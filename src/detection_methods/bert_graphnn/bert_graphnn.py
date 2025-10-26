# sentence_gnn_min.py
from __future__ import annotations

import logging
import os
import re
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv, global_mean_pool
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer  # lazy import

os.environ["TOKENIZERS_PARALLELISM"] = "false"


# =========================
# 1) Graph creation class
# =========================
class SentenceGraphBuilder:
    """
    Builds a sentence-level graph from an item dict:
      - Split text into sentences
      - Encode each sentence with frozen BERT (mean-pooled)
      - Add optional positional channel
      - Create edges via cosine similarity >= tau, with top_k fallback

    Returns a PyG Data(x, edge_index, y, gid) where:
      - x: [T, 768(+1)]
      - y: scalar label (0/1)
      - gid: original item["idx"]
    """

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        add_positional: bool = True,
        tau: float = 0.85,
        top_k: int = 8,
        max_sentences: int | None = None,
        device: str | None = None,
    ):
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.enc = AutoModel.from_pretrained(model_name).to(self.device).eval()
        for p in self.enc.parameters():
            p.requires_grad = False

        self.add_positional = add_positional
        self.tau = tau
        self.top_k = top_k
        self.max_sentences = max_sentences
        self.in_dim = 768 + (1 if add_positional else 0)

    # ----- public -----
    def build(
        self,
        item: dict[str, Any],
        text_key: str = "output",
        label_key: str = "Hallucinating",
    ) -> Data:
        text = str(item.get(text_key, "")).strip()
        sents = self._split_sentences(text)
        if self.max_sentences is not None and len(sents) > self.max_sentences:
            sents = sents[: self.max_sentences]

        emb = self._encode_sentences(sents)  # [T, 768] CPU float32
        T = emb.size(0)
        x = emb

        if self.add_positional:
            pos = torch.linspace(0, 1, steps=T, dtype=torch.float32).unsqueeze(
                1
            )  # [T,1]
            x = torch.cat([x, pos], dim=1)  # [T, 768(+1)]

        # Edges from semantic part (exclude position channel)
        edge_index = self._cosine_graph(emb, tau=self.tau, top_k=self.top_k)

        y = int(item.get(label_key, 0))
        data = Data(x=x, edge_index=edge_index, y=torch.tensor(y, dtype=torch.long))
        return data

    # ----- helpers -----
    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        # Simple, robust sentence splitter (keep it dependency-light).
        # Splits on punctuation followed by space/newline; collapses blanks.
        chunks = re.split(r"(?<=[.!?])\s+", text.replace("\n", " ").strip())
        sents = [s.strip() for s in chunks if len(s.strip()) > 0]
        return sents if sents else [""]  # ensure at least one

    @torch.no_grad()
    def _encode_sentences(self, sentences: list[str]) -> torch.Tensor:
        batch = self.tok(
            sentences,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt",
        ).to(self.device)
        last = self.enc(**batch).last_hidden_state  # [B, L, 768]
        mask = batch["attention_mask"].unsqueeze(-1)  # [B, L, 1]
        emb = (last * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)  # [B, 768]
        return emb.detach().cpu().to(torch.float32)

    @staticmethod
    def _cosine_graph(
        node_feats: torch.Tensor, tau: float = 0.85, top_k: int = 8
    ) -> torch.Tensor:
        X = F.normalize(node_feats, dim=1)  # [T, D]
        S = X @ X.T  # [T, T]
        T = X.size(0)
        src, dst = [], []
        for i in range(T):
            sims = S[i].clone()
            sims[i] = -1.0
            cand = (sims >= tau).nonzero(as_tuple=False).view(-1).tolist()
            if not cand:
                k = min(top_k, T - 1)
                if k > 0:
                    _, idxs = torch.topk(sims, k)
                    cand = idxs.tolist()
            for j in cand:
                src.append(i)
                dst.append(j)
                src.append(j)
                dst.append(i)  # undirected
        if len(src) == 0:
            return torch.zeros((2, 0), dtype=torch.long)
        edge_index = torch.tensor([src, dst], dtype=torch.long)
        edge_index = torch.unique(edge_index, dim=1)
        return edge_index


# =========================
# 2) End-to-end trainer
# =========================
class SentenceGATTrainer:
    """
    End-to-end: uses SentenceGraphBuilder to convert raw dicts -> graphs,
    then trains Reducer(768(+1)->32) + 2x GAT + classifier with CE loss.
    """

    def __init__(
        self,
        builder: SentenceGraphBuilder,
        gat_hidden: int = 128,
        heads: int = 4,
        num_classes: int = 2,
        device: str | None = None,
    ):
        self.builder = builder
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.reducer = nn.Sequential(
            nn.Linear(builder.in_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(256, 32),
        ).to(self.device)

        self.gat1 = GATConv(32, gat_hidden // heads, heads=heads, dropout=0.1).to(
            self.device
        )
        self.gat2 = GATConv(
            gat_hidden, gat_hidden // heads, heads=heads, dropout=0.1
        ).to(self.device)

        self.dropout = nn.Dropout(0.3).to(self.device)
        self.classifier = nn.Sequential(
            nn.Linear(gat_hidden, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes),
        ).to(self.device)

        self._compiled = False
        self.train_loader = self.val_loader = self.test_loader = None

    # ----- data wiring -----
    def set_data(
        self,
        train_items: list[dict[str, Any]],
        val_items: list[dict[str, Any]],
        test_items: list[dict[str, Any]],
        *,
        text_key: str = "output",
        label_key: str = "Hallucinating",
        batch_size: int = 32,
        num_workers: int = 4,
        prefetch_factor: int = 2,
        persistent_workers: bool = True,
        pin_memory: bool = True,
    ):
        def build_graphs(items):
            out = []
            for it in tqdm(items, desc="Building graphs"):
                try:
                    out.append(
                        self.builder.build(it, text_key=text_key, label_key=label_key)
                    )
                except Exception as e:
                    logging.warning(
                        f"[GraphBuilder] Skipping idx={it.get('idx')} due to: {e}"
                    )
            return out

        train_graphs = build_graphs(train_items)
        val_graphs = build_graphs(val_items)
        test_graphs = build_graphs(test_items)

        # Use lists directly (PyG DataLoader can handle list[Data])
        use_pw = persistent_workers and num_workers > 0
        use_pf = prefetch_factor if num_workers > 0 else None
        pin = pin_memory and (self.device.type == "cuda")  # avoid MPS warnings

        self.train_loader = DataLoader(
            train_graphs,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin,
            persistent_workers=use_pw,
            prefetch_factor=use_pf,
        )
        self.val_loader = DataLoader(
            val_graphs,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin,
            persistent_workers=use_pw,
            prefetch_factor=use_pf,
        )
        self.test_loader = DataLoader(
            test_graphs,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin,
            persistent_workers=use_pw,
            prefetch_factor=use_pf,
        )

    # ----- core forward -----
    def _forward(self, batch: Data) -> torch.Tensor:
        x, edge_index, b = batch.x, batch.edge_index, batch.batch
        x = self.reducer(x)  # [N, 32]
        x = F.relu(self.gat1(x, edge_index))  # [N, H]
        x = self.dropout(x)
        x = F.relu(self.gat2(x, edge_index))  # [N, H]
        x = global_mean_pool(x, b)  # [B, H]
        logits = self.classifier(x)  # [B, C]
        return logits

    # ----- train / predict -----
    def fit(
        self,
        epochs: int = 10,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        grad_clip: float | None = 1.0,
        compile_model: bool = False,
        save: bool = False,
        dataset: str | None = None,
    ):
        assert self.train_loader is not None, "Call set_data(...) first."

        # Optional torch.compile wrapper around a lightweight nn.Module
        if compile_model and (self.device.type == "cuda") and hasattr(torch, "compile"):

            class _Wrapper(nn.Module):
                def __init__(self, outer):
                    super().__init__()
                    self.outer = outer

                def forward(self, data):
                    return self.outer._forward(data)

            self._compiled = True
            self._compiled_model = torch.compile(_Wrapper(self))  # type: ignore

        params = (
            list(self.reducer.parameters())
            + list(self.gat1.parameters())
            + list(self.gat2.parameters())
            + list(self.classifier.parameters())
        )
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
        criterion = nn.CrossEntropyLoss()

        for ep in range(1, epochs + 1):
            self._train_one_epoch(ep, opt, criterion, grad_clip)
            self._validate(ep)

    def _train_one_epoch(self, ep, opt, criterion, grad_clip):
        self.reducer.train()
        self.gat1.train()
        self.gat2.train()
        self.classifier.train()
        tot = 0.0
        for batch in self.train_loader:
            batch = batch.to(self.device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            logits = (
                self._compiled_model(batch) if self._compiled else self._forward(batch)
            )
            loss = criterion(logits, batch.y)
            loss.backward()
            if grad_clip is not None:
                nn.utils.clip_grad_norm_(
                    [p for p in opt.param_groups[0]["params"] if p.requires_grad],
                    grad_clip,
                )
            opt.step()
            tot += float(loss)
        avg = tot / max(1, len(self.train_loader))
        logging.info(f"[Train] epoch {ep} loss={avg:.4f}")

    @torch.no_grad()
    def _validate(self, ep):
        if self.val_loader is None:
            return
        self.reducer.eval()
        self.gat1.eval()
        self.gat2.eval()
        self.classifier.eval()
        vsum, n = 0.0, 0
        for batch in self.val_loader:
            batch = batch.to(self.device, non_blocking=True)
            logits = self._forward(batch)
            vsum += float(F.cross_entropy(logits, batch.y, reduction="sum"))
            n += batch.y.numel()
        if n:
            logging.info(f"[Val]   epoch {ep} loss={vsum/n:.4f}")  # noqa

    @torch.no_grad()
    def predict(
        self, loader: DataLoader | None = None
    ) -> tuple[list[int], list[float]]:
        loader = loader or self.test_loader
        assert loader is not None, "Provide a loader or call set_data(...)."
        self.reducer.eval()
        self.gat1.eval()
        self.gat2.eval()
        self.classifier.eval()
        y_pred, y_score = [], []
        for batch in loader:
            batch = batch.to(self.device, non_blocking=True)
            probs = F.softmax(self._forward(batch), dim=1)
            preds = probs.argmax(dim=1)
            y_pred.extend(preds.cpu().tolist())
            y_score.extend(probs[:, 1].cpu().tolist())
        return y_pred, y_score


if __name__ == "__main__":
    test_data = []
    for i in range(150):
        q = {
            "idx": i,
            "file": "p00000_prompt_output.txt",
            "prompt": "Question: in football whats the point of wasting the first two plays with a rush - up "
            "the middle - "
            "not regular rush plays i get those\nAnswer the question as precise as possible and add no irrelevant "
            "context.\nExplain your answer by reasoning step by step in a concise manner.\nAnswer:",
            "output": "In football, the point of starting with an option to run up the middle on the "
            "first two plays is to establish a strong baseline of confidence in the offense's "
            "ability to score or to create a sense of dominance. This is typically and typically the "
            "first and the middle of the play\n\nAnswer: The first two plays in football often involve "
            "running up the middle to establish a strong foundation and to assess the defense. "
            'These two plays are "the" first two “...\n\nAnswer: The first two plays in football '
            "often involve "
            "running up the middle to establish a strong foundation and to assess the defense. ",
            "Answer": "Keep the defense honest, get a feel for the pass rush, open up the passing game. An "
            "offense that's too one dimensional will fail. And those rushes up the middle can be "
            "busted wide open sometimes for big yardage.",
            "Hallucinating": 1,
            "Comments": "",
        }

        test_data.append(q)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )

    # your test_data list of dicts goes here...

    # split (adjust sizes to your real dataset)
    train_items, test_items = train_test_split(
        test_data,
        test_size=0.3,
        random_state=42,
        stratify=[d["Hallucinating"] for d in test_data],
    )
    train_items, val_items = train_test_split(
        train_items,
        test_size=0.2,
        random_state=42,
        stratify=[d["Hallucinating"] for d in train_items],
    )

    builder = SentenceGraphBuilder(
        model_name="bert-base-uncased",
        add_positional=True,
        tau=0.85,
        top_k=8,
        max_sentences=40,  # optional cap
    )

    trainer = SentenceGATTrainer(builder, gat_hidden=128, heads=4)

    trainer.fit(
        epochs=5, lr=1e-3, weight_decay=1e-4, grad_clip=1.0, compile_model=False
    )

    y_pred, y_score = trainer.predict()
