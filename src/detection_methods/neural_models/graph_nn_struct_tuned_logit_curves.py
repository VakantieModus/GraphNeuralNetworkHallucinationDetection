import copy
import json
import logging
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    auc,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv, global_mean_pool
from tqdm import tqdm


class StructTunedLogitGraphNN(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim=128,
        num_classes=2,
        heads=4,
        edge_dim: int | None = None,
        device: str | None = None,
    ):
        super().__init__()
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        # IMPORTANT: edge_dim + no implicit self-loops (we won’t have attrs for loops)
        self.conv1 = GATConv(
            input_dim,
            hidden_dim // heads,
            heads=heads,
            dropout=0.1,
            edge_dim=edge_dim,
            add_self_loops=False,
        )
        self.conv2 = GATConv(
            hidden_dim,
            hidden_dim // heads,
            heads=heads,
            dropout=0.1,
            edge_dim=edge_dim,
            add_self_loops=False,
        )
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )
        self.to(self.device)
        self._compiled = False

    def forward(self, data: Data):
        x, edge_index = data.x, data.edge_index
        edge_attr = getattr(data, "edge_attr", None)  # [E, edge_dim] or None
        x = F.relu(self.conv1(x, edge_index, edge_attr))
        x = self.dropout(x)
        x = F.relu(self.conv2(x, edge_index, edge_attr))
        return self.classifier(global_mean_pool(x, data.batch))

    # -------- data plumbing --------
    def set_data(
        self,
        train_data,
        val_data=None,
        test_data=None,
        batch_size: int = 32,
        window: int = 2,
        num_workers: int = 4,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        prefetch_factor: int = 2,
        val_split: float = 0.20,  # NEW: 10% validation
        stratify: bool = True,  # NEW: stratify by "Hallucinating"
        rng_seed: int = 42,  # NEW: reproducibility
    ):
        """
        If val_data is None, split train_data into train/val with given ratio (stratified by label).
        """
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._window = window
        val_data = None

        original_train = train_data
        if val_data is None:
            if not (0.0 < val_split < 1.0):
                raise ValueError("val_split must be in (0, 1).")

            rng = np.random.default_rng(rng_seed)
            if stratify:
                by_label = defaultdict(list)
                for i, it in enumerate(original_train):
                    by_label[int(it["Hallucinating"])].append(i)

                train_idx, val_idx = [], []
                for _, idxs in by_label.items():
                    idxs = np.array(idxs)
                    rng.shuffle(idxs)
                    n_val = max(1, int(round(len(idxs) * val_split)))
                    val_idx.extend(idxs[:n_val].tolist())
                    train_idx.extend(idxs[n_val:].tolist())
            else:
                idxs = np.arange(len(original_train))
                rng.shuffle(idxs)
                n_val = max(1, int(round(len(original_train) * val_split)))
                val_idx = idxs[:n_val].tolist()
                train_idx = idxs[n_val:].tolist()

            train_data = [original_train[i] for i in train_idx]
            val_data = [original_train[i] for i in val_idx]

        # keep raw splits (optional)
        self._train_data_raw = train_data
        self._val_data_raw = val_data
        self._test_data_raw = test_data

        # build datasets (pass neighbor_window=window consistently)
        train_dataset = TokenGraphDatasetGraphNNStructTunedLogit(
            train_data, connect_sentence_clique=True, neighbor_window=window
        )
        val_dataset = (
            TokenGraphDatasetGraphNNStructTunedLogit(
                val_data, connect_sentence_clique=True, neighbor_window=window
            )
            if val_data is not None
            else None
        )
        test_dataset = (
            TokenGraphDatasetGraphNNStructTunedLogit(
                test_data, connect_sentence_clique=True, neighbor_window=window
            )
            if test_data is not None
            else None
        )

        def mk_loader(ds, shuffle):
            if ds is None:
                return None
            use_pw = persistent_workers and num_workers > 0
            use_pf = prefetch_factor if num_workers > 0 else None
            return DataLoader(
                ds,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                pin_memory=pin_memory and (self.device.type == "cuda"),
                persistent_workers=use_pw,
                prefetch_factor=use_pf,
            )

        self.train_loader = mk_loader(train_dataset, True)
        self.val_loader = mk_loader(val_dataset, False)
        self.test_loader = mk_loader(test_dataset, False)

    # -------- train / predict --------
    def fit(
        self,
        epochs=10,
        lr=1e-3,
        grad_clip: float | None = 1.0,
        compile_model: bool = False,
        dataset: str | None = None,
        save: bool = False,
    ):

        if (
            compile_model
            and not getattr(self, "_compiled", False)
            and torch.cuda.is_available()
            and hasattr(torch, "compile")
        ):
            self = torch.compile(self)  # type: ignore
            self._compiled = True

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        criterion = torch.nn.CrossEntropyLoss()

        best_aupr = -float("inf")
        best_epoch = -1
        best_state = None

        for epoch in range(1, epochs + 1):
            # ---- Train ----
            self.train()
            total_loss = 0.0
            for batch in tqdm(
                self.train_loader, desc=f"GNN Epoch {epoch}", leave=False
            ):
                batch = batch.to(self.device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)

                logits = self.forward(batch)
                loss = criterion(logits, batch.y)
                loss.backward()

                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(self.parameters(), grad_clip)

                optimizer.step()
                total_loss += float(loss)

            avg_train = total_loss / max(1, len(self.train_loader))
            logging.info(f"[GNN] Epoch {epoch} — Train Loss: {avg_train:.4f}")

            # ---- Validation (sklearn metrics) ----
            self.eval()
            all_probs: list[float] = []
            all_labels: list[int] = []

            if self.val_loader is not None:
                with torch.no_grad():
                    for batch in self.val_loader:
                        batch = batch.to(self.device, non_blocking=True)
                        logits = self.forward(batch)

                        # Binary probs
                        if logits.shape[-1] == 1:
                            probs = torch.sigmoid(logits).squeeze(-1)
                        else:
                            probs = torch.softmax(logits, dim=-1)[:, 1]

                        all_probs.extend(probs.detach().float().cpu().tolist())
                        all_labels.extend(batch.y.detach().int().cpu().tolist())

            if len(all_labels) > 0:
                # Handle single-class edge case (AUC-PR undefined)
                if len(set(all_labels)) == 1:
                    pos_rate = float(sum(all_labels)) / len(all_labels)
                    aupr = pos_rate
                    ap = pos_rate
                    p = r = f1 = 0.0
                    cm = [[len(all_labels) - sum(all_labels), 0], [0, sum(all_labels)]]
                    logging.warning(
                        "[GNN] Validation has a single class; AUC-PR/AP set to positive rate."
                    )
                else:
                    precision, recall, _ = precision_recall_curve(all_labels, all_probs)
                    aupr = auc(recall, precision)
                    ap = average_precision_score(all_labels, all_probs)

                    # Thresholded metrics at 0.5 (optional but often handy)
                    preds = [1 if p >= 0.5 else 0 for p in all_probs]
                    p = precision_score(all_labels, preds, zero_division=0)
                    r = recall_score(all_labels, preds, zero_division=0)
                    f1 = f1_score(all_labels, preds, zero_division=0)
                    cm = confusion_matrix(all_labels, preds).tolist()

                logging.info(
                    f"[GNN] Epoch {epoch} — Val AUC-PR: {aupr:.4f} | AP: {ap:.4f} | "
                    f"P/R/F1@0.5: {p:.4f}/{r:.4f}/{f1:.4f} | CM: {cm}"
                )

                # Keep best by AUC-PR
                if aupr > best_aupr and epoch >= 5:
                    best_aupr = aupr
                    best_epoch = epoch
                    best_state = copy.deepcopy(self.state_dict())

        # ---- Restore best weights ----
        if best_state is not None:
            self.load_state_dict(best_state)
            logging.info(
                f"[GNN] Restored best checkpoint from epoch {best_epoch} (AUC-PR={best_aupr:.4f})."
            )
        else:
            logging.warning(
                "[GNN] No validation improvement recorded; kept last-epoch weights."
            )

        # ---- Optional: save best only ----
        if save:
            meta = {
                "best_epoch": best_epoch,
                "best_val_aupr": (
                    None if best_aupr == -float("inf") else float(best_aupr)
                ),
                "dataset": dataset,
            }
            try:
                save_gnn(
                    self,
                    out_dir=f"GNN_MODELS/graph_nn_struct_tuned_logit/best_{dataset}",
                    fp16=False,
                    use_safetensors=False,
                    metadata=meta,
                )
                logging.info("[GNN] Saved best checkpoint.")
            except Exception as e:
                logging.exception(f"[GNN] Save failed: {e}")

    def predict(self, data_loader=None):
        self.eval()
        data_loader = data_loader or self.test_loader
        predicted_labels, predicted_scores = [], []

        with torch.no_grad():
            for batch in data_loader:
                batch = batch.to(self.device, non_blocking=True)
                probs = F.softmax(self.forward(batch), dim=1)
                preds = probs.argmax(dim=1)

                predicted_labels.extend(preds.cpu().numpy().tolist())
                predicted_scores.extend(probs[:, 1].float().cpu().numpy().tolist())

        return predicted_labels, predicted_scores


# ---------- your model class is assumed available here ----------


def get_model_config(model: StructTunedLogitGraphNN) -> dict:
    # store what you need to reconstruct the module
    return {
        "input_dim": model.conv1.in_channels,  # D+1
        "hidden_dim": model.classifier[0].in_features,  # matches hidden_dim
        "num_classes": model.classifier[-1].out_features,
        "heads": model.conv1.heads,
    }


def save_gnn(
    model: StructTunedLogitGraphNN,
    out_dir: str | Path,
    fp16: bool = True,
    use_safetensors: bool = False,
    metadata: dict | None = None,
):
    """
    Saves only weights + a small JSON config.
    Set fp16=False if you plan to continue training from this checkpoint.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Save config
    cfg = get_model_config(model)
    if metadata:
        cfg["_meta"] = metadata
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2))

    # 2) Build a CPU state_dict (optionally cast to fp16)
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    if fp16:
        state = {
            k: (v.half() if v.is_floating_point() else v) for k, v in state.items()
        }

    # 3) Save weights
    if use_safetensors:
        from safetensors.torch import save_file

        save_file(state, str(out_dir / "model.safetensors"))
    else:
        # new zipfile format is default; no pickle of the whole module
        torch.save(state, out_dir / "model.pt")


def load_graph_nn_struct_tuned_logit(
    out_dir: str | Path,
    map_device: str | None = None,
    use_safetensors: bool = False,
    expect_fp16: bool = True,
) -> StructTunedLogitGraphNN:
    """
    Recreates the module from config.json and loads weights.
    If weights were saved in fp16 and you want to run on CPU, keep them in fp16 for inference
    or call .float() after loading if you prefer fp32 compute.
    """
    out_dir = Path(out_dir)
    cfg = json.loads((out_dir / "config.json").read_text())

    device = torch.device(
        map_device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model = StructTunedLogitGraphNN(
        input_dim=cfg["input_dim"],
        hidden_dim=cfg["hidden_dim"],
        num_classes=cfg["num_classes"],
        heads=cfg["heads"],
        device=str(device),
    )
    model.eval()  # default

    # Load weights
    if use_safetensors:
        from safetensors.torch import load_file

        state = load_file(str(out_dir / "model.safetensors"))
    else:
        state = torch.load(out_dir / "model.pt", map_location="cpu")

    # If you saved in fp16 but want fp32 compute, upcast after load:
    if expect_fp16:
        model.load_state_dict(state, strict=True)
        # optional upcast:
        # model = model.float()
    else:
        # In case tensors are fp16 on disk but you want fp32 in memory:
        state = {
            k: (v.float() if v.is_floating_point() else v) for k, v in state.items()
        }
        model.load_state_dict(state, strict=True)

    model.to(device)
    return model


class TokenGraphDatasetGraphNNStructTunedLogit(Dataset):
    _SENT_SPLIT_RE = re.compile(r"(?<=[\.!\?])\s+")

    def __init__(
        self,
        data,
        connect_sentence_clique: bool = True,
        neighbor_window: int = 2,
        use_curves: bool = True,
        include_curves: tuple[str] = ("logit_curves",),
    ):
        super().__init__()
        self.data = data
        self.connect_sentence_clique = connect_sentence_clique
        self.neighbor_window = max(1, int(neighbor_window))
        self.use_curves = bool(use_curves)
        self.include_curves = tuple(include_curves)

    # --- PyG expects these ---
    def len(self):
        return len(self.data)

    def get(self, idx):
        return self.__getitem__(idx)

    # ----------------------- helpers -----------------------
    @staticmethod
    def _sentence_spans_from_text_proportional(text: str, T: int):
        """Split by [.?!] and proportionally map char-lengths to contiguous token spans covering [0..T-1]."""
        if T <= 0:
            return []
        text = (text or "").strip()
        if not text:
            return [(0, T - 1)]
        sentences = TokenGraphDatasetGraphNNStructTunedLogit._SENT_SPLIT_RE.split(text)
        lens = [max(1, len(s)) for s in sentences]
        total = sum(lens)
        raw = [L * T / total for L in lens]
        base = [int(np.floor(x)) for x in raw]
        rem = T - sum(base)
        if rem > 0:
            order = np.argsort([-(x - np.floor(x)) for x in raw])
            for k in order[:rem]:
                base[k] += 1
        spans, cur = [], 0
        for cnt in base:
            if cnt <= 0:
                continue
            a, b = cur, cur + cnt - 1
            spans.append((a, b))
            cur = b + 1
        if not spans:
            return [(0, T - 1)]
        if spans[0][0] > 0:
            spans[0] = (0, spans[0][1])
        if spans[-1][1] < T - 1:
            spans[-1] = (spans[-1][0], T - 1)
        return spans

    @staticmethod
    def _as_token_matrix(x, T: int) -> np.ndarray:
        """
        Coerce x to float32 (T, C):
          - None/empty -> (T,0)
          - 2D and first dim == T -> use as-is (per-token features)
          - 2D and first dim == 1 -> broadcast to T
          - 1D (C,)               -> treat as feature vector -> broadcast to (T,C)
          - 1D (T,)               -> ambiguous per-token scalar -> (T,1)
          - Otherwise trim/tile first dim to T
        """
        if x is None:
            return np.zeros((T, 0), dtype=np.float32)

        a = np.asarray(x)
        if a.size == 0:
            return np.zeros((T, 0), dtype=np.float32)

        if a.ndim == 2:
            if a.shape[0] == T:
                return a.astype(np.float32, copy=False)
            if a.shape[0] == 1:
                return np.repeat(a.astype(np.float32, copy=False), T, axis=0)
            # otherwise trim/tile rows to T
            if a.shape[0] < T:
                reps = int(np.ceil(T / a.shape[0]))
                a = np.tile(a, (reps, 1))[:T]
            else:
                a = a[:T]
            return a.astype(np.float32, copy=False)

        if a.ndim == 1:
            if a.shape[0] == T:
                # per-token scalar → (T,1)
                return a.reshape(T, 1).astype(np.float32, copy=False)
            # feature vector → broadcast to (T,C)
            return np.repeat(a.reshape(1, -1).astype(np.float32, copy=False), T, axis=0)

        # fallthrough (unexpected shapes)
        return np.zeros((T, 0), dtype=np.float32)

    @staticmethod
    def _edges_token_window_with_attrs(spans, T: int, curve_feat: np.ndarray, W: int):
        """
        Windowed token↔token edges within each sentence span.
        For i and k=1..W with j=i+k≤b:
          - add i→j and j→i
          - ONLY i→(i+1) carries non-zero attr = curve_feat[i+1]; others are zeros.
        """
        src, dst, attrs = [], [], []
        edge_dim = int(curve_feat.shape[1])
        zero = np.zeros((edge_dim,), dtype=np.float32)
        for a, b in spans:
            for i in range(a, b + 1):
                for k in range(1, W + 1):
                    j = i + k
                    if j > b:
                        break
                    # forward
                    src.append(i)
                    dst.append(j)
                    attrs.append(curve_feat[j] if (k == 1 and edge_dim > 0) else zero)
                    # backward
                    src.append(j)
                    dst.append(i)
                    attrs.append(zero)
        edge_attr = (
            np.stack(attrs, axis=0)
            if attrs
            else np.zeros((0, edge_dim), dtype=np.float32)
        )
        return src, dst, edge_attr

    @staticmethod
    def _edges_token_sentence(spans, T):
        src, dst, supernodes = [], [], []
        for k, (a, b) in enumerate(spans):
            s_idx = T + k
            supernodes.append(s_idx)
            for i in range(a, b + 1):
                src += [i, s_idx]
                dst += [s_idx, i]
        return src, dst, supernodes

    @staticmethod
    def _edges_between_sentences(supernodes, clique: bool):
        src, dst = [], []
        S = len(supernodes)
        if S <= 1:
            return src, dst
        if clique:
            for i in range(S):
                for j in range(i + 1, S):
                    u, v = supernodes[i], supernodes[j]
                    src += [u, v]
                    dst += [v, u]
        else:
            for i in range(S - 1):
                u, v = supernodes[i], supernodes[i + 1]
                src += [u, v]
                dst += [v, u]
        return src, dst

    # --------------------- /helpers -----------------------

    def __getitem__(self, idx):
        item = self.data[idx]

        # token embeddings + label
        X = item["tensor"] if "tensor" in item else item["tensor_last_layer"]
        y = int(item["Hallucinating"])
        X = np.asarray(X, dtype=np.float32)  # (T, D)
        T, D = X.shape

        # sentence spans
        text = (item.get("output") or "").strip()
        spans = self._sentence_spans_from_text_proportional(text, T)

        # node features
        pos = (np.arange(T, dtype=np.float32) / max(1, T)).reshape(-1, 1)
        x_tokens = np.concatenate([X, pos], axis=1).astype(np.float32)
        sent_feats = []
        for a, b in spans:
            if a <= b:
                mean_feat = x_tokens[a : b + 1, :-1].mean(axis=0, keepdims=True)
            else:
                mean_feat = np.zeros((1, D), dtype=np.float32)
            sent_feats.append(
                np.concatenate([mean_feat, np.array([[1.0]], dtype=np.float32)], axis=1)
            )
        x_sent = (
            np.concatenate(sent_feats, axis=0)
            if sent_feats
            else np.zeros((0, D + 1), dtype=np.float32)
        )
        x_all = np.concatenate([x_tokens, x_sent], axis=0).astype(np.float32)

        # ----- OPTIONAL curves -----
        if self.use_curves:
            mats = [
                self._as_token_matrix(item.get(k, None), T) for k in self.include_curves
            ]
            curve_feat = (
                np.concatenate(mats, axis=1)
                if mats and sum(m.shape[1] for m in mats) > 0
                else np.zeros((T, 0), dtype=np.float32)
            )
        else:
            curve_feat = np.zeros((T, 0), dtype=np.float32)
        edge_dim = int(curve_feat.shape[1])

        # edges
        s_tt, d_tt, a_tt = self._edges_token_window_with_attrs(
            spans, T, curve_feat, self.neighbor_window
        )
        s_ts, d_ts, supernodes = self._edges_token_sentence(spans, T)
        a_ts = np.zeros((len(s_ts), edge_dim), dtype=np.float32)
        s_ss, d_ss = self._edges_between_sentences(
            supernodes, clique=self.connect_sentence_clique
        )
        a_ss = np.zeros((len(s_ss), edge_dim), dtype=np.float32)

        src = s_tt + s_ts + s_ss
        dst = d_tt + d_ts + d_ss
        edge_index = torch.tensor([src, dst], dtype=torch.long)

        if edge_dim > 0:
            edge_attr = torch.from_numpy(np.concatenate([a_tt, a_ts, a_ss], axis=0))
        else:
            edge_attr = torch.zeros((edge_index.size(1), 0), dtype=torch.float32)

        return Data(
            x=torch.from_numpy(x_all),  # (T + S, D+1)
            edge_index=edge_index,  # [2, E]
            edge_attr=edge_attr,  # [E, EDGE_DIM] (can be (E,0))
            y=torch.tensor(y, dtype=torch.long),
        )


# ---------- self-test ----------
if __name__ == "__main__":
    from sklearn.metrics import accuracy_score, roc_auc_score
    from sklearn.model_selection import train_test_split

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )

    rng = np.random.default_rng(123)
    N = 500
    D = 32  # base embedding size; dataset will add +1 positional → model input_dim should be D+1

    # Create variable-length token sequences; add class-1 signal via a local neighborhood pattern
    samples = []
    for i in range(N):
        T = int(rng.integers(40, 160))
        X = rng.standard_normal((T, D)).astype(np.float32)
        y = int(rng.integers(0, 2))
        if y == 1:
            # bump a band of features around the center tokens
            center = T // 2
            lo, hi = max(0, center - 5), min(T, center + 5)
            X[lo:hi, :6] += 1.2
        samples.append({"tensor_last_layer": X, "Hallucinating": y})

    train, test = train_test_split(
        samples,
        test_size=0.2,
        random_state=42,
        stratify=[s["Hallucinating"] for s in samples],
    )
    train, val = train_test_split(
        train,
        test_size=0.2,
        random_state=42,
        stratify=[s["Hallucinating"] for s in train],
    )

    model = StructTunedLogitGraphNN(
        input_dim=D + 1, hidden_dim=128, num_classes=2, heads=4, device=None
    )
    model.set_data(
        train,
        val,
        test,
        batch_size=64,
        window=10,
        num_workers=4,
        pin_memory=True,  # now wired correctly
        persistent_workers=True,
        prefetch_factor=2,
    )
    model.fit(epochs=5, lr=1e-3, grad_clip=1.0, compile_model=False)

    y_pred, y_scores = model.predict()
    y_true = [s["Hallucinating"] for s in test]

    acc = accuracy_score(y_true, y_pred)
    try:
        auc = roc_auc_score(y_true, y_scores)  # noqa
    except Exception:
        auc = float("nan")

    logging.info(f"[GNN Self-test] accuracy={acc:.3f}  auc={auc:.3f}")
