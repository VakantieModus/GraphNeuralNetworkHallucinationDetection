#!/usr/bin/env python3

import json
import logging
import time
from pathlib import Path

import faiss
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PATH_KNOWLEDGE_SOURCE = Path("generated_data/knowledge_source")


class DPRFaissRetriever:
    """
    Lightweight DPR question encoder + FAISS index retriever.

    Files must be aligned:
      - index.faiss            : FAISS index with ntotal = N
      - docstore.jsonl         : N lines, one JSON per passage
      - docstore.offsets.npy   : int64 offsets into the jsonl (len = N)
    """

    def __init__(
        self,
        index_path=PATH_KNOWLEDGE_SOURCE / "wiki_full_ivfpq.faiss",
        doc_path=PATH_KNOWLEDGE_SOURCE / "wiki_full_docstore.jsonl",
        offsets_path=PATH_KNOWLEDGE_SOURCE / "wiki_full_docstore.offsets.npy",
        device: str = DEFAULT_DEVICE,
        nprobe: int | None = 16,
        preload_docstore: bool = False,
        use_gpu: bool = False,
    ):
        self.index_path = index_path
        self.doc_path = doc_path
        self.offsets_path = offsets_path
        self.device = device
        self.preload = preload_docstore

        self.index = faiss.read_index(str(self.index_path))
        if hasattr(self.index, "nprobe") and nprobe is not None:
            self.index.nprobe = int(nprobe)

        self.offsets = np.load(self.offsets_path)

        if hasattr(self.index, "ntotal") and self.index.ntotal != self.offsets.shape[0]:
            logging.info(
                f"[warn] index.ntotal({self.index.ntotal}) != offsets({self.offsets.shape[0]})"
            )

        self.qtok = AutoTokenizer.from_pretrained(
            "facebook/dpr-question_encoder-single-nq-base", local_files_only=False
        )
        self.qenc = (
            AutoModel.from_pretrained(
                "facebook/dpr-question_encoder-single-nq-base", local_files_only=False
            )
            .to(self.device)
            .eval()
        )
        try:
            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)
        except Exception:
            pass

        self._docs: list[dict] | None = None
        if self.preload:
            self._docs = self._load_docstore()

        self._preflight()

    def encode(self, questions: list[str]) -> np.ndarray:
        """Return L2-normalized DPR embeddings (float32, shape [B, d])."""
        with torch.no_grad():
            t = self.qtok(
                questions,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=256,
            ).to(self.device)
            v = self.qenc(**t).pooler_output.detach().cpu().numpy().astype("float32")
        v /= np.linalg.norm(v, axis=1, keepdims=True) + 1e-12
        return v

    def search(
        self, query: str, k: int = 5, return_docs: bool = True
    ) -> tuple[np.ndarray, np.ndarray, list[dict] | None, dict[str, float]]:
        """
        Single-query search.
        Returns (scores[D], ids[I], rows[docs or None], timings_ms).
        """
        t0 = time.time()
        qv = self.encode([query])
        t1 = time.time()
        D, I = self.index.search(qv, k)  # noqa
        t2 = time.time()

        rows = self._fetch_docs(I[0]) if return_docs else None
        t3 = time.time()
        timings = {
            "encode_ms": (t1 - t0) * 1000,
            "search_ms": (t2 - t1) * 1000,
            "read_ms": (t3 - t2) * 1000,
        }
        return D[0], I[0], rows, timings

    def search_batch(
        self, queries: list[str], k: int = 5, return_docs: bool = False
    ) -> tuple[np.ndarray, np.ndarray, list[list[dict]] | None, dict[str, float]]:
        """
        Batched search.
        Returns (scores[B,k], ids[B,k], rows[optional], timings_ms).
        """
        t0 = time.time()
        Q = self.encode(queries)
        t1 = time.time()
        D, I = self.index.search(Q, k)  # noqa
        t2 = time.time()

        rows = None
        if return_docs:
            rows = [self._fetch_docs(ids) for ids in I]
        t3 = time.time()
        timings = {
            "encode_ms": (t1 - t0) * 1000,
            "search_ms": (t2 - t1) * 1000,
            "read_ms": (t3 - t2) * 1000,
        }
        return D, I, rows, timings

    def _fetch_docs(self, ids: np.ndarray) -> list[dict]:
        if self._docs is not None:
            return [self._docs[int(i)] for i in ids]

        out: list[dict] = []
        fsize = Path(self.doc_path).stat().st_size
        with open(self.doc_path, "rb") as f:
            for pid in ids:
                off = int(self.offsets[pid])
                if off < 0 or off >= fsize:
                    raise ValueError(
                        f"Bad offset: pid={pid} offset={off} fsize={fsize}"
                    )
                f.seek(off)
                line = f.readline()
                if not line or line[:1] not in (b"{", b"["):
                    line = f.readline()
                try:
                    out.append(json.loads(line.decode("utf-8")))
                except Exception as e:
                    head = line[:60]
                    raise ValueError(
                        f"JSON parse failed @pid={pid} off={off} head={head!r}"
                    ) from e
        return out

    def _load_docstore(self) -> list[dict]:
        docs = []
        with open(self.doc_path, encoding="utf-8") as f:
            for ln in f:
                docs.append(json.loads(ln))
        return docs

    def _preflight(self) -> None:
        if Path(self.doc_path).stat().st_size == 0:
            raise RuntimeError(f"[preflight] docstore is empty: {self.doc_path}")
        n_lines = sum(1 for _ in open(self.doc_path, "rb"))
        if n_lines != self.offsets.shape[0]:
            logging.info(
                f"[warn] doc lines({n_lines}) != offsets({self.offsets.shape[0]})"
            )
        # FAISS id space sanity
        if self.index.ntotal and self.index.ntotal != self.offsets.shape[0]:
            logging.info(
                f"[warn] index.ntotal({self.index.ntotal}) != offsets({self.offsets.shape[0]})"
            )

    @staticmethod
    def _resolve(p: str) -> str:
        path = Path(p)
        return str(
            path if path.is_absolute() else (Path(__file__).resolve().parent / path)
        )


# ---------- Minimal CLI for quick testing ----------
if __name__ == "__main__":

    retriever = DPRFaissRetriever(
        index_path="knowledge_source/wiki_full_ivfpq.faiss",
        doc_path="knowledge_source/wiki_full_docstore.jsonl",
        offsets_path="knowledge_source/wiki_full_docstore.offsets.npy",
        device="cpu",
    )

    scores, ids, rows, timings = retriever.search(
        "Who discovered the electron and when?", k=5, return_docs=True
    )

    contexts = [f"{r.get('title','')}: {r.get('text','')}" for r in rows]  # noqa

    for i, ctx in enumerate(contexts, 1):
        logging.info(
            f"\n=== Context {i} ===\n{ctx[:500]}{'...' if len(ctx) > 500 else ''}"
        )

    logging.info("\n---\n".join(contexts))
