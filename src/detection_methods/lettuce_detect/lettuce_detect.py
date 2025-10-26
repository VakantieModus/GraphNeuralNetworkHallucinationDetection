from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

from lettucedetect.models.inference import HallucinationDetector
from tqdm import tqdm

from src.detection_methods.kilt_retriever.kilt_retriever import DPRFaissRetriever


# -----------------
# Config container
# -----------------
@dataclass
class LettuceConfig:
    model_path: str = "KRLabsOrg/lettucedect-large-modernbert-en-v1"
    device: str | None = None  # e.g., "cuda:0" or "cpu" (None = auto)
    span_conf_threshold: float = 0.5  # spans with conf >= this are kept
    binary_conf_threshold: float = 0.5  # max span conf >= this -> hallucination=1

    # seq-length handling
    max_seq_len: int = 5000  # model context window
    tokenizer_name: str | None = None  # defaults to model_path if None
    token_margin: int = 64  # safety headroom for special tokens/templates


class LettuceDetectCheck:
    """
    Hallucination detector using LettuceDetect (span-level) + DPR retrieval context.

    Binary mapping:
      - If ANY detected span has confidence >= binary_conf_threshold -> pred = 1 (hallucination)
      - Else pred = 0 (not hallucination)
    """

    def __init__(
        self,
        name: str = "lettucedetect_check",
        cfg: LettuceConfig | None = None,
        logger: logging.Logger | None = None,
        number_of_documents: int = 5,
    ):
        self.cfg = cfg or LettuceConfig()
        self.log = logger or logging.getLogger(name)
        self._number_of_documents = number_of_documents

        # DPR retriever (GPU if supported)
        self.retriever = DPRFaissRetriever()

        # LettuceDetect model
        self.detector = HallucinationDetector(
            method="transformer",
            model_path=self.cfg.model_path,
            device=self.cfg.device,  # "cuda:0" to force GPU
        )

        # Lazy tokenizer
        self._tokenizer = None
        self._tokenizer_available = False

    # No-op train/save/load to match your pipeline contract
    def train(self, train_data: list[dict], validation_data: list[dict] | None = None):
        return

    def save_model(self, directory: str, model_name: str):
        return

    def load_model(self, directory: str, model_prefix: str):
        return

    # -------------
    # Public API
    # -------------
    def evaluate(self, test_data: list[dict[str, Any]]) -> dict[str, Any]:
        """Convenience evaluation with metrics + items (for __main__)."""
        preds, scores, items = self._predict_internal(test_data, return_items=True)

        # Collect labels aligned with items that have predictions
        labels: list[int] = []
        use_preds: list[int] = []
        use_scores: list[float] = []
        for it in items:
            if it["pred"] is not None and it["label"] is not None:
                labels.append(int(it["label"]))
                use_preds.append(int(it["pred"]))
                if it["score"] is not None:
                    use_scores.append(float(it["score"]))

        metrics = {}
        if labels:
            from sklearn.metrics import (
                accuracy_score,
                auc,
                average_precision_score,
                f1_score,
                precision_recall_curve,
            )

            metrics["accuracy"] = accuracy_score(labels, use_preds)
            metrics["f1"] = f1_score(labels, use_preds, zero_division=0)
            if use_scores and len(set(use_scores)) > 1:
                p, r, _ = precision_recall_curve(labels, use_scores)
                metrics["average_precision"] = average_precision_score(
                    labels, use_scores
                )
                metrics["pr_auc"] = auc(r, p)

        return {"metrics": metrics, "items": items}

    def predict(
        self, test_data: list[dict[str, Any]]
    ) -> tuple[list[int], list[float | None]]:
        """
        Contract for your experiment runner:
        returns (y_pred, y_scores) aligned with rows that had labels.
        """
        preds, scores, _ = self._predict_internal(test_data, return_items=False)
        return preds, scores

    # -------------
    # Internals
    # -------------
    def _predict_internal(
        self, test_data: list[dict[str, Any]], *, return_items: bool
    ) -> tuple[list[int], list[float | None], list[dict] | None]:
        if not isinstance(test_data, list):
            raise ValueError("test_data must be a list[dict].")

        self._ensure_tokenizer()

        labels_for_metrics: list[int] = []
        preds_for_metrics: list[int] = []
        scores_for_metrics: list[float | None] = []
        items: list[dict] = []

        for sample in tqdm(test_data, desc="Predicting"):
            sid = sample.get("id", sample.get("idx"))
            label = sample.get("label", sample.get("Hallucinating"))
            answer = (sample.get("output") or "").strip()
            raw_prompt = sample.get("prompt", "") or ""
            question = self._extract_question(raw_prompt).strip()

            # Retrieve context; if it fails, proceed with empty context
            context_str = ""
            try:
                _, _, rows, _ = self.retriever.search(
                    question or answer or "",
                    k=self._number_of_documents,
                    return_docs=True,
                )
                contexts = [
                    f"{r.get('title','')}: {r.get('text','')}".strip()  # noqa
                    for r in rows  # noqa
                ]
                context_str = "\n\n-----\n\n".join(s for s in contexts if s)
            except Exception:
                self.log.exception("Retriever search failed for id=%s", sid)
                context_str = ""

            try:
                span_preds, highest_conf = self._predict_spans_chunked(
                    context=context_str, question=question, answer=answer
                )
                keep_spans = [
                    s
                    for s in span_preds
                    if float(s.get("confidence", 0.0)) >= self.cfg.span_conf_threshold
                ]
                pred_bin = 1 if highest_conf >= self.cfg.binary_conf_threshold else 0
                score = float(highest_conf)

                item = {
                    "id": sid,
                    "label": label,
                    "pred": pred_bin,
                    "score": score,
                    "spans": keep_spans,
                    "question": question,
                    "answer": answer,
                    "chunking": {
                        "tokenizer": (
                            type(self._tokenizer).__name__
                            if self._tokenizer_available
                            else None
                        ),
                        "max_seq_len": self.cfg.max_seq_len,
                        "token_margin": self.cfg.token_margin,
                    },
                }
                for k in ("file", "Comments"):
                    if k in sample:
                        item[k] = sample[k]
                items.append(item)

                labels_for_metrics.append(int(label))
                preds_for_metrics.append(int(pred_bin))
                scores_for_metrics.append(float(score))
            except Exception as e:
                self.log.exception("LettuceDetect call failed for id=%s", sid)
                items.append(
                    {
                        "id": sid,
                        "label": label,
                        "pred": None,
                        "score": None,
                        "spans": None,
                        "error": str(e),
                        "question": question,
                        "answer": answer,
                    }
                )

        return preds_for_metrics, scores_for_metrics, (items if return_items else None)

    def _ensure_tokenizer(self):
        if self._tokenizer is not None:
            return
        try:
            from transformers import AutoTokenizer

            tok_name = self.cfg.tokenizer_name or self.cfg.model_path
            self._tokenizer = AutoTokenizer.from_pretrained(tok_name, use_fast=True)
            self._tokenizer_available = True
        except Exception as e:
            self._tokenizer = None
            self._tokenizer_available = False
            self.log.warning(
                "Falling back to char-based splitting (no tokenizer): %s", e
            )

    def _count_tokens(self, text: str) -> int:
        if not self._tokenizer_available:
            return max(1, len(text) // 4)  # crude fallback
        return len(self._tokenizer.encode(text, add_special_tokens=False))

    def _chunk_context_by_tokens(
        self, context: str, question: str, answer: str
    ) -> list[str]:
        """Split context into token-bounded chunks so question+answer+chunk+margin <= max_seq_len."""
        if not context:
            return [""]

        q_tokens = self._count_tokens(question)
        a_tokens = self._count_tokens(answer)
        budget = self.cfg.max_seq_len - q_tokens - a_tokens - self.cfg.token_margin
        budget = max(128, budget)  # ensure a minimum chunk size

        if not self._tokenizer_available:
            approx_ratio = 4  # ~4 chars/token
            char_budget = budget * approx_ratio
            chunks, start = [], 0
            while start < len(context):
                end = min(len(context), start + char_budget)
                cut = context.rfind("\n\n", start, end)
                if cut == -1:
                    cut = context.rfind("\n", start, end)
                if cut == -1:
                    cut = end
                piece = context[start:cut]
                if piece:
                    chunks.append(piece)
                start = cut
            return chunks or [""]

        ids = self._tokenizer.encode(context, add_special_tokens=False)
        chunks: list[str] = []
        for i in range(0, len(ids), budget):
            piece_ids = ids[i : i + budget]
            if not piece_ids:
                continue
            chunks.append(self._tokenizer.decode(piece_ids, skip_special_tokens=True))
        return chunks or [""]

    def _predict_spans_chunked(
        self, context: str, question: str, answer: str
    ) -> tuple[list[dict], float]:
        """Runs LettuceDetect over token-bounded chunks and aggregates (spans, highest_conf)."""
        context_chunks = self._chunk_context_by_tokens(context, question, answer)
        all_spans: list[dict] = []
        highest_conf = 0.0

        for ctx in context_chunks:
            preds = (
                self.detector.predict(
                    context=ctx,
                    question=question,
                    answer=answer,
                    output_format="spans",
                )
                or []
            )
            for p in preds:
                span = {
                    "start": int(p.get("start", 0)),
                    "end": int(p.get("end", 0)),
                    "confidence": float(p.get("confidence", 0.0)),
                    "text": str(p.get("text", "")),
                }
                all_spans.append(span)
                if span["confidence"] > highest_conf:
                    highest_conf = span["confidence"]

        return all_spans, float(highest_conf)

    @staticmethod
    def _extract_question(prompt: str) -> str:
        if not prompt:
            return ""
        m = re.search(
            r"Question:\s*(.*?)(?:\n|$)", prompt, flags=re.IGNORECASE | re.DOTALL
        )
        return m.group(1).strip() if m else prompt.strip()


# -----------------
# Example runner
# -----------------
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )
    log = logging.getLogger("lettucedetect_demo")

    detector = LettuceDetectCheck(
        name="lettucedetect_yesno_style",
        cfg=LettuceConfig(
            model_path="KRLabsOrg/lettucedect-large-modernbert-en-v1",
            device=None,  # auto
            span_conf_threshold=0.5,
            binary_conf_threshold=0.5,
        ),
        number_of_documents=1,
        use_faiss_gpu=True,
    )

    result = detector.evaluate()

    log.info("\n== Metrics ==\n%s", result["metrics"])
    log.info("\n== Items ==")
    for it in result["items"]:
        score_str = f"{it.get('score'):.3f}" if it.get("score") is not None else "None"
        log.info(
            "%s: label=%s pred=%s score=%s spans=%s",
            it.get("id"),
            it.get("label"),
            it.get("pred"),
            score_str,
            it.get("spans"),
        )
