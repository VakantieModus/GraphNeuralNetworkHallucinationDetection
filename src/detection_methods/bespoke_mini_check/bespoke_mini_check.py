from __future__ import annotations

import datetime
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from minicheck.minicheck import MiniCheck  # type: ignore
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    f1_score,
    precision_recall_curve,
)
from tqdm import tqdm

from src.detection_methods.kilt_retriever.kilt_retriever import DPRFaissRetriever
from src.detection_methods.utils import run_experiments_utils

Pooling = Literal["max", "mean", "concat"]


@dataclass
class TransformersConfig:
    model_name: str = "Bespoke-MiniCheck-7B"
    cache_dir: str = "./ckpts"
    enable_prefix_caching: bool = False
    device: str | None = None
    chunk_size: int | None = None
    batch_size: int = 1
    max_model_len: int = 4096
    pooling: Pooling = "max"


class BespokeMiniCheck:
    """
    MiniCheck + retrieval.
    Mapping:
      support=1 -> NOT hallucination -> pred=0
      support=0 -> hallucination     -> pred=1
    """

    def __init__(
        self,
        name: str,
        tcfg: TransformersConfig | None = None,
        logger: logging.Logger | None = None,
        number_of_documents: int = 1,
    ):
        self.tcfg = tcfg or TransformersConfig()
        self.log = logger or logging.getLogger(name)
        self._number_of_documents = int(number_of_documents)
        self.retriever = DPRFaissRetriever(device="cuda:1")

        kwargs = dict(
            model_name=self.tcfg.model_name,
            enable_prefix_caching=self.tcfg.enable_prefix_caching,
            cache_dir=self.tcfg.cache_dir,
            max_model_len=self.tcfg.max_model_len,
        )
        if self.tcfg.device is not None:
            kwargs["device"] = self.tcfg.device

        self._scorer = MiniCheck(**kwargs)

    @property
    def number_of_documents(self) -> int:
        return self._number_of_documents

    def train(self, train_data: list[dict], validation_data: list[dict] | None = None):
        pass

    def save_model(self, directory: str, model_name: str):
        pass

    def load_model(self, directory: str, model_prefix: str):
        pass

    def evaluate(self, test_data: list[dict]) -> dict[str, Any]:
        preds, scores, items = self._predict_internal(test_data, return_items=True)
        labels, use_preds, use_scores = [], [], []
        for it in items:
            if it["pred"] is not None and it["label"] is not None:
                labels.append(int(it["label"]))
                use_preds.append(int(it["pred"]))
                if it["score"] is not None:
                    use_scores.append(float(it["score"]))

        metrics = {}
        if labels:
            metrics["accuracy"] = accuracy_score(labels, use_preds)
            metrics["f1"] = f1_score(labels, use_preds, zero_division=0)
            if use_scores and len(set(use_scores)) > 1:
                p, r, _ = precision_recall_curve(labels, use_scores)
                metrics["average_precision"] = average_precision_score(
                    labels, use_scores
                )
                metrics["pr_auc"] = auc(r, p)
        return {"metrics": metrics, "items": items}

    def predict(self, test_data: list[dict]) -> tuple[list[int], list[float | None]]:
        logging.info("PREDICTING")
        preds, scores, _ = self._predict_internal(test_data, return_items=False)
        return preds, scores

    def _predict_internal(
        self, test_data: list[dict], *, return_items: bool
    ) -> tuple[list[int], list[float | None], list[dict] | None]:
        if not isinstance(test_data, list):
            raise ValueError("test_data must be a list[dict].")

        labels_for_metrics: list[int] = []
        preds_for_metrics: list[int] = []
        scores_for_metrics: list[float | None] = []
        items: list[dict] = []

        for sample in tqdm(
            test_data, desc="Predicting BespokeMiniCheck (transformers)"
        ):
            sid = sample.get("id", sample.get("idx"))
            label = sample.get("label", sample.get("Hallucinating"))
            claim = (sample.get("claim") or sample.get("output") or "").strip()
            if not claim:
                items.append(
                    {
                        "id": sid,
                        "label": label,
                        "pred": None,
                        "score": None,
                        "raw": None,
                        "error": "empty claim",
                    }
                )
                continue

            question = (sample.get("question") or sample.get("prompt") or claim).strip()
            try:
                _, _, rows, _ = self.retriever.search(
                    question, k=self._number_of_documents, return_docs=True
                )
            except Exception as e:
                items.append(
                    {
                        "id": sid,
                        "label": label,
                        "pred": None,
                        "score": None,
                        "raw": None,
                        "error": f"retriever: {e}",
                    }
                )
                continue

            passages = [
                f"{r.get('title','')}: {r.get('text','')}".strip(": ")  # noqa
                for r in rows
                if (r.get("title") or r.get("text"))
            ]
            if not passages:
                passages = ["(no context retrieved)"]

            try:
                doc = "\n\n-----\n\n".join(passages)
                pred_label, raw_prob, _, _ = self._scorer.score(
                    docs=[doc],
                    claims=[claim],
                    **(
                        {"chunk_size": self.tcfg.chunk_size}
                        if self.tcfg.chunk_size
                        else {}
                    ),
                )
                support_prob = float(raw_prob[0])
                pred = 0 if support_prob >= 0.5 else 1
                score = 1.0 - support_prob  # hallucination score
                raw = {"support_prob": support_prob, "pooling": self.tcfg.pooling}

                item = {
                    "id": sid,
                    "label": label,
                    "pred": pred,
                    "score": score,
                    "raw": raw,
                }
                items.append(item)

                preds_for_metrics.append(int(pred))
                labels_for_metrics.append(int(label))
                scores_for_metrics.append(float(score))

            except Exception as e:
                logging.error(f"_predict_internal error {e}")
                items.append(
                    {
                        "id": sid,
                        "label": label,
                        "pred": None,
                        "score": None,
                        "raw": None,
                        "error": f"score: {e}",
                    }
                )

        return preds_for_metrics, scores_for_metrics, items if items else None


# --- JSON-serializable conversion helpers ---
def _to_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(v) for v in obj]
    return obj


def _extract_labels(test_data, label_key="Hallucinating"):
    y = []
    for it in test_data:
        if isinstance(it, dict):
            y.append(int(it[label_key]))
        else:
            y.append(int(getattr(it, label_key)))
    return np.asarray(y, dtype=int)


logger = logging.getLogger("experiments")

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )

    det = BespokeMiniCheck(
        name="minicheck_transformers",
        tcfg=TransformersConfig(
            model_name="Bespoke-MiniCheck-7B",
            cache_dir="./ckpts",
            enable_prefix_caching=False,
            device=None,
            chunk_size=None,
            batch_size=4,
            max_model_len=4096,
            pooling="max",
        ),
        number_of_documents=2,
    )

    datasets = [
        "gptoss_20b_tqa",
        "gptoss_20b_eli5",
        "llama2_13b_tqa",
        "llama2_13b_eli5",
        "t5_gemma_tqa",
        "t5_gemma_eli5",
    ]

    out_dir = Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)
    all_runs_path = out_dir / f"exp1_bespoke_minicheck_{datetime.datetime.now()}.jsonl"

    with all_runs_path.open("w", encoding="utf-8") as f_all:
        for dataset_id in datasets:
            llm, benchmark = run_experiments_utils.parse_dataset_id(dataset_id)
            logger.info(
                f"[Exp1] DatasetID={dataset_id} -> llm={llm} benchmark={benchmark}"
            )

            _, _, test_data = run_experiments_utils.load_dataset(
                benchmark, llm=llm, use_pretrained_embeddings=False, text_only=True
            )

            preds, scores = det.predict(test_data)
            preds = np.asarray(preds)
            scores = np.asarray(scores) if scores is not None else None
            y_true = _extract_labels(test_data)

            # compute your metrics (uses your existing util)
            metrics = run_experiments_utils.compute_metrics(
                y_true=y_true, y_scores=scores, y_pred=preds
            )

            meta = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "dataset_id": dataset_id,
                "llm": llm,
                "benchmark": benchmark,
                "detector_name": det.__class__.__name__,
                "detector_cfg": {
                    "name": det.tcfg.model_name,
                    "batch_size": det.tcfg.batch_size,
                    "max_model_len": det.tcfg.max_model_len,
                    "pooling": det.tcfg.pooling,
                    "number_of_documents": det.number_of_documents,
                },
                "n_items": int(len(y_true)),
            }

            record = {
                "meta": _to_serializable(meta),
                "metrics": _to_serializable(metrics),
                "preds": _to_serializable(preds),
                "scores": _to_serializable(scores) if scores is not None else None,
                "y_true": _to_serializable(y_true),
            }

            # append to JSONL
            f_all.write(json.dumps(record, ensure_ascii=False) + "\n")

            # also write a pretty per-dataset JSON
            per_dataset_path = out_dir / f"metrics_{dataset_id}.json"
            with per_dataset_path.open("w", encoding="utf-8") as f_ds:
                json.dump(record, f_ds, indent=2, ensure_ascii=False)

            logging.info(f"Saved metrics to {per_dataset_path}")

    logging.info(f"Appended all runs to {all_runs_path}")
