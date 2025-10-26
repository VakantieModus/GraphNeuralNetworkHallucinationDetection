from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import requests
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    f1_score,
    precision_recall_curve,
)
from tqdm import tqdm

from src.detection_methods.kilt_retriever.kilt_retriever import DPRFaissRetriever


@dataclass
class OllamaConfig:
    model: str = "bespoke-minicheck:latest"
    base_url: str = "http://localhost:11434"
    timeout: tuple[float, float] = (5, 60)  # (connect, read)
    options: dict[str, Any] | None = None
    max_retries: int = 3
    retry_backoff_sec: float = 0.8  # exponential backoff base


class BespokeMiniCheckOllama:
    """
    Binary detector via Ollama that MUST return 'yes' or 'no'.

    Mapping:
      'yes' -> supported by document -> NOT hallucination -> 0
      'no'  -> not supported         -> hallucination     -> 1

    test_data: list[dict] with keys:
      - preferred "claim" or fallback "output": str
      - optional "label" or "Hallucinating" in {0,1} (for metrics)
      - optional "id" or "idx"
      - optional "question"/"prompt" for retrieval context
    """

    def __init__(
        self,
        name: str,
        cfg: OllamaConfig | None = None,
        logger: logging.Logger | None = None,
        number_of_documents: int = 5,
    ):
        self.cfg = cfg or OllamaConfig()
        self.log = logger or logging.getLogger(name)
        self._number_of_documents = number_of_documents
        self.retriever = DPRFaissRetriever()
        self._http = requests.Session()

    # ---- training API (no-op for this detector) ----
    def train(self, train_data: list[dict], validation_data: list[dict] | None = None):
        return

    def save_model(self, directory: str, model_name: str):
        return

    def load_model(self, directory: str, model_prefix: str):
        return

    # ---- evaluation helpers ----
    def evaluate(self, test_data: list[dict]) -> dict[str, Any]:
        """Convenience method for ad-hoc testing (used in your __main__)."""
        preds, scores, items = self._predict_internal(test_data, return_items=True)

        # Metrics (only for rows with labels)
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
        """
        Contract for your experiment runner:
        returns (y_pred, y_scores) aligned with rows that had labels.
        """
        preds, scores, _ = self._predict_internal(test_data, return_items=False)
        return preds, scores

    # ---- internals ----
    def _predict_internal(
        self, test_data: list[dict], *, return_items: bool
    ) -> tuple[list[int], list[float | None], list[dict] | None]:
        if not isinstance(test_data, list):
            raise ValueError("test_data must be a list[dict].")

        labels_for_metrics: list[int] = []
        preds_for_metrics: list[int] = []
        scores_for_metrics: list[float | None] = []
        items: list[dict] = []

        for sample in tqdm(test_data, desc="Predict BespokeMiniCheck"):
            # id / label
            sid = sample.get("id", sample.get("idx"))
            label = sample.get("label", sample.get("Hallucinating"))

            # claim text
            claim = (sample.get("claim") or sample.get("output") or "").strip()
            if not claim:
                self.log.warning("Skipping sample %r: empty claim/output.", sid)
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

            # retrieval query
            question = (sample.get("question") or sample.get("prompt") or claim).strip()

            try:
                _, _, rows, _ = self.retriever.search(
                    question, k=self._number_of_documents, return_docs=True
                )
            except Exception as e:
                self.log.exception("Retriever.search failed for id=%s", sid)
                rows = []
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

            contexts = [
                f"{r.get('title','')}: {r.get('text','')}"  # noqa
                for r in rows
                if (r.get("title") or r.get("text"))
            ]
            context_str = (
                "\n\n-----\n\n".join(contexts) if contexts else "(no context retrieved)"
            )

            prompt = self._build_prompt(document=context_str, claim=claim)

            try:
                raw = self._call_ollama_with_retry(prompt)
                pred = self._parse_yes_no(raw)  # 0 or 1 or None
                # Use a deterministic "score": 1.0 for 'no'/hallucination, 0.0 for 'yes'/not
                score = float(pred) if pred is not None else np.random.choice([0, 1])
                item = {
                    "id": sid,
                    "label": label,
                    "pred": pred,
                    "score": score,
                    "raw": raw,
                }
                items.append(item)

                if (pred is not None) and (label is not None):
                    preds_for_metrics.append(int(pred))
                    labels_for_metrics.append(int(label))
                    scores_for_metrics.append(score)
            except Exception as e:
                self.log.exception("Ollama call failed for id=%s", sid)
                items.append(
                    {
                        "id": sid,
                        "label": label,
                        "pred": None,
                        "score": None,
                        "raw": None,
                        "error": str(e),
                    }
                )

        return preds_for_metrics, scores_for_metrics, (items if return_items else None)

    @staticmethod
    def _build_prompt(document: str, claim: str) -> str:
        return f"Document: {document}\nClaim: {claim}\n"

    def _call_ollama_with_retry(self, prompt: str) -> str:
        url = f"{self.cfg.base_url.rstrip('/')}/api/generate"
        payload = {"model": self.cfg.model, "prompt": prompt, "stream": False}
        if self.cfg.options:
            payload["options"] = self.cfg.options

        last_err: Exception | None = None
        for attempt in range(self.cfg.max_retries):
            try:
                r = self._http.post(url, json=payload, timeout=self.cfg.timeout)
                r.raise_for_status()
                data = r.json()
                return str(data.get("response", "")).strip()
            except Exception as e:
                last_err = e
                # backoff and retry
                time.sleep(self.cfg.retry_backoff_sec * (2**attempt))

        assert last_err is not None
        raise last_err

    @staticmethod
    def _parse_yes_no(text: str) -> int | None:
        if not text:
            return np.random.choice([0, 1])
        s = text.strip().lower()
        # prefer first token
        first = re.split(r"\s+", re.sub(r"^[^a-zA-Z]*(.*)$", r"\1", s), maxsplit=1)[0]
        if first.startswith("yes"):
            return 0
        if first.startswith("no"):
            return 1
        # fallback: search anywhere
        if re.search(r"\byes\b", s):
            return 0
        if re.search(r"\bno\b", s):
            return 1
        return np.random.choice([0, 1])


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )
    log = logging.getLogger("ollama_yesno_demo")

    detector = BespokeMiniCheckOllama(
        name="ollama_yesno",
        cfg=OllamaConfig(
            model="bespoke-minicheck:latest",
            base_url="http://localhost:11434",
            options={"temperature": 0.0},
        ),
        number_of_documents=3,
    )

    test_data = [
        {
            "idx": 0,
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
        },
        {
            "idx": 1,
            "file": "p00001_prompt_output.txt",
            "prompt": "Question: Why are different tiers (regular < mid < premium) of gas' prices "
            "almost always 10 cents different?\nAnswer the question as precise as possible "
            "and add no irrelevant context.\nExplain your answer by reasoning step by step in a "
            "concise manner.\nAnswer:",
            "output": '10? What are? This \nWe have a question: "Why are different tiers (regular < mid < premium) of '
            "gas' prices almost always 10 cents different?\" The user wants to know why the gas price "
            "tiers differ by about 10 cents. They want a precise answer, no irrelevant context. They want "
            "explanation step by step concisely.\n\nWe need to explain that gasoline pricing is influenced "
            "by the cost of refining, taxes, and market segmentation. Typically, premium gasoline contains "
            "higher ethanol or more additives, so the refiner sells at a higher margin. The 10 cent difference "
            "is due to the refiner's margin structure and the tax structure.\n\nThe question: Why are different "
            "tiers of gas prices almost always 10 cents different? The answer: Because of how refineries price ",
            "Answer": "As someone who uses quality Premium, I wish this was true.",
            "Hallucinating": 1,
            "Comments": "",
        },
    ]

    result = detector.evaluate(test_data)
    log.info("\n== Metrics ==\n%s", result["metrics"])
    log.info("\n== Items ==")
    for it in result["items"]:
        log.info(
            "%s: label=%s pred=%s raw=%r",
            it["id"],
            it["label"],
            it["pred"],
            it.get("raw"),
        )
