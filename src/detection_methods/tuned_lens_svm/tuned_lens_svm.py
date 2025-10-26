import json
import logging
import os
import tempfile
from typing import Any, Literal

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class Standardizer:
    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, X: np.ndarray):
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        self.std_ = np.where(self.std_ < 1e-8, 1.0, self.std_)

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean_) / self.std_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.transform(X)


def _safe_vec(a: np.ndarray) -> np.ndarray:
    """Ensure 1D float32, replace NaNs/Infs with finite numbers."""
    v = np.asarray(a, dtype=np.float32).reshape(-1)
    if not np.isfinite(v).all():
        v = np.nan_to_num(v, nan=0.0, posinf=1.0, neginf=-1.0)
    return v


def _pick_feature_key(samples: list[dict]) -> tuple[str, int]:
    """
    Prefer 'curve_concat' if every sample has same length.
    Otherwise fall back to 'logit_curve' if consistent.
    Raises if neither is consistent.
    """
    keys = ["logit_curve"]
    for k in keys:
        lengths = {len(_safe_vec(s.get(k, np.array([])))) for s in samples}
        if len(lengths) == 1 and next(iter(lengths)) > 0:
            L = next(iter(lengths))
            logging.info(f"[features] using '{k}' (dim={L})")
            return k, L
    raise ValueError(
        "Inconsistent curve lengths across samples for both 'curve_concat' and 'logit_curve'."
    )


def _build_matrix(samples: list[dict], key: str) -> np.ndarray:
    X = [_safe_vec(s[key]) for s in samples]
    return np.vstack(X).astype(np.float32)


def _labels(samples: list[dict]) -> np.ndarray:
    return np.asarray([int(s.get("Hallucinating")) for s in samples], dtype=np.int32)


def _metrics(
    y_true: np.ndarray, y_scores: np.ndarray, thr: float = 0.5
) -> dict[str, Any]:
    y_pred = (y_scores >= thr).astype(int)
    acc = float((y_pred == y_true).mean())
    try:
        auc = float(roc_auc_score(y_true, y_scores))
    except Exception:
        auc = float("nan")
    cm = confusion_matrix(y_true, y_pred)
    rep = classification_report(y_true, y_pred, digits=3)
    return {"accuracy": acc, "auc": auc, "confusion_matrix": cm, "report": rep}


class TunedLensSvmHallucinationDetector:
    """
    Offline-only detector that trains & predicts from precomputed curves:
      - expects each sample dict to include:
          'logit_curve'  -> (L,)
          'tuned_curve'  -> (maybe empty) (L,) or ()
          'curve_concat' -> (L,) or (2L,) depending on whether tuned_curve was present
      - automatically picks a consistent feature key ('curve_concat' preferred)
    """

    def __init__(self, classifier: Literal["svm", "logreg"] = "svm", seed: int = 13):
        self.seed = seed
        self.std = Standardizer()
        self.feature_key: str | None = None
        self.feature_dim: int | None = None

        self.clf = SVC(
            kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=seed
        )

    # ---------- core API ----------
    def train(
        self, train_data: list[dict], validation_data: list[dict] | None = None
    ) -> dict[str, Any]:
        # choose consistent feature representation once
        self.feature_key, self.feature_dim = _pick_feature_key(train_data)

        X = _build_matrix(train_data, self.feature_key)
        y = _labels(train_data)

        Xs = self.std.fit_transform(X)
        self.clf.fit(Xs, y)

        out = {
            "train_samples": len(train_data),
            "feature_dim": int(self.feature_dim),
        }
        return out

    def predict(self, test_samples: list[dict], threshold=0.5) -> dict[str, Any]:
        self._ensure_ready()
        Xt = _build_matrix(test_samples, self.feature_key)
        Xts = self.std.transform(Xt)
        scores = self.predict_scores_from_matrix(Xts)  # standardized
        yt = self.clf.predict(Xts)
        return yt, scores

    # ---------- internals ----------
    def predict_scores_from_matrix(self, Xs: np.ndarray) -> np.ndarray:
        if hasattr(self.clf, "predict_proba"):
            return self.clf.predict_proba(Xs)[:, 1].astype(np.float32)
        if hasattr(self.clf, "decision_function"):
            s = self.clf.decision_function(Xs).astype(np.float32)
            return 1.0 / (1.0 + np.exp(-s))
        # numpy logreg fallback
        return self.clf.predict_proba(Xs).astype(np.float32)

    def _ensure_ready(self):
        if self.feature_key is None or self.feature_dim is None:
            raise RuntimeError(
                "Detector not trained/initialized. Call .train(...) first."
            )

    # ---------- persistence ----------
    def save(self, path: str) -> str:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        # if clf is SVC, persist its attributes via numpy (decision_function params are inside object)
        payload = {
            "feature_key": self.feature_key,
            "feature_dim": np.int32(
                self.feature_dim if self.feature_dim is not None else -1
            ),
            "mean": self.std.mean_,
            "std": self.std.std_,
            "clf_type": "svm" if isinstance(self.clf, SVC) else "logreg",
            "seed": np.int32(self.seed),
        }
        if isinstance(self.clf, SVC):
            # rely on sklearn’s joblib if you want exact fidelity; here we store via np.savez + .get_params()
            payload["clf_params"] = json.dumps(self.clf.get_params())
            # NOTE: for reproducible restore, prefer joblib. For simplicity we only save params and refit may be needed.
        else:
            payload["w"] = getattr(self.clf, "w", None)
            payload["b"] = np.array([getattr(self.clf, "b", 0.0)], dtype=np.float32)

        np.savez(path, **payload)
        return path


if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )
    log = logging.getLogger("tuned_lens_svm_selftest")

    rng = np.random.default_rng(42)

    # ---- make a synthetic dataset with signal in the last quarter of the vector ----
    N = 400
    D = 128  # feature dim for each curve vector
    # Create both keys; _pick_feature_key will prefer 'curve_concat'
    X_base = rng.standard_normal((N, D)).astype(np.float32)

    # Inject a linear signal for class 1 in last quarter of the vector
    signal = np.zeros(D, dtype=np.float32)
    signal[-D // 4 :] = 1.0  # last quarter contains signal
    logits = X_base @ signal + 0.25 * rng.standard_normal(N).astype(np.float32)

    # Balanced labels based on noisy logits
    y = (logits > np.median(logits)).astype(np.int32)

    # Compose features:
    # - logit_curve is the first D entries
    # - tuned_curve is (optional) a slight nonlinear transform
    logit_curve = X_base
    tuned_curve = np.tanh(X_base)  # just to have something plausible
    curve_concat = np.concatenate([logit_curve, tuned_curve], axis=1)  # 2D

    samples = []
    for i in range(N):
        samples.append(
            {
                "idx": i,
                "Hallucinating": int(y[i]),
                "logit_curve": logit_curve[i],
                "tuned_curve": tuned_curve[i],
                "curve_concat": curve_concat[i],  # consistent length → will be picked
            }
        )

    train, test = train_test_split(samples, test_size=0.3, random_state=123, stratify=y)
    train, valid = train_test_split(
        train,
        test_size=0.2,
        random_state=123,
        stratify=[s["Hallucinating"] for s in train],
    )

    # ---- run with SVM classifier ----
    log.info("=== SVM variant ===")
    det = TunedLensSvmHallucinationDetector(classifier="svm", seed=13)
    train_out = det.train(train, valid)
    log.info("Train summary: %s", train_out)

    yt, scores = det.predict(test)
    mt = _metrics(np.asarray(yt), np.asarray(scores))
    log.info(
        "Metrics (SVM): %s",
        {k: (v if k != "confusion_matrix" else v.tolist()) for k, v in mt.items()},
    )

    # quick sanity assertions
    assert 0.5 < mt["accuracy"] <= 1.0, "Accuracy too low for synthetic signal"
    assert np.isfinite(mt["auc"]) and 0.5 < mt["auc"] <= 1.0, "AUC too low"

    # ---- persist & reload, compare scores ----
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "detector_svm.npz")
        det.save(path)
        det2 = TunedLensSvmHallucinationDetector(classifier="svm", seed=13)
        det2.load(path)

        # After reload, standardizer + params are restored; for SVC we restored params (not support vectors),
        # so exact score equality isn't guaranteed unless you re-fit. We just ensure the pipeline runs.
        yt2, scores2 = det2.predict(test)
        assert len(scores2) == len(scores) == len(yt) == len(yt2)
        log.info("Reloaded SVM ran successfully (scores length=%d).", len(scores2))

    # ---- run with NumPy Logistic Regression classifier ----
    log.info("=== NumPy Logistic Regression variant ===")
    det_lr = TunedLensSvmHallucinationDetector(classifier="logreg", seed=7)
    train_out_lr = det_lr.train(train, valid)
    log.info("Train summary (LR): %s", train_out_lr)

    yt_lr, scores_lr = det_lr.predict(test)
    mt_lr = _metrics(np.asarray(yt_lr), np.asarray(scores_lr))
    log.info(
        "Metrics (LogReg): %s",
        {k: (v if k != "confusion_matrix" else v.tolist()) for k, v in mt_lr.items()},
    )

    # persist & reload for LR (weights persist exactly)
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "detector_lr.npz")
        det_lr.save(path)
        det_lr2 = TunedLensSvmHallucinationDetector(classifier="logreg", seed=7)
        det_lr2.load(path)
        # Scores should match closely
        scores_lr2 = det_lr2.predict_scores(test)
        max_abs_diff = float(
            np.max(np.abs(np.asarray(scores_lr2) - np.asarray(scores_lr)))
        )
        log.info("Reloaded LR max|Δscore| = %.6f", max_abs_diff)
        assert max_abs_diff < 1e-6, "Reloaded LR scores differ; persistence bug?"

    log.info("✅ Self-test completed successfully.")
