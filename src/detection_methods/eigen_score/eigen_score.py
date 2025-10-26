from __future__ import annotations

import logging

import numpy as np
from sklearn.metrics import (
    auc,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_curve,
)


class EigenScoreLastToken:
    """
    EigenScore detector using LAST-TOKEN embeddings across K generations.
    - No training, saving, or loading needed; those methods are no-ops.
    - Evaluate expects a list of dicts. For each dict, provide one of:
        * "generations": List[ArrayLike]  # per-generation arrays
          Each array can be:
            - shape (D,): already the last-token vector
            - shape (D, T): we will take the last column as the last token
        * Optionally: "id" for bookkeeping.
    Returns a dict with per-item scores and a mean over all items.
    """

    def __init__(self, name: str = "eigenscore_last_token", alpha: float = 1e-3):
        self.alpha = float(alpha)

    def self_test_with_dummy_data(
        self,
        n_per_class: int = 25,
        K: int = 12,  # generations per item
        D: int = 128,  # hidden dim
        disp_non: float = 0.4,
        disp_hal: float = 1.1,
        seed: int = 42,
    ):
        """
        Quick local self-test:
          1) synthesize validation & test sets,
          2) calibrate threshold τ on validation via self.train(...),
          3) evaluate on test via self.predict(...).
        Returns a dict similar to evaluate(): {"metrics": ..., "items": ...}.
        """

        rng = np.random.default_rng(seed)

        def make_item(item_id: str, label: int, dispersion: float) -> dict:
            # Simulate K generations. LAST column is last-token vector.
            base = rng.normal(0, 1.0, size=D)
            gens = []
            for _ in range(K):
                T = int(rng.integers(8, 21))
                mat = rng.normal(0, 0.5, size=(D, T))
                mat[:, -1] = base + rng.normal(0, dispersion, size=D)
                gens.append(mat)
            return {"id": item_id, "label": label, "generations": gens}

        # Build splits
        val_data = [
            make_item(f"val-non-{i}", 0, disp_non) for i in range(n_per_class)
        ] + [make_item(f"val-hal-{i}", 1, disp_hal) for i in range(n_per_class)]
        test_data = [
            make_item(f"test-non-{i}", 0, disp_non) for i in range(n_per_class)
        ] + [make_item(f"test-hal-{i}", 1, disp_hal) for i in range(n_per_class)]

        # 1) Calibrate τ on validation (your train should set self.threshold)
        self.train(train_data=val_data, validation_data=val_data)
        tau = getattr(self, "threshold", None)

        # 2) Evaluate on validation to report G-Mean/TPR/FPR at τ
        y_val = np.array([it["label"] for it in val_data], dtype=int)
        yv_pred, yv_scores = self.predict(val_data)  # uses current τ internally
        tn, fp, fn, tp = confusion_matrix(y_val, yv_pred).ravel()
        tpr_val = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr_val = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        gmean_val = np.sqrt(tpr_val * (1.0 - fpr_val))

        # 3) Evaluate on test
        y_test = np.array([it["label"] for it in test_data], dtype=int)
        y_pred, y_scores = self.predict(test_data)

        # Basic metrics
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        acc = float(np.mean(y_test == y_pred))
        cm = confusion_matrix(y_test, y_pred)

        # PR curve / AP / PR-AUC only if scores are present & non-constant
        pr_curve_p = np.array([prec], dtype=float)
        pr_curve_r = np.array([rec], dtype=float)
        ap = np.nan
        pr_auc_val = np.nan
        if (y_scores is not None) and (np.unique(y_scores).size > 1):
            p, r, _ = precision_recall_curve(y_test, y_scores)
            pr_auc_val = auc(r, p)
            ap = average_precision_score(y_test, y_scores)
            pr_curve_p, pr_curve_r = p, r

        # Per-item outputs
        items = [
            {
                "id": test_data[i].get("id"),
                "label": int(y_test[i]),
                "pred": int(y_pred[i]),
                "score": (None if y_scores is None else float(y_scores[i])),
            }
            for i in range(len(y_test))
        ]

        # Log a short summary
        tau_str = "None" if tau is None else f"{float(tau):.6f}"
        logging.info(
            f"[Self-test] τ={tau_str} | "
            f"Val: G-Mean={gmean_val:.3f} TPR={tpr_val:.3f} FPR={fpr_val:.3f} | "
            f"Test: Acc={acc:.3f} Prec={prec:.3f} Rec={rec:.3f} F1={f1:.3f}"
        )

        return {
            "threshold": tau,
            "val_stats": {"gmean": gmean_val, "tpr": tpr_val, "fpr": fpr_val},
            "metrics": {
                "accuracy": acc,
                "precision": prec,
                "recall": rec,
                "f1": f1,
                "confusion_matrix": cm,
                "average_precision": ap,
                "pr_auc": pr_auc_val,
                "pr_curve_precision": pr_curve_p,
                "pr_curve_recall": pr_curve_r,
            },
            "items": items,
        }

    def train(self, train_data, validation_data=None):
        """
        Calibrate threshold τ on *train_data* (labels required).
        If validation_data is provided, compute validation stats at the fixed τ
        (for monitoring only). No re-tuning on validation.
        """
        if not train_data:
            logging.warning(
                "[EigenScore] train_data is empty; τ will be picked later on-the-fly."
            )
            return

        # --- collect train scores/labels ---
        tr_scores, tr_labels = [], []
        # --- collect train scores/labels ---
        tr_scores, tr_labels = [], []
        for item in train_data:  # remove tqdm(...) for speed
            gens = item.get("embedding_eigenscore", item.get("generations"))
            if gens is None:
                continue

            if isinstance(gens, np.ndarray):
                if gens.ndim == 2:
                    r, c = gens.shape
                    if r > c:  # likely (D, T) single gen → take last token
                        vecs = [gens[:, -1].astype(np.float32, copy=False)]
                    else:  # likely (K, D) → each row is a generation
                        vecs = [row.astype(np.float32, copy=False) for row in gens]
                elif gens.ndim == 1:
                    vecs = [gens.astype(np.float32, copy=False)]
                else:
                    vecs = []
            else:
                vecs = [self._to_last_token_vec(g) for g in gens if g is not None]
                vecs = [v.astype(np.float32, copy=False) for v in vecs if v is not None]

            score, _ = self._eigenscore_last_token(vecs, alpha=self.alpha)
            tr_scores.append(float(score))
            tr_labels.append(int(item.get("label", item.get("Hallucinating"))))

        tr_scores = np.asarray(tr_scores, dtype=float)
        tr_labels = np.asarray(tr_labels, dtype=int)

        # --- calibrate τ on TRAIN via G-Mean ---
        tau, tr_stats = gmean_threshold(tr_scores, tr_labels)
        self.threshold = tau
        self._train_stats_ = tr_stats

        # --- optional: validation monitoring at fixed τ ---
        if validation_data:
            val_scores, val_labels = [], []
            for item in validation_data:
                gens = item.get("embedding_eigenscore", item.get("generations"))
                if gens is None:
                    continue
                if isinstance(gens, np.ndarray):
                    vecs = (
                        [row.astype(np.float64, copy=False) for row in gens]
                        if gens.ndim == 2
                        else [gens.astype(np.float64, copy=False)]
                    )
                else:
                    vecs = [self._to_last_token_vec(g) for g in gens if g is not None]
                    vecs = [v for v in vecs if v is not None]
                s, _ = self._eigenscore_last_token(vecs, alpha=self.alpha)
                val_scores.append(float(s))
                val_labels.append(int(item.get("label", item.get("Hallucinating"))))

            val_scores = np.asarray(val_scores, dtype=float)
            val_labels = np.asarray(val_labels, dtype=int)

            pred = val_scores >= tau
            P = (val_labels == 1).sum()
            N = (val_labels == 0).sum()
            tp = int((pred & (val_labels == 1)).sum())
            fp = int((pred & (val_labels == 0)).sum())
            tpr = tp / P if P else 0.0
            fpr = fp / N if N else 0.0
            gmean = (tpr * (1 - fpr)) ** 0.5
            self._val_stats_ = {"gmean": gmean, "tpr": tpr, "fpr": fpr}

        logging.info(
            f"[EigenScore] Calibrated τ on TRAIN = {float(self.threshold):.6f}"
        )

    def predict(self, data):
        """
        Returns:
            y_pred   : (n,) int array in {0,1}
            y_scores : (n,) float array of EigenScores (higher = more positive)
        """
        if data is None:
            raise ValueError("predict(data=None): pass an iterable of samples.")

        scores, labels = [], []

        for s in data:
            # Prefer the dataloader's key; fall back to 'generations'
            gens = None
            lab = None
            if isinstance(s, dict):
                gens = s.get("embedding_eigenscore", s.get("generations"))
                lab = s.get("label", s.get("Hallucinating"))
            elif isinstance(s, (list, tuple)) and len(s) >= 1:
                gens = s[0]
                lab = s[1] if len(s) > 1 else None
            else:
                gens = getattr(
                    s, "embedding_eigenscore", getattr(s, "generations", None)
                )
                lab = getattr(s, "label", getattr(s, "y", None))

            if gens is None:
                # Nothing to score for this sample
                continue

            # Normalize input to a list of (D,) vectors
            if isinstance(gens, np.ndarray):
                if gens.ndim == 2:
                    r, c = gens.shape
                    if r > c:  # (D, T)
                        vecs = [gens[:, -1].astype(np.float32, copy=False)]
                    else:  # (K, D)
                        vecs = [row.astype(np.float32, copy=False) for row in gens]
                elif gens.ndim == 1:
                    vecs = [gens.astype(np.float32, copy=False)]
                else:
                    vecs = []
            else:
                vecs = [self._to_last_token_vec(g) for g in gens if g is not None]
                vecs = [v.astype(np.float32, copy=False) for v in vecs if v is not None]

            score, _ = self._eigenscore_last_token(vecs, alpha=self.alpha)
            scores.append(float(score))
            labels.append(None if lab is None else int(lab))

        scores = np.asarray(scores, dtype=float)

        # threshold selection
        tau = getattr(self, "threshold", None)
        if tau is None:
            lbl = np.array(labels, dtype=float)  # may include NaN
            mask = np.isfinite(lbl)
            if mask.sum() >= 2 and np.unique(lbl[mask].astype(int)).size == 2:
                tau, _ = gmean_threshold(scores[mask], lbl[mask].astype(int))
            else:
                tau = float(np.nanmedian(scores))

        preds = (scores >= tau).astype(int)
        return preds, scores

    # --- helpers ---
    @staticmethod
    def _to_numpy(x) -> np.ndarray:
        try:
            import torch

            if isinstance(x, torch.Tensor):
                return x.detach().cpu().numpy()
        except Exception:
            pass
        return np.asarray(x)

    def _to_last_token_vec(self, arr) -> np.ndarray | None:
        a = self._to_numpy(arr)
        if a.ndim == 1:
            return a.astype(np.float32, copy=False)
        if a.ndim == 2:
            D, T = a.shape
            if D >= 1 and T >= 1:
                return a[:, -1].astype(np.float32, copy=False)
        return None

    @staticmethod
    def _eigenscore_last_token(
        last_token_vecs: list[np.ndarray], alpha: float = 1e-3
    ) -> tuple[float, np.ndarray | None]:
        """
        Fast EigenScore on K last-token vectors (each shape (D,)).
        Builds Σ = Z_c^T Z_c where Z has shape (D, K) with columns z_i,
        and Z_c = Z with per-column mean removed (centering over d).
        Complexity ~ O(d K^2 + K^3), no D×D eigen-decomp.
        """
        K = len(last_token_vecs)
        if K < 2:
            return float("nan"), None

        # Stack as H: (K, D) then center each row (equivalent to Z centering over d)
        H = np.vstack(last_token_vecs).astype(np.float32, copy=False)  # (K, D)
        Hc = H - H.mean(axis=1, keepdims=True)  # center over feature dim

        # Σ = Z_c^T Z_c = Hc @ Hc^T  (K×K)
        Sigma = Hc @ Hc.T  # (K, K)

        # Regularize and take eigenvalues (PSD -> eigh)
        eigvals = np.linalg.eigvalsh(Sigma + alpha * np.eye(K, dtype=Sigma.dtype))
        eigvals = np.clip(eigvals, 1e-12, None)

        score = float(
            np.mean(np.log10(eigvals))
        )  # Eq. (6) up to 1/K factor; averaging does the same
        return score, eigvals


def gmean_threshold(scores, labels):
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    if scores.size == 0 or np.unique(labels).size < 2:
        return None, {"gmean": np.nan, "tpr": np.nan, "fpr": np.nan}

    fpr, tpr, thr = roc_curve(labels, scores)
    g = np.sqrt(tpr * (1.0 - fpr))
    i = int(np.argmax(g))
    return float(thr[i]), {
        "gmean": float(g[i]),
        "tpr": float(tpr[i]),
        "fpr": float(fpr[i]),
    }


if __name__ == "__main__":
    detector = EigenScoreLastToken(alpha=1e-3)
    result = detector.self_test_with_dummy_data()
