# =============================================
# Experiment 5 — Abblation study for w-parameter for proposed methods
# =============================================

import copy
import datetime
import logging
import os
import time
from collections.abc import Iterable
from pathlib import Path

import pandas as pd
import torch
from utils import feature_abstraction, run_experiments_utils

# ---------- Torch / Env parity ----------
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

logger = logging.getLogger("experiments")

EMBEDDING_SIZE_MAPPING = {"gptoss_20b": 10, "llama2_13b": 5120, "t5_gemma": 10}


def _stratified_head(items: list[dict], percent: int) -> list[dict]:
    """Take a stratified subset of the training list by label."""
    if percent >= 100:
        return items
    idx_pos = [
        i
        for i, s in enumerate(items)
        if int(s.get("label", s.get("Hallucinating"))) == 1
    ]
    idx_neg = [
        i
        for i, s in enumerate(items)
        if int(s.get("label", s.get("Hallucinating"))) == 0
    ]
    # how many to take from each
    take_pos = max(1, int(len(idx_pos) * percent / 100))
    take_neg = max(1, int(len(idx_neg) * percent / 100))
    # keep order stable but proportional
    chosen = set(idx_pos[:take_pos] + idx_neg[:take_neg])
    return [s for i, s in enumerate(items) if i in chosen]


def run_experiment_w_abblation(
    out_dir: str,
    datasets: list[str],
    methods: list[str],
    use_pretrained_embeddings: bool = False,
    w: Iterable[int] | None = None,
    on_server: bool = False,
):
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s: %(message)s"
    )

    out_dir = Path(out_dir)
    (out_dir / "models").mkdir(parents=True, exist_ok=True)
    (out_dir / "logs").mkdir(parents=True, exist_ok=True)

    metrics_rows, pr_curves, confusions = [], [], []

    def checkpoint(reason: str):
        logger.info(f"[checkpoint] writing partial results ({reason})")
        try:
            run_experiments_utils.write_excels(
                out_dir, metrics_rows, pr_curves, confusions
            )
        except Exception:
            logger.exception("[checkpoint] write_excels failed")

    for dataset_id in datasets:
        llm, benchmark = run_experiments_utils.parse_dataset_id(dataset_id)
        logger.info(f"[Exp4] DatasetID={dataset_id} -> llm={llm} benchmark={benchmark}")

        # Load splits AS PROVIDED
        train_data, validation_data, test_data = run_experiments_utils.load_dataset(
            benchmark,
            llm=llm,
            use_pretrained_embeddings=use_pretrained_embeddings,
            on_server=on_server,
        )

        # Keep pristine bases; avoid cross-contamination
        base_train = copy.deepcopy(train_data)
        base_val = copy.deepcopy(validation_data)
        base_test = copy.deepcopy(test_data)

        for method_name in methods:
            model_tag = method_name
            embedding_size = EMBEDDING_SIZE_MAPPING[llm]
            for perc in w:
                try:
                    # ---- choose subset of TRAIN ONLY (stratified) ----
                    train_subset = _stratified_head(base_train, int(perc))

                    # fresh copies for this run
                    t_train = copy.deepcopy(train_subset)
                    t_val = copy.deepcopy(base_val)
                    t_test = copy.deepcopy(base_test)

                    logger.info(
                        f"[Exp4] {dataset_id} [{method_name}] using {len(t_train)}/{len(base_train)} "
                        f"train samples ({perc}%)"
                    )

                    method = run_experiments_utils.get_detection_method(
                        benchmark, method_name, llm, embedding_size
                    )

                    # Feature abstraction only for HHMM/PoLLMgraph/LUNA
                    if method_name in ["LUNA", "HHMM", "PoLLMgraph"]:
                        t_train, t_val, t_test = (
                            feature_abstraction.abstract_hhmm_features(
                                data=[t_train, t_val, t_test],
                                train=True,
                                benchmark=benchmark,
                                llm=llm,
                                experiment=f"exp4_{perc}",
                            )
                        )

                    # -----------------------
                    # TRAIN (timed)
                    # -----------------------
                    t0 = time.perf_counter()
                    if hasattr(method, "set_data") and hasattr(method, "fit"):
                        method.set_data(t_train, t_val, t_test, batch_size=32)
                        method.fit(epochs=20, lr=1e-3)
                    elif hasattr(method, "train"):
                        method.train(train_data=t_train, validation_data=t_val)
                    else:
                        logger.warning(
                            f"[{model_tag}] No recognized train interface; skipping train."
                        )
                    train_seconds = time.perf_counter() - t0
                    logger.info(
                        f"[Exp4] {dataset_id} [{method_name}] p{perc} train_seconds={train_seconds:.3f}s"
                    )

                    # Save model (include percent in name)
                    run_experiments_utils.save_model(
                        method,
                        out_dir / "models",
                        f"{dataset_id}__{model_tag}__p{perc}",
                    )

                    # -----------------------
                    # INFERENCE (timed)
                    # -----------------------
                    t0 = time.perf_counter()
                    y_true, y_scores, y_pred, subtypes, subtype_vocab = (
                        run_experiments_utils.collect_scores_and_labels(method, t_test)
                    )
                    infer_seconds = time.perf_counter() - t0
                    logger.info(
                        f"[Exp4] {dataset_id} [{method_name}] p{perc} infer_seconds={infer_seconds:.3f}s"
                    )

                    # Evaluate / Metrics
                    m = run_experiments_utils.compute_metrics(y_true, y_scores, y_pred)

                    run_experiments_utils.save_cm_json(
                        out_dir=out_dir,
                        exp=f"exp4_{perc}",
                        dataset=dataset_id,
                        train_tag=f"{dataset_id}__p{perc}",
                        model=model_tag,
                        cm=m["confusion_matrix"],
                    )

                    # store percent + timing + sizes in the table
                    metrics_rows.append(
                        {
                            "experiment": "exp4",
                            "dataset": dataset_id,
                            "train": f"{dataset_id}__p{perc}",
                            "test": dataset_id,
                            "model": model_tag,
                            "percent": perc,
                            "precision": m["precision"],
                            "recall": m["recall"],
                            "f1": m["f1"],
                            "average_precision": m["average_precision"],
                            "pr_auc": m["pr_auc"],
                            "train_seconds": float(train_seconds),
                            "infer_seconds": float(infer_seconds),
                            "total_seconds": float(train_seconds + infer_seconds),
                            "n_train": len(t_train),
                            "n_val": len(t_val),
                            "n_test": len(t_test),
                        }
                    )

                    pr_curves.append(
                        (
                            "exp4",
                            dataset_id,
                            f"{dataset_id}__p{perc}",
                            model_tag,
                            pd.DataFrame(
                                {
                                    "recall": m["pr_curve_recall"],
                                    "precision": m["pr_curve_precision"],
                                }
                            ),
                        )
                    )
                    confusions.append(
                        (
                            "exp4",
                            dataset_id,
                            f"{dataset_id}__p{perc}",
                            model_tag,
                            m["confusion_matrix"],
                        )
                    )

                    checkpoint(reason=f"{dataset_id}/{method_name}/p{perc} complete")

                except Exception as e:
                    logger.exception(
                        f"[Exp4] {dataset_id} -> [{method_name}] p{perc} FAILED"
                    )
                    fail_log = (
                        out_dir
                        / "logs"
                        / f"fail__{dataset_id}__{method_name}__p{perc}.log"
                    )
                    try:
                        fail_log.write_text(
                            f"{type(e).__name__}: {e}\n", encoding="utf-8"
                        )
                    except Exception:
                        logger.exception("[fail-log] could not write failure log")
                    checkpoint(reason=f"{dataset_id}/{method_name}/p{perc} failed")

    run_experiments_utils.write_excels(out_dir, metrics_rows, pr_curves, confusions)


# ===== Hard-coded config (no CLI) =====
now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
OUT_DIR = f"results/exp5_{now}"

DATASETS = [
    "gptoss_20b_tqa",
    "gptoss_20b_eli5",
    "t5_gemma_tqa",
    "t5_gemma_eli5",
]

METHODS = [
    "GraphNN",
    "BertGraphNN",
    "GraphNNStruct",
]

USE_PRETRAINED_EMBEDDINGS = False
SEED = 42


def main():
    run_experiment_w_abblation(
        out_dir=OUT_DIR,
        datasets=DATASETS,
        methods=METHODS,
        use_pretrained_embeddings=USE_PRETRAINED_EMBEDDINGS,
        w=[1, 2, 4, 10, 15, 20],
    )


if __name__ == "__main__":
    main()
