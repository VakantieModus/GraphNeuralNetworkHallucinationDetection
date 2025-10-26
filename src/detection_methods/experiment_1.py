# =============================================
# Experiment 1 — Normal training (70/30)
# with per-method fault isolation + incremental checkpoints
# =============================================

import argparse
import copy
import datetime
import logging
import os
import time
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


def run_experiment1(
    out_dir: str,
    datasets: list[str],
    methods: list[str],
    use_pretrained_embeddings: bool = False,
    on_server: bool = False,
):
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s: %(message)s"
    )

    out_dir = Path(out_dir)
    (out_dir / "models").mkdir(parents=True, exist_ok=True)
    (out_dir / "logs").mkdir(parents=True, exist_ok=True)

    # running aggregations we checkpoint as we go
    metrics_rows, pr_curves, confusions = [], [], []
    subtype_rows = []

    # In checkpoint(), also persist subtype rows
    def checkpoint(reason: str):
        logger.info(f"[checkpoint] writing partial results ({reason})")
        try:
            run_experiments_utils.write_excels(
                out_dir, metrics_rows, pr_curves, confusions, subtype_rows=subtype_rows
            )
        except Exception:
            logger.exception("[checkpoint] write_excels failed")

    for dataset_id in datasets:
        llm, benchmark = run_experiments_utils.parse_dataset_id(dataset_id)
        logger.info(f"[Exp1] DatasetID={dataset_id} -> llm={llm} benchmark={benchmark}")

        # Load splits AS PROVIDED
        train_data, validation_data, test_data = run_experiments_utils.load_dataset(
            benchmark,
            llm=llm,
            use_pretrained_embeddings=use_pretrained_embeddings,
            on_server=on_server,
        )

        # Keep pristine copies to avoid cross-method contamination
        base_train = copy.deepcopy(train_data)
        base_val = copy.deepcopy(validation_data)
        base_test = copy.deepcopy(test_data)

        for method_name in methods:
            train_tag = dataset_id
            model_tag = method_name

            try:
                logger.info(f"[{dataset_id}] -> [{method_name}] starting")

                method = run_experiments_utils.get_detection_method(
                    benchmark, method_name, llm
                )

                # reset splits per method to avoid bleed-over
                train_data = copy.deepcopy(base_train)
                validation_data = copy.deepcopy(base_val)
                test_data = copy.deepcopy(base_test)

                # Feature abstraction only for HHMM/PoLLMgraph/LUNA
                if method_name in ["LUNA", "HHMM", "PoLLMgraph"]:
                    train_data, validation_data, test_data = (
                        feature_abstraction.abstract_hhmm_features(
                            data=[train_data, validation_data, test_data],
                            train=True,
                            benchmark=benchmark,
                            llm=llm,
                            experiment="exp1",
                        )
                    )

                # -----------------------
                # TRAIN (timed)
                # -----------------------
                t0 = time.perf_counter()
                if hasattr(method, "set_data") and hasattr(method, "fit"):
                    method.set_data(
                        train_data, validation_data, test_data, batch_size=32
                    )
                    method.fit(epochs=10, lr=1e-3)
                elif hasattr(method, "train"):
                    method.train(train_data=train_data, validation_data=validation_data)
                else:
                    logger.warning(
                        f"[{model_tag}] No recognized train interface; skipping train."
                    )
                train_seconds = time.perf_counter() - t0
                logger.info(
                    f"[{dataset_id}] -> [{method_name}] train_seconds={train_seconds:.3f}s"
                )

                # Save model (flat path)
                run_experiments_utils.save_model(
                    method, out_dir / "models", f"{train_tag}__{model_tag}"
                )

                # -----------------------
                # INFERENCE (timed)
                # -----------------------
                t0 = time.perf_counter()
                y_true, y_scores, y_pred, subtypes, subtype_vocab = (
                    run_experiments_utils.collect_scores_and_labels(method, test_data)
                )
                infer_seconds = time.perf_counter() - t0
                logger.info(
                    f"[{dataset_id}] -> [{method_name}] infer_seconds={infer_seconds:.3f}s"
                )

                m = run_experiments_utils.compute_metrics(y_true, y_scores, y_pred)

                df_sub, meta = run_experiments_utils.compute_subtype_analysis(
                    y_true, y_pred, subtypes, subtype_vocab
                )

                df_sub.insert(0, "exp", "exp1")
                df_sub.insert(1, "dataset", dataset_id)
                df_sub.insert(2, "train_tag", train_tag)
                df_sub.insert(3, "model", model_tag)

                subtype_rows.append(df_sub)

                run_experiments_utils.save_cm_json(
                    out_dir=out_dir,
                    exp="exp1",
                    dataset=f"{dataset_id}",
                    train_tag=f"{train_tag}_{llm}",
                    model=model_tag,
                    cm=m["confusion_matrix"],
                )

                # attach timing + dataset sizes for Excel
                m["train_seconds"] = float(train_seconds)
                m["infer_seconds"] = float(infer_seconds)
                m["total_seconds"] = float(train_seconds + infer_seconds)
                m["n_train"] = len(train_data)
                m["n_val"] = len(validation_data)
                m["n_test"] = len(test_data)

                run_experiments_utils.append_metrics_row(
                    metrics_rows,
                    exp="exp1",
                    dataset=dataset_id,
                    train_tag=train_tag,
                    test_tag=train_tag,
                    model=model_tag,
                    m=m,
                )

                pr_curves.append(
                    (
                        "exp1",
                        dataset_id,
                        train_tag,
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
                    ("exp1", dataset_id, train_tag, model_tag, m["confusion_matrix"])
                )

                # Flush results for THIS method so later failures don't lose progress
                checkpoint(reason=f"{dataset_id}/{method_name} complete")

            except Exception as e:
                # Log a method-specific failure and continue
                logger.exception(f"[{dataset_id}] -> [{method_name}] FAILED")
                # also write a short failure log for quick triage
                fail_log = out_dir / "logs" / f"fail__{dataset_id}__{method_name}.log"
                try:
                    fail_log.write_text(f"{type(e).__name__}: {e}\n", encoding="utf-8")
                except Exception:
                    logger.exception("[fail-log] could not write failure log")
                # still checkpoint whatever we have so far
                checkpoint(reason=f"{dataset_id}/{method_name} failed")

    # Final write (in case the last loop had nothing to append after the last checkpoint)
    run_experiments_utils.write_excels(
        out_dir, metrics_rows, pr_curves, confusions, subtype_rows=subtype_rows
    )


# ===== Hard-coded config (no CLI) =====
now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
OUT_DIR = f"results/exp1_{now}"
DATASETS = [
    "llama2_13b_tqa",
    "gptoss_20b_tqa",
    "t5_gemma_tqa",
    "llama2_13b_eli5",
    "gptoss_20b_eli5",
    "t5_gemma_eli5",
]
METHODS = [
    "GraphNN",
    "GraphNNStruct",
    "GraphNNStructTunedLogit",
    "BertGraphNN",
    "EigenScoreLastToken",
    "TunedLogitLensSvm",
    "BespokeMiniCheck",
    "LettuceDetect",
    "PoLLMgraph",
]

USE_PRETRAINED_EMBEDDINGS = False


def main():
    parser = argparse.ArgumentParser(
        description="Run hallucination detection experiments."
    )

    parser.add_argument(
        "--datasets",
        nargs="+",
        default=[
            "llama2_13b_tqa",
            "gptoss_20b_tqa",
            "t5_gemma_tqa",
            "llama2_13b_eli5",
            "gptoss_20b_eli5",
            "t5_gemma_eli5",
        ],
        help="List of datasets to test (space-separated).",
    )

    parser.add_argument(
        "--methods",
        nargs="+",
        default=[
            "GraphNN",
            "GraphNNStruct",
            "GraphNNStructTunedLogit",
            "BertGraphNN",
            "EigenScoreLastToken",
            "TunedLogitLensSvm",
            "LettuceDetect",
            # "PoLLMgraph",
        ],
        help="List of detection methods to test (space-separated).",
    )

    parser.add_argument(
        "--out-dir",
        type=str,
        default=f"results/exp1_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}",
        help="Directory where experiment results are saved.",
    )

    parser.add_argument(
        "--use-pretrained-embeddings",
        action="store_true",
        help="Use pretrained embeddings instead of re-generating them.",
    )
    parser.add_argument(
        "--on-server",
        dest="on_server",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run experiments on a server (affects paths/resources).",
    )

    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_experiment1(
        out_dir=str(out_dir),
        datasets=args.datasets,
        methods=args.methods,
        use_pretrained_embeddings=False,
        on_server=args.on_server,
    )


if __name__ == "__main__":
    main()
