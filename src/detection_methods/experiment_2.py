# =============================================
# Experiment 2 — Domain shift (train benchmark i, test benchmark j)
# (Hardcoded config, no CLI — with fault isolation + checkpoints)
# =============================================

import argparse
import copy
import datetime
import logging
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from utils import feature_abstraction
from utils.run_experiments_utils import (
    append_metrics_row,
    collect_scores_and_labels,
    compute_metrics,
    get_detection_method,
    load_dataset,
    parse_dataset_id,
    save_cm_json,
    save_model,
    write_excels,
)

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

# Match exp1 mapping (extend as needed)
EMBEDDING_SIZE_MAPPING = {"gptoss_20b": 10, "llama2_13b": 5120, "t5_gemma": 10}


def _set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_experiment2(
    out_dir: str,
    datasets: list[tuple[str, str]],  # list of (train_id, test_id)
    methods: list[str],
    use_pretrained_embeddings: bool = False,
    seed: int = 42,
    on_server: bool = False,
):
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s: %(message)s"
    )
    _set_seed(seed)

    out_dir = Path(out_dir)
    (out_dir / "models").mkdir(parents=True, exist_ok=True)
    (out_dir / "logs").mkdir(parents=True, exist_ok=True)

    metrics_rows: list[dict] = []
    pr_curves: list[tuple[str, str, str, str, pd.DataFrame]] = []
    confusions: list[tuple[str, str, str, str, np.ndarray]] = []

    def checkpoint(reason: str):
        """Write partial results so far to disk."""
        logger.info(f"[checkpoint] writing partial results ({reason})")
        try:
            write_excels(out_dir, metrics_rows, pr_curves, confusions)
        except Exception:
            logger.exception("[checkpoint] write_excels failed")

    # Preload all unique dataset IDs referenced in the explicit pairs
    cache: dict[str, tuple[list, list, list]] = {}
    all_ids = {ds for pair in datasets for ds in pair}
    for ds_id in all_ids:
        llm, bench = parse_dataset_id(ds_id)
        logger.info(
            f"[Exp2] Preloading dataset {ds_id} -> llm={llm}, benchmark={bench}"
        )
        cache[ds_id] = load_dataset(
            bench,
            llm=llm,
            use_pretrained_embeddings=use_pretrained_embeddings,
            on_server=on_server,
        )

    # Loop over explicit (train_id, test_id) pairs
    for train_id, test_id in datasets:
        train_llm, train_bench = parse_dataset_id(train_id)
        test_llm, test_bench = parse_dataset_id(test_id)

        # base (pristine) copies so each method starts clean
        base_train, base_val, _ = cache[train_id]
        _, _, base_test = cache[test_id]
        base_train = copy.deepcopy(base_train)
        base_val = copy.deepcopy(base_val)
        base_test = copy.deepcopy(base_test)

        for method_name in methods:
            model_tag = method_name
            train_tag = train_id
            test_tag = test_id

            try:
                logger.info(
                    f"[Exp2] {train_id} -> {test_id}  |  [{method_name}] starting"
                )

                embedding_size = EMBEDDING_SIZE_MAPPING.get(train_llm, 512)
                if train_llm not in EMBEDDING_SIZE_MAPPING:
                    logger.warning(
                        f"[{method_name}] No embedding size for llm={train_llm}; defaulting to {embedding_size}."
                    )

                method = get_detection_method(
                    train_bench, method_name, train_llm, embedding_size
                )

                # fresh splits for this method
                train_data = copy.deepcopy(base_train)
                val_data = copy.deepcopy(base_val)
                test_data = copy.deepcopy(base_test)

                # Feature abstraction on the SOURCE (train) domain
                if method_name in ["LUNA", "HHMM", "PoLLMgraph"]:
                    train_data, val_data, test_data = (
                        feature_abstraction.abstract_hhmm_features(
                            data=[train_data, val_data, test_data],
                            train=False,
                            benchmark=train_bench,
                            llm=train_llm,
                            experiment="exp1",
                        )
                    )

                # -----------------------
                # TRAIN (timed)
                # -----------------------
                t0 = time.perf_counter()
                if hasattr(method, "set_data") and hasattr(method, "fit"):
                    method.set_data(train_data, val_data, test_data, batch_size=32)
                    method.fit(epochs=20, lr=1e-3, save=False)
                elif hasattr(method, "train"):
                    method.train(train_data=train_data, validation_data=val_data)
                else:
                    logger.warning(
                        f"[{model_tag}] No recognized train interface; skipping train."
                    )
                train_seconds = time.perf_counter() - t0
                logger.info(
                    f"[Exp2] {train_id}->{test_id} [{method_name}] train_seconds={train_seconds:.3f}s"
                )

                # Save model under exp2/<train_tag>
                save_model(
                    method,
                    out_dir / "models" / "exp2" / train_tag,
                    f"{model_tag}__to__{test_tag}",
                )

                # Optional evaluate hook (not timed, keep consistent with Exp1)
                if hasattr(method, "evaluate"):
                    try:
                        method.evaluate(test_data=test_data)
                    except TypeError:
                        pass

                # -----------------------
                # INFERENCE (timed)
                # -----------------------
                t0 = time.perf_counter()
                y_true, y_scores, y_pred, subtypes, subtype_vocab = (
                    collect_scores_and_labels(method, test_data)
                )
                infer_seconds = time.perf_counter() - t0
                logger.info(
                    f"[Exp2] {train_id}->{test_id} [{method_name}] infer_seconds={infer_seconds:.3f}s"
                )

                # Metrics (+ timing + sizes)
                m = compute_metrics(y_true, y_scores, y_pred)

                save_cm_json(
                    out_dir=out_dir,
                    exp="exp2",
                    dataset=f"{train_tag}_{test_tag}",
                    train_tag=f"{train_tag}_{test_tag}",
                    model=model_tag,
                    cm=m["confusion_matrix"],
                )
                m["train_seconds"] = float(train_seconds)
                m["infer_seconds"] = float(infer_seconds)
                m["total_seconds"] = float(train_seconds + infer_seconds)
                m["n_train"] = len(train_data)
                m["n_val"] = len(val_data)
                m["n_test"] = len(test_data)

                append_metrics_row(
                    metrics_rows,
                    exp="exp2",
                    dataset=f"{train_bench}->{test_bench}",
                    train_tag=train_tag,
                    test_tag=test_tag,
                    model=model_tag,
                    m=m,
                )
                pr_curves.append(
                    (
                        "exp2",
                        f"{train_bench}_to_{test_bench}",
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
                    (
                        "exp2",
                        f"{train_bench}_to_{test_bench}",
                        train_tag,
                        model_tag,
                        m["confusion_matrix"],
                    )
                )

                # Flush results for THIS method
                checkpoint(reason=f"{train_id}__{method_name}__to__{test_id} complete")

            except Exception as e:
                logger.exception(
                    f"[Exp2] {train_id} -> {test_id} | [{method_name}] FAILED"
                )
                # write a small failure log and checkpoint partials
                fail_log = (
                    out_dir
                    / "logs"
                    / f"fail__{train_id}__{method_name}__to__{test_id}.log"
                )
                try:
                    fail_log.write_text(f"{type(e).__name__}: {e}\n", encoding="utf-8")
                except Exception:
                    logger.exception("[fail-log] could not write failure log")

                checkpoint(reason=f"{train_id}__{method_name}__to__{test_id} failed")

    # Final write
    write_excels(out_dir, metrics_rows, pr_curves, confusions)


# ===== Defaults (same as your hard-coded ones) =====
DEFAULT_PAIRS: list[tuple[str, str]] = [
    ("gptoss_20b_tqa", "gptoss_20b_eli5"),
    ("gptoss_20b_eli5", "gptoss_20b_tqa"),
    ("t5_gemma_tqa", "t5_gemma_eli5"),
    ("t5_gemma_eli5", "t5_gemma_tqa"),
    ("llama2_13b_tqa", "llama2_13b_eli5"),
    ("llama2_13b_eli5", "llama2_13b_tqa"),
]

DEFAULT_METHODS = [
    "GraphNN",
    "GraphNNStruct",
    "GraphNNStructTunedLogit",
    "BertGraphNN",
    "EigenScoreLastToken",
    "TunedLogitLensSvm",
    # "PoLLMgraph",
]


def _parse_pair(s: str) -> tuple[str, str]:
    """
    Parse 'train:test' into a (train, test) tuple.
    Example: 'llama2_13b_tqa:llama2_13b_eli5'
    """
    if ":" not in s:
        raise argparse.ArgumentTypeError(
            f"Invalid pair '{s}'. Expected format 'train_id:test_id'."
        )
    train, test = s.split(":", 1)
    train, test = train.strip(), test.strip()
    if not train or not test:
        raise argparse.ArgumentTypeError(
            f"Invalid pair '{s}'. Train/test cannot be empty."
        )
    return (train, test)


def main():
    parser = argparse.ArgumentParser(
        description="Run Experiment 2 (domain shift) for hallucination detection."
    )

    # Accept pairs like:  --pairs a:b c:d ...
    parser.add_argument(
        "--pairs",
        type=_parse_pair,
        nargs="+",
        default=DEFAULT_PAIRS,
        help=(
            "Space-separated train:test dataset pairs. "
            "Example: --pairs llama2_13b_tqa:llama2_13b_eli5 gptoss_20b_tqa:gptoss_20b_eli5"
        ),
    )

    parser.add_argument(
        "--methods",
        nargs="+",
        default=DEFAULT_METHODS,
        help="List of detection methods to test (space-separated).",
    )

    parser.add_argument(
        "--out-dir",
        type=str,
        default=f"results/exp2_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}",
        help="Directory where experiment results are saved.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=87,
        help="Random seed.",
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
        help="Run on a server (may affect paths/resources). Use --no-on-server to force local.",
    )

    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # call your experiment
    run_experiment2(
        out_dir=str(out_dir),
        datasets=args.pairs,  # list[tuple[str, str]]
        methods=args.methods,
        use_pretrained_embeddings=args.use_pretrained_embeddings,
        seed=args.seed,
        on_server=args.on_server,
    )


if __name__ == "__main__":
    main()
