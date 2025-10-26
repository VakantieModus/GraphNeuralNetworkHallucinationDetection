# =============================================
# Shared utilities used by all 4 experiments
# =============================================
# add near the top
import json
import logging
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    auc,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)

from src.detection_methods.bert_graphnn.bert_graphnn import (
    SentenceGATTrainer,
    SentenceGraphBuilder,
)
from src.detection_methods.bespoke_mini_check.bespoke_mini_check_ollama import (
    BespokeMiniCheckOllama,
)
from src.detection_methods.eigen_score.eigen_score import EigenScoreLastToken
from src.detection_methods.lettuce_detect.lettuce_detect import (
    LettuceConfig,
    LettuceDetectCheck,
)
from src.detection_methods.neural_models.graph_nn import GraphNNClassifier
from src.detection_methods.neural_models.graph_nn_struct import StructGraphNN
from src.detection_methods.neural_models.graph_nn_struct_tuned_logit_curves import (
    StructTunedLogitGraphNN,
)
from src.detection_methods.pollmgraph.pollmgraph_gridd_hmm import PoLLMgraphGriddHMM
from src.detection_methods.tuned_lens_svm.tuned_lens_svm import (
    TunedLensSvmHallucinationDetector,
)
from src.white_box_benchmark.data_loaders import tqa_data_loader

logger = logging.getLogger("experiments")


def _write_sheet_append(
    path: Path, sheet_name: str, df: pd.DataFrame, *, index: bool = True
):
    sheet_name = sheet_name[:31]
    if path.exists():
        with pd.ExcelWriter(
            path, engine="openpyxl", mode="a", if_sheet_exists="replace"
        ) as w:
            df.to_excel(w, sheet_name=sheet_name, index=index)
    else:
        with pd.ExcelWriter(path, engine="openpyxl") as w:
            df.to_excel(w, sheet_name=sheet_name, index=index)


def _find_project_root(start: Path | None = None) -> Path:
    """
    Walk up from `start` (or this file) to find the repo root that contains 'src/white_box_benchmark'.
    Falls back to going up 3 levels if not found.
    """
    here = (start or Path(__file__).resolve()).parent
    for anc in [here] + list(here.parents):
        if (anc / "src" / "white_box_benchmark").exists():
            return anc
    # Fallback: heuristic
    logger.warning("[paths] Heuristic fallback for project root (parents[3])")
    return Path(__file__).resolve().parents[3]


PROJECT_ROOT = _find_project_root()


# Local paths
ANNO_DIR = PROJECT_ROOT / "src" / "white_box_benchmark" / "data" / "generated_output"
TENSORS_DIR = PROJECT_ROOT / "generated_data"

# Server paths (relative to working directory on the server)
ANNO_DIR_SERVER = Path("src/white_box_benchmark/data/generated_output")
TENSORS_DIR_SERVER = Path("generated_data")

EMBEDDING_SIZE_MAPPING = {"gptoss_20b": 2880, "llama2_13b": 5120, "t5_gemma": 2048}


def pr_auc_score(y_true, y_pred_scores):
    precision, recall, _ = precision_recall_curve(y_true, y_pred_scores)
    pr_auc = auc(recall, precision)
    return pr_auc, precision, recall


def ensure_dir(path: str | Path):
    Path(path).mkdir(parents=True, exist_ok=True)


def save_model(model: Any, out_dir: str | Path, name: str):
    ensure_dir(out_dir)
    # path = Path(out_dir) / f"{name}.joblib"
    return
    # try:
    #     joblib.dump(model, path)
    #     logger.info(f"[save_model] Saved to {path}")
    # except Exception as e:
    #     logger.warning(f"[save_model] Could not save with joblib: {e}")


def compute_metrics(y_true, y_scores, y_pred):
    m = {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
    }

    if (y_scores is None) or (np.unique(y_scores).size <= 1):
        m["average_precision"] = np.nan
        m["pr_auc"] = np.nan
        m["pr_curve_precision"] = np.array([m["precision"]], dtype=float)
        m["pr_curve_recall"] = np.array([m["recall"]], dtype=float)
        return m

    prec, rec, _ = precision_recall_curve(y_true, y_scores)
    m["pr_auc"] = auc(rec, prec)
    m["average_precision"] = average_precision_score(y_true, y_scores)
    m["pr_curve_precision"] = prec
    m["pr_curve_recall"] = rec
    return m


def _slug(s: str) -> str:
    # safe filename: letters, numbers, dash/underscore only
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(s))[:120]


def save_cm_json(
    out_dir: str | Path,
    exp: str,
    dataset: str,
    train_tag: str,
    model: str,
    cm: np.ndarray | list,
):
    """Dump a single confusion matrix to results/<exp>/cm_json/<file>.json immediately."""
    base = Path(out_dir) / "cm_json"
    base.mkdir(parents=True, exist_ok=True)
    payload = {
        "exp": exp,
        "dataset": dataset,
        "train": train_tag,
        "model": model,
        "confusion_matrix": np.asarray(cm).tolist(),
    }
    fname = f"{_slug(exp)}__{_slug(dataset)}__{_slug(train_tag)}__{_slug(model)}.json"
    (base / fname).write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def collect_scores_and_labels(method, test_data):
    """
    As before, but also returns:
      - subtypes: list[list[str]] per sample
      - subtype_vocab: sorted unique subtypes present in test_data
    """
    if (
        hasattr(method, "test_loader")
        and getattr(method, "test_loader", None) is not None
    ):
        y_pred, y_scores = method.predict()
    else:
        y_pred, y_scores = method.predict(test_data)

    y_true = np.asarray(
        [int(s.get("label", s.get("Hallucinating"))) for s in test_data], dtype=int
    )
    if y_scores is not None:
        y_scores = np.asarray(y_scores)

    # Robustly extract subtypes (multi-label stored as list; allow "", None, or string)
    subtypes = []
    for s in test_data:
        st = s.get("hallu_sub_type", s.get("hallu_subtypes", []))
        if st is None:
            st = []
        elif isinstance(st, str):
            st = [st] if st.strip() else []
        elif isinstance(st, (tuple, set)):
            st = list(st)
        # sanitize whitespace
        st = [x.strip() for x in st if isinstance(x, str) and x.strip()]
        subtypes.append(st)

    subtype_vocab = sorted({t for lst in subtypes for t in lst})

    return y_true, y_scores, np.asarray(y_pred), subtypes, subtype_vocab


def compute_subtype_analysis(y_true, y_pred, subtypes, subtype_vocab):
    """
    y_true: (N,) 0/1
    y_pred: (N,) 0/1
    subtypes: list[list[str]] per sample (subtype labels for positives; negatives should be empty)
    subtype_vocab: list[str]

    Returns:
      - df (tidy): columns = ['subtype','support','frequency_pct','recall','miss_rate','fn_count']
      - meta: dict with totals for convenience
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    N = len(y_true)
    pos_idx = np.where(y_true == 1)[0]
    total_pos = len(pos_idx)

    # Count per-subtype support and FN/TP (multi-label; each positive contributes to all its subtypes)
    rows = []
    for s in subtype_vocab:
        # indices of *true positives* that are annotated with subtype s
        idx_s = [i for i in pos_idx if s in subtypes[i]]
        support = len(idx_s)
        if support == 0:
            rows.append(
                {
                    "subtype": s,
                    "support": 0,
                    "frequency_pct": 0.0,
                    "recall": np.nan,
                    "miss_rate": np.nan,
                    "fn_count": 0,
                }
            )
            continue
        tp_s = sum(1 for i in idx_s if y_pred[i] == 1)
        fn_s = support - tp_s
        recall_s = tp_s / support
        rows.append(
            {
                "subtype": s,
                "support": support,
                "frequency_pct": (
                    (support / total_pos * 100.0) if total_pos > 0 else 0.0
                ),
                "recall": recall_s,
                "miss_rate": 1.0 - recall_s,
                "fn_count": fn_s,
            }
        )

    df = pd.DataFrame(rows).sort_values("subtype").reset_index(drop=True)
    meta = {
        "total_samples": int(N),
        "total_positives": int(total_pos),
        "total_negatives": int(N - total_pos),
        "overall_recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "overall_precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "overall_f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    return df, meta


def append_metrics_row(
    table: list[dict[str, Any]],
    *,
    exp: str,
    dataset: str,
    train_tag: str,
    test_tag: str,
    model: str,
    m: dict[str, Any],
):
    row = {
        "experiment": exp,
        "dataset": dataset,
        "train": train_tag,
        "test": test_tag,
        "model": model,
        "precision": m["precision"],
        "recall": m["recall"],
        "f1": m["f1"],
        "average_precision": m["average_precision"],
        "pr_auc": m["pr_auc"],
    }
    # Optional extras (added by caller)
    for k in (
        "train_seconds",
        "infer_seconds",
        "total_seconds",
        "n_train",
        "n_val",
        "n_test",
    ):
        if k in m:
            row[k] = m[k]

    # NEW: put confusion matrix in a single cell as JSON like [[TN,FP],[FN,TP]]
    cm = m.get("confusion_matrix")
    if cm is not None:
        try:
            if isinstance(cm, np.ndarray):
                row["confusion_matrix"] = json.dumps(cm.tolist())
            else:
                # list-like or something else serializable
                row["confusion_matrix"] = json.dumps(cm)
        except Exception:
            # last resort
            row["confusion_matrix"] = str(cm)

    table.append(row)


def write_excels(
    out_dir: str | Path,
    metrics_rows: list[dict[str, Any]],
    pr_curves: list[tuple[str, str, str, str, pd.DataFrame]],
    confusions: list[
        tuple[str, str, str, str, np.ndarray]
    ],  # kept for signature compat; unused now
    subtype_rows=None,
):
    ensure_dir(out_dir)
    metrics_df = pd.DataFrame(metrics_rows)

    metrics_path = Path(out_dir) / "results.xlsx"
    with pd.ExcelWriter(metrics_path, engine="xlsxwriter") as w:
        if not metrics_df.empty:
            preferred = [
                "experiment",
                "dataset",
                "train",
                "test",
                "model",
                "precision",
                "recall",
                "f1",
                "average_precision",
                "pr_auc",
                "train_seconds",
                "infer_seconds",
                "total_seconds",
                "n_train",
                "n_val",
                "n_test",
                "confusion_matrix",  # <- NEW column in metrics
            ]
            cols = [c for c in preferred if c in metrics_df.columns] + [
                c for c in metrics_df.columns if c not in preferred
            ]
            metrics_df[cols].to_excel(w, sheet_name="metrics", index=False)

            if {"train_seconds", "infer_seconds", "total_seconds"}.issubset(
                metrics_df.columns
            ):
                timings_cols = [
                    c
                    for c in [
                        "dataset",
                        "model",
                        "train",
                        "test",
                        "train_seconds",
                        "infer_seconds",
                        "total_seconds",
                        "n_train",
                        "n_val",
                        "n_test",
                    ]
                    if c in metrics_df.columns
                ]
                metrics_df[timings_cols].sort_values(["dataset", "model"]).to_excel(
                    w, sheet_name="timings", index=False
                )
        else:
            pd.DataFrame(
                columns=["experiment", "dataset", "train", "test", "model"]
            ).to_excel(w, sheet_name="metrics", index=False)
        if subtype_rows:
            df_all = pd.concat(subtype_rows, ignore_index=True)
            # Optional: add a weighted mean row per (exp,dataset,model)
            # but usually it’s clearer to keep it tidy and let LaTeX aggregate.
            df_all.to_excel(w, sheet_name="SubtypeAnalysis", index=False)
        # PR curve sheets stay as separate sheets (unchanged)
        for exp, dataset, train_tag, model, df_curve in pr_curves:
            sheet = _safe_sheet(f"PR_{dataset}_{train_tag}_{model}")
            df_curve.to_excel(w, sheet_name=sheet, index=False)

    logger.info(
        f"[write_excels] Wrote metrics (+timings + CM column) & PR curves to {metrics_path}"
    )


def _safe_sheet(name: str) -> str:
    bad = "[]:*?/\\"
    for ch in bad:
        name = name.replace(ch, "_")
    return name[:31]


def load_dataset(
    dataset_name: str,
    llm: str,
    use_pretrained_embeddings=False,
    on_server=False,
    text_only=False,
):
    # Resolve base dirs
    if on_server:
        base_tensors = TENSORS_DIR_SERVER
        base_annos = ANNO_DIR_SERVER
    else:
        base_tensors = TENSORS_DIR
        base_annos = ANNO_DIR

    # Build paths
    path_tensors = base_tensors / f"{llm}_{dataset_name}_npy"
    path_curves = base_tensors / f"curves_{llm}_{dataset_name}"
    prim = base_annos / f"enriched_{llm}_{dataset_name}.json"
    alt = base_annos / f"v1_enriched_{llm}_{dataset_name}.json"

    # Logging (fix typo)
    logger.info(f"[paths] PROJECT_ROOT={PROJECT_ROOT}")
    logger.info(f"[paths] ANNO_DIR={'(server) ' if on_server else ''}{base_annos}")
    logger.info(f"[paths] TENSORS_DIR={'(server) ' if on_server else ''}{base_tensors}")
    logger.info(f"[paths] CURVES_DIR={path_curves}")

    for path_annotations in (prim, alt):
        if path_annotations.is_file():
            break
    else:
        raise FileNotFoundError(
            "Annotation file not found. Tried:\n  " + "\n  ".join(map(str, (prim, alt)))
        )
    if llm in [
        "llama2_13b",
        "qwen_3_14b",
    ]:
        last_layer = 39
    elif llm in ["gptoss_20b"]:
        last_layer = 23
    elif llm in ["t5_gemma"]:
        last_layer = 24
    else:
        raise Exception(f"Unknown llm: {llm}")

    if use_pretrained_embeddings:
        data_loader = tqa_data_loader.PretrainedEmbeddingLoader(
            str(path_annotations), model_name="bert-base-uncased"
        )
    else:

        assert path_tensors.is_dir(), f"Tensor directory not found: {path_tensors}"
        assert path_curves.is_dir(), f"Curves directory not found: {path_curves}"

        data_loader = tqa_data_loader.DataLoader(
            str(path_tensors),
            str(path_annotations),
            last_layer=last_layer,
            curves_dir=str(path_curves),
        )

    return data_loader.load_data_set(text_only=text_only)


def parse_dataset_id(dataset_id: str) -> tuple[str, str]:
    # dataset_id is "{llm}_{benchmark}" and llm may contain underscores
    llm, benchmark = dataset_id.rsplit("_", 1)
    return llm, benchmark


def get_detection_method(
    dataset_name: str, detection_method: str, llm_name: str, embedding_size: int = None
):
    embedding_size = EMBEDDING_SIZE_MAPPING[llm_name]

    logger.info(
        f"Initializing detection method '{detection_method}' for dataset '{dataset_name}'"
    )
    if detection_method == "PoLLMgraph":
        return PoLLMgraphGriddHMM(dataset=dataset_name)
    elif detection_method == "EigenScoreLastToken":
        return EigenScoreLastToken()
    elif detection_method == "GraphNN":
        return GraphNNClassifier(input_dim=embedding_size + 1)
    elif detection_method == "BertGraphNN":
        builder = SentenceGraphBuilder(
            model_name="bert-base-uncased",
            add_positional=True,
            tau=0.85,
            top_k=8,
            max_sentences=40,  # optional cap
        )

        trainer = SentenceGATTrainer(builder, gat_hidden=128, heads=4)
        return trainer
    elif detection_method == "GraphNNStructTunedLogit":
        return StructTunedLogitGraphNN(input_dim=embedding_size + 1)
    elif detection_method == "GraphNNStruct":
        return StructGraphNN(input_dim=embedding_size + 1)
    elif detection_method == "TunedLogitLensSvm":
        return TunedLensSvmHallucinationDetector(
            classifier="svm",  # or "logreg"
            seed=123,
        )

    elif detection_method == "LettuceDetect":
        return LettuceDetectCheck(
            name="lettucedetect_yesno_style",
            cfg=LettuceConfig(),
            number_of_documents=1,
        )
    elif detection_method == "BespokeMiniCheckOllama":
        return BespokeMiniCheckOllama(
            name="ollama_yesno",
        )
    else:
        raise NotImplementedError("Detection method not implemented")
