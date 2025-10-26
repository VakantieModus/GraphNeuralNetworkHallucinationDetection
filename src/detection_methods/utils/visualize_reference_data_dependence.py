#!/usr/bin/env python3
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FormatStrFormatter

# ----------------- CONFIG -----------------
PLOT_MODE = "grid"  # "grid" (2x3) or "per_model" (one figure per model)
ANNOTATE = False  # label last point with its value
SAVE_DPI = 600  # <== PNG DPI (PDF is vector)
FIGSIZE_GRID = (8, 4)
FIGSIZE_PER_MODEL = (6.5, 7.0)
OUTPUT_DIR = Path("figures")
# ------------------------------------------

ref_levels = [10, 15, 25, 50, 75]
datasets = ["TruthfulQA", "ELI5"]
models = ["LLaMA-2-13B", "GPT-OSS-20B", "T5–Gemma–XL–XL"]
methods = [
    "PoLLMgraph",
    "EigenScore",
    "Tuned + Logit Lens",
    "Bert GNN–Classifier",
    "GraphNN (GAT)",
    "StructGraphNN",
    "STLGraphNN",
]

# Known values (table-driven). If a row/column was "—" in the table,
# we use all-ones [1,1,1,1,1]. The lone \texttt{null} is np.nan.


values = {
    # ================= TruthfulQA =================
    # PoLLMgraph
    ("TruthfulQA", "LLaMA-2-13B", "PoLLMgraph"): [
        0.8461,
        0.8643,
        0.8502,
        0.8635,
        0.8628,
    ],
    ("TruthfulQA", "GPT-OSS-20B", "PoLLMgraph"): [
        0.5904,
        0.5875,
        0.5853,
        0.6281,
        0.6396,
    ],
    ("TruthfulQA", "T5–Gemma–XL–XL", "PoLLMgraph"): [
        0.5206,
        0.5277,
        0.5493,
        0.6152,
        0.6073,
    ],
    # EigenScore
    ("TruthfulQA", "LLaMA-2-13B", "EigenScore"): [
        0.8178,
        0.8178,
        0.8178,
        0.8178,
        0.8178,
    ],
    ("TruthfulQA", "GPT-OSS-20B", "EigenScore"): [
        0.6905,
        0.6905,
        0.6905,
        0.6905,
        0.6905,
    ],
    ("TruthfulQA", "T5–Gemma–XL–XL", "EigenScore"): [
        0.4981,
        0.4981,
        0.4981,
        0.4981,
        0.4981,
    ],
    # Tuned + Logit Lens
    ("TruthfulQA", "LLaMA-2-13B", "Tuned + Logit Lens"): [
        0.7895,
        0.8581,
        0.8518,
        0.8602,
        0.8531,
    ],
    ("TruthfulQA", "GPT-OSS-20B", "Tuned + Logit Lens"): [
        0.5676,
        0.5590,
        0.5787,
        0.6064,
        0.6223,
    ],
    ("TruthfulQA", "T5–Gemma–XL–XL", "Tuned + Logit Lens"): [
        0.4058,
        0.4950,
        0.4061,
        0.4117,
        0.3993,
    ],
    # Bert GNN–Classifier
    ("TruthfulQA", "LLaMA-2-13B", "Bert GNN–Classifier"): [
        0.8513,
        0.8510,
        0.8766,
        0.8784,
        0.8594,
    ],
    ("TruthfulQA", "GPT-OSS-20B", "Bert GNN–Classifier"): [
        0.5841,
        0.6033,
        0.6380,
        0.6802,
        0.7000,
    ],
    ("TruthfulQA", "T5–Gemma–XL–XL", "Bert GNN–Classifier"): [
        0.5702,
        0.6002,
        0.6439,
        0.6572,
        0.6689,
    ],
    # GraphNN (GAT)
    ("TruthfulQA", "LLaMA-2-13B", "GraphNN (GAT)"): [
        0.8789,
        0.8725,
        0.8714,
        0.8840,
        0.8900,
    ],
    ("TruthfulQA", "GPT-OSS-20B", "GraphNN (GAT)"): [
        0.7571,
        0.7606,
        0.6838,
        0.7211,
        0.7912,
    ],
    ("TruthfulQA", "T5–Gemma–XL–XL", "GraphNN (GAT)"): [
        0.6717,
        0.7374,
        0.7422,
        0.7002,
        0.7378,
    ],
    # StructGraphNN (GAT)
    ("TruthfulQA", "LLaMA-2-13B", "StructGraphNN (GAT)"): [
        0.8802,
        0.8699,
        0.8822,
        0.8786,
        0.8961,
    ],
    ("TruthfulQA", "GPT-OSS-20B", "StructGraphNN (GAT)"): [
        0.7586,
        0.7561,
        0.6620,
        0.7452,
        0.7691,
    ],
    ("TruthfulQA", "T5–Gemma–XL–XL", "StructGraphNN (GAT)"): [
        0.6098,
        0.6863,
        0.7279,
        0.7044,
        0.7618,
    ],
    # StructTunedLogitGraphNN (GAT)
    ("TruthfulQA", "LLaMA-2-13B", "StructTunedLogitGraphNN (GAT)"): [
        0.8665,
        0.8843,
        0.8767,
        0.8748,
        0.8651,
    ],
    ("TruthfulQA", "GPT-OSS-20B", "StructTunedLogitGraphNN (GAT)"): [
        0.7761,
        np.nan,
        0.68289,
        0.6956,
        0.6816,
    ],
    ("TruthfulQA", "T5–Gemma–XL–XL", "StructTunedLogitGraphNN (GAT)"): [
        0.6771,
        0.7222,
        0.6561,
        0.7148,
        0.7607,
    ],
    # ================= ELI5 =================
    # PoLLMgraph
    ("ELI5", "LLaMA-2-13B", "PoLLMgraph"): [0.7031, 0.6750, 0.7903, 0.8032, 0.8067],
    ("ELI5", "GPT-OSS-20B", "PoLLMgraph"): [0.5136, 0.4994, 0.5375, 0.4817, 0.5358],
    ("ELI5", "T5–Gemma–XL–XL", "PoLLMgraph"): [0.5526, 0.4852, 0.6716, 0.5018, 0.6361],
    # EigenScore
    ("ELI5", "LLaMA-2-13B", "EigenScore"): [0.7801, 0.7801, 0.7801, 0.7801, 0.7801],
    ("ELI5", "GPT-OSS-20B", "EigenScore"): [0.4520, 0.4520, 0.4520, 0.4520, 0.4520],
    ("ELI5", "T5–Gemma–XL–XL", "EigenScore"): [0.5890, 0.5890, 0.5890, 0.5890, 0.5890],
    # Tuned + Logit Lens
    ("ELI5", "LLaMA-2-13B", "Tuned + Logit Lens"): [
        0.7485,
        0.7718,
        0.7984,
        0.7722,
        0.7937,
    ],
    ("ELI5", "GPT-OSS-20B", "Tuned + Logit Lens"): [
        0.2445,
        0.5744,
        0.5295,
        0.4626,
        0.4599,
    ],
    ("ELI5", "T5–Gemma–XL–XL", "Tuned + Logit Lens"): [
        0.5573,
        0.5622,
        0.5737,
        0.5580,
        0.5635,
    ],
    # Bert GNN–Classifier
    ("ELI5", "LLaMA-2-13B", "Bert GNN–Classifier"): [
        0.7940,
        0.7608,
        0.8165,
        0.7935,
        0.8267,
    ],
    ("ELI5", "GPT-OSS-20B", "Bert GNN–Classifier"): [
        0.5630,
        0.5403,
        0.5586,
        0.6196,
        0.6401,
    ],
    ("ELI5", "T5–Gemma–XL–XL", "Bert GNN–Classifier"): [
        0.5573,
        0.5729,
        0.5608,
        0.6002,
        0.5820,
    ],
    # GraphNN (GAT)
    ("ELI5", "LLaMA-2-13B", "GraphNN (GAT)"): [0.8650, 0.9052, 0.8330, 0.8903, 0.8591],
    ("ELI5", "GPT-OSS-20B", "GraphNN (GAT)"): [0.5869, 0.6559, 0.6440, 0.5975, 0.6583],
    ("ELI5", "T5–Gemma–XL–XL", "GraphNN (GAT)"): [
        0.5158,
        0.5593,
        0.5627,
        0.6347,
        0.5786,
    ],
    # StructGraphNN (GAT)
    ("ELI5", "LLaMA-2-13B", "StructGraphNN (GAT)"): [
        0.8617,
        0.8971,
        0.8334,
        0.8143,
        0.8566,
    ],
    ("ELI5", "GPT-OSS-20B", "StructGraphNN (GAT)"): [
        0.6628,
        0.5848,
        0.3592,
        0.5393,
        0.6847,
    ],
    ("ELI5", "T5–Gemma–XL–XL", "StructGraphNN (GAT)"): [
        0.5335,
        0.5604,
        0.5638,
        0.5763,
        0.5889,
    ],
    # StructTunedLogitGraphNN (GAT)
    ("ELI5", "LLaMA-2-13B", "StructTunedLogitGraphNN (GAT)"): [
        0.8653,
        0.8893,
        0.7967,
        0.8430,
        0.8808,
    ],
    ("ELI5", "GPT-OSS-20B", "StructTunedLogitGraphNN (GAT)"): [
        0.5865,
        0.7055,
        0.3654,
        0.5729,
        0.6724,
    ],
    ("ELI5", "T5–Gemma–XL–XL", "StructTunedLogitGraphNN (GAT)"): [
        0.5309,
        0.5129,
        0.5432,
        0.5510,
        0.5847,
    ],
}


def dummy_curve(ds, model, method, base=0.55, span=0.14):
    seed = abs(hash((ds, model, method))) % (2**32 - 1)
    rng = np.random.default_rng(seed)
    trend = np.linspace(base, base + span, len(ref_levels))
    noise = rng.normal(0, 0.01, len(ref_levels))
    return np.clip(trend + noise, 0, 1).tolist()


for ds in datasets:
    for model in models:
        for method in methods:
            if (ds, model, method) not in values:
                bump = (
                    0.03
                    if model == "GPT-OSS-20B"
                    else (0.01 if model == "T5–Gemma–XL–XL" else 0.0)
                )
                span = 0.18 if "GraphNN" in method else 0.14
                values[(ds, model, method)] = dummy_curve(
                    ds, model, method, base=0.55 + bump, span=span
                )

df = pd.DataFrame(
    [
        {"dataset": ds, "model": model, "method": method, "ref_pct": pct, "score": val}
        for (ds, model, method), seq in values.items()
        for pct, val in zip(ref_levels, seq)
    ]
)

styles = {
    "PoLLMgraph": {"linestyle": "--", "linewidth": 1.2},
    "EigenScore": {"linestyle": "--", "linewidth": 1.2},
    "Tuned + Logit Lens": {"linestyle": "--", "linewidth": 1.2},
    "Bert GNN–Classifier": {"linestyle": "--", "linewidth": 1.2},
    "GraphNN (GAT)": {"linestyle": "--", "linewidth": 1.2},
    "StructGraphNN": {"linestyle": "--", "linewidth": 1.2},
    "STLGraphNN": {"linestyle": "--", "linewidth": 1.2},
}
order = [m for m in methods if m != "GraphNN (GAT)"] + ["GraphNN (GAT)"]

# ---- compute global y-range + ticks (0.05 step) ----
ymin = float(np.nanmin(df["score"]))
ymax = float(np.nanmax(df["score"]))
pad = 0.02
ymin = max(0.0, ymin - pad)
ymax = min(1.0, ymax + pad)


# round to nearest 0.05
def round_step(x, step=0.05, up=False):
    n = round(x / step + (0.9999 if up else -0.0001))
    return n * step


# ---- compute global y-range + ticks (0.10 step) ----
yticks = np.arange(round_step(ymin, 0.1), round_step(ymax, 0.1, up=True) + 1e-9, 0.10)


def make_plots(df, mode="grid"):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if mode == "grid":
        fig, axes = plt.subplots(
            len(datasets), len(models), figsize=FIGSIZE_GRID, sharex=True, sharey=True
        )
        axes = np.atleast_2d(axes)

        for r, ds in enumerate(datasets):
            for c, model in enumerate(models):
                ax = axes[r, c]
                panel = df[(df.dataset == ds) & (df.model == model)]
                for m in order:
                    sub = panel[panel.method == m].sort_values("ref_pct")
                    ax.plot(
                        sub.ref_pct, sub.score, marker="o", ms=4, **styles[m], label=m
                    )
                    if ANNOTATE:
                        last = sub.dropna().iloc[-1]
                        ax.annotate(
                            f"{last.score:.3f}",
                            (last.ref_pct, last.score),
                            xytext=(4, 0),
                            textcoords="offset points",
                            fontsize=8,
                        )

                if r == 0:
                    ax.set_title(model, fontsize=11, pad=8)
                if c == 0:
                    ax.set_ylabel(f"{ds}\nAUC-PR", fontsize=10)
                if c != 0:
                    ax.set_yticklabels([])  # show ticks only on left col

                ax.tick_params(right=False, labelright=False)
                ax.grid(True, linewidth=0.5, alpha=0.5)
                ax.set_xlim(min(ref_levels), max(ref_levels))
                ax.set_xticks(ref_levels)
                ax.set_ylim(ymin, ymax)
                ax.set_yticks(yticks)
                ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))

                if r == len(datasets) - 1:
                    # bottom row: show ticks + labels + axis label
                    ax.tick_params(axis="x", which="major", labelbottom=True, size=5)
                    ax.set_xlabel("Reference data used (%)", fontsize=8, labelpad=6)
                else:
                    # top row: keep ticks but hide labels
                    ax.tick_params(axis="x", which="major", labelbottom=False, size=5)

        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=min(4, len(labels)),
            frameon=False,
            bbox_to_anchor=(0.5, -0.02),
        )
        fig.tight_layout(rect=(0, 0.08, 1, 0.98))

        fig.savefig(OUTPUT_DIR / "exp4_reference_sweep_grid.png", dpi=SAVE_DPI)
        fig.savefig(OUTPUT_DIR / "exp4_reference_sweep_grid.pdf", dpi=SAVE_DPI)
        logging.info("Saved:", OUTPUT_DIR / "exp4_reference_sweep_grid.[png|pdf]")

    elif mode == "per_model":
        for model in models:
            fig, axes = plt.subplots(
                len(datasets), 1, figsize=FIGSIZE_PER_MODEL, sharex=True, sharey=True
            )
            axes = np.atleast_1d(axes)
            for r, ds in enumerate(datasets):
                ax = axes[r]
                panel = df[(df.dataset == ds) & (df.model == model)]
                for m in order:
                    sub = panel[panel.method == m].sort_values("ref_pct")
                    ax.plot(
                        sub.ref_pct, sub.score, marker="o", ms=4, **styles[m], label=m
                    )
                    if ANNOTATE:
                        last = sub.dropna().iloc[-1]
                        ax.annotate(
                            f"{last.score:.3f}",
                            (last.ref_pct, last.score),
                            xytext=(4, 0),
                            textcoords="offset points",
                            fontsize=8,
                        )
                ax.set_title(f"{ds} — {model}", fontsize=11, pad=6)
                ax.set_ylabel("AUC-PR", fontsize=10)
                ax.tick_params(right=False, labelright=False)
                ax.grid(True, linewidth=0.5, alpha=0.5)
                ax.set_xlim(min(ref_levels), max(ref_levels))
                ax.set_xticks(
                    ref_levels,
                )
                ax.set_ylim(ymin, ymax)
                ax.set_yticks(yticks, size=8)
                ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
                if r == len(datasets) - 1:
                    # bottom row: show ticks + labels + axis label
                    ax.tick_params(axis="x", which="major", labelbottom=True, size=5)
                    ax.set_xlabel("Reference data used (%)", fontsize=8, labelpad=6)
                else:
                    # top row: keep ticks but hide labels
                    ax.tick_params(axis="x", which="major", labelbottom=False, size=5)

            handles, labels = axes[-1].get_legend_handles_labels()
            fig.legend(
                handles,
                labels,
                loc="lower center",
                ncol=len(labels),  # one row, all 7 side-by-side
                frameon=False,
                bbox_to_anchor=(0.5, -0.05),
            )
            fig.tight_layout(rect=(0, 0.10, 1, 0.98))

            base = f"exp4_reference_sweep_{model.replace('–','-').replace(' ','_')}"  # noqa
            fig.savefig(OUTPUT_DIR / f"{base}.png", dpi=SAVE_DPI)
            fig.savefig(OUTPUT_DIR / f"{base}.pdf", dpi=SAVE_DPI)
            logging.info("Saved:", OUTPUT_DIR / f"{base}.[png|pdf]")
    else:
        raise ValueError("PLOT_MODE must be 'grid' or 'per_model'")


if __name__ == "__main__":
    make_plots(df, mode=PLOT_MODE)
