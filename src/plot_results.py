"""Combined plots: performance vs label fraction, one panel per dataset.

Uses the shared loader, so any algorithm with a results CSV in results/ shows
up automatically. Semi-supervised and supervised methods are solid lines with
markers; unsupervised methods do not use labels and are drawn as flat dashed
lines across all fractions.

Run from the repo root:
    python src/plot_results.py
"""

import matplotlib

matplotlib.use("Agg")  # write image files, no window needed
import matplotlib.pyplot as plt

from load_results import load_all_results, summarize
from results import RESULTS_DIR

FIGURES_DIR = RESULTS_DIR / "figures"

DATASET_TITLES = {
    "annthyroid": "Annthyroid",
    "breastw": "Breastw",
    "cardiotocography": "Cardiotocography",
    "hepatitis": "Hepatitis",
    "lymphography": "Lymphography",
    "vertebral": "Vertebral",
}
METRIC_LABELS = {"auc_roc": "AUC-ROC", "auc_pr": "AUC-PR"}

# Fixed color per algorithm, so the same method has the same color in every figure.
# Colors are slots 1 to 8 of a colorblind-checked categorical palette; new
# algorithms take the next free slot in alphabetical order.
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
           "#e87ba4", "#008300", "#6250d6", "#e34948"]
KNOWN_ORDER = ["XGBOD", "DevNet", "XGBoost", "RandomForest", "ECOD", "IForest"]
MARKERS = {"semi-supervised": "o", "supervised": "s"}
TEXT, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def algorithm_colors(algorithms):
    """Known algorithms keep their slot; new ones get the next free slots."""
    order = [a for a in KNOWN_ORDER if a in algorithms]
    order += sorted(a for a in algorithms if a not in KNOWN_ORDER)
    return {a: PALETTE[i % len(PALETTE)] for i, a in enumerate(order)}, order


def plot_metric(df, metric):
    """Draw one figure for one metric and return it."""
    summary = summarize(df, metric)
    fractions = sorted(f for f in summary["label_fraction"].unique() if f > 0)
    x_pos = {f: i for i, f in enumerate(fractions)}
    colors, order = algorithm_colors(set(summary["algorithm"]))
    category = summary.drop_duplicates("algorithm").set_index("algorithm")["category"]

    datasets = [d for d in DATASET_TITLES if d in set(summary["dataset"])]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5), sharex=True)
    for ax, dataset in zip(axes.flat, datasets):
        part = summary[summary["dataset"] == dataset]
        for algorithm in order:
            rows = part[part["algorithm"] == algorithm]
            if rows.empty:
                continue
            if category[algorithm] == "unsupervised":
                ax.axhline(rows["mean"].iloc[0], color=colors[algorithm], linewidth=2,
                           linestyle=(0, (5, 3)), zorder=2)
            else:
                rows = rows.sort_values("label_fraction")
                ax.plot([x_pos[f] for f in rows["label_fraction"]], rows["mean"],
                        color=colors[algorithm], linewidth=2,
                        marker=MARKERS.get(category[algorithm], "o"), markersize=6,
                        markeredgecolor="white", markeredgewidth=1.2, zorder=3)
        ax.set_title(DATASET_TITLES[dataset], color=TEXT, fontsize=12, loc="left")
        ax.set_ylim(0, 1.02)
        ax.set_xticks(range(len(fractions)))
        ax.set_xticklabels([f"{f * 100:g}%" for f in fractions])
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.tick_params(colors=MUTED, labelsize=9)

    for ax in axes[:, 0]:
        ax.set_ylabel(f"{METRIC_LABELS[metric]} (mean of 3 seeds)", color=MUTED)
    for ax in axes[1, :]:
        ax.set_xlabel("Labeled anomalies in training", color=MUTED)

    handles = []
    for algorithm in order:
        if category[algorithm] == "unsupervised":
            h, = plt.plot([], [], color=colors[algorithm], linewidth=2, linestyle=(0, (5, 3)))
            label = f"{algorithm} (unsupervised)"
        else:
            h, = plt.plot([], [], color=colors[algorithm], linewidth=2,
                          marker=MARKERS.get(category[algorithm], "o"), markersize=6,
                          markeredgecolor="white", markeredgewidth=1.2)
            label = f"{algorithm} ({category[algorithm]})"
        handles.append((h, label))
    fig.legend([h for h, _ in handles], [l for _, l in handles], loc="upper center",
               ncol=len(handles), frameon=False, fontsize=9, bbox_to_anchor=(0.5, 0.955))
    fig.suptitle(f"{METRIC_LABELS[metric]} vs fraction of labeled anomalies",
                 x=0.01, ha="left", fontsize=14, color=TEXT)
    fig.tight_layout(rect=(0, 0, 1, 0.935))
    return fig


if __name__ == "__main__":
    FIGURES_DIR.mkdir(exist_ok=True)
    results = load_all_results()
    fig = plot_metric(results, "auc_roc")
    out = FIGURES_DIR / "auc_roc_vs_label_fraction.png"
    fig.savefig(out, dpi=200)
    print(f"Saved {out}")
