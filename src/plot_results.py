"""Combined plots: performance vs label fraction, one panel per dataset.

Uses the shared loader, so any algorithm with a results CSV in results/ shows
up automatically. Semi-supervised and supervised methods are solid lines with
markers; unsupervised methods do not use labels and are drawn as flat dashed
lines across all fractions.

Run from the repo root:
    python src/plot_results.py
"""

import math
import warnings

import matplotlib

matplotlib.use("Agg")  # write image files, no window needed
import matplotlib.pyplot as plt

from data import DATASET_NAMES, load_dataset
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
    # Larger datasets, where 1% labeled means 9 to 46 anomalies
    "satellite": "Satellite",
    "landsat": "Landsat",
    "magic_gamma": "Magic Gamma",
}


def dataset_title(dataset):
    """Display name; datasets not listed above get a title made from their file name."""
    return DATASET_TITLES.get(dataset, dataset.replace("_", " ").title())


def dataset_order(datasets):
    """Listed datasets first in their fixed order, any new ones after, alphabetically."""
    present = set(datasets)
    return [d for d in DATASET_TITLES if d in present] + sorted(present - set(DATASET_TITLES))
METRIC_LABELS = {"auc_roc": "AUC-ROC", "auc_pr": "AUC-PR"}

# Fixed color per algorithm, so the same method has the same color in every figure.
# Colors are slots 1 to 8 of a colorblind-checked categorical palette; new
# algorithms take the next free slot in alphabetical order.
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
           "#e87ba4", "#008300", "#6250d6", "#e34948"]
KNOWN_ORDER = ["XGBOD", "DevNet", "XGBoost", "RandomForest", "ECOD", "IForest"]
MARKERS = {"semi-supervised": "o", "supervised": "s"}
TEXT, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BASELINE = "#8a8984"  # neutral gray for the random-guess reference line


def anomaly_ratio(dataset):
    """Share of anomalies in a dataset: the AUC-PR score of a random ranking.

    Returns None if the dataset file has not been downloaded (run download_data.py).
    """
    try:
        _, y = load_dataset(dataset)
    except FileNotFoundError:
        warnings.warn(f"data/{dataset}.npz not found; skipping its random-guess line")
        return None
    return y.mean()


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

    datasets = dataset_order(summary["dataset"])
    ncols = 3
    nrows = math.ceil(len(datasets) / ncols)
    height = 3.4 * nrows + 0.7  # fixed panel height plus room for title and legend
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, height), sharex=True, squeeze=False)
    for ax in axes.flat[len(datasets):]:
        ax.set_visible(False)  # unused panels when datasets do not fill the last row
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
        ratio = anomaly_ratio(dataset) if metric == "auc_pr" else None
        if ratio is not None:
            ax.axhline(ratio, color=BASELINE, linewidth=1.2, linestyle=(0, (1, 2)), zorder=1)
        ax.set_title(dataset_title(dataset), color=TEXT, fontsize=12, loc="left")
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
    for i, ax in enumerate(axes.flat[:len(datasets)]):
        if i + ncols >= len(datasets):  # last visible panel in each column
            ax.set_xlabel("Labeled anomalies in training", color=MUTED)
            ax.tick_params(labelbottom=True)

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
    if metric == "auc_pr":
        h, = plt.plot([], [], color=BASELINE, linewidth=1.2, linestyle=(0, (1, 2)))
        handles.append((h, "Random guess (anomaly ratio)"))
    # One legend row for up to 6 entries, otherwise wrap onto two rows
    ncol = len(handles) if len(handles) <= 6 else math.ceil(len(handles) / 2)
    header = 0.75 if len(handles) <= 6 else 1.0  # inches reserved for title and legend
    top = 1 - header / height
    fig.legend([h for h, _ in handles], [l for _, l in handles], loc="upper center",
               ncol=ncol, frameon=False, fontsize=9, bbox_to_anchor=(0.5, 1 - 0.4 / height))
    fig.suptitle(f"{METRIC_LABELS[metric]} vs fraction of labeled anomalies",
                 x=0.01, y=1 - 0.12 / height, ha="left", fontsize=14, color=TEXT)
    fig.tight_layout(rect=(0, 0, 1, top))
    return fig

def style_axis(ax):
    """Recessive grid and axes, shared by the summary figures."""
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.grid(axis="x", visible=False)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)


def plot_median(df, metric="auc_roc"):
    """One panel: median across all datasets of the mean over seeds."""
    summary = summarize(df, metric)
    n_datasets = summary["dataset"].nunique()
    medians = (summary.groupby(["algorithm", "category", "label_fraction"])["mean"]
                      .median().reset_index())
    fractions = sorted(f for f in medians["label_fraction"].unique() if f > 0)
    x_pos = {f: i for i, f in enumerate(fractions)}
    colors, order = algorithm_colors(set(medians["algorithm"]))

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    handles = []
    for algorithm in order:
        rows = medians[medians["algorithm"] == algorithm].sort_values("label_fraction")
        category = rows["category"].iloc[0]
        if category == "unsupervised":
            h = ax.axhline(rows["mean"].iloc[0], color=colors[algorithm], linewidth=2,
                           linestyle=(0, (5, 3)), zorder=2)
        else:
            h, = ax.plot([x_pos[f] for f in rows["label_fraction"]], rows["mean"],
                         color=colors[algorithm], linewidth=2,
                         marker=MARKERS.get(category, "o"), markersize=7,
                         markeredgecolor="white", markeredgewidth=1.2, zorder=3)
        handles.append((h, f"{algorithm} ({category})"))

    ax.set_xticks(range(len(fractions)))
    ax.set_xticklabels([f"{f * 100:g}%" for f in fractions])
    ax.set_ylim(0.5, 1.0)
    ax.set_xlabel("Labeled anomalies in training", color=MUTED)
    ax.set_ylabel(f"Median {METRIC_LABELS[metric]} across datasets", color=MUTED)
    style_axis(ax)
    ax.legend([h for h, _ in handles], [label for _, label in handles], loc="lower right",
              frameon=False, fontsize=8.5)
    ax.set_title(f"Median {METRIC_LABELS[metric]} across all {n_datasets} datasets "
                 "(mean of 3 seeds per dataset)", loc="left", fontsize=12, color=TEXT)
    fig.tight_layout()
    return fig


def plot_larger_1pct(df, metric="auc_roc", fraction=0.01):
    """Grouped bars: every method at 1% labels on the larger datasets.

    Unsupervised methods use no labels and are shown with a hatched fill.
    Error bars are the standard deviation over the 3 seeds.
    """
    summary = summarize(df, metric)
    datasets = [d for d in dataset_order(summary["dataset"]) if d not in DATASET_NAMES]
    summary = summary[summary["dataset"].isin(datasets)]
    summary = summary[(summary["label_fraction"] == fraction)
                      | (summary["category"] == "unsupervised")]
    colors, order = algorithm_colors(set(summary["algorithm"]))

    fig, ax = plt.subplots(figsize=(8, 4.6))
    width = 0.8 / len(order)
    for i, algorithm in enumerate(order):
        rows = summary[summary["algorithm"] == algorithm].set_index("dataset").loc[datasets]
        category = rows["category"].iloc[0]
        unsupervised = category == "unsupervised"
        xs = [d + (i - (len(order) - 1) / 2) * width for d in range(len(datasets))]
        ax.bar(xs, rows["mean"], width=width, color="white" if unsupervised else colors[algorithm],
               edgecolor=colors[algorithm] if unsupervised else "white",
               hatch="////" if unsupervised else None,
               linewidth=1.5 if unsupervised else 1, label=f"{algorithm} ({category})",
               zorder=3)
        ax.errorbar(xs, rows["mean"], yerr=rows["std"], fmt="none", ecolor=MUTED,
                    elinewidth=1, capsize=2, zorder=4)

    ax.set_xticks(range(len(datasets)))
    ax.set_xticklabels([dataset_title(d) for d in datasets])
    ax.set_ylim(0, 1.0)
    ax.set_ylabel(f"{METRIC_LABELS[metric]} (mean of 3 seeds)", color=MUTED)
    style_axis(ax)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3, frameon=False,
              fontsize=8.5)
    ax.set_title(f"{METRIC_LABELS[metric]} with {fraction * 100:g}% labeled anomalies "
                 "on the larger datasets (unsupervised: no labels)",
                 loc="left", fontsize=11, color=TEXT)
    fig.tight_layout()
    return fig

if __name__ == "__main__":
    FIGURES_DIR.mkdir(exist_ok=True)
    results = load_all_results()
    for metric in ("auc_roc", "auc_pr"):
        fig = plot_metric(results, metric)
        out = FIGURES_DIR / f"{metric}_vs_label_fraction.png"
        fig.savefig(out, dpi=200)
        plt.close(fig)
        print(f"Saved {out}")

    for name, fig in (("median_auc_roc_all_datasets.png", plot_median(results)),
                      ("larger_datasets_1pct.png", plot_larger_1pct(results))):
        out = FIGURES_DIR / name
        fig.savefig(out, dpi=200)
        plt.close(fig)
        print(f"Saved {out}")