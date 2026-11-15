"""Compare our data protocol with ADBench's, one step at a time.

Our main results (results/) use: labeled count rounded down (min 1), StandardScaler,
datasets as provided. ADBench's data generator differs in three ways. This script
re-runs every method under three variants, adding one ADBench step at a time:

    round_up          labeled count rounded up, as ADBench does
    round_up_minmax   + MinMaxScaler in place of StandardScaler
    adbench_full      + datasets under 1,000 rows resampled with replacement to 1,000,
                        and datasets over 10,000 rows subsampled to 10,000

Results go to results_comparison/, outside results/, so the main plots, tables and app
are not affected. A summary table is written to docs/protocol_comparison.md.

Run from the repo root (about 5 to 10 minutes on 10 cores):
    python src/compare_protocols.py
    python src/compare_protocols.py --jobs 4       # fewer cores
    python src/compare_protocols.py --summary-only # rebuild the table from saved results
"""

import argparse
import math
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import data
from download_data import DATASETS
from load_results import load_all_results
from results import COLUMNS

REPO = Path(__file__).resolve().parent.parent
OUT_DIR = REPO / "results_comparison"
SUMMARY_FILE = REPO / "docs" / "protocol_comparison.md"

SEEDS = [1, 2, 3]
SMALL_THRESHOLD = 1000    # ADBench: resample smaller datasets up to this size
LARGE_THRESHOLD = 10000   # ADBench: subsample larger datasets down to this size

VARIANTS = {
    "round_up": "Labeled count rounded up",
    "round_up_minmax": "Rounded up + min-max scaling",
    "adbench_full": "ADBench's full procedure (rounded up + min-max + resampling)",
}
LABELED_METHODS = ["XGBOD", "DevNet", "XGBoost", "RandomForest"]
UNSUPERVISED_METHODS = ["ECOD", "IForest"]


def hide_labels_round_up(y_train, fraction, seed):
    """Same as data.hide_labels, but the labeled count is rounded up (ADBench's ceil)."""
    anomaly_idx = np.where(y_train == 1)[0]
    n_labeled = max(1, math.ceil(fraction * len(anomaly_idx)))
    rng = np.random.default_rng(seed)
    labeled_idx = rng.choice(anomaly_idx, size=n_labeled, replace=False)
    y_semi = np.zeros_like(y_train)
    y_semi[labeled_idx] = 1
    return y_semi, n_labeled


def scale_minmax(X_train, X_test):
    """Min-max scaling fitted on the training set only, as in ADBench."""
    scaler = MinMaxScaler().fit(X_train)
    return scaler.transform(X_train), scaler.transform(X_test)


def split_resampled(X, y, seed, test_size=0.3):
    """ADBench's resampling, then the usual stratified 70/30 split.

    Small datasets are sampled with replacement up to 1,000 rows BEFORE the split,
    so copies of the same row can land in both train and test. Large datasets are
    subsampled without replacement to 10,000 rows.
    """
    rng = np.random.default_rng(seed)
    if len(y) < SMALL_THRESHOLD:
        idx = rng.choice(len(y), size=SMALL_THRESHOLD, replace=True)
        X, y = X[idx], y[idx]
    if len(y) > LARGE_THRESHOLD:
        idx = rng.choice(len(y), size=LARGE_THRESHOLD, replace=False)
        X, y = X[idx], y[idx]
    return train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)


def run_task(variant, algorithm, dataset, fraction, seed):
    """One run of one method under one variant. Executed in a worker process."""
    warnings.filterwarnings("ignore")
    import torch
    torch.set_num_threads(1)  # one core per run; runs are spread across cores instead
    import run_devnet
    import run_supervised
    import run_unsupervised
    import run_xgbod

    # Workers are reused between tasks, so set all three functions explicitly every time.
    hide = hide_labels_round_up
    scale = data.scale_features if variant == "round_up" else scale_minmax
    split = split_resampled if variant == "adbench_full" else data.split_data
    for module in (run_xgbod, run_devnet, run_supervised, run_unsupervised):
        module.split_data = split
        module.scale_features = scale
        if hasattr(module, "hide_labels"):
            module.hide_labels = hide

    if algorithm == "XGBOD":
        row = run_xgbod.run_once(dataset, fraction, seed)
    elif algorithm == "DevNet":
        row = run_devnet.run_once(dataset, fraction, seed)
    elif algorithm in ("RandomForest", "XGBoost"):
        row = run_supervised.run_one(dataset, algorithm, seed, fraction)
    else:
        row = run_unsupervised.run_once(dataset, algorithm, seed)
    return row


def build_tasks(variant):
    tasks = []
    for dataset in DATASETS:
        for seed in SEEDS:
            tasks += [(variant, a, dataset, 0.0, seed) for a in UNSUPERVISED_METHODS]
            for fraction in data.LABEL_FRACTIONS:
                tasks += [(variant, a, dataset, fraction, seed) for a in LABELED_METHODS]
    # Slowest first (XGBOD on the largest datasets), so the cores stay busy to the end
    sizes = {d: len(data.load_dataset(d)[1]) for d in DATASETS}
    tasks.sort(key=lambda t: (t[1] != "XGBOD", -sizes[t[2]]))
    return tasks


def run_variants(jobs):
    OUT_DIR.mkdir(exist_ok=True)
    start = time.time()
    for variant in VARIANTS:
        tasks = build_tasks(variant)
        print(f"{variant}: {len(tasks)} runs on {jobs} cores", flush=True)
        rows = Parallel(n_jobs=jobs)(delayed(run_task)(*t) for t in tasks)
        df = pd.DataFrame(rows)[COLUMNS].sort_values(
            ["dataset", "algorithm", "label_fraction", "seed"])
        df.to_csv(OUT_DIR / f"{variant}.csv", index=False)
        print(f"  saved {variant}.csv at {(time.time() - start) / 60:.1f} min", flush=True)


def medians(df):
    """Median across datasets of the mean over seeds, in percent (the paper's summary)."""
    per_dataset = df.groupby(["dataset", "algorithm", "label_fraction"])["auc_roc"].mean()
    return (per_dataset.groupby(["algorithm", "label_fraction"]).median() * 100).unstack()


def write_summary():
    setups = {"Ours (main results)": load_all_results().drop(columns=["source_file"])}
    for variant, label in VARIANTS.items():
        setups[label] = pd.read_csv(OUT_DIR / f"{variant}.csv")
    tables = {label: medians(df) for label, df in setups.items()}
    fractions = data.LABEL_FRACTIONS

    lines = ["# Protocol comparison: ours vs ADBench's data procedure", "",
             "Generated by `python src/compare_protocols.py`. Median AUC-ROC (%) across all "
             f"{len(DATASETS)} datasets of the mean over 3 seeds. Each setup adds one ADBench "
             "step to the previous one. Raw results are in `results_comparison/`.", "",
             "Resampling draws rows at random, so `adbench_full` follows ADBench's procedure "
             "but not its exact samples.", ""]
    header = "| Method | Setup | " + " | ".join(f"{f * 100:g}%" for f in fractions) + " |"
    for method in LABELED_METHODS:
        lines += [f"## {method}", "", header, "|---|---|" + "---|" * len(fractions)]
        for label, table in tables.items():
            cells = " | ".join(f"{table.loc[method, f]:.1f}" for f in fractions)
            lines.append(f"| {method} | {label} | {cells} |")
        lines.append("")
    lines += ["## Unsupervised baselines (no labels)", "", "| Setup | ECOD | IForest |", "|---|---|---|"]
    for label, table in tables.items():
        lines.append(f"| {label} | {table.loc['ECOD', 0.0]:.1f} | {table.loc['IForest', 0.0]:.1f} |")
    lines.append("")
    SUMMARY_FILE.write_text("\n".join(lines))
    print(f"Wrote {SUMMARY_FILE}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare our data protocol with ADBench's.")
    parser.add_argument("--jobs", type=int, default=10, help="number of CPU cores to use")
    parser.add_argument("--summary-only", action="store_true",
                        help="rebuild docs/protocol_comparison.md from saved results")
    args = parser.parse_args()
    if not args.summary_only:
        run_variants(args.jobs)
    write_summary()
