"""Supervised baselines: Random Forest and XGBoost.

For every dataset, seed and label fraction, train each model with only that
fraction of training anomalies labeled (the rest are treated as normal),
then score the test set and record AUC-ROC and AUC-PR.

Run from the repo root:
    python src/run_supervised.py                                   # the 6 main datasets
    python src/run_supervised.py satellite landsat magic_gamma     # extra datasets only
"""

import sys
import time

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from data import (DATASET_NAMES, LABEL_FRACTIONS, hide_labels, load_dataset,
                  scale_features, split_data)
from results import RESULTS_DIR, evaluate, save_result

SEEDS = [1, 2, 3]
CATEGORY = "supervised"

RESULTS_FILES = {
    "RandomForest": "random_forest_results.csv",
    "XGBoost": "xgboost_results.csv",
}
EXTRA_RESULTS_FILES = {
    "RandomForest": "larger_dataset/random_forest_extra_results.csv",
    "XGBoost": "larger_dataset/xgboost_extra_results.csv",
}

# Each model is built fresh for every run; default settings, as in ADBench.
MODELS = {
    "RandomForest": lambda seed: RandomForestClassifier(
        n_estimators=100, random_state=seed, n_jobs=-1),
    "XGBoost": lambda seed: XGBClassifier(
        n_estimators=100, random_state=seed, n_jobs=-1, eval_metric="logloss",
        base_score=0.5),   # older XGBoost default; matches PyOD's XGBOD
}


def run_one(dataset, algorithm, seed, fraction):
    """Train and evaluate one model on one dataset / seed / label fraction."""
    X, y = load_dataset(dataset)
    X_train, X_test, y_train, y_test = split_data(X, y, seed=seed)
    y_semi, n_labeled = hide_labels(y_train, fraction=fraction, seed=seed)
    X_train, X_test = scale_features(X_train, X_test)

    start = time.time()
    model = MODELS[algorithm](seed)
    model.fit(X_train, y_semi)                    # train
    scores = model.predict_proba(X_test)[:, 1]    # probability of "anomaly"
    runtime = time.time() - start

    auc_roc, auc_pr = evaluate(y_test, scores)
    return {
        "dataset": dataset, "algorithm": algorithm, "category": CATEGORY,
        "label_fraction": fraction, "n_labeled": n_labeled, "seed": seed,
        "auc_roc": auc_roc, "auc_pr": auc_pr, "runtime_sec": runtime,
    }


def main():
    # No arguments: the 6 main datasets. With arguments: only those datasets,
    # saved to separate files so the main results are not touched.
    datasets = sys.argv[1:] or DATASET_NAMES
    files = EXTRA_RESULTS_FILES if sys.argv[1:] else RESULTS_FILES

    for filename in files.values():               # start fresh, no duplicate rows
        out_path = RESULTS_DIR / filename
        if out_path.exists():
            out_path.unlink()

    total = len(datasets) * len(MODELS) * len(SEEDS) * len(LABEL_FRACTIONS)
    done = 0
    for dataset in datasets:
        for algorithm in MODELS:
            for seed in SEEDS:
                for fraction in LABEL_FRACTIONS:
                    row = run_one(dataset, algorithm, seed, fraction)
                    save_result(row, files[algorithm])
                    done += 1
                    print(f"[{done:3d}/{total}] {dataset:16s} {algorithm:12s} seed={seed} "
                          f"frac={fraction:4.2f} labeled={row['n_labeled']:3d}  "
                          f"AUC-ROC={row['auc_roc']:.3f}  AUC-PR={row['auc_pr']:.3f}")

    print(f"\nSaved {done} rows to {', '.join(files.values())} in {RESULTS_DIR}")


if __name__ == "__main__":
    main()