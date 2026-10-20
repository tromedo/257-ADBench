"""XGBOD (semi-supervised) on the ADBench datasets.

XGBOD runs a set of unsupervised detectors (KNN, LOF, HBOS, OCSVM, IForest),
adds their outlier scores to the original features, and trains XGBoost on the
few labeled anomalies. We use PyOD's implementation with its default settings.

Run from the repo root:
    python src/run_xgbod.py
"""

import time
import warnings

from pyod.models.xgbod import XGBOD

from data import DATASET_NAMES, LABEL_FRACTIONS, hide_labels, load_dataset, scale_features, split_data
from results import RESULTS_DIR, evaluate, save_result

SEEDS = [1, 2, 3]
RESULTS_FILE = "xgbod_results.csv"

# Two known, harmless warnings from inside PyOD and XGBoost (see README)
warnings.filterwarnings("ignore", message="y should not be presented")
warnings.filterwarnings("ignore", message=r"[\s\S]*silent")


def run_once(dataset, fraction, seed):
    """Train and evaluate XGBOD for one dataset, label fraction and seed."""
    X, y = load_dataset(dataset)
    X_train, X_test, y_train, y_test = split_data(X, y, seed=seed)
    y_semi, n_labeled = hide_labels(y_train, fraction, seed=seed)
    X_train, X_test = scale_features(X_train, X_test)

    start = time.time()
    model = XGBOD(random_state=seed)
    model.fit(X_train, y_semi)
    scores = model.decision_function(X_test)  # probability of anomaly
    runtime = time.time() - start

    auc_roc, auc_pr = evaluate(y_test, scores)
    return {
        "dataset": dataset,
        "algorithm": "XGBOD",
        "category": "semi-supervised",
        "label_fraction": fraction,
        "n_labeled": n_labeled,
        "seed": seed,
        "auc_roc": auc_roc,
        "auc_pr": auc_pr,
        "runtime_sec": runtime,
    }


def run_all():
    """Run every dataset x label fraction x seed and save each result row."""
    out = RESULTS_DIR / RESULTS_FILE
    if out.exists():
        out.unlink()  # start fresh so re-running does not duplicate rows

    total = len(DATASET_NAMES) * len(LABEL_FRACTIONS) * len(SEEDS)
    done = 0
    for dataset in DATASET_NAMES:
        for fraction in LABEL_FRACTIONS:
            for seed in SEEDS:
                row = run_once(dataset, fraction, seed)
                save_result(row, RESULTS_FILE)
                done += 1
                print(f"[{done:3d}/{total}] {dataset:16s} {int(fraction * 100):3d}% "
                      f"(n_labeled={row['n_labeled']:3d}) seed={seed}  "
                      f"AUC-ROC={row['auc_roc']:.3f}  AUC-PR={row['auc_pr']:.3f}  "
                      f"{row['runtime_sec']:.1f}s")
    print(f"Saved {total} rows to {out}")


if __name__ == "__main__":
    run_all()
