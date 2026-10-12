"""Shared results format: every algorithm writes its scores in the same CSV
layout, so all results can be combined and plotted together at the end.
"""

import csv
from pathlib import Path

from sklearn.metrics import average_precision_score, roc_auc_score

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

COLUMNS = [
    "dataset",         # e.g. "annthyroid"
    "algorithm",       # e.g. "XGBOD"
    "category",        # "supervised", "semi-supervised" or "unsupervised"
    "label_fraction",  # e.g. 0.01 (0.0 for unsupervised)
    "n_labeled",       # actual number of labeled training anomalies
    "seed",            # 1, 2 or 3
    "auc_roc",
    "auc_pr",
    "runtime_sec",
]


def evaluate(y_test, scores):
    """Score a model's anomaly scores against the true test labels.

    scores: one number per test sample, higher = more anomalous.
    Returns (AUC-ROC, AUC-PR). AUC-PR is computed as average precision, as in ADBench.
    """
    return roc_auc_score(y_test, scores), average_precision_score(y_test, scores)


def save_result(row, filename):
    """Append one result row to results/<filename>, writing the header if the file is new."""
    path = RESULTS_DIR / filename
    is_new = not path.exists()
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        if is_new:
            writer.writeheader()
        writer.writerow(row)
