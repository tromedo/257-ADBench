"""XGBOD (semi-supervised) on the ADBench datasets.

XGBOD runs a set of unsupervised detectors (KNN, LOF, HBOS, OCSVM, IForest),
adds their outlier scores to the original features, and trains XGBoost on the
few labeled anomalies. We use PyOD's implementation with its default settings.

Run from the repo root:
    python src/run_xgbod.py
"""

import time

from pyod.models.xgbod import XGBOD

from data import hide_labels, load_dataset, scale_features, split_data
from results import evaluate


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


if __name__ == "__main__":
    # Single test run before looping over everything
    print(run_once("annthyroid", fraction=0.01, seed=1))
