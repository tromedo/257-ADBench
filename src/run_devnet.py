"""DevNet (semi-supervised) on the ADBench datasets.

DevNet trains a small neural network with a deviation loss: unlabeled samples
are pulled toward the score of a standard normal reference, and labeled
anomalies are pushed at least 5 standard deviations above it. The network and
training are in src/devnet.py (PyTorch port of the ADBench implementation).

Run from the repo root:
    python src/run_devnet.py
"""

import time

from data import hide_labels, load_dataset, scale_features, split_data
from devnet import score_devnet, train_devnet
from results import evaluate


def run_once(dataset, fraction, seed):
    """Train and evaluate DevNet for one dataset, label fraction and seed."""
    X, y = load_dataset(dataset)
    X_train, X_test, y_train, y_test = split_data(X, y, seed=seed)
    y_semi, n_labeled = hide_labels(y_train, fraction, seed=seed)
    X_train, X_test = scale_features(X_train, X_test)

    start = time.time()
    model, _ = train_devnet(X_train, y_semi, seed=seed)
    scores = score_devnet(model, X_test)  # higher = more anomalous
    runtime = time.time() - start

    auc_roc, auc_pr = evaluate(y_test, scores)
    return {
        "dataset": dataset,
        "algorithm": "DevNet",
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
