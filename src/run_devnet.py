"""DevNet (semi-supervised) on the ADBench datasets.

DevNet trains a small neural network with a deviation loss: unlabeled samples
are pulled toward the score of a standard normal reference, and labeled
anomalies are pushed at least 5 standard deviations above it. The network and
training are in src/devnet.py (PyTorch port of the ADBench implementation).

Run from the repo root:
    python src/run_devnet.py
"""
import sys
import time

from data import DATASET_NAMES, LABEL_FRACTIONS, hide_labels, load_dataset, scale_features, split_data
from devnet import score_devnet, train_devnet
from results import RESULTS_DIR, evaluate, save_result

SEEDS = [1, 2, 3]
RESULTS_FILE = "devnet_results.csv"
EXTRA_RESULTS_FILE = "larger_dataset/devnet_extra_results.csv"

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


def run_all():
    """Run every dataset x label fraction x seed and save each result row."""
    datasets = sys.argv[1:] or DATASET_NAMES
    results_file = EXTRA_RESULTS_FILE if sys.argv[1:] else RESULTS_FILE
    out = RESULTS_DIR / results_file
    if out.exists():
        out.unlink()  # start fresh so re-running does not duplicate rows

    total = len(datasets) * len(LABEL_FRACTIONS) * len(SEEDS)
    done = 0
    for dataset in datasets:
        for fraction in LABEL_FRACTIONS:
            for seed in SEEDS:
                row = run_once(dataset, fraction, seed)
                save_result(row, results_file)
                done += 1
                print(f"[{done:3d}/{total}] {dataset:16s} {int(fraction * 100):3d}% "
                      f"(n_labeled={row['n_labeled']:3d}) seed={seed}  "
                      f"AUC-ROC={row['auc_roc']:.3f}  AUC-PR={row['auc_pr']:.3f}  "
                      f"{row['runtime_sec']:.1f}s")
    print(f"Saved {total} rows to {out}")


if __name__ == "__main__":
    run_all()
