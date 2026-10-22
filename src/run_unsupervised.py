"""ECOD and IForest (unsupervised) on the ADBench datasets.

Both detectors ignore labels, so they are run once per dataset and seed at
label_fraction 0.0. That score is the baseline the supervised and
semi-supervised methods are compared against at every label fraction.
We use PyOD's implementations with their default settings.

Run from the repo root:
    python src/run_unsupervised.py
"""

import time

from pyod.models.ecod import ECOD
from pyod.models.iforest import IForest

from data import DATASET_NAMES, load_dataset, scale_features, split_data
from results import RESULTS_DIR, evaluate, save_result

SEEDS = [1, 2, 3]
RESULTS_FILE = "unsupervised_results.csv"

# Name -> function that builds a fresh model for a given seed
MODELS = {
    "ECOD": lambda seed: ECOD(),  # deterministic, has no random_state
    "IForest": lambda seed: IForest(random_state=seed),
}


def run_once(dataset, algorithm, seed):
    """Train and evaluate one unsupervised detector for one dataset and seed."""
    X, y = load_dataset(dataset)
    X_train, X_test, y_train, y_test = split_data(X, y, seed=seed)
    X_train, X_test = scale_features(X_train, X_test)

    start = time.time()
    model = MODELS[algorithm](seed)
    model.fit(X_train)  # no labels
    scores = model.decision_function(X_test)  # higher = more anomalous
    runtime = time.time() - start

    auc_roc, auc_pr = evaluate(y_test, scores)
    return {
        "dataset": dataset,
        "algorithm": algorithm,
        "category": "unsupervised",
        "label_fraction": 0.0,
        "n_labeled": 0,
        "seed": seed,
        "auc_roc": auc_roc,
        "auc_pr": auc_pr,
        "runtime_sec": runtime,
    }


def run_all():
    """Run every dataset x algorithm x seed and save each result row."""
    out = RESULTS_DIR / RESULTS_FILE
    if out.exists():
        out.unlink()  # start fresh so re-running does not duplicate rows

    total = len(DATASET_NAMES) * len(MODELS) * len(SEEDS)
    done = 0
    for dataset in DATASET_NAMES:
        for algorithm in MODELS:
            for seed in SEEDS:
                row = run_once(dataset, algorithm, seed)
                save_result(row, RESULTS_FILE)
                done += 1
                print(f"[{done:2d}/{total}] {dataset:16s} {algorithm:8s} seed={seed}  "
                      f"AUC-ROC={row['auc_roc']:.3f}  AUC-PR={row['auc_pr']:.3f}  "
                      f"{row['runtime_sec']:.1f}s")
    print(f"Saved {total} rows to {out}")


if __name__ == "__main__":
    run_all()
