"""Data quality check promised in the proposal: missing values and duplicates.

We keep the datasets exactly as ADBench provides them (no rows removed), so this
script only reports what is there, for the report.

For datasets under 1,000 rows it also shows how many test rows would also be in the
training set under ADBench's resampling (rows drawn with replacement up to 1,000
before the split), which is why our main results do not resample.

Run from the repo root:
    python src/check_data_quality.py
"""

import numpy as np

from compare_protocols import SMALL_THRESHOLD, split_resampled
from data import load_dataset, split_data
from download_data import DATASETS


def duplicate_stats(X, y):
    """Return (extra duplicate rows, duplicate groups whose labels disagree)."""
    _, inverse, counts = np.unique(X, axis=0, return_inverse=True, return_counts=True)
    inverse = inverse.ravel()
    extra_rows = int((counts[counts > 1] - 1).sum())
    conflicting = sum(1 for g in np.where(counts > 1)[0] if len(set(y[inverse == g])) > 1)
    return extra_rows, conflicting


def test_rows_seen_in_train(X_train, X_test):
    """Number of test rows that also appear, identically, in the training set."""
    train_rows = {row.tobytes() for row in X_train}
    return sum(row.tobytes() in train_rows for row in X_test)


if __name__ == "__main__":
    print(f"{'dataset':16s} {'samples':>7s} {'missing':>7s} {'inf':>4s} "
          f"{'dup rows':>8s} {'label conflicts':>15s} {'test rows also in train (seed 1)':>33s} "
          f"{'if resampled (ADBench)':>23s}")
    for name in DATASETS:  # every dataset the download script knows about
        try:
            X, y = load_dataset(name)
        except FileNotFoundError:
            print(f"{name:16s} not downloaded (run python src/download_data.py)")
            continue
        extra, conflicting = duplicate_stats(X, y)
        X_train, X_test, _, _ = split_data(X, y, seed=1)
        overlap = test_rows_seen_in_train(X_train, X_test)
        if len(X) < SMALL_THRESHOLD:  # ADBench would resample this dataset before the split
            X_train_r, X_test_r, _, _ = split_resampled(X, y, seed=1)
            overlap_r = test_rows_seen_in_train(X_train_r, X_test_r)
            resampled = f"{overlap_r:4d} of {len(X_test_r):4d} ({overlap_r / len(X_test_r):4.0%})"
        else:
            resampled = "-"
        print(f"{name:16s} {len(X):7d} {int(np.isnan(X).sum()):7d} {int(np.isinf(X).sum()):4d} "
              f"{extra:8d} {conflicting:15d} {overlap:20d} of {len(X_test):4d} "
              f"({overlap / len(X_test):4.0%}) {resampled:>23s}")
