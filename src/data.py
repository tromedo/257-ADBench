"""Shared data module: every algorithm in the project gets its data from here,
so all three of us use identical splits for the same seed.
"""

from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

DATASET_NAMES = [
    "annthyroid",
    "cardiotocography",
    "breastw",
    "lymphography",
    "vertebral",
    "hepatitis",
]


def load_dataset(name):
    """Return X (features, one row per sample) and y (0 = normal, 1 = anomaly)."""
    data = np.load(DATA_DIR / f"{name}.npz")
    return data["X"].astype(float), data["y"].astype(int)


def split_data(X, y, seed, test_size=0.3):
    """Stratified 70/30 split, as in ADBench.

    Stratified means the anomaly ratio is kept the same in train and test,
    which matters when there are only a handful of anomalies.
    """
    return train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)


if __name__ == "__main__":
    # Quick check: show the split sizes for every dataset with seed 1
    for name in DATASET_NAMES:
        X, y = load_dataset(name)
        X_train, X_test, y_train, y_test = split_data(X, y, seed=1)
        print(f"{name:18s} train={len(y_train):5d} (anomalies {y_train.sum():3d})   "
              f"test={len(y_test):5d} (anomalies {y_test.sum():3d})")
