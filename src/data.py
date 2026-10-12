"""Shared data module: every algorithm in the project gets its data from here,
so all three of us use identical splits for the same seed.
"""

from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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


LABEL_FRACTIONS = [0.01, 0.05, 0.10, 0.25, 0.50, 1.00]


def hide_labels(y_train, fraction, seed):
    """Keep only a fraction of the training anomalies labeled.

    The anomalies that are not chosen are relabeled 0 and stay in the training
    data, mixed in with the normal samples (ADBench's semi-supervised setting).
    At least one anomaly is always kept labeled, so the semi-supervised and
    supervised methods have something to learn from on the small datasets.

    Returns the new labels and the number of anomalies actually labeled.
    """
    anomaly_idx = np.where(y_train == 1)[0]
    n_labeled = max(1, int(fraction * len(anomaly_idx)))

    rng = np.random.default_rng(seed)
    labeled_idx = rng.choice(anomaly_idx, size=n_labeled, replace=False)

    y_semi = np.zeros_like(y_train)
    y_semi[labeled_idx] = 1
    return y_semi, n_labeled


def scale_features(X_train, X_test):
    """Standardize each feature to mean 0 and standard deviation 1.

    The mean and standard deviation come from the training set only, and the
    same transformation is then applied to the test set. Fitting on the test
    set would leak information about it into the model.
    """
    scaler = StandardScaler().fit(X_train)
    return scaler.transform(X_train), scaler.transform(X_test)


if __name__ == "__main__":
    # Quick check: split sizes, then how many anomalies get labeled per fraction (seed 1)
    for name in DATASET_NAMES:
        X, y = load_dataset(name)
        X_train, X_test, y_train, y_test = split_data(X, y, seed=1)
        print(f"{name:18s} train={len(y_train):5d} (anomalies {y_train.sum():3d})   "
              f"test={len(y_test):5d} (anomalies {y_test.sum():3d})")
        counts = [hide_labels(y_train, f, seed=1)[1] for f in LABEL_FRACTIONS]
        print(" " * 19 + "labeled at " + ", ".join(
            f"{int(f * 100)}%: {c}" for f, c in zip(LABEL_FRACTIONS, counts)))
        X_train_s, X_test_s = scale_features(X_train, X_test)
        print(" " * 19 + f"after scaling: train mean {X_train_s.mean():.2f}, train std {X_train_s.std():.2f}, "
              f"test mean {X_test_s.mean():.2f}")
