"""Download the six ADBench datasets used in this project into data/.

Run from the repo root:
    python src/download_data.py
"""

import urllib.request
from pathlib import Path

import numpy as np

BASE_URL = "https://raw.githubusercontent.com/Minqi824/ADBench/main/adbench/datasets/Classical/"

# Short name used in our code -> file name in the ADBench repo
DATASETS = {
    "annthyroid": "2_annthyroid.npz",
    "cardiotocography": "7_Cardiotocography.npz",
    "breastw": "4_breastw.npz",
    "lymphography": "21_Lymphography.npz",
    "vertebral": "39_vertebral.npz",
    "hepatitis": "15_Hepatitis.npz",
    # Larger datasets (more labeled anomalies at 1%)
    "satellite": "30_satellite.npz",
    "landsat": "19_landsat.npz",
    "magic_gamma": "22_magic.gamma.npz",
}

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def download_all():
    DATA_DIR.mkdir(exist_ok=True)
    for name, filename in DATASETS.items():
        path = DATA_DIR / f"{name}.npz"
        if not path.exists():
            urllib.request.urlretrieve(BASE_URL + filename, path)
        data = np.load(path)
        X, y = data["X"], data["y"]
        print(f"{name:18s} samples={X.shape[0]:5d} features={X.shape[1]:3d} anomalies={int(y.sum()):4d}")


if __name__ == "__main__":
    download_all()
