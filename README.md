# Label Efficiency in Anomaly Detection

CMPE 257 group project (SJSU, Fall 2026): a replication of Angle I of ADBench (NeurIPS 2022).
We test how much a small fraction of labeled anomalies improves detection compared to fully unsupervised methods.

Team: Manisha Varthamanan, Dipti Ghiya, Sriram Sridhar

## Scope

Datasets (ADBench Classical): Annthyroid, Cardiotocography, Breastw, Lymphography, Vertebral, Hepatitis

| Category | Algorithms | Owner |
|---|---|---|
| Supervised | Random Forest, XGBoost | Manisha |
| Semi-supervised | DevNet, XGBOD | Sriram |
| Unsupervised | ECOD, IForest | Dipti |

Label fractions: 1%, 5%, 10%, 25%, 50%, 100% of training anomalies (minimum 1 labeled anomaly).
0% is the unsupervised baseline.

Split: stratified 70/30 train/test. Seeds: 1, 2, 3.
Metrics: AUC-ROC and AUC-PR.

## Layout

```
data/      downloaded .npz datasets (not committed)
src/       shared data module and algorithm runners
results/   raw results CSVs and plots
```

## Setup (one time)

Prerequisites:

* Python 3.14 (`python3 --version`). The library versions in `requirements.txt` are pinned
  to match the machine the final results are run on, so the same seed gives the same numbers.
* Mac only: XGBoost needs the OpenMP runtime. Install it once with `brew install libomp`,
  otherwise importing XGBoost fails with "libxgboost.dylib could not be loaded".

1. Clone the repo and go into it:

```
git clone git@github.com:tromedo/257-ADBench.git
cd 257-ADBench
```

2. Create a virtual environment and install packages:

```
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

3. Download the six datasets into `data/` (they are not stored in git):

```
python src/download_data.py
```

4. Check the data module works:

```
python src/data.py
```

With seed 1 you should see, for example, `annthyroid train= 5040 (anomalies 374)` and
`labeled at 1%: 3`. If your numbers match, your splits are identical to everyone else's.

In later sessions you only need `cd 257-ADBench` and `source .venv/bin/activate`.

## Data Exploration

`src/data_exploration.ipynb` looks at the six datasets before any training: dataset sizes, anomaly rates, PCA plots of how separable the anomalies are, the most informative features, and a check of the feature scaling.

To run it: open the notebook in any notebook editor, select `.venv` Python as the kernel, and run all cells. 


## Data quality

`python src/check_data_quality.py` reports missing values and duplicate rows for each dataset.
There are no missing values. Annthyroid, Breastw and Cardiotocography contain duplicate rows;
following ADBench, we keep the datasets as provided and do not remove them. The script also
shows how many test rows also appear in the training set, which matters most for Breastw.

## Running the experiments

Run each script from the repo root with the virtual environment active. Each one deletes its
old results file first, so re-running never duplicates rows.

| Script | Algorithms | Category | Output | Runs | Time (MacBook Pro) |
|---|---|---|---|---|---|
| `python src/run_unsupervised.py` | ECOD, IForest | Unsupervised | `results/unsupervised_results.csv` | 36 | seconds |
| `python src/run_supervised.py` | Random Forest, XGBoost | Supervised | `results/supervised/random_forest_results.csv`, `results/supervised/xgboost_results.csv` | 216 | under a minute |
| `python src/run_xgbod.py` | XGBOD | Semi-supervised | `results/xgbod_results.csv` | 108 | about 3 minutes |
| `python src/run_devnet.py` | DevNet | Semi-supervised | `results/devnet_results.csv` | 108 | under 30 seconds |

Unsupervised methods ignore labels, so they run once per dataset and seed (label fraction 0).
All other methods run every dataset x label fraction (1, 5, 10, 25, 50, 100%) x seed (1, 2, 3).

Notes:

* XGBOD uses PyOD's implementation with default settings, the same one ADBench uses.
* DevNet is a PyTorch port of the ADBench DevNet code (`src/devnet.py`), keeping the original
  architecture and training settings. The original is Keras/TensorFlow 1 code that does not run
  on current Python. Running `python src/devnet.py` prints sanity checks for the loss, the batch
  sampler, training and scoring.

## Adding your algorithm

Every algorithm uses the shared data module and the shared results format, so all results
are comparable and can be plotted together. Put your script in `src/` (for example
`src/run_supervised.py`) and run it from the repo root with `python src/run_supervised.py`.
The pattern for one run:

```python
import time
from data import load_dataset, split_data, hide_labels, scale_features
from results import evaluate, save_result

X, y = load_dataset("annthyroid")
X_train, X_test, y_train, y_test = split_data(X, y, seed=1)
y_semi, n_labeled = hide_labels(y_train, fraction=0.01, seed=1)
X_train, X_test = scale_features(X_train, X_test)

start = time.time()
model = ...                      # your algorithm
model.fit(X_train, y_semi)       # unsupervised methods ignore the labels
scores = ...                     # anomaly scores for X_test, higher = more anomalous
auc_roc, auc_pr = evaluate(y_test, scores)

save_result({
    "dataset": "annthyroid", "algorithm": "YourAlgo", "category": "supervised",
    "label_fraction": 0.01, "n_labeled": n_labeled, "seed": 1,
    "auc_roc": auc_roc, "auc_pr": auc_pr, "runtime_sec": time.time() - start,
}, "your_results.csv")
```

Rules so results stay comparable:

* Seeds 1, 2, 3 for everything.
* Label fractions from `LABEL_FRACTIONS` in `src/data.py` (unsupervised methods use 0.0).
* Always call `scale_features` (fitted on training data only).
* Write each person's results to their own CSV in `results/` to avoid merge conflicts.

## Reference

S. Han, X. Hu, H. Huang, M. Jiang, and Y. Zhao, "ADBench: Anomaly Detection Benchmark," NeurIPS 2022.
https://github.com/Minqi824/ADBench
