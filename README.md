# Label Efficiency in Anomaly Detection

CMPE 257 group project (SJSU, Fall 2026): a replication of Angle I of ADBench (NeurIPS 2022).
We test how much a small fraction of labeled anomalies improves detection compared to fully unsupervised methods.

Team: Manisha Varthamanan, Dipti Ghiya, Sriram Sridhar

## Scope

Datasets (ADBench Classical): Annthyroid, Cardiotocography, Breastw, Lymphography, Vertebral, Hepatitis,
plus three larger ADBench datasets (Satellite, Landsat, Magic Gamma) added to test whether the
1% result depends on having enough labeled anomalies.

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
data/       downloaded .npz datasets (not committed)
docs/       proposal, paper comparison, unsupervised baselines notes
notebooks/  data exploration
src/        shared data module, algorithm runners, plots, app, reproduce script
results/    results CSVs (larger datasets in results/larger_dataset/) and figures/
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

`notebooks/data_exploration.ipynb` looks at the nine datasets before any training: dataset sizes, anomaly rates, PCA plots of how separable the anomalies are, the most informative features, and a check of the feature scaling.

To run it: open the notebook in any notebook editor, select `.venv` Python as the kernel, and run all cells. 


## Reproduce everything (one command)

```
python src/reproduce.py                 # all datasets, all methods, figures (about 40 min)
python src/reproduce.py --skip-larger   # 6 main datasets only (a few minutes)
python src/reproduce.py --plots-only    # redraw figures from existing results
```

Runs the dataset download, the data quality check, every experiment runner and the plots, in
order, and stops at the first failure. Most of the full run is XGBOD on the larger datasets.
With the pinned versions in `requirements.txt`, the regenerated results CSVs should match the
committed ones, so `git status` should show no changes afterwards.

## Settings and hyperparameters

All values below are what the code actually runs with (pinned versions in `requirements.txt`).
Unless listed, every other parameter is the library default.

### Shared experimental setup (all methods)

| Setting | Our value | ADBench |
|---|---|---|
| Train/test split | Stratified 70/30 (`train_test_split`, `stratify=y`) | Same |
| Seeds | 1, 2, 3; results are the mean (and std) over seeds | Same |
| Label fractions | 1, 5, 10, 25, 50, 100% of training anomalies; unsupervised methods use none | Also 0% and 75% |
| Small label counts | `max(1, int(fraction x anomalies))`: rounds down, at least 1 labeled anomaly | `ceil(fraction x anomalies)`: rounds up (its experiments call the generator with `at_least_one_labeled=True`), so also at least 1; differs from ours by at most 1 label |
| Unlabeled anomalies | Stay in training data, labeled 0 | Same |
| Feature scaling | `StandardScaler` fitted on training data only | `MinMaxScaler` fitted on training data only |
| Small datasets (< 1,000 samples) | Used as provided | Resampled with replacement up to 1,000 samples |
| Large datasets (> 10,000 samples) | Used as provided (Magic Gamma: 19,020) | Subsampled to 10,000 |
| Duplicate rows | Kept (see Data quality) | Kept |
| Metrics | AUC-ROC (`roc_auc_score`), AUC-PR (`average_precision_score`) | Same |

### Per algorithm

| Algorithm | Category | Implementation | Settings |
|---|---|---|---|
| ECOD | Unsupervised | PyOD `ECOD` | Defaults; no tunable parameters affect the scores |
| IForest | Unsupervised | PyOD `IForest` | 100 trees, `max_samples="auto"` (min(256, n)), all features, no bootstrap, `random_state=seed` |
| Random Forest | Supervised | scikit-learn `RandomForestClassifier` | 100 trees, Gini, no max depth, `max_features="sqrt"`, bootstrap, `random_state=seed`; score = predicted probability of anomaly |
| XGBoost | Supervised | `xgboost.XGBClassifier` | 100 trees, library defaults (max depth 6, learning rate 0.3), `eval_metric="logloss"`, **`base_score=0.5`**, `random_state=seed` |
| XGBOD | Semi-supervised | PyOD `XGBOD` | 41 unsupervised detectors (KNN and LOF with k in 1, 3, 5, 10, 20, 30, 40, 50; HBOS with 5 to 50 bins; OCSVM with nu from 0.01 to 0.99; IForest with 10 to 200 trees) whose scores are appended as features, then XGBoost with 100 trees, max depth 3, learning rate 0.1, `base_score=0.5`; `random_state=seed` |
| DevNet | Semi-supervised | Our PyTorch port of ADBench's DevNet (`src/devnet.py`) | 1 hidden layer of 20 ReLU units, L2 0.01 on hidden weights, deviation loss with 5,000 N(0, 1) reference scores and margin 5, balanced batches of 512 (half labeled anomalies, sampled with replacement), 20 batches per epoch, 50 epochs, RMSprop (lr 0.001, rho 0.9, eps 1e-7), gradient norm clipping 1.0, weights from the lowest-loss epoch kept |

Notes on deliberate choices:

* **XGBoost `base_score=0.5`:** from XGBoost 2.0 on, `base_score` is estimated from the labels. With
  1 or 2 labeled anomalies this blocks every split and gives constant predictions (AUC 0.5). Setting
  0.5 restores the older default and matches what PyOD's XGBOD uses.
* **DevNet port:** ADBench runs the DevNet authors' Keras/TensorFlow 1 code, which does not run on
  current Python. Our port keeps the architecture and training settings above, and matches Keras
  details (population standard deviation in the loss, Glorot uniform initial weights, zero biases).
* **No hyperparameter tuning:** all methods use default or paper settings, as in ADBench and our
  proposal, since tuning unsupervised methods without labels is not realistic.



## Data quality

`python src/check_data_quality.py` reports missing values and duplicate rows for each dataset.
There are no missing values. Annthyroid, Breastw, Cardiotocography and Magic Gamma contain duplicate rows;
following ADBench, we keep the datasets as provided and do not remove them. The script also
shows how many test rows also appear in the training set, which matters most for Breastw.

## Running the experiments

Run each script from the repo root with the virtual environment active. Each one deletes its
old results file first, so re-running never duplicates rows.

| Script | Algorithms | Category | Output | Runs | Time (MacBook Pro) |
|---|---|---|---|---|---|
| `python src/run_unsupervised.py` | ECOD, IForest | Unsupervised | `results/unsupervised_results.csv` | 36 | seconds |
| `python src/run_supervised.py` | Random Forest, XGBoost | Supervised | `results/random_forest_results.csv`, `results/xgboost_results.csv` | 216 | under a minute |
| `python src/run_xgbod.py` | XGBOD | Semi-supervised | `results/xgbod_results.csv` | 108 | about 3 minutes |
| `python src/run_devnet.py` | DevNet | Semi-supervised | `results/devnet_results.csv` | 108 | under 30 seconds |

To run only specific datasets, add their names after the script. Results then go to
`results/larger_dataset/` instead, so the main results are not touched:

| Script | Output |
|---|---|
| `python src/run_unsupervised.py satellite landsat magic_gamma` | `results/larger_dataset/unsupervised_extra_results.csv` |
| `python src/run_supervised.py satellite landsat magic_gamma` | `results/larger_dataset/random_forest_extra_results.csv`, `xgboost_extra_results.csv` |
| `python src/run_xgbod.py satellite landsat magic_gamma` | `results/larger_dataset/xgbod_extra_results.csv` (~30 min) |
| `python src/run_devnet.py satellite landsat magic_gamma` | `results/larger_dataset/devnet_extra_results.csv` |

Unsupervised methods ignore labels, so they run once per dataset and seed (label fraction 0).
All other methods run every dataset x label fraction (1, 5, 10, 25, 50, 100%) x seed (1, 2, 3).

Notes:

* XGBOD uses PyOD's implementation with default settings, the same one ADBench uses.
* DevNet is a PyTorch port of the ADBench DevNet code (`src/devnet.py`), keeping the original
  architecture and training settings. The original is Keras/TensorFlow 1 code that does not run
  on current Python. Running `python src/devnet.py` prints sanity checks for the loss, the batch
  sampler, training and scoring.


## Plots

`python src/plot_results.py` reads every results CSV in `results/` (via `src/load_results.py`)
and saves two figures to `results/figures/`:

* `auc_roc_vs_label_fraction.png` and `auc_pr_vs_label_fraction.png`: one panel per dataset
* `median_auc_roc_all_datasets.png`: median across all datasets
* `larger_datasets_1pct.png`: every method at 1% labels on the larger datasets

`notebooks/data_exploration.ipynb` also saves `labeled_anomalies_per_fraction.png`.

The two per-dataset figures have one panel per dataset and one line per algorithm (mean of 3 seeds). Unsupervised
methods are flat dashed lines, and the AUC-PR figure marks the random-guess level (the
dataset's anomaly ratio). A new algorithm appears automatically once its runner writes a
results CSV in the shared format. Re-run the script after adding or updating results.

## Results and comparison with the paper

Median AUC-ROC across all 9 datasets (mean of 3 seeds per dataset), the same summary
ADBench uses.

| Method | Category | 1% | 5% | 10% | 25% | 50% | 100% |
|---|---|---|---|---|---|---|---|
| XGBOD | Semi-supervised | **79.2** | **88.5** | **90.7** | **94.2** | 95.4 | 96.7 |
| DevNet | Semi-supervised | 69.5 | 83.3 | 86.8 | 88.9 | 90.3 | 94.5 |
| XGBoost | Supervised | 76.5 | 82.5 | 86.0 | 91.6 | 94.5 | 97.3 |
| Random Forest | Supervised | 60.1 | 80.3 | 85.3 | 91.8 | **96.3** | **97.4** |
| ECOD | Unsupervised | 78.4 | | | | | |
| IForest | Unsupervised | 70.8 | | | | | |

![Median AUC-ROC across all datasets](results/figures/median_auc_roc_all_datasets.png)

| Paper's finding (Angle I) | Our result | Match |
|---|---|---|
| With 1% labeled anomalies, semi-supervised methods beat the best unsupervised method | XGBOD 79.2% vs ECOD 78.4%: holds for XGBOD, narrowly. DevNet (69.5%) does not | Yes for XGBOD |
| Semi-supervised median at 1% is about 75.6% | XGBOD 79.2%, DevNet 69.5% | Close |
| Semi-supervised methods are best at low label rates | XGBOD is the best method from 1% to 25% | Yes |
| Supervised methods need more labels to beat unsupervised ones | Random Forest is lowest at 1% (60.1%); both supervised methods pass ECOD at 5% | Yes |
| Supervised methods catch up with more labels | Random Forest is best at 50% and 100% | Yes |
| All methods converge with full labels | 94.5 to 97.4% at 100% | Yes |

Most findings reproduce. With 1% labels, XGBOD narrowly beats the best unsupervised
method across all 9 datasets; the advantage comes from the larger datasets, where 1%
gives 9 to 46 labeled anomalies instead of 1 to 3. Full analysis:
[docs/paper_comparison.md](docs/paper_comparison.md).

## Results explorer (web app)

```
streamlit run src/app.py
```

Opens a local page in the browser that reads every results CSV through `src/load_results.py`,
so new algorithms appear automatically. Use the "Reload results" button after re-running an
experiment.

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
