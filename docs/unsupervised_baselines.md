# Unsupervised baselines: ECOD and IForest

Owner: Dipti. Code: `src/run_unsupervised.py`. Results: `results/unsupervised_results.csv`.

## Methods

Both detectors ignore labels, so each is run once per dataset and seed at label fraction 0.0.
That score is the flat baseline every label-using model is compared against at each label fraction.
Both use PyOD 3.6.7 with default settings (no tuning), on the shared splits and scaling from `src/data.py`.

* **ECOD** (Empirical Cumulative distribution based Outlier Detection, Li et al. 2022).
  For each feature it estimates the empirical distribution and measures how far into the left
  and right tails a sample falls, then sums these tail probabilities across features.
  It has no hyperparameters and no randomness, so the seed only changes the train/test split.
* **IForest** (Isolation Forest, Liu et al. 2008). Builds random trees that split on random
  features at random thresholds. Anomalies are isolated in fewer splits, so a shorter average
  path length means a higher score. Defaults: 100 trees, 256 samples per tree (or all rows if
  fewer), all features. `random_state` is set to the seed.

## Results

Mean ± std over seeds 1, 2, 3. Higher is better.

| Dataset | ECOD AUC-ROC | IForest AUC-ROC | ECOD AUC-PR | IForest AUC-PR | Random AUC-PR |
|---|---|---|---|---|---|
| annthyroid | 0.787 ± 0.008 | **0.832 ± 0.018** | 0.254 ± 0.021 | **0.291 ± 0.010** | 0.074 |
| cardiotocography | **0.784 ± 0.020** | 0.691 ± 0.013 | **0.499 ± 0.066** | 0.432 ± 0.053 | 0.220 |
| breastw | **0.994 ± 0.003** | 0.991 ± 0.006 | **0.990 ± 0.006** | 0.981 ± 0.012 | 0.350 |
| lymphography | 0.996 ± 0.007 | **1.000 ± 0.000** | 0.944 ± 0.096 | **1.000 ± 0.000** | 0.041 |
| vertebral | **0.391 ± 0.039** | 0.290 ± 0.067 | **0.114 ± 0.011** | 0.095 ± 0.008 | 0.125 |
| hepatitis | **0.788 ± 0.088** | 0.708 ± 0.072 | **0.433 ± 0.197** | 0.313 ± 0.072 | 0.162 |

"Random AUC-PR" is the anomaly rate, which is what a random scorer gets on average.

## Observations

* ECOD is the stronger baseline on 4 of 6 datasets (cardiotocography, breastw, vertebral,
  hepatitis). IForest wins on annthyroid and lymphography.
* Both are worse than random on vertebral: AUC-ROC below 0.5 and AUC-PR below the 0.125
  anomaly rate. The anomalies there do not look unusual to either detector.
* breastw and lymphography are near the ceiling for both detectors (AUC-ROC 0.99 or higher),
  so labels have little room to help there.
* Hepatitis has only 4 test anomalies per split, so its standard deviations are large.

## Against the label-using models at 1% labels

Mean AUC-PR at 1% labeled anomalies, next to the best unsupervised model for each dataset:

| Dataset | Best unsupervised | DevNet | XGBOD | Random Forest | XGBoost |
|---|---|---|---|---|---|
| annthyroid | 0.291 (IForest) | 0.153 | **0.522** | 0.144 | **0.462** |
| cardiotocography | 0.499 (ECOD) | 0.494 | 0.427 | 0.350 | **0.590** |
| breastw | 0.990 (ECOD) | 0.663 | 0.680 | 0.462 | 0.910 |
| lymphography | 1.000 (IForest) | 0.352 | 0.694 | 0.498 | 0.265 |
| vertebral | 0.114 (ECOD) | **0.287** | **0.184** | **0.173** | **0.221** |
| hepatitis | 0.433 (ECOD) | **0.444** | **0.455** | **0.621** | **0.620** |

Bold = beats the best unsupervised model.

* By AUC-PR, a label-using model at 1% beats the best unsupervised model in 11 of 24
  (dataset, model) pairs. By AUC-ROC it does so in only 7 of 24.
* On breastw and lymphography no label-using model at 1% reaches the unsupervised baseline,
  by either metric.
* On vertebral, where both unsupervised detectors are worse than random, all four label-using
  models beat them at 1% by both metrics.
* XGBoost is the most consistent at 1%: it beats the best unsupervised model in 4 of 6
  datasets by AUC-PR and 3 of 6 by AUC-ROC.

The combined figures are made by `python src/plot_results.py`, and `streamlit run src/app.py`
shows the same comparison interactively.

## Differences from ADBench's code

Our shared setup (`src/data.py`) differs from ADBench's data generator in these ways:

* Scaling: we standardize (mean 0, std 1). ADBench uses min-max scaling.
* Labeled count: we use `max(1, int(fraction * n))`, which rounds down. ADBench uses ceil.
* Label fractions: ADBench also includes 75%. We use 1, 5, 10, 25, 50 and 100%.
* Small datasets: ADBench resamples datasets under 1000 rows with replacement up to 1000
  before splitting, so copies of the same row can land in both train and test.
  We keep the original rows. This affects breastw, lymphography, vertebral and hepatitis.
* Seeds: both use 1, 2 and 3.
