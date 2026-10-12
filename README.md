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

## Setup

```
pip install -r requirements.txt
```

## Reference

S. Han, X. Hu, H. Huang, M. Jiang, and Y. Zhao, "ADBench: Anomaly Detection Benchmark," NeurIPS 2022.
https://github.com/Minqi824/ADBench
