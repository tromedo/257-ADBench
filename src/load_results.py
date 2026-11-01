"""Shared results loader for plots and the web app.

Finds every results CSV under results/ (including subfolders) that uses the
shared column layout from results.py, and combines them into one table. A new
algorithm is picked up automatically as soon as its runner writes a CSV there.

Run from the repo root to see what is found:
    python src/load_results.py
"""

import warnings

import pandas as pd

from results import COLUMNS, RESULTS_DIR


def load_all_results(results_dir=RESULTS_DIR):
    """Return one DataFrame with every result row from every matching CSV."""
    frames = []
    for path in sorted(results_dir.rglob("*.csv")):
        df = pd.read_csv(path)
        if list(df.columns) != COLUMNS:
            warnings.warn(f"Skipping {path.relative_to(results_dir)}: columns do not match results.py")
            continue
        df["source_file"] = str(path.relative_to(results_dir))
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No results CSVs found in {results_dir}")
    return pd.concat(frames, ignore_index=True)


def summarize(df, metric="auc_roc"):
    """Mean and standard deviation over seeds for each dataset, algorithm and label fraction."""
    return (df.groupby(["dataset", "category", "algorithm", "label_fraction"])[metric]
              .agg(["mean", "std", "count"])
              .reset_index())


if __name__ == "__main__":
    df = load_all_results()
    print(f"{len(df)} rows from {df['source_file'].nunique()} files\n")
    print(df.groupby(["source_file", "category", "algorithm"]).size().to_string())
    print("\nAUC-ROC at 1% labeled (unsupervised shown at 0%):")
    s = summarize(df)
    low = s[s["label_fraction"].isin([0.0, 0.01])]
    print(low.pivot_table(index="dataset", columns="algorithm", values="mean").round(3).to_string())
