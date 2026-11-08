"""Reproduce every result and figure in this repo with one command.

Runs, in order: dataset download, data quality check, all four runners on the
6 main datasets, all four runners on the larger datasets, and the figures.
Each step calls the existing script, so this file never duplicates their logic.

Run from the repo root:
    python src/reproduce.py                 # everything (about 40 min, mostly XGBOD on larger datasets)
    python src/reproduce.py --skip-larger   # main 6 datasets only (a few minutes)
    python src/reproduce.py --plots-only    # just redraw the figures from existing results

With the pinned versions in requirements.txt, the regenerated results CSVs
should match the committed ones exactly (check with `git status`).
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

from data import DATASET_NAMES
from download_data import DATASETS

SRC = Path(__file__).resolve().parent
REPO = SRC.parent

# Larger datasets: everything download_data.py knows about beyond the 6 main ones
LARGER_DATASETS = [d for d in DATASETS if d not in DATASET_NAMES]
RUNNERS = ["run_unsupervised.py", "run_supervised.py", "run_xgbod.py", "run_devnet.py"]


def build_steps(skip_larger, plots_only):
    """List of (description, script, arguments) in the order they should run."""
    if plots_only:
        return [("Figures", "plot_results.py", [])]
    steps = [("Download datasets", "download_data.py", []),
             ("Data quality check", "check_data_quality.py", [])]
    steps += [(f"{r} on the 6 main datasets", r, []) for r in RUNNERS]
    if not skip_larger:
        steps += [(f"{r} on {', '.join(LARGER_DATASETS)}", r, LARGER_DATASETS) for r in RUNNERS]
    steps.append(("Figures", "plot_results.py", []))
    return steps


def main():
    parser = argparse.ArgumentParser(description="Reproduce all results and figures.")
    parser.add_argument("--skip-larger", action="store_true",
                        help="only the 6 main datasets (skips the slow larger-dataset runs)")
    parser.add_argument("--plots-only", action="store_true",
                        help="only regenerate figures from the existing results CSVs")
    args = parser.parse_args()

    steps = build_steps(args.skip_larger, args.plots_only)
    total_start = time.time()
    for i, (description, script, script_args) in enumerate(steps, start=1):
        print(f"\n=== [{i}/{len(steps)}] {description} ===", flush=True)
        start = time.time()
        result = subprocess.run([sys.executable, str(SRC / script), *script_args], cwd=REPO)
        if result.returncode != 0:
            print(f"\nStopped: {script} failed (exit code {result.returncode}).")
            sys.exit(result.returncode)
        print(f"--- done in {time.time() - start:.0f}s", flush=True)

    noun = "step" if len(steps) == 1 else "steps"
    print(f"\nAll {len(steps)} {noun} finished in {(time.time() - total_start) / 60:.1f} min.")
    print("Run `git status` to confirm the regenerated results match the committed ones.")


if __name__ == "__main__":
    main()
