"""Reproduce the out-of-fold model comparison table (training data only, test set untouched).

    python -m readmission.compare --data-dir data --out reports/oof_comparison.csv

Uses the tuned hyperparameters stored in readmission.models; the notebook shows how they were found.
"""

import argparse
from pathlib import Path

import pandas as pd

from readmission.data import DATA_FILE, prepare_datasets, split_xy
from readmission.evaluation import evaluate_scores, oof_probabilities, select_f1_threshold
from readmission.models import DISPLAY_NAMES, build_model


def compare(data_dir):
    train_df, _ = prepare_datasets(Path(data_dir) / DATA_FILE)
    X_train, y_train, groups = split_xy(train_df)

    rows = {}
    for key, display_name in DISPLAY_NAMES.items():
        scores = oof_probabilities(build_model(key, n_jobs=1), X_train, y_train, groups)
        rows[display_name] = evaluate_scores(y_train, scores, select_f1_threshold(y_train, scores))
    return pd.DataFrame(rows).T.sort_values("PR-AUC", ascending=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--out", default="reports/oof_comparison.csv")
    args = parser.parse_args(argv)

    table = compare(args.data_dir)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    table.round(4).to_csv(args.out)
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(table.round(3).to_string())
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
