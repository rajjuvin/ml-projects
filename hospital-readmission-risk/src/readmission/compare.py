"""Reproduce the out-of-fold model comparison table (training data only, test set untouched).

    python -m readmission.compare --data-dir data
    python -m readmission.compare --models decision_tree xgboost     # only these models; prints, does not save

Uses the tuned hyperparameters in reports/best_params.json (see readmission.tune), or the built-in defaults
in readmission.models when that file does not exist. A full run rewrites reports/oof_comparison.csv; a run
limited with --models only saves if you also pass --out, so a partial table never replaces the full one.
"""

import argparse
from pathlib import Path

import pandas as pd

from readmission.data import DATA_FILE, prepare_datasets, split_xy
from readmission.evaluation import evaluate_scores, oof_probabilities, select_f1_threshold
from readmission.models import DISPLAY_NAMES, build_model, load_params

DEFAULT_OUT = "reports/oof_comparison.csv"


def compare(data_dir, model_keys=None, params=None):
    """Out-of-fold metrics for the given models (default: all), best PR-AUC first.

    `params` is {model_key: hyperparameter overrides}, as returned by models.load_params."""
    train_df, _ = prepare_datasets(Path(data_dir) / DATA_FILE)
    X_train, y_train, groups = split_xy(train_df)
    params = params or {}

    rows = {}
    for key in model_keys or DISPLAY_NAMES:
        model = build_model(key, n_jobs=1, params=params.get(key))
        scores = oof_probabilities(model, X_train, y_train, groups)
        rows[DISPLAY_NAMES[key]] = evaluate_scores(y_train, scores, select_f1_threshold(y_train, scores))
    return pd.DataFrame(rows).T.sort_values("PR-AUC", ascending=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--models", nargs="+", choices=list(DISPLAY_NAMES), default=None,
                        help="compare only these models (default: all five)")
    parser.add_argument("--params", default=None,
                        help="tuned parameter file from readmission.tune (default: reports/best_params.json "
                             "if it exists, otherwise built-in defaults)")
    parser.add_argument("--out", default=None,
                        help=f"where to save the table (default: {DEFAULT_OUT} for a full run, nowhere with --models)")
    args = parser.parse_args(argv)

    out = args.out or (DEFAULT_OUT if args.models is None else None)
    table = compare(args.data_dir, args.models, load_params(args.params))
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(table.round(3).to_string())
    if out is None:
        print("\nNot saved (partial comparison). Pass --out to save it.")
        return
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    table.round(4).to_csv(out)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
