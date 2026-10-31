"""Search for each model's best hyperparameters and save them to reports/best_params.json.

    python -m readmission.tune --data-dir data
    python -m readmission.tune --models decision_tree xgboost
    python -m readmission.tune --fast --out scratch/best_params.json     # quick smoke test, 2 candidates per model

The search uses the training split only (the test set is never touched), the same patient-grouped folds as
the model comparison, and selects on PR-AUC (average precision). `readmission.train` and
`readmission.compare` read the saved file, so tuned values are changed by re-running this script, not by
editing code. A full run takes about 20 minutes, mostly XGBoost.
"""

import argparse
import json
import math
import platform
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import sklearn
import xgboost
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from readmission import config
from readmission.data import DATA_FILE, prepare_datasets, split_xy
from readmission.evaluation import make_cv_splitter
from readmission.models import DISPLAY_NAMES, PARAMS_FILE, build_model
from readmission.train import file_sha256

SCORING = "average_precision"
SCHEMA_VERSION = 1
FAST_N_ITER = 2
# Models searched by random sampling; the others are searched exhaustively over their grid.
RANDOM_SEARCH_N_ITER = {"xgboost": 30}


def search_space(name, y):
    """Candidate values per hyperparameter. Same search spaces as the notebook."""
    spaces = {
        "logistic_regression": {
            "C": [0.01, 0.1, 1.0],
            "l1_ratio": [0],
            "class_weight": [None, "balanced"],
        },
        "decision_tree": {
            "max_depth": [5, 10, None],
            "min_samples_leaf": [10, 50],
            "class_weight": [None, "balanced"],
        },
        "random_forest": {
            "max_depth": [10],
            "min_samples_leaf": [10],
            "class_weight": [None, "balanced"],
        },
        "xgboost": {
            "n_estimators": [200, 400, 800],
            "learning_rate": [0.01, 0.03, 0.05],
            "max_depth": [2, 3, 4, 6],
            "min_child_weight": [1, 10, 50],
            "subsample": [0.7, 0.9],
            "colsample_bytree": [0.3, 0.6],
            "reg_lambda": [1, 10],
            "scale_pos_weight": [1, float((y == 0).sum() / (y == 1).sum())],
        },
        "mlp": {
            "hidden_layer_sizes": [(64,), (128, 64)],
            "alpha": [0.01, 0.1, 1.0],
        },
    }
    if name not in spaces:
        raise ValueError(f"Unknown model '{name}'. Choose from: {', '.join(DISPLAY_NAMES)}")
    return spaces[name]


def tune_model(name, X, y, groups, n_jobs=-1, n_iter=None):
    """Grouped cross-validated search for one model; returns its best parameters and CV score.

    Searches the whole grid unless `n_iter` (or the model's default) asks for fewer random candidates."""
    space = search_space(name, y)
    grid_size = math.prod(len(values) for values in space.values())
    n_iter = n_iter if n_iter is not None else RANDOM_SEARCH_N_ITER.get(name)
    options = dict(
        estimator=build_model(name, n_jobs=1),
        cv=make_cv_splitter(),
        scoring=SCORING,
        refit=False,
        error_score="raise",
        n_jobs=n_jobs,
    )
    prefixed = {f"model__{key}": values for key, values in space.items()}
    if n_iter is None or n_iter >= grid_size:
        search, kind, n_candidates = GridSearchCV(param_grid=prefixed, **options), "grid", grid_size
    else:
        search = RandomizedSearchCV(param_distributions=prefixed, n_iter=n_iter,
                                    random_state=config.RANDOM_STATE, **options)
        kind, n_candidates = f"randomized (n_iter={n_iter})", n_iter
    search.fit(X, y, groups=groups)

    best = search.best_index_
    return {
        "params": {key.removeprefix("model__"): value for key, value in search.best_params_.items()},
        "cv_pr_auc_mean": float(search.cv_results_["mean_test_score"][best]),
        "cv_pr_auc_std": float(search.cv_results_["std_test_score"][best]),
        "search": kind,
        "n_candidates": n_candidates,
        "search_space": space,
    }


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def tune(data_dir, model_keys, out, n_jobs=-1, n_iter=None):
    """Tune the given models and merge their results into the JSON file `out` (other models are kept)."""
    data_path = Path(data_dir) / DATA_FILE
    train_df, _ = prepare_datasets(data_path)
    X_train, y_train, groups = split_xy(train_df)
    data_sha256 = file_sha256(data_path)

    out = Path(out)
    saved = json.loads(out.read_text(encoding="utf-8"))["models"] if out.exists() else {}
    for key in model_keys:
        started = time.perf_counter()
        print(f"Tuning {DISPLAY_NAMES[key]} ...", flush=True)
        result = tune_model(key, X_train, y_train, groups, n_jobs=n_jobs, n_iter=n_iter)
        result.update({
            "fast_run": n_iter is not None,
            "tuned_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
            "data_file_sha256": data_sha256,
            "n_train_encounters": int(len(train_df)),
            "libraries": {"python": platform.python_version(), "scikit-learn": sklearn.__version__,
                          "xgboost": xgboost.__version__},
        })
        saved[key] = result
        print(f"  best CV PR-AUC {result['cv_pr_auc_mean']:.4f} (+/- {result['cv_pr_auc_std']:.4f}) "
              f"in {time.perf_counter() - started:.0f}s: {result['params']}", flush=True)

    document = {
        "schema_version": SCHEMA_VERSION,
        "scoring": SCORING,
        "cv": {"splitter": "StratifiedGroupKFold", "n_splits": config.N_SPLITS, "shuffle": True,
               "random_state": config.RANDOM_STATE},
        "models": {key: saved[key] for key in DISPLAY_NAMES if key in saved},
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(document, indent=2, default=_json_default) + "\n", encoding="utf-8")
    return document


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default="data", help="folder containing diabetic_data.csv")
    parser.add_argument("--models", nargs="+", choices=list(DISPLAY_NAMES), default=list(DISPLAY_NAMES),
                        help="models to tune (default: all); other models' saved results are kept")
    parser.add_argument("--out", default=None, help=f"parameter file to write (default: {PARAMS_FILE})")
    parser.add_argument("--n-jobs", type=int, default=-1, help="parallel search workers (default: all cores)")
    parser.add_argument("--fast", action="store_true",
                        help=f"smoke test: only {FAST_N_ITER} random candidates per model; requires --out")
    args = parser.parse_args(argv)

    if args.fast and args.out is None:
        parser.error(f"--fast gives rough results; pass --out so {PARAMS_FILE} is not overwritten")
    out = args.out or PARAMS_FILE

    tune(args.data_dir, args.models, out, n_jobs=args.n_jobs, n_iter=FAST_N_ITER if args.fast else None)
    print(f"\nSaved {out}. Review the change (git diff), then re-run `python -m readmission.compare` "
          "to refresh the comparison table.")


if __name__ == "__main__":
    main()
