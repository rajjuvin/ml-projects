"""Model definitions.

Tuned hyperparameters are read from reports/best_params.json (written by `python -m readmission.tune`).
DEFAULT_PARAMS are the values found by the searches in notebooks/hospreadmissionpred.ipynb (refit metric:
PR-AUC) and are used for any model the file does not cover, or when no file exists."""

import json
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from readmission import config
from readmission.features import build_preprocessor

PARAMS_FILE = "reports/best_params.json"

DISPLAY_NAMES = {
    "logistic_regression": "Logistic Regression",
    "decision_tree": "Decision Tree",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
    "mlp": "Neural Network (MLP)",
}

# Only the searched hyperparameters; settings that are never tuned stay fixed in build_model.
DEFAULT_PARAMS = {
    "logistic_regression": {"C": 0.1, "l1_ratio": 0},
    "decision_tree": {"max_depth": 10, "min_samples_leaf": 50},
    "random_forest": {"max_depth": 10, "min_samples_leaf": 10},
    "xgboost": {"n_estimators": 400, "learning_rate": 0.05, "max_depth": 6, "min_child_weight": 10,
                "subsample": 0.9, "colsample_bytree": 0.3, "reg_lambda": 10, "scale_pos_weight": 1},
    "mlp": {"hidden_layer_sizes": (64,), "alpha": 0.1},
}


def _unknown_model(name):
    return ValueError(f"Unknown model '{name}'. Choose from: {', '.join(DISPLAY_NAMES)}")


def model_params(name, params=None):
    """Searched hyperparameters for one model: the defaults, overridden by `params` (a dict for that model)."""
    if name not in DEFAULT_PARAMS:
        raise _unknown_model(name)
    return {**DEFAULT_PARAMS[name], **(params or {})}


def load_params(path=None):
    """Tuned parameters from a best_params.json file, as {model_key: {parameter: value}}.

    With no path, reads PARAMS_FILE and returns {} when it does not exist (the defaults then apply).
    An explicitly given path must exist."""
    explicit = path is not None
    path = Path(path if explicit else PARAMS_FILE)
    if not path.exists():
        if explicit:
            raise FileNotFoundError(f"Tuned parameter file not found: {path}")
        return {}
    entries = json.loads(path.read_text(encoding="utf-8"))["models"]
    return {name: _from_json(entry["params"]) for name, entry in entries.items()}


def params_source(path, overrides):
    """Where a model's hyperparameters came from, for logs and metadata (`overrides`: that model's tuned values)."""
    if not overrides:
        return "built-in defaults"
    return str(path) if path is not None else PARAMS_FILE


def _from_json(params):
    # JSON has no tuples; scikit-learn expects one for the MLP's layer sizes.
    return {key: tuple(value) if key == "hidden_layer_sizes" else value for key, value in params.items()}


def build_model(name, n_jobs=1, params=None):
    """Return an unfitted preprocessing + estimator Pipeline for the named model.

    `params` overrides the default searched hyperparameters for this model (see model_params)."""
    seed = config.RANDOM_STATE
    tuned = model_params(name, params)
    if name == "logistic_regression":
        steps = (True, LogisticRegression(solver="liblinear", **tuned))
    elif name == "decision_tree":
        steps = (False, DecisionTreeClassifier(random_state=seed, **tuned))
    elif name == "random_forest":
        steps = (False, RandomForestClassifier(n_estimators=50, random_state=seed, n_jobs=n_jobs, **tuned))
    elif name == "xgboost":
        steps = (False, XGBClassifier(tree_method="hist", eval_metric="aucpr", random_state=seed,
                                      n_jobs=n_jobs, **tuned))
    else:
        steps = (True, MLPClassifier(early_stopping=True, validation_fraction=0.1, n_iter_no_change=5,
                                     max_iter=100, random_state=seed, **tuned))
    scale_numeric, estimator = steps
    return Pipeline([("preprocess", build_preprocessor(scale_numeric)), ("model", estimator)])
