"""Model definitions. Hyperparameters are the best values found by the grouped cross-validated searches
in notebooks/hospreadmissionpred.ipynb (refit metric: PR-AUC)."""

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from readmission import config
from readmission.features import build_preprocessor

DISPLAY_NAMES = {
    "logistic_regression": "Logistic Regression",
    "decision_tree": "Decision Tree",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
    "mlp": "Neural Network (MLP)",
}


def build_model(name, n_jobs=1):
    """Return an unfitted preprocessing + estimator Pipeline for the named model."""
    seed = config.RANDOM_STATE
    if name == "logistic_regression":
        steps = (True, LogisticRegression(solver="liblinear", C=0.1, l1_ratio=0))
    elif name == "decision_tree":
        steps = (False, DecisionTreeClassifier(max_depth=10, min_samples_leaf=50, random_state=seed))
    elif name == "random_forest":
        steps = (False, RandomForestClassifier(n_estimators=50, max_depth=10, min_samples_leaf=10,
                                               random_state=seed, n_jobs=n_jobs))
    elif name == "xgboost":
        steps = (False, XGBClassifier(
            tree_method="hist", eval_metric="aucpr", random_state=seed, n_jobs=n_jobs,
            n_estimators=400, learning_rate=0.05, max_depth=6, min_child_weight=10,
            subsample=0.9, colsample_bytree=0.3, reg_lambda=10, scale_pos_weight=1))
    elif name == "mlp":
        steps = (True, MLPClassifier(hidden_layer_sizes=(64,), alpha=0.1, early_stopping=True,
                                     validation_fraction=0.1, n_iter_no_change=5, max_iter=100,
                                     random_state=seed))
    else:
        raise ValueError(f"Unknown model '{name}'. Choose from: {', '.join(DISPLAY_NAMES)}")
    scale_numeric, estimator = steps
    return Pipeline([("preprocess", build_preprocessor(scale_numeric)), ("model", estimator)])
