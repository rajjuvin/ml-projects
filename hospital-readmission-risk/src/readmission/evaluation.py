"""Model comparison metrics: threshold-free, fixed-threshold and capacity-based (top-k% flagged)."""

import numpy as np
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict

from readmission import config

CAPACITY_FRACTIONS = [0.05, 0.10, 0.20]


def make_cv_splitter():
    """Patient-grouped, stratified folds. Identical across models so results are comparable."""
    return StratifiedGroupKFold(n_splits=config.N_SPLITS, shuffle=True, random_state=config.RANDOM_STATE)


def oof_probabilities(model, X, y, groups, n_jobs=-1):
    """Out-of-fold P(readmitted < 30 days): each row is scored by a model that never saw its patient."""
    return cross_val_predict(clone(model), X, y, cv=make_cv_splitter(), groups=groups,
                             method="predict_proba", n_jobs=n_jobs)[:, 1]


def select_f1_threshold(y_true, y_score):
    """Probability cutoff that maximizes F1 on the given scores (use out-of-fold training scores only)."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    f1 = 2 * precision[:-1] * recall[:-1] / np.clip(precision[:-1] + recall[:-1], 1e-12, None)
    return float(thresholds[np.argmax(f1)])


def capacity_metrics(y_true, y_score, fractions=CAPACITY_FRACTIONS):
    """Precision, recall and lift when only the top `fraction` of highest-scored encounters are flagged."""
    y_true = np.asarray(y_true)
    order = np.argsort(-np.asarray(y_score), kind="stable")
    base_rate = y_true.mean()
    metrics = {}
    for fraction in fractions:
        n_flagged = max(1, int(round(fraction * len(y_true))))
        hits = y_true[order[:n_flagged]].sum()
        precision = hits / n_flagged
        metrics[f"Precision@Top{fraction:.0%}"] = precision
        metrics[f"Recall@Top{fraction:.0%}"] = hits / y_true.sum()
        metrics[f"Lift@Top{fraction:.0%}"] = precision / base_rate
    return metrics


def evaluate_scores(y_true, y_score, threshold):
    """Threshold-free, fixed-threshold and capacity-based metrics for one set of predicted probabilities."""
    y_pred = (np.asarray(y_score) >= threshold).astype(int)
    metrics = {
        "Base Rate": float(np.mean(y_true)),
        "ROC-AUC": roc_auc_score(y_true, y_score),
        "PR-AUC": average_precision_score(y_true, y_score),
        "Threshold": threshold,
        "Flagged@Threshold": float(y_pred.mean()),
        "Accuracy@Threshold": accuracy_score(y_true, y_pred),
        "Precision@Threshold": precision_score(y_true, y_pred, zero_division=0),
        "Recall@Threshold": recall_score(y_true, y_pred),
        "F1@Threshold": f1_score(y_true, y_pred),
    }
    metrics.update(capacity_metrics(y_true, y_score))
    return metrics
