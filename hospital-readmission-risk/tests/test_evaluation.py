import numpy as np
import pytest

from readmission.evaluation import capacity_metrics, evaluate_scores, select_f1_threshold


def test_capacity_metrics_perfect_ranking():
    y = np.array([1] * 10 + [0] * 90)
    scores = np.linspace(1, 0, 100)  # positives scored highest
    metrics = capacity_metrics(y, scores, fractions=[0.10])
    assert metrics["Precision@Top10%"] == pytest.approx(1.0)
    assert metrics["Recall@Top10%"] == pytest.approx(1.0)
    assert metrics["Lift@Top10%"] == pytest.approx(10.0)  # base rate 0.10


def test_capacity_metrics_random_ranking_has_lift_near_one():
    rng = np.random.default_rng(0)
    y = (rng.random(20000) < 0.1).astype(int)
    metrics = capacity_metrics(y, rng.random(20000), fractions=[0.2])
    assert metrics["Lift@Top20%"] == pytest.approx(1.0, abs=0.1)


def test_capacity_metrics_known_small_case():
    y = np.array([1, 0, 1, 0, 0, 0, 0, 0, 0, 0])
    scores = np.array([0.9, 0.8, 0.7, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    metrics = capacity_metrics(y, scores, fractions=[0.2])  # top 2 flagged: one hit out of two positives
    assert metrics["Precision@Top20%"] == pytest.approx(0.5)
    assert metrics["Recall@Top20%"] == pytest.approx(0.5)
    assert metrics["Lift@Top20%"] == pytest.approx(2.5)


def test_select_f1_threshold_separates_clean_classes():
    y = np.array([0, 0, 0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.15, 0.3, 0.7, 0.8])
    threshold = select_f1_threshold(y, scores)
    assert 0.3 < threshold <= 0.7


def test_evaluate_scores_reports_all_comparison_metrics():
    rng = np.random.default_rng(1)
    y = (rng.random(2000) < 0.12).astype(int)
    scores = np.clip(0.12 + 0.3 * y + rng.normal(0, 0.15, 2000), 0, 1)
    metrics = evaluate_scores(y, scores, threshold=0.3)
    for key in ("Base Rate", "ROC-AUC", "PR-AUC", "Threshold", "Flagged@Threshold", "Accuracy@Threshold",
                "Precision@Threshold", "Recall@Threshold", "F1@Threshold", "Precision@Top10%", "Lift@Top5%"):
        assert key in metrics
    assert metrics["ROC-AUC"] > 0.8
    assert metrics["PR-AUC"] > metrics["Base Rate"]


def test_default_cutoff_with_no_positives_gives_zero_precision_not_an_error():
    y = np.array([0, 1, 0, 1, 0, 0, 0, 0, 0, 0])
    scores = np.full(10, 0.2)  # every score is below 0.5: no encounter is flagged
    metrics = evaluate_scores(y, scores, threshold=0.5)
    assert metrics["Precision@Threshold"] == 0.0
    assert metrics["Recall@Threshold"] == 0.0
