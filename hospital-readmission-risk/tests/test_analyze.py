import json

import numpy as np
import pytest

from readmission.analyze import (
    BOOTSTRAP_METRICS,
    MIN_GROUP_ENCOUNTERS,
    analyze,
    bootstrap_intervals,
    calibration_table,
    group_metrics,
    patient_members,
)
from readmission.analyze import main as analyze_main
from readmission.data import DATA_FILE
from readmission.train import train


def informative_scores(n=600, seed=0):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.2).astype(int)
    score = np.clip(0.15 + 0.25 * y + rng.normal(0, 0.12, n), 0, 1)
    return y, score


def test_patient_members_lists_every_row_once_per_patient():
    members = patient_members(["a", "b", "a", "c", "b", "a"])
    assert sorted(sorted(rows.tolist()) for rows in members) == [[0, 2, 5], [1, 4], [3]]


def test_intervals_surround_the_estimate_and_repeat_with_the_same_seed():
    y, score = informative_scores()
    patients = np.arange(len(y)) // 2
    result = bootstrap_intervals(y, score, patients, threshold=0.25, n_boot=200, seed=1)
    assert list(result) == BOOTSTRAP_METRICS
    for entry in result.values():
        assert entry["lower"] <= entry["estimate"] <= entry["upper"]
    assert result["ROC-AUC"]["lower"] > 0.5  # the scores carry real signal
    assert result == bootstrap_intervals(y, score, patients, threshold=0.25, n_boot=200, seed=1)


def test_intervals_resample_whole_patients_so_clustered_data_gives_wider_intervals():
    rng = np.random.default_rng(3)
    patients = np.repeat(np.arange(30), 60)  # 30 patients with 60 encounters each
    patient_effect = np.repeat(rng.normal(0, 0.15, 30), 60)  # each patient has their own typical score
    score = np.clip(0.3 + patient_effect + rng.normal(0, 0.05, len(patients)), 0, 1)
    y = (rng.random(len(patients)) < 0.3).astype(int)
    by_patient = bootstrap_intervals(y, score, patients, 0.3, n_boot=200, seed=0)["Flagged@Threshold"]
    by_row = bootstrap_intervals(y, score, np.arange(len(y)), 0.3, n_boot=200, seed=0)["Flagged@Threshold"]
    assert by_patient["upper"] - by_patient["lower"] > 2 * (by_row["upper"] - by_row["lower"])


def test_group_metrics_count_every_encounter_and_hide_rates_for_small_groups():
    y = np.array([1, 0, 0, 1] * 60 + [1, 0, 0])
    score = np.array([0.9, 0.1, 0.4, 0.2] * 60 + [0.9, 0.1, 0.1])
    labels = ["big"] * 240 + ["tiny"] * 3
    rows = {row["group"]: row for row in group_metrics(labels, y, score, threshold=0.3)}
    assert rows["big"]["encounters"] + rows["tiny"]["encounters"] == len(y)
    assert rows["big"]["encounters"] >= MIN_GROUP_ENCOUNTERS
    assert rows["big"]["positives"] == 120 and rows["big"]["base_rate"] == pytest.approx(0.5)
    # flagged: 0.9 and 0.4 of every four; of those 1 of 2 is a true positive; 1 of 2 positives is found
    assert rows["big"]["flagged_share"] == pytest.approx(0.5)
    assert rows["big"]["precision"] == pytest.approx(0.5) and rows["big"]["recall"] == pytest.approx(0.5)
    assert rows["tiny"]["encounters"] == 3
    assert all(rows["tiny"][key] is None for key in ("base_rate", "flagged_share", "precision", "recall", "roc_auc"))


def test_calibration_slices_cover_the_test_set_and_follow_the_scores():
    y, score = informative_scores(1000, seed=2)
    table = calibration_table(y, score, bins=10)
    assert [row["slice"] for row in table["slices"]] == list(range(1, 11))
    assert sum(row["encounters"] for row in table["slices"]) == 1000
    means = [row["mean_score"] for row in table["slices"]]
    assert means == sorted(means)
    assert table["slices"][-1]["observed_rate"] > table["slices"][0]["observed_rate"]
    overall = table["overall"]
    assert overall["brier_score"] == pytest.approx(np.mean((score - y) ** 2))
    assert overall["brier_score_of_always_predicting_the_base_rate"] == pytest.approx(y.mean() * (1 - y.mean()))


def test_analyze_reports_everything_from_a_trained_model(model_dir):
    result = analyze(model_dir.parent, model_dir, n_boot=20)
    assert result["matches_saved_test_metrics"] is None  # this model was trained without scoring the test set
    assert set(result["groups"]) == {"age", "gender", "race"}
    assert "Not recorded" in {row["group"] for row in result["groups"]["race"]}  # the "?" marker is renamed
    for rows in result["groups"].values():
        assert sum(row["encounters"] for row in rows) == result["n_test_encounters"]
    assert result["n_test_patients"] <= result["n_test_encounters"]


def test_analyze_agrees_with_the_metrics_saved_at_training_time(in_tmp_cwd, raw_frame):
    raw_frame.to_csv(in_tmp_cwd / DATA_FILE, index=False)
    train(in_tmp_cwd, "logistic_regression", in_tmp_cwd / "models", evaluate_test=True)
    assert analyze(in_tmp_cwd, in_tmp_cwd / "models", n_boot=10)["matches_saved_test_metrics"] is True


def test_analyze_refuses_data_the_model_was_not_trained_on(in_tmp_cwd, raw_frame):
    raw_frame.to_csv(in_tmp_cwd / DATA_FILE, index=False)
    train(in_tmp_cwd, "logistic_regression", in_tmp_cwd / "models")
    raw_frame.head(700).to_csv(in_tmp_cwd / DATA_FILE, index=False)
    with pytest.raises(ValueError, match="checksum"):
        analyze(in_tmp_cwd, in_tmp_cwd / "models", n_boot=10)
    with pytest.raises(SystemExit, match="checksum"):
        analyze_main(["--data-dir", str(in_tmp_cwd), "--model-dir", str(in_tmp_cwd / "models"), "--n-boot", "10"])


def test_the_command_saves_aggregates_only(in_tmp_cwd, raw_frame, capsys):
    raw_frame.to_csv(in_tmp_cwd / DATA_FILE, index=False)
    train(in_tmp_cwd, "logistic_regression", in_tmp_cwd / "models")
    out = in_tmp_cwd / "reports" / "analysis.json"
    analyze_main(["--data-dir", str(in_tmp_cwd), "--model-dir", str(in_tmp_cwd / "models"), "--out", str(out),
                  "--n-boot", "10"])
    text = out.read_text(encoding="utf-8")
    saved = json.loads(text)
    assert saved["interval"]["resamples"] == 10 and "Calibration" in capsys.readouterr().out
    for forbidden in ("patient_nbr", "encounter_id"):
        assert forbidden not in text  # no identifiers, no rows: only counts and rates
