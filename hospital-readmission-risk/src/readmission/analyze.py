"""Describe how the saved model behaves on the held-out test set: uncertainty, groups and calibration.

    python -m readmission.analyze --data-dir data --model-dir models --out reports/test_analysis.json

This is descriptive only. The model, its settings and its cutoff are already fixed, and nothing here is used to
change them, so the held-out test numbers stay an honest estimate. Three things are reported:

- 95% intervals for the main test metrics, from resampling whole patients (a patient can have several
  encounters, so resampling encounters one by one would make the intervals too narrow);
- the same metrics for each age band, gender and race, so differences between groups are visible;
- a calibration table: in each tenth of the test set, ordered by score, the average score next to the share
  of encounters that were really readmitted within 30 days.

Only counts and rates are written, never a row of the data.
"""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from readmission import __version__, config
from readmission.data import DATA_FILE, prepare_datasets, split_xy
from readmission.evaluation import evaluate_scores
from readmission.train import METADATA_FILE, MODEL_FILE, file_sha256

INTERVAL_LEVEL = 0.95
BOOTSTRAP_METRICS = ["ROC-AUC", "PR-AUC", "Flagged@Threshold", "Precision@Threshold", "Recall@Threshold",
                     "Lift@Top10%"]
GROUP_COLUMNS = ["age", "gender", "race"]
MIN_GROUP_ENCOUNTERS = 100  # smaller groups are listed with their counts only
MIN_GROUP_POSITIVES = 20
NOT_RECORDED = "Not recorded"


def patient_members(groups):
    """The row positions of each patient, as a list of arrays."""
    codes, _ = pd.factorize(np.asarray(groups))
    order = np.argsort(codes, kind="stable")
    return np.split(order, np.flatnonzero(np.diff(codes[order])) + 1)


def bootstrap_intervals(y_true, y_score, groups, threshold, n_boot=1000, seed=config.RANDOM_STATE,
                        level=INTERVAL_LEVEL):
    """Point estimate and percentile interval for each metric, resampling whole patients with replacement."""
    y_true, y_score = np.asarray(y_true), np.asarray(y_score)
    members = patient_members(groups)
    rng = np.random.default_rng(seed)
    draws = {name: [] for name in BOOTSTRAP_METRICS}
    for _ in range(n_boot):
        picked = rng.integers(0, len(members), len(members))
        rows = np.concatenate([members[i] for i in picked])
        if y_true[rows].min() == y_true[rows].max():
            continue  # a sample with one outcome only has no ROC-AUC; with real data this does not happen
        metrics = evaluate_scores(y_true[rows], y_score[rows], threshold)
        for name in draws:
            draws[name].append(metrics[name])
    point = evaluate_scores(y_true, y_score, threshold)
    tail = (1 - level) / 2 * 100
    return {name: {"estimate": float(point[name]),
                   "lower": float(np.percentile(values, tail)),
                   "upper": float(np.percentile(values, 100 - tail))}
            for name, values in draws.items()}


def group_metrics(labels, y_true, y_score, threshold):
    """One row per value of `labels` (a column of the test set). Small groups get counts only."""
    y_true, y_score = np.asarray(y_true), np.asarray(y_score)
    flagged = y_score >= threshold
    labels = pd.Series(np.asarray(labels)).astype(str)
    rows = []
    for value, index in labels.groupby(labels).indices.items():
        y, hit = y_true[index], flagged[index]
        row = {"group": value, "encounters": int(len(index)), "positives": int(y.sum())}
        enough = len(index) >= MIN_GROUP_ENCOUNTERS and y.sum() >= MIN_GROUP_POSITIVES
        row.update(
            base_rate=float(y.mean()) if enough else None,
            flagged_share=float(hit.mean()) if enough else None,
            precision=float(y[hit].mean()) if enough and hit.any() else None,
            recall=float(hit[y == 1].mean()) if enough else None,
            roc_auc=float(roc_auc_score(y, y_score[index])) if enough and y.min() != y.max() else None)
        rows.append(row)
    return rows


def calibration_table(y_true, y_score, bins=10):
    """Average score against the observed readmission rate in equal-sized slices of the test set, lowest first."""
    y_true, y_score = np.asarray(y_true), np.asarray(y_score)
    order = np.argsort(y_score, kind="stable")
    slices = [{"slice": number, "encounters": int(len(rows)),
               "mean_score": float(y_score[rows].mean()), "observed_rate": float(y_true[rows].mean())}
              for number, rows in enumerate(np.array_split(order, bins), start=1)]
    overall = {"mean_score": float(y_score.mean()), "observed_rate": float(y_true.mean()),
               "brier_score": float(np.mean((y_score - y_true) ** 2)),
               "brier_score_of_always_predicting_the_base_rate": float(np.var(y_true))}
    return {"slices": slices, "overall": overall}


def analyze(data_dir, model_dir, n_boot=1000, seed=config.RANDOM_STATE):
    model_dir = Path(model_dir)
    metadata = json.loads((model_dir / METADATA_FILE).read_text(encoding="utf-8"))
    data_path = Path(data_dir) / DATA_FILE
    if file_sha256(data_path) != metadata["data_file_sha256"]:
        raise ValueError("the data file is not the one the model was trained on (its checksum differs)")
    model = joblib.load(model_dir / MODEL_FILE)
    threshold = metadata["threshold"]

    _, test_df = prepare_datasets(data_path)
    X_test, y_test, patients = split_xy(test_df)
    score = model.predict_proba(X_test)[:, 1]

    saved = metadata.get("test_metrics")
    fresh = evaluate_scores(y_test, score, threshold)
    matches_saved = None if saved is None else all(abs(fresh[key] - value) < 1e-9 for key, value in saved.items())

    groups = {}
    for column in GROUP_COLUMNS:
        labels = test_df[column].replace({"?": NOT_RECORDED})
        groups[column] = group_metrics(labels, y_test, score, threshold)

    return {
        "created_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "package_version": __version__,
        "model_name": metadata["model_name"],
        "threshold": threshold,
        "n_test_encounters": int(len(test_df)),
        "n_test_patients": int(patients.nunique()),
        "matches_saved_test_metrics": matches_saved,
        "interval": {"level": INTERVAL_LEVEL, "method": "bootstrap over patients", "resamples": n_boot,
                     "seed": seed},
        "metrics": bootstrap_intervals(y_test, score, patients, threshold, n_boot=n_boot, seed=seed),
        "groups": groups,
        "calibration": calibration_table(y_test, score),
        "min_group": {"encounters": MIN_GROUP_ENCOUNTERS, "positives": MIN_GROUP_POSITIVES},
    }


def percent(value):
    return "     -" if value is None else f"{value:6.1%}"


def print_summary(result):
    print(f"{result['model_name']}: {result['n_test_encounters']} test encounters from "
          f"{result['n_test_patients']} patients, cutoff {result['threshold']:.3f}")
    print(f"matches the test metrics saved with the model: {result['matches_saved_test_metrics']}")
    print(f"\n{result['interval']['level']:.0%} intervals ({result['interval']['resamples']} resamples of patients):")
    for name, entry in result["metrics"].items():
        print(f"  {name:20s} {entry['estimate']:.3f}   [{entry['lower']:.3f}, {entry['upper']:.3f}]")
    for column, rows in result["groups"].items():
        print(f"\nBy {column}:      encounters  positives  base rate  flagged  precision  recall  ROC-AUC")
        for row in rows:
            auc = "     -" if row["roc_auc"] is None else f"{row['roc_auc']:6.3f}"
            print(f"  {row['group']:18s} {row['encounters']:8d} {row['positives']:9d} {percent(row['base_rate'])} "
                  f"{percent(row['flagged_share'])} {percent(row['precision'])}   {percent(row['recall'])} {auc}")
    calibration = result["calibration"]
    print("\nCalibration (tenths of the test set, lowest score first):  mean score   observed rate")
    for row in calibration["slices"]:
        print(f"  {row['slice']:2d}  {row['encounters']:5d} encounters   {row['mean_score']:10.3f}   "
              f"{row['observed_rate']:10.3f}")
    overall = calibration["overall"]
    print(f"  overall: mean score {overall['mean_score']:.3f}, observed rate {overall['observed_rate']:.3f}, "
          f"Brier score {overall['brier_score']:.3f} "
          f"(always predicting the base rate: {overall['brier_score_of_always_predicting_the_base_rate']:.3f})")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default="data", help="folder containing diabetic_data.csv")
    parser.add_argument("--model-dir", default="models", help="folder holding the trained model")
    parser.add_argument("--out", default="reports/test_analysis.json")
    parser.add_argument("--n-boot", type=int, default=1000, help="number of patient resamples for the intervals")
    parser.add_argument("--seed", type=int, default=config.RANDOM_STATE)
    args = parser.parse_args(argv)

    try:
        result = analyze(args.data_dir, args.model_dir, args.n_boot, args.seed)
    except ValueError as problem:
        raise SystemExit(f"error: {problem}") from None
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print_summary(result)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
