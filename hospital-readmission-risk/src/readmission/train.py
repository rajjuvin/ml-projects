"""Train the final model, pick its decision threshold from out-of-fold predictions and save the artifact.

    python -m readmission.train --data-dir data --model xgboost --out-dir models

The held-out test set is only scored when --evaluate-test is passed. Do that once, after the model and
threshold are fixed, so the reported test numbers stay an unbiased estimate.
"""

import argparse
import hashlib
import json
import platform
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost

from readmission import __version__, config
from readmission.data import DATA_FILE, prepare_datasets, split_xy
from readmission.evaluation import evaluate_scores, oof_probabilities, select_f1_threshold
from readmission.models import DISPLAY_NAMES, build_model

MODEL_FILE = "readmission_model.joblib"
METADATA_FILE = "metadata.json"


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _plain(metrics):
    return {key: float(value) for key, value in metrics.items()}


def train(data_dir, model_key, out_dir, evaluate_test=False):
    data_path = Path(data_dir) / DATA_FILE
    train_df, test_df = prepare_datasets(data_path)
    X_train, y_train, groups = split_xy(train_df)

    # Out-of-fold scores (folds run in parallel, one thread per model) -> decision threshold.
    oof = oof_probabilities(build_model(model_key, n_jobs=1), X_train, y_train, groups)
    threshold = select_f1_threshold(y_train, oof)
    oof_metrics = evaluate_scores(y_train, oof, threshold)

    # Final fit on the full training split.
    model = build_model(model_key, n_jobs=-1).fit(X_train, y_train)

    test_metrics = None
    if evaluate_test:
        X_test, y_test, _ = split_xy(test_df)
        test_metrics = evaluate_scores(y_test, model.predict_proba(X_test)[:, 1], threshold)

    metadata = {
        "model_key": model_key,
        "model_name": DISPLAY_NAMES[model_key],
        "package_version": __version__,
        "created_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "threshold": threshold,
        "threshold_rule": "maximizes F1 on out-of-fold training predictions",
        "positive_label": config.POSITIVE_LABEL,
        "data_file_sha256": file_sha256(data_path),
        "n_train_encounters": int(len(train_df)),
        "n_test_encounters": int(len(test_df)),
        "train_positive_rate": float(y_train.mean()),
        "numeric_columns": config.NUMERIC_COLUMNS,
        "categorical_columns": config.CATEGORICAL_COLUMNS,
        "oof_metrics": _plain(oof_metrics),
        "test_metrics": _plain(test_metrics) if test_metrics else None,
        "libraries": {
            "python": platform.python_version(),
            "scikit-learn": sklearn.__version__,
            "xgboost": xgboost.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
    }

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_dir / MODEL_FILE)
    (out_dir / METADATA_FILE).write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default="data", help="folder containing diabetic_data.csv")
    parser.add_argument("--model", default="xgboost", choices=list(DISPLAY_NAMES))
    parser.add_argument("--out-dir", default="models")
    parser.add_argument("--evaluate-test", action="store_true",
                        help="also score the held-out test set (run once, after model selection)")
    args = parser.parse_args(argv)

    metadata = train(args.data_dir, args.model, args.out_dir, args.evaluate_test)
    print(f"Saved {metadata['model_name']} to {Path(args.out_dir) / MODEL_FILE} "
          f"(threshold {metadata['threshold']:.3f})")
    for label, key in (("Out-of-fold", "oof_metrics"), ("Test", "test_metrics")):
        if metadata[key]:
            print(f"\n{label} metrics:")
            for name, value in metadata[key].items():
                print(f"  {name:22s} {value:.4f}")


if __name__ == "__main__":
    main()
