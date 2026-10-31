import json

import joblib
import numpy as np
import pytest

from readmission import config
from readmission.data import DATA_FILE, split_xy
from readmission.models import DISPLAY_NAMES, build_model
from readmission.train import METADATA_FILE, MODEL_FILE, main


@pytest.mark.parametrize("name", list(DISPLAY_NAMES))
def test_every_model_fits_and_returns_probabilities(name, frame):
    X, y, _ = split_xy(frame)
    model = build_model(name).fit(X, y)
    proba = model.predict_proba(X)
    assert proba.shape == (len(X), 2)
    assert np.all((proba >= 0) & (proba <= 1))


def test_unknown_model_name_is_rejected():
    with pytest.raises(ValueError, match="Unknown model"):
        build_model("not_a_model")


def test_train_cli_saves_a_model_that_scores_new_encounters(tmp_path, raw_frame, capsys):
    raw_frame.to_csv(tmp_path / DATA_FILE, index=False)
    out_dir = tmp_path / "models"
    main(["--data-dir", str(tmp_path), "--model", "logistic_regression", "--out-dir", str(out_dir)])

    metadata = json.loads((out_dir / METADATA_FILE).read_text(encoding="utf-8"))
    assert 0 < metadata["threshold"] < 1
    assert metadata["test_metrics"] is None  # the test set is untouched unless explicitly requested
    assert metadata["categorical_columns"] == config.CATEGORICAL_COLUMNS
    assert len(metadata["data_file_sha256"]) == 64

    model = joblib.load(out_dir / MODEL_FILE)
    one_encounter = raw_frame.rename(columns={"citoglipton": "sitagliptin"})[config.FEATURE_COLUMNS].head(1)
    assert 0 <= model.predict_proba(one_encounter)[0, 1] <= 1


def test_train_cli_scores_test_set_only_when_asked(tmp_path, raw_frame):
    raw_frame.to_csv(tmp_path / DATA_FILE, index=False)
    out_dir = tmp_path / "models"
    main(["--data-dir", str(tmp_path), "--model", "logistic_regression", "--out-dir", str(out_dir),
          "--evaluate-test"])
    metadata = json.loads((out_dir / METADATA_FILE).read_text(encoding="utf-8"))
    assert "PR-AUC" in metadata["test_metrics"]
