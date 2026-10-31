import json

import joblib
import numpy as np
import pytest

from readmission import config
from readmission.compare import main as compare_main
from readmission.data import DATA_FILE, split_xy
from readmission.models import DEFAULT_PARAMS, DISPLAY_NAMES, build_model
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


def test_train_cli_saves_a_model_that_scores_new_encounters(in_tmp_cwd, raw_frame):
    tmp_path = in_tmp_cwd
    raw_frame.to_csv(tmp_path / DATA_FILE, index=False)
    out_dir = tmp_path / "models"
    main(["--data-dir", str(tmp_path), "--model", "logistic_regression", "--out-dir", str(out_dir)])

    metadata = json.loads((out_dir / METADATA_FILE).read_text(encoding="utf-8"))
    assert 0 < metadata["threshold"] < 1
    assert metadata["test_metrics"] is None  # the test set is untouched unless explicitly requested
    assert metadata["categorical_columns"] == config.CATEGORICAL_COLUMNS
    assert len(metadata["data_file_sha256"]) == 64
    assert metadata["hyperparameters"] == DEFAULT_PARAMS["logistic_regression"]
    assert metadata["hyperparameters_source"] == "built-in defaults"

    model = joblib.load(out_dir / MODEL_FILE)
    one_encounter = raw_frame.rename(columns={"citoglipton": "sitagliptin"})[config.FEATURE_COLUMNS].head(1)
    assert 0 <= model.predict_proba(one_encounter)[0, 1] <= 1


def test_train_cli_scores_test_set_only_when_asked(in_tmp_cwd, raw_frame):
    tmp_path = in_tmp_cwd
    raw_frame.to_csv(tmp_path / DATA_FILE, index=False)
    out_dir = tmp_path / "models"
    main(["--data-dir", str(tmp_path), "--model", "logistic_regression", "--out-dir", str(out_dir),
          "--evaluate-test"])
    metadata = json.loads((out_dir / METADATA_FILE).read_text(encoding="utf-8"))
    assert "PR-AUC" in metadata["test_metrics"]


def test_train_cli_uses_and_records_tuned_parameters(in_tmp_cwd, raw_frame, write_params_file):
    raw_frame.to_csv(in_tmp_cwd / DATA_FILE, index=False)
    params_path = in_tmp_cwd / "best.json"
    write_params_file(params_path, {"decision_tree": {"max_depth": 3}})
    out_dir = in_tmp_cwd / "models"
    main(["--data-dir", str(in_tmp_cwd), "--model", "decision_tree", "--out-dir", str(out_dir),
          "--params", str(params_path)])

    metadata = json.loads((out_dir / METADATA_FILE).read_text(encoding="utf-8"))
    assert metadata["hyperparameters"] == {"max_depth": 3, "min_samples_leaf": 50}
    assert metadata["hyperparameters_source"] == str(params_path)
    assert joblib.load(out_dir / MODEL_FILE).named_steps["model"].max_depth == 3


def test_compare_with_models_prints_but_does_not_save_by_default(in_tmp_cwd, raw_frame, capsys):
    raw_frame.to_csv(in_tmp_cwd / DATA_FILE, index=False)
    compare_main(["--data-dir", str(in_tmp_cwd), "--models", "decision_tree", "logistic_regression"])

    printed = capsys.readouterr().out
    assert "Decision Tree" in printed and "Logistic Regression" in printed and "XGBoost" not in printed
    assert not (in_tmp_cwd / "reports").exists()  # a partial table must not replace the full one


def test_compare_saves_a_partial_table_only_when_given_an_output_path(in_tmp_cwd, raw_frame):
    raw_frame.to_csv(in_tmp_cwd / DATA_FILE, index=False)
    out = in_tmp_cwd / "partial.csv"
    compare_main(["--data-dir", str(in_tmp_cwd), "--models", "decision_tree", "--out", str(out)])
    assert out.read_text(encoding="utf-8").count("\n") == 2  # header + one model
