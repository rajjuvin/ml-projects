"""Synthetic encounter-level data with the same schema as diabetic_data.csv (the real file is not in CI)."""

import json

import numpy as np
import pandas as pd
import pytest

from readmission import config

CATEGORY_CHOICES = {
    "race": ["Caucasian", "AfricanAmerican", "?"],
    "gender": ["Male", "Female"],
    "age": ["[40-50)", "[50-60)", "[60-70)", "[70-80)"],
    "weight": ["?", "?", "?", "[75-100)"],
    "admission_type_id": [1, 2, 3],
    "discharge_disposition_id": [1, 3, 6],
    "admission_source_id": [1, 4, 7],
    "payer_code": ["MC", "BC", "?"],
    "medical_specialty": ["?", "Cardiology", "InternalMedicine"],
    "diag_1": ["250", "428", "414", "?"],
    "diag_2": ["250", "401", "?"],
    "diag_3": ["250", "276", "?"],
    "max_glu_serum": ["None", "None", "Norm", ">200"],
    "A1Cresult": ["None", "None", ">7", "Norm"],
    "change": ["No", "Ch"],
    "diabetesMed": ["Yes", "No"],
}
MEDICATION_CHOICES = ["No", "No", "Steady", "Up", "Down"]


def make_frame(n=800, seed=0, n_patients=500, raw=False):
    """Random encounters. P(readmitted < 30 days) rises with number_inpatient so models have signal."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "encounter_id": np.arange(n),
        "patient_nbr": rng.integers(0, n_patients, n),
    })
    for col in config.NUMERIC_COLUMNS:
        df[col] = rng.integers(0, 12, n)
    for col in config.CATEGORICAL_COLUMNS:
        choices = CATEGORY_CHOICES.get(col, MEDICATION_CHOICES)
        df[col] = rng.choice(choices, n)
    probability = 1 / (1 + np.exp(-(-2.5 + 0.35 * df["number_inpatient"])))
    outcome = rng.random(n) < probability
    df["readmitted"] = np.where(outcome, "<30", rng.choice(["NO", ">30"], n))
    if raw:
        df = df.rename(columns={new: old for old, new in config.COLUMN_RENAMES.items()})
    return df


@pytest.fixture(scope="session")
def model_dir(tmp_path_factory):
    """A small logistic-regression model trained on synthetic data, saved the way readmission.train saves one."""
    from readmission.data import DATA_FILE
    from readmission.train import train

    root = tmp_path_factory.mktemp("trained")
    make_frame(raw=True).to_csv(root / DATA_FILE, index=False)
    with pytest.MonkeyPatch.context() as patch:
        patch.chdir(root)  # so a reports/best_params.json in the repo is not picked up
        train(root, "logistic_regression", root / "models")
    return root / "models"


@pytest.fixture
def write_params_file():
    """Function writing a best_params.json that holds {model_key: {parameter: value}}, as readmission.tune does."""
    def write(path, models):
        entries = {key: {"params": params} for key, params in models.items()}
        path.write_text(json.dumps({"schema_version": 1, "models": entries}), encoding="utf-8")
    return write


@pytest.fixture
def in_tmp_cwd(tmp_path, monkeypatch):
    """Run in an empty folder, so the default reports/best_params.json is never picked up from the repo."""
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.fixture
def frame():
    return make_frame()


@pytest.fixture
def raw_frame():
    return make_frame(raw=True)
