"""Synthetic encounter-level data with the same schema as diabetic_data.csv (the real file is not in CI)."""

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


@pytest.fixture
def frame():
    return make_frame()


@pytest.fixture
def raw_frame():
    return make_frame(raw=True)
