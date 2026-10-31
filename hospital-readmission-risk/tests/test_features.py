import numpy as np
import pandas as pd

from readmission import config
from readmission.features import build_preprocessor, replace_cleanup


def test_replace_cleanup_marks_not_measured_and_missing():
    df = pd.DataFrame({
        "max_glu_serum": ["None", "Norm"],
        "A1Cresult": ["None", ">7"],
        "race": ["?", "Caucasian"],
        "metformin": ["None", "No"],  # "None" outside the two lab columns is left alone
    })
    cleaned = replace_cleanup(df)
    assert cleaned["max_glu_serum"].tolist() == ["not_measured", "Norm"]
    assert cleaned["A1Cresult"].tolist() == ["not_measured", ">7"]
    assert np.isnan(cleaned.loc[0, "race"])
    assert cleaned.loc[0, "metformin"] == "None"
    assert df.loc[0, "race"] == "?"  # input is not modified


def test_preprocessor_output_is_numeric_without_nans(frame):
    X = frame[config.FEATURE_COLUMNS]
    out = build_preprocessor(scale_numeric=True).fit_transform(X)
    dense = out.toarray() if hasattr(out, "toarray") else out
    assert dense.shape[0] == len(X)
    assert dense.shape[1] > len(config.NUMERIC_COLUMNS)
    assert np.isfinite(dense).all()


def test_scaling_is_optional(frame):
    X = frame[config.FEATURE_COLUMNS]
    scaled = build_preprocessor(scale_numeric=True).fit(X).transform(X)
    unscaled = build_preprocessor(scale_numeric=False).fit(X).transform(X)
    n_num = len(config.NUMERIC_COLUMNS)
    first_scaled = scaled[:, :n_num].toarray() if hasattr(scaled, "toarray") else scaled[:, :n_num]
    first_unscaled = unscaled[:, :n_num].toarray() if hasattr(unscaled, "toarray") else unscaled[:, :n_num]
    assert abs(first_scaled.mean()) < 1e-6
    assert first_unscaled.mean() > 1


def test_unseen_and_missing_values_do_not_break_transform(frame):
    X = frame[config.FEATURE_COLUMNS]
    preprocessor = build_preprocessor(scale_numeric=False).fit(X)
    row = X.iloc[[0]].copy()
    row["race"] = "SomethingNew"
    row["diag_1"] = "?"
    row["num_lab_procedures"] = np.nan
    out = preprocessor.transform(row)
    assert out.shape[0] == 1
