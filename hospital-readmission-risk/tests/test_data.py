import pandas as pd

from readmission import config
from readmission.data import (
    DATA_FILE,
    filter_eligible,
    group_train_test_split,
    load_raw,
    make_target,
    prepare_datasets,
    split_xy,
)


def test_load_raw_keeps_markers_and_corrects_column_name(tmp_path, raw_frame):
    raw_frame.to_csv(tmp_path / DATA_FILE, index=False)
    df = load_raw(tmp_path / DATA_FILE)
    assert "sitagliptin" in df.columns and "citoglipton" not in df.columns
    assert (df["race"] == "?").any()  # "?" is kept, cleaned later in the pipeline
    assert (df["max_glu_serum"] == "None").any()  # the string "None" is not parsed as missing
    assert not df.isna().any().any()


def test_filter_eligible_drops_expired_and_hospice(frame):
    frame.loc[:9, "discharge_disposition_id"] = config.EXCLUDED_DISCHARGE_DISPOSITION_IDS + [11, 13, 14, 19]
    filtered = filter_eligible(frame)
    assert len(filtered) == len(frame) - 10
    assert not filtered["discharge_disposition_id"].isin(config.EXCLUDED_DISCHARGE_DISPOSITION_IDS).any()


def test_group_split_never_shares_a_patient(frame):
    train, test = group_train_test_split(frame)
    assert set(train["patient_nbr"]).isdisjoint(test["patient_nbr"])
    assert len(train) + len(test) == len(frame)
    assert 0.1 < len(test) / len(frame) < 0.3


def test_group_split_is_deterministic(frame):
    first, _ = group_train_test_split(frame)
    second, _ = group_train_test_split(frame)
    pd.testing.assert_frame_equal(first, second)


def test_make_target_only_marks_under_30_days():
    df = pd.DataFrame({"readmitted": ["<30", ">30", "NO", "<30"]})
    assert make_target(df).tolist() == [1, 0, 0, 1]


def test_split_xy_returns_model_columns_only(frame):
    X, y, groups = split_xy(frame)
    assert list(X.columns) == config.FEATURE_COLUMNS
    assert set(y.unique()) <= {0, 1}
    assert groups.equals(frame["patient_nbr"])


def test_prepare_datasets_end_to_end(tmp_path, raw_frame):
    raw_frame.to_csv(tmp_path / DATA_FILE, index=False)
    train, test = prepare_datasets(tmp_path / DATA_FILE)
    assert set(train["patient_nbr"]).isdisjoint(test["patient_nbr"])
