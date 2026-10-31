"""Loading, cohort filtering, target construction and the patient-grouped train/test split."""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from readmission import config

DATA_FILE = "diabetic_data.csv"


def load_raw(path):
    """Read diabetic_data.csv. "?" is kept as-is (cleaned in the pipeline) and "None" stays a string."""
    df = pd.read_csv(Path(path), keep_default_na=False, na_values=[""])
    return df.rename(columns=config.COLUMN_RENAMES)


def filter_eligible(df):
    """Drop encounters that ended in death or hospice discharge."""
    excluded = df["discharge_disposition_id"].isin(config.EXCLUDED_DISCHARGE_DISPOSITION_IDS)
    return df.loc[~excluded].reset_index(drop=True)


def group_train_test_split(df, group_col=config.GROUP_COLUMN, test_size=config.TEST_SIZE,
                           random_state=config.RANDOM_STATE):
    """Split so that every encounter of a given patient lands in exactly one of train/test."""
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(splitter.split(df, groups=df[group_col]))
    return df.iloc[train_idx].reset_index(drop=True), df.iloc[test_idx].reset_index(drop=True)


def make_target(df):
    """1 if the encounter was readmitted in under 30 days, else 0."""
    return (df[config.SOURCE_TARGET_COLUMN] == config.POSITIVE_LABEL).astype(int)


def split_xy(df):
    """Return (features, target, patient groups) for a prepared encounter-level frame."""
    return df[config.FEATURE_COLUMNS], make_target(df), df[config.GROUP_COLUMN]


def prepare_datasets(path):
    """Full data preparation: load, filter, grouped split. Returns (train_df, test_df)."""
    return group_train_test_split(filter_eligible(load_raw(path)))
