"""Preprocessing: value cleanup, imputation, scaling and one-hot encoding."""

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from readmission import config


def replace_cleanup(df, replacement_values=None):
    """Replace "not measured" markers with an explicit category, then turn "?" into NaN for imputation."""
    replacement_values = config.NOT_MEASURED_REPLACEMENTS if replacement_values is None else replacement_values
    cleaned = df.copy()
    cleaned = cleaned.replace(replacement_values)
    cleaned = cleaned.replace("?", np.nan)
    return cleaned


def build_preprocessor(scale_numeric):
    """ColumnTransformer shared by all models.

    Tree models do not need scaled numerics (scale_numeric=False); logistic regression and the MLP do.
    Rare categories (fewer than 20 occurrences) are pooled, and unseen categories map to that pooled bucket.
    """
    numeric_steps = [("num_imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("stdscaler", StandardScaler()))

    categorical_pipeline = Pipeline([
        ("replace_cleanup", FunctionTransformer(replace_cleanup, validate=False, feature_names_out="one-to-one")),
        ("cat_imputer", SimpleImputer(strategy="constant", fill_value="not_recorded")),
        ("cat_encoder", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=20)),
    ])

    return ColumnTransformer([
        ("num", Pipeline(numeric_steps), config.NUMERIC_COLUMNS),
        ("cat", categorical_pipeline, config.CATEGORICAL_COLUMNS),
    ])
