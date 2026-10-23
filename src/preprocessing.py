"""Leakage-safe preprocessing and model-pipeline construction."""

from __future__ import annotations

from typing import Final

from sklearn.base import RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler

from src.config import CATEGORICAL_FEATURES, ENGINEERED_NUMERIC_FEATURES, NUMERIC_FEATURES
from src.feature_engineering import HouseFeatureEngineer


ALL_NUMERIC_FEATURES: Final[tuple[str, ...]] = NUMERIC_FEATURES + ENGINEERED_NUMERIC_FEATURES


def build_preprocessor(scale_numeric: bool) -> ColumnTransformer:
    """Build a fitted-in-pipeline transformer for the curated model features.

    Linear models receive robust scaling because coefficients are sensitive to
    feature magnitude and outliers. Tree-based models receive imputation only,
    as split-based learners do not benefit from scaling.
    """
    numeric_steps: list[tuple[str, object]] = [
        ("imputer", SimpleImputer(strategy="median")),
    ]
    if scale_numeric:
        numeric_steps.append(("scaler", RobustScaler()))

    numeric_pipeline = Pipeline(steps=numeric_steps)
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, list(ALL_NUMERIC_FEATURES)),
            ("categorical", categorical_pipeline, list(CATEGORICAL_FEATURES)),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_model_pipeline(
    estimator: RegressorMixin,
    scale_numeric: bool,
) -> Pipeline:
    """Build the complete inference pipeline around a regression estimator."""
    return Pipeline(
        steps=[
            ("feature_engineering", HouseFeatureEngineer()),
            ("preprocessing", build_preprocessor(scale_numeric=scale_numeric)),
            ("model", estimator),
        ]
    )
