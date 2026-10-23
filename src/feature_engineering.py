"""Domain-driven feature engineering compatible with scikit-learn pipelines."""

from __future__ import annotations

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class HouseFeatureEngineer(BaseEstimator, TransformerMixin):
    """Create interpretable area, bathroom, amenity, and age features.

    The transformer is fitted inside each pipeline fold, preserving the same
    training/inference logic and preventing preprocessing leakage.
    """

    required_columns = (
        "TotalBsmtSF",
        "1stFlrSF",
        "2ndFlrSF",
        "FullBath",
        "HalfBath",
        "BsmtFullBath",
        "BsmtHalfBath",
        "WoodDeckSF",
        "OpenPorchSF",
        "YearBuilt",
        "YearRemodAdd",
        "YrSold",
        "GarageCars",
        "GarageArea",
    )

    def fit(self, features: pd.DataFrame, target: pd.Series | None = None) -> "HouseFeatureEngineer":
        """Validate the expected input schema; no learned state is needed."""
        self._validate_input(features)
        return self

    def transform(self, features: pd.DataFrame) -> pd.DataFrame:
        """Return a copy with compact, business-interpretable derived features."""
        self._validate_input(features)
        transformed = features.copy()
        numeric = transformed.loc[:, self.required_columns].apply(
            pd.to_numeric, errors="coerce"
        )
        transformed["TotalSF"] = (
            numeric["TotalBsmtSF"].fillna(0)
            + numeric["1stFlrSF"].fillna(0)
            + numeric["2ndFlrSF"].fillna(0)
        )
        transformed["TotalBath"] = (
            numeric["FullBath"].fillna(0)
            + 0.5 * numeric["HalfBath"].fillna(0)
            + numeric["BsmtFullBath"].fillna(0)
            + 0.5 * numeric["BsmtHalfBath"].fillna(0)
        )
        transformed["TotalPorchSF"] = (
            numeric["WoodDeckSF"].fillna(0) + numeric["OpenPorchSF"].fillna(0)
        )
        transformed["HouseAgeAtSale"] = (
            numeric["YrSold"] - numeric["YearBuilt"]
        ).clip(lower=0)
        transformed["RemodelAgeAtSale"] = (
            numeric["YrSold"] - numeric["YearRemodAdd"]
        ).clip(lower=0)
        transformed["HasGarage"] = (
            (numeric["GarageCars"].fillna(0) > 0)
            | (numeric["GarageArea"].fillna(0) > 0)
        ).astype(int)
        transformed["HasBasement"] = (numeric["TotalBsmtSF"].fillna(0) > 0).astype(int)
        return transformed

    @classmethod
    def _validate_input(cls, features: pd.DataFrame) -> None:
        """Raise a clear error when the inference contract is incomplete."""
        missing = sorted(set(cls.required_columns) - set(features.columns))
        if missing:
            raise ValueError(f"Missing columns required for feature engineering: {missing}")
