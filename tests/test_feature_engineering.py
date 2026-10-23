"""Tests for deterministic business-feature generation."""

from __future__ import annotations

import pandas as pd
import pytest

from src.feature_engineering import HouseFeatureEngineer


def test_feature_engineer_creates_expected_values() -> None:
    """Derived property totals and flags should follow their documented formulas."""
    frame = pd.DataFrame(
        [
            {
                "TotalBsmtSF": 500,
                "1stFlrSF": 750,
                "2ndFlrSF": 250,
                "FullBath": 2,
                "HalfBath": 1,
                "BsmtFullBath": 1,
                "BsmtHalfBath": 0,
                "WoodDeckSF": 100,
                "OpenPorchSF": 50,
                "YearBuilt": 2000,
                "YearRemodAdd": 2005,
                "YrSold": 2010,
                "GarageCars": 2,
                "GarageArea": 450,
            }
        ]
    )

    transformed = HouseFeatureEngineer().fit_transform(frame)

    assert transformed.loc[0, "TotalSF"] == 1500
    assert transformed.loc[0, "TotalBath"] == 3.5
    assert transformed.loc[0, "TotalPorchSF"] == 150
    assert transformed.loc[0, "HouseAgeAtSale"] == 10
    assert transformed.loc[0, "RemodelAgeAtSale"] == 5
    assert transformed.loc[0, "HasGarage"] == 1
    assert transformed.loc[0, "HasBasement"] == 1


def test_feature_engineer_rejects_incomplete_contract() -> None:
    """A clear error should identify feature-engineering contract violations."""
    with pytest.raises(ValueError, match="Missing columns"):
        HouseFeatureEngineer().fit(pd.DataFrame({"TotalBsmtSF": [100]}))
