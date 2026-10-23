"""Tests for validated API input conversion."""

from __future__ import annotations

from api.schemas import HouseFeatures


def valid_payload() -> dict[str, object]:
    """Return a schema-complete canonical API request."""
    return {
        "OverallQual": 7, "OverallCond": 5, "GrLivArea": 1710, "TotalBsmtSF": 856,
        "1stFlrSF": 856, "2ndFlrSF": 854, "FullBath": 2, "HalfBath": 1,
        "BsmtFullBath": 1, "BsmtHalfBath": 0, "BedroomAbvGr": 3, "TotRmsAbvGrd": 8,
        "YearBuilt": 2003, "YearRemodAdd": 2003, "YrSold": 2008, "GarageCars": 2,
        "GarageArea": 548, "WoodDeckSF": 0, "OpenPorchSF": 61, "LotArea": 8450,
        "MSZoning": "RL", "Neighborhood": "CollgCr", "BldgType": "1Fam",
        "HouseStyle": "2Story", "ExterQual": "Gd", "KitchenQual": "Gd",
        "HeatingQC": "Ex", "BsmtQual": "Gd", "GarageFinish": "RFn",
        "SaleCondition": "Normal", "CentralAir": "Y", "LotShape": "Reg",
    }


def test_schema_preserves_model_column_aliases() -> None:
    """Pydantic aliases must recreate the exact DataFrame column contract."""
    frame = HouseFeatures(**valid_payload()).to_frame()

    assert "1stFlrSF" in frame.columns
    assert "2ndFlrSF" in frame.columns
    assert "first_floor_area" not in frame.columns
    assert frame.loc[0, "GrLivArea"] == 1710
