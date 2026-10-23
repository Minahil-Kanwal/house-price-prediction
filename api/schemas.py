"""Pydantic request and response contracts for the prediction API."""

from __future__ import annotations

from typing import Annotated

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field


QualityScore = Annotated[int, Field(ge=1, le=10, description="Overall quality score (1–10).")]


class HouseFeatures(BaseModel):
    """Business-friendly property attributes expected by the trained pipeline."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "OverallQual": 7,
                    "OverallCond": 5,
                    "GrLivArea": 1710,
                    "TotalBsmtSF": 856,
                    "1stFlrSF": 856,
                    "2ndFlrSF": 854,
                    "FullBath": 2,
                    "HalfBath": 1,
                    "BsmtFullBath": 1,
                    "BsmtHalfBath": 0,
                    "BedroomAbvGr": 3,
                    "TotRmsAbvGrd": 8,
                    "YearBuilt": 2003,
                    "YearRemodAdd": 2003,
                    "YrSold": 2008,
                    "GarageCars": 2,
                    "GarageArea": 548,
                    "WoodDeckSF": 0,
                    "OpenPorchSF": 61,
                    "LotArea": 8450,
                    "MSZoning": "RL",
                    "Neighborhood": "CollgCr",
                    "BldgType": "1Fam",
                    "HouseStyle": "2Story",
                    "ExterQual": "Gd",
                    "KitchenQual": "Gd",
                    "HeatingQC": "Ex",
                    "BsmtQual": "Gd",
                    "GarageFinish": "RFn",
                    "SaleCondition": "Normal",
                    "CentralAir": "Y",
                    "LotShape": "Reg",
                }
            ]
        },
    )

    OverallQual: QualityScore
    OverallCond: QualityScore
    GrLivArea: Annotated[float, Field(gt=0, le=20_000)]
    TotalBsmtSF: Annotated[float, Field(ge=0, le=20_000)]
    first_floor_area: Annotated[float, Field(gt=0, le=20_000, alias="1stFlrSF")]
    second_floor_area: Annotated[float, Field(ge=0, le=20_000, alias="2ndFlrSF")]
    FullBath: Annotated[int, Field(ge=0, le=10)]
    HalfBath: Annotated[int, Field(ge=0, le=10)]
    BsmtFullBath: Annotated[int, Field(ge=0, le=10)]
    BsmtHalfBath: Annotated[int, Field(ge=0, le=10)]
    BedroomAbvGr: Annotated[int, Field(ge=0, le=20)]
    TotRmsAbvGrd: Annotated[int, Field(ge=1, le=30)]
    YearBuilt: Annotated[int, Field(ge=1800, le=2100)]
    YearRemodAdd: Annotated[int, Field(ge=1800, le=2100)]
    YrSold: Annotated[int, Field(ge=2006, le=2100)]
    GarageCars: Annotated[int, Field(ge=0, le=8)]
    GarageArea: Annotated[float, Field(ge=0, le=5_000)]
    WoodDeckSF: Annotated[float, Field(ge=0, le=10_000)]
    OpenPorchSF: Annotated[float, Field(ge=0, le=10_000)]
    LotArea: Annotated[float, Field(gt=0, le=500_000)]
    MSZoning: Annotated[str, Field(min_length=1, max_length=20)]
    Neighborhood: Annotated[str, Field(min_length=1, max_length=40)]
    BldgType: Annotated[str, Field(min_length=1, max_length=20)]
    HouseStyle: Annotated[str, Field(min_length=1, max_length=20)]
    ExterQual: Annotated[str, Field(min_length=1, max_length=10)]
    KitchenQual: Annotated[str, Field(min_length=1, max_length=10)]
    HeatingQC: Annotated[str, Field(min_length=1, max_length=10)]
    BsmtQual: Annotated[str, Field(min_length=1, max_length=10)]
    GarageFinish: Annotated[str, Field(min_length=1, max_length=10)]
    SaleCondition: Annotated[str, Field(min_length=1, max_length=20)]
    CentralAir: Annotated[str, Field(pattern="^[YN]$")]
    LotShape: Annotated[str, Field(min_length=1, max_length=10)]

    def to_frame(self) -> pd.DataFrame:
        """Convert the validated request into the one-row model input frame."""
        return pd.DataFrame([self.model_dump(by_alias=True)])


class PredictionResponse(BaseModel):
    """Successful API response."""

    predicted_price: float = Field(ge=0, description="Predicted sale price in USD.")
    currency: str = "USD"
    model_version: str


class HealthResponse(BaseModel):
    """Lightweight service readiness response."""

    status: str
    model_available: bool


class ErrorResponse(BaseModel):
    """Safe public error shape for predictable service failures."""

    detail: str
