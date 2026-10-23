"""Central configuration and immutable project paths."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final


PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
DATA_DIR: Final[Path] = PROJECT_ROOT / "data"
RAW_DATA_DIR: Final[Path] = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Final[Path] = DATA_DIR / "processed"
MODELS_DIR: Final[Path] = PROJECT_ROOT / "models"
REPORTS_DIR: Final[Path] = PROJECT_ROOT / "reports"
FIGURES_DIR: Final[Path] = REPORTS_DIR / "figures"
RESULTS_DIR: Final[Path] = REPORTS_DIR / "results"

RAW_DATA_PATH: Final[Path] = RAW_DATA_DIR / "ames_housing.csv"
MODEL_PATH: Final[Path] = MODELS_DIR / "house_price_pipeline.joblib"
METADATA_PATH: Final[Path] = MODELS_DIR / "model_metadata.json"
MODEL_RESULTS_PATH: Final[Path] = RESULTS_DIR / "model_comparison.csv"
VALIDATION_REPORT_PATH: Final[Path] = RESULTS_DIR / "data_validation.json"

TARGET_COLUMN: Final[str] = "SalePrice"
ID_COLUMN: Final[str] = "Id"
RANDOM_STATE: Final[int] = 42
TEST_SIZE: Final[float] = 0.20
CV_FOLDS: Final[int] = 5
OPENML_DATA_ID: Final[int] = 42165

# The API deliberately uses a compact, business-meaningful input contract rather
# than asking users to provide every column from the competition dataset.
NUMERIC_FEATURES: Final[tuple[str, ...]] = (
    "OverallQual",
    "OverallCond",
    "GrLivArea",
    "TotalBsmtSF",
    "1stFlrSF",
    "2ndFlrSF",
    "FullBath",
    "HalfBath",
    "BsmtFullBath",
    "BsmtHalfBath",
    "BedroomAbvGr",
    "TotRmsAbvGrd",
    "YearBuilt",
    "YearRemodAdd",
    "YrSold",
    "GarageCars",
    "GarageArea",
    "WoodDeckSF",
    "OpenPorchSF",
    "LotArea",
)
CATEGORICAL_FEATURES: Final[tuple[str, ...]] = (
    "MSZoning",
    "Neighborhood",
    "BldgType",
    "HouseStyle",
    "ExterQual",
    "KitchenQual",
    "HeatingQC",
    "BsmtQual",
    "GarageFinish",
    "SaleCondition",
    "CentralAir",
    "LotShape",
)
ENGINEERED_NUMERIC_FEATURES: Final[tuple[str, ...]] = (
    "TotalSF",
    "TotalBath",
    "TotalPorchSF",
    "HouseAgeAtSale",
    "RemodelAgeAtSale",
    "HasGarage",
    "HasBasement",
)


@dataclass(frozen=True)
class TrainingConfig:
    """Parameters shared by the deterministic training workflow."""

    random_state: int = RANDOM_STATE
    test_size: float = TEST_SIZE
    cv_folds: int = CV_FOLDS
    target_column: str = TARGET_COLUMN


def ensure_project_directories() -> None:
    """Create directories that hold generated data, models, and reports."""
    for directory in (
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        MODELS_DIR,
        FIGURES_DIR,
        RESULTS_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)
