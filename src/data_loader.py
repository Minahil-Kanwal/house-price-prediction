"""Data acquisition and loading utilities for the Ames Housing dataset."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.datasets import fetch_openml

from src.config import OPENML_DATA_ID, RAW_DATA_PATH, TARGET_COLUMN, ensure_project_directories


def download_ames_housing(destination: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Download the canonical Ames Housing data from OpenML and cache it as CSV.

    OpenML data ID 42165 mirrors the Kaggle House Prices training schema and
    avoids committing a third-party dataset to the repository.
    """
    ensure_project_directories()
    dataset = fetch_openml(data_id=OPENML_DATA_ID, as_frame=True, parser="auto")
    features = dataset.data.copy()
    target = pd.to_numeric(dataset.target, errors="raise").rename(TARGET_COLUMN)
    data = pd.concat([features, target], axis=1)
    data.to_csv(destination, index=False)
    return data


def load_housing_data(data_path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Load a cached housing CSV and validate the target column is present."""
    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {data_path}. Run `python -m src.cli download` "
            "or `python -m src.cli train --download` first."
        )

    data = pd.read_csv(data_path)
    if TARGET_COLUMN not in data.columns:
        raise ValueError(f"Expected target column '{TARGET_COLUMN}' in {data_path}.")
    return data
