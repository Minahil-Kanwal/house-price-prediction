"""Explicit data-quality checks used before any model is trained."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd
from pandas.api.types import is_numeric_dtype

from src.config import ID_COLUMN, TARGET_COLUMN


@dataclass(frozen=True)
class DataValidationReport:
    """Serializable summary of dataset quality checks."""

    row_count: int
    column_count: int
    duplicate_rows: int
    duplicate_ids: int
    missing_value_percentages: dict[str, float]
    empty_string_counts: dict[str, int]
    constant_columns: list[str]
    near_constant_columns: list[str]
    sparse_columns: list[str]
    invalid_numeric_counts: dict[str, int]
    target_missing_count: int

    def as_dict(self) -> dict[str, Any]:
        """Return the report as a JSON-safe mapping."""
        return asdict(self)


def _near_constant_columns(data: pd.DataFrame, threshold: float = 0.995) -> list[str]:
    """Find columns whose modal value dominates the observed values."""
    columns: list[str] = []
    for column in data.columns:
        observed = data[column].dropna()
        if not observed.empty and observed.value_counts(normalize=True).iloc[0] >= threshold:
            columns.append(column)
    return columns


def _invalid_numeric_counts(data: pd.DataFrame) -> dict[str, int]:
    """Count numeric values that violate basic domain constraints."""
    invalid: dict[str, int] = {}
    non_negative_tokens = ("SF", "Area", "Bath", "Cars", "Porch", "Bedroom", "Rms")
    for column in data.select_dtypes(include="number").columns:
        invalid_count = 0
        if any(token in column for token in non_negative_tokens):
            invalid_count += int((data[column] < 0).sum())
        if column.startswith("Year") or column == "YrSold":
            invalid_count += int(((data[column] < 1800) | (data[column] > 2100)).sum())
        if invalid_count:
            invalid[column] = invalid_count
    return invalid


def validate_housing_data(data: pd.DataFrame) -> DataValidationReport:
    """Profile schema, completeness, duplication, sparsity, and basic validity."""
    missing = (data.isna().mean() * 100).round(2)
    missing_percentages = {
        column: float(value) for column, value in missing.items() if value > 0
    }
    empty_strings = {
        column: int((data[column].astype("string").str.strip() == "").sum())
        for column in data.columns
        if not is_numeric_dtype(data[column])
        and int((data[column].astype("string").str.strip() == "").sum()) > 0
    }
    constant = [column for column in data.columns if data[column].nunique(dropna=False) <= 1]
    sparse = [column for column, value in missing.items() if value >= 80.0]
    duplicate_ids = 0
    if ID_COLUMN in data.columns:
        duplicate_ids = int(data[ID_COLUMN].duplicated().sum())

    return DataValidationReport(
        row_count=len(data),
        column_count=len(data.columns),
        duplicate_rows=int(data.duplicated().sum()),
        duplicate_ids=duplicate_ids,
        missing_value_percentages=missing_percentages,
        empty_string_counts=empty_strings,
        constant_columns=constant,
        near_constant_columns=_near_constant_columns(data),
        sparse_columns=sparse,
        invalid_numeric_counts=_invalid_numeric_counts(data),
        target_missing_count=int(data[TARGET_COLUMN].isna().sum()),
    )
