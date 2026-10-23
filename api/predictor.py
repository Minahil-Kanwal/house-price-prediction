"""Thin API-facing adapter around the shared inference module."""

from __future__ import annotations

import json
from pathlib import Path

from api.config import API_VERSION
from api.schemas import HouseFeatures
from src.config import METADATA_PATH, MODEL_PATH
from src.prediction import predict_price


def model_is_available(model_path: Path = MODEL_PATH) -> bool:
    """Return whether an inference artifact is present without loading it."""
    return model_path.is_file()


def prediction_model_version(metadata_path: Path = METADATA_PATH) -> str:
    """Read the model identity from metadata, with a safe API-version fallback."""
    if not metadata_path.is_file():
        return API_VERSION
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return API_VERSION
    return str(metadata.get("trained_at_utc", API_VERSION))


def predict_house_price(features: HouseFeatures) -> float:
    """Score a validated property through the serialized sklearn pipeline."""
    return predict_price(features.to_frame())
