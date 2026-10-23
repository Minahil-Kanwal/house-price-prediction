"""Model loading and inference helpers shared by the API and Streamlit UI."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.config import MODEL_PATH


class ModelNotAvailableError(RuntimeError):
    """Raised when a prediction is requested before model training completed."""


@lru_cache(maxsize=2)
def load_model(model_path: Path = MODEL_PATH) -> Pipeline:
    """Load and cache the serialized inference pipeline."""
    if not model_path.exists():
        raise ModelNotAvailableError(
            f"No trained model exists at {model_path}. Run `python -m src.cli train --download`."
        )
    try:
        model = joblib.load(model_path)
    except (OSError, ValueError, TypeError) as error:
        raise ModelNotAvailableError("The trained model could not be loaded.") from error
    if not hasattr(model, "predict"):
        raise ModelNotAvailableError("The serialized artifact is not a prediction pipeline.")
    return model


def predict_price(features: pd.DataFrame, model_path: Path = MODEL_PATH) -> float:
    """Produce one non-negative dollar price from a schema-aligned feature frame."""
    model = load_model(model_path)
    try:
        prediction = float(np.asarray(model.predict(features)).reshape(-1)[0])
    except (ValueError, TypeError, KeyError) as error:
        raise ValueError("The provided property attributes could not be scored.") from error
    return max(prediction, 0.0)
