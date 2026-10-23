"""HTTP route definitions, deliberately separated from app construction."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from api.config import API_TITLE, API_VERSION
from api.predictor import model_is_available, predict_house_price, prediction_model_version
from api.schemas import ErrorResponse, HealthResponse, HouseFeatures, PredictionResponse
from src.prediction import ModelNotAvailableError


router = APIRouter()


@router.get("/", summary="Service information")
def root() -> dict[str, str]:
    """Describe the service and provide a stable health-check link."""
    return {
        "service": API_TITLE,
        "version": API_VERSION,
        "documentation": "/docs",
        "health": "/health",
    }


@router.get("/health", response_model=HealthResponse, summary="Service health")
def health() -> HealthResponse:
    """Return API readiness without forcing a costly model deserialization."""
    available = model_is_available()
    return HealthResponse(status="ok" if available else "degraded", model_available=available)


@router.post(
    "/predict",
    response_model=PredictionResponse,
    responses={
        503: {"model": ErrorResponse, "description": "Model artifact is unavailable."},
        422: {"description": "Validation failed."},
    },
    summary="Predict a house sale price",
    description=(
        "Scores one property using the versioned, serialized training pipeline. "
        "Unknown categorical values are handled safely by the fitted encoder."
    ),
)
def predict(features: HouseFeatures) -> PredictionResponse:
    """Validate and score a property request without exposing internal errors."""
    try:
        price = predict_house_price(features)
    except ModelNotAvailableError as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Prediction model is unavailable. Train the model before serving requests.",
        ) from error
    except ValueError as error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="The property attributes could not be evaluated.",
        ) from error
    return PredictionResponse(
        predicted_price=round(price, 2),
        model_version=prediction_model_version(),
    )
