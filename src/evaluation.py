"""Regression metrics, diagnostics, and report visualizations."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(actual: pd.Series, predicted: np.ndarray) -> dict[str, float]:
    """Calculate standard regression metrics on the original dollar scale."""
    mse = float(mean_squared_error(actual, predicted))
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "r2": float(r2_score(actual, predicted)),
    }


def build_error_analysis(
    actual: pd.Series,
    predicted: np.ndarray,
    identifiers: pd.Series | None = None,
) -> pd.DataFrame:
    """Build a row-level residual table, sorted by largest absolute error."""
    residual = actual.to_numpy() - predicted
    denominator = np.maximum(np.abs(actual.to_numpy()), 1.0)
    analysis = pd.DataFrame(
        {
            "actual_price": actual.to_numpy(),
            "predicted_price": predicted,
            "residual": residual,
            "absolute_error": np.abs(residual),
            "absolute_percentage_error": np.abs(residual) / denominator * 100,
        },
        index=actual.index,
    )
    if identifiers is not None:
        analysis.insert(0, "id", identifiers.reindex(actual.index).to_numpy())
    return analysis.sort_values("absolute_error", ascending=False)


def save_diagnostic_plots(
    actual: pd.Series,
    predicted: np.ndarray,
    output_directory: Path,
) -> None:
    """Save actual-vs-predicted and residual plots for model review."""
    output_directory.mkdir(parents=True, exist_ok=True)
    residuals = actual.to_numpy() - predicted
    axis_min = min(float(actual.min()), float(np.min(predicted)))
    axis_max = max(float(actual.max()), float(np.max(predicted)))

    figure, axis = plt.subplots(figsize=(8, 6))
    axis.scatter(actual, predicted, alpha=0.7, edgecolor="none")
    axis.plot([axis_min, axis_max], [axis_min, axis_max], "--", color="#d62728")
    axis.set_title("Actual vs. Predicted Sale Price")
    axis.set_xlabel("Actual sale price ($)")
    axis.set_ylabel("Predicted sale price ($)")
    figure.tight_layout()
    figure.savefig(output_directory / "actual_vs_predicted.png", dpi=160)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8, 6))
    axis.scatter(predicted, residuals, alpha=0.7, edgecolor="none")
    axis.axhline(0, linestyle="--", color="#d62728")
    axis.set_title("Residuals vs. Predicted Sale Price")
    axis.set_xlabel("Predicted sale price ($)")
    axis.set_ylabel("Residual ($)")
    figure.tight_layout()
    figure.savefig(output_directory / "residuals_vs_predicted.png", dpi=160)
    plt.close(figure)


def save_feature_importance_plot(
    feature_names: np.ndarray,
    importances: np.ndarray,
    output_path: Path,
    top_n: int = 20,
) -> None:
    """Save a horizontal plot of the most influential model features."""
    ranking = pd.DataFrame({"feature": feature_names, "importance": importances})
    ranking = ranking.nlargest(top_n, "importance").sort_values("importance")
    figure, axis = plt.subplots(figsize=(9, 7))
    axis.barh(ranking["feature"], ranking["importance"], color="#2463a5")
    axis.set_title(f"Top {len(ranking)} Feature Importances")
    axis.set_xlabel("Importance")
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
