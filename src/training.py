"""Model comparison, tuning, serialization, and training orchestration."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import RegressorMixin, clone
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.model_selection import KFold, RandomizedSearchCV, cross_validate, train_test_split
from sklearn.tree import DecisionTreeRegressor

from src.config import (
    CATEGORICAL_FEATURES,
    FIGURES_DIR,
    ID_COLUMN,
    METADATA_PATH,
    MODEL_PATH,
    MODEL_RESULTS_PATH,
    NUMERIC_FEATURES,
    RESULTS_DIR,
    TARGET_COLUMN,
    TrainingConfig,
    ensure_project_directories,
)
from src.evaluation import (
    build_error_analysis,
    regression_metrics,
    save_diagnostic_plots,
    save_feature_importance_plot,
)
from src.preprocessing import build_model_pipeline


@dataclass(frozen=True)
class ModelCandidate:
    """Regression estimator metadata used by the comparison workflow."""

    name: str
    estimator: RegressorMixin
    scale_numeric: bool


@dataclass(frozen=True)
class ModelResult:
    """Comparable holdout and cross-validation metrics for one estimator."""

    name: str
    mae: float
    mse: float
    rmse: float
    r2: float
    cv_rmse_mean: float
    cv_rmse_std: float


def get_model_candidates(random_state: int) -> tuple[ModelCandidate, ...]:
    """Return a deliberately diverse set of regularized and tree models."""
    return (
        ModelCandidate("Linear Regression", LinearRegression(), True),
        ModelCandidate("Ridge", Ridge(alpha=20.0), True),
        ModelCandidate("Lasso", Lasso(alpha=250.0, max_iter=20_000), True),
        ModelCandidate(
            "ElasticNet",
            ElasticNet(alpha=0.05, l1_ratio=0.5, max_iter=20_000, random_state=random_state),
            True,
        ),
        ModelCandidate(
            "Decision Tree",
            DecisionTreeRegressor(min_samples_leaf=3, random_state=random_state),
            False,
        ),
        ModelCandidate(
            "Random Forest",
            RandomForestRegressor(
                n_estimators=400,
                min_samples_leaf=2,
                n_jobs=-1,
                random_state=random_state,
            ),
            False,
        ),
        ModelCandidate(
            "Gradient Boosting",
            GradientBoostingRegressor(random_state=random_state),
            False,
        ),
    )


def _evaluate_candidate(
    candidate: ModelCandidate,
    features: pd.DataFrame,
    target: pd.Series,
    config: TrainingConfig,
) -> tuple[ModelResult, Any, pd.DataFrame, pd.Series]:
    """Fit one pipeline and return unbiased holdout plus CV performance."""
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=config.test_size,
        random_state=config.random_state,
    )
    pipeline = build_model_pipeline(candidate.estimator, candidate.scale_numeric)
    cv = KFold(n_splits=config.cv_folds, shuffle=True, random_state=config.random_state)
    cv_scores = cross_validate(
        clone(pipeline),
        x_train,
        y_train,
        cv=cv,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
    )
    fitted_pipeline = pipeline.fit(x_train, y_train)
    predictions = fitted_pipeline.predict(x_test)
    metrics = regression_metrics(y_test, predictions)
    result = ModelResult(
        name=candidate.name,
        **metrics,
        cv_rmse_mean=float(-cv_scores["test_score"].mean()),
        cv_rmse_std=float(cv_scores["test_score"].std()),
    )
    return result, fitted_pipeline, x_test, y_test


def compare_models(
    features: pd.DataFrame,
    target: pd.Series,
    config: TrainingConfig,
) -> tuple[pd.DataFrame, ModelCandidate]:
    """Evaluate required model families and select by cross-validated RMSE."""
    results: list[ModelResult] = []
    candidates = get_model_candidates(config.random_state)
    for candidate in candidates:
        result, _, _, _ = _evaluate_candidate(candidate, features, target, config)
        results.append(result)
    comparison = pd.DataFrame(asdict(result) for result in results).sort_values(
        "cv_rmse_mean", ascending=True
    )
    best_name = str(comparison.iloc[0]["name"])
    best_candidate = next(candidate for candidate in candidates if candidate.name == best_name)
    return comparison.reset_index(drop=True), best_candidate


def _parameter_distributions(model_name: str) -> dict[str, list[Any]]:
    """Supply bounded, model-specific tuning ranges with material impact."""
    distributions: dict[str, dict[str, list[Any]]] = {
        "Linear Regression": {"model__fit_intercept": [True, False]},
        "Ridge": {"model__alpha": [0.1, 1.0, 5.0, 20.0, 50.0, 100.0]},
        "Lasso": {"model__alpha": [10.0, 50.0, 100.0, 250.0, 500.0, 1_000.0]},
        "ElasticNet": {
            "model__alpha": [0.005, 0.01, 0.05, 0.1, 0.5],
            "model__l1_ratio": [0.15, 0.35, 0.5, 0.7, 0.9],
        },
        "Decision Tree": {
            "model__max_depth": [4, 6, 8, 12, None],
            "model__min_samples_leaf": [1, 2, 3, 5, 10],
            "model__min_samples_split": [2, 5, 10, 20],
        },
        "Random Forest": {
            "model__n_estimators": [300, 500, 700],
            "model__max_depth": [None, 10, 20, 30],
            "model__min_samples_leaf": [1, 2, 4],
            "model__max_features": ["sqrt", 0.7, 1.0],
        },
        "Gradient Boosting": {
            "model__n_estimators": [200, 350, 500],
            "model__learning_rate": [0.02, 0.04, 0.06, 0.1],
            "model__max_depth": [2, 3, 4],
            "model__min_samples_leaf": [2, 4, 6],
            "model__subsample": [0.7, 0.85, 1.0],
        },
    }
    return distributions[model_name]


def tune_model(
    candidate: ModelCandidate,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    config: TrainingConfig,
) -> RandomizedSearchCV:
    """Tune only the selected model using CV on training data alone."""
    distributions = _parameter_distributions(candidate.name)
    estimator = build_model_pipeline(candidate.estimator, candidate.scale_numeric)
    combinations = int(np.prod([len(values) for values in distributions.values()]))
    search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=distributions,
        n_iter=min(18, combinations),
        scoring="neg_root_mean_squared_error",
        cv=KFold(n_splits=config.cv_folds, shuffle=True, random_state=config.random_state),
        random_state=config.random_state,
        n_jobs=-1,
        refit=True,
    )
    return search.fit(x_train, y_train)


def _save_feature_importance(final_pipeline: Any) -> None:
    """Persist importances when the selected estimator exposes them."""
    model = final_pipeline.named_steps["model"]
    if not hasattr(model, "feature_importances_"):
        return
    preprocessor = final_pipeline.named_steps["preprocessing"]
    feature_names = preprocessor.get_feature_names_out()
    save_feature_importance_plot(
        feature_names,
        model.feature_importances_,
        FIGURES_DIR / "feature_importance.png",
    )


def train_final_model(data: pd.DataFrame, config: TrainingConfig = TrainingConfig()) -> dict[str, Any]:
    """Run comparison, CV tuning, diagnostics, and model serialization.

    The test set is held out before comparison and is never used in tuning;
    all imputation, encoding, scaling, and feature engineering occur inside
    the pipeline fitted independently within each validation fold.
    """
    ensure_project_directories()
    feature_columns = list(NUMERIC_FEATURES + CATEGORICAL_FEATURES)
    missing_columns = sorted(set(feature_columns + [config.target_column]) - set(data.columns))
    if missing_columns:
        raise ValueError(f"Training data is missing required columns: {missing_columns}")
    clean_data = data.dropna(subset=[config.target_column]).copy()
    features = clean_data.loc[:, feature_columns]
    target = pd.to_numeric(clean_data[config.target_column], errors="raise")
    comparison, selected_candidate = compare_models(features, target, config)
    comparison.to_csv(MODEL_RESULTS_PATH, index=False)

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=config.test_size,
        random_state=config.random_state,
    )
    search = tune_model(selected_candidate, x_train, y_train, config)
    final_pipeline = search.best_estimator_
    predictions = final_pipeline.predict(x_test)
    final_metrics = regression_metrics(y_test, predictions)
    identifiers = clean_data[ID_COLUMN] if ID_COLUMN in clean_data.columns else None
    error_analysis = build_error_analysis(y_test, predictions, identifiers)
    error_analysis.to_csv(RESULTS_DIR / "error_analysis.csv", index=False)
    save_diagnostic_plots(y_test, predictions, FIGURES_DIR)
    _save_feature_importance(final_pipeline)

    joblib.dump(final_pipeline, MODEL_PATH)
    metadata: dict[str, Any] = {
        "model_name": selected_candidate.name,
        "selected_by": "lowest mean 5-fold CV RMSE on training partition",
        "best_cv_rmse": float(-search.best_score_),
        "test_metrics": final_metrics,
        "best_parameters": search.best_params_,
        "raw_feature_columns": feature_columns,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "training_rows": int(len(x_train)),
        "test_rows": int(len(x_test)),
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata
