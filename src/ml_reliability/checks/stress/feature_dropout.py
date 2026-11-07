from collections.abc import Sequence
from typing import Any

import pandas as pd

from ml_reliability.checks.model_performance import check_model_performance

from .base import StressTestOutcome, calculate_degradation


def apply_feature_dropout(X: pd.DataFrame, features: Sequence[str]) -> pd.DataFrame:
    """Return a copy of X with selected features masked as missing."""
    if isinstance(features, (str, bytes)) or not isinstance(features, Sequence):
        raise TypeError("features must be a sequence of column names")

    if any(not isinstance(feature, str) for feature in features):
        raise ValueError("features must contain only column names")

    stressed_X = X.copy()

    if stressed_X.columns.has_duplicates:
        duplicated_columns = (
            stressed_X.columns[stressed_X.columns.duplicated()].unique().tolist()
        )

        requested_duplicates = [
            feature for feature in features if feature in duplicated_columns
        ]

        if requested_duplicates:
            raise ValueError(
                f"duplicate column names are ambiguous: {requested_duplicates}"
            )

    if not features:
        return stressed_X

    missing_features = [
        feature for feature in features if feature not in stressed_X.columns
    ]
    if missing_features:
        raise ValueError(f"Features not found in X: {missing_features}")

    for feature in features:
        stressed_X[feature] = stressed_X[feature].mask(
            pd.Series(True, index=stressed_X.index)
        )

    return stressed_X


def evaluate_feature_dropout_stress(
    model: Any,
    X_test: pd.DataFrame,
    y_test: Any,
    task: str,
    features: Sequence[str],
) -> StressTestOutcome:
    """Evaluate model performance after dropping selected features."""

    stressed_X = apply_feature_dropout(
        X_test,
        features=features,
    )

    baseline_result = check_model_performance(
        model=model,
        X_test=X_test,
        y_test=y_test,
        task=task,
    )

    if not baseline_result.passed:
        raise ValueError(
            f"Baseline model performance could not be calculated: "
            f"{baseline_result.message}"
        )

    stressed_result = check_model_performance(
        model=model,
        X_test=stressed_X,
        y_test=y_test,
        task=task,
    )

    baseline_metric = float(baseline_result.evidence["metric_value"])

    if not stressed_result.passed:
        return StressTestOutcome(
            stress_type="feature_dropout",
            baseline_metric=baseline_metric,
            stressed_metric=float("nan"),
            degradation=float("inf"),
            passed=False,
            evidence={
                "features": list(features),
                "baseline_metric": baseline_metric,
                "stress_prediction_failed": True,
                "error": stressed_result.message,
            },
        )

    stressed_metric = float(stressed_result.evidence["metric_value"])

    degradation = calculate_degradation(
        baseline_metric,
        stressed_metric,
    )

    return StressTestOutcome(
        stress_type="feature_dropout",
        baseline_metric=baseline_metric,
        stressed_metric=stressed_metric,
        degradation=degradation,
        passed=True,
        evidence={
            "features": list(features),
            "baseline_metric": baseline_metric,
            "stressed_metric": stressed_metric,
            "degradation": degradation,
        },
    )
