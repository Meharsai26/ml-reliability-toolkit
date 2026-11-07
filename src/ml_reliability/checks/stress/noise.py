from typing import Any

import numpy as np
import pandas as pd

from ml_reliability.checks.model_performance import check_model_performance

from .base import (
    StressTestOutcome,
    calculate_degradation,
    validate_rate,
)


def inject_feature_noise(
    X: pd.DataFrame,
    noise_level: float,
    random_state: int = 42,
) -> pd.DataFrame:
    """Return a copy of X with controlled Gaussian noise added to numeric features."""

    validate_rate(noise_level)

    if (
        isinstance(random_state, (bool, np.bool_))
        or not isinstance(random_state, (int, np.integer))
        or random_state < 0
    ):
        raise ValueError("random_state must be a non-negative integer")

    stressed_X = X.copy()

    if stressed_X.columns.has_duplicates:
        raise ValueError("duplicate column names are ambiguous for feature noise")

    numeric_columns = stressed_X.select_dtypes(include=np.number).columns

    if len(numeric_columns) > 0:
        numeric_values = stressed_X[numeric_columns].to_numpy(dtype=float)

        invalid_values = np.isinf(numeric_values)

        if invalid_values.any():
            raise ValueError("feature noise produced non-finite values")

    if stressed_X.empty or noise_level == 0.0:
        return stressed_X

    numeric_columns = stressed_X.select_dtypes(include=np.number).columns

    if len(numeric_columns) == 0:
        return stressed_X

    rng = np.random.default_rng(random_state)

    for column in numeric_columns:
        if stressed_X[column].isna().all():
            continue

        with np.errstate(over="ignore", invalid="ignore"):
            feature_std = stressed_X[column].std()

        if not np.isfinite(feature_std):
            raise ValueError("feature noise produced non-finite values")

        noise_scale = feature_std * noise_level

        if not np.isfinite(noise_scale):
            raise ValueError("feature noise produced non-finite values")

        noise = rng.normal(
            loc=0.0,
            scale=noise_scale,
            size=len(stressed_X),
        )

        stressed_X[column] = stressed_X[column] + noise

        stressed_values = stressed_X[column].to_numpy(dtype=float)

        if np.isinf(stressed_values).any():
            raise ValueError("feature noise produced non-finite values")

    return stressed_X


def evaluate_noise_stress(
    model: Any,
    X_test: pd.DataFrame,
    y_test: Any,
    task: str,
    noise_level: float,
    random_state: int = 42,
) -> StressTestOutcome:
    """Evaluate model performance after adding noise to numeric features."""

    stressed_X = inject_feature_noise(
        X_test,
        noise_level=noise_level,
        random_state=random_state,
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
            stress_type="feature_noise",
            baseline_metric=baseline_metric,
            stressed_metric=float("nan"),
            degradation=float("inf"),
            passed=False,
            evidence={
                "noise_level": noise_level,
                "random_state": random_state,
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
        stress_type="feature_noise",
        baseline_metric=baseline_metric,
        stressed_metric=stressed_metric,
        degradation=degradation,
        passed=True,
        evidence={
            "noise_level": noise_level,
            "random_state": random_state,
            "baseline_metric": baseline_metric,
            "stressed_metric": stressed_metric,
            "degradation": degradation,
        },
    )
