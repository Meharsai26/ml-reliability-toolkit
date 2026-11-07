import math
from typing import Any

import numpy as np
import pandas as pd

from ml_reliability.checks.model_performance import check_model_performance

from .base import StressTestOutcome, calculate_degradation


def perturb_feature_distribution(
    X: pd.DataFrame,
    shift: float = 0.0,
    scale: float = 1.0,
    random_state: int = 42,
) -> pd.DataFrame:
    """Return a copy of X with controlled numeric distribution perturbation."""

    if (
        isinstance(random_state, (bool, np.bool_))
        or not isinstance(random_state, (int, np.integer))
        or random_state < 0
    ):
        raise ValueError("random_state must be a non-negative integer")

    for name, value in (("shift", shift), ("scale", scale)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, float, np.integer, np.floating)
        ):
            raise TypeError(f"{name} must be a numeric value")

        try:
            is_finite = math.isfinite(value)
        except OverflowError:
            is_finite = False

        if not is_finite:
            if name == "scale":
                raise ValueError("scale must be a finite number greater than 0")
            raise ValueError("shift must be a finite number")

    if scale <= 0:
        raise ValueError("scale must be a finite number greater than 0")
    stressed_X = X.copy()

    if stressed_X.empty:
        return stressed_X

    numeric_columns = stressed_X.select_dtypes(include=np.number).columns

    if len(numeric_columns) == 0:
        return stressed_X

    with np.errstate(over="ignore", invalid="ignore"):
        for column in numeric_columns:
            stressed_X[column] = (stressed_X[column] * scale) + shift

    original_values = X[numeric_columns].to_numpy(dtype=float)
    stressed_values = stressed_X[numeric_columns].to_numpy(dtype=float)

    original_missing = np.isnan(original_values)
    stressed_missing = np.isnan(stressed_values)

    new_non_finite = ~np.isfinite(stressed_values) & ~original_missing

    if new_non_finite.any():
        raise ValueError("distribution perturbation produced non-finite values")

    if np.any(stressed_missing & ~original_missing):
        raise ValueError("distribution perturbation produced non-finite values")

    return stressed_X


def evaluate_distribution_perturbation_stress(
    model: Any,
    X_test: pd.DataFrame,
    y_test: Any,
    task: str,
    shift: float = 0.0,
    scale: float = 1.0,
    random_state: int = 42,
) -> StressTestOutcome:
    """Evaluate model performance after distribution perturbation."""

    stressed_X = perturb_feature_distribution(
        X_test,
        shift=shift,
        scale=scale,
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
            "Baseline model performance could not be calculated: "
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
            stress_type="distribution_perturbation",
            baseline_metric=baseline_metric,
            stressed_metric=float("nan"),
            degradation=float("inf"),
            passed=False,
            evidence={
                "shift": shift,
                "scale": scale,
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
        stress_type="distribution_perturbation",
        baseline_metric=baseline_metric,
        stressed_metric=stressed_metric,
        degradation=degradation,
        passed=True,
        evidence={
            "shift": shift,
            "scale": scale,
            "random_state": random_state,
            "baseline_metric": baseline_metric,
            "stressed_metric": stressed_metric,
            "degradation": degradation,
        },
    )
