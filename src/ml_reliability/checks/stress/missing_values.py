from typing import Any

import numpy as np
import pandas as pd

from ml_reliability.checks.model_performance import check_model_performance

from .base import (
    StressTestOutcome,
    calculate_degradation,
    validate_rate,
)


def inject_missing_values(
    X: pd.DataFrame,
    rate: float,
    random_state: int = 42,
) -> pd.DataFrame:
    """Return a copy of X with a controlled proportion of eligible values replaced by NaN."""

    if (
        isinstance(random_state, bool)
        or not isinstance(random_state, (int, np.integer))
        or random_state < 0
    ):
        raise ValueError("random_state must be a non-negative integer")

    validate_rate(rate)

    stressed_X = X.copy()

    if X.empty or rate == 0.0:
        return stressed_X

    # Sample only existing non-missing cells so the requested corruption
    # count represents newly introduced missing values.
    valid_mask = ~stressed_X.isna().to_numpy()
    valid_positions = np.flatnonzero(valid_mask.ravel())

    values_to_corrupt = min(
        len(valid_positions),
        max(1, int(len(valid_positions) * rate)),
    )

    if values_to_corrupt == 0:
        return stressed_X

    random_generator = np.random.default_rng(random_state)

    selected_positions = random_generator.choice(
        valid_positions,
        size=values_to_corrupt,
        replace=False,
    )

    rows, columns = np.unravel_index(
        selected_positions,
        stressed_X.shape,
    )

    for row, column in zip(rows, columns):
        stressed_X.iat[row, column] = float("nan")

    return stressed_X


def evaluate_missing_value_stress(
    model: Any,
    X_test: pd.DataFrame,
    y_test: Any,
    task: str,
    rate: float,
    random_state: int = 42,
) -> StressTestOutcome:
    """Evaluate model performance after injecting missing values."""

    stressed_X = inject_missing_values(
        X_test,
        rate=rate,
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
            stress_type="missing_values",
            baseline_metric=baseline_metric,
            stressed_metric=float("nan"),
            degradation=float("inf"),
            passed=False,
            evidence={
                "rate": rate,
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
        stress_type="missing_values",
        baseline_metric=baseline_metric,
        stressed_metric=stressed_metric,
        degradation=degradation,
        passed=True,
        evidence={
            "rate": rate,
            "random_state": random_state,
            "baseline_metric": baseline_metric,
            "stressed_metric": stressed_metric,
            "degradation": degradation,
        },
    )
