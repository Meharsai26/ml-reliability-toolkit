from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from ml_reliability.checks.model_performance import check_model_performance

from .base import (
    StressTestOutcome,
    calculate_degradation,
    validate_rate,
)


def inject_categorical_corruption(
    X: pd.DataFrame,
    columns: Sequence[str],
    rate: float,
    random_state: int = 42,
) -> pd.DataFrame:
    """Return a copy of X with categorical values replaced by other valid values."""

    if isinstance(columns, (str, bytes)) or not isinstance(columns, Sequence):
        raise TypeError("columns must be a sequence of column names")

    if any(not isinstance(column, str) for column in columns):
        raise ValueError("columns must contain only column names")

    validate_rate(rate)
    if (
        isinstance(random_state, (bool, np.bool_))
        or not isinstance(random_state, (int, np.integer))
        or random_state < 0
    ):
        raise ValueError("random_state must be a non-negative integer")

    if len(columns) != len(set(columns)):
        raise ValueError("duplicate requested column names are not allowed")

    stressed_X = X.copy()

    if stressed_X.columns.has_duplicates:
        duplicated_columns = (
            stressed_X.columns[stressed_X.columns.duplicated()].unique().tolist()
        )
        requested_duplicates = [
            column for column in columns if column in duplicated_columns
        ]
        if requested_duplicates:
            raise ValueError(
                f"duplicate column names are ambiguous: {requested_duplicates}"
            )

    missing_columns = [column for column in columns if column not in stressed_X.columns]

    if missing_columns:
        raise ValueError(f"Columns not found in X: {missing_columns}")

    if not columns or stressed_X.empty or rate == 0.0:
        return stressed_X

    rng = np.random.default_rng(random_state)

    for column in columns:
        unique_values = stressed_X[column].dropna().unique()

        if len(unique_values) < 2:
            continue

        valid_positions = np.flatnonzero(stressed_X[column].notna().to_numpy())

        values_to_corrupt = int(len(valid_positions) * rate)

        if rate > 0.0 and len(valid_positions) > 0:
            values_to_corrupt = max(1, values_to_corrupt)

        if values_to_corrupt == 0:
            continue

        selected_positions = rng.choice(
            valid_positions,
            size=values_to_corrupt,
            replace=False,
        )

        for position in selected_positions:
            current_value = stressed_X[column].iloc[position]

            alternatives = [value for value in unique_values if value != current_value]

            if not alternatives:
                continue

            alternative_index = rng.integers(len(alternatives))
            stressed_X.iat[position, stressed_X.columns.get_loc(column)] = alternatives[
                alternative_index
            ]

    return stressed_X


def evaluate_categorical_corruption_stress(
    model: Any,
    X_test: pd.DataFrame,
    y_test: Any,
    task: str,
    columns: Sequence[str],
    rate: float,
    random_state: int = 42,
) -> StressTestOutcome:
    """Evaluate model performance after categorical value corruption."""

    stressed_X = inject_categorical_corruption(
        X_test,
        columns=columns,
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
            stress_type="categorical_corruption",
            baseline_metric=baseline_metric,
            stressed_metric=float("nan"),
            degradation=float("inf"),
            passed=False,
            evidence={
                "columns": list(columns),
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
        stress_type="categorical_corruption",
        baseline_metric=baseline_metric,
        stressed_metric=stressed_metric,
        degradation=degradation,
        passed=True,
        evidence={
            "columns": list(columns),
            "rate": rate,
            "random_state": random_state,
            "baseline_metric": baseline_metric,
            "stressed_metric": stressed_metric,
            "degradation": degradation,
        },
    )
