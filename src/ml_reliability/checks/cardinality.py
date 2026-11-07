from typing import Any

import pandas as pd

from ml_reliability.core.result import CheckResult


def check_cardinality(X: Any) -> CheckResult:
    """
    Detect high-cardinality and potentially identifier-like columns.
    """

    if X is None:
        return CheckResult(
            check_id="data.cardinality",
            title="Cardinality / ID Detection",
            category="data_quality",
            passed=False,
            severity="critical",
            score=0.0,
            message="Input data is missing.",
            recommendation="Provide X_train before running cardinality checks.",
        )

    if not isinstance(X, pd.DataFrame):
        return CheckResult(
            check_id="data.cardinality",
            title="Cardinality / ID Detection",
            category="data_quality",
            passed=False,
            severity="critical",
            score=0.0,
            message="Input must be a pandas DataFrame.",
            recommendation="Provide tabular data as a pandas DataFrame.",
        )

    if X.empty:
        return CheckResult(
            check_id="data.cardinality",
            title="Cardinality / ID Detection",
            category="data_quality",
            passed=False,
            severity="critical",
            score=0.0,
            message="Input DataFrame contains zero rows.",
            evidence={
                "rows": 0,
                "columns": int(X.shape[1]),
            },
            recommendation="Provide a DataFrame containing at least one row.",
        )

    if X.shape[1] == 0:
        return CheckResult(
            check_id="data.cardinality",
            title="Cardinality / ID Detection",
            category="data_quality",
            passed=False,
            severity="critical",
            score=0.0,
            message="Input DataFrame contains zero columns.",
            evidence={
                "rows": int(X.shape[0]),
                "columns": 0,
            },
            recommendation="Provide a DataFrame containing at least one feature column.",
        )

    column_analysis = {}
    high_cardinality_columns = []
    identifier_like_columns = []
    # Cardinality signals are unreliable on very small datasets.
    # We still report the evidence, but avoid penalizing the dataset.
    small_dataset = len(X) < 10

    for column in X.columns:
        total_values = int(X[column].notna().sum())
        unique_values = int(X[column].nunique(dropna=True))

        if total_values == 0:
            uniqueness_percentage = 0.0
        else:
            uniqueness_percentage = (unique_values / total_values) * 100

        uniqueness_percentage = round(uniqueness_percentage, 2)

        column_analysis[str(column)] = {
            "total_values": total_values,
            "unique_values": unique_values,
            "uniqueness_percentage": uniqueness_percentage,
        }

        if not small_dataset:
            if uniqueness_percentage >= 80:
                high_cardinality_columns.append(str(column))

            if uniqueness_percentage > 95:
                identifier_like_columns.append(str(column))

    if small_dataset:
        severity = "info"
        score = 100.0
        passed = True
        message = (
            "Dataset is too small for reliable cardinality-based identifier detection."
        )
    elif identifier_like_columns:
        severity = "warning"
        score = 50.0
        passed = True
        message = (
            f"Detected {len(identifier_like_columns)} potentially "
            "identifier-like column(s)."
        )
    elif high_cardinality_columns:
        severity = "warning"
        score = 70.0
        passed = True
        message = (
            f"Detected {len(high_cardinality_columns)} high-cardinality column(s)."
        )
    else:
        severity = "info"
        score = 100.0
        passed = True
        message = "No high-cardinality columns were detected."

    return CheckResult(
        check_id="data.cardinality",
        title="Cardinality / ID Detection",
        category="data_quality",
        passed=passed,
        severity=severity,
        score=score,
        message=message,
        evidence={
            "total_columns": int(X.shape[1]),
            "high_cardinality_columns": high_cardinality_columns,
            "identifier_like_columns": identifier_like_columns,
            "columns": column_analysis,
            "small_dataset": small_dataset,
            "minimum_rows_for_detection": 10,
        },
        recommendation=(
            "Review identifier-like columns and determine whether they should "
            "be excluded from model features."
            if identifier_like_columns
            else (
                "Review high-cardinality columns for potential identifiers "
                "or leakage before model training."
                if high_cardinality_columns
                else ""
            )
        ),
    )
