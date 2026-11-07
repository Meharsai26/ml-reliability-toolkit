from typing import Any

from ml_reliability.core.result import CheckResult


def check_missing_values(X: Any) -> CheckResult:
    """
    Check a dataset for missing values.

    Returns a CheckResult containing:
    - total missing values
    - affected columns
    - missing percentage
    """

    if X is None:
        return CheckResult(
            check_id="data.missing_values",
            title="Missing Values",
            category="data_quality",
            passed=False,
            severity="critical",
            score=0.0,
            message="Input dataset is missing.",
            recommendation="Provide a valid dataset before running reliability checks.",
        )

    # Pandas DataFrame support
    if not hasattr(X, "isna") or not hasattr(X, "columns"):
        return CheckResult(
            check_id="data.missing_values",
            title="Missing Values",
            category="data_quality",
            passed=False,
            severity="critical",
            score=0.0,
            message="Input must be a pandas DataFrame.",
            recommendation="Provide the dataset as a pandas DataFrame.",
        )

    missing_by_column = X.isna().sum()
    missing_by_column = missing_by_column[missing_by_column > 0]

    total_missing = int(missing_by_column.sum())
    total_cells = int(X.shape[0] * X.shape[1])

    if total_cells == 0:
        missing_percentage = 0.0
    else:
        missing_percentage = (total_missing / total_cells) * 100

    if total_missing == 0:
        return CheckResult(
            check_id="data.missing_values",
            title="Missing Values",
            category="data_quality",
            passed=True,
            severity="info",
            score=100.0,
            message="No missing values were detected.",
            evidence={
                "total_missing": 0,
                "missing_percentage": 0.0,
                "affected_columns": {},
            },
        )

    affected_columns = {
        str(column): int(count) for column, count in missing_by_column.items()
    }

    # Initial severity thresholds
    if missing_percentage >= 20:
        severity = "critical"
        score = 0.0
    elif missing_percentage >= 5:
        severity = "warning"
        score = 50.0
    else:
        severity = "info"
        score = 80.0

    return CheckResult(
        check_id="data.missing_values",
        title="Missing Values",
        category="data_quality",
        passed=False,
        severity=severity,
        score=score,
        message=(
            f"Detected {total_missing} missing values "
            f"({missing_percentage:.2f}% of all cells)."
        ),
        evidence={
            "total_missing": total_missing,
            "missing_percentage": round(missing_percentage, 2),
            "affected_columns": affected_columns,
        },
        recommendation=(
            "Investigate affected columns and apply an appropriate "
            "missing-value handling strategy."
        ),
    )
