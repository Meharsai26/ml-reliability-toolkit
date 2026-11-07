from typing import Any

from ml_reliability.core.result import CheckResult


def check_duplicates(X: Any) -> CheckResult:
    """
    Check a dataset for duplicate rows.

    Duplicate rows are counted after the first occurrence.
    """

    if X is None:
        return CheckResult(
            check_id="data.duplicates",
            title="Duplicate Rows",
            category="data_quality",
            passed=False,
            severity="critical",
            score=0.0,
            message="Input dataset is missing.",
            recommendation="Provide a valid dataset before running reliability checks.",
        )

    if not hasattr(X, "duplicated") or not hasattr(X, "shape"):
        return CheckResult(
            check_id="data.duplicates",
            title="Duplicate Rows",
            category="data_quality",
            passed=False,
            severity="critical",
            score=0.0,
            message="Input must be a pandas DataFrame.",
            recommendation="Provide the dataset as a pandas DataFrame.",
        )

    total_rows = int(X.shape[0])

    if total_rows == 0:
        return CheckResult(
            check_id="data.duplicates",
            title="Duplicate Rows",
            category="data_quality",
            passed=True,
            severity="info",
            score=100.0,
            message="Dataset is empty; no duplicate rows were detected.",
            evidence={
                "total_rows": 0,
                "duplicate_rows": 0,
                "duplicate_percentage": 0.0,
            },
        )

    duplicate_rows = int(X.duplicated().sum())
    duplicate_percentage = (duplicate_rows / total_rows) * 100

    if duplicate_rows == 0:
        return CheckResult(
            check_id="data.duplicates",
            title="Duplicate Rows",
            category="data_quality",
            passed=True,
            severity="info",
            score=100.0,
            message="No duplicate rows were detected.",
            evidence={
                "total_rows": total_rows,
                "duplicate_rows": 0,
                "duplicate_percentage": 0.0,
            },
        )

    if duplicate_percentage > 20:
        severity = "critical"
        score = 0.0
    elif duplicate_percentage >= 5:
        severity = "warning"
        score = 50.0
    else:
        severity = "info"
        score = 80.0

    return CheckResult(
        check_id="data.duplicates",
        title="Duplicate Rows",
        category="data_quality",
        passed=False,
        severity=severity,
        score=score,
        message=(
            f"Detected {duplicate_rows} duplicate rows "
            f"({duplicate_percentage:.2f}% of all rows)."
        ),
        evidence={
            "total_rows": total_rows,
            "duplicate_rows": duplicate_rows,
            "duplicate_percentage": round(duplicate_percentage, 2),
        },
        recommendation=(
            "Investigate duplicate records and remove or handle them "
            "appropriately before model training."
        ),
    )
