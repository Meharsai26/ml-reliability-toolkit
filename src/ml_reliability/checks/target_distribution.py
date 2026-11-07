from typing import Any

import pandas as pd

from ml_reliability.core.result import CheckResult


def _infer_task_type(y: Any) -> str:
    """Infer whether the target is classification or regression."""

    series = pd.Series(y)

    if not pd.api.types.is_numeric_dtype(series):
        return "classification"

    unique_values = series.nunique(dropna=True)

    if unique_values <= 10:
        return "classification"

    return "regression"


def check_target_distribution(
    y: Any,
    task_type: str | None = None,
) -> CheckResult:
    """
    Check the distribution of a machine-learning target.

    Supports classification and regression targets.
    """

    if y is None:
        return CheckResult(
            check_id="target.distribution",
            title="Target Distribution",
            category="target_quality",
            passed=False,
            severity="critical",
            score=0.0,
            message="Target data is missing.",
            recommendation="Provide y_train before running target distribution checks.",
        )

    try:
        series = pd.Series(y)
    except Exception:  # noqa: BLE001 - Handle invalid target conversion.
        return CheckResult(
            check_id="target.distribution",
            title="Target Distribution",
            category="target_quality",
            passed=False,
            severity="critical",
            score=0.0,
            message="Target could not be converted into a valid series.",
            recommendation="Provide a valid one-dimensional target.",
        )

    if series.empty:
        return CheckResult(
            check_id="target.distribution",
            title="Target Distribution",
            category="target_quality",
            passed=False,
            severity="critical",
            score=0.0,
            message="Target contains zero samples.",
            evidence={"total_samples": 0},
            recommendation="Provide a target containing at least one sample.",
        )

    if series.isna().any():
        missing_count = int(series.isna().sum())

        return CheckResult(
            check_id="target.distribution",
            title="Target Distribution",
            category="target_quality",
            passed=False,
            severity="critical",
            score=0.0,
            message=f"Target contains {missing_count} missing values.",
            evidence={
                "total_samples": len(series),
                "missing_values": missing_count,
            },
            recommendation="Remove or appropriately handle missing target values.",
        )

    if task_type is None:
        task_type = _infer_task_type(series)

    if task_type not in {"classification", "regression"}:
        return CheckResult(
            check_id="target.distribution",
            title="Target Distribution",
            category="target_quality",
            passed=False,
            severity="critical",
            score=0.0,
            message=f"Unsupported task type: {task_type}.",
            recommendation="Use 'classification' or 'regression'.",
        )

    if task_type == "classification":
        counts = series.value_counts()
        percentages = (counts / len(series)) * 100

        class_counts = {str(label): int(count) for label, count in counts.items()}

        class_percentages = {
            str(label): round(float(percentage), 2)
            for label, percentage in percentages.items()
        }

        minority_percentage = float(percentages.min())

        if minority_percentage >= 20:
            severity = "info"
            score = 100.0
            passed = True
        elif minority_percentage >= 5:
            severity = "warning"
            score = 70.0
            passed = True
        else:
            severity = "critical"
            score = 30.0
            passed = False

        return CheckResult(
            check_id="target.distribution",
            title="Target Distribution",
            category="target_quality",
            passed=passed,
            severity=severity,
            score=score,
            message=(
                f"Classification target contains {len(counts)} classes. "
                f"Minority class represents {minority_percentage:.2f}% of samples."
            ),
            evidence={
                "task_type": "classification",
                "total_samples": len(series),
                "class_counts": class_counts,
                "class_percentages": class_percentages,
                "minority_percentage": round(minority_percentage, 2),
            },
            recommendation=(
                "Consider class-balancing techniques or appropriate evaluation "
                "metrics if the target distribution is highly imbalanced."
                if not passed
                else ""
            ),
        )

    # Regression
    numeric_series = pd.to_numeric(series, errors="coerce")

    if numeric_series.isna().any():
        return CheckResult(
            check_id="target.distribution",
            title="Target Distribution",
            category="target_quality",
            passed=False,
            severity="critical",
            score=0.0,
            message="Regression target contains non-numeric values.",
            recommendation="Provide a numeric target for regression.",
        )

    return CheckResult(
        check_id="target.distribution",
        title="Target Distribution",
        category="target_quality",
        passed=True,
        severity="info",
        score=100.0,
        message="Regression target distribution was successfully analyzed.",
        evidence={
            "task_type": "regression",
            "total_samples": len(numeric_series),
            "mean": float(numeric_series.mean()),
            "std": float(numeric_series.std()),
            "min": float(numeric_series.min()),
            "max": float(numeric_series.max()),
        },
    )
