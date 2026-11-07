from typing import Any

import pandas as pd

from ml_reliability.core.result import CheckResult


def check_distribution_shift(
    X_train: Any,
    X_test: Any,
) -> CheckResult:
    """
    Compare numeric feature distributions between training and test data.
    """

    if X_train is None or X_test is None:
        return CheckResult(
            check_id="data.distribution_shift",
            title="Train/Test Distribution Shift",
            category="distribution",
            passed=False,
            severity="critical",
            score=0.0,
            message="Both training and test data are required.",
            recommendation="Provide X_train and X_test before running distribution shift checks.",
        )

    if not isinstance(X_train, pd.DataFrame) or not isinstance(X_test, pd.DataFrame):
        return CheckResult(
            check_id="data.distribution_shift",
            title="Train/Test Distribution Shift",
            category="distribution",
            passed=False,
            severity="critical",
            score=0.0,
            message="Training and test data must both be pandas DataFrames.",
            recommendation="Provide X_train and X_test as pandas DataFrames.",
        )

    if X_train.empty or X_test.empty:
        return CheckResult(
            check_id="data.distribution_shift",
            title="Train/Test Distribution Shift",
            category="distribution",
            passed=False,
            severity="critical",
            score=0.0,
            message="Training and test data must both contain at least one row.",
            evidence={
                "train_rows": len(X_train),
                "test_rows": len(X_test),
            },
            recommendation="Provide non-empty training and test datasets.",
        )

    shared_columns = [column for column in X_train.columns if column in X_test.columns]

    numeric_columns = [
        column
        for column in shared_columns
        if pd.api.types.is_numeric_dtype(X_train[column])
        and pd.api.types.is_numeric_dtype(X_test[column])
    ]

    if not numeric_columns:
        return CheckResult(
            check_id="data.distribution_shift",
            title="Train/Test Distribution Shift",
            category="distribution",
            passed=True,
            severity="info",
            score=100.0,
            message="No shared numeric features were available for distribution comparison.",
            evidence={
                "shared_columns": [str(column) for column in shared_columns],
                "numeric_features_compared": 0,
                "shifted_features": [],
            },
            recommendation="Add numeric features or implement categorical distribution checks for categorical data.",
        )

    feature_statistics = {}
    moderate_shift_features = []
    significant_shift_features = []

    for column in numeric_columns:
        train_series = pd.to_numeric(X_train[column], errors="coerce").dropna()
        test_series = pd.to_numeric(X_test[column], errors="coerce").dropna()

        if train_series.empty or test_series.empty:
            continue

        train_mean = float(train_series.mean())
        test_mean = float(test_series.mean())

        absolute_difference = abs(test_mean - train_mean)

        if train_mean != 0:
            relative_shift = (absolute_difference / abs(train_mean)) * 100
        else:
            relative_shift = 0.0 if test_mean == 0 else float("inf")

        if relative_shift < 10:
            status = "stable"
        elif relative_shift <= 20:
            status = "moderate"
            moderate_shift_features.append(str(column))
        else:
            status = "significant"
            significant_shift_features.append(str(column))

        feature_statistics[str(column)] = {
            "train_mean": round(train_mean, 4),
            "test_mean": round(test_mean, 4),
            "absolute_difference": round(absolute_difference, 4),
            "relative_shift_percentage": (
                round(relative_shift, 2) if relative_shift != float("inf") else None
            ),
            "status": status,
        }

    if significant_shift_features:
        severity = "warning"
        score = 50.0
        passed = True
        message = (
            f"Detected significant distribution shift in "
            f"{len(significant_shift_features)} feature(s)."
        )
    elif moderate_shift_features:
        severity = "warning"
        score = 80.0
        passed = True
        message = (
            f"Detected moderate distribution shift in "
            f"{len(moderate_shift_features)} feature(s)."
        )
    else:
        severity = "info"
        score = 100.0
        passed = True
        message = "No significant numeric distribution shift was detected."

    return CheckResult(
        check_id="data.distribution_shift",
        title="Train/Test Distribution Shift",
        category="distribution",
        passed=passed,
        severity=severity,
        score=score,
        message=message,
        evidence={
            "train_rows": len(X_train),
            "test_rows": len(X_test),
            "shared_columns": [str(column) for column in shared_columns],
            "numeric_features_compared": len(numeric_columns),
            "moderate_shift_features": moderate_shift_features,
            "significant_shift_features": significant_shift_features,
            "feature_statistics": feature_statistics,
        },
        recommendation=(
            "Investigate significant train/test distribution differences "
            "before relying on model evaluation results."
            if significant_shift_features
            else (
                "Monitor moderate distribution differences and verify that "
                "they are expected."
                if moderate_shift_features
                else ""
            )
        ),
    )
