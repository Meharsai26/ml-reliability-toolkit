import math
from math import sqrt
from typing import Any

from ml_reliability.core.result import CheckResult


def check_regression_metrics(
    model: Any,
    X_test: Any,
    y_test: Any,
    max_acceptable_mae: float | None = None,
    max_scoring_mae: float | None = None,
) -> CheckResult:
    """Calculate regression MAE and RMSE and optionally assess quality."""

    check_id = "model.regression_metrics"
    title = "Regression Metrics"
    category = "performance"

    def configuration_error(message: str) -> CheckResult:
        return CheckResult(
            check_id=check_id,
            title=title,
            category=category,
            passed=False,
            severity="critical",
            score=0.0,
            message=message,
            recommendation="Correct the regression metric configuration.",
        )

    for name, value in (
        ("max_acceptable_mae", max_acceptable_mae),
        ("max_scoring_mae", max_scoring_mae),
    ):
        if value is None:
            continue

        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
            or (name == "max_scoring_mae" and value == 0)
        ):
            requirement = (
                "a finite, strictly positive number"
                if name == "max_scoring_mae"
                else "a finite, non-negative number"
            )
            return configuration_error(f"{name} must be {requirement}.")

    if model is None:
        return CheckResult(
            check_id=check_id,
            title=title,
            category=category,
            passed=False,
            severity="critical",
            score=0.0,
            message="A trained model is required.",
            recommendation="Provide a trained model with a predict() method.",
        )

    if X_test is None or y_test is None:
        return CheckResult(
            check_id=check_id,
            title=title,
            category=category,
            passed=False,
            severity="critical",
            score=0.0,
            message="X_test and y_test are required.",
            recommendation="Provide both X_test and y_test.",
        )

    if not callable(getattr(model, "predict", None)):
        return CheckResult(
            check_id=check_id,
            title=title,
            category=category,
            passed=False,
            severity="critical",
            score=0.0,
            message="The model must provide a callable predict() method.",
            recommendation="Provide a trained model implementing predict().",
        )

    try:
        predictions = list(model.predict(X_test))
        actual = list(y_test)

        for name, values in (
            ("predictions", predictions),
            ("targets", actual),
        ):
            for index, value in enumerate(values):
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                ):
                    raise ValueError(
                        f"Non-finite or invalid numeric value in "
                        f"{name} at index {index}: {value!r}."
                    )

        if not actual:
            raise ValueError("y_test must contain at least one sample.")

        if len(actual) != len(predictions):
            raise ValueError("Predictions and y_test must have the same length.")

        absolute_errors = [
            abs(prediction - target) for prediction, target in zip(predictions, actual)
        ]
        mae = sum(absolute_errors) / len(actual)

        squared_errors = [
            (prediction - target) ** 2
            for prediction, target in zip(predictions, actual)
        ]
        rmse = sqrt(sum(squared_errors) / len(actual))

        if not math.isfinite(mae) or not math.isfinite(rmse):
            raise ValueError(
                "Calculated MAE and RMSE must be finite; "
                f"received mae={mae!r}, rmse={rmse!r}."
            )

        passed = True if max_acceptable_mae is None else mae <= max_acceptable_mae

        if max_scoring_mae is not None:
            score = max(
                0.0,
                min(100.0, (1.0 - mae / max_scoring_mae) * 100.0),
            )
        elif max_acceptable_mae is not None:
            score = 100.0 if passed else 0.0
        else:
            score = None

        severity = "info" if max_acceptable_mae is None or passed else "warning"

        message = (
            f"Regression metrics calculated successfully: "
            f"mae={mae:.4f}, rmse={rmse:.4f}."
        )

        if max_acceptable_mae is None:
            message += (
                " No MAE tolerance configured; "
                "model quality was not assessed against a pass/fail threshold."
            )
        else:
            message += (
                f" MAE tolerance={max_acceptable_mae:.4f}; "
                f"tolerance {'met' if passed else 'exceeded'}."
            )

        if max_scoring_mae is not None:
            message += f" Scoring scale={max_scoring_mae:.4f}; score={score:.2f}/100."

        recommendation = (
            ""
            if passed
            else (
                "Investigate prediction errors or revise the configured MAE tolerance."
            )
        )

        return CheckResult(
            check_id=check_id,
            title=title,
            category=category,
            passed=passed,
            severity=severity,
            score=score,
            message=message,
            evidence={
                "metric": "mae",
                "mae": round(float(mae), 4),
                "rmse": round(float(rmse), 4),
                "test_samples": len(actual),
                "max_acceptable_mae": max_acceptable_mae,
                "max_scoring_mae": max_scoring_mae,
            },
            recommendation=recommendation,
        )

    except Exception as exc:  # noqa: BLE001 - Convert evaluation failures to check results.
        return CheckResult(
            check_id=check_id,
            title=title,
            category=category,
            passed=False,
            severity="critical",
            score=0.0,
            message=f"Regression metric calculation failed: {exc}",
            recommendation=("Verify that predictions and test targets are valid."),
        )
