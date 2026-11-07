import math
from typing import Any

from ml_reliability.core.result import CheckResult


def check_calibration(
    model: Any,
    X_test: Any,
    y_test: Any,
) -> CheckResult:
    """
    Calculate mean absolute probability error for binary classification.

    The model must provide predict_proba().
    """

    if model is None:
        return CheckResult(
            check_id="model.calibration",
            title="Classification Calibration",
            category="performance",
            passed=False,
            severity="critical",
            score=0.0,
            message="A trained model is required.",
            recommendation="Provide a trained classification model.",
        )

    if X_test is None or y_test is None:
        return CheckResult(
            check_id="model.calibration",
            title="Classification Calibration",
            category="performance",
            passed=False,
            severity="critical",
            score=0.0,
            message="X_test and y_test are required.",
            recommendation="Provide both X_test and y_test.",
        )

    if not hasattr(model, "predict_proba") or not callable(model.predict_proba):
        return CheckResult(
            check_id="model.calibration",
            title="Classification Calibration",
            category="performance",
            passed=False,
            severity="critical",
            score=0.0,
            message="The model must provide a callable predict_proba() method.",
            recommendation=(
                "Provide a classification model implementing predict_proba()."
            ),
        )

    try:
        probabilities = list(model.predict_proba(X_test))
        actual = list(y_test)

        if not actual:
            raise ValueError("y_test must contain at least one sample.")

        if len(actual) != len(probabilities):
            raise ValueError(
                "Predicted probabilities and y_test must have the same length."
            )

        positive_probabilities = []

        for probability in probabilities:
            if len(probability) != 2:
                raise ValueError(
                    "Binary classification probabilities must contain exactly two class probabilities."
                )

            try:
                class_probabilities = [
                    float(probability[0]),
                    float(probability[1]),
                ]
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(
                    "Class probabilities must be valid finite numbers."
                ) from exc

            if not all(math.isfinite(value) for value in class_probabilities):
                raise ValueError("Class probabilities must be finite numbers.")

            if any(value < 0.0 or value > 1.0 for value in class_probabilities):
                raise ValueError("Class probabilities must be between 0 and 1.")

            if not math.isclose(
                sum(class_probabilities),
                1.0,
                rel_tol=0.0,
                abs_tol=1e-6,
            ):
                raise ValueError("Binary class probabilities must sum to 1.")

            positive_probabilities.append(class_probabilities[1])

        absolute_errors = [
            abs(probability - target)
            for probability, target in zip(
                positive_probabilities,
                actual,
            )
        ]

        calibration_error = sum(absolute_errors) / len(actual)

        score = round(
            max(0.0, min(100.0, (1.0 - calibration_error) * 100.0)),
            2,
        )

        return CheckResult(
            check_id="model.calibration",
            title="Classification Calibration",
            category="performance",
            passed=True,
            severity="info",
            score=score,
            message=(
                "Classification probability calibration error "
                f"calculated successfully: {calibration_error:.4f}. "
                f"Calibration score: {score:.2f}/100."
            ),
            evidence={
                "metric": "mean_absolute_probability_error",
                "calibration_error": round(
                    float(calibration_error),
                    4,
                ),
                "test_samples": len(actual),
            },
            recommendation="",
        )

    except Exception as exc:  # noqa: BLE001 - Convert evaluation failures to check results.
        return CheckResult(
            check_id="model.calibration",
            title="Classification Calibration",
            category="performance",
            passed=False,
            severity="critical",
            score=0.0,
            message=f"Calibration calculation failed: {exc}",
            recommendation=(
                "Verify that predicted probabilities and test targets are valid."
            ),
        )
