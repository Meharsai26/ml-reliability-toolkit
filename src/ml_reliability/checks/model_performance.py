import math
from typing import Any

from ml_reliability.core.result import CheckResult


def check_model_performance(
    model: Any,
    X_test: Any,
    y_test: Any,
    task: str,
) -> CheckResult:
    """
    Evaluate basic model performance on test data.

    Supported tasks:
    - classification: accuracy
    - regression: RÂ²

    The reliability score is based on the calculated metric:
    - Classification: accuracy * 100
    - Regression: RÂ² * 100, bounded between 0 and 100

    A successful calculation does not necessarily mean the model
    has acceptable performance. Check the score and metric evidence.
    """

    check_id = "model.performance"
    title = "Model Performance"
    category = "performance"

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

    if task not in {"classification", "regression"}:
        return CheckResult(
            check_id=check_id,
            title=title,
            category=category,
            passed=False,
            severity="critical",
            score=0.0,
            message=f"Unsupported task type: {task}.",
            recommendation=("Use task='classification' or task='regression'."),
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
        predictions = model.predict(X_test)
    except Exception as exc:  # noqa: BLE001 - Convert evaluation failures to check results.
        return CheckResult(
            check_id=check_id,
            title=title,
            category=category,
            passed=False,
            severity="critical",
            score=0.0,
            message=f"Model prediction failed: {exc}",
            recommendation=(
                "Verify that the model is trained and compatible with X_test."
            ),
        )

    try:
        actual = list(y_test)
        predicted = list(predictions)

        if not actual:
            raise ValueError("y_test must contain at least one sample.")

        if len(actual) != len(predicted):
            raise ValueError("Predictions and y_test must have the same length.")

        def contains_non_finite_numeric_value(values: list[Any]) -> bool:
            for value in values:
                if isinstance(value, (int, float)):
                    if not math.isfinite(value):
                        return True
                elif hasattr(value, "item"):
                    # Handle NumPy scalar values without rejecting strings.
                    scalar = value.item()
                    if isinstance(scalar, (int, float)) and not math.isfinite(scalar):
                        return True
            return False

        if contains_non_finite_numeric_value(
            predicted
        ) or contains_non_finite_numeric_value(actual):
            raise ValueError(
                "Predictions and targets must not contain non-finite numeric values."
            )

        if task == "classification":
            correct = sum(
                prediction == target for prediction, target in zip(predicted, actual)
            )

            metric_name = "accuracy"
            metric_value = correct / len(actual)

        else:
            mean_actual = sum(actual) / len(actual)

            total_sum_squares = sum((target - mean_actual) ** 2 for target in actual)

            if total_sum_squares == 0:
                raise ValueError(
                    "RÂ² cannot be calculated when y_test has zero variance."
                )

            residual_sum_squares = sum(
                (target - prediction) ** 2
                for target, prediction in zip(actual, predicted)
            )

            metric_name = "r2"
            metric_value = 1 - residual_sum_squares / total_sum_squares

        score = max(
            0.0,
            min(100.0, metric_value * 100.0),
        )

    except Exception as exc:  # noqa: BLE001 - Convert evaluation failures to check results.
        return CheckResult(
            check_id=check_id,
            title=title,
            category=category,
            passed=False,
            severity="critical",
            score=0.0,
            message=f"Performance calculation failed: {exc}",
            recommendation=("Verify that predictions and test targets are valid."),
        )

    return CheckResult(
        check_id=check_id,
        title=title,
        category=category,
        passed=True,
        severity="info",
        score=round(float(score), 2),
        message=(
            f"Model {metric_name} calculated successfully: "
            f"{metric_value:.4f}. "
            f"Reliability score: {score:.2f}/100."
        ),
        evidence={
            "task_type": task,
            "metric": metric_name,
            "metric_value": round(float(metric_value), 4),
            "test_samples": len(actual),
        },
        recommendation="",
    )
