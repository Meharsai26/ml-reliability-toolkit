import math
from numbers import Real
from typing import Any

from ml_reliability.core.result import CheckResult


def check_generalization_gap(
    model: Any,
    X_train: Any,
    y_train: Any,
    X_test: Any,
    y_test: Any,
    task: str,
    threshold: float = 0.10,
) -> CheckResult:
    """
    Measure the performance gap between training and test data.

    Supported tasks:
    - classification: accuracy
    - regression: RÂ²
    """

    if model is None:
        return CheckResult(
            check_id="model.generalization_gap",
            title="Generalization Gap",
            category="performance",
            passed=False,
            severity="critical",
            score=0.0,
            message="A trained model is required.",
            recommendation="Provide a trained model with a predict() method.",
        )

    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, Real)
        or not math.isfinite(threshold)
        or threshold < 0
    ):
        return CheckResult(
            check_id="model.generalization_gap",
            title="Generalization Gap",
            category="performance",
            passed=False,
            severity="critical",
            score=0.0,
            message=(
                "Invalid threshold: threshold must be a finite, non-negative number."
            ),
            evidence={"threshold": repr(threshold)},
            recommendation=(
                "Provide a finite, non-negative threshold for the generalization gap."
            ),
        )

    if X_train is None or y_train is None or X_test is None or y_test is None:
        return CheckResult(
            check_id="model.generalization_gap",
            title="Generalization Gap",
            category="performance",
            passed=False,
            severity="critical",
            score=0.0,
            message="Training and test data are required.",
            recommendation=("Provide X_train, y_train, X_test, and y_test."),
        )

    if task not in {"classification", "regression"}:
        return CheckResult(
            check_id="model.generalization_gap",
            title="Generalization Gap",
            category="performance",
            passed=False,
            severity="critical",
            score=0.0,
            message=f"Unsupported task type: {task}.",
            recommendation=("Use task='classification' or task='regression'."),
        )

    if not hasattr(model, "predict") or not callable(model.predict):
        return CheckResult(
            check_id="model.generalization_gap",
            title="Generalization Gap",
            category="performance",
            passed=False,
            severity="critical",
            score=0.0,
            message="The model must provide a callable predict() method.",
            recommendation="Provide a trained model implementing predict().",
        )

    try:
        train_predictions = model.predict(X_train)
        test_predictions = model.predict(X_test)

        train_actual = list(y_train)
        test_actual = list(y_test)

        train_predicted = list(train_predictions)
        test_predicted = list(test_predictions)

        if not train_actual or not test_actual:
            raise ValueError(
                "Training and test targets must contain at least one sample."
            )

        if len(train_actual) != len(train_predicted):
            raise ValueError(
                "Training predictions and y_train must have the same length."
            )

        if len(test_actual) != len(test_predicted):
            raise ValueError("Test predictions and y_test must have the same length.")

        if task == "classification":
            train_correct = sum(
                prediction == target
                for prediction, target in zip(
                    train_predicted,
                    train_actual,
                )
            )

            test_correct = sum(
                prediction == target
                for prediction, target in zip(
                    test_predicted,
                    test_actual,
                )
            )

            train_metric = train_correct / len(train_actual)
            test_metric = test_correct / len(test_actual)

            metric_name = "accuracy"

        else:
            train_mean = sum(train_actual) / len(train_actual)
            test_mean = sum(test_actual) / len(test_actual)

            train_total = sum((target - train_mean) ** 2 for target in train_actual)

            test_total = sum((target - test_mean) ** 2 for target in test_actual)

            if train_total == 0 or test_total == 0:
                raise ValueError(
                    "RÂ² cannot be calculated when target variance is zero."
                )

            train_residual = sum(
                (target - prediction) ** 2
                for target, prediction in zip(
                    train_actual,
                    train_predicted,
                )
            )

            test_residual = sum(
                (target - prediction) ** 2
                for target, prediction in zip(
                    test_actual,
                    test_predicted,
                )
            )

            train_metric = 1 - train_residual / train_total
            test_metric = 1 - test_residual / test_total

            metric_name = "r2"

        generalization_gap = train_metric - test_metric

        passed = generalization_gap <= threshold

        return CheckResult(
            check_id="model.generalization_gap",
            title="Generalization Gap",
            category="performance",
            passed=passed,
            severity="info" if passed else "warning",
            score=100.0 if passed else 0.0,
            message=(
                f"Generalization gap calculated successfully: {generalization_gap:.4f}."
            ),
            evidence={
                "task_type": task,
                "metric": metric_name,
                "train_metric": round(float(train_metric), 4),
                "test_metric": round(float(test_metric), 4),
                "generalization_gap": round(
                    float(generalization_gap),
                    4,
                ),
                "threshold": threshold,
            },
            recommendation=(
                ""
                if passed
                else "Investigate possible overfitting or "
                "distribution differences between training and test data."
            ),
        )

    except Exception as exc:  # noqa: BLE001 - Convert evaluation failures to check results.
        return CheckResult(
            check_id="model.generalization_gap",
            title="Generalization Gap",
            category="performance",
            passed=False,
            severity="critical",
            score=0.0,
            message=f"Generalization gap calculation failed: {exc}",
            recommendation=("Verify that the model and training/test data are valid."),
        )
