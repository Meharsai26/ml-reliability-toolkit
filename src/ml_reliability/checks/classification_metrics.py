from typing import Any

import numpy as np

from ml_reliability.core.result import CheckResult


def check_classification_metrics(
    model: Any,
    X_test: Any,
    y_test: Any,
) -> CheckResult:
    """
    Calculate classification precision and recall.

    This check currently supports binary classification.
    """

    if model is None:
        return CheckResult(
            check_id="model.classification_metrics",
            title="Classification Metrics",
            category="performance",
            passed=False,
            severity="critical",
            score=0.0,
            message="A trained model is required.",
            recommendation="Provide a trained model with a predict() method.",
        )

    if X_test is None or y_test is None:
        return CheckResult(
            check_id="model.classification_metrics",
            title="Classification Metrics",
            category="performance",
            passed=False,
            severity="critical",
            score=0.0,
            message="X_test and y_test are required.",
            recommendation="Provide both X_test and y_test.",
        )

    if not hasattr(model, "predict") or not callable(model.predict):
        return CheckResult(
            check_id="model.classification_metrics",
            title="Classification Metrics",
            category="performance",
            passed=False,
            severity="critical",
            score=0.0,
            message="The model must provide a callable predict() method.",
            recommendation="Provide a trained model implementing predict().",
        )

    try:
        predictions = list(model.predict(X_test))
        actual = list(y_test)

        valid_labels = {0, 1}

        for name, values in (
            ("predictions", predictions),
            ("targets", actual),
        ):
            for index, value in enumerate(values):
                if (
                    isinstance(value, (bool, np.bool_))
                    or not isinstance(
                        value,
                        (int, float, np.integer, np.floating),
                    )
                    or value not in valid_labels
                ):
                    raise ValueError(
                        f"Invalid binary label in {name} at index "
                        f"{index}: {value!r}. Labels must be 0 or 1."
                    )

        if not actual:
            raise ValueError("y_test must contain at least one sample.")

        if len(actual) != len(predictions):
            raise ValueError("Predictions and y_test must have the same length.")

        true_positive = sum(
            prediction == 1 and target == 1
            for prediction, target in zip(predictions, actual)
        )

        false_positive = sum(
            prediction == 1 and target != 1
            for prediction, target in zip(predictions, actual)
        )

        false_negative = sum(
            prediction != 1 and target == 1
            for prediction, target in zip(predictions, actual)
        )

        if true_positive + false_positive == 0:
            raise ValueError(
                "Precision cannot be calculated because the model "
                "predicted no positive samples."
            )

        precision = true_positive / (true_positive + false_positive)

        if true_positive + false_negative == 0:
            raise ValueError(
                "Recall cannot be calculated because there are no "
                "positive samples in y_test."
            )

        recall = true_positive / (true_positive + false_negative)

        f1 = 2 * precision * recall / (precision + recall)

        return CheckResult(
            check_id="model.classification_metrics",
            title="Classification Metrics",
            category="performance",
            passed=True,
            severity="info",
            score=round(float(f1 * 100.0), 2),
            message=(
                f"Classification metrics calculated successfully: "
                f"precision={precision:.4f}, "
                f"recall={recall:.4f}, "
                f"f1={f1:.4f}."
            ),
            evidence={
                "metric": "classification_metrics",
                "precision": round(float(precision), 4),
                "recall": round(float(recall), 4),
                "f1": round(float(f1), 4),
                "test_samples": len(actual),
            },
            recommendation="",
        )

    except Exception as exc:  # noqa: BLE001 - Convert evaluation failures to check results.
        return CheckResult(
            check_id="model.classification_metrics",
            title="Classification Metrics",
            category="performance",
            passed=False,
            severity="critical",
            score=0.0,
            message=f"Classification metric calculation failed: {exc}",
            recommendation=("Verify that predictions and test targets are valid."),
        )
