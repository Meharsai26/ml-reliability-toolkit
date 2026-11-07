from typing import Any

from ml_reliability.core.result import CheckResult


def check_slice_analysis(
    model: Any,
    X_test: Any,
    y_test: Any,
    slices: Any,
    task: str,
) -> CheckResult:
    """
    Calculate classification accuracy separately for each slice.

    The current implementation supports classification only.
    """

    if model is None:
        return CheckResult(
            check_id="model.slice_analysis",
            title="Slice Analysis",
            category="performance",
            passed=False,
            severity="critical",
            score=0.0,
            message="A trained model is required.",
            recommendation="Provide a trained model with a predict() method.",
        )

    if X_test is None or y_test is None or slices is None:
        return CheckResult(
            check_id="model.slice_analysis",
            title="Slice Analysis",
            category="performance",
            passed=False,
            severity="critical",
            score=0.0,
            message="X_test, y_test, and slices are required.",
            recommendation="Provide test data, targets, and slice labels.",
        )

    if task != "classification":
        return CheckResult(
            check_id="model.slice_analysis",
            title="Slice Analysis",
            category="performance",
            passed=False,
            severity="critical",
            score=0.0,
            message="Slice analysis currently supports classification only.",
            recommendation="Use task='classification'.",
        )

    if not hasattr(model, "predict") or not callable(model.predict):
        return CheckResult(
            check_id="model.slice_analysis",
            title="Slice Analysis",
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
        slice_labels = list(slices)

        if not actual:
            raise ValueError("y_test must contain at least one sample.")

        if len(actual) != len(predictions):
            raise ValueError("Predictions and y_test must have the same length.")

        if len(actual) != len(slice_labels):
            raise ValueError("Slices and y_test must have the same length.")

        grouped_results = {}

        for prediction, target, slice_label in zip(
            predictions,
            actual,
            slice_labels,
        ):
            if slice_label not in grouped_results:
                grouped_results[slice_label] = {
                    "correct": 0,
                    "total": 0,
                }

            grouped_results[slice_label]["total"] += 1

            if prediction == target:
                grouped_results[slice_label]["correct"] += 1

        slice_metrics = {}

        for slice_label, values in grouped_results.items():
            accuracy = values["correct"] / values["total"]

            slice_metrics[str(slice_label)] = {
                "accuracy": round(float(accuracy), 4),
                "samples": values["total"],
            }

        accuracies = [metrics["accuracy"] for metrics in slice_metrics.values()]

        slice_accuracy_gap = max(accuracies) - min(accuracies)

        score = round(
            max(0.0, min(100.0, (1.0 - slice_accuracy_gap) * 100.0)),
            2,
        )

        return CheckResult(
            check_id="model.slice_analysis",
            title="Slice Analysis",
            category="performance",
            passed=True,
            severity="info",
            score=score,
            message=(
                "Classification accuracy calculated for each slice. "
                f"Accuracy gap: {slice_accuracy_gap:.4f}. "
                f"Slice consistency score: {score:.2f}/100."
            ),
            evidence={
                "task": "classification",
                "metric": "accuracy",
                "slices": slice_metrics,
                "slice_accuracy_gap": round(
                    float(slice_accuracy_gap),
                    4,
                ),
                "test_samples": len(actual),
            },
            recommendation="",
        )

    except Exception as exc:  # noqa: BLE001 - Convert evaluation failures to check results.
        return CheckResult(
            check_id="model.slice_analysis",
            title="Slice Analysis",
            category="performance",
            passed=False,
            severity="critical",
            score=0.0,
            message=f"Slice analysis failed: {exc}",
            recommendation=(
                "Verify that predictions, targets, and slice labels are valid."
            ),
        )
