from ml_reliability.checks.model_performance import (
    check_model_performance,
)


class DummyClassificationModel:
    def predict(self, X):
        return [0, 1, 1, 0]


class DummyRegressionModel:
    def predict(self, X):
        return [1.0, 2.0, 3.0, 4.0]


class InvalidModel:
    pass


def test_classification_accuracy():
    result = check_model_performance(
        model=DummyClassificationModel(),
        X_test=[1, 2, 3, 4],
        y_test=[0, 1, 1, 0],
        task="classification",
    )

    assert result.passed is True
    assert result.score == 100.0
    assert result.evidence["task_type"] == "classification"
    assert result.evidence["metric"] == "accuracy"
    assert result.evidence["metric_value"] == 1.0


def test_regression_r2():
    result = check_model_performance(
        model=DummyRegressionModel(),
        X_test=[1, 2, 3, 4],
        y_test=[1.0, 2.0, 3.0, 4.0],
        task="regression",
    )

    assert result.passed is True
    assert result.score == 100.0
    assert result.evidence["task_type"] == "regression"
    assert result.evidence["metric"] == "r2"
    assert result.evidence["metric_value"] == 1.0


def test_missing_model():
    result = check_model_performance(
        model=None,
        X_test=[1, 2],
        y_test=[0, 1],
        task="classification",
    )

    assert result.passed is False
    assert result.score == 0.0
    assert result.severity == "critical"


def test_missing_test_data():
    result = check_model_performance(
        model=DummyClassificationModel(),
        X_test=None,
        y_test=None,
        task="classification",
    )

    assert result.passed is False
    assert result.score == 0.0
    assert result.severity == "critical"


def test_invalid_task():
    result = check_model_performance(
        model=DummyClassificationModel(),
        X_test=[1, 2],
        y_test=[0, 1],
        task="unknown",
    )

    assert result.passed is False
    assert result.score == 0.0
    assert result.severity == "critical"


def test_invalid_model():
    result = check_model_performance(
        model=InvalidModel(),
        X_test=[1, 2],
        y_test=[0, 1],
        task="classification",
    )

    assert result.passed is False
    assert result.score == 0.0
    assert result.severity == "critical"


def test_prediction_length_mismatch():
    class ShortPredictionModel:
        def predict(self, X):
            return [0]

    result = check_model_performance(
        model=ShortPredictionModel(),
        X_test=[1, 2],
        y_test=[0, 1],
        task="classification",
    )

    assert result.passed is False
    assert result.score == 0.0
    assert result.severity == "critical"


def test_zero_variance_regression_target():
    result = check_model_performance(
        model=DummyRegressionModel(),
        X_test=[1, 2, 3],
        y_test=[5.0, 5.0, 5.0],
        task="regression",
    )

    assert result.passed is False
    assert result.score == 0.0
    assert result.severity == "critical"


def test_classification_score_reflects_actual_accuracy():
    class PoorClassificationModel:
        def predict(self, X):
            return [1, 0, 0, 1]

    result = check_model_performance(
        model=PoorClassificationModel(),
        X_test=[1, 2, 3, 4],
        y_test=[0, 1, 1, 0],
        task="classification",
    )

    assert result.passed is True
    assert result.evidence["metric_value"] == 0.0
    assert result.score == 0.0


def test_regression_score_reflects_negative_r2():
    class PoorRegressionModel:
        def predict(self, X):
            return [10.0, 10.0, 10.0, 10.0]

    result = check_model_performance(
        model=PoorRegressionModel(),
        X_test=[1, 2, 3, 4],
        y_test=[1.0, 2.0, 3.0, 4.0],
        task="regression",
    )

    assert result.passed is True
    assert result.evidence["metric_value"] < 0.0
    assert result.score < 100.0


def test_regression_non_finite_predictions_fail():
    class NonFiniteRegressionModel:
        def predict(self, X):
            return [1.0, float("nan"), 3.0, 4.0]

    result = check_model_performance(
        model=NonFiniteRegressionModel(),
        X_test=[1, 2, 3, 4],
        y_test=[1.0, 2.0, 3.0, 4.0],
        task="regression",
    )

    assert result.passed is False
    assert result.score == 0.0
    assert "Performance calculation failed" in result.message


def test_regression_infinite_predictions_fail():
    class InfiniteRegressionModel:
        def predict(self, X):
            return [1.0, float("inf"), 3.0, 4.0]

    result = check_model_performance(
        model=InfiniteRegressionModel(),
        X_test=[1, 2, 3, 4],
        y_test=[1.0, 2.0, 3.0, 4.0],
        task="regression",
    )

    assert result.passed is False
    assert result.score == 0.0
    assert "non-finite numeric values" in result.message


def test_regression_non_finite_targets_fail():
    result = check_model_performance(
        model=DummyRegressionModel(),
        X_test=[1, 2, 3, 4],
        y_test=[1.0, float("inf"), 3.0, 4.0],
        task="regression",
    )

    assert result.passed is False
    assert result.score == 0.0
    assert "non-finite numeric values" in result.message


def test_classification_nan_predictions_fail():
    class NaNClassificationModel:
        def predict(self, X):
            return [0, float("nan"), 1, 0]

    result = check_model_performance(
        model=NaNClassificationModel(),
        X_test=[1, 2, 3, 4],
        y_test=[0, 1, 1, 0],
        task="classification",
    )

    assert result.passed is False
    assert result.score == 0.0


def test_classification_non_finite_targets_fail():
    result = check_model_performance(
        model=DummyClassificationModel(),
        X_test=[1, 2, 3, 4],
        y_test=[0, float("inf"), 1, 0],
        task="classification",
    )

    assert result.passed is False
    assert result.score == 0.0


def test_classification_supports_string_labels():
    class StringClassificationModel:
        def predict(self, X):
            return ["cat", "dog", "dog", "cat"]

    result = check_model_performance(
        model=StringClassificationModel(),
        X_test=[1, 2, 3, 4],
        y_test=["cat", "dog", "dog", "cat"],
        task="classification",
    )

    assert result.passed is True
    assert result.evidence["metric_value"] == 1.0
    assert result.score == 100.0
