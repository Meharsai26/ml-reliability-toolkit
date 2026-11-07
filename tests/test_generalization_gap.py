import pytest

from ml_reliability.checks.generalization_gap import (
    check_generalization_gap,
)


class DummyClassificationModel:
    def predict(self, X):
        if X == "train":
            return [0, 1, 1, 0]

        return [0, 1, 0, 0]


def test_classification_generalization_gap():
    result = check_generalization_gap(
        model=DummyClassificationModel(),
        X_train="train",
        y_train=[0, 1, 1, 0],
        X_test="test",
        y_test=[0, 1, 1, 0],
        task="classification",
    )

    assert result.passed is False
    assert result.severity == "warning"
    assert result.evidence["task_type"] == "classification"
    assert result.evidence["metric"] == "accuracy"
    assert result.evidence["train_metric"] == 1.0
    assert result.evidence["test_metric"] == 0.75
    assert result.evidence["generalization_gap"] == 0.25


def test_classification_small_generalization_gap_passes():
    class Model:
        def predict(self, X):
            if X == "train":
                return [0, 1, 1, 0]

            return [0, 1, 1, 0]

    result = check_generalization_gap(
        model=Model(),
        X_train="train",
        y_train=[0, 1, 1, 0],
        X_test="test",
        y_test=[0, 1, 1, 0],
        task="classification",
    )

    assert result.passed is True
    assert result.severity == "info"
    assert result.evidence["train_metric"] == 1.0
    assert result.evidence["test_metric"] == 1.0
    assert result.evidence["generalization_gap"] == 0.0


def test_regression_generalization_gap():
    class Model:
        def predict(self, X):
            if X == "train":
                return [1.0, 2.0, 3.0, 4.0]

            return [1.0, 2.0, 3.5, 4.5]

    result = check_generalization_gap(
        model=Model(),
        X_train="train",
        y_train=[1.0, 2.0, 3.0, 4.0],
        X_test="test",
        y_test=[1.0, 2.0, 3.0, 4.0],
        task="regression",
    )

    assert result.passed is True
    assert result.severity == "info"
    assert result.evidence["metric"] == "r2"
    assert result.evidence["train_metric"] == 1.0
    assert result.evidence["test_metric"] == 0.9
    assert result.evidence["generalization_gap"] == 0.1


def test_regression_large_generalization_gap_fails():
    class Model:
        def predict(self, X):
            if X == "train":
                return [1.0, 2.0, 3.0, 4.0]

            return [4.0, 3.0, 2.0, 1.0]

    result = check_generalization_gap(
        model=Model(),
        X_train="train",
        y_train=[1.0, 2.0, 3.0, 4.0],
        X_test="test",
        y_test=[1.0, 2.0, 3.0, 4.0],
        task="regression",
    )

    assert result.passed is False
    assert result.severity == "warning"
    assert result.evidence["metric"] == "r2"
    assert result.evidence["train_metric"] == 1.0
    assert result.evidence["test_metric"] == -3.0
    assert result.evidence["generalization_gap"] == 4.0


def test_missing_model():
    result = check_generalization_gap(
        model=None,
        X_train=[1, 2],
        y_train=[0, 1],
        X_test=[3, 4],
        y_test=[0, 1],
        task="classification",
    )

    assert result.passed is False
    assert result.score == 0.0
    assert result.severity == "critical"


def test_missing_data():
    class Model:
        def predict(self, X):
            return [0, 1]

    result = check_generalization_gap(
        model=Model(),
        X_train=None,
        y_train=[0, 1],
        X_test=[3, 4],
        y_test=[0, 1],
        task="classification",
    )

    assert result.passed is False
    assert result.score == 0.0
    assert result.severity == "critical"


def test_invalid_task():
    class Model:
        def predict(self, X):
            return [0, 1]

    result = check_generalization_gap(
        model=Model(),
        X_train=[1, 2],
        y_train=[0, 1],
        X_test=[3, 4],
        y_test=[0, 1],
        task="unknown",
    )

    assert result.passed is False
    assert result.score == 0.0
    assert result.severity == "critical"


def test_prediction_failure():
    class FailingModel:
        def predict(self, X):
            raise RuntimeError("prediction failed")

    result = check_generalization_gap(
        model=FailingModel(),
        X_train=[1, 2],
        y_train=[0, 1],
        X_test=[3, 4],
        y_test=[0, 1],
        task="classification",
    )

    assert result.passed is False
    assert result.score == 0.0
    assert result.severity == "critical"


def test_custom_threshold():
    class Model:
        def predict(self, X):
            if X == "train":
                return [0, 1, 1, 0]

            return [0, 1, 0, 0]

    result = check_generalization_gap(
        model=Model(),
        X_train="train",
        y_train=[0, 1, 1, 0],
        X_test="test",
        y_test=[0, 1, 1, 0],
        task="classification",
        threshold=0.30,
    )

    assert result.passed is True
    assert result.evidence["threshold"] == 0.30
    assert result.evidence["generalization_gap"] == 0.25


def test_generalization_gap_rejects_negative_threshold():
    class Model:
        def predict(self, X):
            return [0, 1, 0, 1]

    result = check_generalization_gap(
        model=Model(),
        X_train=[1, 2, 3, 4],
        y_train=[0, 1, 0, 1],
        X_test=[1, 2, 3, 4],
        y_test=[0, 1, 0, 1],
        task="classification",
        threshold=-0.01,
    )

    assert result.passed is False
    assert result.severity == "critical"
    assert result.score == 0.0
    assert "threshold" in result.message.lower()


@pytest.mark.parametrize(
    "threshold",
    [float("nan"), float("inf"), float("-inf")],
)
def test_generalization_gap_rejects_non_finite_threshold(threshold):
    class Model:
        def predict(self, X):
            return [0, 1, 0, 1]

    result = check_generalization_gap(
        model=Model(),
        X_train=[1, 2, 3, 4],
        y_train=[0, 1, 0, 1],
        X_test=[1, 2, 3, 4],
        y_test=[0, 1, 0, 1],
        task="classification",
        threshold=threshold,
    )

    assert result.passed is False
    assert result.severity == "critical"
    assert result.score == 0.0
    assert "threshold" in result.message.lower()


@pytest.mark.parametrize("threshold", ["0.1", None, True])
def test_generalization_gap_rejects_invalid_threshold_type(threshold):
    class Model:
        def predict(self, X):
            return [0, 1, 0, 1]

    result = check_generalization_gap(
        model=Model(),
        X_train=[1, 2, 3, 4],
        y_train=[0, 1, 0, 1],
        X_test=[1, 2, 3, 4],
        y_test=[0, 1, 0, 1],
        task="classification",
        threshold=threshold,
    )

    assert result.passed is False
    assert result.severity == "critical"
    assert result.score == 0.0
    assert "threshold" in result.message.lower()


def test_invalid_threshold_preserves_check_metadata():
    class Model:
        def predict(self, X):
            return [0, 1, 0, 1]

    result = check_generalization_gap(
        model=Model(),
        X_train=[1, 2, 3, 4],
        y_train=[0, 1, 0, 1],
        X_test=[1, 2, 3, 4],
        y_test=[0, 1, 0, 1],
        task="classification",
        threshold=-0.01,
    )

    assert result.check_id == "model.generalization_gap"
    assert result.category == "performance"
