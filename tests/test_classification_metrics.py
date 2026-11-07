import numpy as np

from ml_reliability.checks.classification_metrics import (
    check_classification_metrics,
)


class DummyClassificationModel:
    def predict(self, X):
        return [1, 1, 0, 0]


def test_classification_precision():
    result = check_classification_metrics(
        model=DummyClassificationModel(),
        X_test=[1, 2, 3, 4],
        y_test=[1, 0, 1, 0],
    )

    assert result.passed is True
    assert result.evidence["precision"] == 0.5


def test_precision_when_no_positive_predictions():
    class Model:
        def predict(self, X):
            return [0, 0, 0, 0]

    result = check_classification_metrics(
        model=Model(),
        X_test=[1, 2, 3, 4],
        y_test=[1, 0, 1, 0],
    )

    assert result.passed is False
    assert result.score == 0.0
    assert result.severity == "critical"


def test_classification_recall():
    class Model:
        def predict(self, X):
            return [1, 1, 0, 0]

    result = check_classification_metrics(
        model=Model(),
        X_test=[1, 2, 3, 4],
        y_test=[1, 0, 1, 0],
    )

    assert result.passed is True
    assert result.evidence["recall"] == 0.5


def test_recall_when_no_positive_samples():
    class Model:
        def predict(self, X):
            return [0, 0, 0, 0]

    result = check_classification_metrics(
        model=Model(),
        X_test=[1, 2, 3, 4],
        y_test=[0, 0, 0, 0],
    )

    assert result.passed is False
    assert result.score == 0.0
    assert result.severity == "critical"


def test_classification_f1():
    class Model:
        def predict(self, X):
            return [1, 1, 0, 0]

    result = check_classification_metrics(
        model=Model(),
        X_test=[1, 2, 3, 4],
        y_test=[1, 0, 1, 0],
    )

    assert result.passed is True
    assert result.evidence["f1"] == 0.5


def test_classification_metrics_prediction_length_mismatch():
    class Model:
        def predict(self, X):
            return [1, 0]

    result = check_classification_metrics(
        model=Model(),
        X_test=[1, 2, 3, 4],
        y_test=[1, 0, 1, 0],
    )

    assert result.passed is False
    assert result.score == 0.0
    assert result.severity == "critical"


def test_poor_classification_metrics_are_not_awarded_full_score():
    class PoorModel:
        def predict(self, X):
            return [1, 0, 0, 1]

    result = check_classification_metrics(
        model=PoorModel(),
        X_test=[1, 2, 3, 4],
        y_test=[1, 1, 0, 0],
    )

    assert result.passed is True
    assert result.score < 100.0
    assert result.evidence["precision"] == 0.5
    assert result.evidence["recall"] == 0.5
    assert result.evidence["f1"] == 0.5


def test_perfect_classification_metrics_receive_full_score():
    class PerfectModel:
        def predict(self, X):
            return [1, 1, 0, 0]

    result = check_classification_metrics(
        model=PerfectModel(),
        X_test=[1, 2, 3, 4],
        y_test=[1, 1, 0, 0],
    )

    assert result.passed is True
    assert result.score == 100.0
    assert result.evidence["precision"] == 1.0
    assert result.evidence["recall"] == 1.0
    assert result.evidence["f1"] == 1.0


def test_classification_metrics_reject_invalid_target_labels():
    class Model:
        def predict(self, X):
            return [1, 0, 1, 0]

    result = check_classification_metrics(
        model=Model(),
        X_test=[1, 2, 3, 4],
        y_test=[1, 2, 0, 0],
    )

    assert result.passed is False
    assert result.severity == "critical"
    assert result.score == 0.0
    assert "invalid" in result.message.lower()


def test_classification_metrics_reject_invalid_prediction_labels():
    class Model:
        def predict(self, X):
            return [1, 2, 1, 0]

    result = check_classification_metrics(
        model=Model(),
        X_test=[1, 2, 3, 4],
        y_test=[1, 0, 1, 0],
    )

    assert result.passed is False
    assert result.severity == "critical"
    assert result.score == 0.0
    assert "invalid" in result.message.lower()


def test_classification_metrics_accept_numpy_integer_labels():
    class NumpyModel:
        def predict(self, X):
            return np.array([1, 1, 0, 0], dtype=np.int64)

    result = check_classification_metrics(
        model=NumpyModel(),
        X_test=[1, 2, 3, 4],
        y_test=np.array([1, 0, 1, 0], dtype=np.int64),
    )

    assert result.passed is True
    assert result.evidence["precision"] == 0.5
    assert result.evidence["recall"] == 0.5
    assert result.evidence["f1"] == 0.5
