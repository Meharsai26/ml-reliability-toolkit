from ml_reliability.checks.calibration import (
    check_calibration,
)


def test_classification_calibration():
    class Model:
        def predict_proba(self, X):
            return [
                [0.1, 0.9],
                [0.2, 0.8],
                [0.8, 0.2],
                [0.9, 0.1],
            ]

    result = check_calibration(
        model=Model(),
        X_test=[1, 2, 3, 4],
        y_test=[1, 1, 0, 0],
    )

    assert result.passed is True
    assert result.evidence["calibration_error"] == 0.15
    assert result.score == 85.0


def test_calibration_rejects_non_binary_probabilities():
    class Model:
        def predict_proba(self, X):
            return [
                [0.1, 0.2, 0.7],
                [0.2, 0.3, 0.5],
            ]

    result = check_calibration(
        model=Model(),
        X_test=[1, 2],
        y_test=[1, 0],
    )

    assert result.passed is False
    assert result.score == 0.0
    assert result.severity == "critical"


def test_calibration_prediction_length_mismatch():
    class Model:
        def predict_proba(self, X):
            return [
                [0.1, 0.9],
                [0.2, 0.8],
            ]

    result = check_calibration(
        model=Model(),
        X_test=[1, 2, 3, 4],
        y_test=[1, 1, 0, 0],
    )

    assert result.passed is False
    assert result.score == 0.0
    assert result.severity == "critical"


def test_calibration_rejects_invalid_probability_values():
    class Model:
        def predict_proba(self, X):
            return [
                [-0.1, 1.1],
                [0.2, 0.8],
            ]

    result = check_calibration(
        model=Model(),
        X_test=[1, 2],
        y_test=[1, 0],
    )

    assert result.passed is False
    assert result.score == 0.0
    assert result.severity == "critical"
