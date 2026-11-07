import math

from ml_reliability.checks.regression_metrics import (
    check_regression_metrics,
)


def test_regression_mae():
    class Model:
        def predict(self, X):
            return [2, 4, 6, 8]

    result = check_regression_metrics(
        model=Model(),
        X_test=[1, 2, 3, 4],
        y_test=[1, 5, 5, 10],
    )

    assert result.passed is True
    assert result.evidence["mae"] == 1.25


def test_regression_mae_perfect_predictions():
    class Model:
        def predict(self, X):
            return [1, 5, 5, 10]

    result = check_regression_metrics(
        model=Model(),
        X_test=[1, 2, 3, 4],
        y_test=[1, 5, 5, 10],
    )

    assert result.passed is True
    assert result.evidence["mae"] == 0.0


def test_regression_rmse():
    class Model:
        def predict(self, X):
            return [2, 4, 6, 8]

    result = check_regression_metrics(
        model=Model(),
        X_test=[1, 2, 3, 4],
        y_test=[1, 5, 5, 10],
    )

    assert result.passed is True
    assert result.evidence["rmse"] == 1.3229


def test_regression_metrics_prediction_length_mismatch():
    class Model:
        def predict(self, X):
            return [2, 4]

    result = check_regression_metrics(
        model=Model(),
        X_test=[1, 2, 3, 4],
        y_test=[1, 5, 5, 10],
    )

    assert result.passed is False
    assert result.score == 0.0
    assert result.severity == "critical"


def test_regression_metrics_respect_configured_mae_tolerance():
    class Model:
        def predict(self, X):
            return [2, 4, 6, 8]

    result = check_regression_metrics(
        model=Model(),
        X_test=[1, 2, 3, 4],
        y_test=[1, 5, 5, 10],
        max_acceptable_mae=1.0,
    )

    assert result.passed is False
    assert result.score == 0.0
    assert result.evidence["mae"] == 1.25


def test_regression_metrics_without_tolerance_are_unscored():
    class Model:
        def predict(self, X):
            return [2, 4, 6, 8]

    result = check_regression_metrics(
        model=Model(),
        X_test=[1, 2, 3, 4],
        y_test=[1, 5, 5, 10],
    )

    assert result.passed is True
    assert result.score is None
    assert result.evidence["mae"] == 1.25
    assert result.evidence["rmse"] == 1.3229
    assert "tolerance" in result.message.lower()


def test_regression_metrics_tolerance_boundary_passes():
    class Model:
        def predict(self, X):
            return [2, 4, 6, 8]

    result = check_regression_metrics(
        model=Model(),
        X_test=[1, 2, 3, 4],
        y_test=[1, 5, 5, 10],
        max_acceptable_mae=1.25,
    )

    assert result.passed is True
    assert result.score == 100.0
    assert result.evidence["mae"] == 1.25


def test_regression_metrics_zero_tolerance_accepts_perfect_predictions():
    class Model:
        def predict(self, X):
            return [1, 5, 5, 10]

    result = check_regression_metrics(
        model=Model(),
        X_test=[1, 2, 3, 4],
        y_test=[1, 5, 5, 10],
        max_acceptable_mae=0.0,
    )

    assert result.passed is True
    assert result.score == 100.0
    assert result.evidence["mae"] == 0.0


def test_regression_metrics_calculate_proportional_score():
    class Model:
        def predict(self, X):
            return [2, 4, 6, 8]

    result = check_regression_metrics(
        model=Model(),
        X_test=[1, 2, 3, 4],
        y_test=[1, 5, 5, 10],
        max_acceptable_mae=2.0,
        max_scoring_mae=4.0,
    )

    assert result.passed is True
    assert result.score == 68.75
    assert result.evidence["mae"] == 1.25


def test_regression_metrics_score_is_independent_of_pass_fail():
    class Model:
        def predict(self, X):
            return [2, 4, 6, 8]

    result = check_regression_metrics(
        model=Model(),
        X_test=[1, 2, 3, 4],
        y_test=[1, 5, 5, 10],
        max_acceptable_mae=1.0,
        max_scoring_mae=4.0,
    )

    assert result.passed is False
    assert result.score == 68.75
    assert result.evidence["mae"] == 1.25


def test_regression_metrics_reject_zero_scoring_scale():
    class Model:
        def predict(self, X):
            return [1, 2, 3]

    result = check_regression_metrics(
        model=Model(),
        X_test=[1, 2, 3],
        y_test=[1, 2, 3],
        max_scoring_mae=0.0,
    )

    assert result.passed is False
    assert result.score == 0.0
    assert result.severity == "critical"
    assert "max_scoring_mae" in result.message


def test_regression_metrics_reject_negative_scoring_scale():
    class Model:
        def predict(self, X):
            return [1, 2, 3]

    result = check_regression_metrics(
        model=Model(),
        X_test=[1, 2, 3],
        y_test=[1, 2, 3],
        max_scoring_mae=-1.0,
    )

    assert result.passed is False
    assert result.score == 0.0
    assert result.severity == "critical"
    assert "max_scoring_mae" in result.message


def test_regression_metrics_reject_nan_prediction():
    class Model:
        def predict(self, X):
            return [1.0, float("nan"), 3.0]

    result = check_regression_metrics(
        model=Model(),
        X_test=[1, 2, 3],
        y_test=[1.0, 2.0, 3.0],
    )

    assert result.passed is False
    assert result.severity == "critical"
    assert result.score == 0.0
    assert "non-finite" in result.message.lower()


def test_regression_metrics_reject_infinite_prediction():
    class Model:
        def predict(self, X):
            return [1.0, float("inf"), 3.0]

    result = check_regression_metrics(
        model=Model(),
        X_test=[1, 2, 3],
        y_test=[1.0, 2.0, 3.0],
    )

    assert result.passed is False
    assert result.severity == "critical"
    assert result.score == 0.0
    assert "non-finite" in result.message.lower()


def test_regression_metrics_reject_nan_target():
    class Model:
        def predict(self, X):
            return [1.0, 2.0, 3.0]

    result = check_regression_metrics(
        model=Model(),
        X_test=[1, 2, 3],
        y_test=[1.0, float("nan"), 3.0],
    )

    assert result.passed is False
    assert result.severity == "critical"
    assert result.score == 0.0
    assert "non-finite" in result.message.lower()


def test_regression_metrics_reject_infinite_target():
    class Model:
        def predict(self, X):
            return [1.0, 2.0, 3.0]

    result = check_regression_metrics(
        model=Model(),
        X_test=[1, 2, 3],
        y_test=[1.0, float("-inf"), 3.0],
    )

    assert result.passed is False
    assert result.severity == "critical"
    assert result.score == 0.0
    assert "non-finite" in result.message.lower()


def test_regression_metrics_reject_overflow_from_finite_inputs():
    class Model:
        def predict(self, X):
            return [1e308]

    result = check_regression_metrics(
        model=Model(),
        X_test=[1],
        y_test=[-1e308],
    )

    assert result.passed is False
    assert result.severity == "critical"
    assert result.score == 0.0
    assert "finite" in result.message.lower()


def test_regression_metrics_score_remains_finite_and_bounded():
    class Model:
        def predict(self, X):
            return [1e154, 1e154]

    result = check_regression_metrics(
        model=Model(),
        X_test=[1, 2],
        y_test=[0.0, 0.0],
        max_scoring_mae=1e308,
    )

    assert result.score is not None
    assert math.isfinite(result.score)
    assert 0.0 <= result.score <= 100.0
