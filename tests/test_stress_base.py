from ml_reliability.checks.stress import (
    StressTestOutcome,
    calculate_degradation,
    validate_rate,
)


def test_stress_test_outcome_creation():
    outcome = StressTestOutcome(
        stress_type="missing_values",
        baseline_metric=0.90,
        stressed_metric=0.82,
        degradation=0.08,
        passed=True,
    )

    assert outcome.stress_type == "missing_values"
    assert outcome.baseline_metric == 0.90
    assert outcome.stressed_metric == 0.82
    assert outcome.degradation == 0.08
    assert outcome.passed is True


def test_stress_test_outcome_default_evidence():
    outcome = StressTestOutcome(
        stress_type="noise",
        baseline_metric=0.90,
        stressed_metric=0.85,
        degradation=0.05,
        passed=True,
    )

    assert outcome.evidence == {}


def test_stress_test_outcomes_have_independent_evidence():
    first = StressTestOutcome(
        stress_type="noise",
        baseline_metric=0.90,
        stressed_metric=0.85,
        degradation=0.05,
        passed=True,
    )

    second = StressTestOutcome(
        stress_type="dropout",
        baseline_metric=0.90,
        stressed_metric=0.70,
        degradation=0.20,
        passed=False,
    )

    first.evidence["feature"] = "age"

    assert first.evidence == {"feature": "age"}
    assert second.evidence == {}


def test_calculate_degradation():
    degradation = calculate_degradation(
        baseline_metric=0.90,
        stressed_metric=0.82,
    )

    assert degradation == 0.08


def test_no_degradation():
    degradation = calculate_degradation(
        baseline_metric=0.90,
        stressed_metric=0.90,
    )

    assert degradation == 0.0


def test_negative_degradation_when_stress_improves_metric():
    degradation = calculate_degradation(
        baseline_metric=0.80,
        stressed_metric=0.85,
    )

    assert degradation == -0.05


def test_validate_rate_accepts_zero():
    assert validate_rate(0.0) == 0.0


def test_validate_rate_accepts_one():
    assert validate_rate(1.0) == 1.0


def test_validate_rate_accepts_valid_rate():
    assert validate_rate(0.25) == 0.25


def test_validate_rate_rejects_negative_rate():
    import pytest

    with pytest.raises(ValueError, match="rate must be between 0.0 and 1.0"):
        validate_rate(-0.1)


def test_validate_rate_rejects_rate_above_one():
    import pytest

    with pytest.raises(ValueError, match="rate must be between 0.0 and 1.0"):
        validate_rate(1.1)


import math

import pytest


@pytest.mark.parametrize(
    "rate",
    [
        float("nan"),
        float("inf"),
        float("-inf"),
    ],
)
def test_validate_rate_rejects_non_finite_values(rate):
    with pytest.raises(
        ValueError,
        match="rate must be between 0.0 and 1.0",
    ):
        validate_rate(rate)


@pytest.mark.parametrize(
    "rate",
    [
        True,
        False,
        "0.5",
        None,
    ],
)
def test_validate_rate_rejects_invalid_types(rate):
    with pytest.raises(
        ValueError,
        match="rate must be between 0.0 and 1.0",
    ):
        validate_rate(rate)


@pytest.mark.parametrize(
    ("baseline_metric", "stressed_metric"),
    [
        (float("nan"), 0.8),
        (0.9, float("inf")),
        (float("-inf"), 0.8),
        (True, 0.8),
        (0.9, False),
        ("0.9", 0.8),
        (0.9, None),
    ],
)
def test_calculate_degradation_rejects_invalid_metrics(
    baseline_metric,
    stressed_metric,
):
    with pytest.raises(ValueError):
        calculate_degradation(baseline_metric, stressed_metric)


@pytest.mark.parametrize(
    "metric_field",
    [
        "baseline_metric",
        "stressed_metric",
        "degradation",
    ],
)
@pytest.mark.parametrize(
    "invalid_value",
    [
        float("nan"),
        float("inf"),
        float("-inf"),
        True,
        "invalid",
        None,
    ],
)
def test_stress_test_outcome_preserves_invalid_metrics(
    metric_field,
    invalid_value,
):
    values = {
        "stress_type": "noise",
        "baseline_metric": 0.90,
        "stressed_metric": 0.80,
        "degradation": 0.10,
        "passed": False,
    }
    values[metric_field] = invalid_value

    outcome = StressTestOutcome(**values)

    actual_value = getattr(outcome, metric_field)

    if isinstance(invalid_value, float) and math.isnan(invalid_value):
        assert math.isnan(actual_value)
    else:
        assert actual_value == invalid_value


import numpy as np


@pytest.mark.parametrize("rate", [np.float64(0.5), np.int64(1)])
def test_validate_rate_accepts_numpy_numeric_types(rate):
    assert validate_rate(rate) == rate


def test_calculate_degradation_accepts_numpy_numeric_types():
    result = calculate_degradation(
        baseline_metric=np.float64(0.9),
        stressed_metric=np.float64(0.8),
    )

    assert result == pytest.approx(0.1)


def test_calculate_degradation_accepts_numpy_integer_metrics():
    result = calculate_degradation(
        baseline_metric=np.int64(1),
        stressed_metric=np.int64(0),
    )

    assert result == 1.0


def test_validate_rate_rejects_extremely_large_integer():
    with pytest.raises(
        ValueError,
        match="rate must be between 0.0 and 1.0",
    ):
        validate_rate(10**1000)


@pytest.mark.parametrize(
    ("baseline_metric", "stressed_metric"),
    [
        (10**1000, 0.8),
        (0.9, 10**1000),
    ],
)
def test_calculate_degradation_rejects_extremely_large_integer_metrics(
    baseline_metric,
    stressed_metric,
):
    with pytest.raises(
        ValueError,
        match="baseline_metric and stressed_metric must be finite numbers",
    ):
        calculate_degradation(baseline_metric, stressed_metric)


def test_calculate_degradation_supports_negative_r2_values():
    degradation = calculate_degradation(
        baseline_metric=-0.25,
        stressed_metric=-0.75,
    )

    assert degradation == 0.5


def test_calculate_degradation_rejects_overflow_result():
    with pytest.raises(
        ValueError,
        match="degradation must be finite",
    ):
        calculate_degradation(
            baseline_metric=1e308,
            stressed_metric=-1e308,
        )


from ml_reliability.checks import stress


def test_stress_package_all_exports_include_missing_value_evaluator():
    assert "evaluate_missing_value_stress" in stress.__all__
    assert hasattr(stress, "evaluate_missing_value_stress")
