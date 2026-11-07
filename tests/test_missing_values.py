import pandas as pd

from ml_reliability.checks.missing_values import check_missing_values


def test_no_missing_values():
    df = pd.DataFrame(
        {
            "age": [20, 21, 22, 23],
            "salary": [30000, 40000, 50000, 60000],
        }
    )

    result = check_missing_values(df)

    assert result.passed is True
    assert result.score == 100.0
    assert result.evidence["total_missing"] == 0


def test_missing_values_detected():
    df = pd.DataFrame(
        {
            "age": [20, 21, None, 23],
            "salary": [30000, None, 40000, 50000],
        }
    )

    result = check_missing_values(df)

    assert result.passed is False
    assert result.evidence["total_missing"] == 2
    assert result.evidence["missing_percentage"] == 25.0
    assert result.evidence["affected_columns"]["age"] == 1
    assert result.evidence["affected_columns"]["salary"] == 1


def test_none_input():
    result = check_missing_values(None)

    assert result.passed is False
    assert result.severity == "critical"
    assert result.score == 0.0


def test_invalid_input():
    result = check_missing_values([1, 2, 3])

    assert result.passed is False
    assert result.severity == "critical"
    assert result.score == 0.0
