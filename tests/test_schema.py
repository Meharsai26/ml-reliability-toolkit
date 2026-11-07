import pandas as pd

from ml_reliability.checks.schema import check_schema


def test_valid_schema():
    df = pd.DataFrame(
        {
            "age": [20, 21, 22],
            "salary": [30000, 40000, 50000],
        }
    )

    result = check_schema(df)

    assert result.passed is True
    assert result.score == 100.0
    assert result.evidence["rows"] == 3
    assert result.evidence["columns"] == 2
    assert result.evidence["column_names"] == ["age", "salary"]


def test_none_input():
    result = check_schema(None)

    assert result.passed is False
    assert result.severity == "critical"
    assert result.score == 0.0


def test_invalid_input():
    result = check_schema([1, 2, 3])

    assert result.passed is False
    assert result.severity == "critical"
    assert result.score == 0.0


def test_empty_rows():
    df = pd.DataFrame(
        {
            "age": [],
            "salary": [],
        }
    )

    result = check_schema(df)

    assert result.passed is False
    assert result.severity == "critical"
    assert result.score == 0.0
    assert result.evidence["rows"] == 0


def test_empty_columns():
    df = pd.DataFrame(index=[0, 1, 2])

    result = check_schema(df)

    assert result.passed is False
    assert result.severity == "critical"
    assert result.score == 0.0
    assert result.evidence["columns"] == 0
