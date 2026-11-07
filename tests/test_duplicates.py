import pandas as pd

from ml_reliability.checks.duplicates import check_duplicates


def test_no_duplicate_rows():
    df = pd.DataFrame(
        {
            "age": [20, 21, 22, 23],
            "salary": [30000, 40000, 50000, 60000],
        }
    )

    result = check_duplicates(df)

    assert result.passed is True
    assert result.score == 100.0
    assert result.evidence["total_rows"] == 4
    assert result.evidence["duplicate_rows"] == 0
    assert result.evidence["duplicate_percentage"] == 0.0


def test_duplicate_rows_detected():
    df = pd.DataFrame(
        {
            "age": [20, 21, 20, 23],
            "salary": [30000, 40000, 30000, 50000],
        }
    )

    result = check_duplicates(df)

    assert result.passed is False
    assert result.evidence["total_rows"] == 4
    assert result.evidence["duplicate_rows"] == 1
    assert result.evidence["duplicate_percentage"] == 25.0
    assert result.severity == "critical"
    assert result.score == 0.0


def test_none_input():
    result = check_duplicates(None)

    assert result.passed is False
    assert result.severity == "critical"
    assert result.score == 0.0


def test_invalid_input():
    result = check_duplicates([1, 2, 3])

    assert result.passed is False
    assert result.severity == "critical"
    assert result.score == 0.0


def test_empty_dataframe():
    df = pd.DataFrame()

    result = check_duplicates(df)

    assert result.passed is True
    assert result.score == 100.0
    assert result.evidence["total_rows"] == 0
    assert result.evidence["duplicate_rows"] == 0
