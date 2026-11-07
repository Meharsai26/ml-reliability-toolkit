import pandas as pd

from ml_reliability.checks.cardinality import check_cardinality


def test_no_high_cardinality_columns():
    df = pd.DataFrame(
        {
            "age": [20, 20, 21, 21, 22],
            "city": ["A", "A", "B", "B", "C"],
        }
    )

    result = check_cardinality(df)

    assert result.passed is True
    assert result.score == 100.0
    assert result.severity == "info"
    assert result.evidence["high_cardinality_columns"] == []


def test_high_cardinality_column_detected():
    df = pd.DataFrame(
        {
            "customer_id": list(range(1, 11)),
            "age": [20, 20, 21, 21, 22, 22, 23, 23, 24, 24],
        }
    )

    result = check_cardinality(df)

    assert result.passed is True
    assert result.score == 50.0
    assert result.severity == "warning"
    assert "customer_id" in result.evidence["identifier_like_columns"]


def test_moderate_high_cardinality_detected():
    df = pd.DataFrame(
        {
            "feature": [1, 2, 3, 4, 5, 6, 7, 8, 8, 8],
            "category": ["A", "A", "B", "B", "B", "C", "C", "C", "C", "C"],
        }
    )

    result = check_cardinality(df)

    assert result.passed is True
    assert result.score == 70.0
    assert result.severity == "warning"
    assert "feature" in result.evidence["high_cardinality_columns"]


def test_none_input():
    result = check_cardinality(None)

    assert result.passed is False
    assert result.score == 0.0
    assert result.severity == "critical"


def test_invalid_input():
    result = check_cardinality([1, 2, 3])

    assert result.passed is False
    assert result.score == 0.0
    assert result.severity == "critical"


def test_empty_dataframe():
    df = pd.DataFrame()

    result = check_cardinality(df)

    assert result.passed is False
    assert result.score == 0.0
    assert result.severity == "critical"


def test_column_evidence():
    df = pd.DataFrame(
        {
            "customer_id": [1, 2, 3, 4],
        }
    )

    result = check_cardinality(df)

    evidence = result.evidence["columns"]["customer_id"]

    assert evidence["total_values"] == 4
    assert evidence["unique_values"] == 4
    assert evidence["uniqueness_percentage"] == 100.0
