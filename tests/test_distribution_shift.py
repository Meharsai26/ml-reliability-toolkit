import pandas as pd

from ml_reliability.checks.distribution_shift import check_distribution_shift


def test_no_distribution_shift():
    train = pd.DataFrame(
        {
            "age": [20, 21, 22, 23, 24],
            "salary": [30000, 40000, 50000, 60000, 70000],
        }
    )

    test = pd.DataFrame(
        {
            "age": [20, 21, 22, 23, 24],
            "salary": [30000, 40000, 50000, 60000, 70000],
        }
    )

    result = check_distribution_shift(train, test)

    assert result.passed is True
    assert result.score == 100.0
    assert result.severity == "info"
    assert result.evidence["significant_shift_features"] == []


def test_moderate_distribution_shift():
    train = pd.DataFrame(
        {
            "age": [20, 21, 22, 23, 24],
        }
    )

    test = pd.DataFrame(
        {
            "age": [23, 24, 25, 26, 27],
        }
    )

    result = check_distribution_shift(train, test)

    assert result.passed is True
    assert result.score == 80.0
    assert result.severity == "warning"
    assert "age" in result.evidence["moderate_shift_features"]


def test_significant_distribution_shift():
    train = pd.DataFrame(
        {
            "age": [20, 21, 22, 23, 24],
        }
    )

    test = pd.DataFrame(
        {
            "age": [30, 31, 32, 33, 34],
        }
    )

    result = check_distribution_shift(train, test)

    assert result.passed is True
    assert result.score == 50.0
    assert result.severity == "warning"
    assert "age" in result.evidence["significant_shift_features"]


def test_zero_training_mean():
    train = pd.DataFrame(
        {
            "feature": [-1, 0, 1],
        }
    )

    test = pd.DataFrame(
        {
            "feature": [1, 1, 1],
        }
    )

    result = check_distribution_shift(train, test)

    assert result.passed is True
    assert result.score == 50.0
    assert (
        result.evidence["feature_statistics"]["feature"]["relative_shift_percentage"]
        is None
    )


def test_missing_train_data():
    test = pd.DataFrame({"age": [20, 21, 22]})

    result = check_distribution_shift(None, test)

    assert result.passed is False
    assert result.score == 0.0
    assert result.severity == "critical"


def test_missing_test_data():
    train = pd.DataFrame({"age": [20, 21, 22]})

    result = check_distribution_shift(train, None)

    assert result.passed is False
    assert result.score == 0.0
    assert result.severity == "critical"


def test_invalid_input():
    train = [1, 2, 3]
    test = pd.DataFrame({"age": [20, 21, 22]})

    result = check_distribution_shift(train, test)

    assert result.passed is False
    assert result.score == 0.0
    assert result.severity == "critical"


def test_empty_dataset():
    train = pd.DataFrame()
    test = pd.DataFrame({"age": [20, 21, 22]})

    result = check_distribution_shift(train, test)

    assert result.passed is False
    assert result.score == 0.0
    assert result.severity == "critical"


def test_categorical_only_data():
    train = pd.DataFrame(
        {
            "city": ["A", "B", "A"],
        }
    )

    test = pd.DataFrame(
        {
            "city": ["A", "C", "C"],
        }
    )

    result = check_distribution_shift(train, test)

    assert result.passed is True
    assert result.score == 100.0
    assert result.severity == "info"
    assert result.evidence["numeric_features_compared"] == 0
