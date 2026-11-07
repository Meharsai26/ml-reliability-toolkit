from ml_reliability.checks.target_distribution import check_target_distribution


def test_balanced_classification_target():
    y = [0, 0, 0, 0, 1, 1, 1, 1]

    result = check_target_distribution(
        y,
        task_type="classification",
    )

    assert result.passed is True
    assert result.score == 100.0
    assert result.severity == "info"
    assert result.evidence["task_type"] == "classification"
    assert result.evidence["minority_percentage"] == 50.0


def test_moderately_imbalanced_classification_target():
    y = [0] * 95 + [1] * 5

    result = check_target_distribution(
        y,
        task_type="classification",
    )

    assert result.passed is True
    assert result.score == 70.0
    assert result.severity == "warning"
    assert result.evidence["minority_percentage"] == 5.0


def test_severely_imbalanced_classification_target():
    y = [0] * 99 + [1]

    result = check_target_distribution(
        y,
        task_type="classification",
    )

    assert result.passed is False
    assert result.score == 30.0
    assert result.severity == "critical"
    assert result.evidence["minority_percentage"] == 1.0


def test_regression_target():
    y = [10.2, 12.5, 11.8, 15.4, 13.1]

    result = check_target_distribution(
        y,
        task_type="regression",
    )

    assert result.passed is True
    assert result.score == 100.0
    assert result.severity == "info"
    assert result.evidence["task_type"] == "regression"
    assert result.evidence["total_samples"] == 5


def test_missing_target():
    result = check_target_distribution(None)

    assert result.passed is False
    assert result.score == 0.0
    assert result.severity == "critical"


def test_empty_target():
    result = check_target_distribution([])

    assert result.passed is False
    assert result.score == 0.0
    assert result.severity == "critical"


def test_target_with_missing_values():
    y = [0, 1, None, 1]

    result = check_target_distribution(
        y,
        task_type="classification",
    )

    assert result.passed is False
    assert result.score == 0.0
    assert result.severity == "critical"
    assert result.evidence["missing_values"] == 1


def test_invalid_task_type():
    y = [0, 1, 0, 1]

    result = check_target_distribution(
        y,
        task_type="invalid",
    )

    assert result.passed is False
    assert result.score == 0.0
    assert result.severity == "critical"
