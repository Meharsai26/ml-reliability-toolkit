from ml_reliability.checks.slice_analysis import (
    check_slice_analysis,
)


def test_classification_slice_accuracy():
    class Model:
        def predict(self, X):
            return [1, 1, 0, 0]

    result = check_slice_analysis(
        model=Model(),
        X_test=[1, 2, 3, 4],
        y_test=[1, 0, 0, 1],
        slices=["A", "A", "B", "B"],
        task="classification",
    )

    assert result.passed is True
    assert result.evidence["slices"]["A"]["accuracy"] == 0.5
    assert result.evidence["slices"]["B"]["accuracy"] == 0.5


def test_slice_analysis_length_mismatch():
    class Model:
        def predict(self, X):
            return [1, 1, 0, 0]

    result = check_slice_analysis(
        model=Model(),
        X_test=[1, 2, 3, 4],
        y_test=[1, 0, 0, 1],
        slices=["A", "A"],
        task="classification",
    )

    assert result.passed is False
    assert result.score == 0.0
    assert result.severity == "critical"


def test_slice_analysis_rejects_regression():
    class Model:
        def predict(self, X):
            return [1.0, 2.0, 3.0, 4.0]

    result = check_slice_analysis(
        model=Model(),
        X_test=[1, 2, 3, 4],
        y_test=[1.0, 2.0, 3.0, 4.0],
        slices=["A", "A", "B", "B"],
        task="regression",
    )

    assert result.passed is False
    assert result.score == 0.0
    assert result.severity == "critical"


def test_slice_analysis_empty_slices():
    class Model:
        def predict(self, X):
            return []

    result = check_slice_analysis(
        model=Model(),
        X_test=[],
        y_test=[],
        slices=[],
        task="classification",
    )

    assert result.passed is False
    assert result.score == 0.0
    assert result.severity == "critical"


def test_slice_analysis_reports_accuracy_gap():
    class Model:
        def predict(self, X):
            return [1, 1, 1, 0]

    result = check_slice_analysis(
        model=Model(),
        X_test=[1, 2, 3, 4],
        y_test=[1, 0, 0, 1],
        slices=["A", "A", "B", "B"],
        task="classification",
    )

    assert result.passed is True
    assert result.evidence["slice_accuracy_gap"] == 0.5
    assert result.score == 50.0


def test_slice_analysis_multiple_slices():
    class Model:
        def predict(self, X):
            return [1, 0, 1, 0, 1, 1]

    result = check_slice_analysis(
        model=Model(),
        X_test=[1, 2, 3, 4, 5, 6],
        y_test=[1, 1, 1, 0, 0, 1],
        slices=["A", "A", "B", "B", "C", "C"],
        task="classification",
    )

    assert result.passed is True
    assert result.evidence["slices"]["A"]["accuracy"] == 0.5
    assert result.evidence["slices"]["B"]["accuracy"] == 1.0
    assert result.evidence["slices"]["C"]["accuracy"] == 0.5
    assert result.evidence["slice_accuracy_gap"] == 0.5
