import pandas as pd
import pytest

from ml_reliability.checks.stress.categorical import (
    inject_categorical_corruption,
)


def test_categorical_corruption_does_not_modify_original():
    X = pd.DataFrame(
        {
            "color": ["red", "blue", "green", "red"],
            "age": [20, 25, 30, 35],
        }
    )

    original = X.copy()

    stressed = inject_categorical_corruption(
        X,
        columns=["color"],
        rate=0.5,
        random_state=42,
    )

    pd.testing.assert_frame_equal(X, original)
    assert stressed is not X


def test_categorical_corruption_preserves_shape():
    X = pd.DataFrame(
        {
            "color": ["red", "blue", "green", "red"],
            "city": ["hyd", "del", "mum", "hyd"],
        }
    )

    stressed = inject_categorical_corruption(
        X,
        columns=["color"],
        rate=0.5,
        random_state=42,
    )

    assert stressed.shape == X.shape
    assert list(stressed.columns) == list(X.columns)


def test_categorical_corruption_changes_selected_values():
    X = pd.DataFrame(
        {
            "color": ["red", "blue", "green", "red"],
            "age": [20, 25, 30, 35],
        }
    )

    stressed = inject_categorical_corruption(
        X,
        columns=["color"],
        rate=0.5,
        random_state=42,
    )

    assert (stressed["color"] != X["color"]).sum() == 2
    assert stressed["age"].equals(X["age"])


def test_categorical_corruption_uses_valid_values():
    X = pd.DataFrame(
        {
            "color": ["red", "blue", "green", "red"],
        }
    )

    stressed = inject_categorical_corruption(
        X,
        columns=["color"],
        rate=0.5,
        random_state=42,
    )

    original_values = set(X["color"].unique())
    stressed_values = set(stressed["color"].unique())

    assert stressed_values.issubset(original_values)


def test_categorical_corruption_is_reproducible():
    X = pd.DataFrame(
        {
            "color": ["red", "blue", "green", "red"],
            "city": ["hyd", "del", "mum", "hyd"],
        }
    )

    first = inject_categorical_corruption(
        X,
        columns=["color", "city"],
        rate=0.5,
        random_state=42,
    )

    second = inject_categorical_corruption(
        X,
        columns=["color", "city"],
        rate=0.5,
        random_state=42,
    )

    pd.testing.assert_frame_equal(first, second)


def test_zero_rate_returns_unchanged_copy():
    X = pd.DataFrame(
        {
            "color": ["red", "blue", "green"],
        }
    )

    stressed = inject_categorical_corruption(
        X,
        columns=["color"],
        rate=0.0,
    )

    pd.testing.assert_frame_equal(stressed, X)
    assert stressed is not X


def test_unknown_column_raises_error():
    X = pd.DataFrame(
        {
            "color": ["red", "blue", "green"],
        }
    )

    with pytest.raises(ValueError, match="Columns not found"):
        inject_categorical_corruption(
            X,
            columns=["unknown"],
            rate=0.5,
        )


def test_single_category_column_is_unchanged():
    X = pd.DataFrame(
        {
            "color": ["red", "red", "red", "red"],
        }
    )

    stressed = inject_categorical_corruption(
        X,
        columns=["color"],
        rate=0.5,
        random_state=42,
    )

    pd.testing.assert_frame_equal(stressed, X)


def test_categorical_corruption_evaluation_classification():
    from ml_reliability.checks.stress.categorical import (
        evaluate_categorical_corruption_stress,
    )

    class Model:
        def predict(self, X):
            return [1 if color == "red" else 0 for color in X["color"]]

    X = pd.DataFrame(
        {
            "color": ["red", "blue", "red", "blue"],
        }
    )

    y = [1, 0, 1, 0]

    result = evaluate_categorical_corruption_stress(
        model=Model(),
        X_test=X,
        y_test=y,
        task="classification",
        columns=["color"],
        rate=0.5,
        random_state=42,
    )

    assert result.stress_type == "categorical_corruption"
    assert result.baseline_metric == 1.0
    assert 0.0 <= result.stressed_metric <= 1.0
    assert result.degradation >= 0.0
    assert result.passed is True


def test_categorical_corruption_evaluation_regression():
    from ml_reliability.checks.stress.categorical import (
        evaluate_categorical_corruption_stress,
    )

    class Model:
        def predict(self, X):
            return [10.0 if category == "A" else 20.0 for category in X["category"]]

    X = pd.DataFrame(
        {
            "category": ["A", "B", "A", "B"],
        }
    )

    y = [10.0, 20.0, 10.0, 20.0]

    result = evaluate_categorical_corruption_stress(
        model=Model(),
        X_test=X,
        y_test=y,
        task="regression",
        columns=["category"],
        rate=0.5,
        random_state=42,
    )

    assert result.stress_type == "categorical_corruption"
    assert result.baseline_metric == 1.0
    assert result.stressed_metric <= 1.0
    assert result.degradation >= 0.0
    assert result.passed is True


def test_categorical_corruption_stress_prediction_failure():
    from ml_reliability.checks.stress.categorical import (
        evaluate_categorical_corruption_stress,
    )

    class Model:
        def predict(self, X):
            if (X["color"] != ["red", "blue"]).any():
                raise RuntimeError("Unknown category")
            return [1, 0]

    X = pd.DataFrame(
        {
            "color": ["red", "blue"],
        }
    )

    y = [1, 0]

    result = evaluate_categorical_corruption_stress(
        model=Model(),
        X_test=X,
        y_test=y,
        task="classification",
        columns=["color"],
        rate=1.0,
        random_state=42,
    )

    assert result.passed is False
    assert result.evidence["stress_prediction_failed"] is True
    assert result.degradation == float("inf")


def test_categorical_corruption_baseline_failure():
    from ml_reliability.checks.stress.categorical import (
        evaluate_categorical_corruption_stress,
    )

    class Model:
        def predict(self, X):
            raise RuntimeError("Baseline prediction failed")

    X = pd.DataFrame(
        {
            "color": ["red", "blue"],
        }
    )

    y = [1, 0]

    with pytest.raises(
        ValueError,
        match="Baseline model performance could not be calculated",
    ):
        evaluate_categorical_corruption_stress(
            model=Model(),
            X_test=X,
            y_test=y,
            task="classification",
            columns=["color"],
            rate=0.5,
        )


@pytest.mark.parametrize("columns", ["color", 123, None])
def test_categorical_corruption_rejects_invalid_columns_argument(columns):
    X = pd.DataFrame({"color": ["red", "blue", "green"]})

    with pytest.raises(
        (TypeError, ValueError),
        match="columns must be a sequence of column names",
    ):
        inject_categorical_corruption(X, columns=columns, rate=0.5)


def test_categorical_corruption_rejects_non_string_column_names():
    X = pd.DataFrame({"color": ["red", "blue", "green"]})

    with pytest.raises(
        ValueError,
        match="columns must contain only column names",
    ):
        inject_categorical_corruption(
            X,
            columns=["color", 123],
            rate=0.5,
        )


def test_categorical_corruption_preserves_existing_missing_values():
    X = pd.DataFrame({"color": ["red", None, "blue", "green", None]})

    stressed = inject_categorical_corruption(
        X,
        columns=["color"],
        rate=1.0,
        random_state=42,
    )

    assert stressed["color"].isna().equals(X["color"].isna())
    assert stressed["color"].dropna().isin(X["color"].dropna().unique()).all()
    assert X["color"].isna().sum() == 2


def test_categorical_corruption_handles_non_default_index():
    X = pd.DataFrame(
        {"color": ["red", "blue", "green", "red"]},
        index=[10, 20, 30, 40],
    )

    stressed = inject_categorical_corruption(
        X,
        columns=["color"],
        rate=0.5,
        random_state=42,
    )

    assert (stressed["color"] != X["color"]).sum() == 2
    assert stressed.index.equals(X.index)


def test_categorical_corruption_handles_duplicate_index():
    X = pd.DataFrame(
        {"color": ["red", "blue", "green", "red"]},
        index=[1, 1, 2, 2],
    )

    stressed = inject_categorical_corruption(
        X,
        columns=["color"],
        rate=0.5,
        random_state=42,
    )

    assert (stressed["color"] != X["color"]).sum() == 2
    assert stressed.index.equals(X.index)


@pytest.mark.parametrize(
    "random_state",
    ["invalid", None, -1, -1.5, True],
)
def test_inject_categorical_corruption_rejects_invalid_random_state(
    random_state,
):
    X = pd.DataFrame({"category": ["A", "B", "C", "A"]})

    with pytest.raises((TypeError, ValueError)):
        inject_categorical_corruption(
            X,
            columns=["category"],
            rate=0.5,
            random_state=random_state,
        )


@pytest.mark.parametrize(
    "X, columns, rate",
    [
        (pd.DataFrame({"category": ["A", "B"]}), [], 0.5),
        (pd.DataFrame(), ["category"], 0.5),
        (pd.DataFrame({"category": ["A", "B"]}), ["category"], 0.0),
    ],
)
def test_inject_categorical_corruption_validates_random_state_on_no_op(
    X, columns, rate
):
    with pytest.raises(ValueError, match="random_state"):
        inject_categorical_corruption(
            X,
            columns=columns,
            rate=rate,
            random_state=None,
        )


@pytest.mark.parametrize(
    "X, columns, rate",
    [
        (pd.DataFrame({"color": ["red", "blue"]}), ["unknown"], 0.0),
        (pd.DataFrame(), ["unknown"], 0.5),
    ],
)
def test_categorical_corruption_validates_columns_before_no_op(X, columns, rate):
    with pytest.raises(ValueError, match="Columns not found"):
        inject_categorical_corruption(
            X,
            columns=columns,
            rate=rate,
        )


def test_categorical_corruption_handles_mixed_type_categories():
    X = pd.DataFrame({"category": ["A", 1, "B", 2, "A", 1]})

    stressed = inject_categorical_corruption(
        X,
        columns=["category"],
        rate=1.0,
        random_state=42,
    )

    assert stressed["category"].isin(X["category"]).all()
    assert (stressed["category"] != X["category"]).all()
    pd.testing.assert_frame_equal(
        X,
        pd.DataFrame({"category": ["A", 1, "B", 2, "A", 1]}),
    )


def test_categorical_corruption_rejects_duplicate_column_names():
    X = pd.DataFrame(
        [["red", "blue"], ["blue", "red"], ["green", "green"]],
        columns=["color", "color"],
    )

    with pytest.raises(ValueError, match="duplicate column names"):
        inject_categorical_corruption(
            X,
            columns=["color"],
            rate=0.5,
            random_state=42,
        )


def test_categorical_corruption_small_positive_rate_changes_at_least_one_value():
    X = pd.DataFrame({"color": ["red", "blue", "green"]})

    stressed = inject_categorical_corruption(
        X,
        columns=["color"],
        rate=0.1,
        random_state=42,
    )

    changed_count = (stressed["color"] != X["color"]).sum()

    assert changed_count >= 1
    assert changed_count <= len(X)
    assert stressed["color"].isin(X["color"]).all()


def test_categorical_corruption_rejects_duplicate_requested_columns():
    X = pd.DataFrame({"color": ["red", "blue", "green"]})

    with pytest.raises(
        ValueError,
        match="duplicate requested column names",
    ):
        inject_categorical_corruption(
            X,
            columns=["color", "color"],
            rate=0.5,
            random_state=42,
        )
