import pandas as pd
import pandas.testing as pdt
import pytest

from ml_reliability.checks.stress import apply_feature_dropout


def test_feature_dropout_does_not_modify_original_data():
    X = pd.DataFrame(
        {
            "age": [20, 25, 30],
            "income": [30000, 40000, 50000],
            "city": ["A", "B", "C"],
        }
    )

    original = X.copy()

    stressed = apply_feature_dropout(
        X,
        features=["income"],
    )

    pdt.assert_frame_equal(X, original)
    assert stressed is not X


def test_feature_dropout_masks_selected_features():
    X = pd.DataFrame(
        {
            "age": [20, 25, 30],
            "income": [30000, 40000, 50000],
            "city": ["A", "B", "C"],
        }
    )

    stressed = apply_feature_dropout(
        X,
        features=["income"],
    )

    assert stressed["income"].isna().all()
    assert stressed["age"].tolist() == X["age"].tolist()
    assert stressed["city"].tolist() == X["city"].tolist()


def test_feature_dropout_can_mask_multiple_features():
    X = pd.DataFrame(
        {
            "age": [20, 25, 30],
            "income": [30000, 40000, 50000],
            "experience": [1, 3, 5],
        }
    )

    stressed = apply_feature_dropout(
        X,
        features=["income", "experience"],
    )

    assert stressed["income"].isna().all()
    assert stressed["experience"].isna().all()
    assert stressed["age"].tolist() == X["age"].tolist()


def test_feature_dropout_with_empty_feature_list():
    X = pd.DataFrame(
        {
            "age": [20, 25, 30],
            "income": [30000, 40000, 50000],
        }
    )

    stressed = apply_feature_dropout(
        X,
        features=[],
    )

    pdt.assert_frame_equal(stressed, X)
    assert stressed is not X


def test_feature_dropout_rejects_unknown_features():
    X = pd.DataFrame(
        {
            "age": [20, 25, 30],
            "income": [30000, 40000, 50000],
        }
    )

    with pytest.raises(
        ValueError,
        match="Features not found in X",
    ):
        apply_feature_dropout(
            X,
            features=["unknown_feature"],
        )


def test_feature_dropout_preserves_dataframe_shape():
    X = pd.DataFrame(
        {
            "age": [20, 25, 30],
            "income": [30000, 40000, 50000],
        }
    )

    stressed = apply_feature_dropout(
        X,
        features=["income"],
    )

    assert stressed.shape == X.shape
    assert list(stressed.columns) == list(X.columns)


def test_feature_dropout_evaluation_classification():
    from ml_reliability.checks.stress.feature_dropout import (
        evaluate_feature_dropout_stress,
    )

    class Model:
        def predict(self, X):
            return [1 if value > 5 else 0 for value in X["feature"]]

    X = pd.DataFrame(
        {
            "feature": [10.0, 1.0, 20.0, 2.0],
            "other": [1.0, 1.0, 1.0, 1.0],
        }
    )

    y = [1, 0, 1, 0]

    result = evaluate_feature_dropout_stress(
        model=Model(),
        X_test=X,
        y_test=y,
        task="classification",
        features=["feature"],
    )

    assert result.stress_type == "feature_dropout"
    assert result.baseline_metric == 1.0
    assert 0.0 <= result.stressed_metric <= 1.0
    assert result.degradation >= 0.0
    assert result.passed is True


def test_feature_dropout_evaluation_regression():
    from ml_reliability.checks.stress.feature_dropout import (
        evaluate_feature_dropout_stress,
    )

    class Model:
        def predict(self, X):
            return X["feature"].fillna(0.0).tolist()

    X = pd.DataFrame(
        {
            "feature": [1.0, 2.0, 3.0, 4.0],
            "other": [0.0, 0.0, 0.0, 0.0],
        }
    )

    y = [1.0, 2.0, 3.0, 4.0]

    result = evaluate_feature_dropout_stress(
        model=Model(),
        X_test=X,
        y_test=y,
        task="regression",
        features=["feature"],
    )

    assert result.stress_type == "feature_dropout"
    assert result.baseline_metric == 1.0
    assert result.stressed_metric <= 1.0
    assert result.degradation >= 0.0
    assert result.passed is True


def test_feature_dropout_stress_prediction_failure():
    from ml_reliability.checks.stress.feature_dropout import (
        evaluate_feature_dropout_stress,
    )

    class Model:
        def predict(self, X):
            if X["feature"].isna().any():
                raise RuntimeError("Missing feature")

            return [1, 0]

    X = pd.DataFrame(
        {
            "feature": [1.0, 2.0],
        }
    )

    y = [1, 0]

    result = evaluate_feature_dropout_stress(
        model=Model(),
        X_test=X,
        y_test=y,
        task="classification",
        features=["feature"],
    )

    assert result.passed is False
    assert result.evidence["stress_prediction_failed"] is True
    assert result.degradation == float("inf")


def test_feature_dropout_baseline_failure():
    from ml_reliability.checks.stress.feature_dropout import (
        evaluate_feature_dropout_stress,
    )

    class Model:
        def predict(self, X):
            raise RuntimeError("Baseline prediction failed")

    X = pd.DataFrame(
        {
            "feature": [1.0, 2.0],
        }
    )

    y = [1, 0]

    with pytest.raises(
        ValueError,
        match="Baseline model performance could not be calculated",
    ):
        evaluate_feature_dropout_stress(
            model=Model(),
            X_test=X,
            y_test=y,
            task="classification",
            features=["feature"],
        )


@pytest.mark.parametrize(
    "features",
    [
        "income",
        123,
        None,
    ],
)
def test_feature_dropout_rejects_invalid_features_argument(features):
    X = pd.DataFrame(
        {
            "age": [20, 25, 30],
            "income": [30000, 40000, 50000],
        }
    )

    with pytest.raises(
        (TypeError, ValueError),
        match="features must be a sequence of column names",
    ):
        apply_feature_dropout(X, features=features)


def test_feature_dropout_rejects_non_string_feature_names():
    X = pd.DataFrame(
        {
            "age": [20, 25, 30],
            "income": [30000, 40000, 50000],
        }
    )

    with pytest.raises(
        ValueError,
        match="features must contain only column names",
    ):
        apply_feature_dropout(X, features=["income", 123])


def test_feature_dropout_supports_categorical_features():
    X = pd.DataFrame(
        {
            "age": [20, 25, 30],
            "city": ["Hyderabad", "Nizamabad", "Warangal"],
        }
    )
    original = X.copy()

    stressed = apply_feature_dropout(X, features=["city"])

    assert stressed["city"].isna().all()
    assert stressed["age"].tolist() == X["age"].tolist()
    pdt.assert_frame_equal(X, original)


def test_feature_dropout_handles_duplicate_feature_names():
    X = pd.DataFrame(
        {
            "age": [20, 25, 30],
            "income": [30000, 40000, 50000],
        }
    )

    stressed = apply_feature_dropout(
        X,
        features=["age", "age"],
    )

    assert stressed["age"].isna().all()
    assert stressed["income"].tolist() == X["income"].tolist()
    assert stressed.shape == X.shape


def test_feature_dropout_evaluation_with_empty_feature_list():
    from ml_reliability.checks.stress.feature_dropout import (
        evaluate_feature_dropout_stress,
    )

    class Model:
        def predict(self, X):
            return [1 if value > 5 else 0 for value in X["feature"]]

    X = pd.DataFrame(
        {
            "feature": [10.0, 1.0, 20.0, 2.0],
            "other": [1.0, 1.0, 1.0, 1.0],
        }
    )
    y = [1, 0, 1, 0]

    result = evaluate_feature_dropout_stress(
        model=Model(),
        X_test=X,
        y_test=y,
        task="classification",
        features=[],
    )

    assert result.stress_type == "feature_dropout"
    assert result.baseline_metric == 1.0
    assert result.stressed_metric == result.baseline_metric
    assert result.degradation == 0.0
    assert result.passed is True
    assert result.evidence["features"] == []


def test_feature_dropout_rejects_duplicate_dataframe_column_names():
    X = pd.DataFrame(
        [
            [20, 30000, 35000],
            [25, 40000, 45000],
            [30, 50000, 55000],
        ],
        columns=["age", "income", "income"],
    )

    with pytest.raises(ValueError, match="duplicate column names"):
        apply_feature_dropout(
            X,
            features=["income"],
        )


def test_feature_dropout_validates_features_before_no_op():
    X = pd.DataFrame()

    with pytest.raises(
        ValueError,
        match="Features not found in X",
    ):
        apply_feature_dropout(
            X,
            features=["unknown"],
        )


def test_feature_dropout_preserves_nullable_integer_dtype():
    X = pd.DataFrame(
        {
            "age": pd.Series([20, 25, 30], dtype="Int64"),
            "income": [30000, 40000, 50000],
        }
    )

    stressed = apply_feature_dropout(
        X,
        features=["age"],
    )

    assert stressed["age"].isna().all()
    assert str(stressed["age"].dtype) == "Int64"
    assert stressed["income"].tolist() == X["income"].tolist()


def test_feature_dropout_handles_numpy_integer_dtype():
    X = pd.DataFrame({"age": pd.Series([20, 25, 30], dtype="int64")})

    stressed = apply_feature_dropout(X, features=["age"])

    assert stressed["age"].isna().all()
    assert X["age"].tolist() == [20, 25, 30]


def test_feature_dropout_preserves_categorical_dtype():
    X = pd.DataFrame(
        {
            "city": pd.Series(
                ["Hyderabad", "Nizamabad", "Warangal"],
                dtype="category",
            )
        }
    )

    stressed = apply_feature_dropout(X, features=["city"])

    assert stressed["city"].isna().all()
    assert str(stressed["city"].dtype) == "category"
    assert X["city"].notna().all()
