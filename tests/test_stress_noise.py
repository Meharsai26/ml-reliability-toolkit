import math

import pandas as pd
import pandas.testing as pdt
import pytest

from ml_reliability.checks.stress import evaluate_noise_stress, inject_feature_noise


def test_inject_feature_noise_does_not_modify_original_data():
    X = pd.DataFrame(
        {
            "age": [20.0, 25.0, 30.0, 35.0],
            "income": [30000.0, 40000.0, 50000.0, 60000.0],
        }
    )

    original = X.copy()

    stressed = inject_feature_noise(
        X,
        noise_level=0.10,
        random_state=42,
    )

    pdt.assert_frame_equal(X, original)
    assert stressed is not X


def test_inject_feature_noise_preserves_non_numeric_columns():
    X = pd.DataFrame(
        {
            "age": [20.0, 25.0, 30.0, 35.0],
            "city": ["A", "B", "C", "D"],
        }
    )

    stressed = inject_feature_noise(
        X,
        noise_level=0.10,
        random_state=42,
    )

    assert stressed["city"].tolist() == X["city"].tolist()
    assert not stressed["age"].equals(X["age"])


def test_inject_feature_noise_at_zero_level():
    X = pd.DataFrame(
        {
            "age": [20.0, 25.0, 30.0],
            "income": [30000.0, 40000.0, 50000.0],
        }
    )

    stressed = inject_feature_noise(
        X,
        noise_level=0.0,
        random_state=42,
    )

    pdt.assert_frame_equal(stressed, X)


def test_inject_feature_noise_is_reproducible():
    X = pd.DataFrame(
        {
            "age": [20.0, 25.0, 30.0, 35.0],
            "income": [30000.0, 40000.0, 50000.0, 60000.0],
        }
    )

    first = inject_feature_noise(
        X,
        noise_level=0.10,
        random_state=42,
    )

    second = inject_feature_noise(
        X,
        noise_level=0.10,
        random_state=42,
    )

    pdt.assert_frame_equal(first, second)


def test_inject_feature_noise_handles_no_numeric_columns():
    X = pd.DataFrame(
        {
            "city": ["A", "B", "C"],
            "department": ["IT", "HR", "Finance"],
        }
    )

    stressed = inject_feature_noise(
        X,
        noise_level=0.10,
        random_state=42,
    )

    pdt.assert_frame_equal(stressed, X)
    assert stressed is not X


def test_inject_feature_noise_handles_empty_dataframe():
    X = pd.DataFrame()

    stressed = inject_feature_noise(
        X,
        noise_level=0.10,
        random_state=42,
    )

    assert stressed.empty
    assert stressed is not X


def test_inject_feature_noise_rejects_invalid_level():
    X = pd.DataFrame({"age": [20.0, 30.0, 40.0]})

    with pytest.raises(
        ValueError,
        match="rate must be between 0.0 and 1.0",
    ):
        inject_feature_noise(
            X,
            noise_level=1.1,
            random_state=42,
        )


def test_inject_feature_noise_changes_numeric_features():
    X = pd.DataFrame(
        {
            "age": [20.0, 25.0, 30.0, 35.0, 40.0],
        }
    )

    stressed = inject_feature_noise(
        X,
        noise_level=0.10,
        random_state=42,
    )

    assert not stressed["age"].equals(X["age"])


def test_evaluate_noise_stress_classification():
    class Model:
        def predict(self, X):
            predictions = []

            for _, row in X.iterrows():
                predictions.append(1 if row["feature"] >= 0 else 0)

            return predictions

    X = pd.DataFrame(
        {
            "feature": [1.0, 1.0, 1.0, 1.0],
        }
    )

    y = [1, 1, 1, 1]

    outcome = evaluate_noise_stress(
        model=Model(),
        X_test=X,
        y_test=y,
        task="classification",
        noise_level=0.10,
        random_state=42,
    )

    assert outcome.stress_type == "feature_noise"
    assert outcome.baseline_metric == 1.0
    assert 0.0 <= outcome.stressed_metric <= 1.0
    assert outcome.degradation >= 0.0
    assert outcome.passed is True


def test_evaluate_noise_stress_regression():
    class Model:
        def predict(self, X):
            return X["feature"].tolist()

    X = pd.DataFrame(
        {
            "feature": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )

    y = [1.0, 2.0, 3.0, 4.0, 5.0]

    outcome = evaluate_noise_stress(
        model=Model(),
        X_test=X,
        y_test=y,
        task="regression",
        noise_level=0.10,
        random_state=42,
    )

    assert outcome.stress_type == "feature_noise"
    assert outcome.baseline_metric == 1.0
    assert outcome.stressed_metric < 1.0
    assert outcome.degradation > 0.0
    assert outcome.passed is True


def test_evaluate_noise_stress_records_prediction_failure():
    class Model:
        def predict(self, X):
            if X["feature"].max() > 3.5:
                raise ValueError("Stressed feature values are unsupported")
            return [1, 1, 1, 1]

    X = pd.DataFrame({"feature": [0.0, 1.0, 2.0, 3.0]})
    y = [1, 1, 1, 1]

    outcome = evaluate_noise_stress(
        model=Model(),
        X_test=X,
        y_test=y,
        task="classification",
        noise_level=1.0,
        random_state=42,
    )

    assert outcome.passed is False
    assert outcome.evidence["stress_prediction_failed"] is True
    assert "Stressed feature values are unsupported" in outcome.evidence["error"]
    assert math.isnan(outcome.stressed_metric)
    assert outcome.degradation == float("inf")


def test_evaluate_noise_stress_rejects_invalid_baseline():
    class Model:
        def predict(self, X):
            raise ValueError("Model is not ready")

    X = pd.DataFrame(
        {
            "feature": [1.0, 2.0, 3.0],
        }
    )

    y = [1, 2, 3]

    import pytest

    with pytest.raises(
        ValueError,
        match="Baseline model performance could not be calculated",
    ):
        evaluate_noise_stress(
            model=Model(),
            X_test=X,
            y_test=y,
            task="regression",
            noise_level=0.10,
            random_state=42,
        )


def test_inject_feature_noise_rejects_overflowed_results():
    X = pd.DataFrame({"feature": [1e308, -1e308, 1e308, -1e308]})

    with pytest.raises(
        ValueError,
        match="feature noise produced non-finite values",
    ):
        inject_feature_noise(
            X,
            noise_level=1.0,
            random_state=42,
        )


def test_inject_feature_noise_rejects_non_finite_input_values():
    X = pd.DataFrame({"feature": [1.0, float("inf"), 3.0, 4.0]})

    with pytest.raises(
        ValueError,
        match="feature noise produced non-finite values",
    ):
        inject_feature_noise(
            X,
            noise_level=0.1,
            random_state=42,
        )


@pytest.mark.parametrize(
    "random_state",
    ["invalid", None, -1, -1.5, True],
)
def test_inject_feature_noise_rejects_invalid_random_state(
    random_state,
):
    X = pd.DataFrame({"feature": [1.0, 2.0, 3.0]})

    with pytest.raises((TypeError, ValueError)):
        inject_feature_noise(
            X,
            noise_level=0.1,
            random_state=random_state,
        )


@pytest.mark.parametrize(
    "X, noise_level",
    [
        (pd.DataFrame(), 0.1),
        (pd.DataFrame({"category": ["A", "B"]}), 0.1),
        (pd.DataFrame({"feature": [1.0, 2.0]}), 0.0),
    ],
)
def test_inject_feature_noise_validates_random_state_on_no_op(X, noise_level):
    with pytest.raises(ValueError, match="random_state"):
        inject_feature_noise(
            X,
            noise_level=noise_level,
            random_state=None,
        )


def test_inject_feature_noise_preserves_constant_numeric_columns():
    X = pd.DataFrame(
        {
            "constant": [5.0, 5.0, 5.0, 5.0],
            "variable": [1.0, 2.0, 3.0, 4.0],
        }
    )

    stressed = inject_feature_noise(
        X,
        noise_level=0.5,
        random_state=42,
    )

    pdt.assert_series_equal(
        stressed["constant"],
        X["constant"],
    )
    assert not stressed["variable"].equals(X["variable"])


def test_inject_feature_noise_preserves_existing_nan_values():
    X = pd.DataFrame(
        {
            "numeric": [1.0, float("nan"), 3.0, 4.0],
            "category": ["A", "B", "C", "D"],
        }
    )

    stressed = inject_feature_noise(
        X,
        noise_level=0.5,
        random_state=42,
    )

    assert pd.isna(stressed.loc[1, "numeric"])
    assert stressed["category"].equals(X["category"])
    assert pd.isna(X.loc[1, "numeric"])


def test_inject_feature_noise_rejects_infinite_input_values():
    X = pd.DataFrame(
        {
            "numeric": [1.0, float("inf"), 3.0],
        }
    )

    with pytest.raises(
        ValueError,
        match="feature noise produced non-finite values",
    ):
        inject_feature_noise(
            X,
            noise_level=0.5,
            random_state=42,
        )


def test_inject_feature_noise_handles_all_nan_numeric_column():
    X = pd.DataFrame(
        {
            "numeric": [float("nan"), float("nan"), float("nan")],
            "category": ["A", "B", "C"],
        }
    )

    stressed = inject_feature_noise(
        X,
        noise_level=0.5,
        random_state=42,
    )

    assert stressed["numeric"].isna().all()
    assert stressed["category"].equals(X["category"])
    assert X["numeric"].isna().all()


def test_inject_feature_noise_rejects_duplicate_column_names():
    X = pd.DataFrame(
        [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]],
        columns=["feature", "feature"],
    )

    with pytest.raises(
        ValueError,
        match="duplicate column names",
    ):
        inject_feature_noise(
            X,
            noise_level=0.1,
            random_state=42,
        )


def test_inject_feature_noise_handles_nullable_integer_dtype():
    X = pd.DataFrame(
        {
            "feature": pd.Series(
                [10, 20, 30, 40],
                dtype="Int64",
            )
        }
    )

    stressed = inject_feature_noise(
        X,
        noise_level=0.1,
        random_state=42,
    )

    assert stressed.shape == X.shape
    assert stressed["feature"].dtype.kind == "f"
    assert not stressed["feature"].equals(X["feature"])
    assert str(X["feature"].dtype) == "Int64"


def test_inject_feature_noise_preserves_nullable_integer_missing_values():
    X = pd.DataFrame(
        {
            "feature": pd.Series(
                [10, pd.NA, 30, 40],
                dtype="Int64",
            )
        }
    )

    stressed = inject_feature_noise(
        X,
        noise_level=0.1,
        random_state=42,
    )

    assert pd.isna(stressed.loc[1, "feature"])
    assert stressed["feature"].notna().sum() == 3
    assert pd.isna(X.loc[1, "feature"])


def test_inject_feature_noise_handles_near_constant_values():
    X = pd.DataFrame(
        {
            "feature": [
                1.0,
                1.0 + 1e-12,
                1.0 + 2e-12,
                1.0 + 3e-12,
            ]
        }
    )

    stressed = inject_feature_noise(
        X,
        noise_level=0.5,
        random_state=42,
    )

    assert stressed.shape == X.shape
    assert not stressed["feature"].equals(X["feature"])
    assert stressed["feature"].notna().all()
    assert stressed["feature"].map(math.isfinite).all()


def test_inject_feature_noise_rejects_overflow_in_standard_deviation():
    X = pd.DataFrame(
        {
            "feature": [
                1e308,
                -1e308,
                1e308,
                -1e308,
            ]
        }
    )

    with pytest.raises(
        ValueError,
        match="feature noise produced non-finite values",
    ):
        inject_feature_noise(
            X,
            noise_level=0.1,
            random_state=42,
        )
