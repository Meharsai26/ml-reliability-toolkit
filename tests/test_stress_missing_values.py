import numpy as np
import pandas as pd

from ml_reliability.checks.stress import (
    evaluate_missing_value_stress,
    inject_missing_values,
)


def test_inject_missing_values_does_not_modify_original_data():
    X = pd.DataFrame(
        {
            "age": [20, 25, 30, 35],
            "income": [30000, 40000, 50000, 60000],
        }
    )

    original = X.copy()

    stressed = inject_missing_values(
        X,
        rate=0.25,
        random_state=42,
    )

    pd.testing.assert_frame_equal(X, original)
    assert stressed is not X


def test_inject_missing_values_at_zero_rate():
    X = pd.DataFrame(
        {
            "age": [20, 25, 30],
            "income": [30000, 40000, 50000],
        }
    )

    stressed = inject_missing_values(
        X,
        rate=0.0,
        random_state=42,
    )

    pd.testing.assert_frame_equal(stressed, X)


def test_inject_missing_values_changes_expected_number_of_values():
    X = pd.DataFrame(
        {
            "a": range(10),
            "b": range(10, 20),
        }
    )

    stressed = inject_missing_values(
        X,
        rate=0.20,
        random_state=42,
    )

    missing_count = int(stressed.isna().sum().sum())

    assert missing_count == 4


def test_inject_missing_values_is_reproducible():
    X = pd.DataFrame(
        {
            "a": range(10),
            "b": range(10, 20),
        }
    )

    first = inject_missing_values(
        X,
        rate=0.20,
        random_state=42,
    )

    second = inject_missing_values(
        X,
        rate=0.20,
        random_state=42,
    )

    pd.testing.assert_frame_equal(first, second)


def test_inject_missing_values_handles_empty_dataframe():
    X = pd.DataFrame()

    stressed = inject_missing_values(
        X,
        rate=0.20,
        random_state=42,
    )

    assert stressed.empty
    assert stressed is not X


def test_evaluate_missing_value_stress_classification():
    class Model:
        def predict(self, X):
            predictions = []

            for _, row in X.iterrows():
                if pd.isna(row["feature"]):
                    predictions.append(0)
                else:
                    predictions.append(1 if row["feature"] >= 3 else 0)

            return predictions

    X = pd.DataFrame(
        {
            "feature": [1, 2, 3, 4, 5],
        }
    )

    y = [0, 0, 1, 1, 1]

    outcome = evaluate_missing_value_stress(
        model=Model(),
        X_test=X,
        y_test=y,
        task="classification",
        rate=0.2,
        random_state=42,
    )

    assert outcome.stress_type == "missing_values"
    assert outcome.baseline_metric == 1.0
    assert 0.0 <= outcome.stressed_metric <= 1.0
    assert outcome.degradation >= 0.0
    assert outcome.passed is True


def test_evaluate_missing_value_stress_regression():
    class Model:
        def predict(self, X):
            predictions = []

            for _, row in X.iterrows():
                if pd.isna(row["feature"]):
                    predictions.append(0.0)
                else:
                    predictions.append(float(row["feature"]))

            return predictions

    X = pd.DataFrame(
        {
            "feature": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )

    y = [1.0, 2.0, 3.0, 4.0, 5.0]

    outcome = evaluate_missing_value_stress(
        model=Model(),
        X_test=X,
        y_test=y,
        task="regression",
        rate=0.2,
        random_state=42,
    )

    assert outcome.stress_type == "missing_values"
    assert outcome.baseline_metric == 1.0
    assert outcome.stressed_metric < outcome.baseline_metric
    assert outcome.degradation > 0


def test_evaluate_missing_value_stress_records_prediction_failure():
    class Model:
        def predict(self, X):
            if X.isna().any().any():
                raise ValueError("NaN values are not supported")

            return [0, 0, 1, 1, 1]

    X = pd.DataFrame(
        {
            "feature": [1, 2, 3, 4, 5],
        }
    )

    y = [0, 0, 1, 1, 1]

    outcome = evaluate_missing_value_stress(
        model=Model(),
        X_test=X,
        y_test=y,
        task="classification",
        rate=0.2,
        random_state=42,
    )

    assert outcome.passed is False
    assert outcome.evidence["stress_prediction_failed"] is True
    assert outcome.evidence["rate"] == 0.2
    assert outcome.baseline_metric == 1.0


def test_evaluate_missing_value_stress_rejects_invalid_baseline():
    class Model:
        def predict(self, X):
            raise ValueError("Model is not ready")

    X = pd.DataFrame(
        {
            "feature": [1, 2, 3],
        }
    )

    y = [0, 1, 1]

    import pytest

    with pytest.raises(
        ValueError,
        match="Baseline model performance could not be calculated",
    ):
        evaluate_missing_value_stress(
            model=Model(),
            X_test=X,
            y_test=y,
            task="classification",
            rate=0.2,
            random_state=42,
        )


def test_evaluate_missing_value_stress_detects_degradation():
    class Model:
        def predict(self, X):
            predictions = []

            for _, row in X.iterrows():
                if pd.isna(row["feature"]):
                    predictions.append(0)
                else:
                    predictions.append(1 if row["feature"] >= 0 else 0)

            return predictions

    X = pd.DataFrame(
        {
            "feature": [1.0, 1.0, 1.0, 1.0],
        }
    )

    y = [1, 1, 1, 1]

    outcome = evaluate_missing_value_stress(
        model=Model(),
        X_test=X,
        y_test=y,
        task="classification",
        rate=0.25,
        random_state=42,
    )

    assert outcome.baseline_metric == 1.0
    assert outcome.stressed_metric == 0.75
    assert outcome.degradation == 0.25
    assert outcome.passed is True


import pytest


@pytest.mark.parametrize("random_state", ["invalid", None, -1.5])
def test_inject_missing_values_rejects_invalid_random_state(random_state):
    X = pd.DataFrame({"feature": [1, 2, 3, 4, 5]})

    with pytest.raises((TypeError, ValueError)):
        inject_missing_values(
            X,
            rate=0.4,
            random_state=random_state,
        )


def test_inject_missing_values_adds_requested_missing_values_when_input_has_nans():
    X = pd.DataFrame(
        {
            "a": [np.nan, 2.0, 3.0, 4.0],
            "b": [5.0, 6.0, 7.0, 8.0],
        }
    )

    original = X.copy()
    original_missing = X.isna()

    stressed = inject_missing_values(
        X,
        rate=0.25,
        random_state=42,
    )

    newly_missing = stressed.isna() & ~original_missing

    assert int(newly_missing.sum().sum()) == 1
    pd.testing.assert_frame_equal(X, original)


def test_inject_missing_values_rejects_numpy_boolean_random_state():
    X = pd.DataFrame({"feature": [1, 2, 3, 4, 5]})

    with pytest.raises(ValueError, match="random_state must be a non-negative integer"):
        inject_missing_values(
            X,
            rate=0.4,
            random_state=np.bool_(True),
        )


@pytest.mark.parametrize(
    "rate",
    [
        -0.1,
        1.1,
        float("nan"),
        float("inf"),
        float("-inf"),
        True,
        np.bool_(False),
        "0.5",
    ],
)
def test_inject_missing_values_rejects_invalid_rates(rate):
    X = pd.DataFrame({"feature": [1, 2, 3, 4, 5]})

    with pytest.raises(ValueError, match="rate must be between 0.0 and 1.0"):
        inject_missing_values(
            X,
            rate=rate,
            random_state=42,
        )


def test_inject_missing_values_uses_eligible_values_for_corruption_rate():
    X = pd.DataFrame(
        {
            "a": [np.nan, np.nan, 3.0, 4.0],
            "b": [np.nan, 6.0, 7.0, 8.0],
        }
    )

    original_missing = X.isna()

    stressed = inject_missing_values(
        X,
        rate=0.5,
        random_state=42,
    )

    newly_missing = stressed.isna() & ~original_missing

    # Five eligible cells Ã— 50% = 2 newly introduced missing values.
    assert int(newly_missing.sum().sum()) == 2


def test_inject_missing_values_preserves_nullable_integer_dtype():
    X = pd.DataFrame(
        {
            "feature": pd.Series(
                [10, 20, 30, 40],
                dtype="Int64",
            )
        }
    )

    stressed = inject_missing_values(
        X,
        rate=0.25,
        random_state=42,
    )

    assert str(stressed["feature"].dtype) == "Int64"
    assert stressed["feature"].isna().sum() == 1
    assert str(X["feature"].dtype) == "Int64"
    assert X["feature"].notna().all()


def test_inject_missing_values_with_small_positive_rate():
    X = pd.DataFrame({"feature": [10.0, 20.0, 30.0]})

    stressed = inject_missing_values(
        X,
        rate=0.01,
        random_state=42,
    )

    assert stressed["feature"].isna().sum() == 1
