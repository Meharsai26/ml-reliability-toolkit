import pandas as pd
import pytest

from ml_reliability.checks.stress.distribution import (
    perturb_feature_distribution,
)


def test_distribution_perturbation_does_not_modify_original():
    X = pd.DataFrame(
        {
            "age": [20.0, 30.0, 40.0],
            "income": [100.0, 200.0, 300.0],
            "city": ["hyd", "del", "mum"],
        }
    )

    original = X.copy()

    stressed = perturb_feature_distribution(
        X,
        shift=10.0,
        scale=2.0,
    )

    pd.testing.assert_frame_equal(X, original)
    assert stressed is not X


def test_distribution_perturbation_applies_shift():
    X = pd.DataFrame({"feature": [10.0, 20.0, 30.0]})

    stressed = perturb_feature_distribution(
        X,
        shift=5.0,
        scale=1.0,
    )

    expected = pd.DataFrame({"feature": [15.0, 25.0, 35.0]})

    pd.testing.assert_frame_equal(stressed, expected)


def test_distribution_perturbation_applies_scale():
    X = pd.DataFrame({"feature": [10.0, 20.0, 30.0]})

    stressed = perturb_feature_distribution(
        X,
        shift=0.0,
        scale=2.0,
    )

    expected = pd.DataFrame({"feature": [20.0, 40.0, 60.0]})

    pd.testing.assert_frame_equal(stressed, expected)


def test_distribution_perturbation_applies_shift_and_scale():
    X = pd.DataFrame({"feature": [10.0, 20.0, 30.0]})

    stressed = perturb_feature_distribution(
        X,
        shift=5.0,
        scale=2.0,
    )

    expected = pd.DataFrame({"feature": [25.0, 45.0, 65.0]})

    pd.testing.assert_frame_equal(stressed, expected)


def test_non_numeric_columns_are_unchanged():
    X = pd.DataFrame(
        {
            "feature": [10.0, 20.0, 30.0],
            "city": ["hyd", "del", "mum"],
        }
    )

    stressed = perturb_feature_distribution(
        X,
        shift=5.0,
        scale=2.0,
    )

    assert stressed["city"].equals(X["city"])


def test_shape_and_columns_are_preserved():
    X = pd.DataFrame(
        {
            "feature": [10.0, 20.0, 30.0],
            "city": ["hyd", "del", "mum"],
        }
    )

    stressed = perturb_feature_distribution(
        X,
        shift=5.0,
        scale=2.0,
    )

    assert stressed.shape == X.shape
    assert list(stressed.columns) == list(X.columns)


def test_invalid_scale_raises_error():
    X = pd.DataFrame({"feature": [10.0, 20.0, 30.0]})

    with pytest.raises(
        ValueError,
        match="scale must be a finite number greater than 0",
    ):
        perturb_feature_distribution(X, scale=0.0)


def test_empty_dataframe_is_supported():
    X = pd.DataFrame()

    stressed = perturb_feature_distribution(
        X,
        shift=5.0,
        scale=2.0,
    )

    assert stressed.empty
    assert stressed is not X


def test_no_numeric_columns_are_unchanged():
    X = pd.DataFrame({"city": ["hyd", "del", "mum"]})

    stressed = perturb_feature_distribution(
        X,
        shift=5.0,
        scale=2.0,
    )

    pd.testing.assert_frame_equal(stressed, X)


def test_distribution_perturbation_evaluation_classification():
    from ml_reliability.checks.stress.distribution import (
        evaluate_distribution_perturbation_stress,
    )

    class Model:
        def predict(self, X):
            return [1 if value >= 0 else 0 for value in X["feature"]]

    X = pd.DataFrame({"feature": [1.0, -1.0, 2.0, -2.0]})
    y = [1, 0, 1, 0]

    result = evaluate_distribution_perturbation_stress(
        model=Model(),
        X_test=X,
        y_test=y,
        task="classification",
        shift=1.0,
        scale=1.0,
    )

    assert result.stress_type == "distribution_perturbation"
    assert result.baseline_metric == 1.0
    assert 0.0 <= result.stressed_metric <= 1.0
    assert result.degradation >= 0.0
    assert result.passed is True


def test_distribution_perturbation_evaluation_regression():
    from ml_reliability.checks.stress.distribution import (
        evaluate_distribution_perturbation_stress,
    )

    class Model:
        def predict(self, X):
            return X["feature"].tolist()

    X = pd.DataFrame({"feature": [1.0, 2.0, 3.0, 4.0]})
    y = [1.0, 2.0, 3.0, 4.0]

    result = evaluate_distribution_perturbation_stress(
        model=Model(),
        X_test=X,
        y_test=y,
        task="regression",
        shift=1.0,
        scale=1.0,
    )

    assert result.stress_type == "distribution_perturbation"
    assert result.baseline_metric == 1.0
    assert result.stressed_metric <= 1.0
    assert result.degradation >= 0.0
    assert result.passed is True


def test_distribution_perturbation_stress_prediction_failure():
    from ml_reliability.checks.stress.distribution import (
        evaluate_distribution_perturbation_stress,
    )

    class Model:
        def predict(self, X):
            if (X["feature"] > 10).any():
                raise RuntimeError("Distribution outside supported range")
            return [1, 0]

    X = pd.DataFrame({"feature": [1.0, -1.0]})
    y = [1, 0]

    result = evaluate_distribution_perturbation_stress(
        model=Model(),
        X_test=X,
        y_test=y,
        task="classification",
        shift=20.0,
        scale=1.0,
    )

    assert result.passed is False
    assert result.evidence["stress_prediction_failed"] is True
    assert result.degradation == float("inf")


def test_distribution_perturbation_baseline_failure():
    from ml_reliability.checks.stress.distribution import (
        evaluate_distribution_perturbation_stress,
    )

    class Model:
        def predict(self, X):
            raise RuntimeError("Baseline prediction failed")

    X = pd.DataFrame({"feature": [1.0, 2.0]})
    y = [1, 2]

    with pytest.raises(
        ValueError,
        match="Baseline model performance could not be calculated",
    ):
        evaluate_distribution_perturbation_stress(
            model=Model(),
            X_test=X,
            y_test=y,
            task="classification",
            shift=1.0,
        )


# Invalid scale types must raise TypeError.
@pytest.mark.parametrize("scale", [True, "2.0", None])
def test_invalid_scale_types_raise_type_error(scale):
    X = pd.DataFrame({"feature": [1.0, 2.0, 3.0]})

    with pytest.raises(TypeError):
        perturb_feature_distribution(X, scale=scale)


# Invalid scale values must raise ValueError.
@pytest.mark.parametrize(
    "scale",
    [
        float("nan"),
        float("inf"),
        float("-inf"),
        0,
        -1,
    ],
)
def test_invalid_scale_values_raise_value_error(scale):
    X = pd.DataFrame({"feature": [1.0, 2.0, 3.0]})

    with pytest.raises(ValueError):
        perturb_feature_distribution(X, scale=scale)


# Invalid shift types must raise TypeError.
@pytest.mark.parametrize("shift", [True, "2.0", None])
def test_invalid_shift_types_raise_type_error(shift):
    X = pd.DataFrame({"feature": [1.0, 2.0, 3.0]})

    with pytest.raises(TypeError):
        perturb_feature_distribution(X, shift=shift)


# Invalid shift values must raise ValueError.
@pytest.mark.parametrize(
    "shift",
    [
        float("nan"),
        float("inf"),
        float("-inf"),
    ],
)
def test_invalid_shift_values_raise_value_error(shift):
    X = pd.DataFrame({"feature": [1.0, 2.0, 3.0]})

    with pytest.raises(ValueError):
        perturb_feature_distribution(X, shift=shift)


def test_distribution_perturbation_rejects_overflowed_results():
    X = pd.DataFrame({"feature": [1e308, 2.0]})

    with pytest.raises(
        ValueError,
        match="distribution perturbation produced non-finite values",
    ):
        perturb_feature_distribution(X, scale=2.0)


def test_distribution_perturbation_rejects_overflow_from_shift():
    X = pd.DataFrame({"feature": [1e308, 2.0]})

    with pytest.raises(
        ValueError,
        match="distribution perturbation produced non-finite values",
    ):
        perturb_feature_distribution(X, shift=1e308)


def test_extremely_large_integer_shift_raises_value_error():
    X = pd.DataFrame({"feature": [1.0, 2.0]})

    with pytest.raises(
        ValueError,
        match="shift must be a finite number",
    ):
        perturb_feature_distribution(X, shift=10**1000)


def test_extremely_large_integer_scale_raises_value_error():
    X = pd.DataFrame({"feature": [1.0, 2.0]})

    with pytest.raises(
        ValueError,
        match="scale must be a finite number greater than 0",
    ):
        perturb_feature_distribution(X, scale=10**1000)


@pytest.mark.parametrize(
    "random_state",
    ["invalid", None, -1, -1.5, True],
)
def test_distribution_perturbation_rejects_invalid_random_state(
    random_state,
):
    X = pd.DataFrame({"feature": [1.0, 2.0, 3.0]})

    with pytest.raises(ValueError, match="random_state"):
        perturb_feature_distribution(
            X,
            random_state=random_state,
        )


@pytest.mark.parametrize(
    "X",
    [
        pd.DataFrame(),
        pd.DataFrame({"category": ["A", "B"]}),
    ],
)
def test_distribution_perturbation_validates_random_state_on_no_op(X):
    with pytest.raises(ValueError, match="random_state"):
        perturb_feature_distribution(
            X,
            random_state=None,
        )


def test_distribution_perturbation_preserves_missing_values():
    X = pd.DataFrame(
        {
            "feature": [10.0, float("nan"), 30.0],
            "city": ["hyd", "del", "mum"],
        }
    )

    stressed = perturb_feature_distribution(
        X,
        shift=5.0,
        scale=2.0,
    )

    assert stressed["feature"].iloc[0] == 25.0
    assert pd.isna(stressed["feature"].iloc[1])
    assert stressed["feature"].iloc[2] == 65.0
    assert stressed["city"].equals(X["city"])
    assert pd.isna(X["feature"].iloc[1])
