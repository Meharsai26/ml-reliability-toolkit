import math

import pandas as pd

from ml_reliability import ReliabilitySuite
from ml_reliability.core.result import CheckResult


def get_check(report, check_id):
    return next(check for check in report.checks if check.check_id == check_id)


def test_suite_with_clean_training_data():
    df = pd.DataFrame(
        {
            "age": [20, 21, 22, 23],
            "salary": [30000, 40000, 50000, 60000],
        }
    )

    report = ReliabilitySuite().run(X_train=df)

    assert report.score == 100.0
    assert report.grade == "A"
    assert len(report.checks) == 4

    assert get_check(report, "data.schema").passed is True
    assert get_check(report, "data.missing_values").passed is True
    assert get_check(report, "data.duplicates").passed is True


def test_suite_detects_missing_training_data():
    df = pd.DataFrame(
        {
            "age": [20, 21, None, 23],
            "salary": [30000, None, 40000, 50000],
        }
    )

    report = ReliabilitySuite().run(X_train=df)

    assert report.score == 75.0
    assert report.grade == "B"
    assert len(report.checks) == 4

    schema_check = get_check(report, "data.schema")
    missing_check = get_check(report, "data.missing_values")
    duplicate_check = get_check(report, "data.duplicates")

    assert schema_check.passed is True
    assert schema_check.score == 100.0

    assert missing_check.passed is False
    assert missing_check.score == 0.0

    assert duplicate_check.passed is True
    assert duplicate_check.score == 100.0


def test_suite_detects_duplicate_training_data():
    df = pd.DataFrame(
        {
            "age": [20, 21, 20, 23],
            "salary": [30000, 40000, 30000, 50000],
        }
    )

    report = ReliabilitySuite().run(X_train=df)

    assert report.score == 75.0
    assert report.grade == "B"
    assert len(report.checks) == 4

    schema_check = get_check(report, "data.schema")
    missing_check = get_check(report, "data.missing_values")
    duplicate_check = get_check(report, "data.duplicates")

    assert schema_check.passed is True
    assert schema_check.score == 100.0

    assert missing_check.passed is True
    assert missing_check.score == 100.0

    assert duplicate_check.passed is False
    assert duplicate_check.score == 0.0


def test_suite_includes_target_distribution():
    X = pd.DataFrame(
        {
            "age": [20, 21, 22, 23],
            "salary": [30000, 40000, 50000, 60000],
        }
    )

    y = [0, 0, 1, 1]

    report = ReliabilitySuite().run(
        X_train=X,
        y_train=y,
    )

    assert len(report.checks) == 5

    target_check = get_check(
        report,
        "target.distribution",
    )

    assert target_check.passed is True
    assert target_check.score == 100.0
    assert target_check.evidence["task_type"] == "classification"


def test_suite_detects_target_imbalance():
    X = pd.DataFrame(
        {
            "age": [20, 21, 22, 23],
            "salary": [30000, 40000, 50000, 60000],
        }
    )

    y = [0] * 95 + [1] * 5

    report = ReliabilitySuite().run(
        X_train=pd.concat([X] * 25, ignore_index=True).iloc[:100],
        y_train=y,
    )

    target_check = get_check(
        report,
        "target.distribution",
    )

    assert target_check.passed is True
    assert target_check.score == 70.0
    assert target_check.severity == "warning"
    assert target_check.evidence["minority_percentage"] == 5.0


def test_suite_includes_distribution_shift():
    X_train = pd.DataFrame(
        {
            "age": [20, 21, 22, 23, 24],
            "salary": [30000, 40000, 50000, 60000, 70000],
        }
    )

    X_test = pd.DataFrame(
        {
            "age": [20, 21, 22, 23, 24],
            "salary": [30000, 40000, 50000, 60000, 70000],
        }
    )

    report = ReliabilitySuite().run(
        X_train=X_train,
        X_test=X_test,
    )

    assert len(report.checks) == 5
    assert any(check.check_id == "data.distribution_shift" for check in report.checks)


def test_suite_skips_distribution_shift_without_test_data():
    X_train = pd.DataFrame(
        {
            "age": [20, 21, 22, 23],
            "salary": [30000, 40000, 50000, 60000],
        }
    )

    report = ReliabilitySuite().run(X_train=X_train)

    assert len(report.checks) == 4
    assert not any(
        check.check_id == "data.distribution_shift" for check in report.checks
    )


class DummyClassificationModel:
    def predict(self, X):
        return [0, 1, 1, 0]


class DummyRegressionModel:
    def predict(self, X):
        return [1.0, 2.0, 3.0, 4.0]


def test_suite_includes_classification_model_performance():
    X_train = pd.DataFrame(
        {
            "age": [20, 21, 22, 23],
            "salary": [30000, 40000, 50000, 60000],
        }
    )

    y_train = [0, 1, 1, 0]

    X_test = pd.DataFrame(
        {
            "age": [24, 25, 26, 27],
            "salary": [40000, 50000, 60000, 70000],
        }
    )

    y_test = [0, 1, 1, 0]

    report = ReliabilitySuite().run(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        model=DummyClassificationModel(),
    )

    performance_check = get_check(
        report,
        "model.performance",
    )

    assert performance_check.passed is True
    assert performance_check.score == 100.0
    assert performance_check.evidence["task_type"] == "classification"
    assert performance_check.evidence["metric"] == "accuracy"
    assert performance_check.evidence["metric_value"] == 1.0


def test_suite_includes_regression_model_performance():
    X_train = pd.DataFrame(
        {
            "feature": [1, 2, 3, 4],
        }
    )

    y_train = [1.0, 2.0, 3.0, 4.0]

    X_test = pd.DataFrame(
        {
            "feature": [5, 6, 7, 8],
        }
    )

    y_test = [1.0, 2.0, 3.0, 4.0]

    report = ReliabilitySuite(task_type="regression").run(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        model=DummyRegressionModel(),
    )

    performance_check = get_check(
        report,
        "model.performance",
    )

    assert performance_check.passed is True
    assert performance_check.score == 100.0
    assert performance_check.evidence["task_type"] == "regression"
    assert performance_check.evidence["metric"] == "r2"
    assert performance_check.evidence["metric_value"] == 1.0


def test_suite_skips_model_performance_without_model():
    X_train = pd.DataFrame(
        {
            "age": [20, 21, 22, 23],
        }
    )

    y_train = [0, 1, 0, 1]

    X_test = pd.DataFrame(
        {
            "age": [24, 25, 26, 27],
        }
    )

    y_test = [0, 1, 0, 1]

    report = ReliabilitySuite().run(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )

    assert not any(check.check_id == "model.performance" for check in report.checks)


def test_suite_skips_model_performance_without_test_data():
    X_train = pd.DataFrame(
        {
            "age": [20, 21, 22, 23],
        }
    )

    y_train = [0, 1, 0, 1]

    report = ReliabilitySuite().run(
        X_train=X_train,
        y_train=y_train,
        model=DummyClassificationModel(),
    )

    assert not any(check.check_id == "model.performance" for check in report.checks)


def test_suite_defaults_to_automatic_task_inference():
    suite = ReliabilitySuite()

    assert suite.task_type is None


def test_suite_accepts_classification_task_type():
    suite = ReliabilitySuite(task_type="classification")

    assert suite.task_type == "classification"


def test_suite_accepts_regression_task_type():
    suite = ReliabilitySuite(task_type="regression")

    assert suite.task_type == "regression"


def test_suite_rejects_invalid_task_type():
    import pytest

    with pytest.raises(ValueError):
        ReliabilitySuite(task_type="invalid")


def test_suite_includes_classification_generalization_gap():
    X_train = pd.DataFrame(
        {
            "age": [20, 21, 22, 23],
            "salary": [30000, 40000, 50000, 60000],
        }
    )

    y_train = [0, 1, 1, 0]

    X_test = pd.DataFrame(
        {
            "age": [24, 25, 26, 27],
            "salary": [40000, 50000, 60000, 70000],
        }
    )

    y_test = [0, 1, 0, 0]

    report = ReliabilitySuite().run(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        model=DummyClassificationModel(),
    )

    generalization_check = get_check(
        report,
        "model.generalization_gap",
    )

    assert generalization_check.passed is False
    assert generalization_check.severity == "warning"
    assert generalization_check.evidence["task_type"] == "classification"
    assert generalization_check.evidence["metric"] == "accuracy"
    assert generalization_check.evidence["train_metric"] == 1.0
    assert generalization_check.evidence["test_metric"] == 0.75
    assert generalization_check.evidence["generalization_gap"] == 0.25


def test_suite_includes_classification_metrics():
    class Model:
        def predict(self, X):
            return [1, 1, 0, 0]

    suite = ReliabilitySuite(task_type="classification")

    report = suite.run(
        X_train=[1, 2, 3, 4],
        y_train=[1, 0, 1, 0],
        X_test=[5, 6, 7, 8],
        y_test=[1, 0, 1, 0],
        model=Model(),
    )

    check_ids = [check.check_id for check in report.checks]

    assert "model.classification_metrics" in check_ids


def test_suite_includes_regression_metrics():
    class Model:
        def predict(self, X):
            return [2, 4, 6, 8]

    suite = ReliabilitySuite(task_type="regression")

    report = suite.run(
        X_train=[1, 2, 3, 4],
        y_train=[1, 5, 5, 10],
        X_test=[5, 6, 7, 8],
        y_test=[1, 5, 5, 10],
        model=Model(),
    )

    check_ids = [check.check_id for check in report.checks]

    assert "model.regression_metrics" in check_ids


def test_suite_includes_calibration():
    class Model:
        def predict(self, X):
            return [1, 1, 0, 0]

        def predict_proba(self, X):
            return [
                [0.1, 0.9],
                [0.2, 0.8],
                [0.8, 0.2],
                [0.9, 0.1],
            ]

    suite = ReliabilitySuite(task_type="classification")

    report = suite.run(
        X_train=[1, 2, 3, 4],
        y_train=[1, 0, 1, 0],
        X_test=[5, 6, 7, 8],
        y_test=[1, 1, 0, 0],
        model=Model(),
    )

    check_ids = [check.check_id for check in report.checks]

    assert "model.calibration" in check_ids


def test_suite_includes_slice_analysis():
    class Model:
        def predict(self, X):
            return [1, 1, 0, 0]

    suite = ReliabilitySuite(task_type="classification")

    report = suite.run(
        X_train=[1, 2, 3, 4],
        y_train=[1, 0, 1, 0],
        X_test=[5, 6, 7, 8],
        y_test=[1, 0, 0, 1],
        model=Model(),
        slices=["A", "A", "B", "B"],
    )

    check_ids = [check.check_id for check in report.checks]

    assert "model.slice_analysis" in check_ids


def test_suite_accepts_stress_tests_argument():
    suite = ReliabilitySuite()

    report = suite.run(
        stress_tests=None,
    )

    assert report.stress_tests == []


def test_suite_runs_missing_value_stress():
    class Model:
        def predict(self, X):
            return [1 if value > 5 else 0 for value in X["feature"]]

    X_train = pd.DataFrame(
        {
            "feature": [10.0, 1.0, 20.0, 2.0],
        }
    )

    y_train = [1, 0, 1, 0]

    X_test = pd.DataFrame(
        {
            "feature": [10.0, 1.0, 20.0, 2.0],
        }
    )

    y_test = [1, 0, 1, 0]

    report = ReliabilitySuite(task_type="classification").run(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        model=Model(),
        stress_tests={
            "missing_values": {
                "rate": 0.1,
                "random_state": 42,
            }
        },
    )

    assert len(report.stress_tests) == 1

    stress_result = report.stress_tests[0]

    assert stress_result.stress_type == "missing_values"
    assert stress_result.baseline_metric == 1.0
    assert stress_result.passed is True


def test_suite_runs_feature_noise_stress():
    class Model:
        def predict(self, X):
            return [1 if value > 5 else 0 for value in X["feature"]]

    X_train = pd.DataFrame(
        {
            "feature": [10.0, 1.0, 20.0, 2.0],
        }
    )

    y_train = [1, 0, 1, 0]

    X_test = pd.DataFrame(
        {
            "feature": [10.0, 1.0, 20.0, 2.0],
        }
    )

    y_test = [1, 0, 1, 0]

    report = ReliabilitySuite(task_type="classification").run(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        model=Model(),
        stress_tests={
            "feature_noise": {
                "noise_level": 0.1,
                "random_state": 42,
            }
        },
    )

    assert len(report.stress_tests) == 1

    stress_result = report.stress_tests[0]

    assert stress_result.stress_type == "feature_noise"
    assert stress_result.baseline_metric == 1.0
    assert 0.0 <= stress_result.stressed_metric <= 1.0
    assert stress_result.passed is True


def test_suite_runs_feature_dropout_stress():
    class Model:
        def predict(self, X):
            return [1 if value > 5 else 0 for value in X["feature"]]

    X_train = pd.DataFrame(
        {
            "feature": [10.0, 1.0, 20.0, 2.0],
            "other_feature": [1.0, 2.0, 3.0, 4.0],
        }
    )

    y_train = [1, 0, 1, 0]

    X_test = pd.DataFrame(
        {
            "feature": [10.0, 1.0, 20.0, 2.0],
            "other_feature": [1.0, 2.0, 3.0, 4.0],
        }
    )

    y_test = [1, 0, 1, 0]

    report = ReliabilitySuite(task_type="classification").run(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        model=Model(),
        stress_tests={
            "feature_dropout": {
                "features": ["feature"],
            }
        },
    )

    assert len(report.stress_tests) == 1

    stress_result = report.stress_tests[0]

    assert stress_result.stress_type == "feature_dropout"
    assert stress_result.baseline_metric == 1.0
    assert stress_result.passed is True


def test_suite_runs_categorical_corruption_stress():
    class Model:
        def predict(self, X):
            return [1 if value == "A" else 0 for value in X["category"]]

    X_train = pd.DataFrame(
        {
            "category": ["A", "B", "A", "B"],
            "feature": [10.0, 1.0, 20.0, 2.0],
        }
    )

    y_train = [1, 0, 1, 0]

    X_test = pd.DataFrame(
        {
            "category": ["A", "B", "A", "B"],
            "feature": [10.0, 1.0, 20.0, 2.0],
        }
    )

    y_test = [1, 0, 1, 0]

    report = ReliabilitySuite(task_type="classification").run(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        model=Model(),
        stress_tests={
            "categorical_corruption": {
                "columns": ["category"],
                "rate": 0.5,
                "random_state": 42,
            }
        },
    )

    assert len(report.stress_tests) == 1

    stress_result = report.stress_tests[0]

    assert stress_result.stress_type == "categorical_corruption"
    assert stress_result.baseline_metric == 1.0
    assert 0.0 <= stress_result.stressed_metric <= 1.0
    assert stress_result.passed is True


def test_suite_runs_distribution_perturbation_stress():
    class Model:
        def predict(self, X):
            return [1 if value > 5 else 0 for value in X["feature"]]

    X_train = pd.DataFrame(
        {
            "feature": [10.0, 1.0, 20.0, 2.0],
        }
    )

    y_train = [1, 0, 1, 0]

    X_test = pd.DataFrame(
        {
            "feature": [10.0, 1.0, 20.0, 2.0],
        }
    )

    y_test = [1, 0, 1, 0]

    report = ReliabilitySuite(task_type="classification").run(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        model=Model(),
        stress_tests={
            "distribution_perturbation": {
                "shift": 1.0,
                "scale": 1.2,
                "random_state": 42,
            }
        },
    )

    assert len(report.stress_tests) == 1

    stress_result = report.stress_tests[0]

    assert stress_result.stress_type == "distribution_perturbation"
    assert stress_result.baseline_metric == 1.0
    assert 0.0 <= stress_result.stressed_metric <= 1.0
    assert stress_result.passed is True


def test_empty_suite_report_indicates_insufficient_evidence():

    report = ReliabilitySuite().run()
    assert report.score is None
    assert report.grade == "N/A"
    assert report.checks == []
    assert report.stress_tests == []


def test_stress_tests_do_not_change_overall_reliability_score():
    class Model:
        def predict(self, X):
            return [1 if value > 5 else 0 for value in X["feature"]]

    X_train = pd.DataFrame({"feature": [10.0, 1.0, 20.0, 2.0]})
    y_train = [1, 0, 1, 0]
    X_test = pd.DataFrame({"feature": [10.0, 1.0, 20.0, 2.0]})
    y_test = [1, 0, 1, 0]

    suite = ReliabilitySuite(task_type="classification")

    baseline_report = suite.run(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        model=Model(),
    )

    stress_report = suite.run(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        model=Model(),
        stress_tests={
            "missing_values": {
                "rate": 0.1,
                "random_state": 42,
            }
        },
    )

    assert len(stress_report.stress_tests) == 1
    assert stress_report.score == baseline_report.score
    assert stress_report.grade == baseline_report.grade


def test_suite_excludes_unscored_checks_from_report_average():
    from ml_reliability.core.result import CheckResult

    checks = [
        CheckResult(
            check_id="data.schema",
            title="Schema",
            category="data_quality",
            passed=True,
            severity="info",
            score=80.0,
            message="Schema check completed.",
        ),
        CheckResult(
            check_id="model.regression_metrics",
            title="Regression Metrics",
            category="performance",
            passed=True,
            severity="info",
            score=None,
            message="Metrics calculated without a quality threshold.",
        ),
    ]

    scored_checks = [check for check in checks if check.score is not None]

    score = sum(check.score for check in scored_checks) / len(scored_checks)

    assert score == 80.0


def test_suite_returns_na_when_all_checks_are_unscored(monkeypatch):
    import ml_reliability.suite as suite_module
    from ml_reliability.core.result import CheckResult

    def unscored_check(*args, **kwargs):
        return CheckResult(
            check_id="test.unscored",
            title="Unscored Check",
            category="performance",
            passed=True,
            severity="info",
            score=None,
            message="No quality threshold configured.",
        )

    monkeypatch.setattr(
        suite_module,
        "check_schema",
        unscored_check,
    )
    monkeypatch.setattr(
        suite_module,
        "check_missing_values",
        unscored_check,
    )
    monkeypatch.setattr(
        suite_module,
        "check_duplicates",
        unscored_check,
    )
    monkeypatch.setattr(
        suite_module,
        "check_cardinality",
        unscored_check,
    )

    import pandas as pd

    report = suite_module.ReliabilitySuite().run(
        X_train=pd.DataFrame({"feature": [1, 2, 3]}),
    )

    assert len(report.checks) == 4
    assert report.score is None
    assert report.grade == "N/A"


def test_suite_excludes_unscored_checks_from_actual_report_average(
    monkeypatch,
):
    import pandas as pd

    import ml_reliability.suite as suite_module
    from ml_reliability.core.result import CheckResult

    def make_check(check_id, score):
        def check(*args, **kwargs):
            return CheckResult(
                check_id=check_id,
                title=check_id,
                category="test",
                passed=True,
                severity="info",
                score=score,
                message="Test check completed.",
            )

        return check

    monkeypatch.setattr(
        suite_module,
        "check_schema",
        make_check("test.schema", 80.0),
    )
    monkeypatch.setattr(
        suite_module,
        "check_missing_values",
        make_check("test.missing", None),
    )
    monkeypatch.setattr(
        suite_module,
        "check_duplicates",
        make_check("test.duplicates", 60.0),
    )
    monkeypatch.setattr(
        suite_module,
        "check_cardinality",
        make_check("test.cardinality", None),
    )

    report = suite_module.ReliabilitySuite().run(
        X_train=pd.DataFrame({"feature": [1, 2, 3]}),
    )

    assert len(report.checks) == 4
    assert report.score == 70.0
    assert report.grade == "C"
    assert [check.score for check in report.checks] == [80.0, None, 60.0, None]


def test_suite_does_not_report_non_finite_check_score(monkeypatch):
    import pandas as pd

    import ml_reliability.suite as suite_module
    from ml_reliability.core.result import CheckResult

    def invalid_score_check(*args, **kwargs):
        return CheckResult(
            check_id="test.invalid_score",
            title="Invalid Score",
            category="test",
            passed=True,
            severity="info",
            score=float("nan"),
            message="Simulated invalid score.",
        )

    monkeypatch.setattr(
        suite_module,
        "check_schema",
        invalid_score_check,
    )

    report = suite_module.ReliabilitySuite().run(
        X_train=pd.DataFrame({"feature": [1, 2, 3]}),
    )

    assert report.score is not None
    assert math.isfinite(report.score)
    assert 0.0 <= report.score <= 100.0


def test_suite_excludes_out_of_range_scores_from_average(monkeypatch):
    import pandas as pd

    import ml_reliability.suite as suite_module
    from ml_reliability.core.result import CheckResult

    def make_check(check_id, score):
        return lambda *args, **kwargs: CheckResult(
            check_id=check_id,
            title=check_id,
            category="test",
            passed=True,
            severity="info",
            score=score,
            message="Test check.",
            evidence={},
            recommendation="",
        )

    monkeypatch.setattr(
        suite_module,
        "check_schema",
        make_check("schema", 150.0),
    )
    monkeypatch.setattr(
        suite_module,
        "check_missing_values",
        make_check("missing", 80.0),
    )
    monkeypatch.setattr(
        suite_module,
        "check_duplicates",
        make_check("duplicates", 60.0),
    )
    monkeypatch.setattr(
        suite_module,
        "check_cardinality",
        make_check("cardinality", 100.0),
    )

    report = suite_module.ReliabilitySuite().run(
        X_train=pd.DataFrame({"feature": [1, 2, 3]})
    )

    assert report.score == 80.0
    assert report.grade == "B"
    assert [check.check_id for check in report.invalid_score_checks] == ["schema"]


import pytest


@pytest.mark.parametrize(
    ("score", "expected_grade"),
    [
        (0.0, "F"),
        (39.99, "F"),
        (40.0, "D"),
        (59.99, "D"),
        (60.0, "C"),
        (74.99, "C"),
        (75.0, "B"),
        (89.99, "B"),
        (90.0, "A"),
        (100.0, "A"),
    ],
)
def test_suite_grade_boundaries(
    monkeypatch,
    score,
    expected_grade,
):
    import pandas as pd

    import ml_reliability.suite as suite_module
    from ml_reliability.core.result import CheckResult

    def make_scored_check(check_id):
        def check(*args, **kwargs):
            return CheckResult(
                check_id=check_id,
                title=check_id,
                category="test",
                passed=True,
                severity="info",
                score=score,
                message="Boundary test.",
            )

        return check

    monkeypatch.setattr(suite_module, "check_schema", make_scored_check("schema"))
    monkeypatch.setattr(
        suite_module,
        "check_missing_values",
        make_scored_check("missing"),
    )
    monkeypatch.setattr(
        suite_module,
        "check_duplicates",
        make_scored_check("duplicates"),
    )
    monkeypatch.setattr(
        suite_module,
        "check_cardinality",
        make_scored_check("cardinality"),
    )

    report = suite_module.ReliabilitySuite().run(
        X_train=pd.DataFrame({"feature": [1, 2, 3]}),
    )

    assert report.score == pytest.approx(score)
    assert report.grade == expected_grade


def test_suite_handles_non_numeric_check_scores(monkeypatch):
    invalid_check = CheckResult(
        check_id="invalid_score",
        title="Invalid score",
        category="test",
        passed=True,
        severity="info",
        score="invalid",
        message="Test check with an invalid score",
        evidence={},
        recommendation="",
    )

    monkeypatch.setattr(
        "ml_reliability.suite.check_schema",
        lambda X: invalid_check,
    )

    report = ReliabilitySuite().run(X_train=pd.DataFrame({"feature": [1, 2, 3]}))

    assert report.score is not None
    assert [check.check_id for check in report.invalid_score_checks] == [
        "invalid_score"
    ]


def test_suite_excludes_boolean_check_scores(monkeypatch):
    import pandas as pd

    import ml_reliability.suite as suite_module
    from ml_reliability.core.result import CheckResult

    def make_check(check_id, score):
        def check(*args, **kwargs):
            return CheckResult(
                check_id=check_id,
                title=check_id,
                category="test",
                passed=True,
                severity="info",
                score=score,
                message="Boolean score regression test.",
            )

        return check

    monkeypatch.setattr(
        suite_module,
        "check_schema",
        make_check("schema", True),
    )
    monkeypatch.setattr(
        suite_module,
        "check_missing_values",
        make_check("missing", 80.0),
    )
    monkeypatch.setattr(
        suite_module,
        "check_duplicates",
        make_check("duplicates", 60.0),
    )
    monkeypatch.setattr(
        suite_module,
        "check_cardinality",
        make_check("cardinality", 100.0),
    )

    report = suite_module.ReliabilitySuite().run(
        X_train=pd.DataFrame({"feature": [1, 2, 3]})
    )

    assert report.score == 80.0
    assert report.grade == "B"
    assert [check.check_id for check in report.invalid_score_checks] == ["schema"]


def test_suite_averages_only_valid_scores_when_mixed_with_invalid_scores(
    monkeypatch,
):
    import pandas as pd

    import ml_reliability.suite as suite_module
    from ml_reliability.core.result import CheckResult

    def make_check(check_id, score):
        def check(*args, **kwargs):
            return CheckResult(
                check_id=check_id,
                title=check_id,
                category="test",
                passed=True,
                severity="info",
                score=score,
                message="Mixed score regression test.",
            )

        return check

    checks_to_patch = [
        ("check_schema", "schema", 80.0),
        ("check_missing_values", "missing_values", float("nan")),
        ("check_duplicates", "duplicates", True),
        ("check_cardinality", "cardinality", None),
    ]

    for function_name, check_id, score in checks_to_patch:
        monkeypatch.setattr(
            suite_module,
            function_name,
            make_check(check_id, score),
        )

    report = suite_module.ReliabilitySuite().run(
        X_train=pd.DataFrame({"feature": [1, 2, 3]})
    )

    assert report.score == 80.0
    assert report.grade == "B"
    assert [check.check_id for check in report.invalid_score_checks] == [
        "missing_values",
        "duplicates",
    ]

    assert math.isfinite(report.score)


@pytest.mark.parametrize(
    "stress_tests",
    [
        [],
        "invalid",
        123,
        True,
    ],
)
def test_suite_rejects_non_dict_stress_tests(stress_tests):
    with pytest.raises(
        TypeError,
        match="stress_tests must be a dictionary or None",
    ):
        ReliabilitySuite().run(stress_tests=stress_tests)


@pytest.mark.parametrize(
    "config_key",
    [
        "missing_values",
        "feature_noise",
        "feature_dropout",
        "categorical_corruption",
        "distribution_perturbation",
    ],
)
def test_suite_rejects_non_dict_stress_test_config(
    config_key,
):
    import pandas as pd
    import pytest

    X = pd.DataFrame({"feature": [1, 2, 3, 4]})
    y = pd.Series([0, 1, 0, 1])

    with pytest.raises(
        TypeError,
        match=f"stress_tests\\['{config_key}'\\] must be a dictionary",
    ):
        ReliabilitySuite().run(
            X_train=X,
            y_train=y,
            X_test=X,
            y_test=y,
            model=object(),
            stress_tests={config_key: []},
        )


def test_suite_rejects_unknown_stress_test_config_key():
    import pytest

    with pytest.raises(
        ValueError,
        match="Unsupported stress test",
    ):
        ReliabilitySuite().run(stress_tests={"missing_value": {"rate": 0.1}})


def test_suite_continues_after_a_stress_prediction_failure():
    class Model:
        def predict(self, X):
            if X["feature"].isna().any():
                raise ValueError("Model cannot handle missing values")

            return [1 if value > 5 else 0 for value in X["feature"]]

    X = pd.DataFrame(
        {
            "feature": [10.0, 1.0, 20.0, 2.0],
        }
    )
    y = [1, 0, 1, 0]

    report = ReliabilitySuite(task_type="classification").run(
        X_train=X,
        y_train=y,
        X_test=X,
        y_test=y,
        model=Model(),
        stress_tests={
            "missing_values": {
                "rate": 0.5,
                "random_state": 42,
            },
            "feature_noise": {
                "noise_level": 0.1,
                "random_state": 42,
            },
        },
    )

    assert len(report.stress_tests) == 2

    missing_result, noise_result = report.stress_tests

    assert missing_result.stress_type == "missing_values"
    assert missing_result.passed is False
    assert missing_result.evidence["stress_prediction_failed"] is True

    assert noise_result.stress_type == "feature_noise"
    assert noise_result.passed is True


from ml_reliability.checks.stress import StressTestOutcome
from ml_reliability.suite import ReliabilityReport


def test_report_counts_successful_and_failed_stress_tests():
    successful = StressTestOutcome(
        stress_type="feature_noise",
        baseline_metric=0.9,
        stressed_metric=0.7,
        degradation=0.2,
        passed=True,
    )
    failed = StressTestOutcome(
        stress_type="missing_values",
        baseline_metric=0.9,
        stressed_metric=float("nan"),
        degradation=float("inf"),
        passed=False,
        evidence={"stress_prediction_failed": True},
    )

    report = ReliabilityReport(
        score=90.0,
        grade="A",
        stress_tests=[successful, failed],
    )

    assert report.successful_stress_tests == [successful]
    assert report.failed_stress_tests == [failed]


def test_worst_stress_degradation_ignores_failed_evaluations():
    successful = StressTestOutcome(
        stress_type="feature_noise",
        baseline_metric=0.9,
        stressed_metric=0.7,
        degradation=0.2,
        passed=True,
    )
    failed = StressTestOutcome(
        stress_type="missing_values",
        baseline_metric=0.9,
        stressed_metric=float("nan"),
        degradation=float("inf"),
        passed=False,
    )

    report = ReliabilityReport(
        score=90.0,
        grade="A",
        stress_tests=[successful, failed],
    )

    assert report.worst_stress_degradation == 0.2


def test_worst_stress_degradation_ignores_non_finite_successful_results():
    valid = StressTestOutcome(
        stress_type="feature_noise",
        baseline_metric=0.9,
        stressed_metric=0.7,
        degradation=0.2,
        passed=True,
    )
    invalid = StressTestOutcome(
        stress_type="distribution_perturbation",
        baseline_metric=0.9,
        stressed_metric=0.5,
        degradation=float("nan"),
        passed=True,
    )

    report = ReliabilityReport(
        score=90.0,
        grade="A",
        stress_tests=[valid, invalid],
    )

    assert report.worst_stress_degradation == 0.2


def test_worst_stress_degradation_is_none_without_valid_results():
    invalid = StressTestOutcome(
        stress_type="feature_noise",
        baseline_metric=0.9,
        stressed_metric=float("nan"),
        degradation=float("inf"),
        passed=True,
    )

    report = ReliabilityReport(
        score=90.0,
        grade="A",
        stress_tests=[invalid],
    )

    assert report.worst_stress_degradation is None


def test_suite_rejects_unknown_stress_test_config_option():
    import pytest

    with pytest.raises(
        ValueError,
        match="Unsupported option.*noise_levl",
    ):
        ReliabilitySuite().run(
            stress_tests={
                "feature_noise": {
                    "noise_levl": 0.5,
                }
            }
        )


@pytest.mark.parametrize(
    ("config_key", "invalid_option"),
    [
        ("missing_values", "invalid_option"),
        ("feature_noise", "noise_levl"),
        ("feature_dropout", "invalid_option"),
        ("categorical_corruption", "invalid_option"),
        ("distribution_perturbation", "invalid_option"),
    ],
)
def test_suite_rejects_unknown_options_for_all_stress_tests(
    config_key,
    invalid_option,
):
    with pytest.raises(
        ValueError,
        match="Unsupported option",
    ):
        ReliabilitySuite().run(
            stress_tests={
                config_key: {
                    invalid_option: 0.5,
                }
            }
        )


@pytest.mark.parametrize(
    ("stress_type", "config"),
    [
        ("missing_values", {"rate": -0.1}),
        ("feature_noise", {"noise_level": 5.0}),
        ("categorical_corruption", {"rate": 1.5}),
        ("distribution_perturbation", {"scale": 0.0}),
        ("distribution_perturbation", {"shift": float("nan")}),
    ],
)
def test_suite_rejects_invalid_stress_values_without_model(
    stress_type,
    config,
):
    with pytest.raises((TypeError, ValueError)):
        ReliabilitySuite().run(stress_tests={stress_type: config})


@pytest.mark.parametrize(
    ("stress_type", "config"),
    [
        ("missing_values", {"random_state": -1}),
        ("feature_noise", {"random_state": 1.5}),
        ("categorical_corruption", {"random_state": True}),
        ("distribution_perturbation", {"random_state": -1}),
        ("distribution_perturbation", {"scale": float("inf")}),
        ("distribution_perturbation", {"shift": 10**10000}),
        ("feature_dropout", {"features": "feature_a"}),
        ("feature_dropout", {"features": [1]}),
        ("categorical_corruption", {"columns": "category"}),
        ("categorical_corruption", {"columns": ["category", 1]}),
    ],
)
def test_suite_rejects_invalid_stress_config_values_without_model(
    stress_type,
    config,
):
    with pytest.raises((TypeError, ValueError, OverflowError)):
        ReliabilitySuite().run(stress_tests={stress_type: config})


def test_suite_skips_configured_stress_tests_without_model():
    import pandas as pd

    X_train = pd.DataFrame({"feature": [1.0, 2.0, 3.0, 4.0]})
    y_train = [0, 0, 1, 1]
    X_test = pd.DataFrame({"feature": [1.0, 2.0, 3.0, 4.0]})
    y_test = [0, 0, 1, 1]

    report = ReliabilitySuite().run(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        stress_tests={
            "feature_noise": {"noise_level": 0.1},
        },
    )

    assert report.stress_tests == []


def test_suite_skips_configured_stress_tests_when_training_target_is_invalid():
    class Model:
        def predict(self, X):
            return [0] * len(X)

    X_train = pd.DataFrame({"feature": [1.0, 2.0, 3.0, 4.0]})
    y_train = [0, 1, None, 1]

    X_test = pd.DataFrame({"feature": [1.0, 2.0, 3.0, 4.0]})
    y_test = [0, 1, 0, 1]

    report = ReliabilitySuite().run(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        model=Model(),
        stress_tests={
            "feature_noise": {
                "noise_level": 0.1,
                "random_state": 42,
            },
        },
    )

    target_check = get_check(report, "target.distribution")

    assert target_check.passed is False
    assert report.stress_tests == []


def test_suite_continues_when_schema_check_raises(monkeypatch):
    import pandas as pd

    import ml_reliability.suite as suite_module
    from ml_reliability.core.result import CheckResult

    def broken_schema_check(X):
        raise RuntimeError("Simulated schema implementation failure")

    def successful_check(check_id):
        return CheckResult(
            check_id=check_id,
            title=check_id,
            category="data_quality",
            passed=True,
            severity="info",
            score=100.0,
            message="Check completed.",
            evidence={},
            recommendation="",
        )

    monkeypatch.setattr(suite_module, "check_schema", broken_schema_check)
    monkeypatch.setattr(
        suite_module,
        "check_missing_values",
        lambda X: successful_check("missing_values"),
    )
    monkeypatch.setattr(
        suite_module,
        "check_duplicates",
        lambda X: successful_check("duplicates"),
    )
    monkeypatch.setattr(
        suite_module,
        "check_cardinality",
        lambda X: successful_check("cardinality"),
    )

    report = suite_module.ReliabilitySuite().run(
        X_train=pd.DataFrame({"feature": [1, 2, 3]})
    )

    assert len(report.checks) == 4
    assert report.checks[0].passed is False
    assert "Simulated schema implementation failure" in (report.checks[0].message)
    assert {check.check_id for check in report.checks[1:]} == {
        "missing_values",
        "duplicates",
        "cardinality",
    }
    assert report.checks[0].score is None
    assert report.checks[0].severity == "critical"
    assert report.checks[0].evidence["error_type"] == "RuntimeError"


def test_suite_continues_when_stress_test_raises(monkeypatch):
    import pandas as pd

    import ml_reliability.suite as suite_module
    from ml_reliability.checks.stress import StressTestOutcome

    class DummyModel:
        def predict(self, X):
            return [0, 1, 0, 1][: len(X)]

    def failing_stress_test(**kwargs):
        raise RuntimeError("Simulated stress test failure")

    def successful_stress_test(**kwargs):
        return StressTestOutcome(
            stress_type="feature_noise",
            baseline_metric=0.90,
            stressed_metric=0.85,
            degradation=0.05,
            passed=True,
            evidence={"test": "successful"},
        )

    monkeypatch.setattr(
        suite_module,
        "evaluate_missing_value_stress",
        failing_stress_test,
    )

    monkeypatch.setattr(
        suite_module,
        "evaluate_noise_stress",
        successful_stress_test,
    )

    X_train = pd.DataFrame(
        {
            "feature": [1, 2, 3, 4, 5, 6, 7, 8],
        }
    )

    X_test = pd.DataFrame(
        {
            "feature": [9, 10, 11, 12],
        }
    )

    y_train = pd.Series([0, 1, 0, 1, 0, 1, 0, 1])
    y_test = pd.Series([0, 1, 0, 1])

    report = suite_module.ReliabilitySuite(task_type="classification").run(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        model=DummyModel(),
        stress_tests={
            "missing_values": {"rate": 0.2},
            "feature_noise": {"noise_level": 0.1},
        },
    )

    assert len(report.stress_tests) == 2

    failed = report.stress_tests[0]
    successful = report.stress_tests[1]

    assert failed.stress_type == "missing_values"
    assert failed.passed is False
    assert failed.evidence["error_type"] == "RuntimeError"

    assert successful.stress_type == "feature_noise"
    assert successful.passed is True


def test_suite_runs_stress_tests_without_training_targets():
    import pandas as pd

    class Model:
        def predict(self, X):
            return [0, 1, 0, 1][: len(X)]

    X_test = pd.DataFrame({"feature": [1.0, 2.0, 3.0, 4.0]})
    y_test = [0, 1, 0, 1]

    report = ReliabilitySuite(task_type="classification").run(
        X_test=X_test,
        y_test=y_test,
        model=Model(),
        stress_tests={
            "feature_noise": {
                "noise_level": 0.1,
                "random_state": 42,
            }
        },
    )

    assert len(report.stress_tests) == 1
    assert report.stress_tests[0].stress_type == "feature_noise"
    assert report.stress_tests[0].passed is True


def test_suite_skips_generalization_gap_without_training_features():
    import pandas as pd

    class Model:
        def predict(self, X):
            return [0, 1, 0, 1][: len(X)]

    X_test = pd.DataFrame({"feature": [1, 2, 3, 4]})
    y_train = [0, 1, 0, 1]
    y_test = [0, 1, 0, 1]

    report = ReliabilitySuite(task_type="classification").run(
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        model=Model(),
    )

    assert not any(
        check.check_id == "model.generalization_gap" for check in report.checks
    )


def test_suite_rejects_duplicate_categorical_columns():
    with pytest.raises(
        ValueError,
        match="duplicate requested column names",
    ):
        ReliabilitySuite().run(
            stress_tests={
                "categorical_corruption": {
                    "columns": ["category", "category"],
                    "rate": 0.2,
                }
            }
        )


@pytest.mark.parametrize(
    "stress_tests",
    [
        {"unknown_stress": {}, 123: {}},
        {"feature_noise": {"unknown_option": 0.1, 123: 0.2}},
    ],
)
def test_suite_rejects_mixed_type_stress_configuration_keys(
    stress_tests,
):
    with pytest.raises(ValueError):
        ReliabilitySuite().run(stress_tests=stress_tests)


@pytest.mark.parametrize(
    ("config", "expected_exception"),
    [
        ({"shift": True}, TypeError),
        ({"shift": "2.0"}, TypeError),
        ({"shift": None}, TypeError),
        ({"scale": True}, TypeError),
        ({"scale": "2.0"}, TypeError),
        ({"scale": None}, TypeError),
        ({"shift": float("nan")}, ValueError),
        ({"shift": float("inf")}, ValueError),
        ({"scale": float("nan")}, ValueError),
        ({"scale": float("inf")}, ValueError),
        ({"scale": 0}, ValueError),
        ({"scale": -1}, ValueError),
    ],
)
def test_suite_validates_distribution_parameter_types_and_values(
    config,
    expected_exception,
):
    with pytest.raises(expected_exception):
        ReliabilitySuite().run(stress_tests={"distribution_perturbation": config})
