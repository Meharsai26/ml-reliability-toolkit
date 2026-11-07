from ml_reliability.core.result import CheckResult
from ml_reliability.suite import ReliabilityReport


def make_check(check_id, passed, severity):
    return CheckResult(
        check_id=check_id,
        title=check_id,
        category="test",
        passed=passed,
        severity=severity,
        score=100.0 if passed else 0.0,
        message="test",
    )


def test_failed_checks_returns_only_failed_checks():
    checks = [
        make_check("check_1", True, "info"),
        make_check("check_2", False, "critical"),
        make_check("check_3", False, "warning"),
    ]

    report = ReliabilityReport(
        score=50.0,
        grade="D",
        checks=checks,
    )

    assert [check.check_id for check in report.failed_checks] == [
        "check_2",
        "check_3",
    ]


def test_passed_checks_returns_only_passed_checks():
    checks = [
        make_check("check_1", True, "info"),
        make_check("check_2", False, "critical"),
        make_check("check_3", True, "info"),
    ]

    report = ReliabilityReport(
        score=66.7,
        grade="C",
        checks=checks,
    )

    assert [check.check_id for check in report.passed_checks] == [
        "check_1",
        "check_3",
    ]


def test_warnings_returns_only_warning_checks():
    checks = [
        make_check("check_1", True, "info"),
        make_check("check_2", False, "warning"),
        make_check("check_3", False, "critical"),
        make_check("check_4", True, "warning"),
    ]

    report = ReliabilityReport(
        score=50.0,
        grade="D",
        checks=checks,
    )

    assert [check.check_id for check in report.warnings] == [
        "check_2",
        "check_4",
    ]


def test_report_properties_are_empty_when_no_checks():
    report = ReliabilityReport(
        score=None,
        grade="N/A",
        checks=[],
    )

    assert report.failed_checks == []
    assert report.passed_checks == []
    assert report.warnings == []


from ml_reliability.checks.stress import StressTestOutcome


def test_stress_test_counts_distinguish_execution_results():
    outcomes = [
        StressTestOutcome(
            stress_type="feature_noise",
            baseline_metric=0.9,
            stressed_metric=0.7,
            degradation=0.2,
            passed=True,
        ),
        StressTestOutcome(
            stress_type="missing_values",
            baseline_metric=0.9,
            stressed_metric=float("nan"),
            degradation=float("inf"),
            passed=False,
        ),
    ]

    report = ReliabilityReport(
        score=90.0,
        grade="A",
        stress_tests=outcomes,
    )

    assert len(report.stress_tests) == 2
    assert sum(result.passed for result in report.stress_tests) == 1
    assert sum(not result.passed for result in report.stress_tests) == 1


def test_successful_stress_tests_returns_only_completed_evaluations():
    outcomes = [
        StressTestOutcome(
            stress_type="feature_noise",
            baseline_metric=0.9,
            stressed_metric=0.7,
            degradation=0.2,
            passed=True,
        ),
        StressTestOutcome(
            stress_type="missing_values",
            baseline_metric=0.9,
            stressed_metric=float("nan"),
            degradation=float("inf"),
            passed=False,
        ),
    ]

    report = ReliabilityReport(
        score=90.0,
        grade="A",
        stress_tests=outcomes,
    )

    assert report.successful_stress_tests == [outcomes[0]]


def test_failed_stress_tests_returns_only_failed_evaluations():
    outcomes = [
        StressTestOutcome(
            stress_type="feature_noise",
            baseline_metric=0.9,
            stressed_metric=0.7,
            degradation=0.2,
            passed=True,
        ),
        StressTestOutcome(
            stress_type="missing_values",
            baseline_metric=0.9,
            stressed_metric=float("nan"),
            degradation=float("inf"),
            passed=False,
        ),
    ]

    report = ReliabilityReport(
        score=90.0,
        grade="A",
        stress_tests=outcomes,
    )

    assert report.failed_stress_tests == [outcomes[1]]


def test_worst_stress_degradation_is_none_without_stress_tests():
    report = ReliabilityReport(
        score=100.0,
        grade="A",
    )

    assert report.worst_stress_degradation is None


def test_worst_stress_degradation_returns_largest_valid_degradation():
    outcomes = [
        StressTestOutcome(
            stress_type="feature_noise",
            baseline_metric=0.9,
            stressed_metric=0.8,
            degradation=0.1,
            passed=True,
        ),
        StressTestOutcome(
            stress_type="missing_values",
            baseline_metric=0.9,
            stressed_metric=0.5,
            degradation=0.4,
            passed=True,
        ),
        StressTestOutcome(
            stress_type="feature_dropout",
            baseline_metric=0.9,
            stressed_metric=0.95,
            degradation=-0.05,
            passed=True,
        ),
    ]

    report = ReliabilityReport(
        score=90.0,
        grade="A",
        stress_tests=outcomes,
    )

    assert report.worst_stress_degradation == 0.4


def test_worst_stress_degradation_ignores_failed_evaluations():
    outcomes = [
        StressTestOutcome(
            stress_type="feature_noise",
            baseline_metric=0.9,
            stressed_metric=0.8,
            degradation=0.1,
            passed=True,
        ),
        StressTestOutcome(
            stress_type="missing_values",
            baseline_metric=0.9,
            stressed_metric=float("nan"),
            degradation=float("inf"),
            passed=False,
        ),
    ]

    report = ReliabilityReport(
        score=90.0,
        grade="A",
        stress_tests=outcomes,
    )

    assert report.worst_stress_degradation == 0.1


def test_worst_stress_degradation_ignores_non_finite_values():
    outcomes = [
        StressTestOutcome(
            stress_type="feature_noise",
            baseline_metric=0.9,
            stressed_metric=float("nan"),
            degradation=float("nan"),
            passed=True,
        ),
        StressTestOutcome(
            stress_type="feature_dropout",
            baseline_metric=0.9,
            stressed_metric=0.7,
            degradation=0.2,
            passed=True,
        ),
    ]

    report = ReliabilityReport(
        score=90.0,
        grade="A",
        stress_tests=outcomes,
    )

    assert report.worst_stress_degradation == 0.2


def test_worst_stress_degradation_is_none_when_no_valid_results():
    outcomes = [
        StressTestOutcome(
            stress_type="missing_values",
            baseline_metric=0.9,
            stressed_metric=float("nan"),
            degradation=float("inf"),
            passed=False,
        ),
        StressTestOutcome(
            stress_type="feature_noise",
            baseline_metric=0.9,
            stressed_metric=float("nan"),
            degradation=float("nan"),
            passed=True,
        ),
    ]

    report = ReliabilityReport(
        score=90.0,
        grade="A",
        stress_tests=outcomes,
    )

    assert report.worst_stress_degradation is None


def test_worst_stress_degradation_when_all_stress_tests_improve_performance():
    outcomes = [
        StressTestOutcome(
            stress_type="feature_noise",
            baseline_metric=0.8,
            stressed_metric=0.9,
            degradation=-0.1,
            passed=True,
        ),
        StressTestOutcome(
            stress_type="feature_dropout",
            baseline_metric=0.8,
            stressed_metric=0.85,
            degradation=-0.05,
            passed=True,
        ),
    ]

    report = ReliabilityReport(
        score=90.0,
        grade="A",
        stress_tests=outcomes,
    )

    assert report.worst_stress_degradation == -0.05


def test_worst_stress_degradation_ignores_non_finite_metrics():
    outcomes = [
        StressTestOutcome(
            stress_type="feature_noise",
            baseline_metric=0.9,
            stressed_metric=float("nan"),
            degradation=0.8,
            passed=True,
        ),
        StressTestOutcome(
            stress_type="feature_dropout",
            baseline_metric=0.9,
            stressed_metric=0.7,
            degradation=0.2,
            passed=True,
        ),
    ]

    report = ReliabilityReport(
        score=90.0,
        grade="A",
        stress_tests=outcomes,
    )

    assert report.worst_stress_degradation == 0.2


def test_invalid_score_checks_identifies_non_finite_scores():
    checks = [
        make_check("valid", True, "info"),
        make_check("nan_score", True, "info"),
        make_check("infinite_score", True, "info"),
    ]
    checks[1].score = float("nan")
    checks[2].score = float("inf")

    report = ReliabilityReport(
        score=100.0,
        grade="A",
        checks=checks,
    )

    assert [check.check_id for check in report.invalid_score_checks] == [
        "nan_score",
        "infinite_score",
    ]


def test_invalid_score_checks_identifies_out_of_range_scores():
    checks = [
        make_check("negative_score", True, "info"),
        make_check("above_max_score", True, "info"),
        make_check("valid_score", True, "info"),
    ]
    checks[0].score = -1.0
    checks[1].score = 101.0
    checks[2].score = 75.0

    report = ReliabilityReport(
        score=75.0,
        grade="B",
        checks=checks,
    )

    assert [check.check_id for check in report.invalid_score_checks] == [
        "negative_score",
        "above_max_score",
    ]


def test_invalid_score_checks_is_empty_for_valid_scores():
    report = ReliabilityReport(
        score=100.0,
        grade="A",
        checks=[make_check("valid", True, "info")],
    )

    assert report.invalid_score_checks == []


def test_invalid_score_checks_identifies_non_numeric_scores():
    checks = [
        make_check("valid", True, "info"),
        make_check("string_score", True, "info"),
    ]
    checks[1].score = "invalid"

    report = ReliabilityReport(
        score=100.0,
        grade="A",
        checks=checks,
    )

    assert [check.check_id for check in report.invalid_score_checks] == ["string_score"]


def test_worst_stress_degradation_ignores_non_numeric_metrics():
    outcomes = [
        StressTestOutcome(
            stress_type="feature_noise",
            baseline_metric=0.9,
            stressed_metric=0.7,
            degradation=0.2,
            passed=True,
        ),
        StressTestOutcome(
            stress_type="missing_values",
            baseline_metric="invalid",
            stressed_metric=0.5,
            degradation=0.4,
            passed=True,
        ),
    ]

    report = ReliabilityReport(
        score=90.0,
        grade="A",
        stress_tests=outcomes,
    )

    assert report.worst_stress_degradation == 0.2


def test_invalid_score_checks_identifies_boolean_scores():
    checks = [
        make_check("valid_score", True, "info"),
        make_check("boolean_score", True, "info"),
    ]
    checks[1].score = True

    report = ReliabilityReport(
        score=100.0,
        grade="A",
        checks=checks,
    )

    assert [check.check_id for check in report.invalid_score_checks] == [
        "boolean_score"
    ]
