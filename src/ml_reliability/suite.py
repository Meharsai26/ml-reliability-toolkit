import math
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from ml_reliability.checks.calibration import check_calibration
from ml_reliability.checks.cardinality import check_cardinality
from ml_reliability.checks.classification_metrics import check_classification_metrics
from ml_reliability.checks.distribution_shift import check_distribution_shift
from ml_reliability.checks.duplicates import check_duplicates
from ml_reliability.checks.generalization_gap import check_generalization_gap
from ml_reliability.checks.missing_values import check_missing_values
from ml_reliability.checks.model_performance import check_model_performance
from ml_reliability.checks.regression_metrics import check_regression_metrics
from ml_reliability.checks.schema import check_schema
from ml_reliability.checks.slice_analysis import check_slice_analysis
from ml_reliability.checks.stress import (
    StressTestOutcome,
    evaluate_categorical_corruption_stress,
    evaluate_distribution_perturbation_stress,
    evaluate_feature_dropout_stress,
    evaluate_missing_value_stress,
    evaluate_noise_stress,
)
from ml_reliability.checks.stress.base import validate_rate
from ml_reliability.checks.target_distribution import check_target_distribution
from ml_reliability.core.result import CheckResult


@dataclass(slots=True)
class ReliabilityReport:
    score: float | None
    grade: str
    checks: list[CheckResult] = field(default_factory=list)
    stress_tests: list[StressTestOutcome] = field(default_factory=list)

    @property
    def failed_checks(self) -> list[CheckResult]:
        return [check for check in self.checks if not check.passed]

    @property
    def passed_checks(self) -> list[CheckResult]:
        return [check for check in self.checks if check.passed]

    @property
    def warnings(self) -> list[CheckResult]:
        return [check for check in self.checks if check.severity == "warning"]

    @property
    def invalid_score_checks(self) -> list[CheckResult]:
        return [
            check
            for check in self.checks
            if check.score is not None
            and (
                isinstance(check.score, bool)
                or not isinstance(check.score, (int, float))
                or not math.isfinite(check.score)
                or not 0.0 <= check.score <= 100.0
            )
        ]

    @property
    def successful_stress_tests(self) -> list[StressTestOutcome]:
        return [result for result in self.stress_tests if result.passed]

    @property
    def failed_stress_tests(self) -> list[StressTestOutcome]:
        return [result for result in self.stress_tests if not result.passed]

    @property
    def worst_stress_degradation(self) -> float | None:
        degradations = [
            result.degradation
            for result in self.successful_stress_tests
            if all(
                isinstance(metric, (int, float))
                and not isinstance(metric, bool)
                and math.isfinite(metric)
                for metric in (
                    result.baseline_metric,
                    result.stressed_metric,
                    result.degradation,
                )
            )
        ]
        return max(degradations) if degradations else None

    def to_json(self, path: str) -> None:
        """Export this report as a JSON file."""
        from ml_reliability.reporting import write_json_report

        write_json_report(self, path)

    def to_markdown(self, path: str) -> None:
        """Export this report as a Markdown file."""
        from ml_reliability.reporting import write_markdown_report

        write_markdown_report(self, path)

    def to_html(self, path: str) -> None:
        """Export this report as an HTML file."""
        from ml_reliability.reporting import write_html_report

        write_html_report(self, path)


def _run_check_safely(
    check_id,
    title,
    category,
    check_function,
    *args,
    **kwargs,
):
    try:
        return check_function(*args, **kwargs)
    except Exception as exc:  # noqa: BLE001 - Isolate individual check failures.
        return CheckResult(
            check_id=check_id,
            title=title,
            category=category,
            passed=False,
            severity="critical",
            score=None,
            message=(f"Check execution failed with {type(exc).__name__}: {exc}"),
            evidence={
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            },
            recommendation=(
                "Investigate the check implementation and "
                "the input that caused the execution failure."
            ),
        )


def _run_stress_test_safely(
    stress_type,
    evaluator,
    *args,
    **kwargs,
):
    try:
        return evaluator(*args, **kwargs)
    except Exception as exc:  # noqa: BLE001 - Isolate individual stress-test failures.
        return StressTestOutcome(
            stress_type=stress_type,
            baseline_metric=float("nan"),
            stressed_metric=float("nan"),
            degradation=float("inf"),
            passed=False,
            evidence={
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "execution_failed": True,
            },
        )


class ReliabilitySuite:
    def __init__(self, task_type: str | None = None):
        if task_type not in {None, "classification", "regression"}:
            raise ValueError(
                "task_type must be 'classification', 'regression', or None."
            )
        self.task_type = task_type

    def run(
        self,
        *,
        X_train=None,
        y_train=None,
        X_test=None,
        y_test=None,
        model=None,
        slices=None,
        stress_tests=None,
    ) -> ReliabilityReport:
        if stress_tests is not None and not isinstance(stress_tests, dict):
            raise TypeError("stress_tests must be a dictionary or None")

        if stress_tests is not None:
            supported_stress_types = (
                "missing_values",
                "feature_noise",
                "feature_dropout",
                "categorical_corruption",
                "distribution_perturbation",
            )

            unsupported_stress_types = stress_tests.keys() - set(supported_stress_types)

            if unsupported_stress_types:
                unsupported = sorted(
                    unsupported_stress_types,
                    key=lambda value: (type(value).__name__, repr(value)),
                )
                raise ValueError(
                    f"Unsupported stress test(s): {unsupported}. "
                    f"Supported stress tests are: "
                    f"{list(supported_stress_types)}"
                )

            supported_config_options = {
                "missing_values": {"rate", "random_state"},
                "feature_noise": {"noise_level", "random_state"},
                "feature_dropout": {"features"},
                "categorical_corruption": {
                    "columns",
                    "rate",
                    "random_state",
                },
                "distribution_perturbation": {
                    "shift",
                    "scale",
                    "random_state",
                },
            }

            for config_key in supported_stress_types:
                config = stress_tests.get(config_key)

                if config is None:
                    continue

                if not isinstance(config, dict):
                    raise TypeError(
                        f"stress_tests['{config_key}'] must be a dictionary"
                    )

                unsupported_options = (
                    config.keys() - supported_config_options[config_key]
                )

                if unsupported_options:
                    unsupported = sorted(
                        unsupported_options,
                        key=lambda value: (type(value).__name__, repr(value)),
                    )
                    raise ValueError(
                        f"Unsupported option(s) for stress test "
                        f"'{config_key}': {unsupported}. "
                        f"Supported options are: "
                        f"{sorted(supported_config_options[config_key])}"
                    )

                rate_options = {
                    "missing_values": "rate",
                    "feature_noise": "noise_level",
                    "categorical_corruption": "rate",
                }

                if config_key in rate_options:
                    option_name = rate_options[config_key]
                    validate_rate(config.get(option_name, 0.1))

                if config_key in {
                    "missing_values",
                    "feature_noise",
                    "categorical_corruption",
                    "distribution_perturbation",
                }:
                    random_state = config.get("random_state", 42)

                    if (
                        isinstance(random_state, (bool, np.bool_))
                        or not isinstance(random_state, (int, np.integer))
                        or random_state < 0
                    ):
                        raise ValueError("random_state must be a non-negative integer")

                if config_key == "distribution_perturbation":
                    shift = config.get("shift", 0.0)
                    scale = config.get("scale", 1.0)

                    for name, value in (("shift", shift), ("scale", scale)):
                        if isinstance(value, (bool, np.bool_)) or not isinstance(
                            value, (int, float, np.integer, np.floating)
                        ):
                            raise TypeError(f"{name} must be a numeric value")

                        try:
                            is_finite = math.isfinite(value)
                        except OverflowError:
                            is_finite = False

                        if not is_finite:
                            raise ValueError(f"{name} must be a finite number")

                    if scale <= 0:
                        raise ValueError("scale must be a finite number greater than 0")

                if config_key == "feature_dropout":
                    features = config.get("features", [])

                    if isinstance(features, (str, bytes)) or not isinstance(
                        features, Sequence
                    ):
                        raise TypeError("features must be a sequence of column names")

                    if any(not isinstance(feature, str) for feature in features):
                        raise ValueError("features must contain only column names")

                if config_key == "categorical_corruption":
                    columns = config.get("columns", [])

                    if isinstance(columns, (str, bytes)) or not isinstance(
                        columns, Sequence
                    ):
                        raise TypeError("columns must be a sequence of column names")

                    if any(not isinstance(column, str) for column in columns):
                        raise ValueError("columns must contain only column names")

                    if len(columns) != len(set(columns)):
                        raise ValueError(
                            "duplicate requested column names are not allowed"
                        )

        checks: list[CheckResult] = []
        stress_results: list[StressTestOutcome] = []
        detected_task_type = self.task_type

        if X_train is not None:
            checks.append(
                _run_check_safely(
                    "data.schema",
                    "Dataset Schema",
                    "data_quality",
                    check_schema,
                    X_train,
                )
            )
            checks.append(
                _run_check_safely(
                    "data.missing_values",
                    "Missing Values",
                    "data_quality",
                    check_missing_values,
                    X_train,
                )
            )
            checks.append(
                _run_check_safely(
                    "data.duplicates",
                    "Duplicate Rows",
                    "data_quality",
                    check_duplicates,
                    X_train,
                )
            )
            checks.append(
                _run_check_safely(
                    "data.cardinality",
                    "Feature Cardinality",
                    "data_quality",
                    check_cardinality,
                    X_train,
                )
            )

        if X_train is not None and X_test is not None:
            checks.append(
                _run_check_safely(
                    "data.distribution_shift",
                    "Train/Test Distribution Shift",
                    "distribution",
                    check_distribution_shift,
                    X_train,
                    X_test,
                )
            )

        if y_train is not None:
            target_check = _run_check_safely(
                "target.distribution",
                "Target Distribution",
                "target_quality",
                check_target_distribution,
                y_train,
                task_type=self.task_type,
            )
            checks.append(target_check)

            detected_task_type = target_check.evidence.get(
                "task_type",
                self.task_type,
            )

        if (
            model is not None
            and X_test is not None
            and y_test is not None
            and detected_task_type in {"classification", "regression"}
        ):
            checks.append(
                _run_check_safely(
                    "model.performance",
                    "Model Performance",
                    "performance",
                    check_model_performance,
                    model=model,
                    X_test=X_test,
                    y_test=y_test,
                    task=detected_task_type,
                )
            )

            if X_train is not None and y_train is not None:
                checks.append(
                    _run_check_safely(
                        "model.generalization_gap",
                        "Generalization Gap",
                        "performance",
                        check_generalization_gap,
                        model=model,
                        X_train=X_train,
                        y_train=y_train,
                        X_test=X_test,
                        y_test=y_test,
                        task=detected_task_type,
                    )
                )

            if detected_task_type == "classification":
                checks.append(
                    _run_check_safely(
                        "model.classification_metrics",
                        "Classification Metrics",
                        "performance",
                        check_classification_metrics,
                        model=model,
                        X_test=X_test,
                        y_test=y_test,
                    )
                )

                checks.append(
                    _run_check_safely(
                        "model.calibration",
                        "Classification Calibration",
                        "performance",
                        check_calibration,
                        model=model,
                        X_test=X_test,
                        y_test=y_test,
                    )
                )

                if slices is not None:
                    checks.append(
                        _run_check_safely(
                            "model.slice_analysis",
                            "Slice Analysis",
                            "performance",
                            check_slice_analysis,
                            model=model,
                            X_test=X_test,
                            y_test=y_test,
                            slices=slices,
                            task=detected_task_type,
                        )
                    )

            if detected_task_type == "regression":
                checks.append(
                    _run_check_safely(
                        "model.regression_metrics",
                        "Regression Metrics",
                        "performance",
                        check_regression_metrics,
                        model=model,
                        X_test=X_test,
                        y_test=y_test,
                    )
                )

        # Stress tests execute independently of ordinary model diagnostics.
        if (
            stress_tests is not None
            and model is not None
            and X_test is not None
            and y_test is not None
            and detected_task_type in {"classification", "regression"}
        ):
            missing_value_config = stress_tests.get("missing_values")

            if missing_value_config is not None:
                stress_results.append(
                    _run_stress_test_safely(
                        "missing_values",
                        evaluate_missing_value_stress,
                        model=model,
                        X_test=X_test,
                        y_test=y_test,
                        task=detected_task_type,
                        rate=missing_value_config.get("rate", 0.1),
                        random_state=missing_value_config.get("random_state", 42),
                    )
                )

            feature_noise_config = stress_tests.get("feature_noise")

            if feature_noise_config is not None:
                stress_results.append(
                    _run_stress_test_safely(
                        "feature_noise",
                        evaluate_noise_stress,
                        model=model,
                        X_test=X_test,
                        y_test=y_test,
                        task=detected_task_type,
                        noise_level=feature_noise_config.get("noise_level", 0.1),
                        random_state=feature_noise_config.get("random_state", 42),
                    )
                )

            feature_dropout_config = stress_tests.get("feature_dropout")

            if feature_dropout_config is not None:
                stress_results.append(
                    _run_stress_test_safely(
                        "feature_dropout",
                        evaluate_feature_dropout_stress,
                        model=model,
                        X_test=X_test,
                        y_test=y_test,
                        task=detected_task_type,
                        features=feature_dropout_config.get("features", []),
                    )
                )

            categorical_corruption_config = stress_tests.get("categorical_corruption")

            if categorical_corruption_config is not None:
                stress_results.append(
                    _run_stress_test_safely(
                        "categorical_corruption",
                        evaluate_categorical_corruption_stress,
                        model=model,
                        X_test=X_test,
                        y_test=y_test,
                        task=detected_task_type,
                        columns=categorical_corruption_config.get("columns", []),
                        rate=categorical_corruption_config.get("rate", 0.1),
                        random_state=categorical_corruption_config.get(
                            "random_state", 42
                        ),
                    )
                )

            distribution_perturbation_config = stress_tests.get(
                "distribution_perturbation"
            )

            if distribution_perturbation_config is not None:
                stress_results.append(
                    _run_stress_test_safely(
                        "distribution_perturbation",
                        evaluate_distribution_perturbation_stress,
                        model=model,
                        X_test=X_test,
                        y_test=y_test,
                        task=detected_task_type,
                        shift=distribution_perturbation_config.get("shift", 0.0),
                        scale=distribution_perturbation_config.get("scale", 1.0),
                        random_state=distribution_perturbation_config.get(
                            "random_state", 42
                        ),
                    )
                )

        scored_checks = [
            check
            for check in checks
            if check.score is not None
            and not isinstance(check.score, bool)
            and isinstance(check.score, (int, float))
            and math.isfinite(check.score)
            and 0.0 <= check.score <= 100.0
        ]

        if not scored_checks:
            return ReliabilityReport(
                score=None,
                grade="N/A",
                checks=checks,
                stress_tests=stress_results,
            )

        score = sum(check.score for check in scored_checks) / len(scored_checks)

        if score >= 90:
            grade = "A"
        elif score >= 75:
            grade = "B"
        elif score >= 60:
            grade = "C"
        elif score >= 40:
            grade = "D"
        else:
            grade = "F"

        return ReliabilityReport(
            score=score,
            grade=grade,
            checks=checks,
            stress_tests=stress_results,
        )
