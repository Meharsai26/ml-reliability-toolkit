import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np


def _is_finite_number(value: Any) -> bool:
    """Return whether a supported numeric value is finite."""
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


@dataclass
class StressTestOutcome:
    """Standard result produced by a stress test.

    The ``passed`` field indicates whether the stress evaluation
    completed successfully. It does not indicate that the model
    was robust to the applied stress.

    Metric validation is handled by the calculation and reporting
    layers so failed evaluations can preserve diagnostic results.
    """

    stress_type: str
    baseline_metric: float
    stressed_metric: float
    degradation: float
    passed: bool
    evidence: dict[str, Any] = field(default_factory=dict)


def calculate_degradation(
    baseline_metric: float,
    stressed_metric: float,
) -> float:
    """Calculate degradation from two finite numeric metrics."""

    metrics = (baseline_metric, stressed_metric)

    if any(
        isinstance(metric, (bool, np.bool_))
        or not isinstance(
            metric,
            (int, float, np.integer, np.floating),
        )
        or not _is_finite_number(metric)
        for metric in metrics
    ):
        raise ValueError("baseline_metric and stressed_metric must be finite numbers")

    degradation = float(baseline_metric) - float(stressed_metric)

    if not math.isfinite(degradation):
        raise ValueError("degradation must be finite")

    return round(degradation, 10)


def validate_rate(rate: float) -> float:
    """Validate that a stress rate is finite and between 0 and 1."""

    if (
        isinstance(rate, (bool, np.bool_))
        or not isinstance(rate, (int, float, np.integer, np.floating))
        or not _is_finite_number(rate)
        or not 0.0 <= rate <= 1.0
    ):
        raise ValueError("rate must be between 0.0 and 1.0")

    return rate
