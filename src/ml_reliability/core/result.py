from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class CheckResult:
    check_id: str
    title: str
    category: str
    passed: bool
    severity: str
    score: float | None
    message: str
    evidence: dict[str, Any] = field(default_factory=dict)
    recommendation: str = ""
