"""Serialization utilities for ML Reliability Toolkit reports."""

from __future__ import annotations

import html
import json
import math
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np


def _normalize_for_json(value: Any) -> Any:
    """Convert supported values to JSON-compatible data.

    Non-finite floats become None, which JSON serializes as null.
    NumPy scalar values are converted to their Python equivalents.
    The input objects are not modified.
    """
    if isinstance(value, np.generic):
        return _normalize_for_json(value.item())

    if value is None or isinstance(value, (str, bool, int)):
        return value

    if isinstance(value, float):
        return value if math.isfinite(value) else None

    if isinstance(value, dict):
        return {str(key): _normalize_for_json(item) for key, item in value.items()}

    if isinstance(value, (list, tuple)):
        return [_normalize_for_json(item) for item in value]

    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _normalize_for_json(getattr(value, field.name))
            for field in fields(value)
        }

    raise TypeError(
        f"Object of type {type(value).__name__} "
        "is not supported by the reporting JSON serializer."
    )


def report_to_dict(report: Any) -> dict[str, Any]:
    """Convert a reliability report into a JSON-compatible dictionary."""
    normalized = _normalize_for_json(report)

    if not isinstance(normalized, dict):
        raise TypeError("A reliability report must serialize to a dictionary.")

    if hasattr(report, "invalid_score_checks"):
        normalized["invalid_score_checks"] = [
            check.check_id for check in report.invalid_score_checks
        ]

    return normalized


def write_json_report(
    report: Any,
    path: str | Path,
) -> None:
    """Write a reliability report as strict, indented JSON."""
    output_path = Path(path)
    data = report_to_dict(report)

    with output_path.open("w", encoding="utf-8", newline="\n") as file:
        json.dump(
            data,
            file,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        file.write("\n")


def _format_console_number(value: Any, precision: int = 4) -> str:
    """Format finite numbers consistently for console reports."""
    if (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    ):
        return f"{value:.{precision}f}"

    return "N/A"


def format_console_report(report: Any) -> str:
    """Format a reliability report as readable console text."""
    critical_failures = [
        check
        for check in report.checks
        if not check.passed and check.severity == "critical"
    ]

    lines = [
        "=" * 60,
        "ML RELIABILITY REPORT",
        "=" * 60,
        "",
        "OVERALL ASSESSMENT",
        "-" * 60,
        f"Score: {_format_console_number(report.score, precision=2)}",
        f"Grade: {report.grade}",
    ]

    if critical_failures:
        lines.extend(
            [
                "",
                f"CRITICAL ISSUES DETECTED: {len(critical_failures)}",
                "Review critical failures before relying on this assessment.",
            ]
        )

    lines.extend(
        [
            "",
            "CHECK SUMMARY",
            "-" * 60,
            f"Total checks: {len(report.checks)}",
            f"Passed: {len(report.passed_checks)}",
            f"Failed: {len(report.failed_checks)}",
            f"Warnings: {len(report.warnings)}",
            "",
            "CHECK DETAILS",
            "-" * 60,
        ]
    )

    if report.checks:
        for check in report.checks:
            status = "PASSED" if check.passed else "FAILED"

            lines.extend(
                [
                    f"[{status}] {check.title}",
                    f"  Category: {check.category}",
                    f"  Severity: {check.severity}",
                    f"  Score: {_format_check_score(check.score)}",
                    f"  Message: {check.message}",
                ]
            )
            if not check.passed and check.recommendation:
                lines.append(f"  Recommendation: {check.recommendation}")

            lines.append("")
    else:
        lines.append("No checks to display.")
        lines.append("")

    lines.extend(
        [
            "STRESS TEST SUMMARY",
            "-" * 60,
            f"Total stress tests: {len(report.stress_tests)}",
            f"Successful: {len(report.successful_stress_tests)}",
            f"Failed: {len(report.failed_stress_tests)}",
        ]
    )

    worst_degradation = report.worst_stress_degradation

    lines.append(
        "Worst valid degradation: "
        + (f"{worst_degradation:.4f}" if worst_degradation is not None else "N/A")
    )

    lines.append("")
    lines.append("STRESS TEST DETAILS")
    lines.append("-" * 60)

    if report.stress_tests:
        for result in report.stress_tests:
            status = "COMPLETED" if result.passed else "FAILED"

            lines.extend(
                [
                    f"[{status}] {result.stress_type}",
                    (
                        f"  Baseline metric: "
                        f"{_format_console_number(result.baseline_metric)}"
                    ),
                    (
                        f"  Stressed metric: "
                        f"{_format_console_number(result.stressed_metric)}"
                    ),
                    f"  Degradation: {_format_console_number(result.degradation)}",
                    "",
                ]
            )
    else:
        lines.append("No stress tests to display.")
        lines.append("")

    lines.append("=" * 60)

    return "\n".join(lines)


def _escape_markdown_table_cell(value: Any) -> str:
    """Escape text for safe use inside a Markdown table cell."""
    text = str(value)
    text = text.replace("\\", "\\\\")
    text = text.replace("|", "\\|")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\n", "<br>")
    return text


def _escape_markdown_text(value: Any) -> str:
    """Escape special Markdown syntax in ordinary text."""
    text = str(value)

    replacements = {
        "\\": "&#92;",
        "`": "&#96;",
        "*": "&#42;",
        "_": "&#95;",
        "{": "&#123;",
        "}": "&#125;",
        "[": "&#91;",
        "]": "&#93;",
        "<": "&lt;",
        ">": "&gt;",
        "|": "\\|",
    }

    return "".join(replacements.get(char, char) for char in text)


def _format_check_score(value: Any) -> str:
    """Format a check score while identifying invalid values."""
    if value is None:
        return "N/A"

    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not 0.0 <= value <= 100.0
    ):
        return "Invalid score"

    return f"{value:.2f}"


def format_markdown_report(report: Any) -> str:
    """Format a reliability report as a Markdown document."""
    score = (
        f"{report.score:.2f}"
        if isinstance(report.score, (int, float))
        and not isinstance(report.score, bool)
        and math.isfinite(report.score)
        else "N/A"
    )

    lines = [
        "# ML Reliability Report",
        "",
        "## Overall Assessment",
        "",
        f"- **Score:** {score}",
        f"- **Grade:** {report.grade}",
        "",
        "## Check Summary",
        "",
        f"- **Total checks:** {len(report.checks)}",
        f"- **Passed:** {len(report.passed_checks)}",
        f"- **Failed:** {len(report.failed_checks)}",
        f"- **Warnings:** {len(report.warnings)}",
        "",
        "## Check Results",
        "",
    ]
    critical_failures = [
        check
        for check in report.checks
        if not check.passed and check.severity == "critical"
    ]

    if critical_failures:
        warning_lines = [
            f"**CRITICAL ISSUES DETECTED: {len(critical_failures)}**",
            "",
            "Review critical failures before relying on this assessment.",
            "",
        ]
        lines[6:6] = warning_lines

    if report.checks:
        lines.extend(
            [
                "| Check | Category | Status | Severity | Score |",
                "|---|---|---|---|---:|",
            ]
        )

        for check in report.checks:
            status = "Passed" if check.passed else "Failed"
            check_score = _format_check_score(check.score)

            lines.append(
                f"| {_escape_markdown_table_cell(check.title)} "
                f"| {_escape_markdown_table_cell(check.category)} "
                f"| {status} "
                f"| {_escape_markdown_table_cell(check.severity)} "
                f"| {check_score} |"
            )

            lines.extend(
                [
                    "",
                    f"**Message:** {_escape_markdown_text(check.message)}",
                ]
            )

            if not check.passed and check.recommendation:
                lines.append(
                    f"**Recommendation:** {_escape_markdown_text(check.recommendation)}"
                )

            lines.append("")
    else:
        lines.extend(["No checks to display.", ""])

    lines.extend(
        [
            "## Stress Test Summary",
            "",
            f"- **Total stress tests:** {len(report.stress_tests)}",
            f"- **Completed:** {len(report.successful_stress_tests)}",
            f"- **Failed:** {len(report.failed_stress_tests)}",
        ]
    )

    worst_degradation = report.worst_stress_degradation
    degradation_text = (
        f"{worst_degradation:.4f}" if worst_degradation is not None else "N/A"
    )

    lines.extend(
        [
            f"- **Worst valid degradation:** {degradation_text}",
            "",
            "## Stress Test Results",
            "",
        ]
    )

    if report.stress_tests:
        lines.extend(
            [
                "| Stress Test | Status | Baseline | Stressed | Degradation |",
                "|---|---|---:|---:|---:|",
            ]
        )

        for result in report.stress_tests:
            status = "Completed" if result.passed else "Failed"

            def format_metric(value: Any) -> str:
                if (
                    isinstance(value, (int, float))
                    and not isinstance(value, bool)
                    and math.isfinite(value)
                ):
                    return f"{value:.4f}"
                return "N/A"

            lines.append(
                f"| {_escape_markdown_table_cell(result.stress_type)} "
                f"| {status} "
                f"| {format_metric(result.baseline_metric)} "
                f"| {format_metric(result.stressed_metric)} "
                f"| {format_metric(result.degradation)} |"
            )
    else:
        lines.append("No stress tests to display.")

    lines.append("")
    return "\n".join(lines)


def write_markdown_report(
    report: Any,
    path: str | Path,
) -> None:
    """Write a reliability report as a Markdown file."""
    output_path = Path(path)
    content = format_markdown_report(report)

    with output_path.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as file:
        file.write(content)
        file.write("\n")


def format_html_report(report: Any) -> str:
    """Format a reliability report as a standalone HTML document."""

    def escape(value: Any) -> str:
        """Escape dynamic values before inserting them into HTML."""
        return html.escape(str(value), quote=True)

    def format_number(value: Any, precision: int = 2) -> str:
        """Format finite numeric values, returning N/A otherwise."""
        if (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
        ):
            return f"{value:.{precision}f}"
        return "N/A"

    score = format_number(report.score)

    lines = [
        "<!DOCTYPE html>",
        '<html lang="en">',
        "<head>",
        '    <meta charset="UTF-8">',
        '    <meta name="viewport" content="width=device-width, initial-scale=1.0">',
        "    <title>ML Reliability Report</title>",
        "    <style>",
        (
            "        body { font-family: Arial, sans-serif; max-width: 1100px; "
            "margin: 2rem auto; padding: 0 1rem; color: #222; }"
        ),
        "        h1, h2 { color: #17365d; }",
        "        .summary { display: flex; gap: 1rem; flex-wrap: wrap; }",
        (
            "        .card { border: 1px solid #ddd; border-radius: 8px; "
            "padding: 1rem; min-width: 140px; }"
        ),
        (
            "        table { width: 100%; border-collapse: collapse; "
            "margin: 1rem 0 2rem; }"
        ),
        (
            "        th, td { border: 1px solid #ddd; padding: 0.65rem; "
            "text-align: left; overflow-wrap: anywhere; }"
        ),
        "        th { background: #f2f5f9; }",
        "        .passed { color: #176b35; }",
        "        .failed { color: #a32121; }",
        "        .details { margin: 0.75rem 0 1.25rem; }",
        (
            "        @media (max-width: 600px) { "
            "body { margin: 1rem auto; } table { font-size: 0.85rem; } }"
        ),
        "    </style>",
        "</head>",
        "<body>",
        "    <h1>ML Reliability Report</h1>",
        "    <section aria-labelledby='assessment-heading'>",
        "        <h2 id='assessment-heading'>Overall Assessment</h2>",
        "        <div class='summary'>",
        f"            <div class='card'><strong>Score</strong><p>{score}</p></div>",
        f"            <div class='card'><strong>Grade</strong><p>{escape(report.grade)}</p></div>",
        "        </div>",
        "    </section>",
        "    <section aria-labelledby='checks-heading'>",
        "        <h2 id='checks-heading'>Check Summary</h2>",
        (
            f"        <p>{len(report.checks)} checks | "
            f"{len(report.passed_checks)} passed | "
            f"{len(report.failed_checks)} failed | "
            f"{len(report.warnings)} warnings</p>"
        ),
        "        <h2>Check Results</h2>",
    ]
    critical_failures = [
        check
        for check in report.checks
        if not check.passed and check.severity == "critical"
    ]

    if critical_failures:
        lines.extend(
            [
                "        <div class='critical-warning' role='alert'>",
                (
                    f"            <strong>CRITICAL ISSUES DETECTED: "
                    f"{len(critical_failures)}</strong>"
                ),
                (
                    "            <p>Review critical failures before relying "
                    "on this assessment.</p>"
                ),
                "        </div>",
            ]
        )

    if report.checks:
        lines.extend(
            [
                "        <table>",
                (
                    "            <thead><tr><th>Check</th><th>Category</th>"
                    "<th>Status</th><th>Severity</th><th>Score</th></tr></thead>"
                ),
                "            <tbody>",
            ]
        )

        for check in report.checks:
            status = "Passed" if check.passed else "Failed"
            status_class = "passed" if check.passed else "failed"

            lines.extend(
                [
                    "                <tr>",
                    f"                    <td>{escape(check.title)}</td>",
                    f"                    <td>{escape(check.category)}</td>",
                    f"                    <td class='{status_class}'>{status}</td>",
                    f"                    <td>{escape(check.severity)}</td>",
                    f"                    <td>{escape(_format_check_score(check.score))}</td>",
                    "                </tr>",
                    "                <tr><td colspan='5'>",
                    (
                        f"                    <div class='details'><strong>Message:</strong> "
                        f"{escape(check.message)}"
                    ),
                ]
            )

            if not check.passed and check.recommendation:
                lines.append(
                    f"<br><strong>Recommendation:</strong> "
                    f"{escape(check.recommendation)}"
                )

            lines.extend(
                [
                    "                    </div>",
                    "                </td></tr>",
                ]
            )

        lines.extend(
            [
                "            </tbody>",
                "        </table>",
            ]
        )
    else:
        lines.append("        <p>No checks to display.</p>")

    lines.extend(
        [
            "    </section>",
            "    <section aria-labelledby='stress-heading'>",
            "        <h2 id='stress-heading'>Stress Test Summary</h2>",
            (
                f"        <p>{len(report.stress_tests)} stress tests | "
                f"{len(report.successful_stress_tests)} completed | "
                f"{len(report.failed_stress_tests)} failed</p>"
            ),
            "        <h2>Stress Test Results</h2>",
        ]
    )

    worst_degradation = report.worst_stress_degradation
    lines.append(
        "        <p><strong>Worst valid degradation:</strong> "
        f"{format_number(worst_degradation, 4)}</p>"
    )

    if report.stress_tests:
        lines.extend(
            [
                "        <table>",
                (
                    "            <thead><tr><th>Stress Test</th><th>Status</th>"
                    "<th>Baseline</th><th>Stressed</th><th>Degradation</th></tr></thead>"
                ),
                "            <tbody>",
            ]
        )

        for result in report.stress_tests:
            status = "Completed" if result.passed else "Failed"
            status_class = "passed" if result.passed else "failed"

            lines.extend(
                [
                    "                <tr>",
                    f"                    <td>{escape(result.stress_type)}</td>",
                    f"                    <td class='{status_class}'>{status}</td>",
                    f"                    <td>{format_number(result.baseline_metric, 4)}</td>",
                    f"                    <td>{format_number(result.stressed_metric, 4)}</td>",
                    f"                    <td>{format_number(result.degradation, 4)}</td>",
                    "                </tr>",
                ]
            )

        lines.extend(
            [
                "            </tbody>",
                "        </table>",
            ]
        )
    else:
        lines.append("        <p>No stress tests to display.</p>")

    lines.extend(
        [
            "    </section>",
            "</body>",
            "</html>",
        ]
    )

    return "\n".join(lines)


def write_html_report(
    report: Any,
    path: str | Path,
) -> None:
    """Write a reliability report as a standalone HTML file."""
    output_path = Path(path)
    content = format_html_report(report)

    with output_path.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as file:
        file.write(content)
        file.write("\n")
