import json
import math
from html.parser import HTMLParser

import numpy as np

from ml_reliability.checks.stress import StressTestOutcome
from ml_reliability.core.result import CheckResult
from ml_reliability.reporting import (
    format_console_report,
    format_html_report,
    format_markdown_report,
    report_to_dict,
)
from ml_reliability.suite import ReliabilityReport


def make_report():
    checks = [
        CheckResult(
            check_id="missing_values",
            title="Missing Values Check",
            category="data_quality",
            passed=False,
            severity="warning",
            score=75.0,
            message="Some missing values were found.",
            evidence={
                "missing_count": 5,
                "columns": ["age", "salary"],
            },
            recommendation="Investigate missing values.",
        ),
        CheckResult(
            check_id="duplicates",
            title="Duplicate Check",
            category="data_quality",
            passed=True,
            severity="info",
            score=100.0,
            message="No duplicate rows found.",
        ),
    ]

    stress_tests = [
        StressTestOutcome(
            stress_type="feature_noise",
            baseline_metric=0.9,
            stressed_metric=0.7,
            degradation=0.2,
            passed=True,
            evidence={"noise_level": 0.1},
        ),
        StressTestOutcome(
            stress_type="missing_values",
            baseline_metric=0.9,
            stressed_metric=float("nan"),
            degradation=float("inf"),
            passed=False,
        ),
    ]

    return ReliabilityReport(
        score=87.5,
        grade="B",
        checks=checks,
        stress_tests=stress_tests,
    )


def test_to_json_creates_a_valid_json_file(tmp_path):
    report = make_report()
    output_path = tmp_path / "reliability_report.json"

    report.to_json(output_path)

    assert output_path.exists()

    with output_path.open(encoding="utf-8") as file:
        data = json.load(file)

    assert isinstance(data, dict)


def test_json_contains_report_summary_and_all_results(tmp_path):
    report = make_report()
    output_path = tmp_path / "reliability_report.json"

    report.to_json(output_path)

    with output_path.open(encoding="utf-8") as file:
        data = json.load(file)

    assert data["score"] == 87.5
    assert data["grade"] == "B"

    assert len(data["checks"]) == 2
    assert len(data["stress_tests"]) == 2

    assert data["checks"][0]["check_id"] == "missing_values"
    assert data["checks"][0]["passed"] is False

    assert data["stress_tests"][0]["stress_type"] == "feature_noise"
    assert data["stress_tests"][0]["passed"] is True


def test_json_preserves_check_evidence_and_recommendations(tmp_path):
    report = make_report()
    output_path = tmp_path / "reliability_report.json"

    report.to_json(output_path)

    with output_path.open(encoding="utf-8") as file:
        data = json.load(file)

    check = data["checks"][0]

    assert check["evidence"] == {
        "missing_count": 5,
        "columns": ["age", "salary"],
    }
    assert check["recommendation"] == "Investigate missing values."


def test_json_converts_non_finite_numbers_to_null(tmp_path):
    report = make_report()
    report.checks[0].evidence["invalid_measurement"] = float("nan")

    output_path = tmp_path / "reliability_report.json"

    report.to_json(output_path)

    with output_path.open(encoding="utf-8") as file:
        raw_json = file.read()

    # Standard JSON must not contain bare NaN or Infinity tokens.
    assert "NaN" not in raw_json
    assert "Infinity" not in raw_json

    data = json.loads(raw_json)

    assert data["checks"][0]["evidence"]["invalid_measurement"] is None
    assert data["stress_tests"][1]["stressed_metric"] is None
    assert data["stress_tests"][1]["degradation"] is None


def test_json_serialization_does_not_modify_original_report(tmp_path):
    report = make_report()

    original_score = report.score
    original_check_count = len(report.checks)
    original_stressed_metric = report.stress_tests[1].stressed_metric

    report.to_json(tmp_path / "reliability_report.json")

    assert report.score == original_score
    assert len(report.checks) == original_check_count
    assert math.isnan(original_stressed_metric)
    assert math.isnan(report.stress_tests[1].stressed_metric)


def test_json_supports_report_without_checks_or_stress_tests(tmp_path):
    report = ReliabilityReport(
        score=None,
        grade="N/A",
    )
    output_path = tmp_path / "empty_report.json"

    report.to_json(output_path)

    with output_path.open(encoding="utf-8") as file:
        data = json.load(file)

    assert data["score"] is None
    assert data["grade"] == "N/A"
    assert data["checks"] == []
    assert data["stress_tests"] == []


def test_console_report_contains_overall_summary():
    report = make_report()

    output = format_console_report(report)

    assert "87.5" in output
    assert "B" in output
    assert "Missing Values Check" in output
    assert "Duplicate Check" in output


def test_console_report_shows_failed_checks_and_recommendations():
    report = make_report()

    output = format_console_report(report)

    assert "Some missing values were found." in output
    assert "Investigate missing values." in output


def test_console_report_shows_stress_test_summary():
    report = make_report()

    output = format_console_report(report)

    assert "feature_noise" in output
    assert "missing_values" in output
    assert "0.2" in output


def test_console_report_handles_empty_report():
    report = ReliabilityReport(
        score=None,
        grade="N/A",
    )

    output = format_console_report(report)

    assert "N/A" in output
    assert "No checks" in output or "0 checks" in output
    assert "No stress tests" in output or "0 stress tests" in output


def test_markdown_report_contains_summary():
    report = make_report()

    output = format_markdown_report(report)

    assert "# ML Reliability Report" in output
    assert "87.5" in output
    assert "B" in output


def test_markdown_report_contains_check_results():
    report = make_report()

    output = format_markdown_report(report)

    assert "Missing Values Check" in output
    assert "Duplicate Check" in output
    assert "Some missing values were found." in output
    assert "Investigate missing values." in output


def test_markdown_report_contains_stress_test_results():
    report = make_report()

    output = format_markdown_report(report)

    assert "feature_noise" in output
    assert "missing_values" in output
    assert "Stress Test" in output


def test_markdown_report_handles_empty_report():
    report = ReliabilityReport(score=None, grade="N/A")

    output = format_markdown_report(report)

    assert "N/A" in output
    assert "No checks" in output or "0 checks" in output
    assert "No stress tests" in output or "0 stress tests" in output


def test_to_markdown_creates_a_readable_file(tmp_path):
    report = make_report()
    output_path = tmp_path / "reliability_report.md"

    report.to_markdown(output_path)

    assert output_path.exists()

    content = output_path.read_text(encoding="utf-8")

    assert "# ML Reliability Report" in content
    assert "Missing Values Check" in content
    assert "feature_noise" in content


def test_html_report_contains_overall_summary():
    report = make_report()

    output = format_html_report(report)

    assert "<html" in output.lower()
    assert "ML Reliability Report" in output
    assert "87.5" in output
    assert "Grade" in output
    assert "Missing Values Check" in output


def test_html_report_contains_check_details_and_recommendations():
    report = make_report()

    output = format_html_report(report)

    assert "Some missing values were found." in output
    assert "Investigate missing values." in output
    assert "Duplicate Check" in output


def test_html_report_contains_stress_test_results():
    report = make_report()

    output = format_html_report(report)

    assert "feature_noise" in output
    assert "missing_values" in output
    assert "Stress Test" in output


def test_html_report_handles_empty_report():
    report = ReliabilityReport(score=None, grade="N/A")

    output = format_html_report(report)

    assert "<html" in output.lower()
    assert "N/A" in output
    assert "No checks" in output or "0 checks" in output
    assert "No stress tests" in output or "0 stress tests" in output


def test_html_report_escapes_untrusted_text():
    report = make_report()
    report.checks[0].message = "<script>alert('xss')</script>"

    output = format_html_report(report)

    assert "<script>alert('xss')</script>" not in output
    assert "&lt;script&gt;" in output


def test_to_html_creates_a_readable_file(tmp_path):
    report = make_report()
    output_path = tmp_path / "reliability_report.html"

    report.to_html(output_path)

    assert output_path.exists()

    content = output_path.read_text(encoding="utf-8")

    assert "<!DOCTYPE html>" in content
    assert "ML Reliability Report" in content
    assert "Missing Values Check" in content
    assert "feature_noise" in content
    assert content.rstrip().endswith("</html>")


def test_markdown_report_escapes_table_special_characters():
    report = make_report()
    report.checks[0].title = "Feature | Data Quality"
    report.checks[0].category = "data_quality\nextra"
    report.stress_tests[0].stress_type = "feature|noise"

    output = format_markdown_report(report)

    assert "Feature \\| Data Quality" in output
    assert "data_quality<br>extra" in output
    assert "feature\\|noise" in output


def test_console_report_handles_non_finite_numbers():
    report = make_report()
    report.score = float("nan")
    report.stress_tests[0].baseline_metric = float("inf")
    report.stress_tests[0].stressed_metric = float("-inf")
    report.stress_tests[0].degradation = float("nan")

    output = format_console_report(report)

    assert "Score: N/A" in output
    assert "Baseline metric: N/A" in output
    assert "Stressed metric: N/A" in output
    assert "Degradation: N/A" in output


def test_console_report_identifies_critical_failed_check():
    report = make_report()
    report.checks[0].severity = "critical"

    output = format_console_report(report)

    assert "[FAILED] Missing Values Check" in output
    assert "Severity: critical" in output


def test_markdown_report_identifies_critical_failed_check():
    report = make_report()
    report.checks[0].severity = "critical"

    output = format_markdown_report(report)

    assert "Missing Values Check" in output
    assert "critical" in output


def test_console_report_warns_about_failed_critical_checks():
    report = make_report()
    report.checks[0].severity = "critical"

    output = format_console_report(report)

    assert "CRITICAL ISSUES DETECTED: 1" in output
    assert "Score: 87.50" in output
    assert "Grade: B" in output


def test_console_report_counts_multiple_failed_critical_checks():
    report = make_report()
    report.checks[0].severity = "critical"

    report.checks.append(
        CheckResult(
            check_id="schema",
            title="Schema Check",
            category="data_quality",
            passed=False,
            severity="critical",
            score=0.0,
            message="Required columns are missing.",
            recommendation="Correct the input schema.",
        )
    )

    output = format_console_report(report)

    assert "CRITICAL ISSUES DETECTED: 2" in output


def test_console_report_does_not_count_passed_critical_checks():
    report = make_report()
    report.checks[1].severity = "critical"

    output = format_console_report(report)

    assert "CRITICAL ISSUES DETECTED" not in output


def test_console_report_omits_warning_when_no_critical_check_failed():
    report = make_report()

    output = format_console_report(report)

    assert "CRITICAL ISSUES DETECTED" not in output


def test_console_critical_warning_appears_in_overall_assessment():
    report = make_report()
    report.checks[0].severity = "critical"

    output = format_console_report(report)

    warning_position = output.index("CRITICAL ISSUES DETECTED: 1")
    summary_position = output.index("CHECK SUMMARY")

    assert warning_position < summary_position


def test_markdown_report_warns_about_failed_critical_checks():
    report = make_report()
    report.checks[0].severity = "critical"

    output = format_markdown_report(report)

    assert "CRITICAL ISSUES DETECTED: 1" in output
    assert "Score:** 87.50" in output
    assert "Grade:** B" in output


def test_html_report_warns_about_failed_critical_checks():
    report = make_report()
    report.checks[0].severity = "critical"

    output = format_html_report(report)

    assert "CRITICAL ISSUES DETECTED: 1" in output
    assert "Score</strong><p>87.50</p>" in output
    assert "Grade</strong><p>B</p>" in output


def test_html_report_closes_all_sections():
    report = make_report()

    output = format_html_report(report)

    assert output.count("<section ") == output.count("</section>")


def test_markdown_report_handles_special_characters_in_check_details():
    report = make_report()
    report.checks[
        0
    ].message = "Invalid | value\n**Important** [link](https://example.com)"
    report.checks[0].recommendation = "Review | records\n<script>alert('xss')</script>"

    output = format_markdown_report(report)

    assert "Invalid \\| value" in output
    assert "**Important**" not in output
    assert "[link](https://example.com)" not in output
    assert "Review \\| records" in output
    assert "<script>" not in output


def test_markdown_report_identifies_invalid_check_scores():
    report = make_report()

    report.checks[0].score = 150.0
    report.checks[1].score = float("nan")

    output = format_markdown_report(report)

    assert "Invalid score" in output
    assert "150.00" not in output
    assert "nan" not in output.lower()


def test_html_report_identifies_invalid_check_scores():
    report = make_report()

    report.checks[0].score = 150.0
    report.checks[1].score = float("nan")

    output = format_html_report(report)

    assert "Invalid score" in output
    assert "150.00" not in output
    assert "nan" not in output.lower()


def test_all_report_formats_preserve_consistent_assessment():
    report = make_report()
    report.checks[0].severity = "critical"

    json_data = report_to_dict(report)
    console = format_console_report(report)
    markdown = format_markdown_report(report)
    html_output = format_html_report(report)

    # Overall assessment
    assert json_data["score"] == 87.5
    assert json_data["grade"] == "B"

    assert "Score: 87.50" in console
    assert "Grade: B" in console

    assert "Score:** 87.50" in markdown
    assert "Grade:** B" in markdown

    assert "Score</strong><p>87.50</p>" in html_output
    assert "Grade</strong><p>B</p>" in html_output

    # Check summary
    assert len(json_data["checks"]) == 2
    assert "Total checks: 2" in console
    assert "Total checks:** 2" in markdown
    assert "2 checks |" in html_output

    # Critical failures
    assert json_data["checks"][0]["severity"] == "critical"
    assert "CRITICAL ISSUES DETECTED: 1" in console
    assert "CRITICAL ISSUES DETECTED: 1" in markdown
    assert "CRITICAL ISSUES DETECTED: " in html_output

    # Stress tests
    assert len(json_data["stress_tests"]) == 2
    assert "Total stress tests: 2" in console
    assert "Total stress tests:** 2" in markdown
    assert "2 stress tests |" in html_output

    # Stress-test outcomes
    assert json_data["stress_tests"][0]["passed"] is True
    assert json_data["stress_tests"][1]["passed"] is False
    assert "Successful: 1" in console
    assert "Completed:** 1" in markdown
    assert "1 completed | 1 failed" in html_output


def test_json_identifies_invalid_check_scores(tmp_path):
    report = make_report()

    report.checks[0].score = 150.0
    report.checks[1].score = float("nan")

    output_path = tmp_path / "invalid_scores.json"
    report.to_json(output_path)

    with output_path.open(encoding="utf-8") as file:
        data = json.load(file)

    assert data["checks"][0]["score"] == 150.0
    assert data["checks"][1]["score"] is None

    assert data["invalid_score_checks"] == [
        "missing_values",
        "duplicates",
    ]


def test_console_report_identifies_invalid_check_scores():
    report = make_report()

    report.checks[0].score = 150.0
    report.checks[1].score = float("nan")

    report.checks.append(
        CheckResult(
            check_id="unscored",
            title="Unscored Check",
            category="test",
            passed=True,
            severity="info",
            score=None,
            message="No score available.",
        )
    )

    output = format_console_report(report)

    assert output.count("Score: Invalid score") == 2
    assert "Score: 150.0000" not in output
    assert "Score: N/A" in output
    assert "Unscored Check" in output


def test_html_report_escapes_all_dynamic_check_and_stress_fields():
    report = make_report()

    malicious = "<script>alert('xss')</script>"

    report.grade = malicious

    report.checks[0].title = malicious
    report.checks[0].category = malicious
    report.checks[0].severity = malicious
    report.checks[0].message = malicious
    report.checks[0].recommendation = malicious

    report.stress_tests[0].stress_type = malicious

    output = format_html_report(report)

    # No raw script tags should appear.
    assert malicious not in output

    # All dynamic fields should be HTML escaped.
    assert output.count("&lt;script&gt;") >= 7
    assert output.count("&lt;/script&gt;") >= 7


def test_html_report_has_valid_document_structure():
    class ReportHTMLParser(HTMLParser):
        def __init__(self):
            super().__init__()
            self.errors = []
            self.sections = 0
            self.html_opened = False
            self.html_closed = False
            self.body_opened = False
            self.body_closed = False

        def handle_starttag(self, tag, attrs):
            if tag == "html":
                self.html_opened = True
            elif tag == "body":
                self.body_opened = True
            elif tag == "section":
                self.sections += 1

        def handle_endtag(self, tag):
            if tag == "html":
                self.html_closed = True
            elif tag == "body":
                self.body_closed = True
            elif tag == "section":
                self.sections -= 1
                if self.sections < 0:
                    self.errors.append("Unexpected closing section")

    report = make_report()
    output = format_html_report(report)

    parser = ReportHTMLParser()
    parser.feed(output)
    parser.close()

    assert not parser.errors
    assert parser.html_opened
    assert parser.html_closed
    assert parser.body_opened
    assert parser.body_closed
    assert parser.sections == 0


def test_json_serializer_handles_numpy_scalar_values():
    report = make_report()

    report.checks[0].evidence = {
        "boolean": np.bool_(True),
        "integer": np.int64(10),
        "float": np.float64(2.5),
        "non_finite": np.float64(np.nan),
    }

    data = report_to_dict(report)

    assert data["checks"][0]["evidence"] == {
        "boolean": True,
        "integer": 10,
        "float": 2.5,
        "non_finite": None,
    }
