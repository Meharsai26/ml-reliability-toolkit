# ML Reliability Toolkit

A Python toolkit for assessing machine-learning reliability beyond basic accuracy. It combines data-quality validation, model diagnostics, distribution analysis, and controlled stress testing to identify potential issues before deployment.

## Features

### 1. Data Quality Checks
- Missing-value detection
- Duplicate-row detection
- Dataset schema validation
- Feature-cardinality and identifier detection
- Target-distribution analysis

### 2. Model Evaluation
- Classification performance using accuracy
- Regression performance using R-squared
- Generalization-gap analysis
- Classification metrics (precision, recall, and F1-score)
- Regression metrics (MAE and RMSE)
- Classification calibration analysis
- Slice-based performance analysis
- Train/test distribution-shift analysis

### 3. Stress Testing

Evaluate model behavior under controlled input perturbations:

| Stress test | Purpose |
|---|---|
| `missing_values` | Inject missing values into test features |
| `feature_noise` | Add noise to numerical features |
| `feature_dropout` | Mask selected feature values |
| `categorical_corruption` | Corrupt selected categorical values |
| `distribution_perturbation` | Shift and scale numerical features |

Stress tests require a compatible model, test features, test targets, and a recognized task type.

### 4. Reporting

Reporting is available in four formats:

- Console output
- JSON
- Markdown
- HTML

JSON, Markdown, and HTML reports can be exported to files. Reports include the overall assessment, individual check results, recommendations, and stress-test outcomes.

## Requirements

- Python 3.10 or newer
- NumPy 2.0 or newer
- pandas 2.3.3 or newer
- scikit-learn 1.7.2 or newer (required for the included demos)

## Installation

### Install from source

Clone the repository and navigate to the project directory.

Install the package and development dependencies:

```bash
uv sync
```

Install the optional dependencies required for the demos:

```bash
uv sync --extra demo
```

### Install a locally built wheel

```bash
uv pip install ./dist/ml_reliability_toolkit-0.1.0-py3-none-any.whl
```

### PyPI publication

ML Reliability Toolkit is officially published on PyPI. Install the latest available version using:

```bash
pip install ml-reliability-toolkit
```

## Basic Usage

The following example runs applicable data-quality and target-distribution checks.

```python
import pandas as pd
from ml_reliability import ReliabilitySuite

X_train = pd.DataFrame(
    {
        "age": [22, 25, 30, 35, 40],
        "income": [25000, 32000, 45000, 52000, 65000],
    }
)

y_train = pd.Series([0, 0, 1, 1, 1])

report = ReliabilitySuite().run(
    X_train=X_train,
    y_train=y_train,
)

print("Reliability score:", report.score)
print("Grade:", report.grade)

for check in report.checks:
    print(check.title, check.passed, check.message)
```

Only applicable checks are executed. Model-performance checks require a model and test data.

## Model Evaluation

The toolkit supports classification and regression workflows.

Set the task type explicitly when automatic target inference is unsuitable.

```python
suite = ReliabilitySuite(task_type="classification")

report = suite.run(
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    model=trained_model,
)
```

Supported basic performance metrics:

- Classification: Accuracy
- Regression: R-squared

Additional checks cover classification metrics, regression metrics, calibration, generalization gaps, and slice-based analysis.

A successfully calculated metric does not automatically mean the model is reliable. Review the metric values and check results.

## Running the Demos

The project includes two executable examples demonstrating classification and regression workflows.

Install the optional demo dependency first:

```bash
uv sync --extra demo
```

### Classification Demo

Uses the Breast Cancer Wisconsin dataset and a Logistic Regression model.

Run:

```bash
uv run python demo/demo_classification.py
```

The demo:
- Loads and splits the dataset.
- Trains a Logistic Regression pipeline.
- Runs the reliability checks.
- Executes stress tests.
- Displays individual check results and evidence.
- Generates JSON, Markdown, and HTML reports.

### Regression Demo

Uses the California Housing dataset and a Random Forest Regressor.

Run:

```bash
uv run python demo/demo_regression.py
```

The demo:
- Loads and splits the dataset.
- Trains a Random Forest regression pipeline.
- Runs the reliability checks.
- Executes stress tests.
- Displays individual check results and evidence.
- Generates JSON, Markdown, and HTML reports.

**Note:** Both demos create a `demo_outputs/` directory containing their generated reports. The first execution of the regression demo may require an internet connection to download the California Housing dataset.

## Stress Testing

Stress tests can be configured through the `stress_tests` argument.

```python
report = suite.run(
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    model=trained_model,
    stress_tests={
        "missing_values": {
            "rate": 0.1,
            "random_state": 42,
        },
        "feature_noise": {
            "noise_level": 0.1,
            "random_state": 42,
        },
        "feature_dropout": {
            "features": ["age"],
        },
        "categorical_corruption": {
            "columns": ["city"],
            "rate": 0.1,
            "random_state": 42,
        },
        "distribution_perturbation": {
            "shift": 0.2,
            "scale": 1.1,
            "random_state": 42,
        },
    },
)
```

All five configurations are optional. Use only those relevant to your dataset and model.

Stress-test results are reported separately from the overall reliability score.

A stress test's `passed` field indicates whether its evaluation completed successfully. It does not indicate that the model met a robustness threshold.

## Reporting and Export

### Console

```python
from ml_reliability.reporting import format_console_report

print(format_console_report(report))
```

### JSON

```python
report.to_json("reliability_report.json")
```

JSON output contains report details, check results, and stress-test outcomes. Non-finite numeric values are converted to `null` to produce standards-compliant JSON.

### Markdown

```python
report.to_markdown("reliability_report.md")
```

### HTML

```python
report.to_html("reliability_report.html")
```

HTML reports can be opened in a web browser.

## Understanding the Reliability Score

The overall score is calculated as the arithmetic mean of valid scored checks.

A check score must be numeric, finite, non-boolean, and between 0 and 100 to contribute to the overall score.

| Grade | Score |
|---|---:|
| A | 90-100 |
| B | 75-89.99 |
| C | 60-74.99 |
| D | 40-59.99 |
| F | Below 40 |
| N/A | No valid scored checks |

Important considerations:

- Checks without a score do not contribute to the average.
- Invalid check scores are excluded and identified in the report.
- Critical failures are highlighted separately; they do not receive an additional severity-based scoring penalty.
- Stress-test outcomes are excluded from the overall score.
- Regression MAE and RMSE are calculated, but do not contribute to the overall score when no scoring thresholds are configured.
- A high score does not guarantee that a model is accurate, fair, robust, or production-ready.

The score represents the applicable checks performed with the supplied inputs, not a universal measure of model quality.

## Known Limitations

- Cardinality detection is heuristic. High-cardinality numerical features are not necessarily identifiers.
- Distribution perturbation applies synthetic transformations and may not represent realistic production data changes.
- Stress-test outcomes depend on the selected dataset, model, perturbation settings, and random seed.
- Regression metrics require suitable thresholds to contribute to the overall reliability score.
- Reliability results should be interpreted alongside domain-specific validation and appropriate evaluation datasets.

## Development and Testing

Run the complete test suite:

```bash
uv run pytest -q
```

Run Python compilation checks:

```bash
uv run python -m compileall -q src demo
```

## Project Structure

```text
ml-reliability-toolkit/
|-- demo/
|   |-- demo_classification.py
|   `-- demo_regression.py
|-- src/
|   `-- ml_reliability/
|       |-- checks/
|       |   `-- stress/
|       |-- core/
|       |-- reporting.py
|       `-- suite.py
|-- tests/
|-- pyproject.toml
|-- README.md
`-- uv.lock
```

## Current Status

- Package: `ml-reliability-toolkit`
- Version: `0.1.0`
- Python: `>=3.10`
- Latest verified full test run: 479 tests passed.

The toolkit is intended to support pre-deployment assessment. Its results should be interpreted alongside domain-specific validation, appropriate evaluation datasets, and human review. It is not a guarantee of production reliability.
