from typing import Any

from ml_reliability.core.result import CheckResult


def check_schema(X: Any) -> CheckResult:
    """
    Validate the basic structure of a tabular dataset.
    """

    if X is None:
        return CheckResult(
            check_id="data.schema",
            title="Dataset Schema",
            category="data_quality",
            passed=False,
            severity="critical",
            score=0.0,
            message="Input dataset is missing.",
            recommendation="Provide a valid pandas DataFrame.",
        )

    if not hasattr(X, "shape") or not hasattr(X, "columns") or not hasattr(X, "dtypes"):
        return CheckResult(
            check_id="data.schema",
            title="Dataset Schema",
            category="data_quality",
            passed=False,
            severity="critical",
            score=0.0,
            message="Input must be a pandas DataFrame.",
            recommendation="Provide the dataset as a pandas DataFrame.",
        )

    rows = int(X.shape[0])
    columns = int(X.shape[1])

    if rows == 0:
        return CheckResult(
            check_id="data.schema",
            title="Dataset Schema",
            category="data_quality",
            passed=False,
            severity="critical",
            score=0.0,
            message="Dataset contains zero rows.",
            evidence={
                "rows": rows,
                "columns": columns,
                "column_names": [str(column) for column in X.columns],
                "dtypes": {
                    str(column): str(dtype) for column, dtype in X.dtypes.items()
                },
            },
            recommendation="Provide a dataset containing at least one row.",
        )

    if columns == 0:
        return CheckResult(
            check_id="data.schema",
            title="Dataset Schema",
            category="data_quality",
            passed=False,
            severity="critical",
            score=0.0,
            message="Dataset contains zero columns.",
            evidence={
                "rows": rows,
                "columns": columns,
                "column_names": [],
                "dtypes": {},
            },
            recommendation="Provide a dataset containing at least one column.",
        )

    column_names = [str(column) for column in X.columns]

    dtypes = {str(column): str(dtype) for column, dtype in X.dtypes.items()}

    return CheckResult(
        check_id="data.schema",
        title="Dataset Schema",
        category="data_quality",
        passed=True,
        severity="info",
        score=100.0,
        message="Dataset schema is valid.",
        evidence={
            "rows": rows,
            "columns": columns,
            "column_names": column_names,
            "dtypes": dtypes,
        },
    )
