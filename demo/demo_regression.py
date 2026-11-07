from pathlib import Path

from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from ml_reliability import ReliabilitySuite


def main():
    print("\n" + "=" * 60)
    print("ML RELIABILITY TOOLKIT - REGRESSION DEMO")
    print("=" * 60)

    # 1. Load dataset
    print("\n[1] DATASET INFORMATION")

    dataset = fetch_california_housing(as_frame=True)

    X = dataset.data
    y = dataset.target

    print("Dataset: California Housing")
    print("Features:", X.shape[1])
    print("Samples:", X.shape[0])
    print("Target: Median house value")

    # 2. Split dataset
    print("\n[2] DATA SPLIT")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    print("Training samples:", len(X_train))
    print("Testing samples:", len(X_test))

    # 3. Train regression model
    print("\n[3] MODEL TRAINING")

    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "regressor",
                RandomForestRegressor(
                    n_estimators=50,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    model.fit(X_train, y_train)

    print("Model: Random Forest Regressor")
    print("Preprocessing: Median Imputation")
    print("Training completed.")

    # 4. Run reliability suite
    print("\n[4] RELIABILITY EVALUATION")

    suite = ReliabilitySuite(task_type="regression")

    report = suite.run(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        model=model,
        stress_tests={
            "missing_values": {
                "rate": 0.1,
                "random_state": 42,
            },
            "feature_noise": {
                "noise_level": 0.1,
                "random_state": 42,
            },
            "distribution_perturbation": {
                "shift": 0.1,
                "scale": 1.1,
                "random_state": 42,
            },
        },
    )

    # 5. Display reliability summary
    print("\n[5] RELIABILITY SUMMARY")

    print("Overall Score:", report.score)
    print("Overall Grade:", report.grade)
    print("Total Checks:", len(report.checks))
    print("Passed Checks:", len(report.passed_checks))
    print("Failed Checks:", len(report.failed_checks))
    print("Warnings:", len(report.warnings))

    # 6. Display individual check results
    print("\n[6] INDIVIDUAL CHECK RESULTS")

    for check in report.checks:
        print("-" * 60)
        print("Check:", check.title)
        print("Category:", check.category)
        print("Status:", "PASSED" if check.passed else "FAILED")
        print("Severity:", check.severity)
        print("Score:", check.score)
        print("Message:", check.message)

        if check.evidence:
            print("Evidence:", check.evidence)

        if check.recommendation:
            print("Recommendation:", check.recommendation)

    # 7. Display stress-test results
    print("\n[7] STRESS TEST RESULTS")

    for result in report.stress_tests:
        print("-" * 60)
        print("Test:", result.stress_type)
        print("Evaluation Completed:", result.passed)
        print("Baseline R²:", result.baseline_metric)
        print("Stressed R²:", result.stressed_metric)
        print("R² Degradation:", result.degradation)

    # 8. Generate reports
    print("\n[8] REPORT GENERATION")

    output_dir = Path("demo_outputs")
    output_dir.mkdir(exist_ok=True)

    json_path = output_dir / "regression_report.json"
    markdown_path = output_dir / "regression_report.md"
    html_path = output_dir / "regression_report.html"

    report.to_json(json_path)
    report.to_markdown(markdown_path)
    report.to_html(html_path)

    print("JSON report:", json_path)
    print("Markdown report:", markdown_path)
    print("HTML report:", html_path)

    print("\n" + "=" * 60)
    print("REGRESSION DEMO COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()
