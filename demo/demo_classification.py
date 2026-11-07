from pathlib import Path

from sklearn.datasets import load_breast_cancer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ml_reliability import ReliabilitySuite

# 1. Load the classification dataset
print("\n" + "=" * 60)
print("ML RELIABILITY TOOLKIT - CLASSIFICATION DEMO")
print("=" * 60)

dataset = load_breast_cancer(as_frame=True)

X = dataset.data
y = dataset.target

print("\n[1] DATASET INFORMATION")
print("Dataset:", "Breast Cancer Wisconsin")
print("Features:", X.shape[1])
print("Samples:", X.shape[0])
print("Target classes:", y.nunique())


# 2. Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

print("\n[2] DATA SPLIT")
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# 3. Train a classification model
model = Pipeline(
    [
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=2000)),
    ]
)

model.fit(X_train, y_train)

print("\n[3] MODEL TRAINING")
print("Model: Logistic Regression")
print("Preprocessing: Median Imputation + Standard Scaling")
print("Training completed.")


# 4. Evaluate model reliability
suite = ReliabilitySuite(task_type="classification")

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


# 5. Display reliability results
print("\n[4] RELIABILITY SUMMARY")
print("Overall Score:", report.score)
print("Overall Grade:", report.grade)
print("Total Checks:", len(report.checks))
print("Passed Checks:", len(report.passed_checks))
print("Failed Checks:", len(report.failed_checks))
print("Warnings:", len(report.warnings))

print("\n[5] INDIVIDUAL CHECK RESULTS")

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


# 6. Display stress test results
print("\n[6] STRESS TEST RESULTS")

for result in report.stress_tests:
    print("-" * 60)
    print("Test:", result.stress_type)
    print("Evaluation Completed:", result.passed)
    print("Baseline Accuracy:", result.baseline_metric)
    print("Stressed Accuracy:", result.stressed_metric)
    print("Accuracy Degradation:", result.degradation)


# 7. Export reports
output_dir = Path("demo_outputs")
output_dir.mkdir(exist_ok=True)

report.to_json(output_dir / "classification_report.json")
report.to_markdown(output_dir / "classification_report.md")
report.to_html(output_dir / "classification_report.html")

print("\n[7] REPORT GENERATION")
print("JSON report: demo_outputs/classification_report.json")
print("Markdown report: demo_outputs/classification_report.md")
print("HTML report: demo_outputs/classification_report.html")

print("\n" + "=" * 60)
print("CLASSIFICATION DEMO COMPLETED")
print("=" * 60)
