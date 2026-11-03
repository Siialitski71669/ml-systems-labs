import pandas as pd
from sklearn.datasets import fetch_openml
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

# 1. Problem definition
problem_definition = """
Problem Definition

The goal of this project is to predict whether an adult earns more than
$50,000 per year using the UCI Adult dataset.

The model uses 14 input features describing personal and employment
information. The target variable is income, with two classes:
<=50K and >50K.

The models will be evaluated using accuracy and F1-score.
The pipeline must prevent data leakage by fitting preprocessing
steps only on training data. It must also be reproducible, using
a fixed random seed and a documented configuration.

The dataset contains numerical and categorical features.
Missing values, invalid ranges, duplicates and label quality
will be checked before model training.
"""

(ROOT / "problem_definition.md").write_text(
    problem_definition, encoding="utf-8"
)

# 2. Data ingestion
print("Loading UCI Adult dataset...")

adult = fetch_openml(name="adult", version=2, as_frame=True)
df = adult.frame.copy()
df = df.rename(columns={"class": "income"})

print(f"Rows: {df.shape[0]}")
print(f"Columns: {df.shape[1]}")

# Save the raw data without cleaning or modification
df.to_csv(RESULTS / "raw_adult.csv", index=False)

# 3. Validation
schema = {
    "age": "int",
    "workclass": "category",
    "fnlwgt": "int",
    "education": "category",
    "education-num": "int",
    "marital-status": "category",
    "occupation": "category",
    "relationship": "category",
    "race": "category",
    "sex": "category",
    "capital-gain": "int",
    "capital-loss": "int",
    "hours-per-week": "int",
    "native-country": "category",
    "income": "category",
}

ranges = {
    "age": (17, 90),
    "hours-per-week": (1, 99),
    "capital-gain": (0, 99999),
    "capital-loss": (0, 4356),
}

report = []

report.append("ADULT DATA VALIDATION REPORT")
report.append(f"Rows: {len(df)}")
report.append(f"Columns: {len(df.columns)}")

report.append("\nMISSING VALUES")
df_validation = df.replace("?", pd.NA)

for column in df.columns:
    missing = int(df_validation[column].isna().sum())
    report.append(f"{column}: {missing}")

report.append("\nRANGE VIOLATIONS")
for column, (minimum, maximum) in ranges.items():
    values = pd.to_numeric(df_validation[column], errors="coerce")
    invalid = int(((values < minimum) | (values > maximum)).sum())
    report.append(
        f"{column}: {invalid} values outside {minimum}-{maximum}"
    )

report.append("\nSCHEMA CHECK")
for column, expected_type in schema.items():
    if column not in df.columns:
        report.append(f"{column}: MISSING COLUMN")
        continue

    if expected_type == "int":
        converted = pd.to_numeric(df_validation[column], errors="coerce")
        invalid = int(converted.isna().sum())
        report.append(f"{column}: numeric conversion failures = {invalid}")
    else:
        report.append(f"{column}: categorical column present")

report.append("\nDUPLICATES")
report.append(f"Exact duplicate rows: {int(df.duplicated().sum())}")

report_text = "\n".join(report)
(RESULTS / "validation_report.txt").write_text(
    report_text, encoding="utf-8"
)

print("\n".join(report))
print("\nRaw data and validation report saved.")

