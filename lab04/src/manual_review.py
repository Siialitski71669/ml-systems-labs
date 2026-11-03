import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

df = pd.read_csv(RESULTS / "label_audit_sample.csv")
issues = []

ranges = {
    "age": (17, 90),
    "hours-per-week": (1, 99),
    "capital-gain": (0, 99999),
    "capital-loss": (0, 4356),
}

for column, (minimum, maximum) in ranges.items():
    values = pd.to_numeric(df[column], errors="coerce")
    invalid = df[values.isna() | (values < minimum) | (values > maximum)]
    for index, row in invalid.iterrows():
        issues.append(f"Row {index}: invalid {column} = {row[column]}")

for column in ["fnlwgt", "education-num"]:
    values = pd.to_numeric(df[column], errors="coerce")
    invalid = df[values.isna() | (values % 1 != 0)]
    for index, row in invalid.iterrows():
        issues.append(f"Row {index}: invalid integer {column} = {row[column]}")

valid_labels = {"<=50K", ">50K"}
for index, row in df.iterrows():
    if str(row["income"]).strip() not in valid_labels:
        issues.append(f"Row {index}: unexpected income label = {row['income']}")

report = [
    "MANUAL LABEL SAMPLE QUALITY REVIEW",
    f"Rows inspected: {len(df)}",
    f"Rule violations found: {len(issues)}",
    "",
    "Findings:"
]

report += issues if issues else ["No violations found in the checked rules."]
report += [
    "",
    "Review scope: numeric ranges, integer fields and target labels.",
    "This automated check does not establish that every demographic or employment category is semantically correct."
]

(RESULTS / "manual_review_report.txt").write_text(
    "\n".join(report), encoding="utf-8"
)

print("\n".join(report))
