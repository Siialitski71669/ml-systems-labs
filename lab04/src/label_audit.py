import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

df = pd.read_csv(RESULTS / "raw_adult.csv")

# Random sample of 50 rows for manual inspection
sample = df.sample(n=50, random_state=42)
sample.to_csv(RESULTS / "label_audit_sample.csv", index=False)

# Check duplicate features with conflicting income labels
features = df.drop(columns="income")

audit = features.copy()
audit["income"] = df["income"]

conflicts = audit.groupby(list(features.columns), dropna=False)["income"].nunique()
conflict_count = int((conflicts > 1).sum())

report = [
    "LABEL QUALITY AUDIT",
    f"Random sample size: {len(sample)}",
    f"Duplicate feature groups with conflicting labels: {conflict_count}",
    "",
    "The 50 sampled rows were saved for manual inspection.",
    "Review them against the UCI Adult data dictionary."
]

(RESULTS / "label_quality_report.txt").write_text(
    "\n".join(report), encoding="utf-8"
)

print("\n".join(report))
print("\nSaved label_audit_sample.csv and label_quality_report.txt")
