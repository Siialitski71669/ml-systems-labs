# Lab 4 — Data Pipeline and Data Quality

## 1. Goal

The goal of this laboratory was to build a reproducible machine learning data pipeline using the UCI Adult Income dataset. The task was to predict whether a person's annual income exceeds $50K while ensuring data quality and preventing data leakage.

## 2. Method

- Loaded the UCI Adult dataset using fetch_openml.
- Validated the dataset schema, missing values, and numeric ranges.
- Split the data into training (70%) and testing (30%) sets using stratification.
- Handled missing values using median imputation for numerical features and most-frequent imputation for categorical features.
- Removed exact duplicate rows from the training set only.
- Applied 1st and 99th percentile clipping to numerical features.
- Created two additional features: capital_net and hours_per_age.
- Compared Logistic Regression models with and without derived features.
- Performed a label quality audit using a random sample of 50 records.
- Saved the fitted preprocessing pipeline and SHA256 hash.

## 3. Results

| Metric | Without derived features | With derived features |
|---|---:|---:|
| Accuracy | 0.853955 | 0.857777 |
| F1-score | 0.660102 | 0.669626 |

Additional results:
- Original dataset: 48,842 rows and 15 columns.
- Training samples after duplicate removal: 34,165.
- Testing samples: 14,653.
- Exact duplicate rows in original dataset: 52.
- Duplicate rows removed from training data: 24.
- Conflicting duplicate feature groups: 5.
- Random sample reviewed: 50 records.
- Rule violations found: 0.

## 4. Conclusions

1. Data validation found no numeric range violations, although some categorical features contained missing values.
2. Stratified splitting and fitting preprocessing transformations only on training data helped prevent data leakage.
3. Feature engineering slightly improved both accuracy and F1-score.
