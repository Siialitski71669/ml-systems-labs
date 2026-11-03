# Lab 4 — Data Pipeline and Data Quality

## 1. Goal

The objective of this laboratory was to build a reproducible data processing pipeline and perform a data quality assessment using the UCI Adult Income dataset.

The dataset contains demographic and employment-related information about individuals. The machine learning task was to predict whether a person's annual income exceeds $50,000.

The laboratory focused on several important aspects of data engineering and machine learning:

- Inspecting the structure and quality of a real-world dataset.
- Identifying missing values and duplicate records.
- Validating data types and numerical ranges.
- Preventing data leakage during preprocessing.
- Handling missing values and numerical outliers.
- Performing feature engineering.
- Training and evaluating machine learning models.
- Auditing target labels and reviewing a sample of records.
- Saving the pipeline configuration, processed data, and validation reports.

## 2. Method

### Step 1. Dataset Selection and Loading

The UCI Adult Income dataset was selected for this laboratory. It was loaded from OpenML using the `fetch_openml` functionality.

The dataset contains 48,842 records and 15 columns, including demographic, educational, employment, and financial attributes.

The target variable represents whether an individual's annual income exceeds $50,000.

After loading the dataset, the initial structure was inspected to understand the available features and identify numerical and categorical columns.

### Step 2. Initial Data Inspection

The dataset was examined to determine its dimensions, column names, data types, and general structure.

The inspection confirmed that the dataset contains both numerical and categorical features.

The following columns contained missing values:

| Feature | Missing values |
|---|---:|
| workclass | 2,799 |
| occupation | 2,809 |
| native-country | 857 |

The remaining columns did not contain missing values.

This inspection was important because missing data can affect preprocessing and model performance.

### Step 3. Schema and Data Validation

A validation procedure was implemented to check whether the dataset followed the expected schema.

The checks included:

- Verifying that all expected columns were present.
- Checking the data types of individual columns.
- Identifying schema conversion failures.
- Checking numerical features for invalid values.
- Validating numerical ranges and identifying values outside expected boundaries.

The validation process detected zero schema conversion failures and zero numerical range violations.

The results were saved in `results/validation_report.txt`.

### Step 4. Duplicate Detection

The dataset was checked for exact duplicate records.

A total of 52 duplicate rows were identified in the original dataset.

To avoid removing potentially useful information from the test set, duplicate removal was performed only on the training data.

After splitting the dataset, 24 duplicate rows were removed from the training set.

The resulting training dataset contained 34,165 records, while the testing dataset contained 14,653 records.

### Step 5. Training and Testing Split

The dataset was divided into training and testing subsets.

A 70/30 split was used:

- 70% for training.
- 30% for testing.

Stratified splitting was applied to preserve the distribution of the target classes in both subsets.

A fixed random state was used to make the split reproducible.

The test dataset was kept separate from the training preprocessing operations to reduce the risk of data leakage.

### Step 6. Missing Value Handling

Missing values were handled separately for numerical and categorical features.

For numerical features, missing values were replaced using median imputation.

For categorical features, missing values were replaced using the most frequent category.

The imputation values were learned from the training data and subsequently applied to the test data.

This approach ensured that information from the test set was not used to determine preprocessing parameters.

### Step 7. Numerical Outlier Handling

A custom quantile clipping transformer was implemented to limit the influence of extreme numerical values.

The lower and upper clipping boundaries were calculated using the 1st and 99th percentiles of the training data.

Values below the lower percentile were replaced with the lower boundary, while values above the upper percentile were replaced with the upper boundary.

The calculated boundaries were then applied to the testing data.

This method helped reduce the influence of extreme observations without removing entire records.

### Step 8. Feature Engineering

Two additional numerical features were created to provide the model with potentially useful information.

**1. capital_net**

This feature represents the difference between capital gains and capital losses:

`capital_net = capital-gain - capital-loss`

It combines two financial attributes into one feature representing the net capital amount.

**2. hours_per_age**

This feature represents the ratio between weekly working hours and age:

`hours_per_age = hours-per-week / age`

It provides an additional representation of working intensity relative to age.

The engineered features were included in the extended model to evaluate whether they improved predictive performance.

### Step 9. Preprocessing Pipeline

A preprocessing pipeline was created to organize the data preparation operations.

The pipeline included:

- Numerical missing value imputation.
- Categorical missing value imputation.
- Numerical outlier clipping.
- Encoding of categorical variables.
- Preparation of numerical features for model training.

The preprocessing components were fitted using the training dataset.

The same fitted transformations were then applied to the testing dataset.

The fitted preprocessing pipeline was saved as `results/preprocessor.joblib` so that it could be reused later.

### Step 10. Model Training

Logistic Regression was selected as the classification algorithm.

Two model configurations were evaluated:

- A baseline model using the original features.
- An extended model using the original features and the two engineered features.

The maximum number of iterations was configured to allow the optimization algorithm to converge.

Both models were trained using the same training and testing split to ensure a fair comparison.

### Step 11. Model Evaluation

The models were evaluated using two classification metrics:

- Accuracy — the proportion of correctly classified observations.
- F1-score — the harmonic mean of precision and recall.

The results were:

| Metric | Without derived features | With derived features |
|---|---:|---:|
| Accuracy | 0.853955 | 0.857777 |
| F1-score | 0.660102 | 0.669626 |

The extended model achieved slightly better results for both metrics.

### Step 12. Label Quality Audit

A separate label quality audit was performed to investigate potential inconsistencies in the target variable.

A random sample of 50 records was selected using a fixed random state.

The audit also examined duplicate feature groups with conflicting target labels.

Five conflicting duplicate feature groups were identified.

These records were treated as potential label inconsistencies requiring attention rather than automatically being classified as incorrect.

The sample and audit findings were saved in:

- `results/label_audit_sample.csv`
- `results/label_quality_report.txt`

### Step 13. Manual Review

An additional automated review was performed on a sample of 50 records.

The review checked:

- Numerical value ranges.
- Integer-valued fields.
- Target label values.

No rule violations were detected in the reviewed sample.

The findings were saved in `results/manual_review_report.txt`.

This check covered the implemented validation rules and did not establish that every record was semantically correct.

### Step 14. Reproducibility and Saved Artifacts

The pipeline configuration was stored in `config.yaml`.

The configuration included the random state, training/testing split, outlier percentile boundaries, and model parameters.

A SHA256 hash was calculated and saved in `results/sha256.txt` to support data integrity verification.

The script `src/versions.py` was created to display the Python version and the versions of the main libraries used in the laboratory.

The main generated artifacts included:

- `results/validation_report.txt`
- `results/label_quality_report.txt`
- `results/manual_review_report.txt`
- `results/label_audit_sample.csv`
- `results/processed_train.npz`
- `results/processed_train_labels.csv`
- `results/preprocessor.joblib`
- `results/metrics.csv`
- `results/sha256.txt`

## 3. Results

### Dataset Statistics

| Parameter | Result |
|---|---:|
| Original records | 48,842 |
| Original columns | 15 |
| Exact duplicate rows | 52 |
| Duplicates removed from training data | 24 |
| Training records after duplicate removal | 34,165 |
| Testing records | 14,653 |
| Records in manual review sample | 50 |
| Conflicting duplicate feature groups | 5 |
| Manual review rule violations | 0 |
| Numerical range violations | 0 |
| Schema conversion failures | 0 |

### Model Performance

| Metric | Baseline Model | Extended Model |
|---|---:|---:|
| Accuracy | 0.853955 | 0.857777 |
| F1-score | 0.660102 | 0.669626 |

The extended model improved accuracy by approximately 0.38 percentage points and F1-score by approximately 0.95 percentage points.

### Data Quality Findings

The dataset contained missing values in three categorical columns.

Duplicate detection identified 52 exact duplicate rows in the original dataset. Duplicate removal was restricted to the training subset.

The numerical validation checks did not identify any range violations or schema conversion failures.

The label audit identified five groups of records with identical features but conflicting target labels. These cases indicate potential inconsistencies in the source data.


The automated manual review found no violations of the implemented validation rules in the 50 inspected records.

## 4. Conclusions

1. The laboratory demonstrated how to build a reproducible data pipeline that includes schema validation, missing value handling, duplicate detection, outlier treatment, and preprocessing.

2. Separating training and testing data before fitting preprocessing transformations helped prevent data leakage and ensured a more reliable evaluation.

3. Feature engineering slightly improved both accuracy and F1-score. However, the label audit also revealed conflicting target labels, showing that data quality issues can remain even after automated validation.
