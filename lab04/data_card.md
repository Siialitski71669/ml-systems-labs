# Data Card: UCI Adult Dataset

## Dataset source
- Name: Adult (Census Income)
- Repository: UCI Machine Learning Repository
- URL: https://archive.ics.uci.edu/dataset/2/adult
- Original authors: Barry Becker and Ronny Kohavi
- DOI: https://doi.org/10.24432/C5XW20
- License: Creative Commons Attribution 4.0 International (CC BY 4.0)

## Dataset size
- Rows: 48,842
- Input features: 14
- Target: income
- Feature types: numerical and categorical
- Missing values: present in workclass, occupation and native-country

## Purpose
The dataset is used to predict whether an individual's annual income exceeds $50,000.

## Known limitations and biases
- The data originates from the 1994 US Census and may not represent current economic conditions.
- The dataset represents the United States population and should not be assumed to generalize to other countries.
- Demographic attributes such as sex and race are included.
- Historical social and economic inequalities may be reflected in the data.
- The target variable is income and does not describe a person's overall financial situation.

## Privacy and ethical considerations
- The dataset contains demographic and employment-related information.
- Race and sex are sensitive attributes and require careful handling.
- Model predictions should not be used as the sole basis for employment, lending or other high-impact decisions.
- The dataset is historical census data, not a live source of personal information.

## Processing
- Train/test split: 70/30, stratified.
- Random state: 42.
- Missing numerical values: median imputation.
- Missing categorical values: most frequent value.
- Numerical outliers: clipped to training 1st and 99th percentiles.
- Exact duplicates: removed from training data only.
- Feature engineering: capital_net and hours_per_age.
- Model: Logistic Regression.
