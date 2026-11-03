
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
