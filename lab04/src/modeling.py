import pandas as pd
import numpy as np
import joblib
import yaml
import hashlib

from pathlib import Path
from scipy import sparse
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.base import BaseEstimator, TransformerMixin

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

with open(ROOT / "config.yaml", encoding="utf-8") as file:
    config = yaml.safe_load(file)

RANDOM_STATE = config["random_state"]
TEST_SIZE = config["test_size"]
MAX_ITER = config["max_iter"]


class QuantileClipper(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.lower_ = np.nanpercentile(
            X, config["outlier_lower_percentile"], axis=0
        )
        self.upper_ = np.nanpercentile(
            X, config["outlier_upper_percentile"], axis=0
        )
        return self

    def transform(self, X):
        return np.clip(X, self.lower_, self.upper_)


df = pd.read_csv(RESULTS / "raw_adult.csv")

X = df.drop(columns="income")
y = df["income"].astype(str).str.strip()

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

X_train = X_train.replace("?", np.nan)
X_test = X_test.replace("?", np.nan)

train_data = X_train.copy()
train_data["income"] = y_train.values

before = len(train_data)
train_data = train_data.drop_duplicates()

duplicates_removed = before - len(train_data)
y_train = train_data.pop("income")
X_train = train_data


def add_features(data):
    data = data.copy()
    data["capital_net"] = data["capital-gain"] - data["capital-loss"]
    data["hours_per_age"] = data["hours-per-week"] / (data["age"] - 16)
    return data


X_train = add_features(X_train)
X_test = add_features(X_test)

numeric_features = [
    "age", "fnlwgt", "education-num", "capital-gain",
    "capital-loss", "hours-per-week", "capital_net", "hours_per_age"
]

categorical_features = [
    "workclass", "education", "marital-status", "occupation",
    "relationship", "race", "sex", "native-country"
]


def create_preprocessor(numeric_columns):
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clipper", QuantileClipper()),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    return ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_columns),
        ("categorical", categorical_pipeline, categorical_features)
    ])


baseline_numeric = numeric_features[:-2]
baseline_preprocessor = create_preprocessor(baseline_numeric)

X_train_base = X_train.drop(columns=["capital_net", "hours_per_age"])
X_test_base = X_test.drop(columns=["capital_net", "hours_per_age"])

X_train_base_processed = baseline_preprocessor.fit_transform(X_train_base)
X_test_base_processed = baseline_preprocessor.transform(X_test_base)

baseline_model = LogisticRegression(max_iter=MAX_ITER)
baseline_model.fit(X_train_base_processed, y_train)
baseline_predictions = baseline_model.predict(X_test_base_processed)

preprocessor = create_preprocessor(numeric_features)

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

model = LogisticRegression(max_iter=MAX_ITER)
model.fit(X_train_processed, y_train)
predictions = model.predict(X_test_processed)

joblib.dump(preprocessor, RESULTS / "preprocessor.joblib")

results = pd.DataFrame([
    {
        "Model": "Without derived features",
        "Accuracy": accuracy_score(y_test, baseline_predictions),
        "F1": f1_score(y_test, baseline_predictions, pos_label=">50K")
    },
    {
        "Model": "With derived features",
        "Accuracy": accuracy_score(y_test, predictions),
        "F1": f1_score(y_test, predictions, pos_label=">50K")
    }
])

results.to_csv(RESULTS / "metrics.csv", index=False)

if sparse.issparse(X_train_processed):
    processed = X_train_processed.tocsr()
    hash_data = (
        processed.data.tobytes()
        + processed.indices.tobytes()
        + processed.indptr.tobytes()
    )
    sparse.save_npz(RESULTS / "processed_train.npz", processed)
else:
    hash_data = np.asarray(X_train_processed).tobytes()
    np.save(RESULTS / "processed_train.npy", X_train_processed)

hash_data += "\n".join(y_train.astype(str)).encode("utf-8")
sha256 = hashlib.sha256(hash_data).hexdigest()

(RESULTS / "sha256.txt").write_text(sha256, encoding="utf-8")
y_train.to_csv(RESULTS / "processed_train_labels.csv", index=False)

print("TRAINING AND TRANSFORMATION RESULTS")
print(f"Training rows after duplicate removal: {len(X_train)}")
print(f"Test rows: {len(X_test)}")
print(f"Duplicates removed from training: {duplicates_removed}")
print(results.to_string(index=False))
print(f"\nSHA256: {sha256}")
print("\nConfiguration loaded from config.yaml.")
