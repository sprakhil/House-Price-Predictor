"""
preprocessing.py
Handles all data preprocessing: validation, encoding, scaling, splitting.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import os

REQUIRED_COLUMNS = [
    "location", "area_sqft", "bedrooms", "bathrooms", "balconies",
    "floors", "parking", "property_age", "furnishing_status",
    "property_type", "distance_from_city_center", "nearby_schools",
    "nearby_hospitals", "crime_rate", "price",
]

CATEGORICAL_FEATURES = ["location", "furnishing_status", "property_type"]
NUMERICAL_FEATURES = [
    "area_sqft", "bedrooms", "bathrooms", "balconies", "floors",
    "parking", "property_age", "distance_from_city_center",
    "nearby_schools", "nearby_hospitals", "crime_rate",
]

TARGET = "price"


def validate_columns(df: pd.DataFrame) -> tuple[bool, list]:
    """Return (is_valid, missing_columns)."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    return len(missing) == 0, missing


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with nulls in critical columns and remove outlier prices."""
    df = df.copy()
    df.dropna(subset=REQUIRED_COLUMNS, inplace=True)

    # Remove extreme price outliers (beyond 1st / 99th percentile)
    lo = df["price"].quantile(0.01)
    hi = df["price"].quantile(0.99)
    df = df[(df["price"] >= lo) & (df["price"] <= hi)]

    return df.reset_index(drop=True)


def build_preprocessor() -> ColumnTransformer:
    """Build a sklearn ColumnTransformer for numeric + categorical features."""
    numeric_transformer  = Pipeline([("scaler", StandardScaler())])
    categorical_transformer = Pipeline([
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer,  NUMERICAL_FEATURES),
        ("cat", categorical_transformer, CATEGORICAL_FEATURES),
    ])
    return preprocessor


def split_data(df: pd.DataFrame, test_size: float = 0.20, random_state: int = 42):
    """Split the DataFrame into train/test sets and return X/y arrays."""
    X = df[NUMERICAL_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test


def preprocess_input(input_dict: dict, preprocessor) -> np.ndarray:
    """Transform a single prediction input dict using a fitted preprocessor."""
    row = pd.DataFrame([input_dict])
    # Ensure column order matches training
    row = row[NUMERICAL_FEATURES + CATEGORICAL_FEATURES]
    return preprocessor.transform(row)


def save_preprocessor(preprocessor, path: str = "models/preprocessor.pkl"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(preprocessor, path)


def load_preprocessor(path: str = "models/preprocessor.pkl"):
    return joblib.load(path)
