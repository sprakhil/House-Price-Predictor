"""
train_model.py
Trains multiple regression models, evaluates them, and saves the best one.
"""

import numpy as np
import pandas as pd
import joblib
import os
import time

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline

from src.preprocessing import build_preprocessor, split_data, clean_data

# ── Optional XGBoost ──────────────────────────────────────────────────────────
try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


MODEL_PATH       = "models/best_model.pkl"
PREPROCESSOR_PATH = "models/preprocessor.pkl"


def get_models() -> dict:
    """Return a dict of model name → estimator."""
    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree":     DecisionTreeRegressor(max_depth=8, random_state=42),
        "Random Forest":     RandomForestRegressor(n_estimators=150, max_depth=12,
                                                   random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=150,
                                                        learning_rate=0.08,
                                                        max_depth=5,
                                                        random_state=42),
    }
    if XGBOOST_AVAILABLE:
        models["XGBoost"] = XGBRegressor(n_estimators=150, learning_rate=0.08,
                                         max_depth=6, random_state=42,
                                         verbosity=0)
    return models


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute regression metrics."""
    mse  = mean_squared_error(y_true, y_pred)
    return {
        "R2 Score": round(r2_score(y_true, y_pred), 4),
        "MAE":      round(mean_absolute_error(y_true, y_pred), 2),
        "MSE":      round(mse, 2),
        "RMSE":     round(np.sqrt(mse), 2),
    }


def train_all_models(df: pd.DataFrame, progress_callback=None):
    """
    Train all models and return:
        results_df        – DataFrame with metrics
        best_name         – name of the best model
        best_pipeline     – fitted sklearn Pipeline (preprocessor + model)
        y_test, y_pred    – for plotting
    """
    df = clean_data(df)
    X_train, X_test, y_train, y_test = split_data(df)
    preprocessor = build_preprocessor()

    models  = get_models()
    results = []
    best_r2      = -np.inf
    best_name    = None
    best_pipeline = None
    best_pred    = None

    total = len(models)
    for idx, (name, model) in enumerate(models.items()):
        if progress_callback:
            progress_callback(idx / total, f"Training {name}…")

        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", model),
        ])

        t0 = time.time()
        pipeline.fit(X_train, y_train)
        elapsed = round(time.time() - t0, 2)

        y_pred  = pipeline.predict(X_test)
        metrics = evaluate(y_test, y_pred)
        metrics["Model"]        = name
        metrics["Train Time (s)"] = elapsed
        results.append(metrics)

        if metrics["R2 Score"] > best_r2:
            best_r2      = metrics["R2 Score"]
            best_name    = name
            best_pipeline = pipeline
            best_pred    = y_pred

    if progress_callback:
        progress_callback(1.0, "Done!")

    # Save best model
    os.makedirs("models", exist_ok=True)
    joblib.dump(best_pipeline, MODEL_PATH)

    results_df = pd.DataFrame(results).set_index("Model")
    cols = ["R2 Score", "MAE", "MSE", "RMSE", "Train Time (s)"]
    results_df = results_df[cols]

    return results_df, best_name, best_pipeline, y_test, best_pred


def get_feature_importance(pipeline: Pipeline, feature_names: list) -> pd.Series | None:
    """Extract feature importance from tree-based models."""
    model = pipeline.named_steps["model"]
    if not hasattr(model, "feature_importances_"):
        return None

    preprocessor = pipeline.named_steps["preprocessor"]
    # Get OHE feature names
    cat_features = (
        preprocessor.named_transformers_["cat"]
        .named_steps["onehot"]
        .get_feature_names_out(
            ["location", "furnishing_status", "property_type"]
        )
    )
    num_features = [
        "area_sqft", "bedrooms", "bathrooms", "balconies", "floors",
        "parking", "property_age", "distance_from_city_center",
        "nearby_schools", "nearby_hospitals", "crime_rate",
    ]
    all_features = list(num_features) + list(cat_features)

    importance = pd.Series(
        model.feature_importances_, index=all_features
    ).sort_values(ascending=False)
    return importance


def load_best_model():
    """Load saved best model pipeline from disk."""
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)
