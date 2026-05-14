"""
prediction.py
Handles single-property price prediction using the saved best model.
"""

import numpy as np
from src.train_model import load_best_model

PRICE_CATEGORIES = [
    (0,        3_000_000,  "Budget",    "🟢", "#28a745",
     "An affordable property suitable for first-time buyers or investors seeking entry-level real estate."),
    (3_000_000, 7_000_000,  "Mid-Range", "🟡", "#ffc107",
     "A well-balanced property offering good value across key amenities and location factors."),
    (7_000_000, 15_000_000, "Premium",   "🟠", "#fd7e14",
     "A premium property in a desirable area with high-quality features and strong appreciation potential."),
    (15_000_000, np.inf,    "Luxury",    "🔴", "#dc3545",
     "An elite luxury property with exceptional features, prime location, and exclusive amenities."),
]


def categorize_price(price: float) -> dict:
    """Return category metadata for a given price."""
    for lo, hi, label, icon, color, desc in PRICE_CATEGORIES:
        if lo <= price < hi:
            return {"label": label, "icon": icon, "color": color, "description": desc}
    return {"label": "Luxury", "icon": "🔴", "color": "#dc3545",
            "description": PRICE_CATEGORIES[-1][-1]}


def predict_price(input_features: dict) -> dict:
    """
    Run prediction on input_features dict.
    Returns a dict with predicted_price, min_price, max_price, and category info.
    Raises RuntimeError if no model is saved yet.
    """
    pipeline = load_best_model()
    if pipeline is None:
        raise RuntimeError(
            "No trained model found. Please go to 'Model Training' and train the models first."
        )

    import pandas as pd
    from src.preprocessing import NUMERICAL_FEATURES, CATEGORICAL_FEATURES

    row = pd.DataFrame([input_features])
    row = row[NUMERICAL_FEATURES + CATEGORICAL_FEATURES]

    predicted = float(pipeline.predict(row)[0])
    # ±10 % confidence band
    min_price = predicted * 0.90
    max_price = predicted * 1.10
    category  = categorize_price(predicted)

    return {
        "predicted_price": predicted,
        "min_price":        min_price,
        "max_price":        max_price,
        "category":         category,
    }


def format_inr(amount: float) -> str:
    """Format a rupee amount into lakhs / crores."""
    if amount >= 1e7:
        return f"₹{amount / 1e7:.2f} Cr"
    elif amount >= 1e5:
        return f"₹{amount / 1e5:.2f} L"
    else:
        return f"₹{amount:,.0f}"
