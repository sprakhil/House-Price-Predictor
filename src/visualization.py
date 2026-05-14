"""
visualization.py
All Plotly chart helpers used across the Streamlit pages.
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

PALETTE = px.colors.qualitative.Bold
TEMPLATE = "plotly_white"


# ── Dataset Overview ──────────────────────────────────────────────────────────

def price_distribution(df: pd.DataFrame) -> go.Figure:
    fig = px.histogram(
        df, x="price", nbins=60, color_discrete_sequence=["#3B82F6"],
        title="Price Distribution",
        labels={"price": "Price (₹)"},
        template=TEMPLATE,
    )
    fig.update_layout(bargap=0.05)
    return fig


def area_vs_price(df: pd.DataFrame) -> go.Figure:
    fig = px.scatter(
        df, x="area_sqft", y="price", color="location",
        title="Area vs Price by Location",
        labels={"area_sqft": "Area (sq ft)", "price": "Price (₹)"},
        opacity=0.65, template=TEMPLATE,
        color_discrete_sequence=PALETTE,
    )
    return fig


def avg_price_by_location(df: pd.DataFrame) -> go.Figure:
    avg = df.groupby("location")["price"].mean().reset_index().sort_values("price", ascending=False)
    fig = px.bar(
        avg, x="location", y="price", color="location",
        title="Average Price by Location",
        labels={"price": "Avg Price (₹)", "location": "Location"},
        color_discrete_sequence=PALETTE,
        template=TEMPLATE,
        text_auto=".2s",
    )
    fig.update_layout(showlegend=False)
    return fig


def bedrooms_vs_price(df: pd.DataFrame) -> go.Figure:
    fig = px.box(
        df, x="bedrooms", y="price", color="bedrooms",
        title="Bedrooms vs Price",
        labels={"bedrooms": "Bedrooms", "price": "Price (₹)"},
        color_discrete_sequence=PALETTE,
        template=TEMPLATE,
    )
    fig.update_layout(showlegend=False)
    return fig


def correlation_heatmap(df: pd.DataFrame) -> go.Figure:
    num_cols = [
        "area_sqft", "bedrooms", "bathrooms", "balconies", "floors",
        "parking", "property_age", "distance_from_city_center",
        "nearby_schools", "nearby_hospitals", "crime_rate", "price",
    ]
    corr = df[num_cols].corr()
    fig = go.Figure(go.Heatmap(
        z=corr.values,
        x=corr.columns.tolist(),
        y=corr.index.tolist(),
        colorscale="RdBu",
        zmid=0,
        text=np.round(corr.values, 2),
        texttemplate="%{text}",
        textfont={"size": 9},
    ))
    fig.update_layout(
        title="Feature Correlation Heatmap",
        height=520,
        template=TEMPLATE,
    )
    return fig


def property_type_pie(df: pd.DataFrame) -> go.Figure:
    counts = df["property_type"].value_counts().reset_index()
    counts.columns = ["property_type", "count"]
    fig = px.pie(
        counts, names="property_type", values="count",
        title="Property Type Distribution",
        color_discrete_sequence=PALETTE,
        hole=0.40,
    )
    return fig


def furnishing_vs_price(df: pd.DataFrame) -> go.Figure:
    avg = df.groupby("furnishing_status")["price"].mean().reset_index()
    fig = px.bar(
        avg, x="furnishing_status", y="price",
        color="furnishing_status",
        title="Avg Price by Furnishing Status",
        labels={"price": "Avg Price (₹)", "furnishing_status": "Furnishing"},
        color_discrete_sequence=PALETTE,
        template=TEMPLATE,
        text_auto=".2s",
    )
    fig.update_layout(showlegend=False)
    return fig


def distance_vs_price(df: pd.DataFrame) -> go.Figure:
    fig = px.scatter(
        df, x="distance_from_city_center", y="price",
        color="location", trendline="ols",
        title="Distance from City Center vs Price",
        labels={"distance_from_city_center": "Distance (km)", "price": "Price (₹)"},
        opacity=0.60, template=TEMPLATE,
        color_discrete_sequence=PALETTE,
    )
    return fig


# ── Model Comparison ──────────────────────────────────────────────────────────

def model_comparison_bar(results_df: pd.DataFrame) -> go.Figure:
    fig = make_subplots(rows=1, cols=2, subplot_titles=("R² Score", "RMSE"))

    fig.add_trace(
        go.Bar(
            name="R² Score",
            x=results_df.index.tolist(),
            y=results_df["R2 Score"].tolist(),
            marker_color=PALETTE[:len(results_df)],
            text=[f"{v:.4f}" for v in results_df["R2 Score"]],
            textposition="outside",
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Bar(
            name="RMSE",
            x=results_df.index.tolist(),
            y=results_df["RMSE"].tolist(),
            marker_color=PALETTE[:len(results_df)],
            text=[f"{v:,.0f}" for v in results_df["RMSE"]],
            textposition="outside",
        ),
        row=1, col=2,
    )
    fig.update_layout(
        title="Model Performance Comparison",
        showlegend=False,
        height=420,
        template=TEMPLATE,
    )
    return fig


def actual_vs_predicted(y_test: np.ndarray, y_pred: np.ndarray, model_name: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=y_test, y=y_pred,
        mode="markers",
        marker=dict(color="#3B82F6", opacity=0.55, size=6),
        name="Predictions",
    ))
    lo = min(y_test.min(), y_pred.min())
    hi = max(y_test.max(), y_pred.max())
    fig.add_trace(go.Scatter(
        x=[lo, hi], y=[lo, hi],
        mode="lines",
        line=dict(color="#EF4444", dash="dash", width=2),
        name="Perfect Prediction",
    ))
    fig.update_layout(
        title=f"Actual vs Predicted Price — {model_name}",
        xaxis_title="Actual Price (₹)",
        yaxis_title="Predicted Price (₹)",
        template=TEMPLATE,
        height=420,
    )
    return fig


def residual_plot(y_test: np.ndarray, y_pred: np.ndarray) -> go.Figure:
    residuals = y_test - y_pred
    fig = px.scatter(
        x=y_pred, y=residuals,
        labels={"x": "Predicted Price (₹)", "y": "Residuals (₹)"},
        title="Residual Plot",
        color_discrete_sequence=["#8B5CF6"],
        template=TEMPLATE,
        opacity=0.55,
    )
    fig.add_hline(y=0, line_dash="dash", line_color="red")
    return fig


def feature_importance_chart(importance: pd.Series, top_n: int = 15) -> go.Figure:
    top = importance.head(top_n).sort_values()
    fig = go.Figure(go.Bar(
        x=top.values,
        y=top.index.tolist(),
        orientation="h",
        marker_color="#3B82F6",
    ))
    fig.update_layout(
        title=f"Top {top_n} Feature Importances",
        xaxis_title="Importance Score",
        yaxis_title="Feature",
        template=TEMPLATE,
        height=460,
    )
    return fig


# ── Insights Page ─────────────────────────────────────────────────────────────

def top_locations_chart(df: pd.DataFrame) -> go.Figure:
    avg = (
        df.groupby("location")["price"]
        .mean().reset_index()
        .sort_values("price", ascending=False)
    )
    fig = px.bar(
        avg, x="price", y="location", orientation="h",
        title="Locations Ranked by Average Price",
        labels={"price": "Avg Price (₹)", "location": "Location"},
        color="price",
        color_continuous_scale="Blues",
        text_auto=".2s",
        template=TEMPLATE,
    )
    fig.update_layout(coloraxis_showscale=False)
    return fig


def price_per_sqft_by_location(df: pd.DataFrame) -> go.Figure:
    df2 = df.copy()
    df2["price_per_sqft"] = df2["price"] / df2["area_sqft"]
    avg = df2.groupby("location")["price_per_sqft"].mean().reset_index()
    fig = px.bar(
        avg, x="location", y="price_per_sqft",
        title="Avg Price per sq ft by Location",
        labels={"price_per_sqft": "₹ / sq ft"},
        color="price_per_sqft",
        color_continuous_scale="Viridis",
        text_auto=".0f",
        template=TEMPLATE,
    )
    fig.update_layout(coloraxis_showscale=False)
    return fig
