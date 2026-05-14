"""
app.py  –  House Price Prediction | Final Year Major Project
Run:  streamlit run app.py
"""

import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

# ── Page config (MUST be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Local modules ─────────────────────────────────────────────────────────────
from src.data_generator import generate_housing_data
from src.preprocessing  import validate_columns, REQUIRED_COLUMNS
from src.train_model    import (train_all_models, get_feature_importance,
                                 load_best_model, XGBOOST_AVAILABLE)
from src.prediction     import predict_price, format_inr, categorize_price
import src.visualization as viz

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Syne:wght@700;800&display=swap');

html, body, [class*="css"]  { font-family: 'Plus Jakarta Sans', sans-serif; }

/* ── Sidebar ──────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0F172A 0%, #1E293B 100%);
    color: #E2E8F0;
}
[data-testid="stSidebar"] .stRadio label,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #E2E8F0 !important;
}
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label {
    background: rgba(255,255,255,0.05);
    border-radius: 8px;
    padding: 10px 16px;
    margin-bottom: 4px;
    cursor: pointer;
    transition: background 0.2s;
    font-weight: 500;
    font-size: 0.93rem;
}
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label:hover {
    background: rgba(59,130,246,0.35);
}

/* ── Hero title ───────────────────────────────────────────────── */
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 2.8rem;
    font-weight: 800;
    background: linear-gradient(135deg, #3B82F6, #8B5CF6, #06B6D4);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1.15;
    margin-bottom: 0;
}
.hero-sub {
    font-size: 1.05rem;
    color: #64748B;
    margin-top: 6px;
    max-width: 660px;
}

/* ── Metric cards ─────────────────────────────────────────────── */
.metric-card {
    background: white;
    border: 1px solid #E2E8F0;
    border-radius: 14px;
    padding: 22px 24px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06);
    text-align: center;
    transition: transform 0.2s, box-shadow 0.2s;
}
.metric-card:hover { transform: translateY(-3px); box-shadow: 0 8px 24px rgba(0,0,0,0.10); }
.metric-card .icon { font-size: 1.9rem; margin-bottom: 6px; }
.metric-card .value { font-size: 1.7rem; font-weight: 800; color: #1E293B; }
.metric-card .label { font-size: 0.82rem; color: #94A3B8; font-weight: 500; letter-spacing: 0.05em; text-transform: uppercase; }

/* ── Feature cards ────────────────────────────────────────────── */
.feature-card {
    background: linear-gradient(135deg, #F0F9FF, #E0F2FE);
    border-left: 4px solid #3B82F6;
    border-radius: 10px;
    padding: 18px 20px;
    margin-bottom: 12px;
}
.feature-card h4 { margin: 0 0 4px; color: #1E40AF; font-size: 0.97rem; }
.feature-card p  { margin: 0; color: #475569; font-size: 0.88rem; }

/* ── Section headings ─────────────────────────────────────────── */
.section-head {
    font-family: 'Syne', sans-serif;
    font-size: 1.5rem;
    font-weight: 800;
    color: #1E293B;
    border-bottom: 3px solid #3B82F6;
    padding-bottom: 6px;
    margin-bottom: 20px;
}

/* ── Prediction result box ────────────────────────────────────── */
.pred-box {
    background: linear-gradient(135deg, #1E3A5F, #1E293B);
    color: white;
    border-radius: 18px;
    padding: 36px 32px;
    text-align: center;
    box-shadow: 0 12px 40px rgba(30,58,138,0.35);
}
.pred-box .pred-label { font-size: 0.88rem; letter-spacing: 0.1em; text-transform: uppercase; opacity: 0.7; }
.pred-box .pred-price { font-family: 'Syne', sans-serif; font-size: 3.0rem; font-weight: 800; margin: 6px 0; }
.pred-box .pred-range { font-size: 0.92rem; opacity: 0.75; }

/* ── About section ────────────────────────────────────────────── */
.about-card {
    background: white;
    border-radius: 14px;
    padding: 24px 28px;
    border: 1px solid #E2E8F0;
    box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    margin-bottom: 16px;
}
.about-card h3 { color: #1E40AF; margin-top: 0; }

/* ── Divider ──────────────────────────────────────────────────── */
.divider { border-top: 1px solid #E2E8F0; margin: 28px 0; }

/* ── Best model badge ─────────────────────────────────────────── */
.best-badge {
    display: inline-block;
    background: linear-gradient(135deg, #10B981, #059669);
    color: white;
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}
</style>
""", unsafe_allow_html=True)


# ── Session state defaults ────────────────────────────────────────────────────
if "df" not in st.session_state:
    st.session_state.df = None
if "results_df" not in st.session_state:
    st.session_state.results_df = None
if "best_name" not in st.session_state:
    st.session_state.best_name = None
if "best_pipeline" not in st.session_state:
    st.session_state.best_pipeline = None
if "y_test" not in st.session_state:
    st.session_state.y_test = None
if "y_pred" not in st.session_state:
    st.session_state.y_pred = None


# ── Helpers ───────────────────────────────────────────────────────────────────
def get_df() -> pd.DataFrame:
    """Return session dataset or generate default."""
    if st.session_state.df is not None:
        return st.session_state.df
    df = generate_housing_data()
    st.session_state.df = df
    return df


# ── Sidebar navigation ────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 16px 0 24px;'>
        <div style='font-size:2.8rem;'>🏠</div>
        <div style='font-family:Syne,sans-serif; font-weight:800; font-size:1.1rem; color:#F1F5F9; letter-spacing:0.02em;'>
            House Price<br>Predictor
        </div>
        <div style='font-size:0.72rem; color:#64748B; margin-top:4px; letter-spacing:0.08em; text-transform:uppercase;'>
        </div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigate",
        [
            "🏠  Home",
            "📊  Dataset Overview",
            "📈  Data Visualization",
            "🤖  Model Training",
            "💰  Predict Price",
            "💡  Insights",
            "ℹ️  About Project",
        ],
        label_visibility="collapsed",
    )

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    model_exists = os.path.exists("models/best_model.pkl")
    status_icon  = "✅" if model_exists else "⚠️"
    status_text  = f"Model: {'Ready' if model_exists else 'Not Trained'}"
    status_color = "#10B981" if model_exists else "#F59E0B"
    st.markdown(f"""
    <div style='padding:10px 16px; background:rgba(255,255,255,0.06);
         border-radius:8px; font-size:0.83rem; color:{status_color}; font-weight:600;'>
        {status_icon}  {status_text}
    </div>""", unsafe_allow_html=True)

    if st.session_state.best_name:
        st.markdown(f"""
        <div style='margin-top:8px; padding:10px 16px; background:rgba(255,255,255,0.06);
             border-radius:8px; font-size:0.80rem; color:#94A3B8;'>
            🏆 Best: <b style='color:#E2E8F0'>{st.session_state.best_name}</b>
        </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div style='position:fixed; bottom:18px; font-size:0.72rem; color:#475569;
         text-align:center; width:210px;'>
        © 2024 House Price Predictor<br>B.Tech Final Year Project
    </div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════════
# PAGE 1 — HOME
# ════════════════════════════════════════════════════════════════════════════════
if page == "🏠  Home":
    df = get_df()

    st.markdown("""
    <div class='hero-title'>House Price Prediction<br>Using Machine Learning</div>
    <div class='hero-sub'>
        An intelligent, multi-model regression system that predicts real estate prices
        across NCR cities using advanced ML algorithms and interactive analytics.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)

    # ── Quick stats ───────────────────────────────────────────────────────────
    model_exists = os.path.exists("models/best_model.pkl")
    r2_val   = f"{st.session_state.results_df['R2 Score'].max():.4f}" if st.session_state.results_df is not None else "—"
    best_mdl = st.session_state.best_name or "—"

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='icon'>📦</div>
            <div class='value'>{len(df):,}</div>
            <div class='label'>Dataset Records</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='icon'>⚙️</div>
            <div class='value'>{"5" if XGBOOST_AVAILABLE else "4"}</div>
            <div class='label'>ML Algorithms</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='icon'>🎯</div>
            <div class='value'>{r2_val}</div>
            <div class='label'>Best R² Score</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='icon'>🏆</div>
            <div class='value' style='font-size:1.1rem'>{best_mdl}</div>
            <div class='label'>Best Model</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:36px'></div>", unsafe_allow_html=True)

    # ── Features ──────────────────────────────────────────────────────────────
    st.markdown("<div class='section-head'>Key Features</div>", unsafe_allow_html=True)

    left, right = st.columns(2)
    features = [
        ("🧠", "Multi-Algorithm Comparison", "Trains Linear Regression, Decision Tree, Random Forest, Gradient Boosting" + (" & XGBoost" if XGBOOST_AVAILABLE else "") + " and selects the best."),
        ("📊", "Interactive Visualizations", "Explore price distributions, location trends, correlation heatmaps and more using Plotly."),
        ("💰", "Real-Time Price Prediction", "Enter property details and get an instant price estimate with confidence range."),
        ("📁", "Custom Dataset Support", "Upload your own CSV dataset or use the built-in NCR housing data."),
        ("🔍", "Feature Importance Analysis", "Understand which property attributes influence prices the most."),
        ("💡", "Actionable Insights", "Data-driven recommendations for buyers, sellers, and real estate analysts."),
    ]
    for i, (icon, title, desc) in enumerate(features):
        col = left if i % 2 == 0 else right
        with col:
            st.markdown(f"""
            <div class='feature-card'>
                <h4>{icon} {title}</h4>
                <p>{desc}</p>
            </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)

    # ── Quick city price preview ──────────────────────────────────────────────
    st.markdown("<div class='section-head'>NCR City Price Overview</div>", unsafe_allow_html=True)
    avg_by_loc = df.groupby("location")["price"].mean().sort_values(ascending=False)
    fig = px.bar(
        avg_by_loc.reset_index(), x="location", y="price",
        color="price", color_continuous_scale="Blues",
        labels={"price": "Avg Price (₹)", "location": "City"},
        template="plotly_white", text_auto=".2s",
        height=320,
    )
    fig.update_layout(coloraxis_showscale=False, margin=dict(t=20))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    <div style='background:#F0F9FF; border-radius:12px; padding:18px 22px;
         border-left:4px solid #3B82F6; font-size:0.88rem; color:#1E40AF;'>
        <b>ℹ️ Getting Started:</b> Use the sidebar to navigate pages.
        Start with <b>Dataset Overview</b> → <b>Model Training</b> → <b>Predict Price</b>.
    </div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════════
# PAGE 2 — DATASET OVERVIEW
# ════════════════════════════════════════════════════════════════════════════════
elif page == "📊  Dataset Overview":
    st.markdown("<div class='section-head'>📊 Dataset Overview</div>", unsafe_allow_html=True)

    # Upload or default
    uploaded = st.file_uploader("Upload your housing CSV (optional)", type=["csv"])
    if uploaded is not None:
        try:
            df_upload = pd.read_csv(uploaded)
            valid, missing = validate_columns(df_upload)
            if valid:
                st.session_state.df = df_upload
                st.success(f"✅ Dataset uploaded successfully — {len(df_upload):,} rows.")
            else:
                st.error(f"❌ Missing required columns: {missing}")
                st.info(f"Required columns: {REQUIRED_COLUMNS}")
        except Exception as e:
            st.error(f"Error reading file: {e}")

    df = get_df()

    # Download button
    csv_bytes = df.to_csv(index=False).encode()
    st.download_button("⬇️ Download Sample Dataset", csv_bytes,
                       "housing_data.csv", "text/csv")

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{df.shape[0]:,}")
    c2.metric("Columns", df.shape[1])
    c3.metric("Missing Values", int(df.isnull().sum().sum()))
    c4.metric("Duplicate Rows", int(df.duplicated().sum()))

    tab1, tab2, tab3, tab4 = st.tabs(["📋 Preview", "📈 Statistics", "🔎 Data Types", "❓ Missing Values"])

    with tab1:
        st.dataframe(df.head(100), use_container_width=True, height=380)

    with tab2:
        st.dataframe(df.describe().T.style.background_gradient(cmap="Blues"), use_container_width=True)

    with tab3:
        dtype_df = pd.DataFrame({
            "Column": df.columns,
            "Type": df.dtypes.values,
            "Unique Values": [df[c].nunique() for c in df.columns],
            "Sample": [str(df[c].iloc[0]) for c in df.columns],
        })
        st.dataframe(dtype_df, use_container_width=True)

    with tab4:
        missing = df.isnull().sum().reset_index()
        missing.columns = ["Column", "Missing Count"]
        missing["Missing %"] = (missing["Missing Count"] / len(df) * 100).round(2)
        st.dataframe(missing, use_container_width=True)
        if missing["Missing Count"].sum() == 0:
            st.success("✅ No missing values found in the dataset.")


# ════════════════════════════════════════════════════════════════════════════════
# PAGE 3 — DATA VISUALIZATION
# ════════════════════════════════════════════════════════════════════════════════
elif page == "📈  Data Visualization":
    df = get_df()
    st.markdown("<div class='section-head'>📈 Data Visualization</div>", unsafe_allow_html=True)

    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "💰 Price Dist.",
        "📐 Area vs Price",
        "📍 By Location",
        "🛏 Bedrooms",
        "🔗 Correlation",
        "🏘 Property Type",
        "🛋 Furnishing",
    ])

    with tab1:
        st.plotly_chart(viz.price_distribution(df), use_container_width=True)
        st.markdown("""
        <div class='feature-card'>
            <p>The price distribution shows a right-skewed pattern, with most properties in the
            ₹20–80 lakh range. A small segment of premium/luxury properties accounts for the long
            right tail — typical of Indian urban real estate markets.</p>
        </div>""", unsafe_allow_html=True)

    with tab2:
        st.plotly_chart(viz.area_vs_price(df), use_container_width=True)
        st.plotly_chart(viz.distance_vs_price(df), use_container_width=True)

    with tab3:
        st.plotly_chart(viz.avg_price_by_location(df), use_container_width=True)
        st.plotly_chart(viz.price_per_sqft_by_location(df), use_container_width=True)

    with tab4:
        st.plotly_chart(viz.bedrooms_vs_price(df), use_container_width=True)
        avg_bed = df.groupby("bedrooms")["price"].mean().reset_index()
        st.plotly_chart(
            px.line(avg_bed, x="bedrooms", y="price", markers=True,
                    title="Average Price by Number of Bedrooms",
                    labels={"price": "Avg Price (₹)", "bedrooms": "Bedrooms"},
                    template="plotly_white",
                    color_discrete_sequence=["#3B82F6"]),
            use_container_width=True,
        )

    with tab5:
        st.plotly_chart(viz.correlation_heatmap(df), use_container_width=True)

    with tab6:
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(viz.property_type_pie(df), use_container_width=True)
        with col2:
            avg_type = df.groupby("property_type")["price"].mean().reset_index()
            st.plotly_chart(
                px.bar(avg_type, x="property_type", y="price",
                       title="Avg Price by Property Type",
                       color="property_type",
                       color_discrete_sequence=px.colors.qualitative.Bold,
                       labels={"price": "Avg Price (₹)", "property_type": "Type"},
                       template="plotly_white", text_auto=".2s"),
                use_container_width=True,
            )

    with tab7:
        st.plotly_chart(viz.furnishing_vs_price(df), use_container_width=True)
        col1, col2 = st.columns(2)
        with col1:
            vc = df["furnishing_status"].value_counts().reset_index()
            vc.columns = ["Furnishing", "Count"]
            st.plotly_chart(
                px.pie(vc, names="Furnishing", values="Count",
                       title="Furnishing Status Distribution",
                       hole=0.4, color_discrete_sequence=px.colors.qualitative.Bold),
                use_container_width=True,
            )
        with col2:
            box_fig = px.box(df, x="furnishing_status", y="price",
                             color="furnishing_status",
                             title="Price Range by Furnishing",
                             labels={"furnishing_status": "Furnishing", "price": "Price (₹)"},
                             template="plotly_white",
                             color_discrete_sequence=px.colors.qualitative.Bold)
            box_fig.update_layout(showlegend=False)
            st.plotly_chart(box_fig, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════════
# PAGE 4 — MODEL TRAINING
# ════════════════════════════════════════════════════════════════════════════════
elif page == "🤖  Model Training":
    df = get_df()
    st.markdown("<div class='section-head'>🤖 Model Training & Comparison</div>",
                unsafe_allow_html=True)

    alg_list = ["Linear Regression", "Decision Tree", "Random Forest",
                 "Gradient Boosting"] + (["XGBoost"] if XGBOOST_AVAILABLE else [])
    st.markdown(f"""
    <div class='feature-card'>
        <h4>Algorithms: {' · '.join(alg_list)}</h4>
        <p>All models are trained on 80% of the data and evaluated on the remaining 20%.
        Preprocessing includes StandardScaler for numerics and OneHotEncoder for categoricals.</p>
    </div>""", unsafe_allow_html=True)

    col1, col2 = st.columns([1, 3])
    with col1:
        train_btn = st.button("🚀 Train All Models", type="primary", use_container_width=True)

    if train_btn:
        prog_bar = st.progress(0.0, text="Initialising…")

        def update_progress(frac, msg):
            prog_bar.progress(frac, text=msg)

        with st.spinner("Training models — this may take 30–60 seconds…"):
            try:
                results_df, best_name, best_pipeline, y_test, y_pred = train_all_models(
                    df, progress_callback=update_progress
                )
                st.session_state.results_df   = results_df
                st.session_state.best_name    = best_name
                st.session_state.best_pipeline = best_pipeline
                st.session_state.y_test       = y_test
                st.session_state.y_pred       = y_pred
                prog_bar.progress(1.0, text="✅ Done!")
                st.success(f"🏆 Best model: **{best_name}** | R² = {results_df.loc[best_name,'R2 Score']:.4f}")
            except Exception as e:
                st.error(f"Training failed: {e}")

    if st.session_state.results_df is not None:
        results_df = st.session_state.results_df
        best_name  = st.session_state.best_name

        # ── Results table ─────────────────────────────────────────────────────
        st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)
        st.markdown("#### 📋 Model Evaluation Results")

        def highlight_best(row):
            is_best = row.name == best_name
            return ["background-color:#D1FAE5; font-weight:700" if is_best else "" for _ in row]

        styled = results_df.style.apply(highlight_best, axis=1).format({
            "R2 Score": "{:.4f}",
            "MAE":      "{:,.0f}",
            "MSE":      "{:,.0f}",
            "RMSE":     "{:,.0f}",
            "Train Time (s)": "{:.2f}",
        })
        st.dataframe(styled, use_container_width=True)

        st.markdown(f"""
        <div style='margin:8px 0; text-align:right;'>
            <span class='best-badge'>🏆 Best Model: {best_name}</span>
        </div>""", unsafe_allow_html=True)

        # ── Comparison charts ─────────────────────────────────────────────────
        st.plotly_chart(viz.model_comparison_bar(results_df), use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(
                viz.actual_vs_predicted(
                    st.session_state.y_test, st.session_state.y_pred, best_name
                ),
                use_container_width=True,
            )
        with col2:
            st.plotly_chart(
                viz.residual_plot(st.session_state.y_test, st.session_state.y_pred),
                use_container_width=True,
            )

        # ── Feature importance ─────────────────────────────────────────────────
        importance = get_feature_importance(
            st.session_state.best_pipeline,
            feature_names=[]
        )
        if importance is not None:
            st.plotly_chart(viz.feature_importance_chart(importance), use_container_width=True)
        else:
            st.info("Feature importance is not available for Linear Regression.")

    else:
        st.markdown("""
        <div style='text-align:center; padding:60px; background:#F8FAFC;
             border-radius:16px; border:2px dashed #CBD5E1; margin-top:24px;'>
            <div style='font-size:3rem;'>🤖</div>
            <div style='font-size:1.2rem; color:#64748B; margin-top:12px;'>
                Click <b>Train All Models</b> to start training.
            </div>
        </div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════════
# PAGE 5 — PREDICT PRICE
# ════════════════════════════════════════════════════════════════════════════════
elif page == "💰  Predict Price":
    st.markdown("<div class='section-head'>💰 House Price Prediction</div>", unsafe_allow_html=True)

    if not os.path.exists("models/best_model.pkl"):
        st.warning("⚠️ No trained model found. Please go to **Model Training** and train the models first.")
        st.stop()

    df = get_df()

    st.markdown("""
    <div class='feature-card'>
        <p>Fill in the property details below and click <b>Predict Price</b> to get an
        AI-powered price estimate with a confidence range.</p>
    </div>""", unsafe_allow_html=True)

    with st.form("prediction_form"):
        st.markdown("#### 📍 Location & Property Type")
        col1, col2, col3 = st.columns(3)
        with col1:
            location = st.selectbox("Location", sorted(df["location"].unique()))
        with col2:
            property_type = st.selectbox("Property Type", sorted(df["property_type"].unique()))
        with col3:
            furnishing_status = st.selectbox("Furnishing Status", sorted(df["furnishing_status"].unique()))

        st.markdown("#### 📐 Size & Layout")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            area_sqft = st.number_input("Area (sq ft)", min_value=300, max_value=10000, value=1200, step=50)
        with col2:
            bedrooms = st.slider("Bedrooms", 1, 6, 3)
        with col3:
            bathrooms = st.slider("Bathrooms", 1, 5, 2)
        with col4:
            balconies = st.slider("Balconies", 0, 4, 1)

        col1, col2, col3 = st.columns(3)
        with col1:
            floors = st.number_input("Floor Number", min_value=0, max_value=50, value=5)
        with col2:
            parking = st.selectbox("Parking Spots", [0, 1, 2])
        with col3:
            property_age = st.slider("Property Age (years)", 0, 30, 5)

        st.markdown("#### 📍 Location Factors")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            distance_from_city_center = st.number_input("Distance from City Center (km)", 0.5, 40.0, 8.0, 0.5)
        with col2:
            nearby_schools = st.number_input("Nearby Schools", 0, 15, 3)
        with col3:
            nearby_hospitals = st.number_input("Nearby Hospitals", 0, 10, 2)
        with col4:
            crime_rate = st.slider("Crime Rate (1=low, 10=high)", 1.0, 10.0, 4.0, 0.5)

        submitted = st.form_submit_button("🔮 Predict Price", type="primary", use_container_width=True)

    if submitted:
        input_data = {
            "location":                  location,
            "area_sqft":                 area_sqft,
            "bedrooms":                  bedrooms,
            "bathrooms":                 bathrooms,
            "balconies":                 balconies,
            "floors":                    floors,
            "parking":                   parking,
            "property_age":              property_age,
            "furnishing_status":         furnishing_status,
            "property_type":             property_type,
            "distance_from_city_center": distance_from_city_center,
            "nearby_schools":            nearby_schools,
            "nearby_hospitals":          nearby_hospitals,
            "crime_rate":                crime_rate,
        }

        try:
            result   = predict_price(input_data)
            price    = result["predicted_price"]
            cat      = result["category"]

            # ── Main result box ────────────────────────────────────────────────
            st.markdown(f"""
            <div class='pred-box'>
                <div class='pred-label'>Estimated Market Value</div>
                <div class='pred-price'>{format_inr(price)}</div>
                <div class='pred-range'>
                    Range: {format_inr(result['min_price'])} — {format_inr(result['max_price'])}
                </div>
            </div>""", unsafe_allow_html=True)

            st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Min Expected", format_inr(result["min_price"]))
            with col2:
                st.metric("Predicted Price", format_inr(price), delta=None)
            with col3:
                st.metric("Max Expected", format_inr(result["max_price"]))

            # ── Category badge ─────────────────────────────────────────────────
            st.markdown(f"""
            <div style='text-align:center; margin:20px 0;'>
                <span style='background:{cat["color"]}; color:white; padding:8px 24px;
                      border-radius:24px; font-weight:700; font-size:1.0rem;'>
                    {cat["icon"]} {cat["label"]} Property
                </span>
            </div>""", unsafe_allow_html=True)

            st.markdown(f"""
            <div class='about-card'>
                <h3>💬 AI Recommendation</h3>
                <p style='color:#334155'>{cat["description"]}</p>
                <p style='color:#64748B; font-size:0.85rem; margin-top:8px;'>
                    <b>Property Summary:</b> {bedrooms}BHK {property_type} in {location} |
                    {area_sqft} sq ft | {furnishing_status} | Age: {property_age} yrs |
                    {distance_from_city_center} km from city center
                </p>
            </div>""", unsafe_allow_html=True)

            # ── Price breakdown gauge ──────────────────────────────────────────
            import plotly.graph_objects as go
            gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=price / 1e5,
                number={"suffix": " L", "font": {"size": 28}},
                title={"text": "Predicted Price (Lakhs ₹)"},
                gauge={
                    "axis": {"range": [0, 500], "ticksuffix": "L"},
                    "bar": {"color": cat["color"]},
                    "steps": [
                        {"range": [0,   30],  "color": "#D1FAE5"},
                        {"range": [30,  70],  "color": "#FEF3C7"},
                        {"range": [70, 150],  "color": "#FED7AA"},
                        {"range": [150, 500], "color": "#FEE2E2"},
                    ],
                    "threshold": {
                        "line": {"color": "black", "width": 3},
                        "thickness": 0.85,
                        "value": price / 1e5,
                    },
                },
            ))
            gauge.update_layout(height=280, margin=dict(t=40, b=10))
            st.plotly_chart(gauge, use_container_width=True)

        except RuntimeError as e:
            st.error(str(e))
        except Exception as e:
            st.error(f"Prediction error: {e}")


# ════════════════════════════════════════════════════════════════════════════════
# PAGE 6 — INSIGHTS
# ════════════════════════════════════════════════════════════════════════════════
elif page == "💡  Insights":
    df = get_df()
    st.markdown("<div class='section-head'>💡 Insights & Recommendations</div>", unsafe_allow_html=True)

    # ── Top stats ─────────────────────────────────────────────────────────────
    top_loc = df.groupby("location")["price"].mean().idxmax()
    top_type = df.groupby("property_type")["price"].mean().idxmax()
    corr_area = df["area_sqft"].corr(df["price"])

    c1, c2, c3 = st.columns(3)
    c1.metric("Priciest Location", top_loc)
    c2.metric("Priciest Property Type", top_type)
    c3.metric("Area–Price Correlation", f"{corr_area:.3f}")

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["📍 Location Insights", "🏠 Property Insights", "💼 Recommendations"])

    with tab1:
        st.plotly_chart(viz.top_locations_chart(df), use_container_width=True)
        st.plotly_chart(viz.price_per_sqft_by_location(df), use_container_width=True)

        avg_loc = df.groupby("location")["price"].mean().sort_values(ascending=False)
        st.markdown("#### 📌 City-wise Insights")
        for loc, avg in avg_loc.items():
            count = len(df[df["location"] == loc])
            st.markdown(f"""
            <div class='feature-card' style='margin-bottom:8px;'>
                <h4>📍 {loc} — Avg: {format_inr(avg)}</h4>
                <p>{count} properties | Price range: {format_inr(df[df["location"]==loc]["price"].min())}
                 – {format_inr(df[df["location"]==loc]["price"].max())}</p>
            </div>""", unsafe_allow_html=True)

    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(viz.furnishing_vs_price(df), use_container_width=True)
        with col2:
            st.plotly_chart(viz.property_type_pie(df), use_container_width=True)

        st.plotly_chart(viz.distance_vs_price(df), use_container_width=True)

        st.markdown("#### 📌 Key Observations")
        insights = [
            ("📐 Area", f"Correlation with price = {corr_area:.3f}. Larger area strongly predicts higher price."),
            ("🛋 Furnishing", "Furnished properties command ~20% premium over unfurnished ones."),
            ("🏘 Property Type", f"Villas are the most expensive type; Apartments are the most common."),
            ("📍 Distance", "Each km away from city center reduces price by ~1.5% on average."),
            ("🏚 Age", "Properties older than 15 years depreciate noticeably; new properties command a premium."),
        ]
        for icon_title, desc in insights:
            st.markdown(f"""
            <div class='feature-card'>
                <h4>{icon_title}</h4>
                <p>{desc}</p>
            </div>""", unsafe_allow_html=True)

    with tab3:
        recs = [
            ("🏠 For Buyers", [
                "Use the prediction tool to validate if a quoted price matches market value.",
                "Prefer locations like Greater Noida or Ghaziabad for budget-friendly options.",
                "Semi-furnished properties offer the best cost-to-value balance.",
                "Properties within 10 km of city centers hold value better.",
            ]),
            ("💼 For Sellers", [
                "Furnished properties sell 15–25% higher — consider investing in furnishing.",
                "Highlight nearby schools and hospitals in listings to justify higher pricing.",
                "Price new/recent properties at a premium; buyers discount older stock.",
                "Delhi and Gurgaon locations support the highest asking prices.",
            ]),
            ("📊 For Analysts", [
                "Area, location, and furnishing are the top three price drivers.",
                "Crime rate negatively correlates with price — use for risk assessment.",
                "Gradient Boosting and Random Forest outperform linear models for NCR data.",
                "Price per sq ft varies 2×–3× between locations — use for comparative analysis.",
            ]),
        ]
        for title, points in recs:
            st.markdown(f"""
            <div class='about-card'>
                <h3>{title}</h3>
                {"".join(f"<p>• {p}</p>" for p in points)}
            </div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════════
# PAGE 7 — ABOUT PROJECT
# ════════════════════════════════════════════════════════════════════════════════
elif page == "ℹ️  About Project":
    st.markdown("<div class='section-head'>ℹ️ About the Project</div>", unsafe_allow_html=True)

    sections = [
        ("🎯 Problem Statement",
         """House price estimation is a complex, multi-factor challenge. Property values depend on
         location, size, amenities, market trends, and neighbourhood characteristics. Manual
         appraisal is expensive, slow, and subjective. This project develops an ML-based system
         that predicts prices objectively and at scale, supporting buyers, sellers, and analysts."""),

        ("📋 Objectives",
         """• Build and compare multiple ML regression algorithms for price prediction<br>
         • Identify the most influential features driving house prices<br>
         • Provide an interactive data visualization dashboard<br>
         • Develop a user-friendly prediction interface<br>
         • Generate reliable price estimates to support real-estate decisions"""),

        ("🔬 Methodology",
         """<b>Dataset Collection</b> → <b>Data Preprocessing</b> (null handling, encoding, scaling)
         → <b>Feature Engineering</b> → <b>Model Training</b> (5 algorithms) → <b>Model Evaluation</b>
         (R², MAE, MSE, RMSE) → <b>Best Model Selection</b> → <b>Price Prediction</b>
         → <b>Visualization & Insights</b>"""),

        ("⚙️ Tech Stack",
         """<b>Language:</b> Python 3.10+<br>
         <b>Frontend:</b> Streamlit, Plotly<br>
         <b>ML:</b> Scikit-learn, XGBoost (optional), Pandas, NumPy<br>
         <b>Persistence:</b> Joblib<br>
         <b>Algorithms:</b> Linear Regression, Decision Tree, Random Forest,
         Gradient Boosting""" + (", XGBoost" if XGBOOST_AVAILABLE else "")),

        ("🚀 Future Scope",
         """• Integration with real-time property listing APIs (MagicBricks, 99acres)<br>
         • City-level dynamic price index updates<br>
         • Deep learning models (LSTM, Transformers) for time-series price trends<br>
         • Map-based geospatial price visualization<br>
         • Cloud deployment (AWS / Azure / GCP)<br>
         • User authentication and saved prediction history<br>
         • Mobile application"""),

        ("⚠️ Limitations",
         """• Trained on synthetic NCR data — real-world accuracy requires live datasets<br>
         • Market fluctuations, economic events are not modelled<br>
         • Legal disputes or title issues are not factored in<br>
         • Model performance may degrade on cities not in training data"""),
    ]

    for title, content in sections:
        st.markdown(f"""
        <div class='about-card'>
            <h3>{title}</h3>
            <p style='color:#334155; line-height:1.7'>{content}</p>
        </div>""", unsafe_allow_html=True)

    # ── Team / submission info ─────────────────────────────────────────────────
    st.markdown("""
    <div style='background:linear-gradient(135deg,#1E3A5F,#1E293B); color:white;
         border-radius:16px; padding:28px 32px; margin-top:8px; text-align:center;'>
        <div style='font-family:Syne,sans-serif; font-size:1.4rem; font-weight:800; margin-bottom:8px;'>
            House Price Prediction Using Machine Learning
        </div>
        <div style='color:#94A3B8; font-size:0.90rem;'>
            B.Tech Final Year Major Project · Computer Science & Engineering
        </div>
        <div style='color:#64748B; font-size:0.82rem; margin-top:8px;'>
            Built with Python · Streamlit · Scikit-learn · Plotly
        </div>
    </div>""", unsafe_allow_html=True)
