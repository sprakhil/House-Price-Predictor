# 🏠 House Price Prediction Using Machine Learning

> **B.Tech Final Year Major Project** — Computer Science & Engineering

A complete, multi-page Streamlit web application that predicts residential property prices across NCR cities using multiple machine learning algorithms with interactive analytics, model comparison, and an AI-powered prediction interface.

---

## 🚀 Features

| Feature | Details |
|---|---|
| **Multi-Model ML** | Linear Regression, Decision Tree, Random Forest, Gradient Boosting (+ XGBoost if installed) |
| **Model Comparison** | R², MAE, MSE, RMSE side-by-side table & bar charts |
| **Interactive Dashboard** | 10+ Plotly charts across 7 pages |
| **Price Prediction** | Form-based input → price estimate with confidence range & category |
| **Feature Importance** | Bar chart showing top predictors for tree-based models |
| **Custom Dataset** | Upload any compatible CSV or use the built-in 1200-record NCR dataset |
| **Insights** | Actionable recommendations for buyers, sellers & analysts |

---

## 🛠 Tech Stack

- **Language:** Python 3.10+
- **Frontend:** Streamlit, Custom CSS
- **Visualizations:** Plotly
- **ML Library:** Scikit-learn
- **Optional:** XGBoost
- **Data:** Pandas, NumPy
- **Model Persistence:** Joblib

---

## 📁 Project Structure

```
house-price-prediction/
│
├── app.py                    # Main Streamlit application (7 pages)
├── requirements.txt          # Python dependencies
├── README.md
│
├── data/
│   └── housing_data.csv      # Auto-generated on first run
│
├── models/
│   └── best_model.pkl        # Saved after training
│
└── src/
    ├── __init__.py
    ├── data_generator.py     # Synthetic NCR dataset generator
    ├── preprocessing.py      # Encoding, scaling, splitting
    ├── train_model.py        # All ML models + evaluation
    ├── prediction.py         # Single-property prediction helper
    └── visualization.py      # All Plotly chart functions
```

---

## ⚙️ Installation & Setup

### 1. Clone / Download the project

```bash
git clone <your-repo-url>
cd house-price-prediction
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux / macOS
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt

# Optional: install XGBoost for an extra algorithm
pip install xgboost
```

### 4. Run the application

```bash
streamlit run app.py
```

The app opens at **http://localhost:8501** in your browser.

---

## 📊 Dataset

The built-in dataset is **synthetically generated** to reflect realistic NCR real-estate pricing patterns:

| Column | Description |
|---|---|
| `location` | City (Delhi, Gurgaon, Noida, Greater Noida, Ghaziabad, Faridabad) |
| `area_sqft` | Property area in square feet (500–5000) |
| `bedrooms` | Number of bedrooms (1–5) |
| `bathrooms` | Number of bathrooms (1–5) |
| `balconies` | Number of balconies (0–3) |
| `floors` | Floor number (1–24) |
| `parking` | Parking spots (0, 1, 2) |
| `property_age` | Age in years (0–30) |
| `furnishing_status` | Furnished / Semi-Furnished / Unfurnished |
| `property_type` | Apartment / Builder Floor / Independent House / Villa |
| `distance_from_city_center` | Distance in km |
| `nearby_schools` | Count of nearby schools |
| `nearby_hospitals` | Count of nearby hospitals |
| `crime_rate` | Crime index (1 = low, 10 = high) |
| `price` | Target: Property price in INR |

You can also **upload your own CSV** on the Dataset Overview page (must contain the columns above).

---

## 🧭 How to Use

1. **Home** — Overview, quick stats, city price chart
2. **Dataset Overview** — Upload CSV or browse the built-in dataset
3. **Data Visualization** — 10+ interactive charts
4. **Model Training** → Click **Train All Models** (takes ~30–60 s)
5. **Predict Price** → Fill the form → Get instant estimate
6. **Insights** — City & property type analysis + recommendations
7. **About Project** — Problem statement, objectives, future scope

---

## 📈 Model Performance (typical)

| Model | R² Score | RMSE |
|---|---|---|
| Linear Regression | ~0.78 | ~850,000 |
| Decision Tree | ~0.84 | ~720,000 |
| **Random Forest** | **~0.92** | **~520,000** |
| Gradient Boosting | ~0.91 | ~540,000 |
| XGBoost | ~0.93 | ~500,000 |

*Exact values vary with each run.*

---

## 🔮 Future Scope

- Real-time property listing API integration (MagicBricks, 99acres)
- LSTM / Transformer-based time-series price prediction
- Geospatial map-based price visualization
- Cloud deployment (AWS / Azure / GCP)
- User authentication and prediction history
- Mobile application

---

## ⚠️ Limitations

- Trained on synthetic data — for real-world use, replace with live market data
- Does not account for sudden market shifts or legal/title issues
- City coverage limited to NCR region

---

## 📸 Screenshots

*(Add screenshots of each page here)*

---

## 🧑‍💻 Developed By

B.Tech Final Year Student | Computer Science & Engineering  
Session 2024–25
