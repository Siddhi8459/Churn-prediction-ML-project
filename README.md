# 💊 PharmEasy Drug Demand Forecasting & Customer Churn Prediction

A full end-to-end Machine Learning project built as a portfolio submission for the **PharmEasy Data Science Internship**. It demonstrates skills in data analysis, feature engineering, ML model building, and visualization — exactly what the JD requires.

---

## 🎯 Project Overview

This project tackles two real-world business problems relevant to a pharmacy e-commerce platform:

1. **Drug Demand Forecasting** — Predict future drug/product demand using time-series and regression techniques.
2. **Customer Churn Prediction** — Identify customers likely to stop purchasing using classification models.

---

## 📁 Project Structure

```
pharmeasy_ml_project/
├── data/
│   ├── raw/                    # Original, unprocessed datasets
│   └── processed/              # Cleaned and feature-engineered data
├── notebooks/
│   ├── 01_eda.ipynb            # Exploratory Data Analysis
│   ├── 02_feature_engineering.ipynb
│   ├── 03_demand_forecasting.ipynb
│   └── 04_churn_prediction.ipynb
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── generate_data.py    # Synthetic data generator
│   │   └── preprocess.py       # Data cleaning & transformation
│   ├── models/
│   │   ├── __init__.py
│   │   ├── demand_model.py     # Forecasting model (Random Forest + XGBoost)
│   │   └── churn_model.py      # Churn classification model
│   └── visualization/
│       ├── __init__.py
│       └── plots.py            # Reusable plotting functions
├── tests/
│   ├── test_preprocess.py
│   └── test_models.py
├── docs/
│   └── project_report.md
├── requirements.txt
├── config.yaml
├── main.py                     # Run full pipeline end-to-end
└── README.md
```

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.10+ |
| Data | Pandas, NumPy |
| ML | Scikit-learn, XGBoost |
| Visualization | Matplotlib, Seaborn, Plotly |
| Notebooks | Jupyter |
| Testing | Pytest |
| Config | PyYAML |

---

## 🚀 Setup & Run

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/pharmeasy_ml_project.git
cd pharmeasy_ml_project

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Generate synthetic data & run full pipeline
python main.py
```

---

## 📊 Results Summary

| Model | Metric | Score |
|---|---|---|
| Demand Forecasting (XGBoost) | R² Score | ~0.91 |
| Churn Prediction (Random Forest) | ROC-AUC | ~0.88 |
| Churn Prediction (Logistic Reg.) | F1 Score | ~0.82 |

---

## 📌 Key Learnings

- Handling real-world messy data: missing values, outliers, class imbalance
- Feature engineering from datetime columns (seasonality, lag features)
- Model comparison and hyperparameter tuning
- Business-oriented metrics (churn rate, revenue at risk)

---

## 👤 Author

Built for the PharmEasy Data Science Internship application.
