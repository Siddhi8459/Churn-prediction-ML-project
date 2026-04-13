"""
main.py
-------
Orchestrates the full ML pipeline end-to-end:
  1. Generate synthetic data
  2. Preprocess & feature engineering
  3. Train demand forecasting model
  4. Train churn prediction models
  5. Generate evaluation plots
"""

import warnings
warnings.filterwarnings("ignore")

print("=" * 60)
print("  PharmEasy ML Pipeline — Starting")
print("=" * 60)
import pandas as pd
import numpy as np

def generate_sales_data(n_customers=1000, n_products=20):
    
    np.random.seed(42)

    # Generate product data
    products = pd.DataFrame({
        "product_id": range(1, n_products + 1),
        "price": np.random.randint(100, 1000, n_products)
    })

    # Generate dates
    dates = pd.date_range(start="2023-01-01", end="2023-12-31", freq="D")

    sales_data = []

    for _ in range(n_customers):
        date = np.random.choice(dates)

        # ✅ FIX: convert to pandas datetime
        date = pd.to_datetime(date)

        month = date.month   # now this works ✅

        customer_id = np.random.randint(1, n_customers)
        product_id = np.random.randint(1, n_products)

        quantity = np.random.randint(1, 5)

        # Seasonal effect example
        if month == 12:  # December sales boost 🎄
            quantity += 2

        sales_data.append([
            customer_id,
            product_id,
            date,
            month,
            quantity
        ])

    sales = pd.DataFrame(sales_data, columns=[
        "customer_id",
        "product_id",
        "date",
        "month",
        "quantity"
    ])

    return sales, products

# ── Step 1: Generate Data ─────────────────────────────────────
print("\n[1/5] Generating synthetic data...")
from src.data.generate_data import generate_sales_data, generate_customer_data
sales, products = generate_sales_data(n_customers=5000, n_products=50)
customers = generate_customer_data(sales)

# ── Step 2: Preprocess ────────────────────────────────────────
print("\n[2/5] Preprocessing & feature engineering...")
from src.data.preprocess import engineer_demand_features, prepare_churn_features
demand_df = engineer_demand_features(sales)
X_train, X_test, y_train, y_test, scaler, feat_cols = prepare_churn_features(customers)

# ── Step 3: Demand Forecasting ────────────────────────────────
print("\n[3/5] Training demand forecasting model (XGBoost)...")
from src.models.demand_model import train_demand_model
model_d, metrics_d, feat_imp_d, Xd_test, yd_test, yd_pred = train_demand_model(
    demand_df, model_type="xgboost"
)

# ── Step 4: Churn Prediction ──────────────────────────────────
print("\n[4/5] Training churn prediction models...")
from src.models.churn_model import train_churn_models, get_feature_importance
results_churn, best_model_name = train_churn_models(X_train, X_test, y_train, y_test)
feat_imp_churn = get_feature_importance(
    results_churn[best_model_name]["model"], feat_cols
)

# ── Step 5: Plots ─────────────────────────────────────────────
print("\n[5/5] Generating evaluation plots → figures/")
from src.visualization.plots import (
    plot_sales_trend, plot_category_distribution,
    plot_feature_importance, plot_confusion_matrix,
    plot_roc_curves, plot_demand_forecast
)
plot_sales_trend(sales)
plot_category_distribution(sales)
plot_feature_importance(feat_imp_d, title="Demand Model Feature Importance")
plot_demand_forecast(yd_test.values, yd_pred)
plot_confusion_matrix(
    y_test, results_churn[best_model_name]["y_pred"], model_name=best_model_name
)
plot_roc_curves(results_churn, y_test)
if feat_imp_churn is not None:
    plot_feature_importance(feat_imp_churn, title="Churn Model Feature Importance")

print("\n" + "=" * 60)
print("  Pipeline complete!")
print(f"  Demand Forecast R²:  {metrics_d['R2']}")
print(f"  Best Churn AUC:      {results_churn[best_model_name]['ROC-AUC']}")
print("  Figures saved to:    figures/")
print("  Models saved to:     models/")
print("=" * 60)
