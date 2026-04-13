"""
preprocess.py
-------------
Data cleaning, feature engineering, and train/test splits.
No data leakage — scaler is always fit only on training data.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder


def load_data(raw_path="data/raw/"):
    sales = pd.read_csv(f"{raw_path}sales_data.csv", parse_dates=["date"])
    customers = pd.read_csv(f"{raw_path}customer_data.csv",
                            parse_dates=["last_purchase", "first_purchase"])
    products = pd.read_csv(f"{raw_path}products.csv")
    return sales, customers, products


def engineer_demand_features(sales_df):
    """Aggregate daily sales by category and add time-series features."""
    daily = (sales_df.groupby(["date", "category"])
             .agg(total_quantity=("quantity", "sum"),
                  total_revenue=("revenue", "sum"),
                  n_transactions=("transaction_id", "count"))
             .reset_index()
             .sort_values(["category", "date"]))

    for lag in [7, 14, 30]:
        daily[f"qty_lag_{lag}d"] = (
            daily.groupby("category")["total_quantity"].shift(lag)
        )

    daily["qty_rolling_7d"] = (
        daily.groupby("category")["total_quantity"]
        .transform(lambda x: x.rolling(7, min_periods=1).mean())
    )
    daily["qty_rolling_30d"] = (
        daily.groupby("category")["total_quantity"]
        .transform(lambda x: x.rolling(30, min_periods=1).mean())
    )

    daily["day_of_week"] = daily["date"].dt.dayofweek
    daily["month"] = daily["date"].dt.month
    daily["quarter"] = daily["date"].dt.quarter
    daily["is_weekend"] = (daily["day_of_week"] >= 5).astype(int)
    daily["is_winter"] = daily["month"].isin([11, 12, 1, 2]).astype(int)

    le = LabelEncoder()
    daily["category_encoded"] = le.fit_transform(daily["category"])
    daily = daily.dropna()
    print(f"[✓] Demand features ready: {daily.shape}")
    return daily


def prepare_churn_features(customers_df):
    """Select, split, and scale features for churn model."""
    feature_cols = [
        "total_orders", "total_revenue", "avg_order_value",
        "days_since_last_purchase", "customer_tenure_days",
        "purchase_frequency", "unique_categories"
    ]
    df = customers_df[feature_cols + ["churned"]].dropna()
    X, y = df[feature_cols], df["churned"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = pd.DataFrame(scaler.fit_transform(X_train), columns=feature_cols)
    X_test_s = pd.DataFrame(scaler.transform(X_test), columns=feature_cols)

    print(f"[✓] Churn features | Train: {len(X_train)}, Test: {len(X_test)}")
    return X_train_s, X_test_s, y_train, y_test, scaler, feature_cols
