"""
generate_data.py
----------------
Generates synthetic pharmacy e-commerce data that mimics real-world patterns:
- Sales transactions with seasonality (flu season, festive peaks)
- Customer profiles with churn labels
- Product catalog with drug categories
"""

import numpy as np
import pandas as pd
import os


def generate_sales_data(n_customers=5000, n_products=50,
                        date_start="2022-01-01", date_end="2024-12-31",
                        save_path="data/raw/"):
    """
    Generate synthetic drug sales transactions.
    Includes seasonal demand spikes (winter = cold/flu drugs peak).
    """
    np.random.seed(42)
    dates = pd.date_range(start=date_start, end=date_end, freq="D")

    categories = ["Antibiotics", "Vitamins", "Pain Relief",
                  "Diabetes", "Cardiovascular", "Skincare", "Allergy"]
    products = pd.DataFrame({
        "product_id": range(1, n_products + 1),
        "product_name": [f"Drug_{i}" for i in range(1, n_products + 1)],
        "category": np.random.choice(categories, n_products),
        "base_price": np.round(np.random.uniform(50, 800, n_products), 2)
    })

    records = []
    for _ in range(80000):
        date = np.random.choice(dates)
        date = pd.to_datetime(date)  
        month = date.month
        seasonal_factor = 1.4 if month in [11, 12, 1, 2] else 1.0

        product = products.sample(1).iloc[0]
        customer_id = np.random.randint(1, n_customers + 1)
        quantity = int(np.random.poisson(2 * seasonal_factor) + 1)
        discount = np.random.choice([0, 0.05, 0.10, 0.15], p=[0.5, 0.2, 0.2, 0.1])
        revenue = round(product["base_price"] * quantity * (1 - discount), 2)

        records.append({
            "transaction_id": len(records) + 1,
            "customer_id": customer_id,
            "product_id": product["product_id"],
            "category": product["category"],
            "date": date,
            "quantity": quantity,
            "unit_price": product["base_price"],
            "discount": discount,
            "revenue": revenue
        })

    df = pd.DataFrame(records)
    os.makedirs(save_path, exist_ok=True)
    df.to_csv(f"{save_path}sales_data.csv", index=False)
    products.to_csv(f"{save_path}products.csv", index=False)
    print(f"[✓] Generated {len(df)} sales records → {save_path}sales_data.csv")
    return df, products


def generate_customer_data(sales_df, n_customers=5000, save_path="data/raw/"):
    """
    Build customer-level features and assign churn labels.
    Churn = no purchase in the last 90 days of the dataset.
    """
    max_date = sales_df["date"].max()
    cutoff = max_date - pd.Timedelta(days=90)

    customer_stats = sales_df.groupby("customer_id").agg(
        total_orders=("transaction_id", "count"),
        total_revenue=("revenue", "sum"),
        avg_order_value=("revenue", "mean"),
        last_purchase=("date", "max"),
        first_purchase=("date", "min"),
        unique_categories=("category", "nunique")
    ).reset_index()

    customer_stats["days_since_last_purchase"] = (
        max_date - customer_stats["last_purchase"]).dt.days
    customer_stats["customer_tenure_days"] = (
        customer_stats["last_purchase"] - customer_stats["first_purchase"]).dt.days
    customer_stats["purchase_frequency"] = (
        customer_stats["total_orders"] /
        (customer_stats["customer_tenure_days"] + 1)
    )
    customer_stats["churned"] = (
        customer_stats["last_purchase"] < cutoff).astype(int)

    os.makedirs(save_path, exist_ok=True)
    customer_stats.to_csv(f"{save_path}customer_data.csv", index=False)
    churn_rate = customer_stats["churned"].mean() * 100
    print(f"[✓] Generated {len(customer_stats)} customer profiles | Churn rate: {churn_rate:.1f}%")
    return customer_stats
