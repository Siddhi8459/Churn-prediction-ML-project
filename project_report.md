# Project Report: PharmEasy ML Pipeline

## Business Problem

PharmEasy faces two core data science challenges:
1. **Drug demand volatility** — overstocking raises costs; understocking harms service
2. **Customer churn** — retaining buyers is far cheaper than acquiring new ones

---

## Data

Synthetic pharmacy transaction data designed to mimic real-world patterns:
- 80,000+ transactions across 5,000 customers and 50 products
- 7 drug categories with realistic seasonal demand
- 3-year observation window (2022–2024)

---

## Methodology

### Demand Forecasting
- Aggregated daily sales per category
- Lag features (7, 14, 30 days) and rolling averages
- Seasonality indicators: month, quarter, winter flag
- XGBoost vs Random Forest (time-aware split, no shuffle to prevent leakage)

### Churn Prediction
- Churn defined as: no purchase in final 90 days of observation window
- RFM-inspired features: recency, frequency, monetary value, tenure, category diversity
- Compared Logistic Regression, Random Forest, Gradient Boosting
- Class imbalance handled via `class_weight="balanced"`
- Evaluation: ROC-AUC, F1-score, confusion matrix

---

## Results

| Task | Model | Metric | Value |
|---|---|---|---|
| Demand | XGBoost | R² | ~0.91 |
| Demand | Random Forest | R² | ~0.87 |
| Churn | Random Forest | ROC-AUC | ~0.88 |
| Churn | Gradient Boosting | F1 | ~0.84 |

---

## Business Insights

- Winter months (Nov–Feb) drive ~40% higher drug demand → stock accordingly
- Days since last purchase is the strongest churn signal
- Customers buying across 3+ categories churn ~60% less
- Antibiotics and Pain Relief are highest-revenue categories
