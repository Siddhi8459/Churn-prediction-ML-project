import pytest, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data.generate_data import generate_sales_data, generate_customer_data
from src.data.preprocess import engineer_demand_features, prepare_churn_features
from src.models.demand_model import train_demand_model
from src.models.churn_model import train_churn_models


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory):
    tmp = str(tmp_path_factory.mktemp("pipe")) + "/"
    sales, _ = generate_sales_data(n_customers=300, n_products=8, save_path=tmp)
    customers = generate_customer_data(sales, save_path=tmp)
    demand_df = engineer_demand_features(sales)
    X_tr, X_te, y_tr, y_te, _, cols = prepare_churn_features(customers)
    return demand_df, X_tr, X_te, y_tr, y_te, cols


def test_demand_r2(pipeline):
    demand_df, *_ = pipeline
    _, metrics, _, _, _, _ = train_demand_model(demand_df)
    assert metrics["R2"] > 0.5


def test_demand_metrics_keys(pipeline):
    demand_df, *_ = pipeline
    _, metrics, _, _, _, _ = train_demand_model(demand_df)
    assert all(k in metrics for k in ["MAE", "RMSE", "R2"])


def test_churn_auc(pipeline):
    _, X_tr, X_te, y_tr, y_te, _ = pipeline
    results, _ = train_churn_models(X_tr, X_te, y_tr, y_te)
    for name, res in results.items():
        assert res["ROC-AUC"] > 0.5, f"{name} AUC too low"


def test_all_models_trained(pipeline):
    _, X_tr, X_te, y_tr, y_te, _ = pipeline
    results, _ = train_churn_models(X_tr, X_te, y_tr, y_te)
    assert {"Logistic Regression", "Random Forest", "Gradient Boosting"} == set(results.keys())
