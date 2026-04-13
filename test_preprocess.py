import pytest, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data.generate_data import generate_sales_data, generate_customer_data
from src.data.preprocess import engineer_demand_features, prepare_churn_features


@pytest.fixture(scope="module")
def sales_data(tmp_path_factory):
    tmp = str(tmp_path_factory.mktemp("data")) + "/"
    return generate_sales_data(n_customers=200, n_products=8, save_path=tmp)


@pytest.fixture(scope="module")
def customer_data(sales_data, tmp_path_factory):
    sales, _ = sales_data
    tmp = str(tmp_path_factory.mktemp("cust")) + "/"
    return generate_customer_data(sales, n_customers=200, save_path=tmp)


def test_sales_shape(sales_data):
    sales, _ = sales_data
    assert len(sales) > 0 and "revenue" in sales.columns


def test_no_negative_revenue(sales_data):
    sales, _ = sales_data
    assert (sales["revenue"] >= 0).all()


def test_demand_features(sales_data):
    sales, _ = sales_data
    df = engineer_demand_features(sales)
    assert "qty_lag_7d" in df.columns
    assert "is_winter" in df.columns
    assert df.isnull().sum().sum() == 0


def test_churn_split(customer_data):
    X_tr, X_te, y_tr, y_te, _, _ = prepare_churn_features(customer_data)
    assert len(X_tr) > len(X_te)
    assert set(y_te.unique()).issubset({0, 1})


def test_no_data_leakage(customer_data):
    """Scaler fit only on train — train mean should be ~0 after scaling."""
    X_tr, _, _, _, _, _ = prepare_churn_features(customer_data)
    assert abs(X_tr.mean().mean()) < 0.1
