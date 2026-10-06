from fastapi.testclient import TestClient

from ml.train import train


def test_training_and_prediction(tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_DIR", str(tmp_path / "artifacts"))
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("REPORT_DIR", str(tmp_path / "reports"))
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'predictions.db'}")
    monkeypatch.setenv("TRAINING_ROWS", "1200")
    metadata = train(promote_initial=True, register=False)
    assert metadata["threshold"] > 0
    assert (tmp_path / "artifacts" / "champion.json").exists()

    import app.main as main_module

    with TestClient(main_module.app) as client:
        response = client.post(
            "/v1/predict",
            json={
                "customer_id": "C-TEST",
                "tenure_months": 4,
                "monthly_charge": 115,
                "support_calls_30d": 5,
                "payment_failures_90d": 2,
                "usage_change_30d": -0.4,
                "late_payments_12m": 3,
                "contract_type": "month-to-month",
                "payment_method": "check",
                "region": "south"
            },
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert 0 <= body["churn_probability"] <= 1
        assert body["model_version"] == metadata["model_version"]
