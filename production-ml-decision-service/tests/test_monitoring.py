from ml.data import generate_churn_data
from monitoring.monitor import build_drift_report


def test_monitor_detects_large_drift():
    reference = generate_churn_data(1000, seed=1)
    current = generate_churn_data(1000, seed=2)
    current["monthly_charge"] = current["monthly_charge"] + 200
    current["support_calls_30d"] = current["support_calls_30d"] + 8
    report = build_drift_report(reference, current)
    assert report["alert"] is True
    assert "monthly_charge" in report["drifted_features"]

