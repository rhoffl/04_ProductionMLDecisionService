import pandas as pd

from ml.data import generate_churn_data, validate_dataset


def test_generated_dataset_is_valid():
    frame = generate_churn_data(500)
    assert validate_dataset(frame) == []
    assert 0 < frame["churn_within_30d"].mean() < 1


def test_future_feature_is_rejected():
    frame = generate_churn_data(50)
    frame.loc[0, "feature_available_at"] = pd.Timestamp(frame.loc[0, "snapshot_at"]) + pd.to_timedelta(1, unit="D")
    assert any("leakage" in error for error in validate_dataset(frame))
