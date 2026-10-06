from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from ml.config import FEATURES


def generate_churn_data(n: int = 12000, seed: int = 42) -> pd.DataFrame:
    """Generate reproducible, time-indexed demo data with a non-trivial churn signal."""
    rng = np.random.default_rng(seed)
    months = pd.date_range("2024-01-01", "2026-08-01", freq="MS")
    snapshot = pd.to_datetime(rng.choice(months.to_numpy(dtype="datetime64[ns]"), n), utc=True)
    tenure = rng.integers(1, 121, n)
    charge = np.clip(rng.normal(75, 25, n), 15, 180)
    support = rng.poisson(1.2, n)
    failures = rng.poisson(0.35, n)
    usage_change = np.clip(rng.normal(-0.03, 0.23, n), -0.9, 0.8)
    late = rng.poisson(1.1, n)
    contract = rng.choice(["month-to-month", "annual", "two-year"], n, p=[0.55, 0.3, 0.15])
    payment = rng.choice(["card", "bank", "check"], n, p=[0.48, 0.4, 0.12])
    region = rng.choice(["north", "south", "east", "west"], n)

    logit = (
        -2.25
        - 0.018 * tenure
        + 0.017 * (charge - 70)
        + 0.34 * support
        + 0.55 * failures
        - 2.0 * usage_change
        + 0.12 * late
        + 1.05 * (contract == "month-to-month")
        + 0.32 * (payment == "check")
    )
    probability = 1 / (1 + np.exp(-logit))
    churn = rng.binomial(1, probability)
    frame = pd.DataFrame(
        {
            "customer_id": [f"C{i:07d}" for i in range(n)],
            "snapshot_at": pd.to_datetime(snapshot, utc=True),
            "feature_available_at": pd.to_datetime(snapshot, utc=True),
            "label_available_at": pd.to_datetime(snapshot, utc=True) + pd.to_timedelta(45, unit="D"),
            "tenure_months": tenure,
            "monthly_charge": charge.round(2),
            "support_calls_30d": support,
            "payment_failures_90d": failures,
            "usage_change_30d": usage_change.round(4),
            "late_payments_12m": late,
            "contract_type": contract,
            "payment_method": payment,
            "region": region,
            "churn_within_30d": churn,
        }
    )
    return frame.sort_values(["snapshot_at", "customer_id"]).reset_index(drop=True)


def validate_dataset(df: pd.DataFrame, require_label: bool = True) -> list[str]:
    errors: list[str] = []
    required = {"customer_id", "snapshot_at", *FEATURES}
    if require_label:
        required |= {"churn_within_30d", "feature_available_at", "label_available_at"}
    missing = required - set(df.columns)
    if missing:
        return [f"missing columns: {sorted(missing)}"]
    if df[["customer_id", "snapshot_at"]].duplicated().any():
        errors.append("duplicate customer_id/snapshot_at keys")
    if df[FEATURES].isna().any().any():
        errors.append("null values found in model features")
    if (df["monthly_charge"] < 0).any() or (df["tenure_months"] < 0).any():
        errors.append("invalid negative customer values")
    if require_label:
        snapshot = pd.to_datetime(df["snapshot_at"], utc=True)
        available = pd.to_datetime(df["feature_available_at"], utc=True)
        if (available > snapshot).any():
            errors.append("feature leakage: feature_available_at occurs after snapshot_at")
        if not set(df["churn_within_30d"].unique()).issubset({0, 1}):
            errors.append("label must be binary")
    return errors


def write_versioned_snapshot(df: pd.DataFrame, directory: Path) -> tuple[Path, Path, dict]:
    directory.mkdir(parents=True, exist_ok=True)
    payload = df.to_csv(index=False).encode()
    digest = hashlib.sha256(payload).hexdigest()
    version = f"churn-{pd.Timestamp.utcnow().strftime('%Y%m%d')}-{digest[:8]}"
    data_path = directory / f"{version}.csv"
    manifest_path = directory / f"{version}.manifest.json"
    data_path.write_bytes(payload)
    manifest = {
        "dataset_version": version,
        "created_at": datetime.now(UTC).isoformat(),
        "rows": len(df),
        "positive_rate": float(df["churn_within_30d"].mean()),
        "observation_start": str(df["snapshot_at"].min()),
        "observation_end": str(df["snapshot_at"].max()),
        "sha256": digest,
        "schema_version": "1.0",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return data_path, manifest_path, manifest
