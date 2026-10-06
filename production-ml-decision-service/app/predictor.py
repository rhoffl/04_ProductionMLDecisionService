from __future__ import annotations

import json
import threading
import uuid
from datetime import UTC, datetime

import joblib
import pandas as pd

from ml.config import FEATURES, Settings


class ModelNotReady(RuntimeError):
    pass


class ModelManager:
    def __init__(self, alias: str = "champion.json") -> None:
        self.settings = Settings()
        self.alias = alias
        self.lock = threading.RLock()
        self.model = None
        self.metadata: dict = {}
        self.alias_mtime = 0.0

    def load(self, force: bool = False) -> None:
        alias_path = self.settings.model_dir / self.alias
        if not alias_path.exists():
            raise ModelNotReady("Champion model not found; run the training workflow")
        mtime = alias_path.stat().st_mtime
        if not force and self.model is not None and mtime <= self.alias_mtime:
            return
        alias = json.loads(alias_path.read_text())
        bundle = self.settings.model_dir / alias["bundle"]
        model = joblib.load(bundle / "model.joblib")
        metadata = json.loads((bundle / "metadata.json").read_text())
        with self.lock:
            self.model, self.metadata, self.alias_mtime = model, metadata, mtime

    def predict_one(self, features: dict) -> dict:
        self.load()
        frame = pd.DataFrame([{k: features[k] for k in FEATURES}])
        with self.lock:
            probability = float(self.model.predict_proba(frame)[0, 1])
            threshold = float(self.metadata["threshold"])
            metadata = dict(self.metadata)
        return {
            "prediction_id": str(uuid.uuid4()),
            "customer_id": features["customer_id"],
            "churn_probability": probability,
            "decision": "contact" if probability >= threshold else "do_not_contact",
            "threshold": threshold,
            "model_version": metadata["model_version"],
            "dataset_version": metadata["dataset_version"],
            "scored_at": datetime.now(UTC),
            "reasons": reason_codes(features),
        }


def reason_codes(row: dict) -> list[dict]:
    """Operational reason codes; full SHAP values are produced in the model report."""
    candidates = [
        (abs(row["usage_change_30d"]), "usage_change_30d", row["usage_change_30d"], "increases risk" if row["usage_change_30d"] < 0 else "decreases risk"),
        (row["support_calls_30d"] / 4, "support_calls_30d", row["support_calls_30d"], "increases risk"),
        (row["payment_failures_90d"] / 2, "payment_failures_90d", row["payment_failures_90d"], "increases risk"),
        (1 if row["contract_type"] == "month-to-month" else 0.2, "contract_type", row["contract_type"], "increases risk" if row["contract_type"] == "month-to-month" else "decreases risk"),
        (min(row["tenure_months"] / 60, 1), "tenure_months", row["tenure_months"], "decreases risk"),
    ]
    return [
        {"feature": name, "value": value, "direction": direction}
        for _, name, value, direction in sorted(candidates, reverse=True)[:3]
    ]

