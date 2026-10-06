import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    model_dir: Path = field(default_factory=lambda: Path(os.getenv("MODEL_DIR", "artifacts")))
    data_dir: Path = field(default_factory=lambda: Path(os.getenv("DATA_DIR", "data")))
    report_dir: Path = field(default_factory=lambda: Path(os.getenv("REPORT_DIR", "reports")))
    mlflow_uri: str = field(default_factory=lambda: os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5004"))
    experiment_name: str = "churn-decision-service"
    registered_model_name: str = "customer-churn"
    capacity: int = field(default_factory=lambda: int(os.getenv("INTERVENTION_CAPACITY", "1500")))
    min_precision: float = field(default_factory=lambda: float(os.getenv("MIN_PRECISION", "0.25")))
    retained_margin: float = field(default_factory=lambda: float(os.getenv("RETAINED_MARGIN", "300")))
    save_probability: float = field(default_factory=lambda: float(os.getenv("SAVE_PROBABILITY", "0.25")))
    intervention_cost: float = field(default_factory=lambda: float(os.getenv("INTERVENTION_COST", "12")))
    random_seed: int = 42

    def ensure_dirs(self) -> None:
        for path in (self.model_dir, self.data_dir, self.report_dir):
            path.mkdir(parents=True, exist_ok=True)


FEATURES = [
    "tenure_months",
    "monthly_charge",
    "support_calls_30d",
    "payment_failures_90d",
    "usage_change_30d",
    "late_payments_12m",
    "contract_type",
    "payment_method",
    "region",
]

NUMERIC_FEATURES = FEATURES[:6]
CATEGORICAL_FEATURES = FEATURES[6:]
