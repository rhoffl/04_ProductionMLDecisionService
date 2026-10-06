from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class CustomerFeatures(BaseModel):
    model_config = ConfigDict(extra="forbid")
    customer_id: str = Field(min_length=1, max_length=80)
    tenure_months: int = Field(ge=0, le=240)
    monthly_charge: float = Field(ge=0, le=10000)
    support_calls_30d: int = Field(ge=0, le=100)
    payment_failures_90d: int = Field(ge=0, le=100)
    usage_change_30d: float = Field(ge=-1, le=5)
    late_payments_12m: int = Field(ge=0, le=100)
    contract_type: Literal["month-to-month", "annual", "two-year"]
    payment_method: Literal["card", "bank", "check"]
    region: Literal["north", "south", "east", "west"]


class Reason(BaseModel):
    feature: str
    value: str | int | float
    direction: Literal["increases risk", "decreases risk"]


class Prediction(BaseModel):
    prediction_id: str
    customer_id: str
    churn_probability: float
    decision: Literal["contact", "do_not_contact"]
    threshold: float
    model_version: str
    dataset_version: str
    scored_at: datetime
    reasons: list[Reason]


class Outcome(BaseModel):
    prediction_id: str
    churned: bool
    contacted: bool = False
    retained: bool | None = None
    observed_at: datetime

