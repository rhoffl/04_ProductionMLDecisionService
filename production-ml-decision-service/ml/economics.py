from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score


def threshold_report(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    capacity: int,
    retained_margin: float,
    save_probability: float,
    intervention_cost: float,
    min_precision: float = 0.0,
) -> pd.DataFrame:
    rows = []
    for threshold in np.linspace(0.05, 0.95, 91):
        predicted = probabilities >= threshold
        contacted = int(predicted.sum())
        tp = int(((y_true == 1) & predicted).sum())
        precision = precision_score(y_true, predicted, zero_division=0)
        recall = recall_score(y_true, predicted, zero_division=0)
        value = tp * save_probability * retained_margin - contacted * intervention_cost
        rows.append(
            {
                "threshold": round(float(threshold), 2),
                "contacted": contacted,
                "true_positives": tp,
                "precision": float(precision),
                "recall": float(recall),
                "capacity_utilization": contacted / max(capacity, 1),
                "net_value": float(value),
                "feasible": contacted <= capacity and precision >= min_precision,
            }
        )
    return pd.DataFrame(rows)


def choose_threshold(report: pd.DataFrame) -> dict:
    feasible = report[report["feasible"]]
    if feasible.empty:
        raise ValueError("No threshold satisfies capacity and minimum-precision constraints")
    return feasible.sort_values(["net_value", "precision"], ascending=False).iloc[0].to_dict()

