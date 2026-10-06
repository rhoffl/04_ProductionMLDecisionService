from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from ml.config import CATEGORICAL_FEATURES, NUMERIC_FEATURES


def population_stability_index(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    reference = pd.to_numeric(reference, errors="coerce").dropna()
    current = pd.to_numeric(current, errors="coerce").dropna()
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    ref_counts = pd.cut(reference, edges, include_lowest=True).value_counts(normalize=True, sort=False)
    cur_counts = pd.cut(current, edges, include_lowest=True).value_counts(normalize=True, sort=False)
    ref_p = np.clip(ref_counts.to_numpy(), 1e-6, None)
    cur_p = np.clip(cur_counts.to_numpy(), 1e-6, None)
    return float(np.sum((cur_p - ref_p) * np.log(cur_p / ref_p)))


def categorical_drift(reference: pd.Series, current: pd.Series) -> float:
    categories = sorted(set(reference.dropna().astype(str)) | set(current.dropna().astype(str)))
    ref = reference.astype(str).value_counts(normalize=True).reindex(categories, fill_value=1e-6)
    cur = current.astype(str).value_counts(normalize=True).reindex(categories, fill_value=1e-6)
    return float(0.5 * np.abs(ref - cur).sum())


def build_drift_report(reference: pd.DataFrame, current: pd.DataFrame) -> dict:
    features = {}
    for feature in NUMERIC_FEATURES:
        score = population_stability_index(reference[feature], current[feature])
        features[feature] = {"metric": "psi", "score": score, "drifted": score >= 0.2}
    for feature in CATEGORICAL_FEATURES:
        score = categorical_drift(reference[feature], current[feature])
        features[feature] = {"metric": "total_variation", "score": score, "drifted": score >= 0.1}
    drifted = [name for name, result in features.items() if result["drifted"]]
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "reference_rows": len(reference),
        "current_rows": len(current),
        "drifted_features": drifted,
        "alert": len(drifted) >= 2,
        "features": features,
        "interpretation": "Drift is an investigation trigger, not proof of model-performance degradation.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--current", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("reports/drift_report.json"))
    args = parser.parse_args()
    report = build_drift_report(pd.read_csv(args.reference), pd.read_csv(args.current))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

