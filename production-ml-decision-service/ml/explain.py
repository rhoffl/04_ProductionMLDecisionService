from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap

from ml.config import FEATURES


def create_shap_report(model_path: Path, sample: pd.DataFrame, output: Path) -> dict:
    """Create SHAP values against the fitted estimator's transformed feature space."""
    calibrated = joblib.load(model_path)
    pipeline = calibrated.calibrated_classifiers_[0].estimator
    if hasattr(pipeline, "estimator"):
        pipeline = pipeline.estimator
    transformer = pipeline.named_steps["preprocess"]
    estimator = pipeline.named_steps["model"]
    transformed = transformer.transform(sample[FEATURES].head(min(100, len(sample))))
    names = transformer.get_feature_names_out().tolist()
    background = transformed[: min(40, len(transformed))]
    explained = transformed[: min(20, len(transformed))]
    explainer = shap.Explainer(estimator, background, feature_names=names)
    values = explainer(explained)
    raw = values.values
    if raw.ndim == 3:
        raw = raw[:, :, -1]
    mean_abs = np.abs(raw).mean(axis=0)
    report = {
        "method": "model-agnostic SHAP",
        "sample_size": len(explained),
        "global_importance": [
            {"feature": feature, "mean_absolute_shap": float(score)}
            for feature, score in sorted(zip(names, mean_abs), key=lambda pair: pair[1], reverse=True)
        ],
        "warning": "SHAP describes model behavior; it does not establish causal effects.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2))
    return report
