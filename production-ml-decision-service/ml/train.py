from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from datetime import UTC, datetime

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import ParameterGrid, TimeSeriesSplit

from ml.config import FEATURES, Settings
from ml.data import generate_churn_data, validate_dataset, write_versioned_snapshot
from ml.economics import choose_threshold, threshold_report
from ml.explain import create_shap_report
from ml.pipeline import candidates


def split_by_time(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ordered = df.sort_values("snapshot_at")
    train_end = ordered["snapshot_at"].quantile(0.70)
    calibration_end = ordered["snapshot_at"].quantile(0.85)
    train = ordered[ordered["snapshot_at"] <= train_end]
    calibration = ordered[(ordered["snapshot_at"] > train_end) & (ordered["snapshot_at"] <= calibration_end)]
    test = ordered[ordered["snapshot_at"] > calibration_end]
    return train, calibration, test


def metric_bundle(y: np.ndarray, p: np.ndarray, threshold: float) -> dict[str, float]:
    pred = p >= threshold
    return {
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "log_loss": float(log_loss(y, p)),
        "brier_score": float(brier_score_loss(y, p)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
    }


def temporal_cv_score(model, frame: pd.DataFrame) -> tuple[float, float]:
    ordered = frame.sort_values("snapshot_at")
    splitter = TimeSeriesSplit(n_splits=3)
    scores: list[float] = []
    for train_idx, valid_idx in splitter.split(ordered):
        train_fold, valid_fold = ordered.iloc[train_idx], ordered.iloc[valid_idx]
        model.fit(train_fold[FEATURES], train_fold["churn_within_30d"])
        p = model.predict_proba(valid_fold[FEATURES])[:, 1]
        scores.append(average_precision_score(valid_fold["churn_within_30d"], p))
    return float(np.mean(scores)), float(np.std(scores))


def wait_for_mlflow(uri: str, attempts: int = 20) -> bool:
    import urllib.request

    if uri.startswith(("file:", "sqlite:")):
        return True
    for _ in range(attempts):
        try:
            urllib.request.urlopen(f"{uri.rstrip('/')}/health", timeout=2)
            return True
        except OSError:
            time.sleep(2)
    return False


def train(promote_initial: bool = False, register: bool = True) -> dict:
    settings = Settings()
    settings.ensure_dirs()
    frame = generate_churn_data(n=int(os.getenv("TRAINING_ROWS", "12000")), seed=settings.random_seed)
    errors = validate_dataset(frame)
    if errors:
        raise ValueError(f"Dataset validation failed: {errors}")
    data_path, manifest_path, manifest = write_versioned_snapshot(frame, settings.data_dir)
    shutil.copy2(data_path, settings.data_dir / "training_snapshot.csv")

    train_df, calibration_df, test_df = split_by_time(frame)
    model_results: list[dict] = []
    fitted: dict[str, object] = {}
    search_spaces = {
        "logistic_baseline": {"model__C": [0.5, 2.0]},
        "random_forest": {"model__min_samples_leaf": [5, 12], "model__max_depth": [10]},
        "hist_gradient_boosting": {"model__max_leaf_nodes": [16, 32], "model__l2_regularization": [1.0]},
    }
    for name, template in candidates(settings.random_seed).items():
        trials = []
        for params in ParameterGrid(search_spaces[name]):
            trial = clone(template).set_params(**params)
            cv_mean, cv_std = temporal_cv_score(trial, train_df)
            trials.append((cv_mean, cv_std, params, trial))
        cv_mean, cv_std, best_params, model = max(trials, key=lambda item: item[0])
        model.fit(train_df[FEATURES], train_df["churn_within_30d"])
        p = model.predict_proba(calibration_df[FEATURES])[:, 1]
        fitted[name] = model
        model_results.append(
            {
                "model": name,
                "cv_pr_auc_mean": cv_mean,
                "cv_pr_auc_std": cv_std,
                "calibration_pr_auc": float(average_precision_score(calibration_df["churn_within_30d"], p)),
                "best_params": json.dumps(best_params, sort_keys=True),
            }
        )

    result_frame = pd.DataFrame(model_results).sort_values(
        ["calibration_pr_auc", "cv_pr_auc_mean"], ascending=False
    )
    selected_name = str(result_frame.iloc[0]["model"])
    base_model = fitted[selected_name]
    calibrated = CalibratedClassifierCV(FrozenEstimator(base_model), method="sigmoid")
    calibrated.fit(calibration_df[FEATURES], calibration_df["churn_within_30d"])
    calibration_prob = calibrated.predict_proba(calibration_df[FEATURES])[:, 1]

    capacity_for_sample = max(1, round(settings.capacity * len(calibration_df) / len(frame)))
    economics = threshold_report(
        calibration_df["churn_within_30d"].to_numpy(),
        calibration_prob,
        capacity_for_sample,
        settings.retained_margin,
        settings.save_probability,
        settings.intervention_cost,
        settings.min_precision,
    )
    selected_threshold = choose_threshold(economics)
    threshold = float(selected_threshold["threshold"])
    test_prob = calibrated.predict_proba(test_df[FEATURES])[:, 1]
    metrics = metric_bundle(test_df["churn_within_30d"].to_numpy(), test_prob, threshold)
    metrics.update(
        {
            "expected_net_value": float(selected_threshold["net_value"]),
            "capacity_utilization": float(selected_threshold["capacity_utilization"]),
        }
    )

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    bundle_name = f"model-{timestamp}"
    bundle_dir = settings.model_dir / bundle_name
    bundle_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(calibrated, bundle_dir / "model.joblib")
    metadata = {
        "bundle": bundle_name,
        "created_at": datetime.now(UTC).isoformat(),
        "model_type": selected_name,
        "model_version": timestamp,
        "dataset_version": manifest["dataset_version"],
        "features": FEATURES,
        "threshold": threshold,
        "capacity": settings.capacity,
        "metrics": metrics,
        "business_assumptions": {
            "retained_margin": settings.retained_margin,
            "save_probability": settings.save_probability,
            "intervention_cost": settings.intervention_cost,
        },
    }
    (bundle_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    result_frame.to_csv(settings.report_dir / "candidate_comparison.csv", index=False)
    economics.to_csv(settings.report_dir / "threshold_analysis.csv", index=False)
    (settings.report_dir / "test_metrics.json").write_text(json.dumps(metrics, indent=2))
    try:
        create_shap_report(bundle_dir / "model.joblib", test_df, settings.report_dir / "shap_report.json")
    except (TypeError, ValueError, RuntimeError) as exc:
        (settings.report_dir / "shap_report.json").write_text(
            json.dumps({"status": "failed", "error": str(exc), "action": "Inspect model/explainer compatibility"}, indent=2)
        )

    registry_version = None
    if register and wait_for_mlflow(settings.mlflow_uri):
        mlflow.set_tracking_uri(settings.mlflow_uri)
        mlflow.set_experiment(settings.experiment_name)
        with mlflow.start_run(run_name=bundle_name) as run:
            mlflow.log_params(
                {
                    "model_type": selected_name,
                    "dataset_version": manifest["dataset_version"],
                    "threshold": threshold,
                    **metadata["business_assumptions"],
                }
            )
            mlflow.log_metrics(metrics)
            mlflow.log_artifact(str(manifest_path), artifact_path="dataset")
            mlflow.log_artifact(str(settings.report_dir / "candidate_comparison.csv"), artifact_path="reports")
            mlflow.log_artifact(str(settings.report_dir / "threshold_analysis.csv"), artifact_path="reports")
            mlflow.log_artifact(str(settings.report_dir / "shap_report.json"), artifact_path="reports")
            mlflow.sklearn.log_model(
                calibrated,
                artifact_path="model",
                registered_model_name=settings.registered_model_name,
                input_example=train_df[FEATURES].head(3),
                # These implementation types are created locally by the
                # calibrated logistic/tree candidates in this training job.
                # Keep this allowlist explicit; never derive it from an
                # untrusted model artifact.
                skops_trusted_types=[
                    "numpy.dtype",
                    "sklearn.calibration._CalibratedClassifier",
                    "sklearn.calibration._SigmoidCalibration",
                    "sklearn.tree._tree.Tree",
                ],
            )
            client = mlflow.MlflowClient()
            versions = client.search_model_versions(f"run_id='{run.info.run_id}'")
            if versions:
                registry_version = str(versions[0].version)
                client.set_model_version_tag(
                    settings.registered_model_name, registry_version, "validation_status", "passed"
                )
                if promote_initial and not (settings.model_dir / "champion.json").exists():
                    client.set_registered_model_alias(settings.registered_model_name, "champion", registry_version)
    metadata["registry_version"] = registry_version
    (bundle_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    alias_name = "champion.json" if promote_initial and not (settings.model_dir / "champion.json").exists() else "challenger.json"
    alias = {"bundle": bundle_name, "approved": alias_name == "champion.json", "metadata": metadata}
    (settings.model_dir / alias_name).write_text(json.dumps(alias, indent=2))
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--promote-initial", action="store_true")
    parser.add_argument("--no-register", action="store_true")
    args = parser.parse_args()
    result = train(promote_initial=args.promote_initial, register=not args.no_register)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
