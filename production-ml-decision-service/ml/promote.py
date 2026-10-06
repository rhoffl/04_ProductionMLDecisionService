from __future__ import annotations

import argparse
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

import mlflow

from ml.config import Settings


def promote(candidate_path: Path, approved_by: str, reason: str) -> dict:
    settings = Settings()
    candidate = json.loads(candidate_path.read_text())
    metadata = candidate["metadata"]
    required = {"pr_auc": 0.20, "precision": settings.min_precision}
    failures = [f"{k}={metadata['metrics'].get(k, 0):.3f} below {v}" for k, v in required.items() if metadata["metrics"].get(k, 0) < v]
    if failures:
        raise ValueError("Promotion gates failed: " + "; ".join(failures))
    previous = settings.model_dir / "champion.json"
    if previous.exists():
        shutil.copy2(previous, settings.model_dir / "rollback.json")
    audit = {
        **candidate,
        "approved": True,
        "approved_by": approved_by,
        "approved_at": datetime.now(UTC).isoformat(),
        "reason": reason,
    }
    previous.write_text(json.dumps(audit, indent=2))
    registry_version = metadata.get("registry_version")
    if registry_version:
        mlflow.set_tracking_uri(settings.mlflow_uri)
        mlflow.MlflowClient().set_registered_model_alias(
            settings.registered_model_name, "champion", registry_version
        )
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description="Human-approved model promotion")
    parser.add_argument("--candidate", type=Path, default=Path("artifacts/challenger.json"))
    parser.add_argument("--approved-by", required=True)
    parser.add_argument("--reason", required=True)
    args = parser.parse_args()
    print(json.dumps(promote(args.candidate, args.approved_by, args.reason), indent=2))


if __name__ == "__main__":
    main()

