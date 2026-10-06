from __future__ import annotations

import argparse
import uuid
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from app.predictor import ModelManager
from ml.config import FEATURES
from ml.data import validate_dataset


def score_file(input_path: Path, output_path: Path) -> pd.DataFrame:
    frame = pd.read_csv(input_path)
    errors = validate_dataset(frame, require_label=False)
    if errors:
        raise ValueError(errors)
    manager = ModelManager()
    manager.load()
    probabilities = manager.model.predict_proba(frame[FEATURES])[:, 1]
    threshold = float(manager.metadata["threshold"])
    result = pd.DataFrame(
        {
            "prediction_id": [str(uuid.uuid4()) for _ in range(len(frame))],
            "customer_id": frame["customer_id"],
            "score": probabilities,
            "decision": ["contact" if p >= threshold else "do_not_contact" for p in probabilities],
            "threshold": threshold,
            "model_version": manager.metadata["model_version"],
            "scored_at": datetime.now(UTC).isoformat(),
        }
    )
    result["priority_rank"] = result["score"].rank(method="first", ascending=False).astype(int)
    result.loc[result["priority_rank"] > int(manager.metadata["capacity"]), "decision"] = "do_not_contact"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.sort_values("priority_rank").to_csv(output_path, index=False)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = score_file(args.input, args.output)
    print(f"Scored {len(result)} customers; selected {(result.decision == 'contact').sum()}")


if __name__ == "__main__":
    main()

