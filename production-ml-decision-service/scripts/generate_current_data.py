from pathlib import Path

from ml.config import FEATURES
from ml.data import generate_churn_data


def main() -> None:
    frame = generate_churn_data(3000, seed=2026)
    output = frame[["customer_id", "snapshot_at", *FEATURES]]
    Path("data").mkdir(exist_ok=True)
    output.to_csv("data/current_customers.csv", index=False)


if __name__ == "__main__":
    main()

