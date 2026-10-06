from __future__ import annotations

import json
import os
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager


class PredictionStore:
    def __init__(self) -> None:
        self.url = os.getenv("DATABASE_URL", "sqlite:///data/predictions.db")
        self._init()

    @contextmanager
    def connect(self) -> Iterator:
        if self.url.startswith("postgresql"):
            import psycopg

            with psycopg.connect(self.url) as connection:
                yield connection
        else:
            path = self.url.removeprefix("sqlite:///")
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            with sqlite3.connect(path) as connection:
                yield connection

    def _init(self) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS predictions (
                    prediction_id TEXT PRIMARY KEY,
                    customer_id TEXT NOT NULL,
                    probability DOUBLE PRECISION NOT NULL,
                    decision TEXT NOT NULL,
                    threshold DOUBLE PRECISION NOT NULL,
                    model_version TEXT NOT NULL,
                    dataset_version TEXT NOT NULL,
                    scored_at TEXT NOT NULL,
                    features_json TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS outcomes (
                    prediction_id TEXT PRIMARY KEY,
                    churned INTEGER NOT NULL,
                    contacted INTEGER NOT NULL,
                    retained INTEGER,
                    observed_at TEXT NOT NULL
                )
                """
            )
            connection.commit()

    def save_prediction(self, record: dict, features: dict) -> None:
        values = (
            record["prediction_id"],
            record["customer_id"],
            record["churn_probability"],
            record["decision"],
            record["threshold"],
            record["model_version"],
            record["dataset_version"],
            str(record["scored_at"]),
            json.dumps(features),
        )
        sql = "INSERT INTO predictions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)"
        if self.url.startswith("postgresql"):
            sql = sql.replace("?", "%s")
        with self.connect() as connection:
            connection.execute(sql, values)
            connection.commit()

    def save_outcome(self, outcome: dict) -> None:
        values = (
            outcome["prediction_id"],
            int(outcome["churned"]),
            int(outcome["contacted"]),
            None if outcome["retained"] is None else int(outcome["retained"]),
            str(outcome["observed_at"]),
        )
        placeholder = "%s" if self.url.startswith("postgresql") else "?"
        sql = f"INSERT INTO outcomes VALUES ({', '.join([placeholder] * 5)}) ON CONFLICT(prediction_id) DO UPDATE SET churned=excluded.churned, contacted=excluded.contacted, retained=excluded.retained, observed_at=excluded.observed_at"
        with self.connect() as connection:
            connection.execute(sql, values)
            connection.commit()

