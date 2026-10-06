.PHONY: up down train test lint batch monitor promote

up:
	docker compose up --build

down:
	docker compose down

train:
	python -m ml.train --promote-initial

test:
	pytest -q

lint:
	ruff check .

batch:
	python -m scripts.batch_score --input data/current_customers.csv --output reports/batch_scores.csv

monitor:
	python -m monitoring.monitor --reference data/training_snapshot.csv --current data/current_customers.csv

promote:
	python -m ml.promote --candidate artifacts/challenger.json

