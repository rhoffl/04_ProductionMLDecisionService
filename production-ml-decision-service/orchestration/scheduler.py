from __future__ import annotations

import logging
import os
import subprocess
import sys

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def retrain() -> None:
    logger.info("Starting scheduled challenger training")
    result = subprocess.run([sys.executable, "-m", "ml.train"], check=False)
    if result.returncode:
        logger.error("Retraining failed with code %s", result.returncode)
    else:
        logger.info("Challenger created; human promotion is required")


def main() -> None:
    expression = os.getenv("RETRAIN_CRON", "0 2 1 * *")
    scheduler = BlockingScheduler(timezone="UTC")
    scheduler.add_job(retrain, CronTrigger.from_crontab(expression), id="monthly-retraining", max_instances=1)
    logger.info("Scheduler started with cron: %s", expression)
    scheduler.start()


if __name__ == "__main__":
    main()

