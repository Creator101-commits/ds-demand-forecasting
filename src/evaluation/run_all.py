from __future__ import annotations

from src.evaluation.business_impact import run as run_business
from src.evaluation.evaluate import run as run_evaluate
from src.utils.logger import get_logger

logger = get_logger(__name__)



def run(config_path: str = "config/config.yaml") -> None:
    logger.info("Running metric evaluation")
    run_evaluate(config_path=config_path)

    logger.info("Running business impact evaluation")
    run_business(config_path=config_path)


if __name__ == "__main__":
    run()
