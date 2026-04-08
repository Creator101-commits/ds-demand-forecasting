from __future__ import annotations

from src.models.lightgbm import train_and_predict as run_lgbm
from src.models.prophet import train_and_predict as run_prophet
from src.models.sarima import train_and_predict as run_sarima
from src.utils.logger import get_logger

logger = get_logger(__name__)



def run(config_path: str = "config/config.yaml") -> None:
    logger.info("Running SARIMA")
    run_sarima(config_path=config_path)

    logger.info("Running Prophet")
    run_prophet(config_path=config_path)

    logger.info("Running LightGBM")
    run_lgbm(config_path=config_path)


if __name__ == "__main__":
    run()
