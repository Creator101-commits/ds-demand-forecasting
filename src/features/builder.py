from __future__ import annotations

import pandas as pd

from src.features.event_encoding import add_event_features
from src.features.lag_rolling import add_sales_lag_features
from src.features.macro_features import add_macro_features
from src.features.price_features import add_price_features
from src.features.time_features import add_time_features
from src.utils.config import load_config
from src.utils.logger import get_logger
from src.utils.paths import build_paths

logger = get_logger(__name__)



def build_features(config_path: str = "config/config.yaml") -> pd.DataFrame:
    config = load_config(config_path)
    paths = build_paths(config)

    input_path = paths.processed_dir / "dept_store_daily_base.parquet"
    if not input_path.exists():
        raise FileNotFoundError(f"Expected base dataset not found: {input_path}")

    df = pd.read_parquet(input_path)
    df["date"] = pd.to_datetime(df["date"])

    group_cols = ["store_id", "dept_id"]

    logger.info("Adding time features")
    df = add_time_features(df)

    logger.info("Adding event features")
    df = add_event_features(df)

    logger.info("Adding lag and rolling features")
    df = add_sales_lag_features(
        df,
        group_cols=group_cols,
        lag_days=config["features"]["lag_days"],
        rolling_windows=config["features"]["rolling_windows"],
    )

    logger.info("Adding price features")
    df = add_price_features(df, group_cols=group_cols)

    logger.info("Adding macro features")
    df = add_macro_features(df)

    out_path = paths.processed_dir / "featured_dept_store_daily.parquet"
    df.to_parquet(out_path, index=False)
    logger.info("Saved feature dataset to %s with shape=%s", out_path, df.shape)
    return df



def run(config_path: str = "config/config.yaml") -> None:
    build_features(config_path=config_path)


if __name__ == "__main__":
    run()
