from __future__ import annotations

import json

import pandas as pd

from src.utils.config import load_config
from src.utils.logger import get_logger
from src.utils.paths import build_paths

logger = get_logger(__name__)



def create_temporal_splits(df: pd.DataFrame, val_days: int, test_days: int) -> pd.DataFrame:
    unique_dates = sorted(df["date"].dropna().unique())
    if len(unique_dates) <= (val_days + test_days):
        raise ValueError("Not enough dates to create train/validation/test splits.")

    train_end = unique_dates[-(val_days + test_days)]
    val_end = unique_dates[-test_days]

    out = df.copy()
    out["split"] = "train"
    out.loc[out["date"] >= val_end, "split"] = "test"
    out.loc[(out["date"] >= train_end) & (out["date"] < val_end), "split"] = "val"
    return out



def run(config_path: str = "config/config.yaml") -> pd.DataFrame:
    config = load_config(config_path)
    paths = build_paths(config)

    split_cfg = config["split"]
    input_path = paths.processed_dir / "modeling_dept_store_daily.parquet"
    if not input_path.exists():
        input_path = paths.processed_dir / "featured_dept_store_daily.parquet"

    df = pd.read_parquet(input_path)
    df["date"] = pd.to_datetime(df["date"])

    split_df = create_temporal_splits(df, split_cfg["val_days"], split_cfg["test_days"])

    split_path = paths.processed_dir / "modeling_with_splits.parquet"
    split_df.to_parquet(split_path, index=False)

    split_counts = split_df.groupby("split")["date"].nunique().to_dict()
    meta = {
        "val_days": split_cfg["val_days"],
        "test_days": split_cfg["test_days"],
        "unique_days_per_split": split_counts,
    }
    with (paths.processed_dir / "split_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    logger.info("Saved split dataset to %s", split_path)
    logger.info("Split days: %s", split_counts)
    return split_df


if __name__ == "__main__":
    run()
