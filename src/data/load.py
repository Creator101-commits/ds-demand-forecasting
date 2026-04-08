from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.utils.logger import get_logger
from src.utils.paths import ProjectPaths

logger = get_logger(__name__)


META_COLS = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]



def read_calendar(paths: ProjectPaths) -> pd.DataFrame:
    path = paths.m5_dir / "calendar.csv"
    df = pd.read_csv(path, parse_dates=["date"])
    return df



def read_sales_metadata(paths: ProjectPaths) -> pd.DataFrame:
    path = paths.m5_dir / "sales_train_validation.csv"
    return pd.read_csv(path, usecols=META_COLS)



def read_sales_subset(paths: ProjectPaths, day_cols: list[str]) -> pd.DataFrame:
    path = paths.m5_dir / "sales_train_validation.csv"
    usecols = META_COLS + day_cols
    logger.info("Loading sales subset with %s day columns", len(day_cols))
    df = pd.read_csv(path, usecols=usecols)
    df[day_cols] = df[day_cols].astype("float32")
    return df



def read_sell_prices(paths: ProjectPaths) -> pd.DataFrame:
    path = paths.m5_dir / "sell_prices.csv"
    return pd.read_csv(
        path,
        usecols=["store_id", "item_id", "wm_yr_wk", "sell_price"],
        dtype={"store_id": "string", "item_id": "string", "wm_yr_wk": "int32", "sell_price": "float32"},
    )



def read_walmart_features(paths: ProjectPaths) -> pd.DataFrame:
    path = paths.walmart_dir / "features.csv"
    df = pd.read_csv(path, parse_dates=["Date"], na_values=["NA"])
    df = df.rename(columns={"Date": "date"})
    return df



def read_walmart_train(paths: ProjectPaths) -> pd.DataFrame:
    path = paths.walmart_dir / "train.csv"
    df = pd.read_csv(path, parse_dates=["Date"])
    return df.rename(columns={"Date": "date"})



def read_walmart_stores(paths: ProjectPaths) -> pd.DataFrame:
    path = paths.walmart_dir / "stores.csv"
    return pd.read_csv(path)
