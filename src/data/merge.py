from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.load import (
    read_calendar,
    read_sales_metadata,
    read_sales_subset,
    read_sell_prices,
    read_walmart_features,
)
from src.utils.config import load_config
from src.utils.logger import get_logger
from src.utils.paths import build_paths

logger = get_logger(__name__)



def _extract_overlap_calendar(calendar: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    mask = (calendar["date"] >= pd.to_datetime(start_date)) & (calendar["date"] <= pd.to_datetime(end_date))
    overlap = calendar.loc[mask].copy()
    if overlap.empty:
        raise ValueError("No overlap rows found in calendar for configured window.")
    return overlap



def _build_price_features(paths, sales_meta: pd.DataFrame) -> pd.DataFrame:
    sell_prices = read_sell_prices(paths)
    item_dept_map = sales_meta[["item_id", "dept_id", "store_id"]].drop_duplicates()

    price = sell_prices.merge(item_dept_map, on=["store_id", "item_id"], how="inner")
    price_agg = (
        price.groupby(["store_id", "dept_id", "wm_yr_wk"], as_index=False)
        .agg(
            sell_price_mean=("sell_price", "mean"),
            sell_price_median=("sell_price", "median"),
            sell_price_std=("sell_price", "std"),
            items_in_price_group=("item_id", "nunique"),
        )
    )
    price_agg["sell_price_std"] = price_agg["sell_price_std"].fillna(0.0)
    return price_agg



def _build_macro_features(paths, calendar: pd.DataFrame) -> pd.DataFrame:
    features = read_walmart_features(paths)
    calendar_map = calendar[["date", "wm_yr_wk"]].drop_duplicates()

    macro = features.merge(calendar_map, on="date", how="inner")
    if macro.empty:
        logger.warning("Walmart features did not overlap with calendar date mapping.")
        return pd.DataFrame(columns=["wm_yr_wk", "temperature", "fuel_price", "cpi", "unemployment", "is_holiday_macro"])

    macro_agg = (
        macro.groupby("wm_yr_wk", as_index=False)
        .agg(
            temperature=("Temperature", "mean"),
            fuel_price=("Fuel_Price", "mean"),
            cpi=("CPI", "mean"),
            unemployment=("Unemployment", "mean"),
            is_holiday_macro=("IsHoliday", lambda s: int((s.astype(str).str.upper() == "TRUE").any())),
        )
    )
    return macro_agg



def build_base_dataset(config_path: str = "config/config.yaml") -> pd.DataFrame:
    config = load_config(config_path)
    paths = build_paths(config)

    window_cfg = config["window"]

    calendar = read_calendar(paths)
    overlap_calendar = _extract_overlap_calendar(
        calendar,
        start_date=window_cfg["start_date"],
        end_date=window_cfg["end_date"],
    )
    day_cols = overlap_calendar["d"].tolist()

    sales_subset = read_sales_subset(paths, day_cols=day_cols)
    sales_meta = read_sales_metadata(paths)

    logger.info("Aggregating to department-store level in wide format")
    wide_agg = (
        sales_subset.groupby(["store_id", "state_id", "dept_id"], as_index=False)[day_cols]
        .sum()
    )

    logger.info("Melting department-store wide matrix to long daily format")
    long_df = wide_agg.melt(
        id_vars=["store_id", "state_id", "dept_id"],
        value_vars=day_cols,
        var_name="d",
        value_name="sales",
    )

    keep_calendar_cols = [
        "d",
        "date",
        "wm_yr_wk",
        "weekday",
        "wday",
        "month",
        "year",
        "event_name_1",
        "event_type_1",
        "event_name_2",
        "event_type_2",
        "snap_CA",
        "snap_TX",
        "snap_WI",
    ]
    long_df = long_df.merge(overlap_calendar[keep_calendar_cols], on="d", how="left")

    price_agg = _build_price_features(paths, sales_meta)
    long_df = long_df.merge(price_agg, on=["store_id", "dept_id", "wm_yr_wk"], how="left")

    macro_agg = _build_macro_features(paths, overlap_calendar)
    long_df = long_df.merge(macro_agg, on="wm_yr_wk", how="left")

    long_df["has_event"] = (
        long_df["event_name_1"].notna() | long_df["event_name_2"].notna()
    ).astype("int8")

    long_df["snap_flag"] = np.select(
        [
            long_df["state_id"] == "CA",
            long_df["state_id"] == "TX",
            long_df["state_id"] == "WI",
        ],
        [long_df["snap_CA"], long_df["snap_TX"], long_df["snap_WI"]],
        default=0,
    ).astype("int8")

    long_df = long_df.sort_values(["store_id", "dept_id", "date"]).reset_index(drop=True)

    out_path = paths.processed_dir / "dept_store_daily_base.parquet"
    long_df.to_parquet(out_path, index=False)
    logger.info("Saved base dataset to %s with shape=%s", out_path, long_df.shape)
    return long_df



def run(config_path: str = "config/config.yaml") -> None:
    build_base_dataset(config_path=config_path)


if __name__ == "__main__":
    run()
