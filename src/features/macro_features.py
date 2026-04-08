from __future__ import annotations

import pandas as pd



def add_macro_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    for col in ["temperature", "fuel_price", "cpi", "unemployment"]:
        if col not in out.columns:
            out[col] = pd.NA

    weekly_macro = (
        out[["wm_yr_wk", "temperature", "fuel_price", "cpi", "unemployment"]]
        .drop_duplicates("wm_yr_wk")
        .sort_values("wm_yr_wk")
        .reset_index(drop=True)
    )

    for col in ["temperature", "fuel_price", "cpi", "unemployment"]:
        weekly_macro[f"{col}_lag_4w"] = weekly_macro[col].shift(4)
        weekly_macro[f"{col}_change_4w"] = weekly_macro[col] - weekly_macro[f"{col}_lag_4w"]

    merge_cols = [
        "wm_yr_wk",
        "temperature_lag_4w",
        "temperature_change_4w",
        "fuel_price_lag_4w",
        "fuel_price_change_4w",
        "cpi_lag_4w",
        "cpi_change_4w",
        "unemployment_lag_4w",
        "unemployment_change_4w",
    ]
    out = out.merge(weekly_macro[merge_cols], on="wm_yr_wk", how="left")
    return out
