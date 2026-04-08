from __future__ import annotations

import pandas as pd



def add_price_features(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    out = out.sort_values(group_cols + ["date"]).reset_index(drop=True)

    price_col = "sell_price_mean"
    if price_col not in out.columns:
        out[price_col] = 0.0

    grouped = out.groupby(group_cols, sort=False)

    out["price_change_pct"] = grouped[price_col].pct_change().fillna(0.0)
    rolling_price_mean = grouped[price_col].transform(lambda s: s.shift(1).expanding().mean())
    out["price_relative_to_mean"] = out[price_col] / rolling_price_mean.replace(0, pd.NA)

    dept_week_price = out.groupby(["dept_id", "wm_yr_wk"], as_index=False)[price_col].mean()
    dept_week_price = dept_week_price.rename(columns={price_col: "dept_week_price_mean"})
    out = out.merge(dept_week_price, on=["dept_id", "wm_yr_wk"], how="left")
    out["price_relative_to_category"] = out[price_col] / out["dept_week_price_mean"].replace(0, pd.NA)

    return out
