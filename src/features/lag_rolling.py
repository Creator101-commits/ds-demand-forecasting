from __future__ import annotations

import pandas as pd



def add_sales_lag_features(
    df: pd.DataFrame,
    group_cols: list[str],
    lag_days: list[int],
    rolling_windows: list[int],
) -> pd.DataFrame:
    out = df.copy()
    out = out.sort_values(group_cols + ["date"]).reset_index(drop=True)

    grouped = out.groupby(group_cols, sort=False)
    for lag in lag_days:
        out[f"lag_{lag}"] = grouped["sales"].shift(lag)

    for win in rolling_windows:
        out[f"rolling_mean_{win}"] = grouped["sales"].transform(
            lambda s: s.shift(1).rolling(win, min_periods=1).mean()
        )
        out[f"rolling_std_{win}"] = grouped["sales"].transform(
            lambda s: s.shift(1).rolling(win, min_periods=2).std()
        )

    if "rolling_mean_7" in out.columns and "rolling_mean_28" in out.columns:
        denom = out["rolling_mean_28"].replace(0, pd.NA)
        out["momentum_ratio_7_28"] = out["rolling_mean_7"] / denom

    out["expanding_mean"] = grouped["sales"].transform(lambda s: s.shift(1).expanding().mean())
    return out
