from __future__ import annotations

import pandas as pd



def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    out["day_of_week"] = out["date"].dt.dayofweek.astype("int8")
    out["day_of_month"] = out["date"].dt.day.astype("int8")
    out["week_of_year"] = out["date"].dt.isocalendar().week.astype("int16")
    out["month"] = out["date"].dt.month.astype("int8")
    out["quarter"] = out["date"].dt.quarter.astype("int8")
    out["year"] = out["date"].dt.year.astype("int16")
    out["is_weekend"] = (out["day_of_week"] >= 5).astype("int8")
    out["is_month_start"] = out["date"].dt.is_month_start.astype("int8")
    out["is_month_end"] = out["date"].dt.is_month_end.astype("int8")
    return out
