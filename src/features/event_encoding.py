from __future__ import annotations

import pandas as pd



def add_event_features(df: pd.DataFrame, max_event_dummies: int = 8) -> pd.DataFrame:
    out = df.copy()
    out["event_name_1"] = out["event_name_1"].fillna("None")
    out["event_name_2"] = out["event_name_2"].fillna("None")
    out["has_event"] = ((out["event_name_1"] != "None") | (out["event_name_2"] != "None")).astype("int8")

    top_events_1 = out.loc[out["event_name_1"] != "None", "event_name_1"].value_counts().head(max_event_dummies).index
    top_events_2 = out.loc[out["event_name_2"] != "None", "event_name_2"].value_counts().head(max_event_dummies).index

    out["event_name_1"] = out["event_name_1"].where(out["event_name_1"].isin(top_events_1), "Other")
    out["event_name_2"] = out["event_name_2"].where(out["event_name_2"].isin(top_events_2), "Other")

    d1 = pd.get_dummies(out["event_name_1"], prefix="event1", dtype="int8")
    d2 = pd.get_dummies(out["event_name_2"], prefix="event2", dtype="int8")
    out = pd.concat([out, d1, d2], axis=1)
    return out
