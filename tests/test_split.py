from __future__ import annotations

import pandas as pd

from src.data.split import create_temporal_splits



def test_temporal_split_sizes() -> None:
    dates = pd.date_range("2024-01-01", periods=120, freq="D")
    df = pd.DataFrame(
        {
            "store_id": ["CA_1"] * len(dates),
            "dept_id": ["FOODS_1"] * len(dates),
            "date": dates,
            "sales": 1.0,
        }
    )

    out = create_temporal_splits(df, val_days=28, test_days=28)
    day_counts = out.groupby("split")["date"].nunique().to_dict()

    assert day_counts["val"] == 28
    assert day_counts["test"] == 28
    assert day_counts["train"] == 64
