from __future__ import annotations

from typing import Any, cast

import pandas as pd

from src.utils.config import load_config
from src.utils.logger import get_logger
from src.utils.paths import build_paths

logger = get_logger(__name__)

try:
    from prophet import Prophet
except Exception as exc:  # pragma: no cover
    Prophet = None
    IMPORT_ERROR = exc



def train_and_predict(config_path: str = "config/config.yaml") -> pd.DataFrame:
    if Prophet is None:
        raise ImportError(f"Prophet import failed: {IMPORT_ERROR}")

    config = load_config(config_path)
    paths = build_paths(config)

    regressors: list[str] = config["models"]["prophet"].get("regressors", [])
    yearly = cast(Any, bool(config["models"]["prophet"]["yearly_seasonality"]))
    weekly = cast(Any, bool(config["models"]["prophet"]["weekly_seasonality"]))
    daily = cast(Any, False)

    data_path = paths.processed_dir / "modeling_with_splits.parquet"
    df = pd.read_parquet(data_path)
    df["date"] = pd.to_datetime(df["date"])

    results: list[pd.DataFrame] = []

    for (store_id, dept_id), g in df.groupby(["store_id", "dept_id"], sort=False):
        g = g.sort_values("date").copy()
        train = g[g["split"] == "train"].copy()
        val = g[g["split"] == "val"].copy()
        test = g[g["split"] == "test"].copy()

        if train.empty or val.empty or test.empty:
            continue

        train_df = train[["date", "sales"] + [c for c in regressors if c in g.columns]].rename(
            columns={"date": "ds", "sales": "y"}
        )

        model = Prophet(
            yearly_seasonality=yearly,
            weekly_seasonality=weekly,
            daily_seasonality=daily,
        )

        active_regs = [c for c in regressors if c in train_df.columns]
        for reg in active_regs:
            model.add_regressor(reg)

        model.fit(train_df)

        val_future = val[["date"] + active_regs].rename(columns={"date": "ds"})
        val_pred = model.predict(val_future)[["ds", "yhat"]].rename(columns={"ds": "date", "yhat": "prediction"})

        train_val_df = pd.concat([train, val], ignore_index=True)
        train_val_df = train_val_df[["date", "sales"] + active_regs].rename(columns={"date": "ds", "sales": "y"})

        model_tv = Prophet(
            yearly_seasonality=yearly,
            weekly_seasonality=weekly,
            daily_seasonality=daily,
        )
        for reg in active_regs:
            model_tv.add_regressor(reg)
        model_tv.fit(train_val_df)

        test_future = test[["date"] + active_regs].rename(columns={"date": "ds"})
        test_pred = model_tv.predict(test_future)[["ds", "yhat"]].rename(columns={"ds": "date", "yhat": "prediction"})

        val_out = val[["store_id", "dept_id", "date", "sales", "split"]].merge(val_pred, on="date", how="left")
        val_out["model"] = "prophet"

        test_out = test[["store_id", "dept_id", "date", "sales", "split"]].merge(test_pred, on="date", how="left")
        test_out["model"] = "prophet"

        results.extend([val_out, test_out])

    pred_df = pd.concat(results, ignore_index=True) if results else pd.DataFrame()
    out_path = paths.prediction_dir / "prophet_predictions.parquet"
    pred_df.to_parquet(out_path, index=False)
    logger.info("Saved Prophet predictions: %s", out_path)
    return pred_df


if __name__ == "__main__":
    train_and_predict()
