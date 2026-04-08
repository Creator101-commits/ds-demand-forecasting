from __future__ import annotations

import numpy as np
import pandas as pd
from pmdarima import auto_arima

from src.utils.config import load_config
from src.utils.logger import get_logger
from src.utils.paths import build_paths

logger = get_logger(__name__)



def _fallback_forecast(train_series: pd.Series, periods: int) -> np.ndarray:
    last = float(train_series.iloc[-1]) if len(train_series) else 0.0
    return np.full(shape=periods, fill_value=last, dtype=np.float32)



def train_and_predict(config_path: str = "config/config.yaml") -> pd.DataFrame:
    config = load_config(config_path)
    paths = build_paths(config)

    data_path = paths.processed_dir / "modeling_with_splits.parquet"
    df = pd.read_parquet(data_path)
    df["date"] = pd.to_datetime(df["date"])

    seasonal_period = int(config["models"]["sarima"]["seasonal_period"])
    stepwise = bool(config["models"]["sarima"]["stepwise"])
    sarima_cfg = config["models"]["sarima"]

    results: list[pd.DataFrame] = []

    series_groups = list(df.groupby(["store_id", "dept_id"], sort=False))
    total_series = len(series_groups)

    for idx, ((store_id, dept_id), g) in enumerate(series_groups, start=1):
        logger.info("SARIMA series %s/%s: (%s, %s)", idx, total_series, store_id, dept_id)
        g = g.sort_values("date")
        train = g[g["split"] == "train"]
        val = g[g["split"] == "val"]
        test = g[g["split"] == "test"]

        if train.empty or val.empty or test.empty:
            continue

        train_y = train["sales"].astype(float)

        try:
            model = auto_arima(
                train_y,
                seasonal=True,
                m=seasonal_period,
                stepwise=stepwise,
                start_p=0,
                start_q=0,
                start_P=0,
                start_Q=0,
                max_p=int(sarima_cfg.get("max_p", 2)),
                max_q=int(sarima_cfg.get("max_q", 2)),
                max_P=int(sarima_cfg.get("max_P", 1)),
                max_Q=int(sarima_cfg.get("max_Q", 1)),
                max_d=int(sarima_cfg.get("max_d", 1)),
                max_D=int(sarima_cfg.get("max_D", 1)),
                maxiter=int(sarima_cfg.get("maxiter", 25)),
                n_fits=int(sarima_cfg.get("n_fits", 20)),
                suppress_warnings=True,
                error_action="ignore",
            )
            val_pred = model.predict(n_periods=len(val))
        except Exception as exc:
            logger.warning("SARIMA val fit failed for (%s, %s): %s", store_id, dept_id, exc)
            val_pred = _fallback_forecast(train_y, len(val))

        train_val_y = pd.concat([train["sales"], val["sales"]], ignore_index=True).astype(float)
        try:
            model_tv = auto_arima(
                train_val_y,
                seasonal=True,
                m=seasonal_period,
                stepwise=stepwise,
                start_p=0,
                start_q=0,
                start_P=0,
                start_Q=0,
                max_p=int(sarima_cfg.get("max_p", 2)),
                max_q=int(sarima_cfg.get("max_q", 2)),
                max_P=int(sarima_cfg.get("max_P", 1)),
                max_Q=int(sarima_cfg.get("max_Q", 1)),
                max_d=int(sarima_cfg.get("max_d", 1)),
                max_D=int(sarima_cfg.get("max_D", 1)),
                maxiter=int(sarima_cfg.get("maxiter", 25)),
                n_fits=int(sarima_cfg.get("n_fits", 20)),
                suppress_warnings=True,
                error_action="ignore",
            )
            test_pred = model_tv.predict(n_periods=len(test))
        except Exception as exc:
            logger.warning("SARIMA test fit failed for (%s, %s): %s", store_id, dept_id, exc)
            test_pred = _fallback_forecast(train_val_y, len(test))

        val_out = val[["store_id", "dept_id", "date", "sales", "split"]].copy()
        val_out["prediction"] = val_pred
        val_out["model"] = "sarima"

        test_out = test[["store_id", "dept_id", "date", "sales", "split"]].copy()
        test_out["prediction"] = test_pred
        test_out["model"] = "sarima"

        results.extend([val_out, test_out])

    pred_df = pd.concat(results, ignore_index=True) if results else pd.DataFrame()
    out_path = paths.prediction_dir / "sarima_predictions.parquet"
    pred_df.to_parquet(out_path, index=False)
    logger.info("Saved SARIMA predictions: %s", out_path)
    return pred_df


if __name__ == "__main__":
    train_and_predict()
