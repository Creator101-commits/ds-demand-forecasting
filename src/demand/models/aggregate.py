"""Aggregate-level models: SARIMA and Prophet on level 1-3 series."""

from __future__ import annotations

import numpy as np
import pandas as pd

AGG_LEVELS = ("total", "state", "store")  # level 1..3 series only


def aggregate_panel(
    panel: np.ndarray, ids: pd.DataFrame, level: str
) -> tuple[np.ndarray, list[str]]:
    """Sum the item-store panel into one aggregate level's series."""
    if level == "total":
        return panel.sum(axis=0, keepdims=True), ["total"]
    col = {"state": "state_id", "store": "store_id"}[level]
    groups = ids[col].astype(str)
    codes, uniques = pd.factorize(groups, sort=True)
    out = np.zeros((len(uniques), panel.shape[1]), dtype=np.float64)
    for i in range(len(uniques)):
        out[i] = panel[codes == i].sum(axis=0)
    return out.astype(np.float32), [str(u) for u in uniques]


def sarima_forecast(train_series: np.ndarray, horizon: int) -> np.ndarray:
    """SARIMAX(0,1,1)x(0,1,1,7) forecast; airline-style weekly model."""
    import statsmodels.api as sm  # noqa: PLC0415

    s = pd.Series(train_series.astype(float))
    model = sm.tsa.SARIMAX(
        s,
        order=(0, 1, 1),
        seasonal_order=(0, 1, 1, 7),
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    fit = model.fit(disp=False, maxiter=200)
    fc = fit.forecast(steps=horizon)
    return np.asarray(np.clip(np.asarray(fc, dtype=np.float64), 0.0, None))


def prophet_forecast(train_series: np.ndarray, horizon: int) -> np.ndarray:
    """Prophet daily forecast with weekly + yearly seasonality."""
    from prophet import Prophet  # noqa: PLC0415

    n = len(train_series)
    df = pd.DataFrame(
        {
            "ds": pd.date_range("2011-01-01", periods=n, freq="D"),
            "y": train_series.astype(float),
        }
    )
    m = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
    )
    m.fit(df)
    future = m.make_future_dataframe(periods=horizon, freq="D")
    fc = m.predict(future)
    return np.asarray(
        np.clip(fc["yhat"].to_numpy()[-horizon:], 0.0, None), dtype=np.float64
    )


def aggregate_forecasts(
    level: str,
    panel: np.ndarray,
    ids: pd.DataFrame,
    train_end_idx: int,
    horizon: int,
    method: str,
) -> tuple[np.ndarray, list[str]]:
    """Fit one aggregate model per series of a level and forecast."""
    agg, names = aggregate_panel(panel[:, : train_end_idx + 1], ids, level)
    fn = sarima_forecast if method == "sarima" else prophet_forecast
    preds = np.stack([fn(s, horizon) for s in agg], axis=0)
    return preds.astype(np.float32), names
