"""Global LightGBM demand model with a direct 28-day horizon.

Rows are (series, origin, horizon_step) pooled samples built as numpy
arrays. Features use only data at least 28 days stale relative to the
origin plus calendar/price attributes of the target day, so nothing leaks
past the forecast origin. Tweedie objective suits intermittent counts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

SEED = 42
LAGS = (28, 35, 56)
ROLL_WINDOW = 28
N_ORIGINS = 40
GRID = [
    {"num_leaves": 63, "learning_rate": 0.08},
    {"num_leaves": 63, "learning_rate": 0.03},
    {"num_leaves": 255, "learning_rate": 0.08},
    {"num_leaves": 255, "learning_rate": 0.03},
]
MAX_ROUNDS = 400
EARLY_STOP = 50
STATIC_COLS = ("store_id", "item_id", "dept_id", "cat_id", "state_id")
NUM_BASE_FEATURES = len(LAGS) + 2 + 1 + len(STATIC_COLS)  # lags+roll+price+static


def sample_origins(train_start: int, train_end: int, n: int = N_ORIGINS) -> np.ndarray:
    """Evenly spaced origin indices with enough history for lags."""
    lo = max(60, train_start)
    hi = train_end - 28  # need 28 horizon days inside the window
    if hi <= lo:
        return np.array([], dtype=int)
    return np.unique(np.linspace(lo, hi, n).round().astype(int))


def _static_codes(ids: pd.DataFrame) -> np.ndarray:
    """(n_series, 5) int32 codes for the static categorical columns."""
    out = np.empty((len(ids), len(STATIC_COLS)), dtype=np.int32)
    for j, c in enumerate(STATIC_COLS):
        out[:, j] = pd.factorize(ids[c])[0].astype(np.int32)
    return out


def _calendar_arrays(cal: pd.DataFrame) -> dict[str, np.ndarray]:
    snap_cols = [c for c in cal.columns if c.startswith("snap_")]
    return {
        "wday": cal["wday"].to_numpy(dtype=np.int32),
        "month": cal["month"].to_numpy(dtype=np.int32),
        "week": pd.factorize(cal["wm_yr_wk"])[0].astype(np.int32),
        "event1": pd.factorize(cal["event_name_1"].fillna(""))[0].astype(np.int32),
        "event2": pd.factorize(cal["event_name_2"].fillna(""))[0].astype(np.int32),
        "snap": (
            cal[snap_cols].to_numpy(dtype=np.float32)
            if snap_cols
            else np.zeros((len(cal), 0), dtype=np.float32)
        ),
        "n_snap": len(snap_cols),
    }


def build_xy(
    panel: np.ndarray,
    price_days: np.ndarray,
    cal: pd.DataFrame,
    ids: pd.DataFrame,
    origins: np.ndarray,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Pooled (series, origin, h) rows -> (X float32, y float32).

    X columns: lag_28/35/56, roll_mean_28, roll_std_28, price, five static
    codes, h, target wday/month/week/event1/event2, snap flags.
    Target day for row (series, o, h) is o + h.
    """
    n_series = panel.shape[0]
    calx = _calendar_arrays(cal)
    static = _static_codes(ids)
    snap_n = calx["n_snap"]
    n_feat = len(LAGS) + 2 + 1 + len(STATIC_COLS) + 1 + 5 + snap_n
    n_rows = len(origins) * horizon * n_series
    X = np.empty((n_rows, n_feat), dtype=np.float32)
    y = np.empty(n_rows, dtype=np.float32)

    row = 0
    for o in origins:
        base_feats = np.empty((n_series, NUM_BASE_FEATURES), dtype=np.float32)
        for j, lag in enumerate(LAGS):
            base_feats[:, j] = panel[:, o - lag]
        k = len(LAGS)
        win = panel[:, o - 27 - ROLL_WINDOW : o - 27].astype(np.float64)
        base_feats[:, k] = win.mean(axis=1)
        base_feats[:, k + 1] = win.std(axis=1)
        base_feats[:, k + 2] = price_days[:, o]
        base_feats[:, k + 3 : k + 3 + len(STATIC_COLS)] = static

        for h in range(1, horizon + 1):
            t = o + h
            block = X[row : row + n_series]
            block[:, :NUM_BASE_FEATURES] = base_feats
            block[:, NUM_BASE_FEATURES] = h
            c0 = NUM_BASE_FEATURES + 1
            block[:, c0] = calx["wday"][t]
            block[:, c0 + 1] = calx["month"][t]
            block[:, c0 + 2] = calx["week"][t]
            block[:, c0 + 3] = calx["event1"][t]
            block[:, c0 + 4] = calx["event2"][t]
            if snap_n:
                block[:, c0 + 5 : c0 + 5 + snap_n] = calx["snap"][t]
            y[row : row + n_series] = panel[:, t]
            row += n_series
    return X, y


def predict_origin(
    model: Any,
    panel: np.ndarray,
    price_days: np.ndarray,
    cal: pd.DataFrame,
    ids: pd.DataFrame,
    origin_idx: int,
    horizon: int,
) -> np.ndarray:
    """Forecast all series for days origin_idx+1 .. +horizon -> (n_series, h)."""
    X, _ = build_xy(
        panel, price_days, cal, ids, np.array([origin_idx]), horizon
    )
    raw = model.predict(X)
    n_series = panel.shape[0]
    pred = np.asarray(raw).reshape(horizon, n_series).T
    return np.asarray(np.clip(pred, 0.0, None), dtype=np.float32)


@dataclass
class LGBMResult:
    model: Any
    best_iteration: int
    config: dict[str, Any]
    val_wrmsse: float
    seed: int


def train_config(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    config: dict[str, Any],
    seed: int,
) -> tuple[Any, int]:
    """Fit one grid config with early stopping on the validation rows."""
    import lightgbm as lgb  # noqa: PLC0415

    params = {
        "objective": "tweedie",
        "tweedie_variance_power": 1.5,
        "metric": "l2",
        "seed": seed,
        "num_threads": -1,
        "verbosity": -1,
        "max_bin": 255,
    }
    params.update(config)
    dtrain = lgb.Dataset(X_train, label=y_train)
    dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)
    model = lgb.train(
        params,
        dtrain,
        num_boost_round=MAX_ROUNDS,
        valid_sets=[dval],
        callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False)],
    )
    best = int(model.best_iteration or MAX_ROUNDS)
    return model, best
