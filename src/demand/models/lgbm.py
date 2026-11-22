"""Global LightGBM demand model with a direct 28-day horizon (v2).

Rows are (series, origin, horizon_step) pooled samples. Every sales
feature uses only panel data at or before the origin ``o``; exogenous
features (price, calendar, SNAP) are known in advance from the M5
calendar and sell prices, which cover the private window. A
perturbation test proves features at ``o`` are identical after
overwriting ``panel[:, o+1:]``.

Feature layout (33 columns):
- 0-6:   y[o], y[o-1], ..., y[o-6]
- 7-11:  rolling means over 7/14/28/56/112 days ending at o
- 12:    rolling std over 28 days ending at o
- 13:    zero fraction over the last 28 days
- 14:    days since last non-zero sale, capped at 365
- 15-19: static codes item, dept, cat, store, state (categorical)
- 20:    last observed same-weekday value for the target day
- 21:    mean of the 4 most recent observed same-weekday values
- 22:    sell price at the target day
- 23:    sell price at the origin
- 24:    price_t / price_o
- 25:    price_t / trailing max weekly price up to week(t)
- 26-30: target wday, month, event_name_1, event_type_1, event_name_2
         (categorical)
- 31:    SNAP flag for the series' own state at the target day
- 32:    horizon step h
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from demand.data.m5 import M5Data

SEED = 42
N_ORIGINS = 40
MIN_ORIGIN = 120  # longest lookback is 112 days plus margin
MAX_LOOKBACK = 112
GRID = [
    {"num_leaves": 63, "learning_rate": 0.08},
    {"num_leaves": 63, "learning_rate": 0.03},
    {"num_leaves": 255, "learning_rate": 0.08},
    {"num_leaves": 255, "learning_rate": 0.03},
]
MAX_ROUNDS = 1500
EARLY_STOP = 50
ROLL_WINDOWS = (7, 14, 28, 56, 112)
STATIC_COLS = ("item_id", "dept_id", "cat_id", "store_id", "state_id")
SNAP_STATES = ("CA", "TX", "WI")
EPS_PRICE = 1e-6

FEATURE_NAMES = [
    "lag_0", "lag_1", "lag_2", "lag_3", "lag_4", "lag_5", "lag_6",
    "rollmean_7", "rollmean_14", "rollmean_28", "rollmean_56", "rollmean_112",
    "rollstd_28", "zero_frac_28", "days_since_nonzero",
    "static_item", "static_dept", "static_cat", "static_store", "static_state",
    "same_wd_last", "same_wd_mean4",
    "price_t", "price_o", "price_ratio", "price_rel_max",
    "wday", "month", "event_name_1", "event_type_1", "event_name_2",
    "snap_state", "h",
]
CATEGORICAL_IDX = [15, 16, 17, 18, 19, 26, 27, 28, 29, 30]
N_FEATURES = len(FEATURE_NAMES)


def sample_origins(
    min_origin: int, max_origin: int, n: int = N_ORIGINS
) -> np.ndarray:
    """Evenly spaced origin indices within [min_origin, max_origin]."""
    lo = max(MIN_ORIGIN, min_origin)
    hi = max_origin
    if hi <= lo:
        return np.array([], dtype=int)
    return np.unique(np.linspace(lo, hi, n).round().astype(int))


@dataclass
class FeatureContext:
    """Precomputed per-dataset feature inputs (built once per run)."""

    wday: np.ndarray  # (n_cal_days,) int32
    month: np.ndarray
    event1: np.ndarray  # factorized codes
    event1_type: np.ndarray
    event2: np.ndarray
    snap: np.ndarray  # (n_cal_days, 3) float32 CA/TX/WI
    snap_col_of_series: np.ndarray  # (n_series,) int64
    week_pos_day: np.ndarray  # (n_cal_days,) int32
    trail_max: np.ndarray  # (n_series, n_weeks) float32 expanding max
    static: np.ndarray  # (n_series, 5) int32
    last_nz: np.ndarray  # (n_series, panel_days) int32, -1 if never
    n_panel_days: int


def build_feature_context(data: M5Data) -> FeatureContext:
    """Build the shared feature context for one loaded dataset."""
    cal = data.calendar
    snap_cols = [f"snap_{s}" for s in SNAP_STATES]
    snap = cal[snap_cols].to_numpy(dtype=np.float32)
    state_to_col = {s: i for i, s in enumerate(SNAP_STATES)}
    snap_col = data.ids["state_id"].map(state_to_col).fillna(-1).to_numpy()
    if (snap_col < 0).any():
        raise ValueError("series state_id outside CA/TX/WI")

    static = np.empty((data.n_series, len(STATIC_COLS)), dtype=np.int32)
    for j, c in enumerate(STATIC_COLS):
        static[:, j] = pd.factorize(data.ids[c])[0].astype(np.int32)

    trail_max = np.maximum.accumulate(
        data.price_weekly.astype(np.float32), axis=1
    )

    panel = data.panel
    n_panel = panel.shape[1]
    cols = np.arange(n_panel, dtype=np.int32)[None, :]
    nz_idx = np.where(panel > 0, cols, np.int32(-1))
    last_nz = np.maximum.accumulate(nz_idx, axis=1).astype(np.int32)

    return FeatureContext(
        wday=cal["wday"].to_numpy(dtype=np.int32),
        month=cal["month"].to_numpy(dtype=np.int32),
        event1=pd.factorize(cal["event_name_1"].fillna(""))[0].astype(np.int32),
        event1_type=pd.factorize(cal["event_type_1"].fillna(""))[0].astype(np.int32),
        event2=pd.factorize(cal["event_name_2"].fillna(""))[0].astype(np.int32),
        snap=snap,
        snap_col_of_series=snap_col.astype(np.int64),
        week_pos_day=data.day_week_pos.astype(np.int32),
        trail_max=np.nan_to_num(trail_max, nan=0.0),
        static=static,
        last_nz=last_nz,
        n_panel_days=n_panel,
    )


def _same_wd_back(h: int) -> int:
    """Days back from origin to the last observed same-weekday day."""
    return (7 - h) % 7


def build_xy(
    panel: np.ndarray,
    ctx: FeatureContext,
    data: M5Data,
    origins: np.ndarray,
    horizon: int,
    *,
    with_labels: bool = True,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Pooled (series, origin, h) rows -> (X float32, y float32|None).

    Only ``panel[:, <= origin]`` is read for sales features; target
    labels are read only when ``with_labels`` and the target day lies
    inside the panel. ``data`` supplies ``price_days`` for target days
    that may lie beyond the panel (private window).
    """
    n_series = panel.shape[0]
    n_rows = len(origins) * horizon * n_series
    X = np.empty((n_rows, N_FEATURES), dtype=np.float32)
    y = np.empty(n_rows, dtype=np.float32) if with_labels else None
    price_days = data.price_days
    row = 0
    for o in origins:
        o = int(o)
        base = np.empty((n_series, 20), dtype=np.float32)
        for k in range(7):
            base[:, k] = panel[:, o - k]
        for j, w in enumerate(ROLL_WINDOWS):
            base[:, 7 + j] = panel[:, o - w + 1 : o + 1].mean(axis=1)
        base[:, 12] = panel[:, o - 27 : o + 1].std(axis=1)
        base[:, 13] = (panel[:, o - 27 : o + 1] == 0).mean(axis=1)
        last = ctx.last_nz[:, o]
        # last == -1 means never sold; o - (-1) exceeds the cap anyway.
        base[:, 14] = np.minimum(o - last, 365)
        base[:, 15:20] = ctx.static

        price_o = price_days[:, o]
        for h in range(1, horizon + 1):
            t = o + h
            block = X[row : row + n_series]
            block[:, :20] = base
            sw = o - _same_wd_back(h)
            block[:, 20] = panel[:, sw]
            block[:, 21] = panel[:, sw - 21 : sw + 1 : 7].mean(axis=1)
            price_t = price_days[:, t]
            block[:, 22] = price_t
            block[:, 23] = price_o
            block[:, 24] = price_t / np.maximum(price_o, EPS_PRICE)
            tmax = ctx.trail_max[:, ctx.week_pos_day[t]]
            block[:, 25] = price_t / np.maximum(tmax, EPS_PRICE)
            block[:, 26] = ctx.wday[t]
            block[:, 27] = ctx.month[t]
            block[:, 28] = ctx.event1[t]
            block[:, 29] = ctx.event1_type[t]
            block[:, 30] = ctx.event2[t]
            block[:, 31] = ctx.snap[t][ctx.snap_col_of_series]
            block[:, 32] = h
            if y is not None:
                if t >= panel.shape[1]:
                    raise ValueError(
                        "target day beyond panel; build with_labels=False"
                    )
                y[row : row + n_series] = panel[:, t]
            row += n_series
    return X, y


def predict_origin(
    model: Any,
    panel: np.ndarray,
    ctx: FeatureContext,
    data: M5Data,
    origin_idx: int,
    horizon: int,
) -> np.ndarray:
    """Forecast all series for days origin_idx+1 .. +horizon."""
    X, _ = build_xy(
        panel, ctx, data, np.array([origin_idx]), horizon, with_labels=False
    )
    raw = model.predict(X)
    n_series = panel.shape[0]
    pred = np.asarray(raw).reshape(horizon, n_series).T
    return np.asarray(np.clip(pred, 0.0, None), dtype=np.float32)


def lgbm_params(config: dict[str, Any], seed: int) -> dict[str, Any]:
    """Shared LightGBM parameters for train and refit."""
    return {
        "objective": "tweedie",
        "tweedie_variance_power": 1.5,
        "metric": "l2",
        "seed": seed,
        "num_threads": -1,
        "verbosity": -1,
        "max_bin": 255,
        "deterministic": True,
        **config,
    }


def _dataset(
    X: np.ndarray, y: np.ndarray | None, reference: Any = None
) -> Any:
    import lightgbm as lgb  # noqa: PLC0415

    cat_names = [FEATURE_NAMES[i] for i in CATEGORICAL_IDX]
    return lgb.Dataset(
        X,
        label=y,
        feature_name=FEATURE_NAMES,
        categorical_feature=cat_names,
        reference=reference,
    )


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

    dtrain = _dataset(X_train, y_train)
    dval = _dataset(X_val, y_val, reference=dtrain)
    model = lgb.train(
        lgbm_params(config, seed),
        dtrain,
        num_boost_round=MAX_ROUNDS,
        valid_sets=[dval],
        callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False)],
    )
    best = int(model.best_iteration or MAX_ROUNDS)
    return model, best


def cap_bound(
    best_iteration: int,
    max_rounds: int = MAX_ROUNDS,
    early_stop: int = EARLY_STOP,
) -> bool:
    """True when the round cap binds: best iter is within early_stop of it.

    Early stopping fires after ``early_stop`` rounds without improvement,
    so a best iteration in the last ``early_stop`` rounds means the run
    ended at ``num_boost_round`` rather than on a validation plateau.
    """
    return best_iteration >= max_rounds - early_stop


def refit_fixed_iterations(
    X: np.ndarray,
    y: np.ndarray,
    config: dict[str, Any],
    num_boost_round: int,
    seed: int,
) -> Any:
    """Refit on train+val origins with the validation-selected iteration."""
    import lightgbm as lgb  # noqa: PLC0415

    dtrain = _dataset(X, y)
    return lgb.train(
        lgbm_params(config, seed), dtrain, num_boost_round=num_boost_round
    )
