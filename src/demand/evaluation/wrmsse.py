"""Official M5 WRMSSE plus RMSSE/MAE/bias helpers.

WRMSSE follows the competition definition exactly:

- 12 aggregation levels over the item-store panel (42,840 series total):
  total (L1); state (L2); store (L3); category (L4); department (L5);
  state x category (L6); state x department (L7); store x category (L8);
  store x department (L9); item (L10); state x item (L11);
  item x store (L12).
- RMSSE_i = sqrt( mean_{h}(y - yhat)^2 / scale_i ) where scale_i is the
  mean squared one-step naive error of the series computed on the
  in-sample (pre-origin) data, restricted to periods after the first
  non-zero demand. Aggregation happens before scaling, so each level's
  series gets its own scale.
- Weight w_i is the series' cumulative dollar sales over the last 28
  days ending at the forecast origin (units x weekly sell price,
  float64 from the CSV prices). Each level's weights are normalized
  within the level and the 12 level scores are averaged equally; this
  equals global normalization only because every level partitions the
  same total dollar sales.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from demand.data.m5 import HORIZON, WEIGHT_WINDOW, M5Data, series_dollar_sales_window

EPS = 1e-8

# (level_name, official Level_id, grouping columns in ids, weight-file
# key columns producing Agg_Level_1/Agg_Level_2; None means literal
# "X" except for "total" which emits ("Total", "X")).
LEVEL_SPECS: list[tuple[str, str, tuple[str, ...], tuple[str | None, str | None]]] = [
    ("total", "Level1", (), (None, None)),
    ("state", "Level2", ("state_id",), ("state_id", None)),
    ("store", "Level3", ("store_id",), ("store_id", None)),
    ("category", "Level4", ("cat_id",), ("cat_id", None)),
    ("department", "Level5", ("dept_id",), ("dept_id", None)),
    ("state_x_category", "Level6", ("state_id", "cat_id"), ("state_id", "cat_id")),
    ("state_x_department", "Level7", ("state_id", "dept_id"), ("state_id", "dept_id")),
    ("store_x_category", "Level8", ("store_id", "cat_id"), ("store_id", "cat_id")),
    ("store_x_department", "Level9", ("store_id", "dept_id"), ("store_id", "dept_id")),
    ("item", "Level10", ("item_id",), ("item_id", None)),
    # Official weight files order Level11 as (state, item) and Level12
    # as (item, store); the groupings themselves are unchanged.
    ("item_x_state", "Level11", ("item_id", "state_id"), ("state_id", "item_id")),
    ("item_x_store", "Level12", ("id",), ("item_id", "store_id")),
]

LEVEL_GROUPS: list[tuple[str, tuple[str, ...]]] = [
    (name, keys) for name, _, keys, _ in LEVEL_SPECS
]
LEVEL_IDS = {name: lid for name, lid, _, _ in LEVEL_SPECS}
WEIGHT_KEY_COLS = {name: wkey for name, _, _, wkey in LEVEL_SPECS}
N_BOOT_DEFAULT = 500


def first_nonzero_idx(series: np.ndarray) -> int:
    """Index of the first non-zero demand, or len(series) if all zero."""
    nz = np.flatnonzero(series != 0)
    return int(nz[0]) if nz.size else len(series)


def series_scale(train: np.ndarray) -> np.ndarray:
    """Per-series RMSSE scale from in-sample one-step naive errors.

    ``train`` is (n_series, n_days). For each series, diffs are taken only
    over the active period starting at the first non-zero demand.
    """
    s = np.asarray(train, dtype=np.float64)
    n, t = s.shape
    if t < 2:
        return np.full(n, np.nan)
    nz = s != 0
    has = nz.any(axis=1)
    fnz = np.where(has, np.argmax(nz, axis=1), t)
    d = np.diff(s, axis=1)
    idx = np.arange(t - 1)[None, :]
    mask = idx >= fnz[:, None]
    counts = mask.sum(axis=1)
    ss = (d * d * mask).sum(axis=1)
    return np.where(counts > 0, ss / np.maximum(counts, 1), np.nan)


def rmsse_values(
    y_true: np.ndarray, y_pred: np.ndarray, scales: np.ndarray
) -> np.ndarray:
    """Per-series RMSSE for horizon arrays (n_series, horizon)."""
    se = ((y_true - y_pred) ** 2).mean(axis=1)
    scale = np.maximum(scales, EPS)
    return np.asarray(np.sqrt(se / scale))


def _group_indices(ids: pd.DataFrame, keys: tuple[str, ...]) -> list[np.ndarray]:
    """Row-index groups of the panel for one aggregation level."""
    if not keys:
        return [np.arange(len(ids))]
    codes, _ = pd.factorize(ids[list(keys)].astype(str).agg("|".join, axis=1))
    order = np.argsort(codes, kind="stable")
    sorted_codes = codes[order]
    bounds = np.flatnonzero(np.r_[True, sorted_codes[1:] != sorted_codes[:-1], True])
    return [order[bounds[i] : bounds[i + 1]] for i in range(len(bounds) - 1)]


def level_group_map(ids: pd.DataFrame) -> dict[str, list[np.ndarray]]:
    """All 12 levels' member row indices, built once per dataset."""
    return {name: _group_indices(ids, keys) for name, keys in LEVEL_GROUPS}


def unit_weight_keys(
    ids: pd.DataFrame, group_map: dict[str, list[np.ndarray]]
) -> dict[str, list[tuple[str, str]]]:
    """(Agg_Level_1, Agg_Level_2) weight-file keys per level per unit."""
    out: dict[str, list[tuple[str, str]]] = {}
    for name, groups in group_map.items():
        c1, c2 = WEIGHT_KEY_COLS[name]
        keys: list[tuple[str, str]] = []
        for idx in groups:
            row = ids.iloc[int(idx[0])]
            k1 = "Total" if name == "total" else str(row[c1])
            k2 = "X" if c2 is None else str(row[c2])
            keys.append((k1, k2))
        out[name] = keys
    return out


def _is_identity(groups: list[np.ndarray], n: int) -> bool:
    return all(len(g) == 1 for g in groups) and np.array_equal(
        np.concatenate(groups), np.arange(n)
    )


def _aggregate(panel: np.ndarray, groups: list[np.ndarray]) -> np.ndarray:
    """Sum member rows of ``panel`` (n_series, ...) into (n_groups, ...)."""
    if _is_identity(groups, panel.shape[0]):
        return np.asarray(panel, dtype=np.float64)
    out = np.zeros((len(groups), panel.shape[1]), dtype=np.float64)
    for i, idx in enumerate(groups):
        out[i] = panel[idx].sum(axis=0)
    return out


def _aggregate_h(panel: np.ndarray, groups: list[np.ndarray]) -> np.ndarray:
    """Aggregate a (n_series, horizon) array the same way."""
    if _is_identity(groups, panel.shape[0]):
        return np.asarray(panel, dtype=np.float64)
    out = np.zeros((len(groups), panel.shape[1]), dtype=np.float64)
    for i, idx in enumerate(groups):
        out[i] = panel[idx].sum(axis=0)
    return out


@dataclass
class WRMSSEResult:
    """WRMSSE breakdown for one forecast origin."""

    wrmsse: float
    per_level: dict[str, float]
    n_units: int
    unit_rmsse: np.ndarray  # (n_units,) RMSSE per evaluated series
    unit_weight: np.ndarray  # (n_units,) unnormalized dollar-sales weight
    unit_level: np.ndarray  # (n_units,) level index 0..11


def evaluate_wrmsse(
    data: M5Data,
    group_map: dict[str, list[np.ndarray]],
    y_pred: np.ndarray,
    *,
    origin_idx: int,
    y_true: np.ndarray | None = None,
    horizon: int = HORIZON,
    weight_days: int = WEIGHT_WINDOW,
) -> WRMSSEResult:
    """Compute official WRMSSE for forecasts at one origin.

    ``origin_idx`` is the 0-based index of the last observed day (the
    forecast origin). In-sample scales use panel days ``0..origin_idx``;
    weights use dollar sales over the ``weight_days`` ending at the
    origin. ``y_pred`` and ``y_true`` are (n_series, horizon); when
    ``y_true`` is None it defaults to the panel slice after the origin,
    which only exists for origins inside the panel (test truth for the
    private window must be passed explicitly).
    """
    panel = data.panel
    if y_true is None:
        y_true = panel[:, origin_idx + 1 : origin_idx + 1 + horizon]
    if y_true.shape != (data.n_series, horizon):
        raise ValueError(
            f"y_true shape {y_true.shape} != ({data.n_series}, {horizon})"
        )
    if y_pred.shape != (data.n_series, horizon):
        raise ValueError(
            f"y_pred shape {y_pred.shape} != ({data.n_series}, {horizon})"
        )
    train_panel = panel[:, : origin_idx + 1]
    w0 = max(0, origin_idx + 1 - weight_days)
    w1 = origin_idx + 1  # exclusive; window ends at the origin
    dollar = series_dollar_sales_window(data, w0, w1 - 1)

    unit_rmsse_parts: list[np.ndarray] = []
    unit_weight_parts: list[np.ndarray] = []
    unit_level_parts: list[np.ndarray] = []
    per_level: dict[str, float] = {}

    for level_i, (name, groups) in enumerate(group_map.items()):
        tr = _aggregate(train_panel, groups)
        scales = series_scale(tr)
        yt = _aggregate_h(np.asarray(y_true, dtype=np.float64), groups)
        yp = _aggregate_h(np.asarray(y_pred, dtype=np.float64), groups)
        rmsse = rmsse_values(yt, yp, scales)
        # Dollar-sales weight of an aggregate = sum of member series weights.
        w = np.array([dollar[idx].sum() for idx in groups])
        valid = w > 0
        rmsse_safe = np.where(valid, rmsse, 0.0)
        contrib = w * rmsse_safe
        denom = w.sum()
        per_level[name] = float(contrib.sum() / denom) if denom > 0 else 0.0
        unit_rmsse_parts.append(rmsse)
        unit_weight_parts.append(w)
        unit_level_parts.append(np.full(len(groups), level_i, dtype=np.int8))

    all_r = np.concatenate(unit_rmsse_parts)
    all_w = np.concatenate(unit_weight_parts)
    all_l = np.concatenate(unit_level_parts)
    wrmsse = float(np.mean(list(per_level.values())))
    return WRMSSEResult(
        wrmsse=wrmsse,
        per_level=per_level,
        n_units=int(all_r.size),
        unit_rmsse=all_r,
        unit_weight=all_w,
        unit_level=all_l,
    )


def point_metrics(
    y_true: np.ndarray, y_pred: np.ndarray
) -> dict[str, float]:
    """MAE, RMSE, and bias (mean error) over the horizon."""
    err = y_pred - y_true
    return {
        "mae": float(np.abs(err).mean()),
        "rmse": float(np.sqrt((err**2).mean())),
        "bias": float(err.mean()),
    }


def rmsse_unweighted(
    train_series: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray
) -> float:
    """Plain RMSSE for one series (aggregate-level comparisons)."""
    s = train_series.astype(np.float64)
    fnz = first_nonzero_idx(s)
    active = s[fnz:]
    scale = float(np.mean(np.diff(active) ** 2)) if active.size >= 2 else np.nan
    scale = max(scale, EPS) if np.isfinite(scale) else EPS
    return float(np.sqrt(((y_true - y_pred) ** 2).mean() / scale))
