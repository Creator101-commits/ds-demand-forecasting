"""Official M5 WRMSSE plus RMSSE/MAE/bias helpers.

WRMSSE follows the competition definition exactly:

- 12 aggregation levels over the item-store panel (42,840 series total):
  total; state; store; category; department; state x category;
  state x department; store x category; store x department; item;
  item x state; item x store.
- RMSSE_i = sqrt( mean_{h}(y - yhat)^2 / scale_i ) where scale_i is the
  mean squared one-step naive error of the series computed on the
  in-sample (pre-origin) data, restricted to periods after the first
  non-zero demand. Aggregation happens before scaling, so each level's
  series gets its own scale.
- Weight w_i is the series' cumulative dollar sales over the last 28
  days before the forecast origin (units x weekly sell price); weights
  are normalized globally, so each level contributes proportionally to
  its dollar sales.
- WRMSSE = sum_i w_i * RMSSE_i / sum_i w_i over all 42,840 series.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from demand.data.m5 import M5Data

EPS = 1e-8
N_BOOT_DEFAULT = 1000

# (level_name, grouping columns in ids). Level 12 uses the series id.
LEVEL_GROUPS: list[tuple[str, tuple[str, ...]]] = [
    ("total", ()),
    ("state", ("state_id",)),
    ("store", ("store_id",)),
    ("category", ("cat_id",)),
    ("department", ("dept_id",)),
    ("state_x_category", ("state_id", "cat_id")),
    ("state_x_department", ("state_id", "dept_id")),
    ("store_x_category", ("store_id", "cat_id")),
    ("store_x_department", ("store_id", "dept_id")),
    ("item", ("item_id",)),
    ("item_x_state", ("item_id", "state_id")),
    ("item_x_store", ("id",)),
]


def first_nonzero_idx(series: np.ndarray) -> int:
    """Index of the first non-zero demand, or len(series) if all zero."""
    nz = np.flatnonzero(series != 0)
    return int(nz[0]) if nz.size else len(series)


def series_scale(train: np.ndarray) -> np.ndarray:
    """Per-series RMSSE scale from in-sample one-step naive errors.

    ``train`` is (n_series, n_days). For each series, diffs are taken only
    over the active period starting at the first non-zero demand.
    """
    n, t = train.shape
    scales = np.full(n, np.nan, dtype=np.float64)
    for i in range(n):
        s = train[i]
        fnz = first_nonzero_idx(s)
        active = s[fnz:]
        if active.size < 2:
            continue
        diffs = np.diff(active.astype(np.float64))
        scales[i] = float(np.mean(diffs * diffs))
    return scales


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


def _is_identity(groups: list[np.ndarray], n: int) -> bool:
    return all(len(g) == 1 for g in groups) and np.array_equal(
        np.concatenate(groups), np.arange(n)
    )


def _aggregate(panel: np.ndarray, groups: list[np.ndarray]) -> np.ndarray:
    """Sum member rows of ``panel`` (n_series, ...) into (n_groups, ...)."""
    if _is_identity(groups, panel.shape[0]):
        return panel.astype(np.float64, copy=True)
    out = np.zeros((len(groups), panel.shape[1]), dtype=np.float64)
    for i, idx in enumerate(groups):
        out[i] = panel[idx].sum(axis=0)
    return out


def _aggregate_h(panel: np.ndarray, groups: list[np.ndarray]) -> np.ndarray:
    """Aggregate a (n_series, horizon) array the same way."""
    if _is_identity(groups, panel.shape[0]):
        return panel.astype(np.float64, copy=True)
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
    train_end_idx: int,
    horizon_start_idx: int,
    horizon: int = 28,
    weight_days: int = 28,
) -> WRMSSEResult:
    """Compute official WRMSSE for forecasts at one origin.

    ``y_pred`` is (n_series, horizon) item-store forecasts covering days
    ``horizon_start_idx : horizon_start_idx + horizon``. Scales use days
    ``0..train_end_idx``; weights use dollar sales over the ``weight_days``
    immediately before ``horizon_start_idx``.
    """
    panel = data.panel.astype(np.float64)
    y_true = panel[:, horizon_start_idx : horizon_start_idx + horizon]
    train_panel = panel[:, : train_end_idx + 1]

    w0 = max(0, horizon_start_idx - weight_days)
    w1 = horizon_start_idx  # exclusive
    dollar = (data.panel[:, w0:w1].astype(np.float64) * data.price_days[:, w0:w1]).sum(axis=1)

    unit_rmsse_parts: list[np.ndarray] = []
    unit_weight_parts: list[np.ndarray] = []
    unit_level_parts: list[np.ndarray] = []
    per_level: dict[str, float] = {}

    for level_i, (name, groups) in enumerate(group_map.items()):
        tr = _aggregate(train_panel, groups)
        scales = series_scale(tr)
        yt = _aggregate_h(y_true, groups)
        yp = _aggregate_h(y_pred, groups)
        rmsse = rmsse_values(yt, yp, scales)
        # Dollar-sales weight of an aggregate = sum of member series weights.
        w = np.array([dollar[idx].sum() for idx in groups])
        valid = w > 0
        contrib = np.where(valid, w * rmsse, 0.0)
        denom = w.sum()
        per_level[name] = float(contrib.sum() / denom) if denom > 0 else 0.0
        unit_rmsse_parts.append(rmsse)
        unit_weight_parts.append(w)
        unit_level_parts.append(np.full(len(groups), level_i, dtype=np.int8))

    # Overall WRMSSE: weights normalized globally across all units.
    all_r = np.concatenate(unit_rmsse_parts)
    all_w = np.concatenate(unit_weight_parts)
    all_l = np.concatenate(unit_level_parts)
    total_w = all_w.sum()
    wrmsse = float((all_w * all_r).sum() / total_w) if total_w > 0 else float("nan")
    return WRMSSEResult(
        wrmsse=wrmsse,
        per_level=per_level,
        n_units=int(all_r.size),
        unit_rmsse=all_r,
        unit_weight=all_w,
        unit_level=all_l,
    )


def bootstrap_wrmsse(
    result: WRMSSEResult, n_boot: int = N_BOOT_DEFAULT, seed: int = 0
) -> dict[str, float]:
    """Bootstrap CI for WRMSSE resampling evaluated units i.i.d.

    Units are the 42,840 (level, series) entries; resampling ignores the
    hierarchy's dependence structure, which is the standard pragmatic
    choice for a scalar CI here.
    """
    rng = np.random.default_rng(seed)
    r, w = result.unit_rmsse, result.unit_weight
    n = r.size
    stats = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        wb = w[idx].sum()
        stats[b] = float((w[idx] * r[idx]).sum() / wb) if wb > 0 else np.nan
    stats = stats[np.isfinite(stats)]
    return {
        "mean": float(stats.mean()),
        "ci_low": float(np.quantile(stats, 0.025)),
        "ci_high": float(np.quantile(stats, 0.975)),
    }


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
