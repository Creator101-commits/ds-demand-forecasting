"""Inventory scenario analysis (per-period newsvendor).

Declared cost assumptions (not measured): underage (stockout) cost =
sell price per unit, overage (unsold leftover) cost = 0.25 x sell
price per unit. The critical fractile is Cu / (Cu + Co) = 1 / 1.25 =
0.8, so the primary order quantile is the 0.8 residual quantile with a
0.95 sensitivity row. Order-up-to levels come from point forecasts
plus validation residual quantiles, simulated day-by-day on the test
window. This is scenario analysis, not a measured saving.
"""

from __future__ import annotations

import numpy as np

OVERAGE_FRAC = 0.25
PRIMARY_FRACTILE = 0.8
SENSITIVITY_FRACTILE = 0.95


def residual_quantile(
    y_val: np.ndarray, y_pred_val: np.ndarray, fractile: float
) -> np.ndarray:
    """Per-series quantile of (actual - forecast) on validation."""
    resid = y_val - y_pred_val  # (n_series, h)
    return np.asarray(np.quantile(resid, fractile, axis=1), dtype=np.float64)


def order_up_to(y_pred: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Order-up-to level per (series, day): forecast + residual quantile."""
    return np.asarray(np.clip(y_pred + q[:, None], 0.0, None))


def simulate_inventory(
    y_true: np.ndarray,
    order_levels: np.ndarray,
    price_days_window: np.ndarray,
    overage_frac: float = OVERAGE_FRAC,
) -> dict[str, float]:
    """Day-by-day newsvendor simulation on the test window.

    ``y_true`` and ``order_levels`` are (n_series, horizon);
    ``price_days_window`` is (n_series, horizon) sell price per day.
    Each day: stock = order_up_to; sales = min(demand, stock);
    underage cost = price * (demand - sales); overage cost =
    overage_frac * price * (stock - sales). No pipeline stock carried
    beyond one day (per-period newsvendor).
    """
    n, h = y_true.shape
    total_cost = 0.0
    total_stockout = 0.0
    total_overage = 0.0
    total_lost_units = 0.0
    for t in range(h):
        demand = y_true[:, t]
        avail = order_levels[:, t]
        sales = np.minimum(demand, avail)
        shortage = demand - sales
        leftover = avail - sales
        price = price_days_window[:, t]
        stockout_cost = float((shortage * price).sum())
        overage_cost = float((leftover * price * overage_frac).sum())
        total_cost += stockout_cost + overage_cost
        total_stockout += stockout_cost
        total_overage += overage_cost
        total_lost_units += float(shortage.sum())
    return {
        "total_cost": total_cost,
        "stockout_cost": total_stockout,
        "overage_cost": total_overage,
        "lost_units": total_lost_units,
        "per_series_day_cost": total_cost / (n * h),
    }
