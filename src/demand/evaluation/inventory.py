"""Inventory scenario analysis (newsvendor order-up-to).

Declared cost assumptions (not measured): holding cost = 25% of sell
price per unit-period, stockout cost = sell price per unit. Order-up-to
levels come from point forecasts plus validation residual quantiles at
target service level 0.95, simulated day-by-day on the test window.
This is scenario analysis, not a measured saving.
"""

from __future__ import annotations

import numpy as np

SERVICE_LEVEL = 0.95
HOLDING_FRAC = 0.25


def residual_quantile(
    y_val: np.ndarray, y_pred_val: np.ndarray, service: float = SERVICE_LEVEL
) -> np.ndarray:
    """Per-series quantile of (actual - forecast) on validation."""
    resid = y_val - y_pred_val  # (n_series, h)
    return np.asarray(np.quantile(resid, service, axis=1), dtype=np.float64)


def order_up_to(y_pred: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Order-up-to level per (series, day): forecast + residual quantile."""
    return np.asarray(np.clip(y_pred + q[:, None], 0.0, None))


def simulate_inventory(
    y_true: np.ndarray,
    order_levels: np.ndarray,
    price_days_window: np.ndarray,
    holding_frac: float = HOLDING_FRAC,
) -> dict[str, float]:
    """Day-by-day newsvendor simulation on the test window.

    ``y_true`` and ``order_levels`` are (n_series, horizon);
    ``price_days_window`` is (n_series, horizon) sell price per day.
    Each day: stock = leftover + order_up_to; sales = min(demand, stock);
    shortage cost = price * (demand - sales); holding cost =
    holding_frac * price * (stock - sales). No pipeline stock carried
    beyond one day (per-period newsvendor).
    """
    n, h = y_true.shape
    total_cost = 0.0
    total_stockout = 0.0
    total_holding = 0.0
    total_lost_units = 0.0
    for t in range(h):
        demand = y_true[:, t]
        avail = order_levels[:, t]
        sales = np.minimum(demand, avail)
        shortage = demand - sales
        leftover = avail - sales
        price = price_days_window[:, t]
        stockout_cost = float((shortage * price).sum())
        holding_cost = float((leftover * price * holding_frac).sum())
        total_cost += stockout_cost + holding_cost
        total_stockout += stockout_cost
        total_holding += holding_cost
        total_lost_units += float(shortage.sum())
    return {
        "total_cost": total_cost,
        "stockout_cost": total_stockout,
        "holding_cost": total_holding,
        "lost_units": total_lost_units,
        "per_series_day_cost": total_cost / (n * h),
    }
