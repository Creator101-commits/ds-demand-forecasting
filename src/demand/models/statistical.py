"""Statistical baselines: naive, seasonal naive (7), 28-day mean.

All operate on the item-store panel (n_series, n_days) float32 and return
(n_series, horizon) forecasts for the days after ``origin_idx``.
"""

from __future__ import annotations

import numpy as np

SEASON = 7
MA_DAYS = 28


def naive(panel: np.ndarray, origin_idx: int, horizon: int) -> np.ndarray:
    """Repeat the last observed value for every horizon step."""
    last = panel[:, origin_idx]
    return np.repeat(last[:, None], horizon, axis=1)


def seasonal_naive(
    panel: np.ndarray, origin_idx: int, horizon: int, season: int = SEASON
) -> np.ndarray:
    """Repeat the last ``season`` observations cyclically."""
    tail = panel[:, origin_idx - season + 1 : origin_idx + 1]
    reps = int(np.ceil(horizon / season))
    tiled = np.tile(tail, (1, reps))[:, :horizon]
    return tiled.astype(np.float32, copy=False)


def moving_average(
    panel: np.ndarray, origin_idx: int, horizon: int, window: int = MA_DAYS
) -> np.ndarray:
    """Constant forecast = mean of the last ``window`` observations."""
    tail = panel[:, origin_idx - window + 1 : origin_idx + 1].astype(np.float64)
    mean = tail.mean(axis=1)
    return np.repeat(mean[:, None], horizon, axis=1).astype(np.float32)


BASELINE_BUILDERS = {
    "naive": naive,
    "seasonal_naive": seasonal_naive,
    "ma28": moving_average,
}
