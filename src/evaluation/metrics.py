from __future__ import annotations

import numpy as np



def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))



def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))



def mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-6) -> float:
    denom = np.where(np.abs(y_true) < eps, eps, np.abs(y_true))
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)



def smape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-6) -> float:
    denom = np.maximum(np.abs(y_true) + np.abs(y_pred), eps)
    return float(np.mean(2.0 * np.abs(y_pred - y_true) / denom) * 100.0)



def wrmsse_proxy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    train_series: np.ndarray,
    weights: np.ndarray | None = None,
    eps: float = 1e-6,
) -> float:
    if train_series.size < 2:
        scale = 1.0
    else:
        scale = np.mean(np.diff(train_series) ** 2)
    scale = max(scale, eps)

    rmsse = np.sqrt(np.mean((y_true - y_pred) ** 2) / scale)
    if weights is None:
        return float(rmsse)
    return float(np.average(np.repeat(rmsse, len(weights)), weights=weights))
