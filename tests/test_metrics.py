from __future__ import annotations

import numpy as np

from src.evaluation.metrics import mae, rmse, smape



def test_metric_values_are_finite() -> None:
    y_true = np.array([10.0, 20.0, 30.0])
    y_pred = np.array([12.0, 18.0, 33.0])

    assert np.isfinite(rmse(y_true, y_pred))
    assert np.isfinite(mae(y_true, y_pred))
    assert np.isfinite(smape(y_true, y_pred))
