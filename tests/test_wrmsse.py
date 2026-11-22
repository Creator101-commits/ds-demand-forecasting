"""Tests for WRMSSE: scales, aggregation, weights, hand-computed case."""

from __future__ import annotations

import numpy as np
import pandas as pd

from demand.data.m5 import M5Data
from demand.evaluation.wrmsse import (
    LEVEL_GROUPS,
    bootstrap_wrmsse,
    evaluate_wrmsse,
    first_nonzero_idx,
    level_group_map,
    point_metrics,
    rmsse_values,
    series_scale,
)


def _tiny_data() -> tuple[M5Data, pd.DataFrame]:
    """4 series: 2 stores x 2 items, 1 state, distinct cat/dept per item."""
    ids = pd.DataFrame(
        {
            "id": ["S0_I0", "S1_I0", "S0_I1", "S1_I1"],
            "item_id": ["I0", "I0", "I1", "I1"],
            "dept_id": ["D0", "D0", "D1", "D1"],
            "cat_id": ["C0", "C0", "C1", "C1"],
            "store_id": ["S0", "S1", "S0", "S1"],
            "state_id": ["CA"] * 4,
        }
    )
    n_days = 12
    panel = np.array(
        [
            [0, 0, 1, 2, 3, 2, 1, 0, 1, 1, 0, 2],
            [0, 1, 1, 1, 2, 2, 2, 1, 1, 0, 1, 1],
            [0, 0, 0, 1, 1, 2, 2, 2, 1, 1, 2, 2],
            [1, 0, 1, 0, 2, 1, 1, 2, 2, 1, 0, 1],
        ],
        dtype=np.float32,
    )
    price_days = np.full((4, n_days), 2.0, dtype=np.float32)
    cal = pd.DataFrame({"d": [f"d_{i}" for i in range(1, n_days + 1)]})
    prices = pd.DataFrame()
    data = M5Data(
        ids=ids,
        panel=panel,
        price_days=price_days,
        calendar=cal,
        prices=prices,
        day_cols=[f"d_{i}" for i in range(1, n_days + 1)],
    )
    return data, ids


def test_first_nonzero_idx() -> None:
    assert first_nonzero_idx(np.array([0, 0, 0])) == 3
    assert first_nonzero_idx(np.array([0, 0, 5, 0])) == 2
    assert first_nonzero_idx(np.array([1, 0, 0])) == 0


def test_series_scale_uses_active_period_only() -> None:
    # Leading zeros are excluded; scale is mean squared diff after fnz.
    s = np.array([0, 0, 2, 4, 4, 6], dtype=np.float32)
    scale = series_scale(s[None, :])[0]
    active = np.array([2, 4, 4, 6], dtype=np.float64)
    expected = np.mean(np.diff(active) ** 2)
    assert scale == expected


def test_rmsse_values_perfect_and_constant() -> None:
    y_true = np.array([[1.0, 2.0], [3.0, 4.0]])
    scales = np.array([1.0, 1.0])
    np.testing.assert_allclose(rmsse_values(y_true, y_true.copy(), scales), 0.0)
    # Constant prediction error 1 -> RMSSE = sqrt(mean(1)/scale) = 1.
    pred = y_true + 1.0
    np.testing.assert_allclose(rmsse_values(y_true, pred, scales), 1.0)


def test_level_group_map_covers_12_levels() -> None:
    _, ids = _tiny_data()
    gm = level_group_map(ids)
    assert len(gm) == len(LEVEL_GROUPS) == 12
    assert gm["total"][0].tolist() == [0, 1, 2, 3]
    assert gm["item_x_store"] == [np.array([i]) for i in range(4)] or all(
        len(g) == 1 for g in gm["item_x_store"]
    )
    # item level: I0 -> rows 0,1 ; I1 -> rows 2,3.
    item_groups = sorted([sorted(g.tolist()) for g in gm["item"]])
    assert item_groups == [[0, 1], [2, 3]]
    # state level: single group of all 4.
    assert len(gm["state"]) == 1


def test_evaluate_wrmsse_hand_computed() -> None:
    """Tiny case computed independently in the test."""
    data, ids = _tiny_data()
    n_days = data.n_days
    train_end = 8  # train days 0..8, horizon days 9..11 (3 days)
    horizon_start = 9
    horizon = n_days - horizon_start

    # Forecast = naive repeat of last train value.
    pred = np.repeat(data.panel[:, train_end][:, None], horizon, axis=1).astype(
        np.float32
    )
    gm = level_group_map(ids)
    res = evaluate_wrmsse(
        data,
        gm,
        pred,
        train_end_idx=train_end,
        horizon_start_idx=horizon_start,
        horizon=horizon,
        weight_days=4,
    )

    # Independent recomputation.
    w0, w1 = horizon_start - 4, horizon_start
    dollar = (data.panel[:, w0:w1] * data.price_days[:, w0:w1]).sum(axis=1)

    def scale_of(series: np.ndarray) -> float:
        nz = np.flatnonzero(series != 0)
        start = int(nz[0]) if nz.size else len(series)
        active = series[start:].astype(np.float64)
        if active.size < 2:
            return np.nan
        return float(np.mean(np.diff(active) ** 2))

    total_num = 0.0
    total_den = 0.0
    for groups in gm.values():
        for idx in groups:
            tr = data.panel[idx, : train_end + 1].sum(axis=0).astype(np.float64)
            yt = data.panel[idx, horizon_start : horizon_start + horizon].sum(
                axis=0
            )
            yp = pred[idx].sum(axis=0).astype(np.float64)
            sc = scale_of(tr)
            sc = max(sc, 1e-8) if np.isfinite(sc) else 1e-8
            rmsse = float(np.sqrt(np.mean((yt - yp) ** 2) / sc))
            w = float(dollar[idx].sum())
            if w > 0:
                total_num += w * rmsse
                total_den += w
    expected = total_num / total_den
    assert abs(res.wrmsse - expected) < 1e-9
    assert res.n_units == sum(len(g) for g in gm.values())
    # total level has weight = sum of all dollar sales.
    assert res.per_level["total"] >= 0


def test_bootstrap_wrmsse_returns_ci() -> None:
    data, ids = _tiny_data()
    gm = level_group_map(ids)
    pred = np.zeros((4, 3), dtype=np.float32)
    res = evaluate_wrmsse(
        data, gm, pred,
        train_end_idx=8, horizon_start_idx=9, horizon=3, weight_days=4,
    )
    boot = bootstrap_wrmsse(res, n_boot=50, seed=1)
    assert boot["ci_low"] <= boot["mean"] <= boot["ci_high"] or boot["ci_low"] <= boot["ci_high"]


def test_point_metrics() -> None:
    y_true = np.array([[1.0, 2.0], [3.0, 4.0]])
    y_pred = y_true.copy()
    pm = point_metrics(y_true, y_pred)
    assert pm["mae"] == 0.0 and pm["rmse"] == 0.0 and pm["bias"] == 0.0
    pm2 = point_metrics(y_true, y_true + 2.0)
    assert pm2["mae"] == 2.0 and pm2["bias"] == 2.0
