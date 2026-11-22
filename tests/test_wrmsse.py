"""Tests for WRMSSE: scales, aggregation, weights, bootstrap, gates."""

from __future__ import annotations

import numpy as np
import pandas as pd

from demand.data.m5 import M5Data, series_dollar_sales_window
from demand.evaluation.bootstrap import (
    _replicate_wrmsse,
    prepare_bootstrap,
)
from demand.evaluation.wrmsse import (
    LEVEL_GROUPS,
    LEVEL_SPECS,
    evaluate_wrmsse,
    first_nonzero_idx,
    level_group_map,
    point_metrics,
    rmsse_values,
    series_scale,
    unit_weight_keys,
)


def _tiny_m5() -> tuple[M5Data, pd.DataFrame]:
    """M5Data fixture: 8 series over 2 items x 2 stores x 2 states."""
    rows = []
    for i in range(2):
        for s in range(2):
            for st in range(2):
                rows.append(
                    {
                        "id": f"I{i}_S{s}_{st}",
                        "item_id": f"I{i}",
                        "dept_id": f"D{i}",
                        "cat_id": f"C{i}",
                        "store_id": f"S{s}",
                        "state_id": f"ST{st}",
                    }
                )
    ids = pd.DataFrame(rows)
    n_days = 40
    rng = np.random.default_rng(0)
    panel = rng.integers(0, 4, size=(8, n_days)).astype(np.float32)
    price_days = np.full((8, n_days), 2.0, dtype=np.float32)
    cal = pd.DataFrame({"d": [f"d_{i}" for i in range(1, n_days + 1)]})
    cal["wm_yr_wk"] = [f"W{(i - 1) // 7}" for i in range(1, n_days + 1)]
    data = M5Data(
        ids=ids,
        panel=panel,
        price_days=price_days,
        price_weekly=np.full((8, 6), 2.0),
        week_cols=[f"W{i}" for i in range(6)],
        day_week_pos=np.array(
            [(i - 1) // 7 % 6 for i in range(1, n_days + 1)], dtype=np.int32
        ),
        calendar=cal,
        prices=pd.DataFrame(),
        day_cols=[f"d_{i}" for i in range(1, n_days + 1)],
    )
    return data, ids


def test_first_nonzero_idx() -> None:
    assert first_nonzero_idx(np.array([0, 0, 0])) == 3
    assert first_nonzero_idx(np.array([0, 0, 5, 0])) == 2
    assert first_nonzero_idx(np.array([1, 0, 0])) == 0


def test_series_scale_uses_active_period_only() -> None:
    s = np.array([0, 0, 2, 4, 4, 6], dtype=np.float32)
    scale = series_scale(s[None, :])[0]
    active = np.array([2, 4, 4, 6], dtype=np.float64)
    expected = np.mean(np.diff(active) ** 2)
    assert scale == expected


def test_series_scale_vectorized_matches_loop() -> None:
    rng = np.random.default_rng(3)
    s = rng.integers(0, 3, size=(50, 60)).astype(np.float64)
    s[:, :10] = 0  # leading zeros exercise first-nonzero masking
    fast = series_scale(s)

    def slow(row: np.ndarray) -> float:
        nz = np.flatnonzero(row != 0)
        start = int(nz[0]) if nz.size else len(row)
        active = row[start:]
        if active.size < 2:
            return np.nan
        return float(np.mean(np.diff(active) ** 2))

    for i in range(50):
        assert np.isclose(fast[i], slow(s[i]), equal_nan=True)


def test_rmsse_values_perfect_and_constant() -> None:
    y_true = np.array([[1.0, 2.0], [3.0, 4.0]])
    scales = np.array([1.0, 1.0])
    np.testing.assert_allclose(rmsse_values(y_true, y_true.copy(), scales), 0.0)
    pred = y_true + 1.0
    np.testing.assert_allclose(rmsse_values(y_true, pred, scales), 1.0)


def test_level_group_map_covers_12_levels() -> None:
    _, ids = _tiny_m5()
    gm = level_group_map(ids)
    assert len(gm) == len(LEVEL_GROUPS) == len(LEVEL_SPECS) == 12
    assert gm["total"][0].tolist() == list(range(8))
    assert all(len(g) == 1 for g in gm["item_x_store"])
    item_groups = sorted([sorted(g.tolist()) for g in gm["item"]])
    assert item_groups == [[0, 1, 2, 3], [4, 5, 6, 7]]
    assert len(gm["state"]) == 2


def test_unit_weight_keys_match_official_layout() -> None:
    _, ids = _tiny_m5()
    gm = level_group_map(ids)
    keys = unit_weight_keys(ids, gm)
    # Level11 official order is (state, item).
    l11 = set(keys["item_x_state"])
    assert ("ST0", "I0") in l11 and ("ST1", "I1") in l11
    # Level12 official order is (item, store).
    l12 = set(keys["item_x_store"])
    assert ("I0", "S0") in l12 and ("I1", "S1") in l12
    assert keys["total"] == [("Total", "X")]
    assert ("ST0", "X") in set(keys["state"])


def test_evaluate_wrmsse_hand_computed() -> None:
    data, ids = _tiny_m5()
    origin_idx = 30  # in-sample 0..30, horizon days 31..39 (9 days)
    horizon = data.n_days - origin_idx - 1
    pred = np.repeat(
        data.panel[:, origin_idx][:, None], horizon, axis=1
    ).astype(np.float32)
    gm = level_group_map(ids)
    res = evaluate_wrmsse(
        data, gm, pred, origin_idx=origin_idx, horizon=horizon, weight_days=4
    )

    # Independent recomputation of the official per-level score.
    w0, w1 = origin_idx + 1 - 4, origin_idx + 1
    dollar = series_dollar_sales_window(data, w0, w1 - 1)
    y_true = data.panel[:, origin_idx + 1 : origin_idx + 1 + horizon]

    def scale_of(series: np.ndarray) -> float:
        nz = np.flatnonzero(series != 0)
        start = int(nz[0]) if nz.size else len(series)
        active = series[start:].astype(np.float64)
        if active.size < 2:
            return np.nan
        return float(np.mean(np.diff(active) ** 2))

    level_scores = []
    for groups in gm.values():
        num = den = 0.0
        for idx in groups:
            tr = data.panel[idx, : origin_idx + 1].sum(axis=0).astype(np.float64)
            yt = y_true[idx].sum(axis=0).astype(np.float64)
            yp = pred[idx].sum(axis=0).astype(np.float64)
            sc = scale_of(tr)
            sc = max(sc, 1e-8) if np.isfinite(sc) else 1e-8
            rmsse = float(np.sqrt(np.mean((yt - yp) ** 2) / sc))
            w = float(dollar[idx].sum())
            if w > 0:
                num += w * rmsse
                den += w
        level_scores.append(num / den if den > 0 else 0.0)
    expected = float(np.mean(level_scores))
    assert abs(res.wrmsse - expected) < 1e-9
    assert res.n_units == sum(len(g) for g in gm.values())


def test_evaluate_wrmsse_explicit_truth() -> None:
    """y_true may be supplied externally (private window)."""
    data, ids = _tiny_m5()
    origin_idx = 30
    horizon = 8
    rng = np.random.default_rng(5)
    y_true = rng.integers(0, 4, size=(8, horizon)).astype(np.float32)
    pred = np.zeros((8, horizon), dtype=np.float32)
    gm = level_group_map(ids)
    res = evaluate_wrmsse(
        data, gm, pred, origin_idx=origin_idx, y_true=y_true, horizon=horizon
    )
    assert np.isfinite(res.wrmsse)
    # Perfect prediction on external truth -> zero error part.
    res2 = evaluate_wrmsse(
        data, gm, y_true.copy(), origin_idx=origin_idx, y_true=y_true,
        horizon=horizon,
    )
    for lv in res2.per_level.values():
        assert lv == 0.0


def test_bootstrap_ones_counts_reproduce_point() -> None:
    """All-ones item counts must reproduce evaluate_wrmsse exactly."""
    data, ids = _tiny_m5()
    origin_idx = 30
    horizon = 8
    rng = np.random.default_rng(7)
    y_true = data.panel[:, origin_idx + 1 : origin_idx + 1 + horizon]
    pred_a = np.asarray(rng.random((8, horizon)) * 3, dtype=np.float32)
    pred_b = np.asarray(rng.random((8, horizon)) * 3, dtype=np.float32)
    gm = level_group_map(ids)
    preds = {"a": pred_a, "b": pred_b}
    results = {
        m: evaluate_wrmsse(
            data, gm, p, origin_idx=origin_idx, y_true=y_true, horizon=horizon
        )
        for m, p in preds.items()
    }
    prep = prepare_bootstrap(
        data, gm, preds, results,
        origin_idx=origin_idx, y_true=y_true, horizon=horizon,
    )
    k = np.ones(prep.n_items, dtype=np.int64)
    wr = _replicate_wrmsse(prep, k)
    for m in preds:
        assert abs(wr[m] - results[m].wrmsse) < 1e-5, (m, wr[m], results[m].wrmsse)


def test_bootstrap_uniformly_better_model() -> None:
    """Model A uniformly better than B -> paired diff < 0 every replicate."""
    data, ids = _tiny_m5()
    origin_idx = 30
    horizon = 8
    y_true = data.panel[:, origin_idx + 1 : origin_idx + 1 + horizon]
    rng = np.random.default_rng(11)
    base = y_true.astype(np.float64)
    pred_good = np.clip(base + rng.normal(0, 0.5, base.shape), 0, None).astype(np.float32)
    pred_bad = np.clip(base + rng.normal(0, 3.0, base.shape), 0, None).astype(np.float32)
    gm = level_group_map(ids)
    preds = {"seasonal_naive": pred_bad, "challenger": pred_good}
    results = {
        m: evaluate_wrmsse(
            data, gm, p, origin_idx=origin_idx, y_true=y_true, horizon=horizon
        )
        for m, p in preds.items()
    }
    prep = prepare_bootstrap(
        data, gm, preds, results,
        origin_idx=origin_idx, y_true=y_true, horizon=horizon,
    )
    probs = np.full(prep.n_items, 1.0 / prep.n_items)
    rng2 = np.random.default_rng(0)
    for _ in range(10):
        k = rng2.multinomial(prep.n_items, probs)
        wr = _replicate_wrmsse(prep, k)
        assert wr["challenger"] < wr["seasonal_naive"]


def test_point_metrics() -> None:
    y_true = np.array([[1.0, 2.0], [3.0, 4.0]])
    y_pred = y_true.copy()
    pm = point_metrics(y_true, y_pred)
    assert pm["mae"] == 0.0 and pm["rmse"] == 0.0 and pm["bias"] == 0.0
    pm2 = point_metrics(y_true, y_true + 2.0)
    assert pm2["mae"] == 2.0 and pm2["bias"] == 2.0
