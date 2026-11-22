"""Tests for baselines, v2 feature builder, leakage guard, inventory."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from demand.data.m5 import M5Data
from demand.evaluation.inventory import (
    OVERAGE_FRAC,
    PRIMARY_FRACTILE,
    order_up_to,
    residual_quantile,
    simulate_inventory,
)
from demand.models.aggregate import prophet_frame
from demand.models.lgbm import (
    CATEGORICAL_IDX,
    FEATURE_NAMES,
    MAX_ROUNDS,
    N_FEATURES,
    FeatureContext,
    build_feature_context,
    build_xy,
    cap_bound,
    predict_origin,
    sample_origins,
)
from demand.models.statistical import (
    BASELINE_BUILDERS,
    moving_average,
    naive,
    seasonal_naive,
)

N_PANEL = 200
N_CAL = 228  # calendar outlives the panel, like d_1969 vs d_1941


def _panel() -> np.ndarray:
    rng = np.random.default_rng(1)
    return rng.integers(0, 4, size=(6, N_PANEL)).astype(np.float32)


def _cal(n: int = N_CAL) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "d": [f"d_{i}" for i in range(1, n + 1)],
            "date": pd.date_range("2011-01-29", periods=n, freq="D"),
            "wm_yr_wk": [11101 + (i - 1) // 7 for i in range(1, n + 1)],
            "wday": [(i - 1) % 7 for i in range(1, n + 1)],
            "month": [1 + (i - 1) // 30 for i in range(1, n + 1)],
            "year": [2011] * n,
            "event_name_1": [""] * n,
            "event_type_1": [""] * n,
            "event_name_2": [""] * n,
            "event_type_2": [""] * n,
            "snap_CA": [0] * n,
            "snap_TX": [0] * n,
            "snap_WI": [0] * n,
        }
    )


def _ids(n: int = 6) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": [f"ITEM{i % 2}_S{i % 3}_ST{i % 2}" for i in range(n)],
            "item_id": [f"ITEM{i % 2}" for i in range(n)],
            "dept_id": [f"DEPT{i % 2}" for i in range(n)],
            "cat_id": [f"CAT{i % 2}" for i in range(n)],
            "store_id": [f"S{i % 3}" for i in range(n)],
            "state_id": ["CA" if i % 2 == 0 else "TX" for i in range(n)],
        }
    )


def _m5_data(panel: np.ndarray | None = None) -> M5Data:
    panel = _panel() if panel is None else panel
    n_series, n_days = panel.shape
    n_weeks = 40
    cal = _cal(N_CAL)
    day_week_pos = np.array(
        [int(c["wm_yr_wk"]) - 11101 for _, c in cal.iterrows()], dtype=np.int32
    )
    price_weekly = np.full((n_series, n_weeks), 3.0, dtype=np.float64)
    price_days = price_weekly[:, day_week_pos].astype(np.float32)
    return M5Data(
        ids=_ids(n_series),
        panel=panel,
        price_days=price_days,
        price_weekly=price_weekly,
        week_cols=[11101 + i for i in range(n_weeks)],
        day_week_pos=day_week_pos,
        calendar=cal,
        prices=pd.DataFrame(),
        day_cols=[f"d_{i}" for i in range(1, n_days + 1)],
    )


def _ctx(data: M5Data) -> FeatureContext:
    return build_feature_context(data)


def test_naive_repeats_last_value() -> None:
    panel = _panel()
    fc = naive(panel, 40, 5)
    assert fc.shape == (6, 5)
    np.testing.assert_array_equal(
        fc, np.repeat(panel[:, 40][:, None], 5, axis=1)
    )


def test_seasonal_naive_cycle() -> None:
    panel = _panel()
    fc = seasonal_naive(panel, 40, 10, season=7)
    np.testing.assert_array_equal(fc[:, :7], panel[:, 34:41])


def test_moving_average_constant() -> None:
    panel = np.tile(np.arange(28, dtype=np.float32), (2, 8))[:, :200]
    fc = moving_average(panel, 40, 4, window=28)
    assert fc.shape == (2, 4)
    expected = panel[:, 13:41].mean(axis=1)
    np.testing.assert_allclose(fc[:, 0], expected, rtol=1e-5)
    np.testing.assert_allclose(fc, np.repeat(expected[:, None], 4, axis=1))


def test_all_baselines_registered() -> None:
    assert set(BASELINE_BUILDERS) == {"naive", "seasonal_naive", "ma28"}


def test_sample_origins_respects_min_lookback() -> None:
    origins = sample_origins(0, 500, n=10)
    # Longest lookback is 112 days; MIN_ORIGIN=120 adds margin. The
    # caller passes max_origin already reduced by the label horizon.
    assert len(origins) == 10
    assert origins.min() >= 120
    assert origins.max() <= 500
    assert origins[-1] == 500


def test_feature_layout() -> None:
    assert len(FEATURE_NAMES) == N_FEATURES == 33
    assert len(CATEGORICAL_IDX) == 10
    assert all(0 <= i < N_FEATURES for i in CATEGORICAL_IDX)


def test_build_xy_labels_and_shape() -> None:
    data = _m5_data()
    ctx = _ctx(data)
    origins = np.array([130, 140])
    X, y = build_xy(data.panel, ctx, data, origins, 28)
    assert X.shape == (2 * 28 * 6, N_FEATURES)
    assert y is not None and y.shape == (X.shape[0],)
    # Target for (origin 130, h=1, series s) is panel[s, 131].
    np.testing.assert_allclose(
        y[:6], data.panel[:, 131], rtol=1e-5
    )


def test_build_xy_no_future_leakage() -> None:
    """Randomizing panel[:, o+1:] must not change any feature at o."""
    import dataclasses

    data = _m5_data()
    ctx = _ctx(data)
    origin = np.array([130])
    X1, _ = build_xy(data.panel, ctx, data, origin, 28)

    corrupted = data.panel.copy()
    rng = np.random.default_rng(9)
    corrupted[:, 131:] = rng.integers(0, 9, size=corrupted[:, 131:].shape)
    X2, _ = build_xy(corrupted, ctx, data, origin, 28)
    np.testing.assert_array_equal(X1, X2)

    # FeatureContext itself must be causal: rebuild it on the corrupted
    # panel (last_nz and friends) and the features must still match.
    data_c = dataclasses.replace(data, panel=corrupted)
    ctx_c = _ctx(data_c)
    X3, _ = build_xy(corrupted, ctx_c, data_c, origin, 28)
    np.testing.assert_array_equal(X1, X3)

    # Sanity: labels DO see the future, so y must differ.
    _, y1 = build_xy(data.panel, ctx, data, origin, 28)
    _, y2 = build_xy(corrupted, ctx, data, origin, 28)
    assert not np.allclose(y1, y2)


def test_build_xy_rejects_labels_beyond_panel() -> None:
    data = _m5_data()
    ctx = _ctx(data)
    with pytest.raises(ValueError, match="beyond panel"):
        # origin 180 -> labels reach day 208 > panel 200.
        build_xy(data.panel, ctx, data, np.array([180]), 28)
    # with_labels=False supports targets beyond the panel.
    X, y = build_xy(
        data.panel, ctx, data, np.array([180]), 28, with_labels=False
    )
    assert y is None and X.shape[0] == 28 * 6


def test_predict_origin_shape() -> None:
    class _Const:
        def predict(self, X: np.ndarray) -> np.ndarray:
            return np.full(X.shape[0], 2.5)

    data = _m5_data()
    ctx = _ctx(data)
    pred = predict_origin(_Const(), data.panel, ctx, data, 130, 28)
    assert pred.shape == (6, 28)
    np.testing.assert_allclose(pred, 2.5)


def test_prophet_frame_uses_real_dates() -> None:
    dates = pd.Series(pd.date_range("2011-01-29", periods=50, freq="D"))
    series = np.arange(50, dtype=float)
    df = prophet_frame(series, dates)
    assert list(df.columns) == ["ds", "y"]
    pd.testing.assert_series_equal(
        df["ds"].reset_index(drop=True),
        pd.to_datetime(dates).reset_index(drop=True),
        check_names=False,
    )
    np.testing.assert_array_equal(df["y"].to_numpy(), series)


def test_inventory_scenario_overage() -> None:
    y_true = np.array([[5.0, 3.0], [2.0, 8.0]])
    order = np.array([[6.0, 4.0], [1.0, 9.0]])
    price = np.ones_like(y_true) * 10.0
    stats = simulate_inventory(y_true, order, price, overage_frac=OVERAGE_FRAC)
    # Day 0: s0 sells min(5,6)=5 leftover 1 -> overage 2.5; s1 sells
    # min(2,1)=1 shortage 1 -> stockout 10, leftover 0.
    # Day 1: s0 sells min(3,4)=3 leftover 1 -> 2.5; s1 sells min(8,9)=8
    # leftover 1 -> 2.5.
    assert stats["lost_units"] == 1.0
    assert stats["stockout_cost"] == 10.0
    assert stats["overage_cost"] == 7.5
    assert stats["total_cost"] == 17.5
    assert "holding_cost" not in stats


def test_cap_bound_flags_only_cap_region() -> None:
    # v2 evidence: best_iteration 399 at MAX_ROUNDS=400 with
    # EARLY_STOP=50 bound the cap (399 >= 400 - 50).
    assert cap_bound(399, max_rounds=400, early_stop=50)
    assert cap_bound(MAX_ROUNDS)
    assert cap_bound(MAX_ROUNDS - 50)
    assert not cap_bound(MAX_ROUNDS - 51)
    assert not cap_bound(100)


def test_residual_quantile_fractile() -> None:
    y_val = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    y_pred = np.zeros_like(y_val)
    q = residual_quantile(y_val, y_pred, PRIMARY_FRACTILE)
    assert q.shape == (2,)
    assert q[0] > 2.0 and q[0] <= 3.0
    levels = order_up_to(np.array([[10.0], [20.0]]), q)
    assert levels.shape == (2, 1)
    assert levels[0, 0] > 10.0
