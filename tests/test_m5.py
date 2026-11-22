"""Tests for M5 loading, protocol windows, truth loading, dollar sales."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from demand.data.m5 import (
    FULL_BOUNDS,
    PANEL_END_DAY,
    SMOKE_BOUNDS,
    TEST_END_DAY,
    day_index,
    load_m5,
    load_test_truth,
    origin_indices,
    series_dollar_sales_window,
    sha256_file,
    window_indices,
)

SERIES_KEY = ["item_id", "dept_id", "cat_id", "store_id", "state_id"]


def _write_m5_files(tmp: Path, n_series: int = 4) -> Path:
    """Write a tiny but structurally valid M5 dataset.

    Sales cover d_1..d_1941 like the real file; calendar covers
    d_1..d_1969 and prices cover every week, matching production.
    """
    rng = np.random.default_rng(0)
    days = [f"d_{i}" for i in range(1, PANEL_END_DAY + 1)]
    ids = pd.DataFrame(
        {
            "id": [f"ITEM{i}_S{i % 2}_CA_evaluation" for i in range(n_series)],
            "item_id": [f"ITEM{i}" for i in range(n_series)],
            "dept_id": [f"DEPT{i % 2}" for i in range(n_series)],
            "cat_id": [f"CAT{i % 2}" for i in range(n_series)],
            "store_id": [f"S{i % 2}" for i in range(n_series)],
            "state_id": ["CA"] * n_series,
        }
    )
    sales = rng.integers(0, 5, size=(n_series, PANEL_END_DAY)).astype(float)
    ev = pd.concat([ids, pd.DataFrame(sales, columns=days)], axis=1)
    ev.to_csv(tmp / "sales_train_evaluation.csv", index=False)

    cal_days = list(range(1, TEST_END_DAY + 1))
    cal = pd.DataFrame(
        {
            "d": [f"d_{d}" for d in cal_days],
            "date": pd.date_range("2011-01-29", periods=TEST_END_DAY, freq="D"),
            "wm_yr_wk": [11101 + (d - 1) // 7 for d in cal_days],
            "weekday": ["x"] * TEST_END_DAY,
            "wday": [(d - 1) % 7 for d in cal_days],
            "month": [1 + (d - 1) // 30 for d in cal_days],
            "year": [2011] * TEST_END_DAY,
            "event_name_1": [""] * TEST_END_DAY,
            "event_type_1": [""] * TEST_END_DAY,
            "event_name_2": [""] * TEST_END_DAY,
            "event_type_2": [""] * TEST_END_DAY,
            "snap_CA": [0] * TEST_END_DAY,
            "snap_TX": [0] * TEST_END_DAY,
            "snap_WI": [0] * TEST_END_DAY,
        }
    )
    cal.to_csv(tmp / "calendar.csv", index=False)

    weeks = sorted(cal["wm_yr_wk"].unique())
    price_rows = []
    for _, row in ids.iterrows():
        for w in weeks:
            price_rows.append(
                {
                    "store_id": row["store_id"],
                    "item_id": row["item_id"],
                    "wm_yr_wk": w,
                    "sell_price": 2.5,
                }
            )
    pd.DataFrame(price_rows).to_csv(tmp / "sell_prices.csv", index=False)
    return tmp


def _write_truth(
    tmp: Path, data_ids: pd.DataFrame, n_drop: int = 0
) -> Path:
    """Write a truth file in the official d_1942..d_1969 layout."""
    rng = np.random.default_rng(1)
    n = len(data_ids) - n_drop
    day_cols = [f"d_{d}" for d in range(1942, 1970)]
    vals = rng.integers(0, 4, size=(n, 28)).astype(float)
    truth = pd.concat(
        [
            data_ids[SERIES_KEY].iloc[:n].reset_index(drop=True),
            pd.DataFrame(vals, columns=day_cols),
        ],
        axis=1,
    )
    path = tmp / "sales_test_evaluation.csv"
    truth.to_csv(path, index=False)
    return path


def _patch_sha(monkeypatch: pytest.MonkeyPatch, path: Path) -> None:
    monkeypatch.setattr("demand.data.m5.TEST_TRUTH_SHA256", sha256_file(path))


def test_load_m5_shapes(tmp_path: Path) -> None:
    _write_m5_files(tmp_path)
    data = load_m5(tmp_path)
    assert data.n_series == 4
    assert data.n_days == PANEL_END_DAY
    assert data.panel.shape == (4, PANEL_END_DAY)
    assert data.price_days.shape == (4, TEST_END_DAY)
    assert data.price_weekly.shape[0] == 4
    assert len(data.day_week_pos) == TEST_END_DAY
    assert (data.price_days > 0).all()


def test_day_index() -> None:
    assert day_index("d_1") == 0
    assert day_index("d_1969") == 1968


def test_protocol_bounds() -> None:
    w = window_indices(FULL_BOUNDS)
    assert w["train"] == (0, 1912)
    assert w["validation"] == (1913, 1940)
    o = origin_indices(FULL_BOUNDS)
    assert o["validation"] == 1912  # d_1913
    assert o["test"] == 1940  # d_1941
    assert FULL_BOUNDS.test_start_day == 1942
    assert FULL_BOUNDS.test_end_day == 1969
    ws = window_indices(SMOKE_BOUNDS)
    assert ws["train"] == (0, 1884)
    assert ws["validation"] == (1885, 1912)
    os_ = origin_indices(SMOKE_BOUNDS)
    assert os_["test"] == 1912  # pseudo-test d_1914..d_1941
    assert SMOKE_BOUNDS.test_end_day == 1941


def test_series_dollar_sales_window(tmp_path: Path) -> None:
    _write_m5_files(tmp_path)
    data = load_m5(tmp_path)
    # Price 2.5 everywhere: dollar = 2.5 * unit sales over the window.
    dollars = series_dollar_sales_window(data, 0, 6)
    expected = data.panel[:, :7].sum(axis=1) * 2.5
    np.testing.assert_allclose(dollars, expected, rtol=1e-5)
    # A window reaching past the panel must fail.
    with pytest.raises(ValueError, match="beyond"):
        series_dollar_sales_window(data, 0, PANEL_END_DAY)


def test_sha256_file(tmp_path: Path) -> None:
    p = tmp_path / "x.bin"
    p.write_bytes(b"abc")
    assert sha256_file(p) == (
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )


def test_load_test_truth_roundtrip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_m5_files(tmp_path)
    data = load_m5(tmp_path)
    path = _write_truth(tmp_path, data.ids)
    _patch_sha(monkeypatch, path)
    panel, epoch = load_test_truth(path, data.ids)
    assert panel.shape == (4, 28)
    assert epoch > 0
    df = pd.read_csv(path)
    np.testing.assert_array_equal(
        panel,
        df[[c for c in df.columns if c.startswith("d_")]].to_numpy(np.float32),
    )


def test_load_test_truth_rejects_bad_sha(tmp_path: Path) -> None:
    _write_m5_files(tmp_path)
    data = load_m5(tmp_path)
    path = _write_truth(tmp_path, data.ids)
    with pytest.raises(ValueError, match="sha256"):
        load_test_truth(path, data.ids)


def test_load_test_truth_rejects_missing_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_m5_files(tmp_path)
    data = load_m5(tmp_path)
    path = _write_truth(tmp_path, data.ids, n_drop=1)
    _patch_sha(monkeypatch, path)
    with pytest.raises(ValueError, match="missing series keys"):
        load_test_truth(path, data.ids)
