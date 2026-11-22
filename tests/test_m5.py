"""Tests for M5 loading and frozen split boundaries."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from demand.data.m5 import (
    TEST_END_DAY,
    TRAIN_END_DAY,
    VAL_END_DAY,
    day_index,
    load_m5,
    series_dollar_sales,
    split_indices,
)

BOUNDARY = 20  # tiny-fixture analog of the frozen d_1913 boundary


def _write_m5_files(tmp: Path, n_series: int = 4, n_days: int = 40) -> Path:
    """Write a tiny but structurally valid M5 dataset.

    Mimics the real files: the validation file covers days through the
    boundary and the evaluation file covers all days (its early columns
    duplicate the validation file).
    """
    rng = np.random.default_rng(0)
    days = [f"d_{i}" for i in range(1, n_days + 1)]
    ids = pd.DataFrame(
        {
            "id": [f"S{i}_ITEM{i}" for i in range(n_series)],
            "item_id": [f"ITEM{i}" for i in range(n_series)],
            "dept_id": [f"DEPT{i % 2}" for i in range(n_series)],
            "cat_id": [f"CAT{i % 2}" for i in range(n_series)],
            "store_id": [f"S{i % 2}" for i in range(n_series)],
            "state_id": ["CA" for _ in range(n_series)],
        }
    )
    sales = rng.integers(0, 5, size=(n_series, n_days)).astype(float)
    val = pd.concat(
        [ids, pd.DataFrame(sales[:, :BOUNDARY], columns=days[:BOUNDARY])],
        axis=1,
    )
    ev = pd.concat(
        [ids, pd.DataFrame(sales, columns=days)], axis=1
    )
    val.to_csv(tmp / "sales_train_validation.csv", index=False)
    ev.to_csv(tmp / "sales_train_evaluation.csv", index=False)

    cal_days = list(range(1, n_days + 1))
    cal = pd.DataFrame(
        {
            "d": [f"d_{d}" for d in cal_days],
            "wm_yr_wk": [f"W{(d - 1) // 7}" for d in cal_days],
            "wday": [(d - 1) % 7 for d in cal_days],
            "month": [1 + (d - 1) // 30 for d in cal_days],
            "year": [2011 + (d - 1) // 365 for d in cal_days],
            "event_name_1": [""] * n_days,
            "event_type_1": [""] * n_days,
            "event_name_2": [""] * n_days,
            "event_type_2": [""] * n_days,
            "snap_CA": [0] * n_days,
            "snap_TX": [0] * n_days,
            "snap_WI": [0] * n_days,
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


def test_load_m5_shapes_and_alignment(tmp_path: Path) -> None:
    _write_m5_files(tmp_path)
    data = load_m5(tmp_path, boundary_day=BOUNDARY)
    assert data.n_series == 4
    assert data.n_days == 40
    assert data.panel.shape == (4, 40)
    assert data.price_days.shape == (4, 40)
    # Every series has a positive price on every day.
    assert (data.price_days > 0).all()
    assert list(data.ids.columns)[:3] == ["id", "item_id", "dept_id"]


def test_day_index() -> None:
    assert day_index("d_1") == 0
    assert day_index("d_1941") == 1940


def test_split_indices_frozen_boundaries() -> None:
    splits = split_indices()
    assert splits["train"] == (0, TRAIN_END_DAY - 1)
    assert splits["validation"] == (TRAIN_END_DAY, VAL_END_DAY - 1)
    assert splits["test"] == (VAL_END_DAY, TEST_END_DAY - 1)
    # Windows are contiguous and non-overlapping.
    assert splits["validation"][0] == splits["train"][1] + 1
    assert splits["test"][0] == splits["validation"][1] + 1
    assert TEST_END_DAY == 1941


def test_series_dollar_sales(tmp_path: Path) -> None:
    _write_m5_files(tmp_path)
    data = load_m5(tmp_path, boundary_day=BOUNDARY)
    # Price is 2.5 everywhere, so dollar sales = 2.5 * unit sales.
    dollars = series_dollar_sales(data, 0, 9)
    expected = data.panel[:, :10].sum(axis=1) * 2.5
    np.testing.assert_allclose(dollars, expected, rtol=1e-5)


def test_load_m5_rejects_eval_train_disagreement(tmp_path: Path) -> None:
    _write_m5_files(tmp_path)
    ev_path = tmp_path / "sales_train_evaluation.csv"
    ev = pd.read_csv(ev_path)
    ev.loc[0, "d_1"] = ev.loc[0, "d_1"] + 100
    ev.to_csv(ev_path, index=False)
    with pytest.raises(ValueError, match="disagrees"):
        load_m5(tmp_path, boundary_day=BOUNDARY)


def test_load_m5_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_m5(tmp_path)
