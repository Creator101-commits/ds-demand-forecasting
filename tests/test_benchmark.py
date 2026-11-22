"""Orchestration tests: plan-before-truth ordering and gate failures.

A tiny synthetic M5-like dataset exercises run_benchmark end to end in
both smoke and full modes with the sha-pinned official pieces stubbed.
The truth loaders assert analysis_plan.json already exists when they
are called, which proves the frozen ordering inside the real flow.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from demand.data.m5 import PANEL_END_DAY, TEST_END_DAY
from demand.data.official import OFFICIAL_FILES
from demand.experimentation import benchmark as bm

TINY_GRID = [{"num_leaves": 15, "learning_rate": 0.1}]


def _write_m5(data_dir: Path) -> None:
    """Tiny competition-format dataset: sales to d_1941, cal to d_1969."""
    rng = np.random.default_rng(0)
    combos = [
        (item, f"DEPT{item[-1]}", f"CAT{item[-1]}", store, state)
        for item in ("ITEM0", "ITEM1")
        for store in ("S0", "S1")
        for state in ("CA", "TX")
    ]
    ids = pd.DataFrame(
        combos,
        columns=["item_id", "dept_id", "cat_id", "store_id", "state_id"],
    )
    ids.insert(
        0,
        "id",
        [
            f"{r.item_id}_{r.store_id}_{r.state_id}"
            for r in ids.itertuples()
        ],
    )
    days = [f"d_{i}" for i in range(1, PANEL_END_DAY + 1)]
    sales = rng.integers(0, 5, size=(len(ids), PANEL_END_DAY)).astype(float)
    pd.concat(
        [ids, pd.DataFrame(sales, columns=days)], axis=1
    ).to_csv(data_dir / "sales_train_evaluation.csv", index=False)

    n_cal = TEST_END_DAY
    cal_days = list(range(1, n_cal + 1))
    cal = pd.DataFrame(
        {
            "d": [f"d_{d}" for d in cal_days],
            "date": pd.date_range("2011-01-29", periods=n_cal, freq="D"),
            "wm_yr_wk": [11101 + (d - 1) // 7 for d in cal_days],
            "wday": [(d - 1) % 7 for d in cal_days],
            "month": [1 + (d - 1) // 30 for d in cal_days],
            "year": [2011] * n_cal,
            "event_name_1": [""] * n_cal,
            "event_type_1": [""] * n_cal,
            "event_name_2": [""] * n_cal,
            "event_type_2": [""] * n_cal,
            "snap_CA": [0] * n_cal,
            "snap_TX": [0] * n_cal,
            "snap_WI": [0] * n_cal,
        }
    )
    cal.to_csv(data_dir / "calendar.csv", index=False)

    weeks = sorted(cal["wm_yr_wk"].unique())
    price_rows = [
        {
            "store_id": r.store_id,
            "item_id": r.item_id,
            "wm_yr_wk": w,
            "sell_price": 2.5,
        }
        for r in ids.itertuples()
        for w in weeks
    ]
    pd.DataFrame(price_rows).to_csv(data_dir / "sell_prices.csv", index=False)


def _stub_official(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace every sha-pinned or file-reading official check."""
    monkeypatch.setattr(
        bm,
        "verify_official_dataset",
        lambda d: {name: "0" * 64 for name in OFFICIAL_FILES},
    )
    monkeypatch.setattr(
        bm,
        "check_weights",
        lambda data, d: {"pass": True, "stub": True},
    )
    monkeypatch.setattr(
        bm,
        "check_consistency",
        lambda data, d: {"pass": True, "max_abs_diff": 0.0, "stub": True},
    )
    monkeypatch.setattr(
        bm,
        "official_reference",
        lambda p: {"note": "stub", "ES_bu": None},
    )
    monkeypatch.setattr(bm, "GRID", TINY_GRID)
    monkeypatch.setattr(bm, "AGG_LEVELS", ("total",))


def _truth_loader(monkeypatch: pytest.MonkeyPatch, out_dir: Path) -> None:
    """load_test_truth wrapper asserting the plan exists first."""

    def _load(path: Path, ids: pd.DataFrame):
        assert (out_dir / "analysis_plan.json").exists(), (
            "test truth loaded before analysis_plan.json was written"
        )
        rng = np.random.default_rng(3)
        return (
            rng.integers(0, 4, size=(len(ids), 28)).astype(np.float32),
            time.time(),
        )

    monkeypatch.setattr(bm, "load_test_truth", _load)


def _integrity_stub(monkeypatch: pytest.MonkeyPatch, out_dir: Path) -> None:
    def _check(data, d):
        assert (out_dir / "analysis_plan.json").exists(), (
            "truth integrity check ran before analysis_plan.json"
        )
        return {
            "sha256": "stub",
            "shape": [data.n_series, 28],
            "first_test_read_epoch": time.time(),
            "pass": True,
        }

    monkeypatch.setattr(bm, "check_test_truth_integrity", _check)


def _common_args(tmp_path: Path) -> tuple[Path, Path, Path]:
    data_dir = tmp_path / "m5"
    data_dir.mkdir()
    _write_m5(data_dir)
    official_dir = tmp_path / "official"
    official_dir.mkdir()
    out_dir = tmp_path / "out"
    return data_dir, official_dir, out_dir


def test_run_benchmark_smoke_orders_plan_before_truth(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_dir, official_dir, out_dir = _common_args(tmp_path)
    _stub_official(monkeypatch)
    _integrity_stub(monkeypatch, out_dir)

    res = bm.run_benchmark(
        data_dir,
        official_dir,
        out_dir,
        n_origins=3,
        n_boot=2,
        git_sha="test-sha",
        smoke=True,
    )
    assert res["manifest"]["ordering_ok"] is True
    for name in (
        "analysis_plan.json",
        "metric_verification.json",
        "metrics.json",
        "official_reference.json",
        "manifest.json",
    ):
        assert (out_dir / name).exists(), name
    plan = json.loads((out_dir / "analysis_plan.json").read_text())
    assert plan["smoke"] is True
    assert plan["lgbm_grid"] == TINY_GRID


def test_run_benchmark_full_orders_plan_before_truth(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_dir, official_dir, out_dir = _common_args(tmp_path)
    _stub_official(monkeypatch)
    _truth_loader(monkeypatch, out_dir)
    monkeypatch.setattr(
        bm,
        "check_official_scores",
        lambda *a, **k: {"pass": True, "stub": True},
    )

    res = bm.run_benchmark(
        data_dir,
        official_dir,
        out_dir,
        n_origins=3,
        n_boot=2,
        git_sha="test-sha",
        smoke=False,
    )
    assert res["manifest"]["ordering_ok"] is True
    ver = json.loads((out_dir / "metric_verification.json").read_text())
    assert ver["official_scores"]["pass"] is True
    assert ver["pass"] is True
    plan = json.loads((out_dir / "analysis_plan.json").read_text())
    assert plan["smoke"] is False


def test_run_benchmark_full_stops_on_failed_score_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_dir, official_dir, out_dir = _common_args(tmp_path)
    _stub_official(monkeypatch)
    _truth_loader(monkeypatch, out_dir)
    monkeypatch.setattr(
        bm,
        "check_official_scores",
        lambda *a, **k: {"pass": False, "stub": "failed"},
    )

    def _boom(*a: object, **k: object) -> None:
        raise AssertionError("refit reached despite failed score gate")

    monkeypatch.setattr(bm, "_lgbm_refit", _boom)

    with pytest.raises(RuntimeError, match="official WRMSSE check failed"):
        bm.run_benchmark(
            data_dir,
            official_dir,
            out_dir,
            n_origins=3,
            n_boot=2,
            git_sha="test-sha",
            smoke=False,
        )
    ver = json.loads((out_dir / "metric_verification.json").read_text())
    assert ver["official_scores"]["pass"] is False
    assert (out_dir / "analysis_plan.json").exists()
