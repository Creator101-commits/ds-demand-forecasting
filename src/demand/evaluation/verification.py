"""Protocol gates: official weights, file consistency, score checks.

- ``check_weights``: our dollar sales per (level, unit) over the two
  official weight windows must reproduce ``weights_validation.csv``
  (d_1886..d_1913) and ``weights_evaluation.csv`` (d_1914..d_1941).
- ``check_consistency``: the competition file's d_1914..d_1941 columns
  must equal the organizers' ``sales_test_validation.csv`` exactly.
- ``check_official_scores``: our Naive/sNaive WRMSSE on the private
  window must match the published benchmark table within 1e-5.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from demand.data.m5 import (
    HORIZON,
    SERIES_KEY,
    M5Data,
    load_test_truth,
    series_dollar_sales_window,
)
from demand.data.official import (
    OFFICIAL_FILES,
    SCORE_TOL,
    WEIGHT_DOLLAR_TOL,
    WEIGHT_REL_TOL,
    compare_scores,
    official_targets,
    verify_official_dataset,
)
from demand.evaluation.wrmsse import (
    LEVEL_IDS,
    LEVEL_SPECS,
    evaluate_wrmsse,
    level_group_map,
    unit_weight_keys,
)

# (weight file, day window as 0-based inclusive indices).
WEIGHT_WINDOWS = {
    "weights_validation.csv": (1885, 1912),  # d_1886..d_1913
    "weights_evaluation.csv": (1913, 1940),  # d_1914..d_1941
}


def _our_weights_table(
    data: M5Data,
    group_map: dict[str, list[np.ndarray]],
    day_start: int,
    day_end: int,
) -> pd.DataFrame:
    """Our Dollar_Sales/weight per (Level_id, Agg1, Agg2) for a window."""
    dollar = series_dollar_sales_window(data, day_start, day_end)
    keys = unit_weight_keys(data.ids, group_map)
    rows: list[dict[str, Any]] = []
    for name, _, _, _ in LEVEL_SPECS:
        groups = group_map[name]
        level_total = 0.0
        unit_dollars = np.empty(len(groups))
        for u, idx in enumerate(groups):
            unit_dollars[u] = dollar[idx].sum()
        level_total = float(unit_dollars.sum())
        for u, (k1, k2) in enumerate(keys[name]):
            rows.append(
                {
                    "Level_id": LEVEL_IDS[name],
                    "Agg_Level_1": k1,
                    "Agg_Level_2": k2,
                    "Dollar_Sales": float(unit_dollars[u]),
                    "weight": float(unit_dollars[u] / level_total)
                    if level_total > 0
                    else 0.0,
                }
            )
    return pd.DataFrame(rows)


def check_weights(data: M5Data, official_dir: Path) -> dict[str, Any]:
    """Gate: our weights must reproduce the official weight files."""
    verify_official_dataset(official_dir)
    group_map = level_group_map(data.ids)
    report: dict[str, Any] = {}
    all_pass = True
    for fname, (d0, d1) in WEIGHT_WINDOWS.items():
        ours = _our_weights_table(data, group_map, d0, d1)
        official = pd.read_csv(official_dir / fname)
        merged = official.merge(
            ours,
            on=["Level_id", "Agg_Level_1", "Agg_Level_2"],
            how="outer",
            indicator=True,
            suffixes=("_official", "_ours"),
        )
        n_matched = int((merged["_merge"] == "both").sum())
        n_expected = int(official.shape[0])
        matched = merged[merged["_merge"] == "both"]
        dollar_diff = (
            (matched["Dollar_Sales_ours"] - matched["Dollar_Sales_official"])
            .abs()
            .max()
            if len(matched)
            else float("nan")
        )
        w_off = matched["weight_official"].to_numpy(dtype=np.float64)
        w_our = matched["weight_ours"].to_numpy(dtype=np.float64)
        denom = np.maximum(np.abs(w_off), 1e-12)
        rel_diff = float(np.max(np.abs(w_our - w_off) / denom)) if len(matched) else float("nan")
        ok = (
            n_matched == n_expected == 42840
            and len(matched) == n_expected
            and float(dollar_diff) <= WEIGHT_DOLLAR_TOL
            and rel_diff <= WEIGHT_REL_TOL
        )
        all_pass = all_pass and ok
        report[fname] = {
            "n_matched": n_matched,
            "n_expected": n_expected,
            "max_abs_dollar_diff": float(dollar_diff),
            "max_rel_weight_diff": rel_diff,
            "pass": bool(ok),
        }
    report["pass"] = bool(all_pass)
    return report


def check_consistency(data: M5Data, official_dir: Path) -> dict[str, Any]:
    """Gate: competition d_1914..d_1941 == organizers' validation truth."""
    truth = pd.read_csv(official_dir / "sales_test_validation.csv")
    day_cols = [c for c in truth.columns if c.startswith("d_")]
    merged = data.ids[SERIES_KEY].merge(
        truth, on=SERIES_KEY, how="left", validate="1:1", indicator=True
    )
    if not (merged["_merge"] == "both").all():
        raise ValueError("sales_test_validation.csv missing series keys")
    ours = data.panel[:, 1913:1941]  # d_1914..d_1941
    theirs = merged[day_cols].to_numpy(dtype=np.float64)
    if ours.shape != theirs.shape:
        raise ValueError(f"shape mismatch {ours.shape} vs {theirs.shape}")
    diff = np.abs(ours.astype(np.float64) - theirs)
    return {
        "n_compared": int(ours.size),
        "max_abs_diff": float(diff.max()),
        "pass": bool(diff.max() == 0.0),
    }


def check_test_truth_integrity(
    data: M5Data, official_dir: Path
) -> dict[str, Any]:
    """Load-only integrity check of the private-window truth file."""
    path = official_dir / "sales_test_evaluation.csv"
    panel, epoch = load_test_truth(path, data.ids)
    return {
        "sha256": OFFICIAL_FILES["sales_test_evaluation.csv"],
        "shape": list(panel.shape),
        "first_test_read_epoch": epoch,
        "pass": True,
    }


def check_official_scores(
    data: M5Data,
    group_map: dict[str, list[np.ndarray]],
    preds: dict[str, np.ndarray],
    *,
    origin_idx: int,
    y_true: np.ndarray,
    official_dir: Path,
) -> dict[str, Any]:
    """Gate: Naive/sNaive WRMSSE vs the published benchmark table."""
    targets = official_targets(official_dir / "scores_and_ranks.xlsx")
    ours: dict[str, dict[str, float]] = {}
    # Official targets are keyed L1..L12 in LEVEL_SPECS order.
    level_keys = [f"L{i + 1}" for i in range(len(LEVEL_SPECS))]
    spec_names = [name for name, _, _, _ in LEVEL_SPECS]
    for name in ("naive", "seasonal_naive"):
        res = evaluate_wrmsse(
            data,
            group_map,
            preds[name],
            origin_idx=origin_idx,
            y_true=y_true,
            horizon=HORIZON,
        )
        ours[name] = {
            lk: res.per_level[sn]
            for lk, sn in zip(level_keys, spec_names, strict=True)
        } | {"Avg": res.wrmsse}
    report = compare_scores(ours, targets)
    report["tolerance"] = SCORE_TOL
    return report
