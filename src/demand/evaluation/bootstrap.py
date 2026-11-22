"""Item-cluster bootstrap for WRMSSE (paired across models).

The i.i.d. unit bootstrap is not meaningful here: the single Level1
unit carries 1/12 of all weight. Instead each replicate draws
``k = multinomial(n_items, uniform)`` counts per item and sets every
series count ``c_j = k[item(j)]``.

- Levels 1-9 (aggregates mixing items): recompute c-weighted
  aggregates of the in-sample panel (scales), truth, and each model's
  forecast through CSR membership matrices; unit weight is the
  c-weighted dollar sum; level score is the within-level weighted mean
  RMSSE.
- Levels 10-12 (item, state x item, item x store): every member of a
  unit shares one item, so c-weighting scales the whole unit by
  ``k_item`` and RMSSE is invariant; reuse each model's full-sample
  unit RMSSE and weight ``k_item * w_unit``.
- Replicate WRMSSE = mean of the 12 level scores (official equal-level
  weighting). The point estimate stays each model's full-sample
  WRMSSE.

The interval reflects sensitivity to the sampled product population
with stores and the single 28-day window fixed; it is not temporal
uncertainty.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp

from demand.data.m5 import WEIGHT_WINDOW, M5Data, series_dollar_sales_window
from demand.evaluation.wrmsse import WRMSSEResult, rmsse_values

N_LOWER_LEVELS = 9  # levels 1..9 need c-weighted re-aggregation


@dataclass
class _LevelBlock:
    name: str
    row_start: int
    row_end: int


@dataclass
class BootstrapPrep:
    """Reusable structures for item-cluster bootstrap replicates."""

    membership: sp.csr_matrix  # (units levels 1-9, n_series) 0/1
    blocks: list[_LevelBlock]
    item_of_series: np.ndarray  # (n_series,) item index 0..n_items-1
    n_items: int
    dollar: np.ndarray  # (n_series,) float64 weight-window dollars
    panel_in: np.ndarray  # (n_series, origin_idx+1) float32
    y_true: np.ndarray  # (n_series, horizon) float32
    preds: dict[str, np.ndarray]  # model -> (n_series, horizon)
    models: list[str]
    # Levels 10-12 reuse: per level -> (item_idx per unit, w0 per unit,
    # rmsse0 per unit per model)
    upper_item: dict[str, np.ndarray]
    upper_w0: dict[str, np.ndarray]
    upper_rmsse0: dict[str, np.ndarray]  # (n_units, n_models)
    horizon: int


def _build_membership(
    group_map: dict[str, list[np.ndarray]], n_series: int
) -> tuple[sp.csr_matrix, list[_LevelBlock]]:
    names = list(group_map)[:N_LOWER_LEVELS]
    indptr = [0]
    indices: list[int] = []
    data: list[float] = []
    blocks: list[_LevelBlock] = []
    row = 0
    for name in names:
        start = row
        for idx in group_map[name]:
            indices.extend(int(j) for j in idx)
            data.extend([1.0] * len(idx))
            indptr.append(len(indices))
            row += 1
        blocks.append(_LevelBlock(name=name, row_start=start, row_end=row))
    mat = sp.csr_matrix(
        (
            np.asarray(data, dtype=np.float32),
            np.asarray(indices, dtype=np.int32),
            np.asarray(indptr, dtype=np.int64),
        ),
        shape=(row, n_series),
    )
    return mat, blocks


def prepare_bootstrap(
    data: M5Data,
    group_map: dict[str, list[np.ndarray]],
    preds: dict[str, np.ndarray],
    results: dict[str, WRMSSEResult],
    *,
    origin_idx: int,
    y_true: np.ndarray,
    horizon: int,
) -> BootstrapPrep:
    """Build CSR memberships and per-level caches once."""
    n_series = data.n_series
    membership, blocks = _build_membership(group_map, n_series)
    item_codes, item_uniques = pd.factorize(data.ids["item_id"])
    item_of_series = np.asarray(item_codes, dtype=np.int64)
    n_items = len(item_uniques)
    dollar = series_dollar_sales_window(
        data, origin_idx + 1 - WEIGHT_WINDOW, origin_idx
    )

    # Unit offsets inside the concatenated unit arrays, per level.
    upper_item: dict[str, np.ndarray] = {}
    upper_w0: dict[str, np.ndarray] = {}
    upper_rmsse0: dict[str, np.ndarray] = {}
    models = list(preds)
    offset = 0
    for li, (name, groups) in enumerate(group_map.items()):
        n_units = len(groups)
        if li < N_LOWER_LEVELS:
            offset += n_units
            continue
        item_idx = np.empty(n_units, dtype=np.int64)
        for u, idx in enumerate(groups):
            item_idx[u] = item_of_series[int(idx[0])]
        w0 = results[models[0]].unit_weight[offset : offset + n_units]
        rmsse0 = np.stack(
            [
                results[m].unit_rmsse[offset : offset + n_units]
                for m in models
            ],
            axis=1,
        )
        upper_item[name] = item_idx
        upper_w0[name] = w0.astype(np.float64)
        upper_rmsse0[name] = rmsse0.astype(np.float64)
        offset += n_units

    return BootstrapPrep(
        membership=membership,
        blocks=blocks,
        item_of_series=item_of_series,
        n_items=n_items,
        dollar=dollar,
        panel_in=np.ascontiguousarray(data.panel[:, : origin_idx + 1]),
        y_true=np.asarray(y_true, dtype=np.float32),
        preds={k: np.asarray(v, dtype=np.float32) for k, v in preds.items()},
        models=models,
        upper_item=upper_item,
        upper_w0=upper_w0,
        upper_rmsse0=upper_rmsse0,
        horizon=horizon,
    )


def _replicate_wrmsse(prep: BootstrapPrep, k: np.ndarray) -> dict[str, float]:
    """WRMSSE per model for one item-count draw."""
    c = k[prep.item_of_series].astype(np.float32)
    # membership @ diag(c): scale each nnz by its column's count.
    scaled = sp.csr_matrix(
        (
            prep.membership.data * c[prep.membership.indices],
            prep.membership.indices,
            prep.membership.indptr,
        ),
        shape=prep.membership.shape,
    )

    # c-weighted in-sample aggregates -> per-unit RMSSE scales.
    y_agg = scaled @ prep.panel_in  # (units_1_9, T_in)
    t_in = y_agg.shape[1]
    nz = y_agg > 0
    has = nz.any(axis=1)
    fnz = np.where(has, np.argmax(nz, axis=1), t_in)
    diffs = np.diff(y_agg.astype(np.float64), axis=1)
    idx = np.arange(t_in - 1)[None, :]
    mask = idx >= fnz[:, None]
    counts = mask.sum(axis=1)
    scale = np.where(
        counts > 0,
        (diffs * diffs * mask).sum(axis=1) / np.maximum(counts, 1),
        np.nan,
    )

    w_units = scaled @ prep.dollar.astype(np.float32)
    yt = scaled @ prep.y_true

    out: dict[str, float] = {}
    for mi, model in enumerate(prep.models):
        yp = scaled @ prep.preds[model]
        rmsse = rmsse_values(yt.astype(np.float64), yp.astype(np.float64), scale)
        level_scores: list[float] = []
        for b in prep.blocks:
            sl = slice(b.row_start, b.row_end)
            w = w_units[sl]
            r = rmsse[sl]
            denom = float(w.sum())
            if denom <= 0:
                level_scores.append(float("nan"))
                continue
            contrib = w * np.where(w > 0, r, 0.0)
            level_scores.append(float(contrib.sum() / denom))
        for name in prep.upper_item:
            kw = k[prep.upper_item[name]].astype(np.float64) * prep.upper_w0[name]
            denom = float(kw.sum())
            if denom <= 0:
                level_scores.append(float("nan"))
                continue
            r0 = prep.upper_rmsse0[name][:, mi]
            contrib = kw * np.where(kw > 0, r0, 0.0)
            level_scores.append(float(contrib.sum() / denom))
        out[model] = float(np.mean(level_scores))
    return out


def _paired_summary(d: np.ndarray, a: str, b: str) -> dict[str, Any]:
    return {
        "a": a,
        "b": b,
        "diff_mean": float(d.mean()) if d.size else float("nan"),
        "ci_low": float(np.quantile(d, 0.025)) if d.size else float("nan"),
        "ci_high": float(np.quantile(d, 0.975)) if d.size else float("nan"),
        "share_negative": float((d < 0).mean()) if d.size else float("nan"),
    }


def item_cluster_bootstrap(
    data: M5Data,
    group_map: dict[str, list[np.ndarray]],
    preds: dict[str, np.ndarray],
    results: dict[str, WRMSSEResult],
    *,
    origin_idx: int,
    y_true: np.ndarray,
    horizon: int,
    n_boot: int = 500,
    seed: int = 0,
    val_wrmsse: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Paired item-cluster bootstrap over test WRMSSE.

    Reports per-model percentile CIs and paired differences (every
    model minus seasonal_naive; lightgbm minus the best non-LightGBM
    model by validation WRMSSE) with CI and share of replicates < 0.
    """
    prep = prepare_bootstrap(
        data,
        group_map,
        preds,
        results,
        origin_idx=origin_idx,
        y_true=y_true,
        horizon=horizon,
    )
    rng = np.random.default_rng(seed)
    models = prep.models
    samples = {m: np.empty(n_boot, dtype=np.float64) for m in models}
    probs = np.full(prep.n_items, 1.0 / prep.n_items)
    for b in range(n_boot):
        k = rng.multinomial(prep.n_items, probs)
        wr = _replicate_wrmsse(prep, k)
        for m in models:
            samples[m][b] = wr[m]

    out_models: dict[str, Any] = {}
    for m in models:
        s = samples[m]
        finite = s[np.isfinite(s)]
        out_models[m] = {
            "point": float(results[m].wrmsse),
            "mean": float(finite.mean()) if finite.size else float("nan"),
            "ci_low": (
                float(np.quantile(finite, 0.025)) if finite.size else float("nan")
            ),
            "ci_high": (
                float(np.quantile(finite, 0.975)) if finite.size else float("nan")
            ),
            "n_finite": int(finite.size),
        }

    paired: dict[str, Any] = {}
    ref = "seasonal_naive"
    for m in models:
        if m == ref:
            continue
        d = samples[m] - samples[ref]
        paired[f"{m}_minus_{ref}"] = _paired_summary(
            d[np.isfinite(d)], m, ref
        )

    if val_wrmsse and "lightgbm" in samples:
        candidates = {m: v for m, v in val_wrmsse.items() if m != "lightgbm"}
        if candidates:
            best_other = min(candidates, key=lambda m: candidates[m])
            d = samples["lightgbm"] - samples[best_other]
            paired[f"lightgbm_minus_{best_other}"] = _paired_summary(
                d[np.isfinite(d)], "lightgbm", best_other
            )
    return {
        "n_boot": n_boot,
        "seed": seed,
        "n_items": prep.n_items,
        "models": out_models,
        "paired": paired,
    }
