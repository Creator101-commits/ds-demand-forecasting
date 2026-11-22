"""Frozen-plan M5 benchmark orchestration.

Protocol: train <= d_1885, validation d_1886..d_1913, test d_1914..d_1941.
Baselines forecast both origins; the LightGBM grid is chosen by
validation WRMSSE then refit on train+val origins for test at seeds
0,1,2; SARIMA and Prophet run on the 14 level 1-3 series; inventory is a
labeled scenario analysis. The plan JSON (card_write_epoch) is persisted
before the first test-window metric computation.
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np

from demand.data.m5 import (
    HORIZON,
    M5Data,
    load_m5,
    split_indices,
)
from demand.evaluation.inventory import (
    order_up_to,
    residual_quantile,
    simulate_inventory,
)
from demand.evaluation.wrmsse import (
    WRMSSEResult,
    bootstrap_wrmsse,
    evaluate_wrmsse,
    level_group_map,
    point_metrics,
    rmsse_unweighted,
)
from demand.models.aggregate import AGG_LEVELS, aggregate_forecasts, aggregate_panel
from demand.models.lgbm import (
    EARLY_STOP,
    GRID,
    MAX_ROUNDS,
    N_ORIGINS,
    SEED,
    build_xy,
    predict_origin,
    sample_origins,
    train_config,
)
from demand.models.statistical import BASELINE_BUILDERS

PROTOCOL = "m5-benchmark-v1"
SEEDS = (0, 1, 2)
SERVICE_LEVEL = 0.95
HOLDING_FRAC = 0.25


def _epoch() -> float:
    return time.time()


def _write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")


def _git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def _wrmsse_at_origin(
    data: M5Data,
    group_map: dict[str, list[np.ndarray]],
    pred: np.ndarray,
    train_end_idx: int,
    horizon_start_idx: int,
    n_boot: int,
) -> tuple[WRMSSEResult, dict[str, float]]:
    res = evaluate_wrmsse(
        data,
        group_map,
        pred,
        train_end_idx=train_end_idx,
        horizon_start_idx=horizon_start_idx,
        horizon=HORIZON,
    )
    ci = (
        bootstrap_wrmsse(res, n_boot=n_boot, seed=0)
        if n_boot > 0
        else {"mean": res.wrmsse, "ci_low": res.wrmsse, "ci_high": res.wrmsse}
    )
    return res, ci


def _fit_predict_baselines(
    data: M5Data, splits: dict[str, tuple[int, int]], horizon: int
) -> dict[str, dict[str, np.ndarray]]:
    """Forecast every baseline at the validation and test origins."""
    panel = data.panel
    val_origin = splits["validation"][0] - 1  # last day before val window
    test_origin = splits["test"][0] - 1
    out: dict[str, dict[str, np.ndarray]] = {}
    for name, fn in BASELINE_BUILDERS.items():
        out[name] = {
            "validation": fn(panel, val_origin, horizon).astype(np.float32),
            "test": fn(panel, test_origin, horizon).astype(np.float32),
        }
    return out


def _refit_fixed_iterations(
    X: np.ndarray,
    y: np.ndarray,
    config: dict[str, Any],
    num_boost_round: int,
    seed: int,
) -> Any:
    """Refit on train+val origins with the validation-selected iteration."""
    params = {
        "objective": "tweedie",
        "tweedie_variance_power": 1.5,
        "metric": "l2",
        "seed": seed,
        "num_threads": -1,
        "verbosity": -1,
        "max_bin": 255,
        **config,
    }
    return lgb.train(params, lgb.Dataset(X, label=y), num_boost_round=num_boost_round)


def _lgbm_pipeline(
    data: M5Data,
    splits: dict[str, tuple[int, int]],
    horizon: int,
    n_origins: int,
) -> dict[str, Any]:
    """Grid selection on validation WRMSSE, then seeded refits for test."""
    tr0, tr1 = splits["train"]
    va0, va1 = splits["validation"]
    te0, _ = splits["test"]
    group_map = level_group_map(data.ids)

    train_origins = sample_origins(tr0, tr1, n_origins)
    X_tr, y_tr = build_xy(
        data.panel, data.price_days, data.calendar, data.ids, train_origins, horizon
    )
    val_origin_idx = va0 - 1  # d_1885 -> forecasts d_1886..d_1913
    # Early stopping watches the validation window itself: features at the
    # val origin use only train data (lags >= 28), labels are val actuals.
    X_es, y_es = build_xy(
        data.panel, data.price_days, data.calendar, data.ids,
        np.array([val_origin_idx]), horizon,
    )

    grid_scores: list[dict[str, Any]] = []
    val_predictions: dict[int, np.ndarray] = {}
    for ci, cfg in enumerate(GRID):
        model, best_it = train_config(X_tr, y_tr, X_es, y_es, cfg, SEED)
        pred_val = predict_origin(
            model, data.panel, data.price_days, data.calendar, data.ids,
            val_origin_idx, horizon,
        )
        res, _ = _wrmsse_at_origin(data, group_map, pred_val, tr1, va0, 0)
        grid_scores.append(
            {"config": cfg, "best_iteration": best_it, "val_wrmsse": res.wrmsse}
        )
        val_predictions[ci] = pred_val
        del model
    del X_tr, y_tr, X_es, y_es

    best_i = int(np.argmin([g["val_wrmsse"] for g in grid_scores]))
    best_cfg = grid_scores[best_i]["config"]
    best_iter = grid_scores[best_i]["best_iteration"]

    # Refit on train+val origins for the test window, seeds 0,1,2.
    all_origins = sample_origins(tr0, va1, n_origins)
    X_all, y_all = build_xy(
        data.panel, data.price_days, data.calendar, data.ids, all_origins, horizon
    )
    test_origin_idx = te0 - 1  # d_1913 -> forecasts d_1914..d_1941
    test_predictions = []
    for seed in SEEDS:
        model = _refit_fixed_iterations(X_all, y_all, best_cfg, best_iter, seed)
        test_predictions.append(
            predict_origin(
                model, data.panel, data.price_days, data.calendar, data.ids,
                test_origin_idx, horizon,
            )
        )
        del model

    return {
        "grid_scores": grid_scores,
        "best_config_index": best_i,
        "selected_config": best_cfg,
        "selected_iteration": best_iter,
        "val_wrmsse_selected": grid_scores[best_i]["val_wrmsse"],
        "val_predictions": val_predictions,
        "test_predictions_per_seed": test_predictions,
        "test_prediction_mean": np.mean(np.stack(test_predictions), axis=0).astype(
            np.float32
        ),
        "n_origins": n_origins,
        "early_stop": EARLY_STOP,
        "max_rounds": MAX_ROUNDS,
    }


def _inventory_analysis(
    data: M5Data,
    splits: dict[str, tuple[int, int]],
    val_preds: dict[str, np.ndarray],
    test_preds: dict[str, np.ndarray],
    horizon: int,
) -> dict[str, Any]:
    """Newsvendor scenario: order-up-to from val residual quantiles.

    Scenario analysis only, not a measured saving. Candidates are every
    model with both a validation and a test prediction.
    """
    va0, va1 = splits["validation"]
    te0, _ = splits["test"]
    y_val = data.panel[:, va0 : va1 + 1].astype(np.float64)
    y_test = data.panel[:, te0 : te0 + horizon].astype(np.float64)
    price_test = data.price_days[:, te0 : te0 + horizon].astype(np.float64)

    out: dict[str, Any] = {}
    for name, test_pred in test_preds.items():
        val_pred = val_preds.get(name)
        if val_pred is None:
            continue
        q = residual_quantile(y_val, val_pred.astype(np.float64), SERVICE_LEVEL)
        levels = order_up_to(test_pred.astype(np.float64), q)
        stats = simulate_inventory(y_test, levels, price_test, HOLDING_FRAC)
        out[name] = {
            "service_level": SERVICE_LEVEL,
            "holding_frac": HOLDING_FRAC,
            **stats,
        }
    return out


def run_benchmark(
    data_dir: Path,
    out_dir: Path,
    *,
    n_origins: int = N_ORIGINS,
    n_boot: int = 1000,
    git_sha: str | None = None,
    smoke: bool = False,
) -> dict[str, Any]:
    """Run the frozen protocol and write benchmark artifacts."""
    out_dir.mkdir(parents=True, exist_ok=True)
    data = load_m5(data_dir)
    splits = split_indices()
    horizon = HORIZON
    group_map = level_group_map(data.ids)

    baseline_preds = _fit_predict_baselines(data, splits, horizon)
    lgbm = _lgbm_pipeline(data, splits, horizon, n_origins)

    # Validation WRMSSE for baselines and every grid config's prediction.
    tr1 = splits["train"][1]
    va0 = splits["validation"][0]
    val_results: dict[str, Any] = {}
    val_pred_map: dict[str, np.ndarray] = {}
    for name, preds in baseline_preds.items():
        res, ci = _wrmsse_at_origin(
            data, group_map, preds["validation"], tr1, va0, n_boot
        )
        val_results[name] = {
            "wrmsse": res.wrmsse,
            "ci": ci,
            "per_level": res.per_level,
        }
        val_pred_map[name] = preds["validation"]
    for ci, pred in lgbm["val_predictions"].items():
        cfg = GRID[ci]
        key = f"lightgbm_{cfg['num_leaves']}x{cfg['learning_rate']}"
        res, _ = _wrmsse_at_origin(data, group_map, pred, tr1, va0, 0)
        val_results[key] = {"wrmsse": res.wrmsse, "config": cfg}
    val_pred_map["lightgbm"] = lgbm["val_predictions"][lgbm["best_config_index"]]

    # Freeze the plan before any test-window metric.
    card_write_epoch = _epoch()
    plan = {
        "protocol": PROTOCOL,
        "seed": SEED,
        "seeds": list(SEEDS),
        "horizon": horizon,
        "splits": {
            k: {"start_day": v[0] + 1, "end_day": v[1] + 1}
            for k, v in splits.items()
        },
        "n_origins": n_origins,
        "lgbm_grid": GRID,
        "lgbm_early_stop": {"early_stop": EARLY_STOP, "max_rounds": MAX_ROUNDS},
        "lgbm_selected": {
            "config": lgbm["selected_config"],
            "best_iteration": lgbm["selected_iteration"],
            "val_wrmsse": lgbm["val_wrmsse_selected"],
        },
        "service_level": SERVICE_LEVEL,
        "holding_frac": HOLDING_FRAC,
        "n_bootstrap": n_boot,
        "smoke": smoke,
    }
    _write_json(out_dir / "analysis_plan.json", plan)
    first_test_read_epoch = _epoch()

    # ---- Test window: first and only test metric computation. ----
    te0 = splits["test"][0]
    test_train_end = te0 - 1  # train+val through d_1913
    test_results: dict[str, Any] = {}
    test_pred_map: dict[str, np.ndarray] = {}

    for name, preds in baseline_preds.items():
        pred = preds["test"]
        res, ci = _wrmsse_at_origin(
            data, group_map, pred, test_train_end, te0, n_boot
        )
        pm = point_metrics(
            data.panel[:, te0 : te0 + horizon], pred
        )
        test_results[name] = {
            "wrmsse": res.wrmsse,
            "ci": ci,
            "per_level": res.per_level,
            "point": pm,
        }
        test_pred_map[name] = pred

    pred_mean = lgbm["test_prediction_mean"]
    res, ci = _wrmsse_at_origin(
        data, group_map, pred_mean, test_train_end, te0, n_boot
    )
    pm = point_metrics(data.panel[:, te0 : te0 + horizon], pred_mean)
    per_seed = []
    for seed, pred in zip(SEEDS, lgbm["test_predictions_per_seed"], strict=True):
        r, _ = _wrmsse_at_origin(data, group_map, pred, test_train_end, te0, 0)
        per_seed.append({"seed": seed, "wrmsse": r.wrmsse})
    test_results["lightgbm"] = {
        "wrmsse": res.wrmsse,
        "ci": ci,
        "per_level": res.per_level,
        "point": pm,
        "per_seed": per_seed,
    }
    test_pred_map["lightgbm"] = pred_mean

    # Aggregate-level comparison on test: SARIMA + Prophet vs LGBM bottom-up.
    agg_results: dict[str, Any] = {}
    for level in AGG_LEVELS:
        level_entry: dict[str, Any] = {}
        agg_tr, _ = aggregate_panel(
            data.panel[:, : test_train_end + 1], data.ids, level
        )
        true_agg, agg_names = aggregate_panel(
            data.panel[:, te0 : te0 + horizon], data.ids, level
        )
        lgbm_agg, _ = aggregate_panel(pred_mean, data.ids, level)
        for method in ("sarima", "prophet"):
            fc, _ = aggregate_forecasts(
                level, data.panel, data.ids, test_train_end, horizon, method
            )
            rms = [
                rmsse_unweighted(agg_tr[i], true_agg[i], fc[i])
                for i in range(len(agg_names))
            ]
            level_entry[method] = {
                "rmsse_mean": float(np.mean(rms)),
                "rmsse_per_series": {
                    n: float(v) for n, v in zip(agg_names, rms, strict=True)
                },
            }
        rms = [
            rmsse_unweighted(agg_tr[i], true_agg[i], lgbm_agg[i])
            for i in range(len(agg_names))
        ]
        level_entry["lightgbm_bottomup"] = {
            "rmsse_mean": float(np.mean(rms)),
            "rmsse_per_series": {
                n: float(v) for n, v in zip(agg_names, rms, strict=True)
            },
        }
        agg_results[level] = level_entry

    inv = _inventory_analysis(
        data, splits, val_pred_map, test_pred_map, horizon
    )

    metrics = {
        "protocol": PROTOCOL,
        "validation": val_results,
        "test": test_results,
        "aggregate_test": agg_results,
        "inventory_scenario": inv,
        "smoke": smoke,
    }
    _write_json(out_dir / "metrics.json", metrics)

    manifest = {
        "git_sha": git_sha or _git_sha(),
        "protocol": PROTOCOL,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "card_write_epoch": card_write_epoch,
        "first_test_read_epoch": first_test_read_epoch,
        "ordering_ok": bool(card_write_epoch < first_test_read_epoch),
        "data": {
            "n_series": data.n_series,
            "n_days": data.n_days,
            "files": [
                "sales_train_validation.csv",
                "sales_train_evaluation.csv",
                "calendar.csv",
                "sell_prices.csv",
            ],
            "source": "kaggle m5-forecasting-accuracy",
        },
        "config": plan,
        "artifacts": [
            "analysis_plan.json",
            "metrics.json",
            "manifest.json",
            "report.md",
        ],
    }
    _write_json(out_dir / "manifest.json", manifest)
    (out_dir / "report.md").write_text(
        render_report(metrics, manifest), encoding="utf-8"
    )
    return {"metrics": metrics, "manifest": manifest}


def render_report(metrics: dict[str, Any], manifest: dict[str, Any]) -> str:
    lines = [
        "# M5 Demand Forecasting Report",
        "",
        f"- git SHA: `{manifest['git_sha']}`",
        f"- Protocol: {metrics['protocol']}",
        f"- ordering_ok: {manifest['ordering_ok']}",
        "",
        "## Validation WRMSSE (lower is better)",
        "",
        "| Model | WRMSSE |",
        "| :--- | ---: |",
    ]
    for name, r in metrics["validation"].items():
        lines.append(f"| {name} | {r['wrmsse']:.4f} |")
    lines += [
        "",
        "## Test WRMSSE",
        "",
        "| Model | WRMSSE | 95% CI | MAE | Bias |",
        "| :--- | ---: | :--- | ---: | ---: |",
    ]
    for name, r in metrics["test"].items():
        c = r["ci"]
        lines.append(
            f"| {name} | {r['wrmsse']:.4f} "
            f"| [{c['ci_low']:.4f}, {c['ci_high']:.4f}] "
            f"| {r['point']['mae']:.2f} | {r['point']['bias']:.2f} |"
        )
    lines.append("")
    return "\n".join(lines)
