"""Frozen-plan M5 benchmark orchestration (protocol m5-benchmark-v2).

Train d_1..d_1913, validation d_1914..d_1941 (origin d_1913), test
d_1942..d_1969 (origin d_1941) scored against the organizers' truth
file, which is loaded only after ``analysis_plan.json`` is persisted.
Smoke mode shifts the whole protocol back 28 days so pseudo-test
 d_1914..d_1941 comes from the panel and never touches official truth.

Sequence: weights + consistency gates -> baselines -> LightGBM grid ->
select -> persist plan -> load test truth -> official score gate ->
refit 3 seeds -> test WRMSSE + item-cluster bootstrap -> aggregate
comparison -> inventory scenario -> official reference -> manifest.
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np

from demand.data.m5 import (
    COMPETITION_FILES,
    FULL_BOUNDS,
    HORIZON,
    SMOKE_BOUNDS,
    Bounds,
    M5Data,
    load_m5,
    load_test_truth,
    origin_indices,
    sha256_file,
    window_indices,
)
from demand.data.official import official_reference, verify_official_dataset
from demand.evaluation.bootstrap import item_cluster_bootstrap
from demand.evaluation.inventory import (
    OVERAGE_FRAC,
    PRIMARY_FRACTILE,
    SENSITIVITY_FRACTILE,
    order_up_to,
    residual_quantile,
    simulate_inventory,
)
from demand.evaluation.verification import (
    check_consistency,
    check_official_scores,
    check_test_truth_integrity,
    check_weights,
)
from demand.evaluation.wrmsse import (
    WRMSSEResult,
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
    FeatureContext,
    build_feature_context,
    build_xy,
    cap_bound,
    predict_origin,
    refit_fixed_iterations,
    sample_origins,
    train_config,
)
from demand.models.statistical import BASELINE_BUILDERS

PROTOCOL = "m5-benchmark-v2"
SEEDS = (0, 1, 2)


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
    origin_idx: int,
    y_true: np.ndarray | None = None,
) -> WRMSSEResult:
    return evaluate_wrmsse(
        data,
        group_map,
        pred,
        origin_idx=origin_idx,
        y_true=y_true,
        horizon=HORIZON,
    )


def _fit_predict_baselines(
    data: M5Data, origins: dict[str, int], horizon: int
) -> dict[str, dict[str, np.ndarray]]:
    """Forecast every baseline at the validation and test origins."""
    panel = data.panel
    out: dict[str, dict[str, np.ndarray]] = {}
    for name, fn in BASELINE_BUILDERS.items():
        out[name] = {
            "validation": fn(panel, origins["validation"], horizon).astype(np.float32),
            "test": fn(panel, origins["test"], horizon).astype(np.float32),
        }
    return out


def _lgbm_select(
    data: M5Data,
    ctx: FeatureContext,
    bounds: Bounds,
    horizon: int,
    n_origins: int,
    group_map: dict[str, list[np.ndarray]],
    grid: list[dict[str, Any]],
) -> dict[str, Any]:
    """Grid selection on validation WRMSSE (pre-plan phase)."""
    splits = window_indices(bounds)
    origins = origin_indices(bounds)
    tr1 = splits["train"][1]
    va_origin = origins["validation"]

    # Origins whose 28 labels stay inside the train window.
    train_max_origin = tr1 - HORIZON
    train_origins = sample_origins(0, train_max_origin, n_origins)
    X_tr, y_tr = build_xy(
        data.panel, ctx, data, train_origins, horizon
    )
    X_es, y_es = build_xy(data.panel, ctx, data, np.array([va_origin]), horizon)
    assert y_tr is not None and y_es is not None

    grid_scores: list[dict[str, Any]] = []
    val_predictions: dict[int, np.ndarray] = {}
    for ci, cfg in enumerate(grid):
        model, best_it = train_config(X_tr, y_tr, X_es, y_es, cfg, SEED)
        pred_val = predict_origin(
            model, data.panel, ctx, data, va_origin, horizon
        )
        res = _wrmsse_at_origin(data, group_map, pred_val, va_origin)
        grid_scores.append(
            {
                "config": cfg,
                "best_iteration": best_it,
                "cap_bound": cap_bound(best_it),
                "val_wrmsse": res.wrmsse,
            }
        )
        val_predictions[ci] = pred_val
        del model
    del X_tr, y_tr, X_es, y_es

    best_i = int(np.argmin([g["val_wrmsse"] for g in grid_scores]))
    return {
        "grid": grid,
        "grid_scores": grid_scores,
        "best_config_index": best_i,
        "selected_config": grid_scores[best_i]["config"],
        "selected_iteration": grid_scores[best_i]["best_iteration"],
        "val_wrmsse_selected": grid_scores[best_i]["val_wrmsse"],
        "val_predictions": val_predictions,
        "n_origins": n_origins,
        "train_origins": train_origins,
        "train_max_origin": train_max_origin,
        "early_stop": EARLY_STOP,
        "max_rounds": MAX_ROUNDS,
    }


def _lgbm_refit(
    data: M5Data,
    ctx: FeatureContext,
    bounds: Bounds,
    horizon: int,
    n_origins: int,
    selected: dict[str, Any],
) -> dict[str, Any]:
    """Refit on train+val origins at the selected iteration, seeds 0/1/2."""
    splits = window_indices(bounds)
    origins = origin_indices(bounds)
    va1 = splits["validation"][1]
    te_origin = origins["test"]

    # Origins whose 28 labels stay inside train+validation.
    refit_max_origin = va1 - HORIZON
    all_origins = sample_origins(0, refit_max_origin, n_origins)
    X_all, y_all = build_xy(data.panel, ctx, data, all_origins, horizon)
    assert y_all is not None

    test_predictions = []
    for seed in SEEDS:
        model = refit_fixed_iterations(
            X_all,
            y_all,
            selected["selected_config"],
            selected["selected_iteration"],
            seed,
        )
        test_predictions.append(
            predict_origin(model, data.panel, ctx, data, te_origin, horizon)
        )
        del model
    del X_all, y_all
    return {
        "refit_origins": all_origins,
        "refit_max_origin": refit_max_origin,
        "test_predictions_per_seed": test_predictions,
        "test_prediction_mean": np.mean(
            np.stack(test_predictions), axis=0
        ).astype(np.float32),
    }


def _inventory_analysis(
    data: M5Data,
    bounds: Bounds,
    val_preds: dict[str, np.ndarray],
    test_preds: dict[str, np.ndarray],
    y_val: np.ndarray,
    y_test: np.ndarray,
    horizon: int,
) -> dict[str, Any]:
    """Newsvendor scenario at fractile 0.8 with a 0.95 sensitivity row."""
    splits = window_indices(bounds)
    va0, va1 = splits["validation"]
    te0 = va1 + 1  # first test day index
    price_test = data.price_days[:, te0 : te0 + horizon].astype(np.float64)

    out: dict[str, Any] = {}
    for name, test_pred in test_preds.items():
        val_pred = val_preds.get(name)
        if val_pred is None:
            continue
        entry: dict[str, Any] = {"overage_frac": OVERAGE_FRAC}
        for tag, fractile in (
            ("primary", PRIMARY_FRACTILE),
            ("sensitivity", SENSITIVITY_FRACTILE),
        ):
            q = residual_quantile(
                y_val, val_pred.astype(np.float64), fractile
            )
            levels = order_up_to(test_pred.astype(np.float64), q)
            entry[tag] = {
                "fractile": fractile,
                **simulate_inventory(y_test, levels, price_test),
            }
        out[name] = entry
    return out


def run_benchmark(
    data_dir: Path,
    official_dir: Path,
    out_dir: Path,
    *,
    n_origins: int = N_ORIGINS,
    n_boot: int = 500,
    git_sha: str | None = None,
    smoke: bool = False,
) -> dict[str, Any]:
    """Run the frozen protocol and write benchmark artifacts."""
    out_dir.mkdir(parents=True, exist_ok=True)
    timings: dict[str, float] = {}
    t_stage = _epoch()

    def mark(stage: str) -> None:
        nonlocal t_stage
        now = _epoch()
        timings[stage] = round(now - t_stage, 3)
        print(f"[stage] {stage}: {timings[stage]:.1f}s", flush=True)
        t_stage = now

    bounds = SMOKE_BOUNDS if smoke else FULL_BOUNDS
    data = load_m5(data_dir)
    mark("load_m5")
    competition_sha = {
        name: sha256_file(data_dir / name) for name in COMPETITION_FILES
    }
    official_sha = verify_official_dataset(official_dir)
    mark("verify_official")

    splits = window_indices(bounds)
    origins = origin_indices(bounds)
    horizon = HORIZON
    group_map = level_group_map(data.ids)

    weights_report = check_weights(data, official_dir)
    mark("weights_check")
    consistency_report = check_consistency(data, official_dir)
    mark("consistency_check")

    ctx = build_feature_context(data)
    mark("feature_context")

    baseline_preds = _fit_predict_baselines(data, origins, horizon)
    mark("baselines")
    grid = GRID[:1] if smoke else GRID
    lgbm_sel = _lgbm_select(
        data, ctx, bounds, horizon, n_origins, group_map, grid
    )
    mark("lgbm_grid")

    # Validation WRMSSE: baselines plus every grid config's prediction.
    va_origin = origins["validation"]
    val_results: dict[str, Any] = {}
    val_pred_map: dict[str, np.ndarray] = {}
    val_wrmsse_point: dict[str, float] = {}
    for name, preds in baseline_preds.items():
        res = _wrmsse_at_origin(data, group_map, preds["validation"], va_origin)
        val_results[name] = {"wrmsse": res.wrmsse, "per_level": res.per_level}
        val_pred_map[name] = preds["validation"]
        val_wrmsse_point[name] = res.wrmsse
    for ci, pred in lgbm_sel["val_predictions"].items():
        cfg = lgbm_sel["grid"][ci]
        key = f"lightgbm_{cfg['num_leaves']}x{cfg['learning_rate']}"
        res = _wrmsse_at_origin(data, group_map, pred, va_origin)
        val_results[key] = {
            "wrmsse": res.wrmsse,
            "per_level": res.per_level,
            "config": cfg,
        }
        val_wrmsse_point[key] = res.wrmsse
    selected_key = (
        f"lightgbm_{lgbm_sel['selected_config']['num_leaves']}"
        f"x{lgbm_sel['selected_config']['learning_rate']}"
    )
    val_results["lightgbm"] = val_results[selected_key]
    val_pred_map["lightgbm"] = lgbm_sel["val_predictions"][
        lgbm_sel["best_config_index"]
    ]
    val_wrmsse_point["lightgbm"] = lgbm_sel["val_wrmsse_selected"]
    mark("validation_wrmsse")

    # Freeze the plan before any test-window truth access.
    card_write_epoch = _epoch()
    plan = {
        "protocol": PROTOCOL,
        "seed": SEED,
        "seeds": list(SEEDS),
        "horizon": horizon,
        "bounds": {
            "train_end_day": bounds.train_end_day,
            "val_end_day": bounds.val_end_day,
            "test_start_day": bounds.test_start_day,
            "test_end_day": bounds.test_end_day,
        },
        "smoke_shifted": smoke,
        "n_origins": n_origins,
        "lgbm_grid": lgbm_sel["grid"],
        "lgbm_grid_scores": lgbm_sel["grid_scores"],
        "lgbm_early_stop": {"early_stop": EARLY_STOP, "max_rounds": MAX_ROUNDS},
        "lgbm_selected": {
            "config": lgbm_sel["selected_config"],
            "best_iteration": lgbm_sel["selected_iteration"],
            "cap_bound": cap_bound(lgbm_sel["selected_iteration"]),
            "val_wrmsse": lgbm_sel["val_wrmsse_selected"],
        },
        "primary_fractile": PRIMARY_FRACTILE,
        "sensitivity_fractile": SENSITIVITY_FRACTILE,
        "overage_frac": OVERAGE_FRAC,
        "n_bootstrap": n_boot,
        "smoke": smoke,
    }
    _write_json(out_dir / "analysis_plan.json", plan)
    mark("analysis_plan")

    # ---- Test truth: first and only official-truth access. ----
    if smoke:
        integrity = check_test_truth_integrity(data, official_dir)
        first_test_read_epoch = float(integrity["first_test_read_epoch"])
        # Pseudo-test truth comes from the panel (d_1914..d_1941).
        te0_panel = splits["validation"][1] + 1
        y_test = data.panel[:, te0_panel : te0_panel + horizon]
    else:
        y_test, first_test_read_epoch = load_test_truth(
            official_dir / "sales_test_evaluation.csv", data.ids
        )
    mark("test_truth")

    # Official score gate (full run only): Naive/sNaive vs the
    # published table; cheap stop before the LightGBM refit.
    verification: dict[str, Any] = {
        "weights": weights_report,
        "consistency": consistency_report,
    }
    if smoke:
        verification["test_truth_integrity"] = {
            "sha256": integrity["sha256"],
            "shape": integrity["shape"],
            "pass": integrity["pass"],
        }
        verification["official_scores"] = {
            "pass": None,
            "note": "skipped in smoke; pseudo-test is not the private window",
        }
    else:
        score_report = check_official_scores(
            data,
            group_map,
            {
                "naive": baseline_preds["naive"]["test"],
                "seasonal_naive": baseline_preds["seasonal_naive"]["test"],
            },
            origin_idx=origins["test"],
            y_true=y_test,
            official_dir=official_dir,
        )
        verification["official_scores"] = score_report
        verification["test_truth_sha256"] = official_sha[
            "sales_test_evaluation.csv"
        ]
        verification["pass"] = bool(
            weights_report["pass"]
            and consistency_report["pass"]
            and score_report["pass"]
        )
        _write_json(out_dir / "metric_verification.json", verification)
        if not score_report["pass"]:
            raise RuntimeError(
                "official WRMSSE check failed; see metric_verification.json"
            )
        mark("official_score_check")
    if smoke:
        _write_json(out_dir / "metric_verification.json", verification)

    # ---- Post-plan: refit and all test-window metrics. ----
    lgbm_refit = _lgbm_refit(
        data, ctx, bounds, horizon, n_origins, lgbm_sel
    )
    mark("lgbm_refit")

    te_origin = origins["test"]
    test_results: dict[str, Any] = {}
    test_pred_map: dict[str, np.ndarray] = {}
    test_wrmsse_results: dict[str, WRMSSEResult] = {}

    for name, preds in baseline_preds.items():
        pred = preds["test"]
        res = _wrmsse_at_origin(data, group_map, pred, te_origin, y_test)
        pm = point_metrics(y_test, pred)
        test_results[name] = {
            "wrmsse": res.wrmsse,
            "per_level": res.per_level,
            "point": pm,
        }
        test_pred_map[name] = pred
        test_wrmsse_results[name] = res
    mark("test_baselines")

    pred_mean = lgbm_refit["test_prediction_mean"]
    res = _wrmsse_at_origin(data, group_map, pred_mean, te_origin, y_test)
    pm = point_metrics(y_test, pred_mean)
    per_seed = []
    for seed, pred in zip(
        SEEDS, lgbm_refit["test_predictions_per_seed"], strict=True
    ):
        r = _wrmsse_at_origin(data, group_map, pred, te_origin, y_test)
        per_seed.append({"seed": seed, "wrmsse": r.wrmsse})
        del r
    test_results["lightgbm"] = {
        "wrmsse": res.wrmsse,
        "per_level": res.per_level,
        "point": pm,
        "per_seed": per_seed,
    }
    test_pred_map["lightgbm"] = pred_mean
    test_wrmsse_results["lightgbm"] = res
    mark("test_lightgbm")

    boot = item_cluster_bootstrap(
        data,
        group_map,
        test_pred_map,
        test_wrmsse_results,
        origin_idx=te_origin,
        y_true=y_test,
        horizon=horizon,
        n_boot=n_boot,
        seed=0,
        val_wrmsse={
            m: val_wrmsse_point[m]
            for m in ("naive", "seasonal_naive", "ma28", "lightgbm")
        },
    )
    for m, entry in boot["models"].items():
        test_results[m]["ci"] = {
            "mean": entry["mean"],
            "ci_low": entry["ci_low"],
            "ci_high": entry["ci_high"],
            "n_finite": entry["n_finite"],
            "point": entry["point"],
        }
    mark("bootstrap")

    # Aggregate-level comparison: SARIMA + Prophet vs LGBM bottom-up.
    agg_results: dict[str, Any] = {}
    for level in AGG_LEVELS:
        level_entry: dict[str, Any] = {}
        agg_tr, agg_names = aggregate_panel(
            data.panel[:, : te_origin + 1], data.ids, level
        )
        true_agg, _ = aggregate_panel(y_test, data.ids, level)
        lgbm_agg, _ = aggregate_panel(pred_mean, data.ids, level)
        for method in ("sarima", "prophet"):
            fc, _ = aggregate_forecasts(
                level, data.panel, data.ids, te_origin, horizon, method,
                data.calendar,
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
    mark("aggregate_models")

    # Inventory scenario: validation residual quantiles -> test sim.
    va0, va1 = splits["validation"]
    y_val = data.panel[:, va0 : va1 + 1].astype(np.float64)
    inv = _inventory_analysis(
        data, bounds, val_pred_map, test_pred_map, y_val,
        y_test.astype(np.float64), horizon,
    )
    mark("inventory")

    xlsx = official_dir / "scores_and_ranks.xlsx"
    ref = official_reference(xlsx)
    _write_json(out_dir / "official_reference.json", ref)
    mark("official_reference")

    metrics = {
        "protocol": PROTOCOL,
        "validation": val_results,
        "test": test_results,
        "paired_bootstrap": boot["paired"],
        "bootstrap": {"n_boot": n_boot, "seed": 0, "n_items": boot["n_items"]},
        "aggregate_test": agg_results,
        "inventory_scenario": inv,
        "smoke": smoke,
        "stage_timings_s": timings,
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
            "panel_days": data.n_days,
            "calendar_days": data.n_calendar_days,
            "files": list(COMPETITION_FILES),
            "files_sha256": competition_sha,
            "source": "kaggle m5-forecasting-accuracy",
        },
        "official_files_sha256": official_sha,
        "verification": verification,
        "config": plan,
        "stage_timings_s": timings,
        "artifacts": [
            "analysis_plan.json",
            "metrics.json",
            "metric_verification.json",
            "official_reference.json",
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
        c = r.get("ci")
        ci_txt = (
            f"[{c['ci_low']:.4f}, {c['ci_high']:.4f}]" if c else "n/a"
        )
        lines.append(
            f"| {name} | {r['wrmsse']:.4f} "
            f"| {ci_txt} "
            f"| {r['point']['mae']:.2f} | {r['point']['bias']:.2f} |"
        )
    lines.append("")
    return "\n".join(lines)
