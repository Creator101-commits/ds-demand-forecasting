"""Render image.png from curated results/metrics.json only.

Panels: validation vs test WRMSSE per model, the 12-level WRMSSE
breakdown for the best model, aggregate-level RMSSE comparison
(SARIMA/Prophet/LightGBM bottom-up), and LightGBM seed spread.
Requires Matplotlib; run with
`uv run --with matplotlib python scripts/render_figure.py`.
Set DEMAND_RESULTS_DIR to render from a different results directory.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
RESULTS = Path(os.environ.get("DEMAND_RESULTS_DIR", REPO / "results"))

MODEL_ORDER = ["naive", "seasonal_naive", "ma28", "lightgbm"]
MODEL_LABELS = {
    "naive": "Naive",
    "seasonal_naive": "Seasonal naive",
    "ma28": "MA28",
    "lightgbm": "LightGBM",
}
RUN_LABEL = "v2.1"
COLORS = {
    "naive": "#9e9e9e",
    "seasonal_naive": "#1976d2",
    "ma28": "#f9a825",
    "lightgbm": "#2e7d32",
    "sarima": "#d32f2f",
    "prophet": "#7b1fa2",
    "lightgbm_bottomup": "#2e7d32",
}
LEVEL_ORDER = [
    "total", "state", "store", "category", "department",
    "state_x_category", "state_x_department", "store_x_category",
    "store_x_department", "item", "item_x_state", "item_x_store",
]


def _load() -> dict:
    return json.loads((RESULTS / "metrics.json").read_text())


def _git_sha() -> str:
    manifest = RESULTS / "manifest.json"
    if manifest.exists():
        return str(json.loads(manifest.read_text()).get("git_sha", "unknown"))
    return "unknown"


def main() -> None:
    metrics = _load()
    val = metrics["validation"]
    test = metrics["test"]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(
        "M5 demand forecasting: official WRMSSE "
        f"(protocol {metrics['protocol']}, run {RUN_LABEL}, "
        f"git {_git_sha()[:7]})",
        fontsize=13,
    )

    # Panel 1: validation vs test WRMSSE per headline model.
    ax = axes[0, 0]
    models = [m for m in MODEL_ORDER if m in test]
    x = np.arange(len(models))
    width = 0.38
    for i, m in enumerate(models):
        v = val.get(m, {}).get("wrmsse", np.nan)
        t = test[m]["wrmsse"]
        ci = test[m]["ci"]
        ax.bar(
            x[i] - width / 2, v, width,
            color=COLORS[m], alpha=0.45,
            label="validation" if i == 0 else None,
        )
        ax.bar(
            x[i] + width / 2, t, width,
            color=COLORS[m],
            label="test" if i == 0 else None,
        )
        ax.errorbar(
            x[i] + width / 2, t,
            yerr=[[t - ci["ci_low"]], [ci["ci_high"] - t]],
            fmt="none", ecolor="black", capsize=3,
        )
        ax.text(
            x[i] + width / 2, ci["ci_high"], f"{t:.3f}",
            ha="center", va="bottom", fontsize=8,
        )
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS[m] for m in models], fontsize=9)
    ci_top = max(test[m]["ci"]["ci_high"] for m in models)
    ax.set_ylim(0, ci_top * 1.08)
    ax.set_ylabel("WRMSSE (lower is better)")
    ax.set_title("WRMSSE: validation vs frozen test")
    ax.legend(fontsize=8, loc="upper right")
    ax.text(
        0.5, -0.16,
        "light bars = validation, dark bars = test; "
        "error bars = 95% item-cluster bootstrap CI on test",
        transform=ax.transAxes, ha="center", fontsize=7, color="#444444",
    )

    # Panel 2: per-level WRMSSE for LightGBM vs seasonal naive on test.
    best = min(test, key=lambda m: test[m]["wrmsse"])
    ax = axes[0, 1]
    series = [("lightgbm", "LightGBM"), ("seasonal_naive", "seasonal naive")]
    width = 0.38
    for j, (key, label) in enumerate(series):
        per_level = test[key]["per_level"]
        vals = [per_level.get(lv, np.nan) for lv in LEVEL_ORDER]
        ax.bar(
            np.arange(len(LEVEL_ORDER)) + (j - 0.5) * width, vals, width,
            color=COLORS[key], label=label,
        )
    ax.set_xticks(range(len(LEVEL_ORDER)))
    ax.set_xticklabels([lv.replace("_x_", " x ") for lv in LEVEL_ORDER],
                       rotation=90, ha="right", fontsize=7)
    ax.set_ylabel("WRMSSE")
    ax.set_title("Per-level WRMSSE on test: LightGBM vs seasonal naive")
    ax.legend(fontsize=8)

    # Panel 3: aggregate-level RMSSE comparison.
    ax = axes[1, 0]
    agg = metrics.get("aggregate_test", {})
    methods = ["sarima", "prophet", "lightgbm_bottomup"]
    method_labels = ["SARIMA", "Prophet", "LightGBM\nbottom-up"]
    if agg:
        levels = list(agg.keys())
        x = np.arange(len(levels))
        width = 0.26
        for j, meth in enumerate(methods):
            vals = [agg[lv].get(meth, {}).get("rmsse_mean", np.nan)
                    for lv in levels]
            ax.bar(x + (j - 1) * width, vals, width,
                   color=COLORS[meth], label=method_labels[j])
        ax.set_xticks(x)
        ax.set_xticklabels(levels, fontsize=9)
        ax.set_ylabel("mean RMSSE (lower is better)")
        ax.set_title(
            "Aggregate series RMSSE (test)\n"
            "SARIMA vs Prophet vs LightGBM bottom-up",
            fontsize=11,
        )
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "aggregate results not present",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Aggregate series comparison")

    # Panel 4: LightGBM per-seed WRMSSE spread vs the seed-mean forecast.
    ax = axes[1, 1]
    seeds = test.get("lightgbm", {}).get("per_seed", [])
    if seeds:
        xs = [s["seed"] for s in seeds]
        ys = [s["wrmsse"] for s in seeds]
        ax.plot(xs, ys, "o", color=COLORS["lightgbm"])
        for xv, yv in zip(xs, ys, strict=True):
            ax.text(xv, yv, f"{yv:.3f}", ha="center", va="bottom",
                    fontsize=8)
        mean_val = float(test["lightgbm"]["wrmsse"])
        ax.axhline(mean_val, color="#555555", ls="--", lw=1)
        ax.annotate(
            f"seed-averaged forecast {mean_val:.3f}",
            xy=(xs[0] - 0.15, mean_val),
            xytext=(0, 4),
            textcoords="offset points",
            va="bottom",
            ha="left",
            fontsize=8,
            color="#555555",
        )
        ax.set_xlim(min(xs) - 0.5, max(xs) + 0.5)
        ax.margins(y=0.3)
        ax.set_xlabel("seed")
        ax.set_ylabel("test WRMSSE (single-seed forecast)")
        ax.set_title("LightGBM per-seed WRMSSE (test)")
        ax.set_xticks(xs)
    else:
        ax.text(0.5, 0.5, "seed data not present",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_title("LightGBM per-seed WRMSSE")

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = REPO / "image.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")

    provenance = {
        "source": "results/metrics.json",
        "protocol": metrics["protocol"],
        "run": RUN_LABEL,
        "git_sha": _git_sha(),
        "models_plotted": models,
        "best_test_model": best,
        "levels": LEVEL_ORDER,
    }
    (REPO / "scripts" / "figure-data.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
