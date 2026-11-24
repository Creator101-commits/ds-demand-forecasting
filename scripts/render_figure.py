"""Render image.png from the committed results/metrics.json only.

Panels: (a) validation vs frozen-test WRMSSE per model, (b) per-level test
WRMSSE for all four models across the 12 aggregation levels, (c) aggregate
RMSSE by state for SARIMA/Prophet/LightGBM bottom-up, (d) aggregate-level
mean RMSSE comparison, (e) LightGBM per-seed test WRMSSE spread.
Requires Matplotlib; run with
`uv run --with matplotlib python scripts/render_figure.py`.
Set DEMAND_RESULTS_DIR to render from a different results directory.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
RESULTS = Path(os.environ.get("DEMAND_RESULTS_DIR", REPO / "results"))
OUT = REPO / "image.png"

INK = "#1F2A37"
BLUE = "#3B6EA8"
TEAL = "#2A9D8F"
AMBER = "#E9A23B"
CORAL = "#D1495B"
GRAY = "#9AA5B1"
GRID = "#EEF0F2"
SPINE = "#B8BEC6"
TICK_LABEL = "#4A5361"

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["cmr10"],
        "mathtext.fontset": "cm",
        "axes.formatter.use_mathtext": True,
        "axes.unicode_minus": False,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8,
        "legend.frameon": False,
        "text.color": INK,
        "axes.labelcolor": INK,
        "axes.edgecolor": SPINE,
        "axes.linewidth": 0.8,
        "xtick.color": SPINE,
        "ytick.color": SPINE,
        "xtick.labelcolor": TICK_LABEL,
        "ytick.labelcolor": TICK_LABEL,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
    }
)

DPI = 150
MODEL_ORDER = ["naive", "seasonal_naive", "ma28", "lightgbm"]
MODEL_LABELS = {
    "naive": "Naive",
    "seasonal_naive": "Seasonal naive",
    "ma28": "MA28",
    "lightgbm": "LightGBM",
}
MODEL_COLORS = {
    "naive": GRAY,
    "seasonal_naive": TEAL,
    "ma28": AMBER,
    "lightgbm": BLUE,
}
AGG_METHODS = ["sarima", "prophet", "lightgbm_bottomup"]
AGG_LABELS = ["SARIMA", "Prophet", "LightGBM\nbottom-up"]
AGG_COLORS = [CORAL, AMBER, BLUE]
LEVEL_ORDER = [
    "total", "state", "store", "category", "department",
    "state_x_category", "state_x_department", "store_x_category",
    "store_x_department", "item", "item_x_state", "item_x_store",
]
LEVEL_LABELS = [
    "Total", "State", "Store", "Cat.", "Dept.", r"St.$\times$Cat.",
    r"St.$\times$Dept.", r"Sto.$\times$Cat.", r"Sto.$\times$Dept.",
    "Item", r"Item$\times$St.", r"Item$\times$Sto.",
]


def _fmt2(x: float, _pos: int) -> str:
    """Up to 2 decimals inside mathtext; large values as integers."""
    if x == 0:
        return "$0$"
    if abs(x) >= 1000:
        return f"${x:.0f}$"
    s = f"{x:.2f}".rstrip("0").rstrip(".")
    if s in ("", "-0"):
        return "$0$"
    return f"${s}$"


CATEGORICAL: list[tuple[plt.Axes, str, list[str]]] = []


def style(
    ax: plt.Axes,
    letter: str,
    title: str,
    xlabel: str | None = None,
    ylabel: str | None = None,
    grid: bool = True,
) -> None:
    ax.set_title(f"({letter}) {title}", loc="left", fontsize=11, color=INK)
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=10, color=INK)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=10, color=INK)
    for name in ("top", "right"):
        ax.spines[name].set_visible(False)
    for name in ("left", "bottom"):
        ax.spines[name].set_visible(True)
        ax.spines[name].set_color(SPINE)
        ax.spines[name].set_linewidth(0.8)
    ax.tick_params(
        axis="both",
        direction="out",
        length=3,
        width=0.6,
        color=SPINE,
        labelcolor=TICK_LABEL,
        labelsize=8.5,
    )
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(_fmt2))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_fmt2))
    if grid:
        ax.grid(axis="y", color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)


def cat(
    ax: plt.Axes,
    axis: str,
    positions: list[int] | range,
    labels: list[str],
    rotation: float = 0,
    ha: str | None = None,
    fontsize: float | None = None,
) -> None:
    """Categorical ticks: FixedLocator + FixedFormatter applied AFTER
    style(), so the numeric FuncFormatter cannot clobber the labels.
    Rotation/alignment/fontsize are set on the label Texts directly."""
    labels = list(labels)
    if axis == "x":
        ax.set_xticks(positions)
        ax.xaxis.set_major_locator(mticker.FixedLocator(positions))
        ax.xaxis.set_major_formatter(mticker.FixedFormatter(labels))
        ax.xaxis.set_minor_locator(mticker.NullLocator())
        texts = ax.get_xticklabels()
    else:
        ax.set_yticks(positions)
        ax.yaxis.set_major_locator(mticker.FixedLocator(positions))
        ax.yaxis.set_major_formatter(mticker.FixedFormatter(labels))
        ax.yaxis.set_minor_locator(mticker.NullLocator())
        texts = ax.get_yticklabels()
    for t in texts:
        if rotation:
            t.set_rotation(rotation)
        if ha:
            t.set_ha(ha)
        if fontsize:
            t.set_fontsize(fontsize)
    CATEGORICAL.append((ax, axis, labels))


def verify_categories() -> None:
    bad = []
    for ax, axis, expected in CATEGORICAL:
        got = [
            t.get_text()
            for t in (ax.get_xticklabels() if axis == "x" else ax.get_yticklabels())
        ]
        if got != expected:
            bad.append((axis, expected, got))
    CATEGORICAL.clear()
    if bad:
        details = "; ".join(f"{a} expected={e} got={g}" for a, e, g in bad)
        raise RuntimeError(f"categorical tick labels mismatch: {details}")


def _all_texts(fig: plt.Figure) -> list[plt.Text]:
    items = []
    for ax in fig.axes:
        cand = [ax.title, ax.xaxis.label, ax.yaxis.label]
        cand += list(ax.texts)
        for axis in (ax.xaxis, ax.yaxis):
            try:
                ticks = axis._update_ticks()  # noqa: SLF001
            except Exception:
                ticks = []
            for tick in ticks:
                lab = tick.label1
                if lab.get_text():
                    cand.append(lab)
        leg = ax.get_legend()
        if leg is not None:
            cand += list(leg.get_texts())
        for t in cand:
            try:
                if t.get_text() and t.get_visible():
                    items.append(t)
            except Exception:
                pass
    return items


def check_text_overlap(fig: plt.Figure) -> list[str]:
    fig.canvas.draw()
    fig.canvas.draw()
    ren = fig.canvas.get_renderer()
    W, H = fig.canvas.get_width_height()
    boxes = []
    for t in _all_texts(fig):
        bb = t.get_window_extent(ren)
        boxes.append((t.get_text()[:40], bb))
    problems = []
    for i, (s1, b1) in enumerate(boxes):
        if b1.x0 < -1 or b1.y0 < -1 or b1.x1 > W + 1 or b1.y1 > H + 1:
            problems.append(f"OUT-OF-FIGURE {s1!r} bbox={b1}")
        for s2, b2 in boxes[i + 1 :]:
            ix = min(b1.x1, b2.x1) - max(b1.x0, b2.x0)
            iy = min(b1.y1, b2.y1) - max(b1.y0, b2.y0)
            if ix > 1.5 and iy > 1.5:
                inter = ix * iy
                small = min(b1.width * b1.height, b2.width * b2.height)
                if inter > max(8.0, 0.10 * small):
                    problems.append(f"OVERLAP {s1!r} x {s2!r} area={inter:.0f}px2")
    return problems


def check_legend_data(fig: plt.Figure) -> list[str]:
    """Count data markers/vertices falling inside legend boxes."""
    fig.canvas.draw()
    fig.canvas.draw()
    ren = fig.canvas.get_renderer()
    out = []
    for ax in fig.axes:
        leg = ax.get_legend()
        if leg is None:
            continue
        lb = leg.get_window_extent(ren)
        covered = 0
        total = 0
        for coll in ax.collections:
            offs = coll.get_offsets()
            if len(offs) == 0 or len(offs) > 2000:
                continue
            pts = ax.transData.transform(offs)
            inside = (
                (pts[:, 0] > lb.x0)
                & (pts[:, 0] < lb.x1)
                & (pts[:, 1] > lb.y0)
                & (pts[:, 1] < lb.y1)
            ).sum()
            covered += int(inside)
            total += len(offs)
        for line in ax.lines:
            if line.get_transform() != ax.transData:
                continue
            try:
                pts = ax.transData.transform(line.get_path().vertices)
            except Exception:
                continue
            inside = (
                (pts[:, 0] > lb.x0)
                & (pts[:, 0] < lb.x1)
                & (pts[:, 1] > lb.y0)
                & (pts[:, 1] < lb.y1)
            ).sum()
            covered += int(inside)
            total += len(pts)
        for patch in ax.patches:
            try:
                pb = patch.get_window_extent(ren)
            except Exception:
                continue
            if pb.x0 < lb.x1 and pb.x1 > lb.x0 and pb.y0 < lb.y1 and pb.y1 > lb.y0:
                covered += 1
            total += 1
        if covered:
            out.append(f"legend covers ~{covered}/{total} data items")
    return out


def check_annotation_data(fig: plt.Figure) -> list[str]:
    """In-axes annotations vs plotted data."""
    fig.canvas.draw()
    fig.canvas.draw()
    ren = fig.canvas.get_renderer()
    out = []
    for ax in fig.axes:
        for t in ax.texts:
            if not t.get_text() or not t.get_visible():
                continue
            tb = t.get_window_extent(ren)
            hits = 0
            for line in ax.lines:
                if line.get_transform() != ax.transData:
                    continue
                pts = ax.transData.transform(line.get_path().vertices)
                hits += int(
                    (
                        (pts[:, 0] > tb.x0)
                        & (pts[:, 0] < tb.x1)
                        & (pts[:, 1] > tb.y0)
                        & (pts[:, 1] < tb.y1)
                    ).sum()
                )
            for coll in ax.collections:
                offs = coll.get_offsets()
                if len(offs) == 0 or len(offs) > 5000:
                    continue
                pts = ax.transData.transform(offs)
                hits += int(
                    (
                        (pts[:, 0] > tb.x0)
                        & (pts[:, 0] < tb.x1)
                        & (pts[:, 1] > tb.y0)
                        & (pts[:, 1] < tb.y1)
                    ).sum()
                )
            if hits:
                out.append(f"'{t.get_text()}' overlaps ~{hits} data pts")
    return out


def check_legend_pixels(fig: plt.Figure, path: Path) -> list[str]:
    """Saturated (colored) data pixels showing through legend boxes."""
    if not path.exists():
        return []
    img = plt.imread(path)
    H, W = img.shape[:2]
    rgb = img[..., :3]
    spread = rgb.max(axis=2) - rgb.min(axis=2)
    colored = spread > 0.18
    def _collect(o: object, swatches: list) -> None:
        if type(o).__name__ == "DrawingArea":
            swatches.append(o)
            return
        try:
            children = o.get_visible_children()
        except Exception:
            children = []
        for c in children:
            _collect(c, swatches)

    out = []
    for ax in fig.axes:
        leg = ax.get_legend()
        if leg is None:
            continue
        bb = leg.get_window_extent(fig.canvas.get_renderer())
        x0, x1 = int(bb.x0), min(int(bb.x1) + 1, W)
        y0, y1 = max(0, int(H - bb.y1)), min(H, int(H - bb.y0) + 1)
        if x1 <= x0 or y1 <= y0:
            continue
        region = colored[y0:y1, x0:x1].copy()
        # mask every legend handle swatch (any column) so only data-colored
        # bleed through the legend body counts; swatches live in
        # DrawingArea children of the legend's offset box
        swatches: list = []
        box = getattr(leg, "_legend_handle_box", None)
        if box is not None:
            _collect(box, swatches)
        for sw in swatches:
            try:
                hb = sw.get_window_extent(fig.canvas.get_renderer())
            except Exception:
                continue
            if not all(np.isfinite([hb.x0, hb.x1, hb.y0, hb.y1])):
                continue
            pad = 3
            hx0, hx1 = int(hb.x0) - x0 - pad, int(hb.x1) - x0 + pad
            hy0, hy1 = int(H - hb.y1) - y0 - pad, int(H - hb.y0) - y0 + pad
            rx0, rx1 = max(0, hx0), min(region.shape[1], hx1)
            ry0, ry1 = max(0, hy0), min(region.shape[0], hy1)
            if rx1 > rx0 and ry1 > ry0:
                region[ry0:ry1, rx0:rx1] = False
        frac = region.mean() if region.size else 0.0
        if frac > 0.015:
            out.append(f"legend body ~{frac:.1%} covered by data color")
    return out


def finish(fig: plt.Figure, out: Path) -> None:
    """Verify categorical labels, save, then fail on any check violation."""
    fig.canvas.draw()
    fig.canvas.draw()
    verify_categories()
    fig.savefig(out, dpi=DPI, facecolor="white")
    problems: list[str] = []
    problems += check_text_overlap(fig)
    problems += check_legend_data(fig)
    problems += check_annotation_data(fig)
    problems += check_legend_pixels(fig, out)
    plt.close(fig)
    if problems:
        for p in problems[:30]:
            print(f"  CHECK FAIL: {p}", file=sys.stderr)
        sys.exit(f"figure checks failed for {out.name}: {len(problems)} problem(s)")
    print("wrote", out)


def main() -> None:
    metrics = json.loads((RESULTS / "metrics.json").read_text())
    val = metrics["validation"]
    test = metrics["test"]
    models = [m for m in MODEL_ORDER if m in test]

    fig, axes = plt.subplots(2, 3, figsize=(13.0, 6.4), dpi=DPI, layout="constrained")
    (a1, a2, a3), (a4, a5, _a6) = axes

    # (a) Validation vs frozen-test WRMSSE per model.
    x = np.arange(len(models))
    width = 0.36
    for i, m in enumerate(models):
        v = val.get(m, {}).get("wrmsse", np.nan)
        t = test[m]["wrmsse"]
        ci = test[m]["ci"]
        a1.bar(x[i] - width / 2, v, width, color=MODEL_COLORS[m], alpha=0.4)
        a1.bar(x[i] + width / 2, t, width, color=MODEL_COLORS[m])
        a1.errorbar(
            x[i] + width / 2, t,
            yerr=[[t - ci["ci_low"]], [ci["ci_high"] - t]],
            fmt="none", ecolor=INK, capsize=3, elinewidth=0.8,
        )
        a1.text(
            x[i] + width / 2, ci["ci_high"], f"{t:.2f}",
            ha="center", va="bottom", fontsize=7.5, color=INK,
        )
    ci_top = max(test[m]["ci"]["ci_high"] for m in models)
    a1.set_ylim(0, ci_top * 1.12)
    style(a1, "a", "WRMSSE: validation vs frozen test",
          ylabel="WRMSSE (lower is better)", xlabel="model")
    cat(a1, "x", x, [MODEL_LABELS[m] for m in models])
    a1.bar(x[0] - width / 2, 0, width, color=GRAY, alpha=0.4,
           label="validation")
    a1.bar(x[0] + width / 2, 0, width, color=GRAY, label="test")
    a1.legend(loc="upper right")

    # (b) Per-level test WRMSSE for all four models.
    width = 0.2
    for j, m in enumerate(models):
        per_level = test[m]["per_level"]
        vals = [per_level.get(lv, np.nan) for lv in LEVEL_ORDER]
        a2.bar(
            np.arange(len(LEVEL_ORDER)) + (j - (len(models) - 1) / 2) * width,
            vals, width, color=MODEL_COLORS[m], label=MODEL_LABELS[m],
        )
    vmax_b = max(
        test[m]["per_level"].get(lv, 0.0) for m in models for lv in LEVEL_ORDER
    )
    a2.set_ylim(0, vmax_b * 1.42)
    style(a2, "b", "Per-level WRMSSE on test (12 aggregation levels)",
          ylabel="WRMSSE")
    cat(a2, "x", range(len(LEVEL_ORDER)), LEVEL_LABELS,
        rotation=75, ha="right", fontsize=7)
    a2.legend(loc="upper left")

    # (c) Aggregate RMSSE by state for SARIMA/Prophet/LightGBM bottom-up.
    agg_state = metrics.get("aggregate_test", {}).get("state", {})
    states = ["CA", "TX", "WI"]
    xs = np.arange(len(states))
    width = 0.26
    for j, meth in enumerate(AGG_METHODS):
        vals = [
            agg_state.get(meth, {}).get("rmsse_per_series", {}).get(st, np.nan)
            for st in states
        ]
        a3.bar(xs + (j - 1) * width, vals, width,
               color=AGG_COLORS[j], label=AGG_LABELS[j])
    vmax_c = max(
        agg_state.get(meth, {}).get("rmsse_per_series", {}).get(st, 0.0)
        for meth in AGG_METHODS for st in states
    )
    a3.set_ylim(0, vmax_c * 1.38)
    style(a3, "c", "Aggregate RMSSE by state (test)",
          ylabel="RMSSE (lower is better)", xlabel="state")
    cat(a3, "x", xs, states)
    a3.legend(loc="upper left")

    # (d) Aggregate-level mean RMSSE comparison (kept panel).
    agg = metrics.get("aggregate_test", {})
    levels = [lv for lv in ("total", "state", "store") if lv in agg]
    xl = np.arange(len(levels))
    for j, meth in enumerate(AGG_METHODS):
        vals = [
            agg[lv].get(meth, {}).get("rmsse_mean", np.nan) for lv in levels
        ]
        a4.bar(xl + (j - 1) * width, vals, width,
               color=AGG_COLORS[j], label=AGG_LABELS[j])
    vmax_d = max(
        agg[lv].get(meth, {}).get("rmsse_mean", 0.0)
        for lv in levels for meth in AGG_METHODS
    )
    a4.set_ylim(0, vmax_d * 1.38)
    style(a4, "d", "Aggregate mean RMSSE by level (test)",
          ylabel="mean RMSSE", xlabel="aggregation level")
    cat(a4, "x", xl, [lv.capitalize() for lv in levels])
    a4.legend(loc="upper left")

    # (e) LightGBM per-seed test WRMSSE spread.
    seeds = test.get("lightgbm", {}).get("per_seed", [])
    if seeds:
        sxs = [s["seed"] for s in seeds]
        sys_ = [s["wrmsse"] for s in seeds]
        a5.plot(sxs, sys_, "o", color=BLUE, ms=6)
        mean_val = float(test["lightgbm"]["wrmsse"])
        a5.axhline(mean_val, color=INK, ls="--", lw=1.0)
        a5.annotate(
            f"seed-averaged forecast {mean_val:.2f}",
            xy=(min(sxs), mean_val),
            xytext=(4, 0),
            textcoords="offset points",
            va="center",
            fontsize=8,
            color=TICK_LABEL,
        )
        for xv, yv in zip(sxs, sys_, strict=True):
            a5.text(xv, yv, f"{yv:.2f}", ha="center", va="bottom",
                    fontsize=7.5, color=INK)
        a5.set_xlim(min(sxs) - 0.6, max(sxs) + 0.6)
        a5.margins(y=0.35)
        a5.set_xticks(sxs)
        style(a5, "e", "LightGBM per-seed WRMSSE (test)",
              xlabel="seed", ylabel="test WRMSSE")
    else:
        a5.text(0.5, 0.5, "seed data not present",
                ha="center", va="center", transform=a5.transAxes)
        style(a5, "e", "LightGBM per-seed WRMSSE", grid=False)

    finish(fig, OUT)


if __name__ == "__main__":
    main()
