"""Official M5 evaluation files: verification and score parsing.

The organizers' private-window ground truth, official WRMSSE weight
files, and the published benchmark/leaderboard table live in the
private Kaggle dataset ``sreeharshak101/m5-official-evaluation``
(SOURCES.md inside records provenance). Every file's sha256 is pinned
here and verified before use; the manifest records the same hashes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from demand.data.m5 import sha256_file

OFFICIAL_FILES = {
    "sales_test_evaluation.csv": (
        "73773250dd7f7464f7a1bf3a54874dc7fc8cbb49885cd0bf1efe09bf48fd930d"
    ),
    "sales_test_validation.csv": (
        "964f8ff2c36ff8763fc1bfeb4753f081abf70923d0209e78bc4dd470d49e438a"
    ),
    "weights_evaluation.csv": (
        "b430c288d92a0e55e08b76c20aafb4c538469c7ba95e47ab768e37e385299497"
    ),
    "weights_validation.csv": (
        "c957ad6429e59ac148918ddca13656d71784957c38af19e3314cad5c931572ab"
    ),
    "scores_and_ranks.xlsx": (
        "a1284eb1fd42214ae784366564c9780c1cb40c0694340468e389c9fd46432529"
    ),
}

LEVEL_IDS = [f"Level{i}" for i in range(1, 13)]
LEVEL_NAMES = [f"L{i}" for i in range(1, 13)]
SCORE_TOL = 1e-5
WEIGHT_DOLLAR_TOL = 0.011
# The organizers' stored weight column is only self-consistent to
# ~2.2e-6 relative (weight vs Dollar_Sales/level_total recomputed from
# their own file); 1e-5 sits above that rounding noise while still
# catching window or price bugs, which produce O(1) diffs.
WEIGHT_REL_TOL = 1e-5


def verify_official_dataset(directory: Path) -> dict[str, str]:
    """Verify sha256 of every official file; returns name -> sha256."""
    out: dict[str, str] = {}
    for name, expected in OFFICIAL_FILES.items():
        path = directory / name
        if not path.exists():
            raise FileNotFoundError(f"missing official file {path}")
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"{name} sha256 mismatch: {actual} != {expected}")
        out[name] = actual
    return out


def _sheet_rows(path: Path, sheet: str) -> list[list[Any]]:
    import openpyxl  # noqa: PLC0415

    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    try:
        ws = wb[sheet]
        return [list(row) for row in ws.iter_rows(values_only=True)]
    finally:
        wb.close()


def _score_row(row: list[Any]) -> dict[str, float]:
    """Parse a row whose first cell is a label; L1..L12 then Avg."""
    vals = [float(v) for v in row[1:13]]
    return dict(zip(LEVEL_NAMES, vals, strict=True)) | {
        "Avg": float(row[13])
    }


def parse_scores_xlsx(path: Path) -> dict[str, Any]:
    """Parse the organizers' published scores workbook.

    Returns benchmark rows (Naive, sNaive, ES_bu, ...) and the top-50
    leaderboard (winner plus every ranked team).
    """
    bench_rows = _sheet_rows(path, "Accuracy-Benchmarks (AL)")
    benchmarks: dict[str, dict[str, float]] = {}
    for row in bench_rows[2:]:
        if row and row[0] is not None:
            benchmarks[str(row[0])] = _score_row(row)

    top_rows = _sheet_rows(path, "Accuracy-Top50 (AL)")
    teams = [
        {"team": str(row[0]), **_score_row(row)}
        for row in top_rows[2:]
        if row and row[0] is not None
    ]
    if len(teams) != 50:
        raise ValueError(f"expected 50 top teams, found {len(teams)}")
    return {"benchmarks": benchmarks, "top50": teams}


def official_reference(path: Path) -> dict[str, Any]:
    """Reference rows for results: ES_bu, winner, and 50th place."""
    parsed = parse_scores_xlsx(path)
    bench = parsed["benchmarks"]
    return {
        "note": "official published scores, not rerun here",
        "ES_bu": bench.get("ES_bu"),
        "winner": parsed["top50"][0],
        "fiftieth": parsed["top50"][-1],
        "naive": bench.get("Naive"),
        "seasonal_naive": bench.get("sNaive"),
    }


def official_targets(path: Path) -> dict[str, dict[str, float]]:
    """Official WRMSSE per level + Avg for Naive and sNaive."""
    parsed = parse_scores_xlsx(path)
    bench = parsed["benchmarks"]
    return {
        "naive": bench["Naive"],
        "seasonal_naive": bench["sNaive"],
    }


def compare_scores(
    ours: dict[str, dict[str, float]], targets: dict[str, dict[str, float]]
) -> dict[str, Any]:
    """Max abs diff per model over the 13 numbers (L1..L12, Avg)."""
    report: dict[str, Any] = {}
    for name, target in targets.items():
        got = ours[name]
        diffs = {
            k: float(got[k] - target[k])
            for k in list(LEVEL_NAMES) + ["Avg"]
        }
        max_abs = float(np.max(np.abs(list(diffs.values()))))
        report[name] = {
            "max_abs_diff": max_abs,
            "diffs": diffs,
            "pass": bool(max_abs <= SCORE_TOL),
        }
    report["pass"] = all(r["pass"] for r in report.values())
    return report
