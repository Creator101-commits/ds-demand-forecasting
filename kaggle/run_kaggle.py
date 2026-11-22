"""Kaggle entrypoint for the M5 demand forecasting benchmark (CPU).

Installs ``demand`` from the pinned source dataset uploaded by
``scripts/kaggle_run.sh``, locates the competition CSVs under
``/kaggle/input/m5-forecasting-accuracy`` and the official evaluation
files under ``/kaggle/input/m5-official-evaluation``, runs the frozen
benchmark, and writes ``benchmark/`` artifacts including
``manifest.json`` with the git SHA. ``DEMAND_SMOKE=1`` shifts the
protocol back 28 days and reduces work for a diagnostic run that never
computes metrics on the official test truth.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

KAGGLE_INPUT = Path("/kaggle/input")
WORK = Path("/kaggle/working")
BENCH = WORK / "benchmark"
M5_FILES = {
    "sales_train_evaluation.csv",
    "calendar.csv",
    "sell_prices.csv",
}
OFFICIAL_MARKER = "scores_and_ranks.xlsx"


def _run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def _find_project_root() -> Path | None:
    for candidate in sorted(KAGGLE_INPUT.rglob("pyproject.toml")):
        if (candidate.parent / "src" / "demand").is_dir():
            return candidate.parent
    return None


def _install_package() -> Path:
    proj_root = _find_project_root()
    print(f"project root: {proj_root}", flush=True)
    if proj_root is None:
        raise FileNotFoundError(
            "No installable project found under /kaggle/input "
            "(expected pyproject.toml + src/demand)."
        )
    proj = WORK / "project"
    shutil.copytree(proj_root, proj)
    _run([sys.executable, "-m", "pip", "install", "--quiet", str(proj)])
    return proj


def _git_sha(proj: Path) -> str:
    info = proj / "run_info.json"
    if info.exists():
        return str(json.loads(info.read_text()).get("git_sha", "unknown"))
    return "unknown"


def _find_m5_dir() -> Path:
    for candidate in sorted(KAGGLE_INPUT.iterdir()):
        if candidate.is_dir() and M5_FILES.issubset(
            p.name for p in candidate.iterdir()
        ):
            return candidate
    matches = [
        p
        for p in KAGGLE_INPUT.rglob("sales_train_evaluation.csv")
        if (p.parent / "calendar.csv").exists()
    ]
    if len(matches) != 1:
        raise FileNotFoundError(
            f"expected the m5-forecasting-accuracy input dir, found {matches}"
        )
    return matches[0].parent


def _find_official_dir() -> Path:
    matches = [
        p.parent for p in KAGGLE_INPUT.rglob(OFFICIAL_MARKER)
    ]
    if len(matches) != 1:
        raise FileNotFoundError(
            f"expected the m5-official-evaluation input dir, found {matches}"
        )
    return matches[0]


def _ensure_cmdstan() -> None:
    """Prophet's cmdstan backend needs binaries on first use."""
    try:
        import cmdstanpy  # noqa: PLC0415
    except ImportError:
        return
    try:
        cmdstanpy.cmdstan_path()
        return
    except Exception:
        pass
    print("installing cmdstan for prophet", flush=True)
    cmdstanpy.install_cmdstan()


def main() -> None:
    proj = _install_package()
    sha = _git_sha(proj)
    print(f"git_sha={sha}", flush=True)

    from demand.experimentation.benchmark import run_benchmark  # noqa: PLC0415

    data_dir = _find_m5_dir()
    official_dir = _find_official_dir()
    print(f"m5 data dir: {data_dir}", flush=True)
    print(f"official dir: {official_dir}", flush=True)
    _ensure_cmdstan()

    smoke = bool(os.environ.get("DEMAND_SMOKE"))
    if smoke:
        print("SMOKE MODE: shifted bounds, 1 config, n_origins=6, B=20", flush=True)

    BENCH.mkdir(parents=True, exist_ok=True)
    result = run_benchmark(
        data_dir,
        official_dir,
        BENCH,
        n_origins=6 if smoke else 40,
        n_boot=20 if smoke else 500,
        git_sha=sha,
        smoke=smoke,
    )

    metrics = result["metrics"]
    manifest = result["manifest"]
    print("=== FROZEN ===", flush=True)
    print(
        json.dumps(
            {
                "ordering_ok": manifest["ordering_ok"],
                "n_series": manifest["data"]["n_series"],
            },
            indent=2,
        ),
        flush=True,
    )
    for name, r in metrics["test"].items():
        print(f"test {name} wrmsse: {r['wrmsse']:.4f}", flush=True)


if __name__ == "__main__":
    main()
