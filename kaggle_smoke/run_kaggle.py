"""Smoke kernel: reduced-work diagnostic run of the M5 benchmark.

Reuses the archived main kernel script from the source dataset with
``DEMAND_SMOKE=1`` so the full pipeline (data load, baselines, LightGBM
grid, aggregate models, inventory scenario, frozen plan, test read) is
exercised cheaply before the full run. Smoke outputs are diagnostics
only and are never committed as results.
"""

import os
import runpy
import sys
from pathlib import Path

os.environ["DEMAND_SMOKE"] = "1"

candidates = [
    p
    for p in Path("/kaggle/input").rglob("run_kaggle.py")
    if p.parent.name == "kaggle"
]
if not candidates:
    raise SystemExit("main kernel script not found in the source dataset")
target = sorted(candidates)[0]
print(f"smoke: delegating to {target}", flush=True)
sys.argv = [str(target)]
runpy.run_path(str(target), run_name="__main__")
