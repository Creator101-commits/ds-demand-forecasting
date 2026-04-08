from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ProjectPaths:
    root: Path
    m5_dir: Path
    walmart_dir: Path
    processed_dir: Path
    prediction_dir: Path
    output_dir: Path



def build_paths(config: dict[str, Any], root: str | Path | None = None) -> ProjectPaths:
    root_path = Path(root) if root is not None else Path(__file__).resolve().parents[2]
    path_cfg = config["paths"]

    m5_dir = root_path / path_cfg["m5_dir"]
    walmart_dir = root_path / path_cfg["walmart_dir"]
    processed_dir = root_path / path_cfg["processed_dir"]
    prediction_dir = root_path / path_cfg["prediction_dir"]
    output_dir = root_path / path_cfg["output_dir"]

    for p in (processed_dir, prediction_dir, output_dir):
        p.mkdir(parents=True, exist_ok=True)

    return ProjectPaths(
        root=root_path,
        m5_dir=m5_dir,
        walmart_dir=walmart_dir,
        processed_dir=processed_dir,
        prediction_dir=prediction_dir,
        output_dir=output_dir,
    )
