from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data.aggregate import run as run_aggregate
from src.data.merge import run as run_merge
from src.data.split import run as run_split
from src.features.builder import run as run_features



def main() -> None:
    run_merge()
    run_features()
    run_aggregate()
    run_split()


if __name__ == "__main__":
    main()
