from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from scripts.run_data_pipeline import main as run_data
from src.evaluation.run_all import run as run_eval
from src.models.run_all import run as run_models



def main() -> None:
    run_data()
    run_models()
    run_eval()


if __name__ == "__main__":
    main()
