from __future__ import annotations

from src.utils.config import load_config
from src.utils.logger import get_logger
from src.utils.paths import build_paths
import pandas as pd

logger = get_logger(__name__)



def run(config_path: str = "config/config.yaml") -> pd.DataFrame:
    config = load_config(config_path)
    paths = build_paths(config)

    base_path = paths.processed_dir / "featured_dept_store_daily.parquet"
    if not base_path.exists():
        raise FileNotFoundError(f"Expected feature dataset not found: {base_path}")

    df = pd.read_parquet(base_path)
    out_path = paths.processed_dir / "modeling_dept_store_daily.parquet"
    df.to_parquet(out_path, index=False)
    logger.info("Saved modeling dataset to %s with shape=%s", out_path, df.shape)
    return df


if __name__ == "__main__":
    run()
