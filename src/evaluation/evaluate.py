from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.metrics import mae, mape, rmse, smape, wrmsse_proxy
from src.utils.config import load_config
from src.utils.logger import get_logger
from src.utils.paths import build_paths

logger = get_logger(__name__)



def _load_prediction_files(prediction_dir: Path) -> pd.DataFrame:
    files = sorted(prediction_dir.glob("*_predictions.parquet"))
    if not files:
        raise FileNotFoundError("No prediction files found in prediction directory.")
    frames = [pd.read_parquet(p) for p in files]
    return pd.concat(frames, ignore_index=True)



def _compute_metric_row(model_df: pd.DataFrame, train_df: pd.DataFrame, split_name: str) -> dict[str, float | str]:
    y_true = model_df["sales"].to_numpy(dtype=float)
    y_pred = model_df["prediction"].to_numpy(dtype=float)

    train_series = train_df["sales"].to_numpy(dtype=float)

    return {
        "model": str(model_df["model"].iloc[0]),
        "split": split_name,
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "mape": mape(y_true, y_pred),
        "smape": smape(y_true, y_pred),
        "wrmsse_proxy": wrmsse_proxy(y_true, y_pred, train_series),
    }



def run(config_path: str = "config/config.yaml") -> pd.DataFrame:
    config = load_config(config_path)
    paths = build_paths(config)

    pred_df = _load_prediction_files(paths.prediction_dir)
    pred_df["date"] = pd.to_datetime(pred_df["date"])

    full_df = pd.read_parquet(paths.processed_dir / "modeling_with_splits.parquet")
    full_df["date"] = pd.to_datetime(full_df["date"])

    rows: list[dict[str, float | str]] = []

    for (model_name, split_name), g in pred_df.groupby(["model", "split"], sort=False):
        if split_name not in {"val", "test"}:
            continue

        related_train = full_df[
            (full_df["split"] == "train")
            & (full_df["store_id"].isin(g["store_id"].unique()))
            & (full_df["dept_id"].isin(g["dept_id"].unique()))
        ]
        if related_train.empty:
            related_train = full_df[full_df["split"] == "train"]

        row = _compute_metric_row(g, related_train, split_name)
        row["model"] = model_name
        rows.append(row)

    metrics_df = pd.DataFrame(rows).sort_values(["split", "smape", "rmse"])

    out_path = paths.root / "outputs" / "metrics" / "model_comparison.csv"
    metrics_df.to_csv(out_path, index=False)
    logger.info("Saved model comparison metrics to %s", out_path)

    return metrics_df


if __name__ == "__main__":
    run()
