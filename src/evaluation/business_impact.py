from __future__ import annotations

import pandas as pd

from src.utils.config import load_config
from src.utils.logger import get_logger
from src.utils.paths import build_paths

logger = get_logger(__name__)



def run(config_path: str = "config/config.yaml") -> pd.DataFrame:
    config = load_config(config_path)
    paths = build_paths(config)

    holding_annual = float(config["business"]["holding_cost_annual_pct"])
    margin_pct = float(config["business"]["margin_pct"])

    pred_files = sorted(paths.prediction_dir.glob("*_predictions.parquet"))
    if not pred_files:
        raise FileNotFoundError("No prediction files found. Run model training first.")

    preds = pd.concat([pd.read_parquet(p) for p in pred_files], ignore_index=True)
    preds = preds[preds["split"] == "test"].copy()

    base = pd.read_parquet(paths.processed_dir / "modeling_with_splits.parquet")
    price_map = base[["store_id", "dept_id", "date", "sell_price_mean"]].drop_duplicates()
    price_map["sell_price_mean"] = price_map["sell_price_mean"].fillna(0.0)

    preds = preds.merge(price_map, on=["store_id", "dept_id", "date"], how="left")
    preds["sell_price_mean"] = preds["sell_price_mean"].fillna(0.0)

    preds["overstock_units"] = (preds["prediction"] - preds["sales"]).clip(lower=0)
    preds["stockout_units"] = (preds["sales"] - preds["prediction"]).clip(lower=0)

    # Assume 14 days average holding period for excess units inside the 28-day horizon.
    hold_factor = holding_annual * (14.0 / 365.0)
    preds["overstock_cost"] = preds["overstock_units"] * preds["sell_price_mean"] * hold_factor
    preds["stockout_cost"] = preds["stockout_units"] * preds["sell_price_mean"] * margin_pct
    preds["total_cost"] = preds["overstock_cost"] + preds["stockout_cost"]

    summary = (
        preds.groupby("model", as_index=False)
        .agg(
            overstock_cost=("overstock_cost", "sum"),
            stockout_cost=("stockout_cost", "sum"),
            total_cost=("total_cost", "sum"),
        )
        .sort_values("total_cost")
    )

    baseline_model = "sarima"
    if baseline_model in summary["model"].values:
        baseline_cost = float(summary.loc[summary["model"] == baseline_model, "total_cost"].iloc[0])
        summary["savings_vs_sarima"] = baseline_cost - summary["total_cost"]
    else:
        summary["savings_vs_sarima"] = 0.0

    out_path = paths.root / "outputs" / "metrics" / "business_impact.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_path, index=False)
    logger.info("Saved business impact summary to %s", out_path)
    return summary


if __name__ == "__main__":
    run()
