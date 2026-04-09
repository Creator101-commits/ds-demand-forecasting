from __future__ import annotations

import numpy as np
import pandas as pd
import lightgbm as lgb

from src.utils.config import load_config
from src.utils.logger import get_logger
from src.utils.paths import build_paths

logger = get_logger(__name__)



def _prepare_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    target_col = "sales"
    drop_cols = {
        "sales",
        "date",
        "split",
        "d",
        "event_name_1",
        "event_name_2",
        "event_type_1",
        "event_type_2",
        "model",
        "prediction",
    }
    feature_cols = [c for c in df.columns if c not in drop_cols]

    X = df[feature_cols].copy()
    y = df[target_col].astype(float)

    obj_cols = X.select_dtypes(include=["object", "string"]).columns.tolist()
    X = pd.get_dummies(X, columns=obj_cols, dummy_na=True)
    X = X.fillna(0)

    keys = df[["store_id", "dept_id", "date", "split", "sales"]].copy()
    return X, y, keys



def train_and_predict(config_path: str = "config/config.yaml") -> pd.DataFrame:
    config = load_config(config_path)
    paths = build_paths(config)

    model_cfg = config["models"]["lightgbm"]

    data_path = paths.processed_dir / "modeling_with_splits.parquet"
    df = pd.read_parquet(data_path)
    df["date"] = pd.to_datetime(df["date"])

    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "val"].copy()
    test_df = df[df["split"] == "test"].copy()

    X_train, y_train, _ = _prepare_xy(train_df)
    X_val, y_val, val_keys = _prepare_xy(val_df)
    X_test, y_test, test_keys = _prepare_xy(test_df)

    X_val = X_val.reindex(columns=X_train.columns, fill_value=0)
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_val, label=y_val)

    params = {
        "objective": model_cfg["objective"],
        "metric": model_cfg["metric"],
        "learning_rate": model_cfg["learning_rate"],
        "num_leaves": model_cfg["num_leaves"],
        "min_child_samples": model_cfg["min_child_samples"],
        "subsample": model_cfg["subsample"],
        "colsample_bytree": model_cfg["colsample_bytree"],
        "reg_alpha": model_cfg["reg_alpha"],
        "reg_lambda": model_cfg["reg_lambda"],
        "verbosity": -1,
    }

    booster = lgb.train(
        params,
        train_data,
        valid_sets=[valid_data],
        num_boost_round=model_cfg["num_boost_round"],
        callbacks=[lgb.early_stopping(model_cfg["early_stopping_rounds"], verbose=False)],
    )

    val_pred = booster.predict(X_val, num_iteration=booster.best_iteration)
    test_pred = booster.predict(X_test, num_iteration=booster.best_iteration)
    val_pred = np.asarray(val_pred, dtype=float)
    test_pred = np.asarray(test_pred, dtype=float)

    val_out = val_keys.copy()
    val_out["prediction"] = val_pred
    val_out["model"] = "lightgbm"

    test_out = test_keys.copy()
    test_out["prediction"] = test_pred
    test_out["model"] = "lightgbm"

    pred_df = pd.concat([val_out, test_out], ignore_index=True)
    out_path = paths.prediction_dir / "lightgbm_predictions.parquet"
    pred_df.to_parquet(out_path, index=False)

    model_path = paths.root / "outputs" / "checkpoints" / "lightgbm_model.txt"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(model_path))

    imp = pd.DataFrame({
        "feature": X_train.columns,
        "importance": booster.feature_importance(importance_type="gain"),
    }).sort_values("importance", ascending=False)
    imp_path = paths.root / "outputs" / "metrics" / "lightgbm_feature_importance.csv"
    imp_path.parent.mkdir(parents=True, exist_ok=True)
    imp.to_csv(imp_path, index=False)

    logger.info("Saved LightGBM predictions: %s", out_path)
    return pred_df


if __name__ == "__main__":
    train_and_predict()
