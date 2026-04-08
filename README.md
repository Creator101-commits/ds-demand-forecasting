# Demand Forecasting V1

This repository contains a notebook-first, module-backed implementation for demand forecasting using M5 + Walmart macro features.

## Implemented Scope
- Data pipeline (M5 overlap window + Walmart national weekly macro merge)
- Feature engineering (time, lag/rolling, price, event, macro)
- Temporal split (train, validation 28d, test 28d)
- Models: SARIMA, Prophet, LightGBM
- Evaluation: RMSE, MAE, MAPE, sMAPE, WRMSSE proxy
- Business impact: overstock and stockout proxy cost comparison vs SARIMA baseline

## Project Layout
- `config/config.yaml`: runtime settings
- `src/data`: data loading, merge, split
- `src/features`: feature engineering modules
- `src/models`: model training and prediction
- `src/evaluation`: metric and business impact evaluation
- `scripts`: runnable entrypoints

## Quickstart
1. Create and activate a Python environment.
2. Install dependencies:
   - `pip install -r requirements.txt`
3. Run data pipeline:
   - `python scripts/run_data_pipeline.py`
4. Run models:
   - `python scripts/run_models.py`
5. Run evaluation:
   - `python scripts/run_evaluation.py`

Or run all in one:
- `bash scripts/run_full_pipeline.sh`

## Notes
- The current implementation uses overlap-only dates configured in `config/config.yaml`.
- Walmart macro merge uses national weekly averages as agreed.
- LSTM and ensemble are intentionally deferred to Phase 2.
