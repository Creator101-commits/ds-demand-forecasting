# DS Demand Forecasting

[![Python Version][python-image]][python-url]
[![Build Status][build-image]][kaggle-runner-url]
[![Downloads Stats][downloads-image]][releases-url]

Forecasting demand for Walmart time series using M5 data as the backbone, enriched with macro signals from the Walmart weekly dataset.

This project provides a notebook-first, module-backed pipeline for data prep, feature engineering, model training, evaluation, and business-impact estimation.

## Latest Results

Latest successful Kaggle bundle run completed the LightGBM path end-to-end.

### Model Metrics (LightGBM)

| Split | RMSE | MAE | sMAPE | WRMSSE Proxy |
| --- | ---: | ---: | ---: | ---: |
| Validation | 93.35 | 53.49 | 18.15 | 0.6232 |
| Test | 98.41 | 55.99 | 22.35 | 0.6570 |

Note: Test MAPE can be unstable with near-zero actual demand; sMAPE and WRMSSE proxy are the primary reliable indicators here.

### Business Impact (LightGBM Run)

| Metric | Value |
| --- | ---: |
| Overstock Cost | 2,949.78 |
| Stockout Cost | 38,892.92 |
| Total Cost | 41,842.70 |
| Savings vs SARIMA | 0.00 |

Savings vs SARIMA is zero in this run because SARIMA predictions were not included in the exported bundle.

## Implemented Scope

- Data pipeline: M5 overlap window + Walmart national weekly macro merge
- Feature engineering: time, lag/rolling, price, event/SNAP, macro features
- Temporal split: train, validation (28d), test (28d)
- Models: SARIMA, Prophet, LightGBM (LightGBM path fully validated)
- Evaluation: RMSE, MAE, MAPE, sMAPE, WRMSSE proxy
- Business impact: overstock and stockout proxy cost estimation

## Project Layout

- `config/config.yaml`: runtime settings and model params
- `src/data`: loading, merging, splitting
- `src/features`: feature generation modules
- `src/models`: model train/predict modules
- `src/evaluation`: metrics and business impact logic
- `scripts`: runnable entrypoints
- `notebooks`: local and Kaggle runners

## Installation

### macOS/Linux

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Windows (PowerShell)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Usage Example

Run step-by-step:

```sh
python scripts/run_data_pipeline.py
python scripts/run_models.py
python scripts/run_evaluation.py
```

Or run end-to-end:

```sh
bash scripts/run_full_pipeline.sh
```

## Development Setup

```sh
pip install -r requirements.txt
python -m pytest tests -q
python -m compileall src
```

## Release History

- 1.0.0
  - Initial full project scaffold and modular pipeline
  - Kaggle runner notebook and packaging workflow
  - LightGBM baseline results and business impact reporting

## Notes

- Current default uses overlap-only dates from `config/config.yaml`.
- Walmart macro merge uses national weekly averages.
- LSTM and ensemble are planned as next-phase extensions.

## Meta

- Repository: [https://github.com/Creator101-commits/ds-demand-forecasting](https://github.com/Creator101-commits/ds-demand-forecasting)

## Contributing

1. Fork the repository.
2. Create a branch for your feature.
3. Commit changes with clear messages.
4. Push and open a pull request.

<!-- Markdown badge/link definitions -->
[python-image]: https://img.shields.io/badge/python-3.10%2B-blue.svg?style=flat-square
[python-url]: https://www.python.org/
[build-image]: https://img.shields.io/badge/build-kaggle%20runner-success.svg?style=flat-square
[kaggle-runner-url]: https://www.kaggle.com/code
[downloads-image]: https://img.shields.io/github/downloads/Creator101-commits/ds-demand-forecasting/total.svg?style=flat-square
[releases-url]: https://github.com/Creator101-commits/ds-demand-forecasting/releases
[repo-url]: https://github.com/Creator101-commits/ds-demand-forecasting
