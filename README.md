# DS Demand Forecasting

A reproducible M5 demand-forecasting benchmark on the Kaggle
`m5-forecasting-accuracy` competition data, using the official WRMSSE
metric, a frozen temporal protocol, and a CPU Kaggle kernel.

## Overview

This repository replaces an earlier pipeline that aggregated sales to
`dept_store_day`, restricted the window to overlap an unrelated weekly
dataset, reported a "WRMSSE proxy" instead of the official metric, and
omitted SARIMA from the reported run. The new protocol:

- Uses the competition files directly: `sales_train_validation.csv`
  (d_1..d_1913), `sales_train_evaluation.csv` (d_1914..d_1941),
  `calendar.csv`, and `sell_prices.csv`, mounted via
  `competition_sources`. No data is committed.
- Keeps the official item-store panel (30,490 series x 1,941 days); the
  only aggregation is inside the WRMSSE evaluation itself.
- Freezes the temporal split before modeling: train d_1..d_1885,
  validation d_1886..d_1913, test d_1914..d_1941. Validation selects the
  LightGBM grid config; the test window is read once after
  `analysis_plan.json` is persisted (`ordering_ok` in `manifest.json`).
- Implements official WRMSSE: 12 aggregation levels (42,840 series),
  RMSSE scales from one-step naive errors after the first non-zero sale,
  and weights from cumulative dollar sales (units x weekly sell price)
  over the last 28 days before each forecast origin.
- Baselines: naive, seasonal naive (7), and 28-day mean, forecast at
  both origins.
- LightGBM: pooled (series, origin, horizon) rows with a direct 28-day
  horizon, Tweedie objective, features limited to lags >= 28 plus
  calendar/price attributes of the target day. A fixed 4-config grid is
  chosen by validation WRMSSE with early stopping, then refit on
  train+val origins for test at seeds 0, 1, 2 (mean forecast reported,
  per-seed spread included).
- Aggregate comparison on test: SARIMA(0,1,1)x(0,1,1,7) and Prophet on
  the 14 level 1-3 series (total, 3 states, 10 stores), reported as RMSSE
  per series alongside LightGBM bottom-up aggregation.
- Inventory scenario analysis: newsvendor order-up-to levels from
  validation residual quantiles at service level 0.95, simulated
  day-by-day on test with declared costs (holding 25% of price,
  stockout = price). Labeled scenario analysis, not a measured saving.
- Test CIs come from 1,000 bootstrap resamples over the 42,840 evaluated
  WRMSSE units (seed 0).

## Data

| Item | Value |
| :--- | :--- |
| Source | Kaggle `m5-forecasting-accuracy` competition |
| Series | 30,490 item-store series |
| Days | d_1..d_1941 |
| Files | `sales_train_validation.csv`, `sales_train_evaluation.csv`, `calendar.csv`, `sell_prices.csv` |

## Method

```text
load 4 competition CSVs -> panel (n_series x n_days) + price-per-day
-> frozen splits -> baselines at val/test origins
-> LightGBM grid on train origins, early stop, val WRMSSE selects config
-> persist analysis_plan.json (card_write_epoch)
-> refit on train+val origins, seeds 0/1/2 -> test forecasts
-> official WRMSSE + bootstrap CIs on test -> SARIMA/Prophet aggregate
   comparison -> inventory scenario -> manifest.json + report.md
```

## Results

Pending: the CPU Kaggle benchmark (`ds-demand-forecasting-benchmark`)
has not completed yet. When it does, `results/` will hold unedited kernel
copies of `metrics.json`, `manifest.json`, `analysis_plan.json`, and
`report.md`, and `image.png` will be rendered from `results/` only.

## Evaluation

- Primary metric: official WRMSSE on the 28-day test window (lower is
  better), with per-level breakdown and bootstrap CI.
- Secondary: MAE, RMSE, bias over the horizon; per-seed LightGBM spread.
- Aggregate check: RMSSE of SARIMA/Prophet vs LightGBM bottom-up on the
  14 coarse series.
- Known limits: WRMSSE bootstrap resamples evaluated units i.i.d. and
  ignores hierarchy dependence; Prophet requires the cmdstan backend on
  first fit; inventory numbers are scenario analysis under declared
  costs, not measured savings.

## Reproduction

```bash
uv sync
uv run pytest
scripts/kaggle_run.sh --smoke   # reduced diagnostic kernel
scripts/kaggle_run.sh           # full CPU benchmark
```

Kernel outputs download to `artifacts/kaggle/<timestamp>/`; curate
`benchmark/` into `results/` only after `manifest.git_sha` matches HEAD
and `ordering_ok` is true.

## Provenance

- Protocol: `m5-benchmark-v1`; seed 42; seeds {0,1,2} for the LightGBM
  refit; 1,000 bootstrap resamples (seed 0); service level 0.95;
  holding cost 25% of price.
- Kaggle kernels: `sreeharshak101/ds-demand-forecasting-benchmark` and
  `sreeharshak101/ds-demand-forecasting-smoke` (CPU).
- Source dataset `sreeharshak101/ds-demand-forecasting-src` is versioned
  from `git archive HEAD` by `scripts/kaggle_run.sh`; `run_info.json`
  records the git SHA and `manifest.json` echoes it.
