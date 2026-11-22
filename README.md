# DS Demand Forecasting

A reproducible M5 demand-forecasting benchmark on the Kaggle
`m5-forecasting-accuracy` competition data, scored with the official
WRMSSE against the organizers' private-window ground truth, under a
frozen temporal protocol and verification gates, on a CPU Kaggle
kernel.

## Overview

This repository replaces an earlier pipeline that aggregated sales to
`dept_store_day`, restricted the window to overlap an unrelated weekly
dataset, reported a "WRMSSE proxy" instead of the official metric, and
omitted SARIMA from the reported run. Protocol `m5-benchmark-v2`:

- Uses `sales_train_evaluation.csv` (d_1..d_1941 for 30,490 series),
  `calendar.csv` (d_1..d_1969), and `sell_prices.csv` from the
  competition, plus a private Kaggle dataset
  `sreeharshak101/m5-official-evaluation` holding the organizers'
  evaluation files: `sales_test_evaluation.csv` (private-window truth
  d_1942..d_1969), `sales_test_validation.csv`, the two official weight
  files, and `scores_and_ranks.xlsx`. Every official file is pinned by
  sha256 and verified before use; no data is committed.
- Frozen split: train d_1..d_1913, validation d_1914..d_1941 (public
  leaderboard window, origin d_1913), test d_1942..d_1969 (private
  leaderboard window, origin d_1941). Validation selects the LightGBM
  grid config; `analysis_plan.json` is persisted before the test truth
  file is read for the first time (`ordering_ok` in `manifest.json`).
- Verification gates run before any test metric: our recomputed
  dollar-sales weights must reproduce both official weight files, the
  competition panel's d_1914..d_1941 columns must equal the organizers'
  `sales_test_validation.csv` exactly, and our Naive/sNaive WRMSSE must
  reproduce the published benchmark table within 1e-5 per level and on
  Avg (parsed from the XLSX in code, never transcribed). A failed gate
  stops the run before the LightGBM refit.
- Official WRMSSE: 12 aggregation levels (42,840 series), RMSSE scales
  from one-step naive errors after the first non-zero sale, and weights
  from cumulative dollar sales over the last 28 days before each
  forecast origin, normalized within each level and averaged equally
  over the 12 levels.
- Baselines: naive, seasonal naive (7), and 28-day mean.
- LightGBM: pooled (series, origin, horizon) rows with a direct
  28-day horizon, Tweedie objective, `deterministic: true`, 33
  origin-relative features (fresh lags/rolling stats/same-weekday,
  target-day price and calendar including SNAP, static categorical
  ids), a fixed 4-config grid chosen by validation WRMSSE with early
  stopping, then refit on train+val origins at the selected iteration
  for seeds 0, 1, 2 (mean forecast reported, per-seed spread
  included). A perturbation test proves features at an origin are
  unchanged after randomizing all later panel days.
- Aggregate comparison on test: SARIMA(0,1,1)x(0,1,1,7) and Prophet
  (real calendar dates) on the 14 level 1-3 series, reported as RMSSE
  per series alongside LightGBM bottom-up aggregation.
- Inventory scenario analysis: newsvendor order-up-to levels from
  validation residual quantiles at critical fractile 0.8 (underage
  cost = price, overage cost = 0.25 x price), with a 0.95 sensitivity
  row, simulated day-by-day on test. Labeled scenario analysis, not a
  measured saving.
- Test CIs come from a paired item-cluster bootstrap (500 replicates,
  seed 0): item counts are drawn multinomially and applied identically
  to every model; levels 1-9 are recomputed from count-weighted
  aggregates and levels 10-12 reuse item-weighted unit RMSSE.
- Smoke mode shifts the whole protocol back 28 days (train through
  d_1885, validation d_1886..d_1913, pseudo-test d_1914..d_1941 from the
  panel) with a reduced grid, few origins, and B=20; it runs the same
  gates except the official score check and never computes metrics on
  the private-window truth.

## Data

| Item | Value |
| :--- | :--- |
| Source | Kaggle `m5-forecasting-accuracy` competition |
| Series | 30,490 item-store series |
| Panel | d_1..d_1941 sales; calendar/prices d_1..d_1969 |
| Official files | private dataset `m5-official-evaluation`, sha256-pinned |

## Method

```text
load 3 competition CSVs + verify 5 official files (sha256)
-> weights + consistency gates -> baselines at val/test origins
-> LightGBM grid on train origins, early stop, val WRMSSE selects config
-> persist analysis_plan.json (card_write_epoch)
-> load private-window truth (first_test_read_epoch)
-> official Naive/sNaive score gate (stop on failure)
-> refit on train+val origins, seeds 0/1/2 -> test forecasts
-> official WRMSSE + item-cluster bootstrap CIs on test
-> SARIMA/Prophet aggregate comparison -> inventory scenario
-> manifest.json + report.md + official_reference.json
```

## Results

Pending: the CPU Kaggle benchmark (`ds-demand-forecasting-benchmark`)
has not completed yet. When it does, `results/` will hold unedited
kernel copies of `metrics.json`, `manifest.json`, `analysis_plan.json`,
`metric_verification.json`, `official_reference.json`, and `report.md`,
and `image.png` will be rendered from `results/` only.

## Evaluation

- Primary metric: official WRMSSE on the 28-day private test window
  (lower is better), with per-level breakdown and paired item-cluster
  bootstrap CI.
- Gates: official weight reproduction, competition-vs-organizer
  validation consistency, official Naive/sNaive table reproduction
  within 1e-5.
- Secondary: MAE, RMSE, bias over the horizon; per-seed LightGBM
  spread; aggregate RMSSE comparison; inventory scenario cost.
- Known limits: the bootstrap resamples the item population with
  stores and the single 28-day window fixed, so intervals reflect
  product-population sensitivity, not temporal uncertainty; Prophet
  requires the cmdstan backend on first fit; inventory numbers are
  scenario analysis under declared costs, not measured savings.

## Reproduction

```bash
uv sync
uv run pytest
scripts/kaggle_run.sh --smoke   # shifted-protocol diagnostic kernel
scripts/kaggle_run.sh           # full CPU benchmark
```

Kernel outputs download to `artifacts/kaggle/<timestamp>/`; curate
`benchmark/` into `results/` only after `manifest.git_sha` matches HEAD
and `ordering_ok` is true.

## Provenance

- Protocol: `m5-benchmark-v2`; seed 42; seeds {0,1,2} for the LightGBM
  refit; 500 item-cluster bootstrap replicates (seed 0); critical
  fractile 0.8 with a 0.95 sensitivity row; overage cost 25% of price.
- Kaggle kernels: `sreeharshak101/ds-demand-forecasting-benchmark` and
  `sreeharshak101/ds-demand-forecasting-smoke` (CPU).
- Source dataset `sreeharshak101/ds-demand-forecasting-src` is
  versioned from `git archive HEAD` by `scripts/kaggle_run.sh`;
  `run_info.json` records the git SHA and `manifest.json` echoes it.
- Official evaluation dataset `sreeharshak101/m5-official-evaluation`
  is private and static; `SOURCES.md` inside records file provenance.
  `results/official_reference.json` carries the organizers' published
  scores (ES_bu, winner, 50th place), labeled "official published
  score, not rerun here".
