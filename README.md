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

Primary run: **v2.1**, git `2a0a337`, protocol `m5-benchmark-v2`,
Kaggle CPU kernel `sreeharshak101/ds-demand-forecasting-benchmark`.
`results/` holds unedited kernel copies of `metrics.json`,
`manifest.json`, `analysis_plan.json`, `metric_verification.json`,
`official_reference.json`, and `report.md`; `image.png` renders from
`results/` only. Every verification gate passed: weights reproduce
both official files (max relative weight diff 2.2e-6 over 42,840
units), the competition panel equals `sales_test_validation.csv`
exactly (853,720 values, max diff 0.0), our Naive/sNaive WRMSSE
reproduces the published table within ~4e-15 on all 13 numbers, the
test-truth sha256 matches, and `ordering_ok` is true.

### Test WRMSSE (private window d_1942..d_1969)

| Model | WRMSSE | 95% CI | MAE | Bias |
| :--- | ---: | :--- | ---: | ---: |
| Naive | 1.7520 | [1.6267, 1.8540] | 1.47 | +0.34 |
| Seasonal naive | 0.8470 | [0.8225, 0.9035] | 1.32 | -0.02 |
| MA28 | 0.9462 | [0.9147, 0.9795] | 1.12 | 0.00 |
| LightGBM | **0.6211** | [0.5958, 0.6944] | 1.10 | +0.04 |

LightGBM's 0.6211 is the WRMSSE of the **3-seed averaged forecast**
(seeds 0, 1, 2 refit on train+validation origins at the selected 444
iterations and averaged before scoring). The single-seed forecasts
score 0.6237, 0.6244, and 0.6167 (mean 0.6216); the averaged forecast
is what enters the table, the bootstrap, and the inventory scenario.

Grid selection on validation WRMSSE (early stopping monitors
LightGBM's internal L2 metric on the pooled validation rows, not
WRMSSE): 63x0.08 -> 0.7184 (iter 176), **63x0.03 -> 0.7152 (iter
444, selected)**, 255x0.08 -> 0.7406 (iter 204), 255x0.03 -> 0.7477
(iter 361). All four finished below the cap (`cap_bound=false`,
`MAX_ROUNDS=1500`, `EARLY_STOP=50`, recorded per config in
`analysis_plan.json` and the manifest). Training sets
`deterministic: true` with fixed seeds, so runs on the same data and
hardware reproduce; the v2/v2.1 validation difference below is the
stopping point, not run-to-run noise.

Paired item-cluster bootstrap differences vs seasonal naive
(500 replicates, seed 0; negative favors the first model):

| Contrast | Mean diff | 95% CI |
| :--- | ---: | :--- |
| LightGBM - seasonal naive | -0.2246 | [-0.2848, -0.1630] |
| MA28 - seasonal naive | +0.0841 | [+0.0466, +0.1207] |
| Naive - seasonal naive | +0.8679 | [+0.7612, +0.9853] |

LightGBM beats seasonal naive on every one of the 12 aggregation
levels (per-level WRMSSE, test):

| Level | LightGBM | Seasonal naive |
| :--- | ---: | ---: |
| total | 0.320 | 0.560 |
| state | 0.437 | 0.673 |
| store | 0.523 | 0.718 |
| category | 0.403 | 0.623 |
| department | 0.507 | 0.708 |
| state x category | 0.523 | 0.760 |
| state x department | 0.612 | 0.829 |
| store x category | 0.600 | 0.801 |
| store x department | 0.690 | 0.888 |
| item | 0.992 | 1.223 |
| item x state | 0.949 | 1.205 |
| item x store | 0.897 | 1.176 |

Aggregate-level RMSSE on test (14 coarse series, lower is better):

| Level | SARIMA | Prophet | LightGBM bottom-up |
| :--- | ---: | ---: | ---: |
| total | 0.465 | 0.373 | **0.320** |
| state | 0.601 | 0.531 | **0.450** |
| store | 0.612 | 0.617 | **0.515** |

For context only, the organizers' published private-leaderboard
averages in `official_reference.json` ("official published score,
not rerun here") are: winner 0.5204, 50th place 0.5760, ES_bu
0.6710. This benchmark's LightGBM average of 0.6211 sits between the
published 50th-place and ES_bu scores.

The item-cluster bootstrap intervals are right-skewed: each point
estimate sits near its lower bound (LightGBM 0.6211 in
[0.5958, 0.6944]). This is a property of the method. Each replicate
redraws the 30,490-item population with replacement, which changes
the composition of every aggregation level and therefore both the
squared errors and the RMSSE scales inside the WRMSSE ratio.
Resampled populations are noisier mixtures than the full population,
so the replicate distribution is skewed and percentile intervals need
not be symmetric around the point estimate. Read them as
population-sensitivity bands, not symmetric error bars.

### v2 as-run disclosure

The first full run (**v2 as-run**, artifacts
`artifacts/kaggle/20261121-223604`, git `65250f1`) passed every gate
but had a model-selection defect: the selected LightGBM config
(63 leaves, lr 0.03) recorded `best_iteration` 399 against
`MAX_ROUNDS` 400 with `EARLY_STOP` 50, so training hit the round cap
before early stopping could fire and the cap bound selection. v2.1
raised `MAX_ROUNDS` to 1500 (early stop unchanged) on that
validation-only diagnostic; the grid, data, seeds, and selection rule
are unchanged and the same config was re-selected. The rerun
decision was made **after v2's test numbers had already been
computed**; the justification is the validation-only cap diagnostic,
not the test outcome. On test the uncapped model is not better:

| Model | v2 test WRMSSE | v2 95% CI | v2.1 test WRMSSE | v2.1 95% CI |
| :--- | ---: | :--- | ---: | ---: |
| Naive | 1.7520 | [1.6267, 1.8540] | 1.7520 | [1.6267, 1.8540] |
| Seasonal naive | 0.8470 | [0.8225, 0.9035] | 0.8470 | [0.8225, 0.9035] |
| MA28 | 0.9462 | [0.9147, 0.9795] | 0.9462 | [0.9147, 0.9795] |
| LightGBM | 0.6188 | [0.5940, 0.6912] | 0.6211 | [0.5958, 0.6944] |

The deterministic baselines are bit-identical across the two runs;
only the LightGBM number moved (+0.0024, well inside the overlapping
intervals). v2.1's slightly worse validation WRMSSE (0.7152 at
iteration 444 vs v2's 0.7124 at 399) is expected: early stopping
monitors L2 on pooled validation rows, not WRMSSE, so the
L2-optimal stopping point need not minimize WRMSSE, and the two runs
stopped at different iterations.

### Inventory scenario (not a measured saving)

Newsvendor order-up-to levels from validation residual quantiles,
simulated day-by-day on the test window under declared costs
(underage = price, overage = 0.25 x price). Primary critical fractile
0.8, sensitivity 0.95:

| Model | Fractile | Total cost | Stockout cost | Overage cost | Lost units | Cost/series-day |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| Naive | 0.80 | 2,335,976 | 1,215,571 | 1,120,406 | 369,264 | 2.736 |
| Naive | 0.95 | 2,600,855 | 653,509 | 1,947,347 | 201,275 | 3.046 |
| Seasonal naive | 0.80 | 2,039,162 | 934,921 | 1,104,242 | 283,969 | 2.389 |
| Seasonal naive | 0.95 | 2,414,038 | 375,137 | 2,038,901 | 123,665 | 2.828 |
| MA28 | 0.80 | 1,841,609 | 945,609 | 896,000 | 292,390 | 2.157 |
| MA28 | 0.95 | 2,196,763 | 400,662 | 1,796,101 | 132,701 | 2.573 |
| LightGBM | 0.80 | 1,750,943 | 857,598 | 893,346 | 260,725 | 2.051 |
| LightGBM | 0.95 | 2,143,410 | 351,549 | 1,791,861 | 115,342 | 2.511 |

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
  product-population sensitivity, not temporal uncertainty, and the
  percentile intervals are right-skewed (point estimates sit near the
  lower bound; see Results); Prophet requires the cmdstan backend on
  first fit; inventory numbers are scenario analysis under declared
  costs, not measured savings.

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
- LightGBM grid: `MAX_ROUNDS=1500`, `EARLY_STOP=50`; every grid config
  records `best_iteration` and a `cap_bound` flag
  (`best_iteration >= MAX_ROUNDS - EARLY_STOP`) in
  `analysis_plan.json` and the manifest. Curated run is v2.1 (git
  `2a0a337`); the v2 as-run (git `65250f1`, round cap bound) is kept
  at `artifacts/kaggle/20261121-223604/` for disclosure.
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
