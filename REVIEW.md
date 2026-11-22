# Review

## What changed vs the legacy pipeline

- Data: the old pipeline aggregated to `dept_store_day` and clipped the
  window to overlap an unrelated Walmart weekly dataset. The new code
  keeps all 30,490 item-store series over d_1..d_1941 and uses only the
  competition's own calendar and sell prices.
- Metric: `wrmsse_proxy` (a weighted variant on the aggregated panel)
  is replaced by the official WRMSSE: 12 levels, per-level scales from
  the first non-zero sale, dollar-sales weights over the last 28 days.
- Models: SARIMA(0,1,1)x(0,1,1,7) and Prophet are now reported on the
  14 level 1-3 series; LightGBM uses a pooled direct-28 horizon with
  lags >= 28 and a validation-WRMSSE grid; seeds 0,1,2 for the test refit.
- Discipline: the plan JSON is persisted before the first test read and
  `manifest.json` records `ordering_ok`; test metrics are computed once.

## Verification so far

- `uv run pytest`: 21 tests pass (data loading, splits, WRMSSE hand
  computation, baselines, feature builder, inventory).
- `uv run ruff check .`: clean.
- `uv run mypy src`: strict, clean (13 files).
- Kaggle smoke and full runs: pending.

## Known limitations

- WRMSSE bootstrap resamples the 42,840 evaluated units i.i.d.; it does
  not resample bottom-level series and rebuild the hierarchy, so CIs are
  approximate.
- Prophet depends on the cmdstan backend; first fit pays an install
  cost inside the kernel.
- Inventory figures are a labeled scenario under declared costs, not a
  measured business saving.
