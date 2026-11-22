# Review

## What changed vs the legacy pipeline

- Data: the old pipeline aggregated to `dept_store_day` and clipped the
  window to overlap an unrelated Walmart weekly dataset. The new code
  keeps all 30,490 item-store series over d_1..d_1941 and adds the
  organizers' private-window truth (d_1942..d_1969) plus the official
  weight and score files from a sha256-pinned private dataset.
- Protocol: train d_1..d_1913, validation d_1914..d_1941, test
  d_1942..d_1969 scored against the organizers' truth, loaded only
  after `analysis_plan.json` is persisted.
- Metric: `wrmsse_proxy` is replaced by the official WRMSSE: 12
  levels, per-level scales from the first non-zero sale, dollar-sales
  weights over the last 28 days, equal averaging over levels.
- Verification: weights must reproduce both official weight files;
  the panel's validation window must equal `sales_test_validation.csv`;
  Naive/sNaive WRMSSE must reproduce the published table within 1e-5.
- Uncertainty: the i.i.d. unit bootstrap is replaced by a paired
  item-cluster bootstrap (multinomial item counts shared across models,
  levels 1-9 recomputed from weighted aggregates).
- LightGBM: 33 origin-relative features (lags, rolling stats,
  same-weekday, target-day price/calendar/SNAP, static categoricals),
  `deterministic: true`, labels disabled for beyond-panel prediction;
  a perturbation test proves no post-origin leakage.
- Prophet uses real `calendar.csv` dates instead of a synthetic
  2011-01-01 range.
- Inventory: critical fractile corrected to 0.8 (underage = price,
  overage = 0.25 x price) with a 0.95 sensitivity row; "holding" is
  renamed "overage".

## Verification so far

- `uv run pytest`: 32 tests pass (loading, protocol bounds, truth
  integrity, WRMSSE hand computation, bootstrap properties, feature
  leakage perturbation, inventory).
- `uv run ruff check .`: clean.
- `uv run mypy src`: strict, clean (16 files).
- Local gate check on the real files: `check_weights` matches all
  42,840 units per window (max dollar diff ~5e-10; stored official
  weights are self-consistent only to ~2.2e-6 relative, so the
  tolerance is 1e-5); `check_consistency` max diff 0.0;
  `check_official_scores` reproduces Naive and sNaive within ~4e-15 on
  all 13 numbers.
- Kaggle smoke and full runs: pending.

## Known limitations

- The bootstrap resamples item clusters with stores and the single
  28-day window fixed; intervals reflect product-population
  sensitivity, not temporal uncertainty.
- Prophet depends on the cmdstan backend; first fit pays an install
  cost inside the kernel.
- Inventory figures are a labeled scenario under declared costs, not a
  measured business saving.
