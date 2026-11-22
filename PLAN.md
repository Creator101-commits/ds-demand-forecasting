# Plan

## Status

In progress: package scaffold, data loading, official WRMSSE, models,
and Kaggle infrastructure are implemented and tested locally; the smoke
kernel has not run yet.

## Steps

1. Replace the legacy pipeline (dept_store_day aggregation, unrelated
   weekly overlap, WRMSSE proxy, missing SARIMA) with `src/demand`.
2. Load the four competition CSVs into an item-store panel plus a
   price-per-day matrix for dollar-sales weights.
3. Freeze train/validation/test day windows and persist the plan before
   the first test read.
4. Implement official WRMSSE (12 levels, first-nonzero scales,
   last-28-day dollar weights) with bootstrap CIs.
5. Baselines (naive, seasonal naive, ma28) plus pooled LightGBM with a
   validation-WRMSSE grid and seeded refits; SARIMA and Prophet on the
   14 coarse series; newsvendor inventory scenario.
6. Tests for loading, aggregation, splits, metrics, and model behavior;
   ruff, strict mypy, pytest-cov; CI matrix 3.12/3.13 x ubuntu/macos.
7. CPU Kaggle kernel + smoke kernel with git-SHA guard; estimate runtime
   from smoke; run full only if <= 9h.
8. Curate results, render figure, update docs after the run completes.

## Open questions

- Whether the Kaggle CPU runtime lands under the 9h cap; the smoke run
  measures per-origin LightGBM cost and the aggregate-model cost.
- Prophet needs the cmdstan backend installed on first fit; the runner
  provisions it via `cmdstanpy.install_cmdstan()`.
