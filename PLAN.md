# Plan

## Status

Done: protocol `m5-benchmark-v2` is implemented and the v2.1 full
Kaggle run (git `2a0a337`) passed every gate; `results/` holds the
unedited kernel output and `image.png` renders from it. The earlier
v2 run (git `65250f1`) is retained at
`artifacts/kaggle/20261121-223604/` as "v2 as-run" for disclosure:
its selected LightGBM config hit `best_iteration` 399 at
`MAX_ROUNDS` 400, so v2.1 raised the cap to 1500 and recorded
`cap_bound` per config. All v2.1 grid configs finished uncapped.

## Steps

1. Replace the legacy pipeline (dept_store_day aggregation, unrelated
   weekly overlap, WRMSSE proxy, missing SARIMA) with `src/demand`.
2. Load the three competition CSVs into an item-store panel plus
   price-per-day/week matrices; load the organizers' private-window
   truth only after the plan is persisted.
3. Freeze train d_1..d_1913 / validation d_1914..d_1941 / test
   d_1942..d_1969 and persist `analysis_plan.json` before the first
   test read.
4. Implement official WRMSSE (12 levels, first-nonzero scales,
   last-28-day dollar weights) with the paired item-cluster bootstrap.
5. Gates: our weights vs both official weight files, competition vs
   organizer validation consistency, Naive/sNaive vs the published
   table within 1e-5 (parsed from the XLSX in code).
6. Baselines plus pooled LightGBM with origin-relative features,
   categorical ids, deterministic training, validation-WRMSSE grid,
   and seeded refits; SARIMA and Prophet (real dates) on the 14 coarse
   series; newsvendor scenario at fractile 0.8 with 0.95 sensitivity.
7. Tests for loading, protocol bounds, truth integrity, WRMSSE hand
   computation, bootstrap properties, feature leakage, and inventory;
   ruff, strict mypy, pytest.
8. CPU Kaggle kernel + smoke kernel (protocol shifted -28 days, never
   touches official truth) with git-SHA guard; estimate runtime from
   smoke; run full only if within budget.
9. Curate results, render figure, update docs after the run completes.

## Open questions

- None blocking. The full run took ~2.3 CPU-hours on Kaggle, well
  under the 9h budget; the smoke-based estimate held.
