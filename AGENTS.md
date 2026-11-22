# DS Demand Forecasting

A reproducible M5 demand-forecasting benchmark using the Kaggle
`m5-forecasting-accuracy` competition data, official WRMSSE, a frozen
temporal protocol, and a CPU Kaggle kernel.

## Build commands

```bash
# Verification commands.
uv sync
uv run pytest
uv run ruff check .
uv run mypy src
```

## Platform support

- macOS and Linux; Python 3.12-3.13; CPU only.
- The benchmark runs on a Kaggle CPU kernel via `scripts/kaggle_run.sh`.

## Key conventions

- Frozen protocol `m5-benchmark-v2`: train d_1..d_1913, validation
  d_1914..d_1941, test d_1942..d_1969 scored against the organizers'
  private-window truth. Model and threshold selection use validation
  only; the test truth file is read once after `analysis_plan.json` is
  persisted. Verification gates (official weights, validation
  consistency, Naive/sNaive vs the published table within 1e-5) run
  before any test metric.
- WRMSSE is the official M5 metric: 12 aggregation levels, RMSSE scales
  from the first non-zero sale, dollar-sales weights from the last 28
  days before each forecast origin, equal averaging over levels.
- Test CIs use a paired item-cluster bootstrap, not i.i.d. unit
  resampling; the percentile intervals are right-skewed because
  resampled item multisets change aggregate composition.
- LightGBM early stopping monitors its internal L2 metric on pooled
  validation rows, not WRMSSE. `MAX_ROUNDS=1500`, `EARLY_STOP=50`,
  and every grid config records `best_iteration` plus a `cap_bound`
  flag in `analysis_plan.json` and the manifest.
- Official evaluation files live in the private Kaggle dataset
  `sreeharshak101/m5-official-evaluation`, sha256-pinned in code.
- `results/` holds unedited kernel output; `image.png` renders from
  `results/` only.

## Rules

- Default branch is `main`; never force-push it or rewrite shared history.
- Ask before adding any dependency or package manager.
- Never fabricate data, benchmarks, logs or results.
- Done means the relevant tests actually ran and passed; never skip or mock them to finish.
- No emojis anywhere: code, comments, commits, READMEs, docs.
- No em dashes (the long dash, U+2014) anywhere: code, comments, commits, READMEs, docs. Use a comma, colon, parentheses, or a new sentence instead.
- Keep README in sync with code changes, following its existing template.
- Comments are brief and explain why, only when the code doesn't already say it.
- Keep dataset splits and eval leakage controls intact.

## Commits

- Conventional Commits: `type(scope): summary`. Types: `feat`, `fix`, `refactor`, `perf`, `test`, `docs`, `build`, `ci`, `chore`; scope is optional.
- Summary in imperative mood, lowercase, no trailing period, at most 72 characters.
- One logical change per commit; keep formatting-only changes separate.
- Add a body only when the reason is not obvious: explain why, not what, wrapped at 72.
- Breaking changes: `type!:` in the summary plus a `BREAKING CHANGE:` footer.
- Commit only when asked. Never commit secrets, build output, or failing code.
