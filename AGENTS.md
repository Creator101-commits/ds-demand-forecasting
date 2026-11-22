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

- Frozen protocol: train d_1..d_1885, validation d_1886..d_1913, test
  d_1914..d_1941. Model and threshold selection use validation only; the
  test window is read once after `analysis_plan.json` is persisted.
- WRMSSE is the official M5 metric: 12 aggregation levels, RMSSE scales
  from the first non-zero sale, dollar-sales weights from the last 28
  days before each forecast origin.
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
