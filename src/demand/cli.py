"""Command line interface."""

from __future__ import annotations

from pathlib import Path

import typer

from demand.experimentation.benchmark import run_benchmark

app = typer.Typer(help="M5 demand forecasting benchmark.")


@app.command()
def benchmark(
    data_dir: Path = typer.Option(  # noqa: B008
        Path("data/m5"), "--data-dir", help="Directory with the three M5 CSVs."
    ),
    official_dir: Path = typer.Option(  # noqa: B008
        Path("data/m5-official"),
        "--official-dir",
        help="Directory with the five official evaluation files.",
    ),
    out_dir: Path = typer.Option(  # noqa: B008
        Path("artifacts/benchmark"), "--out-dir", help="Artifact output dir."
    ),
    n_origins: int = typer.Option(40, "--n-origins", help="LGBM training origins."),
    n_boot: int = typer.Option(500, "--n-boot", help="Bootstrap replicates."),
    git_sha: str | None = typer.Option(None, "--git-sha", help="Override git SHA."),
    smoke: bool = typer.Option(False, "--smoke", help="Smoke mode (shifted protocol)."),
) -> None:
    """Run the frozen M5 benchmark protocol."""
    if smoke:
        n_origins = min(n_origins, 6)
        n_boot = min(n_boot, 20)
    run_benchmark(
        data_dir,
        official_dir,
        out_dir,
        n_origins=n_origins,
        n_boot=n_boot,
        git_sha=git_sha,
        smoke=smoke,
    )
    typer.echo(f"wrote artifacts to {out_dir}")


if __name__ == "__main__":
    app()
