from __future__ import annotations

import sys
from pathlib import Path

import typer

from allesfitter._version import __version__

app = typer.Typer(
    name="allesfitter",
    help="A global inference framework for photometry and RV",
    no_args_is_help=True,
)


def _version_callback(value: bool):
    if value:
        typer.echo(f"allesfitter v{__version__}")
        raise typer.Exit()


@app.callback()
def _main(
    version: bool = typer.Option(False, "--version", callback=_version_callback, is_eager=True),
):
    pass


@app.command()
def gui(
    runs_root: str = typer.Option("webgui_runs", help="directory for per-run datadirs"),
    db: str | None = typer.Option(None, help="SQLite path (default: <runs-root>/webgui.sqlite3)"),
    toi_csv: str | None = typer.Option(None, help="local ExoFOP TOI table for auto-fill"),
    host: str = typer.Option("127.0.0.1"),
    port: int = typer.Option(8000),
    no_network: bool = typer.Option(False, "--no-network", help="disable NASA Archive lookups"),
    no_kill_existing: bool = typer.Option(
        False,
        "--no-kill-existing",
        help="do not stop a process already listening on --port before starting",
    ),
):
    """Serve the allesfitter web GUI."""
    from allesfitter.webgui.cli import main as gui_main

    argv = ["allesfitter-gui"]
    argv.append(f"--runs-root={runs_root}")
    if db:
        argv.append(f"--db={db}")
    if toi_csv:
        argv.append(f"--toi-csv={toi_csv}")
    argv.append(f"--host={host}")
    argv.append(f"--port={port}")
    if no_network:
        argv.append("--no-network")
    if no_kill_existing:
        argv.append("--no-kill-existing")
    gui_main(argv)


@app.command()
def prepare(
    toi: int | None = typer.Option(None, "-toi", help="TOI ID"),
    ctoi: int | None = typer.Option(None, "-ctoi", help="CTOI ID"),
    tic: int | None = typer.Option(None, "-tic", help="TIC ID"),
    name: str | None = typer.Option(None, "-name", help="Name"),
    sector: list[str] | None = typer.Option(
        None, "-s", "--sector", help="TESS sector(s); -1=most recent, all=all"
    ),
    campaign: str | None = typer.Option(None, "-c", "--campaign", help="K2 campaign"),
    quarter: str | None = typer.Option(None, "-q", "--quarter", help="Kepler quarter"),
    exptime: float | None = typer.Option(None, "-e", "--exptime", help="exposure time (seconds)"),
    pipeline: str = typer.Option("spoc", "-p", "--pipeline", help="TESS/Kepler data pipeline"),
    filename: list[str] | None = typer.Option(
        None, "-f", "--filename", help="lightcurve filename(s)"
    ),
    mission: str = typer.Option("tess", "-m", "--mission", help="satellite mission"),
    lc_type: str = typer.Option("pdcsap", "-lc", "--lc_type", help="type of light curve"),
    sigma: float | None = typer.Option(None, "-sig", "--sigma", help="sigma for removing outliers"),
    quality: str = typer.Option("default", "-qb", "--quality", help="quality bitmask level"),
    dir: str = typer.Option(".", "-dir", help="base directory"),
    interactive: bool = typer.Option(
        False, "-i", "--interactive", help="manually input missing values"
    ),
    update_db: bool = typer.Option(
        False, "-u", "--update_db", help="update TOI or NExSci database"
    ),
    results_dir: str | None = typer.Option(
        None, "-r", "--results_dir", help="path to previous results"
    ),
    overwrite: bool = typer.Option(False, "-o", "--overwrite", help="overwrite files"),
    debug: bool = typer.Option(False, "--debug"),
    lc_only: bool = typer.Option(False, "--lc-only", help="only download and save lightcurve"),
    ttv: bool = typer.Option(False, "--ttv", help="emit per-transit TTV parameters"),
    bandpass: list[str] | None = typer.Option(
        None,
        "-bp",
        "--bandpass",
        help="bandpass labels for chromatic modeling",
    ),
):
    """Download TESS/Kepler/K2 data and prepare config files."""
    if not any([toi, ctoi, tic, name]):
        typer.echo("Error: one of -toi, -ctoi, -tic, -name is required.")
        raise typer.Exit(1)
    argv = ["prepare_allesfit"]
    if toi is not None:
        argv += ["-toi", str(toi)]
    if ctoi is not None:
        argv += ["-ctoi", str(ctoi)]
    if tic is not None:
        argv += ["-tic", str(tic)]
    if name is not None:
        argv += ["-name", str(name)]
    if sector:
        argv += ["-s"] + sector
    if campaign is not None:
        argv += ["-c", str(campaign)]
    if quarter is not None:
        argv += ["-q", str(quarter)]
    if exptime is not None:
        argv += ["-e", str(exptime)]
    if pipeline != "spoc":
        argv += ["-p", pipeline]
    if filename:
        argv += ["-f"] + filename
    if mission != "tess":
        argv += ["-m", mission]
    if lc_type != "pdcsap":
        argv += ["-lc", lc_type]
    if sigma is not None:
        argv += ["-sig", str(sigma)]
    if quality != "default":
        argv += ["-qb", quality]
    if dir != ".":
        argv += ["-dir", dir]
    if interactive:
        argv += ["-i"]
    if update_db:
        argv += ["-u"]
    if results_dir is not None:
        argv += ["-r", results_dir]
    if overwrite:
        argv += ["-o"]
    if debug:
        argv += ["--debug"]
    if lc_only:
        argv += ["--lc-only"]
    if ttv:
        argv += ["--ttv"]
    if bandpass:
        argv += ["-bp"] + bandpass
    _run_script("prepare_allesfit.py", argv)


@app.command()
def grid(
    grid_dir: str = typer.Argument(..., help="directory holding grid.csv and run_id subdirs"),
    filter: list[str] | None = typer.Option(
        None, "--filter", metavar="KEY=VAL", help="only run rows matching KEY=VAL"
    ),
    sampler: str = typer.Option("mcmc", "--sampler", help="sampler (ns or mcmc)"),
    all: bool = typer.Option(False, "--all", help="re-run rows even if already done"),
    dry_run: bool = typer.Option(False, "--dry-run", help="show what would run"),
):
    """Drive an allesfitter model grid from a grid.csv manifest."""
    argv = ["run_allesfitter_grid", grid_dir]
    if filter:
        for f in filter:
            argv += ["--filter", f]
    if sampler != "mcmc":
        argv += ["--sampler", sampler]
    if all:
        argv += ["--all"]
    if dry_run:
        argv += ["--dry-run"]
    _run_script("run_allesfitter_grid.py", argv)


def _run_script(script_name: str, argv: list[str]) -> None:
    """Run a legacy script via runpy with the given argv."""
    import runpy

    script_path = str(Path(__file__).resolve().parent.parent / "scripts" / script_name)
    old_argv = sys.argv[:]
    try:
        sys.argv = argv
        runpy.run_path(script_path, run_name="__main__")
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    app()
