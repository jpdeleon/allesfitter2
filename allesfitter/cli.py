from __future__ import annotations

import csv
import math
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
    port: int = typer.Option(5100),
    no_network: bool = typer.Option(False, "--no-network", help="disable NASA Archive lookups"),
    no_kill_existing: bool = typer.Option(
        False,
        "--no-kill-existing",
        help="do not stop a process already listening on --port before starting",
    ),
    reload: bool = typer.Option(
        False, "--reload", help="reload the development server when source files change"
    ),
):
    """Serve the allesfitter web GUI."""
    from allesfitter.webgui.cli import main as gui_main

    argv = []
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
    if reload:
        argv.append("--reload")
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


@app.command()
def show_initial_guess(
    dir_path: str = typer.Argument(..., help="path to the data directory"),
    quiet: bool = typer.Option(False, "--quiet", "-q"),
    no_plot: bool = typer.Option(False, "--no-plot", help="skip generating figures"),
    file_extension: str = typer.Option(
        ".pdf", "--file-extension", "-e", help="figure format: pdf, png, jpg, svg, or webp"
    ),
    midtransit: bool = typer.Option(True, "--midtransit/--no-midtransit"),
    ingress: bool = typer.Option(False, "--ingress/--no-ingress"),
    egress: bool = typer.Option(False, "--egress/--no-egress"),
):
    """Plot data with the initial guess model."""
    from allesfitter.general_output import show_initial_guess as _show

    _show(
        dir_path,
        quiet=quiet,
        do_plot=not no_plot,
        plot_midtransit=midtransit,
        plot_ingress=ingress,
        plot_egress=egress,
        file_extension=file_extension,
    )


@app.command()
def optimize(
    dir_path: str = typer.Argument(..., help="path to the data directory"),
    method: str = typer.Option(
        "cmaes",
        "--method",
        "-m",
        help="optimizer: cmaes, dual_annealing, differential_evolution, L-BFGS-B, ...",
    ),
    no_refine: bool = typer.Option(False, "--no-refine", help="skip L-BFGS-B refinement"),
    restarts: int = typer.Option(1, "--restarts", "-n", help="number of multistart restarts"),
    seed: int = typer.Option(42, "--seed", help="random seed"),
    quiet: bool = typer.Option(False, "--quiet", "-q"),
):
    """Global optimization to warm-start MCMC / nested sampling."""
    from allesfitter.optimize import optimize as _optimize

    _optimize(
        dir_path,
        method=method,
        refine=not no_refine,
        n_restarts=restarts,
        seed=seed,
        quiet=quiet,
    )


@app.command()
def mcmc_fit(
    dir_path: str = typer.Argument(..., help="path to the data directory"),
    append: bool = typer.Option(False, "--append", help="append to existing chains"),
):
    """Run MCMC sampling (emcee)."""
    from allesfitter.mcmc import mcmc_fit as _mcmc_fit
    from allesfitter.run_logger import log_run

    with log_run("mcmc_fit", dir_path):
        _mcmc_fit(dir_path, append=append)


@app.command()
def mcmc_output(
    dir_path: str = typer.Argument(..., help="path to the data directory"),
    overwrite: bool = typer.Option(False, "--overwrite", "-o"),
    quiet: bool = typer.Option(False, "--quiet", "-q"),
    file_extension: str = typer.Option(
        ".pdf", "--file-extension", "-e", help="figure format: pdf, png, jpg, svg, or webp"
    ),
):
    """Process MCMC results (posteriors, plots, summaries)."""
    from allesfitter.mcmc_output import mcmc_output as _mcmc_output

    _mcmc_output(dir_path, overwrite=overwrite, quiet=quiet, file_extension=file_extension)


@app.command()
def ns_fit(
    dir_path: str = typer.Argument(..., help="path to the data directory"),
    overwrite: bool = typer.Option(False, "--overwrite", "-o"),
):
    """Run nested sampling (dynesty)."""
    from allesfitter.nested_sampling import ns_fit as _ns_fit
    from allesfitter.run_logger import log_run

    with log_run("ns_fit", dir_path):
        _ns_fit(dir_path, overwrite=overwrite)


@app.command()
def ns_output(
    dir_path: str = typer.Argument(..., help="path to the data directory"),
    overwrite: bool = typer.Option(False, "--overwrite", "-o"),
    file_extension: str = typer.Option(
        ".pdf", "--file-extension", "-e", help="figure format: pdf, png, jpg, svg, or webp"
    ),
):
    """Process nested sampling results (posteriors, evidence, plots)."""
    from allesfitter.nested_sampling_output import ns_output as _ns_output

    _ns_output(dir_path, overwrite=overwrite, file_extension=file_extension)


@app.command()
def show_settings(
    dir_path: str = typer.Argument(..., help="path to the data directory"),
):
    """Print settings.csv as a formatted table."""
    from rich import box
    from rich.console import Console
    from rich.table import Table

    _console = Console()
    path = Path(dir_path).resolve() / "settings.csv"
    if not path.exists():
        _console.print(f"[red]Error:[/] {path} not found")
        raise typer.Exit(1)

    raw = path.read_text().splitlines()
    sec_rows: list[tuple[str, list[tuple[str, str]]]] = []
    current_sec = ""
    current_rows: list[tuple[str, str]] = []
    after_sep = False
    for line in raw:
        s = line.strip()
        if not s:
            continue
        if s.startswith("###"):
            if current_rows:
                sec_rows.append((current_sec, current_rows))
                current_rows = []
                current_sec = ""
            after_sep = True
            continue
        if s.startswith("#") and not s.startswith("##"):
            rest = s.lstrip("# ").rstrip(",").strip()
            if after_sep and rest:
                is_section_title = (
                    "," not in rest
                    or not rest.split(",")[0].replace("_", "").replace(".", "").isalnum()
                )
                if is_section_title:
                    current_sec = rest
                    after_sep = False
            continue
        after_sep = False
        if s.startswith("#"):
            continue
        k, _, v = s.partition(",")
        current_rows.append((k.strip(), v.strip()))
    if current_rows:
        sec_rows.append((current_sec, current_rows))

    for sec_name, rows in sec_rows:
        table = Table(
            title=sec_name, box=box.ROUNDED, title_justify="left", header_style="bold cyan"
        )
        table.add_column("Key", style="green", no_wrap=True)
        table.add_column("Value", style="white")
        for k, v in rows:
            table.add_row(k, v)
        _console.print(table)
        _console.print()


@app.command()
def show_params(
    dir_path: str = typer.Argument(..., help="path to the data directory"),
    star: bool = typer.Option(False, "--star", help="only show params_star.csv"),
):
    """Print params.csv (and params_star.csv) as formatted tables."""
    from rich import box
    from rich.console import Console
    from rich.table import Table

    _console = Console()
    base = Path(dir_path).resolve()

    # ── params_star.csv ──
    star_path = base / "params_star.csv"
    if star_path.exists():
        lines = star_path.read_text().splitlines()
        if len(lines) >= 3:
            headers = [h.strip("# ").strip() for h in lines[0].split(",")]
            units = [u.strip() for u in lines[1].split(",")]
            vals = [v.strip() for v in lines[2].split(",")]
            table = Table(
                title="Stellar Parameters",
                box=box.ROUNDED,
                title_justify="left",
                header_style="bold magenta",
            )
            table.add_column("Parameter", style="cyan", no_wrap=True)
            table.add_column("Value", style="white", justify="right")
            table.add_column("Unit", style="yellow")
            for h, v, u in zip(headers, vals, units):
                table.add_row(h, v, u)
            _console.print(table)
            _console.print()

    if star:
        return

    # ── params.csv ──
    params_path = base / "params.csv"
    if not params_path.exists():
        _console.print(f"[red]Error:[/] {params_path} not found")
        raise typer.Exit(1)

    cols = None
    sections: list[tuple[str, list[list[str]]]] = []
    current_sec = ""
    current_rows: list[list[str]] = []
    first = True
    for line in params_path.read_text().splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("###"):
            if first:
                first = False
            elif current_rows and cols:
                sections.append((current_sec, current_rows))
                current_rows = []
            current_sec = stripped.strip("# ")
            continue
        if stripped.startswith("#") and not first:
            continue
        if stripped.startswith("#") and first:
            cols = [c.strip("# ").strip() for c in stripped.split(",")]
            first = False
            continue
        current_rows.append([c.strip() for c in stripped.split(",")])
    if current_rows and cols:
        sections.append((current_sec, current_rows))

    if not cols:
        _console.print("[red]Error:[/] params.csv has no header row")
        raise typer.Exit(1)

    fit_col = cols.index("fit") if "fit" in cols else -1
    name_col = 0
    bound_col = cols.index("bounds") if "bounds" in cols else -1

    for sec_name, rows in sections:
        table = Table(
            title=sec_name,
            box=box.SIMPLE_HEAVY,
            title_justify="left",
            header_style="bold green",
            collapse_padding=True,
        )
        table.add_column("name", style="cyan", no_wrap=True)
        table.add_column("value", justify="right", no_wrap=True)
        table.add_column("fit?", justify="center", width=4, no_wrap=True)
        table.add_column("bounds", overflow="fold")
        for row in rows:
            if len(row) <= name_col or row[name_col].startswith("#"):
                continue
            name = row[name_col]
            value = row[1] if len(row) > 1 else ""
            is_fitted = fit_col >= 0 and fit_col < len(row) and row[fit_col] == "1"
            status = "[green]✓[/]" if is_fitted else "[dim]✗[/]"
            bounds = row[bound_col] if bound_col >= 0 and bound_col < len(row) else ""
            table.add_row(
                name, value, status, bounds, style="green" if is_fitted else "bright_black"
            )
        _console.print(table)
        _console.print()

    fitted = sum(
        1
        for _, rows in sections
        for r in rows
        if len(r) > name_col
        and not r[name_col].startswith("#")
        and fit_col >= 0
        and fit_col < len(r)
        and r[fit_col] == "1"
    )
    fixed = sum(
        1
        for _, rows in sections
        for r in rows
        if len(r) > name_col
        and not r[name_col].startswith("#")
        and fit_col >= 0
        and fit_col < len(r)
        and r[fit_col] == "0"
    )
    _console.print(f"[dim]{fitted} fitted, {fixed} fixed parameters[/]")


def _find_result_table(base: Path, sampler: str) -> Path | None:
    """Prefer sampler-specific output, then fall back to legacy results/."""
    filename = f"{sampler}_table.csv"
    for directory in (base / f"{sampler}_results", base / "results"):
        candidate = directory / filename
        if candidate.is_file():
            return candidate
    return None


def _read_result_rows(path: Path, first_column: str) -> list[dict[str, str]]:
    """Read an allesfitter result CSV whose header is prefixed with ``#``."""
    header: list[str] | None = None
    rows: list[dict[str, str]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for fields in csv.reader(handle):
            if not fields or not any(field.strip() for field in fields):
                continue
            first = fields[0].strip()
            if first.startswith("#"):
                candidate = first.lstrip("# ").strip()
                if candidate == first_column:
                    fields[0] = candidate
                    header = [field.strip() for field in fields]
                continue
            if header is None:
                continue
            values = [field.strip() for field in fields]
            # Historical tables were written without CSV quoting. Rebuild the
            # two known schemas explicitly so commas inside LaTeX labels do
            # not shift numeric fields into the wrong columns.
            if first_column == "name" and len(values) >= 6:
                rows.append(
                    {
                        "name": values[0],
                        "median": values[1],
                        "lower_error": values[2],
                        "upper_error": values[3],
                        "label": ",".join(values[4:-1]),
                        "unit": values[-1],
                    }
                )
            elif first_column == "property" and len(values) >= 5:
                rows.append(
                    {
                        "property": ",".join(values[:-4]),
                        "value": values[-4],
                        "lower_error": values[-3],
                        "upper_error": values[-2],
                        "source": values[-1],
                    }
                )
            else:
                rows.append(dict(zip(header, values)))
    return rows


def _as_finite_float(value: str) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _format_result_values(
    value: str, lower_error: str, upper_error: str
) -> tuple[str, str, str, bool]:
    """Format a posterior triplet with precision set by its uncertainty."""
    median = _as_finite_float(value)
    lower = _as_finite_float(lower_error)
    upper = _as_finite_float(upper_error)
    fixed = lower is None or upper is None
    if fixed:
        formatted = f"{median:.12g}" if median is not None else value
        return formatted, "—", "—", True

    positive_errors = [abs(error) for error in (lower, upper) if error != 0]
    if not positive_errors:
        formatted = f"{median:.12g}" if median is not None else value
        return formatted, "−0", "+0", False
    else:
        exponent = math.floor(math.log10(min(positive_errors)))
        decimals = max(0, 1 - exponent)  # two significant digits in the errors

    use_scientific = decimals > 10 or (median is not None and median != 0 and abs(median) < 1e-5)
    if use_scientific:
        formatted_value = f"{median:.6e}" if median is not None else value
        formatted_lower = f"−{abs(lower):.2e}"
        formatted_upper = f"+{abs(upper):.2e}"
    else:
        decimals = min(decimals, 10)
        formatted_value = f"{median:.{decimals}f}" if median is not None else value
        formatted_lower = f"−{abs(lower):.{decimals}f}"
        formatted_upper = f"+{abs(upper):.{decimals}f}"
    return formatted_value, formatted_lower, formatted_upper, False


@app.command()
def show_results(
    dir_path: str = typer.Argument(..., help="path to the data directory"),
):
    """Print processed MCMC and nested-sampling results as formatted tables."""
    from rich import box
    from rich.console import Console
    from rich.table import Table

    console = Console()
    base = Path(dir_path).resolve()
    discovered = False

    for sampler, label, color in (
        ("mcmc", "MCMC", "magenta"),
        ("ns", "Nested Sampling", "cyan"),
    ):
        table_path = _find_result_table(base, sampler)
        if table_path is None:
            continue
        discovered = True
        rows = _read_result_rows(table_path, "name")
        relative_path = table_path.relative_to(base)
        table = Table(
            title=f"{label} Posterior Results",
            caption=str(relative_path),
            caption_style="dim",
            box=box.ROUNDED,
            title_justify="left",
            header_style=f"bold {color}",
            collapse_padding=True,
        )
        table.add_column("Parameter", style="cyan", no_wrap=True)
        table.add_column("Median", justify="right", style="bold white", no_wrap=True)
        table.add_column("− error", justify="right", style="yellow", no_wrap=True)
        table.add_column("+ error", justify="right", style="green", no_wrap=True)
        table.add_column("Unit", style="yellow")
        table.add_column("Status", justify="center", no_wrap=True)

        fixed_count = 0
        for row in rows:
            value, lower, upper, fixed = _format_result_values(
                row.get("median", ""),
                row.get("lower_error", ""),
                row.get("upper_error", ""),
            )
            if fixed:
                fixed_count += 1
            table.add_row(
                row.get("name", ""),
                value,
                lower,
                upper,
                row.get("unit", ""),
                "[dim]fixed[/]" if fixed else f"[{color}]posterior[/]",
                style="bright_black" if fixed else None,
            )
        console.print(table)
        console.print(
            f"[dim]{len(rows) - fixed_count} posterior, {fixed_count} fixed parameters[/]"
        )
        console.print()

        derived_path = table_path.with_name(f"{sampler}_derived_table.csv")
        if not derived_path.is_file():
            continue
        derived_rows = _read_result_rows(derived_path, "property")
        derived_table = Table(
            title=f"{label} Derived Results",
            caption=str(derived_path.relative_to(base)),
            caption_style="dim",
            box=box.SIMPLE_HEAVY,
            title_justify="left",
            header_style=f"bold {color}",
            collapse_padding=True,
        )
        derived_table.add_column("Property", style="cyan", overflow="fold")
        derived_table.add_column("Value", justify="right", style="bold white", no_wrap=True)
        derived_table.add_column("− error", justify="right", style="yellow", no_wrap=True)
        derived_table.add_column("+ error", justify="right", style="green", no_wrap=True)
        derived_table.add_column("Source", style="dim", no_wrap=True)
        for row in derived_rows:
            value, lower, upper, _ = _format_result_values(
                row.get("value", ""),
                row.get("lower_error", ""),
                row.get("upper_error", ""),
            )
            derived_table.add_row(
                row.get("property", ""),
                value,
                lower,
                upper,
                row.get("source", ""),
            )
        console.print(derived_table)
        console.print(f"[dim]{len(derived_rows)} derived quantities[/]")
        console.print()

    if not discovered:
        console.print(f"[red]Error:[/] No processed result tables found under {base}")
        console.print(
            "[dim]Run `allesfitter mcmc-output <dir>` or `allesfitter ns-output <dir>` first.[/]"
        )
        raise typer.Exit(1)


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
