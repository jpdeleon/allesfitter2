from __future__ import annotations

import csv
import math
import os
import sys
from pathlib import Path

import typer

from allesfitter._version import __version__

app = typer.Typer(
    name="allesfitter",
    help="A global inference framework for photometry and RV",
    no_args_is_help=True,
)


def _default_workers() -> int:
    """Half of the machine's CPU cores (at least 1), used as the default
    parallelism for differential_evolution."""
    cpu_count = os.cpu_count() or 1
    return max(1, cpu_count // 2)


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
    exptime: list[float] | None = typer.Option(
        None,
        "-e",
        "--exptime",
        help="exposure time(s) in seconds; repeat once per --pipeline, or pass a "
        "single value shared by all pipelines",
    ),
    pipeline: list[str] = typer.Option(
        ["spoc"],
        "-p",
        "--pipeline",
        help="TESS/Kepler data pipeline(s); repeat once per --filename to "
        "download each instrument from its own pipeline in one run",
    ),
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
    period: float | None = typer.Option(None, "--period", help="orbital period (days)"),
    period_err: float | None = typer.Option(
        None, "--period-err", help="orbital-period uncertainty (days)"
    ),
    epoch: float | None = typer.Option(None, "--epoch", help="transit epoch (BJD)"),
    epoch_err: float | None = typer.Option(
        None, "--epoch-err", help="transit-epoch uncertainty (days)"
    ),
    duration: float | None = typer.Option(None, "--duration", help="transit duration (hours)"),
    duration_err: float | None = typer.Option(
        None, "--duration-err", help="transit-duration uncertainty (hours)"
    ),
    depth: float | None = typer.Option(None, "--depth", help="transit depth (ppm)"),
    depth_err: float | None = typer.Option(
        None, "--depth-err", help="transit-depth uncertainty (ppm)"
    ),
    h5: str | None = typer.Option(
        None,
        "--h5",
        help="quicklook TLS results .h5 file for a planet the catalog doesn't have "
        "yet: fills in a raw -tic target's blank ephemeris, or (with -toi) is "
        "appended as an extra companion alongside that target's known catalog "
        "planets; --period/--epoch/--duration/--depth still take priority when "
        "also given",
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
        argv += ["-e"] + [str(v) for v in exptime]
    if pipeline != ["spoc"]:
        argv += ["-p"] + pipeline
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
    for flag, value in (
        ("--period", period),
        ("--period-err", period_err),
        ("--epoch", epoch),
        ("--epoch-err", epoch_err),
        ("--duration", duration),
        ("--duration-err", duration_err),
        ("--depth", depth),
        ("--depth-err", depth_err),
    ):
        if value is not None:
            argv += [flag, str(value)]
    if h5 is not None:
        argv += ["--h5", h5]
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
def gui(
    host: str = typer.Option(
        "127.0.0.1",
        "--host",
        help="Bind address; keep localhost when using SSH port forwarding",
    ),
    port: int = typer.Option(5100, "--port", "-p", min=1, max=65535),
    workspace: str | None = typer.Option(
        None,
        "--workspace",
        "-w",
        help="Directory for target data, results, job logs, and the SQLite "
        "history (default: ~/ql/allesfitter)",
    ),
):
    """Run the browser workbench for targets, preparation, fits, and results."""
    from allesfitter.webapp import serve

    serve(host=host, port=port, workspace=workspace)


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
        "differential_evolution",
        "--method",
        "-m",
        help="optimizer: cmaes, dual_annealing, differential_evolution, L-BFGS-B, ...",
    ),
    no_refine: bool = typer.Option(False, "--no-refine", help="skip L-BFGS-B refinement"),
    restarts: int = typer.Option(1, "--restarts", "-n", help="number of multistart restarts"),
    seed: int = typer.Option(42, "--seed", help="random seed"),
    workers: int = typer.Option(
        _default_workers(),
        "--workers",
        "-w",
        help="parallel workers for differential_evolution (default: half of CPU cores); ignored by other methods",
    ),
    maxfevals: int = typer.Option(
        None,
        "--maxfevals",
        help="per-restart evaluation/iteration budget (default: optimizer's own default)",
    ),
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
        workers=workers,
        maxfevals=maxfevals,
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

    from pathlib import Path

    from allesfitter.results import results_directory

    output_table = Path(results_directory(dir_path, "mcmc")) / "mcmc_table.csv"
    if not output_table.exists():
        from allesfitter.mcmc_output import mcmc_output as _mcmc_output

        _mcmc_output(dir_path, overwrite=False)
    else:
        typer.echo("MCMC output already exists; run 'allesfitter mcmc-output' to refresh it.")


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

    from pathlib import Path

    from allesfitter.results import results_directory

    output_table = Path(results_directory(dir_path, "ns")) / "ns_table.csv"
    if not output_table.exists():
        from allesfitter.nested_sampling_output import ns_output as _ns_output

        _ns_output(dir_path, overwrite=False)
    else:
        typer.echo("NS output already exists; run 'allesfitter ns-output' to refresh it.")


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


@app.command("transit-search")
def transit_search(
    results_dir: str = typer.Argument(
        ..., help="path to an mcmc_results or ns_results directory from a completed fit"
    ),
    sde_min: float = typer.Option(
        8.0,
        "--sde-min",
        help="keep the iterative TLS search going while the found signal's SDE stays "
        "at or above this value; also gates whether a known companion counts as "
        "recovered",
    ),
    period_min: float | None = typer.Option(
        None, "--period-min", help="minimum period to search (days)"
    ),
    period_max: float | None = typer.Option(
        None, "--period-max", help="maximum period to search (days)"
    ),
    mask_width_factor: float = typer.Option(
        1.5,
        "--mask-width-factor",
        help="mask known companions out to this many times their transit duration",
    ),
    outdir: str | None = typer.Option(
        None,
        "--outdir",
        "-o",
        help="output directory (default: <target>/transit_search_results)",
    ),
    file_extension: str = typer.Option(
        ".pdf", "--file-extension", "-e", help="figure format: pdf, png, jpg, svg, or webp"
    ),
    max_candidates: int = typer.Option(
        20, "--max-candidates", help="safety cap on the number of candidates kept"
    ),
    mission: str = typer.Option(
        "tess", "--mission", "-m", help="mission (tess, k2, kepler); sets the h5 BJD offset"
    ),
    skip_known_recovery_check: bool = typer.Option(
        False,
        "--skip-known-recovery-check",
        help="skip checking whether each known companion in params.csv is independently "
        "recoverable before it gets masked out",
    ),
    recovery_period_frac: float = typer.Option(
        0.05,
        "--recovery-period-frac",
        help="half-width of the known-companion recovery check's period-search bracket, "
        "as a fraction of that companion's own period",
    ),
    quiet: bool = typer.Option(False, "--quiet", "-q"),
):
    """Detrend with the best-fit baseline, mask known planets, and blind-search
    for un-modeled transits with TLS until SDE drops below threshold."""
    from allesfitter.detection.blind_search import transit_search as _transit_search

    _transit_search(
        results_dir,
        sde_min=sde_min,
        period_min=period_min,
        period_max=period_max,
        mask_width_factor=mask_width_factor,
        outdir=outdir,
        file_extension=file_extension,
        max_candidates=max_candidates,
        mission=mission,
        check_known_recovery=not skip_known_recovery_check,
        recovery_period_frac=recovery_period_frac,
        quiet=quiet,
    )


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


def _find_result_table(root: Path, sampler: str) -> Path | None:
    """Locate ``<sampler>_table.csv`` under the per-target output ``root``."""
    filename = f"{sampler}_table.csv"
    for directory in (root / f"{sampler}_results", root / "results"):
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

    from allesfitter.results import target_output_directory

    console = Console()
    root = target_output_directory(dir_path)
    discovered = False

    for sampler, label, color in (
        ("mcmc", "MCMC", "magenta"),
        ("ns", "Nested Sampling", "cyan"),
    ):
        table_path = _find_result_table(root, sampler)
        if table_path is None:
            continue
        discovered = True
        rows = _read_result_rows(table_path, "name")
        relative_path = table_path.relative_to(root)
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
            caption=str(derived_path.relative_to(root)),
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
        console.print(f"[red]Error:[/] No processed result tables found under {root}")
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
