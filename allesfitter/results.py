"""Resolve allesfitter result directories.

By default results stay with the data they came from: a manual run of
``allesfitter mcmc-fit <dir_path>`` (or ``run.py``, or a notebook) writes to
``<dir_path>/mcmc_results`` and ``<dir_path>/ns_results``.

Setting the ``ALLESFITTER_RESULTS_DIR`` environment variable switches to a
shared output root instead, where every target gets its own subdirectory named
after its data directory, e.g. ``<root>/HIP67522/mcmc_results``. The browser
workbench (GUI) sets the variable to its workspace, which defaults to
:data:`DEFAULT_OUTPUT_BASE` (``~/ql/allesfitter``), so GUI runs collect their
results there instead of scattering them across the filesystem.
"""

from __future__ import annotations

import csv
import math
import os
from pathlib import Path

RESULTS_DIR_NAMES = {"mcmc": "mcmc_results", "ns": "ns_results"}

#: Environment variable that selects a shared per-target output root.
OUTPUT_BASE_ENV = "ALLESFITTER_RESULTS_DIR"

#: Shared output root used by the GUI when no workspace is given.
DEFAULT_OUTPUT_BASE = Path.home() / "ql" / "allesfitter"


def output_base() -> Path | None:
    """Return the shared per-target output root, or ``None`` for in-place output.

    Reads :data:`OUTPUT_BASE_ENV` at call time; when it is unset or empty each
    target's results belong next to its own data directory. Reading lazily lets
    the GUI (and tests) set the variable per process without re-importing the
    module.
    """
    override = os.environ.get(OUTPUT_BASE_ENV)
    if override:
        return Path(override).expanduser()
    return None


def target_name(datadir) -> str:
    """Return the per-target folder name derived from ``datadir``.

    Uses the resolved basename so ``.`` maps to the current directory's name
    and relative or trailing-slash paths behave consistently.
    """
    return Path(datadir).expanduser().resolve().name


def target_output_directory(datadir) -> Path:
    """Return the directory that holds ``datadir``'s result folders.

    That is ``datadir`` itself by default, or ``<output_base>/<target_name>``
    when a shared output root is configured.
    """
    base = output_base()
    if base is None:
        return Path(datadir).expanduser().resolve()
    return base / target_name(datadir)


def results_directory(datadir, sampler, *, for_write=False):
    """Return the result directory for ``sampler``.

    ``mcmc`` output goes to ``<target_output_directory>/mcmc_results`` and
    ``ns`` output to ``<target_output_directory>/ns_results``. When
    ``for_write`` is true the directory is created; otherwise the path is
    returned whether or not it exists so callers can test for existing output
    themselves.
    """
    try:
        dirname = RESULTS_DIR_NAMES[sampler]
    except KeyError as exc:
        choices = ", ".join(sorted(RESULTS_DIR_NAMES))
        raise ValueError(f"sampler must be one of: {choices}") from exc

    directory = target_output_directory(datadir) / dirname
    if for_write:
        directory.mkdir(parents=True, exist_ok=True)
    return str(directory)


def use_results_directory(basement, sampler, *, for_write=False):
    """Select and return ``basement.outdir`` for a sampler."""
    basement.outdir = results_directory(basement.datadir, sampler, for_write=for_write)
    return basement.outdir


def find_result_file(root: Path, sampler: str, filename: str) -> Path | None:
    """Locate ``filename`` under the per-target output ``root``.

    Checks the modern ``<sampler>_results`` directory first, then the
    legacy ``results`` directory used by pre-refactor allesfitter runs.
    """
    try:
        dirname = RESULTS_DIR_NAMES[sampler]
    except KeyError as exc:
        choices = ", ".join(sorted(RESULTS_DIR_NAMES))
        raise ValueError(f"sampler must be one of: {choices}") from exc

    for directory in (root / dirname, root / "results"):
        candidate = directory / filename
        if candidate.is_file():
            return candidate
    return None


def find_result_table(root: Path, sampler: str) -> Path | None:
    """Locate ``<sampler>_table.csv`` under the per-target output ``root``."""
    return find_result_file(root, sampler, f"{sampler}_table.csv")


def read_result_rows(path: Path, first_column: str) -> list[dict[str, str]]:
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


def as_finite_float(value: str) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def format_result_values(
    value: str, lower_error: str, upper_error: str
) -> tuple[str, str, str, bool]:
    """Format a posterior triplet with precision set by its uncertainty."""
    median = as_finite_float(value)
    lower = as_finite_float(lower_error)
    upper = as_finite_float(upper_error)
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
