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
