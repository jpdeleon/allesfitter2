"""Resolve allesfitter result directories.

All results are written under a single per-target output root. By default the
root is ``~/ql/allesfitter``; each target then gets its own subdirectory named
after the data directory, e.g. ``~/ql/allesfitter/HIP67522/mcmc_results``.

The root can be redirected with the ``ALLESFITTER_RESULTS_DIR`` environment
variable. The browser workbench (GUI) sets this to its workspace so a
workbench keeps its results next to the target data; command-line and script
runs leave it unset and use the default.
"""

from __future__ import annotations

import os
from pathlib import Path

RESULTS_DIR_NAMES = {"mcmc": "mcmc_results", "ns": "ns_results"}

#: Environment variable that overrides the output root.
OUTPUT_BASE_ENV = "ALLESFITTER_RESULTS_DIR"

#: Default output root when the environment variable is unset.
DEFAULT_OUTPUT_BASE = Path.home() / "ql" / "allesfitter"


def output_base() -> Path:
    """Return the root directory that holds every target's results.

    Reads :data:`OUTPUT_BASE_ENV` at call time and falls back to
    :data:`DEFAULT_OUTPUT_BASE`. Reading lazily lets the GUI (and tests) set
    the variable per process without re-importing the module.
    """
    override = os.environ.get(OUTPUT_BASE_ENV)
    if override:
        return Path(override).expanduser()
    return DEFAULT_OUTPUT_BASE


def target_name(datadir) -> str:
    """Return the per-target folder name derived from ``datadir``.

    Uses the resolved basename so ``.`` maps to the current directory's name
    and relative or trailing-slash paths behave consistently.
    """
    return Path(datadir).expanduser().resolve().name


def target_output_directory(datadir) -> Path:
    """Return ``<output_base>/<target_name>`` for ``datadir``."""
    return output_base() / target_name(datadir)


def results_directory(datadir, sampler, *, for_write=False):
    """Return the result directory for ``sampler`` under the output root.

    ``mcmc`` output goes to ``<root>/<target>/mcmc_results`` and ``ns`` output
    to ``<root>/<target>/ns_results``. When ``for_write`` is true the directory
    is created; otherwise the path is returned whether or not it exists so
    callers can test for existing output themselves.
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
