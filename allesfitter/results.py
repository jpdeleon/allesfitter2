"""Resolve sampler-specific and legacy allesfitter result directories."""

from __future__ import annotations

from pathlib import Path

RESULTS_DIR_NAMES = {"mcmc": "mcmc_results", "ns": "ns_results"}


def results_directory(datadir, sampler, *, for_write=False):
    """Return the result directory for ``sampler``.

    New output is written to ``mcmc_results`` or ``ns_results``. Reads fall
    back to the historical shared ``results`` directory when the specific
    directory does not exist, keeping old runs usable.
    """
    try:
        dirname = RESULTS_DIR_NAMES[sampler]
    except KeyError as exc:
        choices = ", ".join(sorted(RESULTS_DIR_NAMES))
        raise ValueError(f"sampler must be one of: {choices}") from exc

    datadir = Path(datadir)
    specific = datadir / dirname
    if for_write:
        specific.mkdir(parents=True, exist_ok=True)
        return str(specific)
    if specific.is_dir():
        return str(specific)
    return str(datadir / "results")


def use_results_directory(basement, sampler, *, for_write=False):
    """Select and return ``basement.outdir`` for a sampler."""
    basement.outdir = results_directory(basement.datadir, sampler, for_write=for_write)
    return basement.outdir
