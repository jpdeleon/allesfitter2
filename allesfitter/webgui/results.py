"""Read results out of a finished run directory.

allesfitter's own ``mcmc_output`` / ``ns_output`` already write posterior tables
and diagnostic figures into ``<run_dir>/results/`` (fit plots as PDF, chains as
JPG, tables as CSV/txt); this module discovers those artifacts and scrapes the
nested-sampling evidence for the UI. Evidence parsing mirrors
``scripts/run_allesfitter_grid.py::extract_logz`` so the web view and the grid
driver agree on the number.
"""

from __future__ import annotations

import re
from pathlib import Path

# Raster figures we can show inline in an <img>; PDFs are linked instead.
RASTER_EXTS: frozenset[str] = frozenset({".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp"})
DOC_EXTS: frozenset[str] = frozenset({".pdf"})
TABLE_EXTS: frozenset[str] = frozenset({".csv", ".txt"})

# Matches e.g. "log(Z) = -1234.5 +- 0.12" as printed by nested_sampling_output.
_LOGZ_RE = re.compile(
    r"log\(Z\)\s*=\s*(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*\+-\s*"
    r"(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)"
)


def _results_dir(run_dir: str | Path) -> Path:
    return Path(run_dir) / "results"


def read_logz(run_dir: str | Path) -> str:
    """Return ``"<value> +- <err>"`` scraped from NS logs, or ``""`` if absent.

    Uses the *last* match in the file (the final, converged evidence).
    """
    results_dir = _results_dir(run_dir)
    if not results_dir.is_dir():
        return ""
    last: re.Match | None = None
    for pattern in ("*.txt", "*.log"):
        for path in sorted(results_dir.glob(pattern)):
            try:
                text = path.read_text(errors="ignore")
            except OSError:
                continue
            for match in _LOGZ_RE.finditer(text):
                last = match
    return f"{last.group(1)} +- {last.group(2)}" if last else ""


def _glob_by_ext(run_dir: str | Path, exts: frozenset[str]) -> list[Path]:
    results_dir = _results_dir(run_dir)
    if not results_dir.is_dir():
        return []
    return sorted(p for p in results_dir.glob("**/*") if p.is_file() and p.suffix.lower() in exts)


def result_images(run_dir: str | Path) -> list[Path]:
    """Raster figures (PNG/JPG/SVG…) that can be shown inline."""
    return _glob_by_ext(run_dir, RASTER_EXTS)


def result_documents(run_dir: str | Path) -> list[Path]:
    """PDF figures (linked, not shown inline)."""
    return _glob_by_ext(run_dir, DOC_EXTS)


def result_tables(run_dir: str | Path) -> list[Path]:
    """Posterior/summary CSV/txt tables."""
    return _glob_by_ext(run_dir, TABLE_EXTS)


def result_files(run_dir: str | Path) -> list[Path]:
    """All servable figure/document artifacts (images + PDFs)."""
    return result_images(run_dir) + result_documents(run_dir)


def find_result_file(run_dir: str | Path, name: str) -> Path | None:
    """Locate a servable result file by basename, guarding against traversal."""
    results_dir = _results_dir(run_dir).resolve()
    for path in result_files(run_dir):
        rp = path.resolve()
        if rp.name == name and results_dir in rp.parents:
            return rp
    return None


def has_results(run_dir: str | Path) -> bool:
    """True once the engine has written any figure/table for this run."""
    return bool(result_files(run_dir) or result_tables(run_dir))
