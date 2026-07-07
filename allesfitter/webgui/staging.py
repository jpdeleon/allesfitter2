"""Stage source light curves into an allesfitter run directory.

Each selected light curve is placed at ``<run_dir>/<label>.csv`` (symlinked by
default so large TESS files are not copied), and its named covariate columns are
surfaced so the fit form can offer them for ``hybrid_linear_multi`` /
``sample_linear_multi`` detrending.

The header sniffing mirrors the contract of
``allesfitter.basement._parse_inst_csv_header`` (two accepted header styles), but
is reimplemented here with only the standard library so the web layer can list
covariates without importing the compiled fitting engine.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import yaml

# The three mandatory leading columns of every instrument CSV.
_REQUIRED_COLUMNS = ("time", "flux", "flux_err")


def sniff_header(path: str | Path) -> list[str]:
    """Return the CSV's column names, or ``[]`` for a legacy positional file.

    Accepts the same two header styles as the engine loader:

    1. ``#``-prefixed schema header whose first token starts with ``time`` and
       whose third token ends with ``_err``.
    2. A plain (pandas-style) header row whose first token is not a float.
    """
    try:
        with open(path) as fh:
            for line in fh:
                s = line.strip()
                if not s:
                    continue
                if s.startswith("#"):
                    toks = [t.strip() for t in s.lstrip("#").strip().split(",")]
                    if (
                        len(toks) >= 3
                        and toks[0].lower().startswith("time")
                        and toks[2].lower().endswith("_err")
                    ):
                        return toks
                    return []
                toks = [t.strip() for t in s.split(",")]
                if len(toks) < 3:
                    return []
                try:
                    float(toks[0])
                    return []  # data row -> no header
                except ValueError:
                    return toks
    except OSError:
        return []
    return []


def covariate_columns(path: str | Path) -> list[str]:
    """Named ancillary regressors (columns past ``time,flux,flux_err``)."""
    header = sniff_header(path)
    return header[3:] if len(header) >= 4 else []


@dataclass
class StagedFile:
    """Record of one staged instrument light curve."""

    label: str
    source: str
    dest: str
    covariate_columns: list[str] = field(default_factory=list)
    method: str = "symlink"


def stage_file(
    source: str | Path,
    label: str,
    run_dir: str | Path,
    *,
    method: str = "symlink",
    overwrite: bool = True,
) -> StagedFile:
    """Stage *source* into ``<run_dir>/<label>.csv``.

    ``method`` is ``"symlink"`` (default, no copy) or ``"copy"``.
    """
    source = Path(source)
    if not source.is_file():
        raise FileNotFoundError(f"source light curve not found: {source}")
    if method not in ("symlink", "copy"):
        raise ValueError(f"method must be 'symlink' or 'copy', got {method!r}")

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    dest = run_dir / f"{label}.csv"
    if dest.exists() or dest.is_symlink():
        if not overwrite:
            raise FileExistsError(f"{dest} already exists (overwrite=False)")
        dest.unlink()

    if method == "symlink":
        os.symlink(source.resolve(), dest)
    else:
        shutil.copy2(source, dest)

    return StagedFile(
        label=label,
        source=str(source.resolve()),
        dest=str(dest),
        covariate_columns=covariate_columns(source),
        method=method,
    )


def stage_all(
    items: list[tuple[str, str | Path]],
    run_dir: str | Path,
    *,
    method: str = "symlink",
) -> list[StagedFile]:
    """Stage many ``(label, source)`` pairs; duplicate labels are rejected."""
    labels = [label for label, _ in items]
    dupes = {label for label in labels if labels.count(label) > 1}
    if dupes:
        raise ValueError(f"duplicate instrument labels in staging set: {sorted(dupes)}")
    staged = [stage_file(source, label, run_dir, method=method) for label, source in items]
    _record_provenance(run_dir, staged)
    return staged


def _record_provenance(run_dir: str | Path, staged: list[StagedFile]) -> None:
    """Append a ``staged`` section to ``meta.yaml`` (created by the config writer)."""
    meta_path = Path(run_dir) / "meta.yaml"
    meta: dict = {}
    if meta_path.exists():
        meta = yaml.safe_load(meta_path.read_text()) or {}
    meta["staged"] = [
        {
            "label": s.label,
            "source": s.source,
            "method": s.method,
            "covariate_columns": s.covariate_columns,
        }
        for s in staged
    ]
    meta_path.write_text(yaml.safe_dump(meta, sort_keys=False))
