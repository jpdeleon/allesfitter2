"""Compare nested-sampling posteriors across multiple allesfitter fit directories.

Reads each directory's already-processed NS output — ``save_ns.pickle.gz``
and ``ns_table.csv`` (both written by ``allesfitter ns-fit`` / ``ns-output``)
— and never touches the global ``config.BASEMENT`` singleton. That keeps
directories with different ``params.csv``/``settings.csv`` safe to compare
side by side (e.g. one fit with an extra free parameter the other doesn't
have), since each directory's parameter names, labels, and posterior
samples are self-described in its own output files.

Produces:

* an overlaid corner plot for the fit parameters shared by every directory
  (``compare_corner<ext>``);
* a side-by-side median +/- error table for every parameter name that
  appears in any directory (``compare_table.csv``), including ones that
  are fixed or only present in a subset of directories;
* a log-evidence (logZ) comparison, since a ΔlogZ between two NS runs is
  the standard way to judge whether an extra free parameter (e.g. a GP
  baseline offset) is actually supported by the data.
"""

from __future__ import annotations

import csv
import gzip
import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ._figure_output import normalize_file_extension
from .results import (
    find_result_file,
    find_result_table,
    read_result_rows,
    target_output_directory,
)

_MAX_CORNER_SAMPLES_DEFAULT = 5000
_MAX_CORNER_INCHES = 40.0
_HARD_NDIM_CAP = 60
_COLORS = (
    "#1b9e77",
    "#d95f02",
    "#7570b3",
    "#e7298a",
    "#66a61e",
    "#e6ab02",
    "#a6761d",
    "#666666",
)


class CompareError(ValueError):
    """Raised when a directory is missing files ``compare`` needs, or the
    requested comparison is otherwise not possible (e.g. no shared params)."""


@dataclass
class DirResult:
    label: str
    dir_path: Path
    root: Path
    fitkeys: list[str]
    samples: dict[str, np.ndarray]  # equal-weight posterior, keyed by fitkey
    table_rows: dict[str, dict[str, str]]


@dataclass
class CompareResult:
    labels: list[str]
    results: list[DirResult]
    shared_params: list[str]
    extra_params: dict[str, list[str]]
    param_labels: dict[str, str]
    param_units: dict[str, str]
    all_names: list[str]
    logz_rows: list[dict] = field(default_factory=list)
    corner_path: Path | None = None
    table_csv_path: Path | None = None
    out_dir: Path | None = None


def _load_ns_pickle(root: Path) -> dict:
    pickle_path = find_result_file(root, "ns", "save_ns.pickle.gz")
    if pickle_path is None:
        raise CompareError(
            f"No 'save_ns.pickle.gz' found under {root} (looked in ns_results/ and "
            "results/). Run `allesfitter ns-fit <dir>` first."
        )
    with gzip.GzipFile(str(pickle_path), "rb") as f:
        return pickle.load(f)


def _equal_weight_samples(results: dict) -> tuple[list[str], np.ndarray]:
    """Return ``(fitkeys, 2d array)`` of resampled equal-weight posterior samples."""
    from dynesty import utils as dyutils

    samples = np.asarray(results["samples"])
    logwt = np.asarray(results["logwt"])
    logz_final = float(np.asarray(results["logz"])[-1])
    weights = np.exp(logwt - logz_final)
    weights = weights / weights.sum()
    eq = dyutils.resample_equal(samples, weights)
    fitkeys = [str(k) for k in np.atleast_1d(results["fitkeys"])]
    return fitkeys, eq


def _load_table_rows(root: Path) -> dict[str, dict[str, str]]:
    table_path = find_result_table(root, "ns")
    if table_path is None:
        raise CompareError(
            f"No 'ns_table.csv' found under {root} (looked in ns_results/ and "
            "results/). Run `allesfitter ns-output <dir>` first."
        )
    return {row["name"]: row for row in read_result_rows(table_path, "name")}


def load_dir_result(dir_path: str, label: str | None = None) -> DirResult:
    """Load one directory's NS posterior + summary table.

    Reads only already-written output files, so this never touches
    ``config.BASEMENT`` and is safe to call repeatedly for directories with
    different fit setups.
    """
    root = target_output_directory(dir_path)
    if not root.is_dir():
        raise CompareError(f"Not a directory: {dir_path}")

    raw_results = _load_ns_pickle(root)
    fitkeys, eq = _equal_weight_samples(raw_results)
    samples = {key: eq[:, i] for i, key in enumerate(fitkeys)}

    return DirResult(
        label=label or Path(dir_path).expanduser().resolve().name,
        dir_path=Path(dir_path).expanduser().resolve(),
        root=root,
        fitkeys=fitkeys,
        samples=samples,
        table_rows=_load_table_rows(root),
    )


def _shared_params(results: list[DirResult], requested: list[str] | None) -> list[str]:
    if requested:
        missing = {r.label: [p for p in requested if p not in r.samples] for r in results}
        missing = {label: names for label, names in missing.items() if names}
        if missing:
            details = "; ".join(f"{label}: {', '.join(names)}" for label, names in missing.items())
            raise CompareError(f"Requested parameter(s) not found in every directory — {details}")
        return list(requested)

    common = set(results[0].fitkeys)
    for r in results[1:]:
        common &= set(r.fitkeys)
    if not common:
        raise CompareError(
            "No fit parameters are shared across all given directories; pass --params explicitly."
        )
    return [k for k in results[0].fitkeys if k in common]  # preserve first dir's order


def _extra_params(results: list[DirResult], shared: list[str]) -> dict[str, list[str]]:
    shared_set = set(shared)
    return {r.label: [k for k in r.fitkeys if k not in shared_set] for r in results}


def _param_label(results: list[DirResult], name: str) -> str:
    for r in results:
        row = r.table_rows.get(name)
        if row and row.get("label"):
            return row["label"]
    return name


def _param_unit(results: list[DirResult], name: str) -> str:
    for r in results:
        row = r.table_rows.get(name)
        if row and row.get("unit"):
            return row["unit"]
    return ""


def _build_overlay_corner(
    results: list[DirResult],
    shared: list[str],
    axis_labels: list[str],
    out_path: Path,
    max_samples: int,
) -> None:
    import corner as _corner
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    ndim = len(shared)
    if ndim < 2:
        raise CompareError(
            f"Need at least 2 shared fit parameters to draw a corner plot (found {ndim}: {shared})."
        )
    if ndim > _HARD_NDIM_CAP:
        raise CompareError(
            f"{ndim} shared parameters exceeds the corner-plot cap of {_HARD_NDIM_CAP}; "
            "narrow the comparison with --params."
        )

    side_inches = min(2.2 * ndim, _MAX_CORNER_INCHES)
    fig = plt.figure(figsize=(side_inches, side_inches))
    rng = np.random.default_rng(42)

    arrays = []
    for r in results:
        arr = np.column_stack([r.samples[p] for p in shared])
        if arr.shape[0] > max_samples:
            idx = rng.choice(arr.shape[0], size=max_samples, replace=False)
            arr = arr[idx]
        arrays.append(arr)

    # Fix each panel's axis limits to the union of every directory's samples
    # (with a small pad) up front. corner.corner() otherwise re-scales axes
    # to whichever dataset was plotted last, silently clipping the others.
    combined = np.concatenate(arrays, axis=0)
    lo, hi = np.nanmin(combined, axis=0), np.nanmax(combined, axis=0)
    pad = 0.05 * np.maximum(hi - lo, 1e-12)
    axis_range = list(zip(lo - pad, hi + pad))

    for i, arr in enumerate(arrays):
        color = _COLORS[i % len(_COLORS)]
        _corner.corner(
            arr,
            labels=axis_labels,
            range=axis_range,
            fig=fig,
            color=color,
            plot_datapoints=False,
            plot_density=False,
            fill_contours=True,
            levels=(0.393, 0.865, 0.989),
            hist_kwargs={"density": True, "linewidth": 1.5, "color": color},
            label_kwargs={
                "fontsize": min(24, 10 + ndim),
                "rotation": 45,
                "horizontalalignment": "right",
            },
        )

    handles = [
        Line2D([0], [0], color=_COLORS[i % len(_COLORS)], lw=3, label=r.label)
        for i, r in enumerate(results)
    ]
    fig.legend(handles=handles, loc="upper right", fontsize=14, frameon=False)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _write_table_csv(results: list[DirResult], all_names: list[str], out_path: Path) -> None:
    header = ["name", "label", "unit"]
    for r in results:
        header += [f"{r.label}_median", f"{r.label}_lower_error", f"{r.label}_upper_error"]

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for name in all_names:
            row = [name, _param_label(results, name), _param_unit(results, name)]
            for r in results:
                entry = r.table_rows.get(name)
                if entry is None:
                    row += ["", "", ""]
                else:
                    row += [
                        entry.get("median", ""),
                        entry.get("lower_error", ""),
                        entry.get("upper_error", ""),
                    ]
            writer.writerow(row)


def compare(
    dirs: list[str],
    labels: list[str] | None = None,
    params: list[str] | None = None,
    out_dir: str | Path | None = None,
    file_extension: str = ".pdf",
    max_samples: int = _MAX_CORNER_SAMPLES_DEFAULT,
) -> CompareResult:
    """Compare NS posteriors across two or more allesfitter fit directories.

    Parameters
    ----------
    dirs : list of str
        Two or more allesfitter data directories, each already processed
        with ``ns-fit`` / ``ns-output``.
    labels : list of str, optional
        One label per directory (default: each directory's basename).
    params : list of str, optional
        Fit-parameter names to compare (default: every fit parameter shared
        by all directories, in the first directory's order).
    out_dir : str or Path, optional
        Where to write ``compare_corner<ext>`` and ``compare_table.csv``
        (default: ``./compare_<label1>_vs_<label2>[...]``).
    file_extension : str
        Corner-plot figure format (default ``.pdf``).
    max_samples : int
        Per-directory posterior subsample cap for the corner plot.
    """
    if len(dirs) < 2:
        raise CompareError("Need at least 2 directories to compare.")

    if labels is not None and len(labels) != len(dirs):
        raise CompareError(
            f"Got {len(labels)} label(s) for {len(dirs)} directories; provide one label per directory."
        )
    resolved_labels = (
        list(labels) if labels else [Path(d).expanduser().resolve().name for d in dirs]
    )
    if len(set(resolved_labels)) != len(resolved_labels):
        raise CompareError(f"Directory labels must be unique; got: {resolved_labels}")

    file_extension = normalize_file_extension(file_extension)

    results = [load_dir_result(d, label=lbl) for d, lbl in zip(dirs, resolved_labels)]

    shared = _shared_params(results, params)
    extra = _extra_params(results, shared)

    if out_dir is None:
        out_dir = Path.cwd() / ("compare_" + "_vs_".join(resolved_labels))
    out_dir = Path(out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    axis_labels = [_param_label(results, p) for p in shared]
    corner_path = out_dir / f"compare_corner{file_extension}"
    _build_overlay_corner(results, shared, axis_labels, corner_path, max_samples)

    all_names: list[str] = []
    for r in results:
        for name in r.table_rows:
            if name not in all_names:
                all_names.append(name)

    table_csv_path = out_dir / "compare_table.csv"
    _write_table_csv(results, all_names, table_csv_path)

    from .postprocessing.nested_sampling_compare_logZ import compare_logz

    logz_rows = compare_logz([r.dir_path for r in results], labels=resolved_labels)

    return CompareResult(
        labels=resolved_labels,
        results=results,
        shared_params=shared,
        extra_params=extra,
        param_labels={p: _param_label(results, p) for p in shared},
        param_units={p: _param_unit(results, p) for p in shared},
        all_names=all_names,
        logz_rows=logz_rows,
        corner_path=corner_path,
        table_csv_path=table_csv_path,
        out_dir=out_dir,
    )
