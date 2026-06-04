"""Shared file-reading and bounds-parsing helpers for the validation package.

Both :mod:`~allesfitter.validation.config_checks` (structural, raises) and
:mod:`~allesfitter.validation.prior_checks` (heuristic, warns) need to read the
same ``params.csv`` / ``settings.csv`` rows and parse the same prior-bounds
strings. This module is the single home for those low-level file/string → data
helpers so the two check layers stay pure and free of duplicated parsing.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple


def read_csv_rows(path) -> List[List[str]]:
    """Split a CSV into stripped token lists, skipping blanks and ``#`` lines.

    Accepts a ``str`` or :class:`os.PathLike`. Returns ``[]`` for a missing
    file so callers can treat "absent" and "empty" alike.
    """
    rows: List[List[str]] = []
    path = os.fspath(path)
    if not os.path.exists(path):
        return rows
    with open(path, encoding="utf-8") as fh:
        for line in fh.read().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            rows.append([p.strip() for p in line.split(",")])
    return rows


def read_param_rows(datadir: str) -> List[List[str]]:
    """Return the non-comment rows of ``<datadir>/params.csv``."""
    return read_csv_rows(os.path.join(datadir, "params.csv"))


def read_settings(datadir: str) -> Dict[str, str]:
    """Return ``settings.csv`` as a ``{key: value}`` dict (first column keyed)."""
    out: Dict[str, str] = {}
    for parts in read_csv_rows(os.path.join(datadir, "settings.csv")):
        if not parts:
            continue
        key = parts[0]
        out[key] = parts[1] if len(parts) >= 2 else ""
    return out


def parse_bounds(bounds_str: str) -> Optional[Tuple[str, Tuple[float, ...]]]:
    """Parse an allesfitter bounds string.

    Recognized forms (whitespace-separated):

    - ``uniform <lo> <hi>``           → ``("uniform", (lo, hi))``
    - ``normal <mean> <sigma>``       → ``("normal", (mean, sigma))``
    - ``trunc_normal <lo> <hi> <mean> <sigma>``
                                      → ``("trunc_normal", (lo, hi, mean, sigma))``

    Returns ``None`` for an empty/whitespace string (the row is fixed or has
    no prior). Returns ``("__unknown__", ())`` for an unrecognized keyword and
    ``("__bad__", ())`` when a known keyword is present but the numbers do not
    parse or the arity is wrong, so callers can flag each case distinctly.
    """
    if bounds_str is None:
        return None
    tokens = bounds_str.split()
    if not tokens:
        return None
    kind = tokens[0]
    nums_raw = tokens[1:]
    arity = {"uniform": 2, "normal": 2, "trunc_normal": 4}
    if kind not in arity:
        return ("__unknown__", ())
    if len(nums_raw) != arity[kind]:
        return ("__bad__", ())
    try:
        nums = tuple(float(x) for x in nums_raw)
    except (TypeError, ValueError):
        return ("__bad__", ())
    return (kind, nums)


def parse_uniform_bounds(bounds_str: str) -> Optional[Tuple[float, float]]:
    """Return ``(lo, hi)`` for a ``uniform <lo> <hi>`` string, else ``None``.

    Thin convenience wrapper over :func:`parse_bounds` for callers (e.g. the
    heuristic GP/noise checks) that only care about uniform priors.
    """
    parsed = parse_bounds(bounds_str)
    if parsed is None or parsed[0] != "uniform":
        return None
    lo, hi = parsed[1]
    return lo, hi
