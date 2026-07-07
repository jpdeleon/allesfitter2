"""Ephemeris auto-fill from the NASA Exoplanet Archive or a local TOI table.

Used to pre-populate a companion's period / epoch / duration in the fit form so
the user starts from a real prior instead of a guess (muscat-db's archive
auto-fill + provenance badge pattern). Every lookup is best-effort and
network-optional: any failure returns an empty :class:`Ephemeris` with
``source="none"`` rather than raising, so the form still renders offline.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

_TOI_RE = re.compile(r"(?:toi[\s_-]*)?(\d+(?:\.\d+)?)", re.IGNORECASE)


@dataclass
class Ephemeris:
    """A transiting-planet ephemeris, with provenance."""

    target: str
    period: float | None = None  # days
    epoch: float | None = None  # BJD
    duration: float | None = None  # days
    depth: float | None = None  # ppm
    source: str = "none"  # "nasa" | "toi_csv" | "none"

    @property
    def found(self) -> bool:
        return self.period is not None or self.epoch is not None


def _to_float(value) -> float | None:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return None if f != f else f  # drop NaN


def _find_col(columns, *needles: str) -> str | None:
    """First column whose lowercased name contains all *needles*."""
    for col in columns:
        low = str(col).lower()
        if all(n in low for n in needles):
            return col
    return None


def _toi_number(target: str) -> str | None:
    m = _TOI_RE.search(target or "")
    if not m:
        return None
    # normalize "6715" / "6715.01" -> integer host id "6715"
    return str(int(float(m.group(1))))


def from_toi_csv(target: str, csv_path: str | Path) -> Ephemeris:
    """Look up *target* in a local ExoFOP-style TOI table (CSV).

    Column names are matched leniently (e.g. ``Period (days)``,
    ``Epoch (BJD)``, ``Duration (hours)``, ``Depth (ppm)``).
    """
    csv_path = Path(csv_path)
    toi_num = _toi_number(target)
    if toi_num is None or not csv_path.is_file():
        return Ephemeris(target=target)

    try:
        import pandas as pd

        df = pd.read_csv(csv_path)
    except Exception:
        return Ephemeris(target=target)

    toi_col = _find_col(df.columns, "toi")
    if toi_col is None:
        return Ephemeris(target=target)

    host = df[df[toi_col].map(lambda v: _toi_number(str(v)) == toi_num)]
    if host.empty:
        return Ephemeris(target=target)
    row = host.iloc[0]

    def get(*needles):
        col = _find_col(df.columns, *needles)
        return _to_float(row[col]) if col is not None else None

    duration_hours = get("duration")
    return Ephemeris(
        target=target,
        period=get("period"),
        epoch=get("epoch"),
        duration=(duration_hours / 24.0) if duration_hours is not None else None,
        depth=get("depth"),
        source="toi_csv",
    )


def from_nasa(target: str) -> Ephemeris:
    """Query the NASA Exoplanet Archive via astroquery. Best-effort (network)."""
    try:
        from astroquery.ipac.nexsci.nasa_exoplanet_archive import NasaExoplanetArchive

        table = NasaExoplanetArchive.query_object(
            target, table="pscomppars", select="pl_orbper,pl_tranmid,pl_trandur,pl_trandep"
        )
    except Exception:
        return Ephemeris(target=target)
    if table is None or len(table) == 0:
        return Ephemeris(target=target)

    row = table[0]

    def get(key):
        try:
            return _to_float(row[key])
        except (KeyError, IndexError):
            return None

    duration_hours = get("pl_trandur")
    return Ephemeris(
        target=target,
        period=get("pl_orbper"),
        epoch=get("pl_tranmid"),
        duration=(duration_hours / 24.0) if duration_hours is not None else None,
        depth=get("pl_trandep"),
        source="nasa",
    )


def lookup(
    target: str,
    *,
    toi_csv: str | Path | None = None,
    allow_network: bool = True,
) -> Ephemeris:
    """Resolve an ephemeris, trying the local TOI table first, then NASA.

    Never raises; returns ``source="none"`` when nothing resolves.
    """
    if toi_csv is not None:
        eph = from_toi_csv(target, toi_csv)
        if eph.found:
            return eph
    if allow_network:
        eph = from_nasa(target)
        if eph.found:
            return eph
    return Ephemeris(target=target)
