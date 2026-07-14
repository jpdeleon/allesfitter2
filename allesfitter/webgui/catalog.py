"""Ephemeris auto-fill from the NASA Exoplanet Archive or a local TOI table.

Used to pre-populate a companion's period / epoch / duration in the fit form so
the user starts from a real prior instead of a guess (muscat-db's archive
auto-fill + provenance badge pattern). Every lookup is best-effort and
network-optional: any failure returns an empty :class:`Ephemeris` with
``source="none"`` rather than raising, so the form still renders offline.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
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


# --- sector / campaign / quarter availability -------------------------------
# So the Prepare form can show what "-s"/"-c"/"-q" would actually accept
# *before* the user submits and waits for a download. Queries the same
# archive `prepare_allesfit` itself uses, with the same result-object access
# pattern (``result.mission`` / ``result.author``, not ``.table.to_pandas()``)
# already proven in ``scripts/prepare_allesfit.py``.


@dataclass
class SectorAvailability:
    """Archived sectors/campaigns/quarters (+ pipelines) found for a target."""

    target: str
    mission: str
    segments: list[str] = field(default_factory=list)  # e.g. ["1", "2", "27"] or ["11a", "11b"]
    pipelines: list[str] = field(default_factory=list)  # lowercased author names found
    exptimes: list[int] = field(default_factory=list)  # unique exposure times (s), ascending
    error: str = ""

    @property
    def ok(self) -> bool:
        return not self.error


def _segment_label(mission_str: str) -> str:
    """Last whitespace token of a lightkurve ``mission`` string, kept as str.

    TESS sectors / Kepler quarters are integers ('TESS Sector 42' -> '42');
    K2 campaigns may carry an alpha suffix ('K2 Campaign 11a' -> '11a').
    """
    return str(mission_str).split()[-1]


def _exptime_values(search_result) -> list[float]:
    """Best-effort list of exposure times (seconds) from a lightkurve result.

    Reads the ``exptime`` column (``.table['exptime']``, falling back to an
    ``.exptime`` attribute), unwrapping an astropy ``Quantity``/``Column`` via
    ``.value``. Never raises — anything unparsable yields ``[]`` so the caller
    can still return sectors/pipelines.
    """
    raw = None
    table = getattr(search_result, "table", None)
    if table is not None:
        try:
            raw = table["exptime"]
        except Exception:
            raw = None
    if raw is None:
        raw = getattr(search_result, "exptime", None)
    if raw is None:
        return []
    values = getattr(raw, "value", raw)  # unwrap astropy Quantity/Column
    out: list[float] = []
    try:
        for v in values:
            try:
                out.append(float(v))
            except (TypeError, ValueError):
                continue
    except TypeError:
        return []
    return out


def _segment_sort_key(label: str):
    """Sort key for segment labels: numeric prefix, then alpha suffix."""
    s = str(label)
    n = 0
    while n < len(s) and s[n].isdigit():
        n += 1
    head = int(s[:n]) if n else float("inf")
    return (head, s[n:])


def _sector_query_name(target: str, id_type: str) -> str:
    id_type = (id_type or "name").strip().lower()
    target = str(target).strip()
    return {"tic": f"TIC {target}", "toi": f"TOI {target}", "ctoi": f"CTOI {target}"}.get(
        id_type, target
    )


def available_sectors(
    target: str,
    *,
    id_type: str = "name",
    mission: str = "tess",
    pipeline: str = "",
    allow_network: bool = True,
) -> SectorAvailability:
    """List archived sectors/campaigns/quarters for *target*, best-effort.

    Runs the same survey query ``prepare_allesfit`` uses
    (``lightkurve.search_lightcurve(query_name, mission=mission)``), so what's
    shown here is exactly what ``-s``/``-c``/``-q`` can select. Never raises —
    any failure comes back as ``SectorAvailability.error``.
    """
    target = (target or "").strip()
    mission = (mission or "tess").strip().lower()
    if not target:
        return SectorAvailability(target=target, mission=mission, error="target is required")
    if not allow_network:
        return SectorAvailability(target=target, mission=mission, error="network is disabled")

    try:
        import lightkurve as lk
    except Exception as exc:
        return SectorAvailability(
            target=target, mission=mission, error=f"lightkurve unavailable: {exc}"
        )

    query_name = _sector_query_name(target, id_type)
    try:
        result = lk.search_lightcurve(query_name, mission=mission)
    except Exception as exc:
        return SectorAvailability(target=target, mission=mission, error=str(exc))

    if len(result) == 0:
        return SectorAvailability(target=target, mission=mission, error="no light curves found")

    pipelines = sorted({str(a).lower() for a in result.author})

    pipeline = (pipeline or "").strip().lower()
    rows = result
    if pipeline:
        idx = [str(a).lower() == pipeline for a in result.author]
        rows = result[idx]
        if len(rows) == 0:
            return SectorAvailability(
                target=target,
                mission=mission,
                pipelines=pipelines,
                error=f"no light curves for pipeline {pipeline!r} (available: "
                f"{', '.join(pipelines) or 'none'})",
            )

    segments = sorted({_segment_label(m) for m in rows.mission}, key=_segment_sort_key)
    exptimes = sorted({round(e) for e in _exptime_values(rows) if e > 0})
    return SectorAvailability(
        target=target,
        mission=mission,
        segments=segments,
        pipelines=pipelines,
        exptimes=exptimes,
    )
