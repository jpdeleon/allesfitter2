"""Read transit parameters and the source light curve out of a quicklook TLS
results HDF5 file.

The `quicklook <https://github.com/jpdeleon/quicklook>`_ pipeline runs
``transitleastsquares`` (TLS) on a light curve and saves the results object to
an HDF5 file (e.g. ``TIC123_s1_tls.h5`` for the primary candidate, or
``TIC123_s1_tls_p2.h5`` for an additional companion found by its iterative
multi-planet search). ``read_h5_transit_params`` extracts the subset of that
TLS results dict needed to seed ``prepare_allesfit.py --h5`` for a raw
``-tic`` target or a new ``-toi`` companion: period, epoch, duration, and
depth (plus errors, where available). ``read_h5_lightcurve`` extracts the
light curve TLS itself ran on, so ``--h5`` can reuse it instead of
re-downloading a segment already on disk in the h5 file.

Two on-disk layouts are supported, matching quicklook's own ``h5io`` loader:
scalars JSON-encoded in a ``_scalar_data`` file attribute (current format),
and scalars stored directly as file-level attributes with tuples as
deepdish/flammkuchen-style groups (legacy format).

``write_h5_tls_result`` is the writer counterpart, used by
``allesfitter.detection.blind_search`` to save each iterative-search
candidate in the same current-format layout, so it round-trips through
``read_h5_transit_params`` / ``read_h5_lightcurve`` exactly like a quicklook
run's own output.
"""

from __future__ import annotations

import json

import h5py
import numpy as np

MISSION_BJD_OFFSET = {"tess": 2_457_000.0, "k2": 2_454_833.0, "kepler": 2_454_833.0}

_LEGACY_ATTR_SKIP = {
    "CLASS",
    "DEEPDISH_IO_VERSION",
    "PYTABLES_FORMAT_VERSION",
    "TITLE",
    "VERSION",
}


def _read_scalar_attrs(h5file: h5py.File) -> dict:
    if "_scalar_data" in h5file.attrs:
        return json.loads(h5file.attrs["_scalar_data"])
    return {k: v for k, v in h5file.attrs.items() if k not in _LEGACY_ATTR_SKIP}


def _read_legacy_tuple(group: h5py.Group) -> tuple | None:
    items = []
    index = 0
    while f"i{index}" in group.attrs:
        items.append(group.attrs[f"i{index}"])
        index += 1
    return tuple(items) if items else None


def _read_value(h5file: h5py.File, scalars: dict, key: str):
    if key in scalars:
        return scalars[key]
    if key not in h5file:
        return None
    obj = h5file[key]
    if isinstance(obj, h5py.Group):
        return _read_legacy_tuple(obj)
    return obj[()]


def _fractional_depth(raw_depth) -> float | None:
    """Convert a TLS ``depth`` (in-transit relative flux, near 1) to the
    fractional dimming ``prepare_allesfit.py`` expects (near 0)."""
    if raw_depth is None:
        return None
    value = float(raw_depth)
    if value > 1.0:
        value /= 1e3
    return value if value < 0.5 else 1.0 - value


def read_h5_transit_params(path: str, mission: str = "tess") -> dict:
    """Extract raw-TIC transit parameters from a quicklook TLS ``.h5`` file.

    Returns a dict with the same keys as ``prepare_allesfit.py``'s
    ``--period/--epoch/--duration/--depth`` (and ``*_err``) options, in the
    units those options expect: days, BJD, hours, and ppm. Fields the TLS
    results dict does not carry (``epoch_err``, ``duration_err``, and
    ``depth_err`` when no ``depth_mean`` uncertainty is available) are left
    as ``None`` for the caller to fill in explicitly.
    """
    offset = MISSION_BJD_OFFSET.get(mission.lower(), MISSION_BJD_OFFSET["tess"])

    with h5py.File(path, "r") as h5file:
        scalars = _read_scalar_attrs(h5file)
        period = _read_value(h5file, scalars, "period")
        period_err = _read_value(h5file, scalars, "period_uncertainty")
        t0 = _read_value(h5file, scalars, "T0")
        duration_days = _read_value(h5file, scalars, "duration")
        raw_depth = _read_value(h5file, scalars, "depth")
        depth_mean = _read_value(h5file, scalars, "depth_mean")

    values = {
        "period": float(period) if period is not None else None,
        "period_err": float(period_err) if period_err is not None else None,
        "epoch": float(t0) + offset if t0 is not None else None,
        "epoch_err": None,
        "duration": float(duration_days) * 24.0 if duration_days is not None else None,
        "duration_err": None,
        "depth": None,
        "depth_err": None,
    }

    frac_depth = _fractional_depth(raw_depth)
    if frac_depth is not None:
        values["depth"] = frac_depth * 1e6

    if depth_mean is not None and len(depth_mean) == 2:
        values["depth_err"] = float(depth_mean[1]) * 1e6

    return values


def read_h5_lightcurve(path: str) -> dict | None:
    """Read the raw (undetrended, normalized) light curve quicklook fed into
    TLS, plus the run metadata (pipeline/sector/exptime/cadence) it stored
    alongside it.

    ``time`` is left in raw BTJD (TESS time - 2457000), unlike
    :func:`read_h5_transit_params`'s ``epoch`` — this is meant to be fed
    straight into ``lightkurve.LightCurve(time=..., format="btjd")``, which
    expects that same convention. Returns ``None`` if the file doesn't carry
    a raw light curve (``time_raw``/``flux_raw``/``err_raw``), e.g. an older
    quicklook run saved without ``time_raw``/``flux_raw``/``err_raw``, or a
    file that only has the TLS results and not the source light curve.
    """
    with h5py.File(path, "r") as h5file:
        scalars = _read_scalar_attrs(h5file)
        time_raw = _read_value(h5file, scalars, "time_raw")
        flux_raw = _read_value(h5file, scalars, "flux_raw")
        err_raw = _read_value(h5file, scalars, "err_raw")
        pipeline = _read_value(h5file, scalars, "pipeline")
        sector = _read_value(h5file, scalars, "sector")
        exptime = _read_value(h5file, scalars, "exptime")
        cadence = _read_value(h5file, scalars, "cadence")

    if time_raw is None or flux_raw is None or err_raw is None:
        return None

    return {
        "time": np.asarray(time_raw, dtype=float),
        "flux": np.asarray(flux_raw, dtype=float),
        "flux_err": np.asarray(err_raw, dtype=float),
        "pipeline": str(pipeline).lower() if pipeline is not None else None,
        "sector": int(sector) if sector is not None else None,
        "exptime": float(exptime) if exptime is not None else None,
        "cadence": str(cadence) if cadence is not None else None,
    }


def write_h5_tls_result(
    path: str,
    results: dict,
    *,
    lightcurve: dict | None = None,
    mission: str = "tess",
    pipeline: str | None = None,
    sector: int | None = None,
    exptime: float | None = None,
    cadence: str | None = None,
) -> None:
    """Write a ``transitleastsquares`` results dict to a quicklook-compatible
    ``.h5`` file, in the current ``_scalar_data``-JSON layout.

    ``results`` is one entry of the ``results_all`` list returned by
    :func:`allesfitter.detection.transit_search.tls_search` (a TLS results
    dict with ``correct_duration`` added). Only the scalar fields
    :func:`read_h5_transit_params` understands are kept.

    ``lightcurve``, if given, is ``{"time": ..., "flux": ..., "flux_err":
    ...}`` with ``time`` in the same absolute BJD convention as the rest of
    allesfitter (i.e. *not* yet mission-offset) — it is converted to the
    mission's raw time convention (BTJD for TESS, etc.) before writing, the
    same convention :func:`read_h5_lightcurve` and ``prepare_allesfit.py``
    expect. ``pipeline``/``sector``/``exptime``/``cadence`` let a caller that
    knows this metadata enable ``_h5_lightcurve_matches_segment`` reuse in
    ``prepare_allesfit.py``; left out, the light curve still seeds
    period/epoch/duration/depth, just without the reuse-instead-of-redownload
    shortcut.
    """
    offset = MISSION_BJD_OFFSET.get(mission.lower(), MISSION_BJD_OFFSET["tess"])

    scalars: dict = {
        "period": float(results["period"]),
        "period_uncertainty": float(results["period_uncertainty"]),
        "T0": float(results["T0"]) - offset,
        "duration": float(results.get("correct_duration", results["duration"])),
        "depth": float(results["depth"]),
    }
    depth_mean = results.get("depth_mean")
    if depth_mean is not None and len(depth_mean) == 2:
        scalars["depth_mean"] = [float(depth_mean[0]), float(depth_mean[1])]
    if pipeline is not None:
        scalars["pipeline"] = pipeline
    if sector is not None:
        scalars["sector"] = int(sector)
    if exptime is not None:
        scalars["exptime"] = float(exptime)
    if cadence is not None:
        scalars["cadence"] = cadence

    with h5py.File(path, "w") as h5file:
        if lightcurve is not None:
            h5file.create_dataset(
                "time_raw", data=np.asarray(lightcurve["time"], dtype=float) - offset
            )
            h5file.create_dataset("flux_raw", data=np.asarray(lightcurve["flux"], dtype=float))
            h5file.create_dataset("err_raw", data=np.asarray(lightcurve["flux_err"], dtype=float))
        h5file.attrs["_scalar_data"] = json.dumps(scalars)
