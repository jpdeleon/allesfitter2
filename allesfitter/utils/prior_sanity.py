"""Prior-bound sanity checks for GP and noise rows in ``params.csv``.

These checks are heuristic — they emit warnings when bounds let parameters
take physically implausible values for the dataset at hand. They never
raise; deliberate users can ignore the warning. The goal is to catch the
common foot-guns where a too-loose GP prior lets the kernel absorb the
transit, or a too-tight noise prior pegs the chain against the wall.

Triggered checks
----------------
``ln_err_flux_<inst>``
    upper bound implies σ > 10% of relative flux (any value above this
    is rarely physical for transit photometry).

``baseline_gp_matern32_lnsigma_flux_<inst>``
    1. upper bound implies σ_GP > 5% — transit-amplitude territory.
    2. upper bound implies σ_GP > 100× the per-cadence RMS — the GP
       can dominate the noise floor and reshape the transit.

``baseline_gp_matern32_lnrho_flux_<inst>``
    1. lower bound implies ρ < 2 × median cadence — GP fits individual
       cadences.
    2. lower bound implies ρ < 0.5 × transit duration (when known) —
       GP fits the transit shape itself.
    3. upper bound implies ρ > observation baseline — unconstrained,
       degenerate with a linear baseline slope.

Public API
----------
:func:`validate_gp_priors` — read the datadir, return a list of warning
strings, optionally piped to a logger.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

GP_LNSIGMA_PREFIX = "baseline_gp_matern32_lnsigma_flux_"
GP_LNRHO_PREFIX = "baseline_gp_matern32_lnrho_flux_"
LN_ERR_PREFIX = "ln_err_flux_"

_AMPL_REL_FLUX_LIMIT = 0.05            # σ_GP > 5% is transit-territory
_AMPL_VS_RMS_RATIO = 100.0             # σ_GP > 100× RMS dominates the noise
_LN_ERR_REL_FLUX_LIMIT = 0.10          # noise > 10% rel flux is suspicious
_RHO_MIN_CADENCES = 2.0                # ρ should span at least 2 cadences
_RHO_VS_TDUR_RATIO = 0.5               # ρ should exceed half tdur


def _read_csv_rows(path):
    out = []
    if not Path(path).exists():
        return out
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = [p.strip() for p in line.split(",")]
        out.append(parts)
    return out


def _parse_uniform(bounds_str):
    if not bounds_str:
        return None
    tokens = bounds_str.split()
    if len(tokens) != 3 or tokens[0] != "uniform":
        return None
    try:
        return float(tokens[1]), float(tokens[2])
    except (TypeError, ValueError):
        return None


def _settings_inst_phot(datadir):
    for parts in _read_csv_rows(Path(datadir) / "settings.csv"):
        if parts and parts[0] == "inst_phot" and len(parts) >= 2:
            return parts[1].split()
    return []


def _load_inst_data(datadir, inst):
    """Return (time, flux) as 1-D float arrays, or (None, None)."""
    fp = Path(datadir) / f"{inst}.csv"
    if not fp.exists():
        return None, None
    try:
        arr = np.genfromtxt(fp, delimiter=",", filling_values=np.nan, encoding="utf-8")
    except Exception:
        return None, None
    if arr.ndim != 2 or arr.shape[1] < 2:
        return None, None
    return arr[:, 0].astype(float), arr[:, 1].astype(float)


def _per_cadence_rms(flux):
    """Robust point-to-point scatter via MAD of successive differences."""
    f = np.asarray(flux, dtype=float)
    f = f[np.isfinite(f)]
    if f.size < 10:
        return float("nan")
    d = np.diff(f)
    if d.size == 0:
        return float("nan")
    mad = np.median(np.abs(d - np.median(d)))
    # σ ≈ 1.4826 * MAD / sqrt(2) for per-cadence noise from differences
    return 1.4826 * mad / math.sqrt(2.0)


def _baseline_span(time):
    t = np.asarray(time, dtype=float)
    t = t[np.isfinite(t)]
    if t.size < 2:
        return float("nan")
    return float(t.max() - t.min())


def _cadence(time):
    t = np.asarray(time, dtype=float)
    t = np.sort(t[np.isfinite(t)])
    if t.size < 2:
        return float("nan")
    return float(np.median(np.diff(t)))


def _w(msg):
    return "prior_sanity: " + msg


def validate_gp_priors(
    datadir,
    *,
    tdur_hours_by_companion: dict | None = None,
    log: Callable[[str], None] | None = None,
) -> list[str]:
    """Walk ``params.csv`` and warn about implausible GP / noise bounds.

    Parameters
    ----------
    datadir : str or pathlib.Path
        allesfitter working directory with ``params.csv``, ``settings.csv``
        and the per-instrument CSV files already written.
    tdur_hours_by_companion : dict, optional
        Mapping companion → transit duration in hours. Used to flag GP
        timescale lower bounds shorter than the transit. When omitted,
        falls back to a permissive 0.1 d (~2.4 h) default.
    log : callable, optional
        Sink for warnings; receives one fully-formatted message per call.
        When ``None`` (default) warnings are returned but not emitted —
        useful for tests.

    Returns
    -------
    list[str]
        Every warning message emitted (also useful for tests).
    """
    msgs: list[str] = []

    params_rows = _read_csv_rows(Path(datadir) / "params.csv")
    if not params_rows:
        return msgs

    insts = _settings_inst_phot(datadir)
    data_cache = {inst: _load_inst_data(datadir, inst) for inst in insts}

    if tdur_hours_by_companion:
        tdur_days = float(np.median(
            [t / 24.0 for t in tdur_hours_by_companion.values() if t]
        ))
    else:
        tdur_days = 0.1   # ~2.4 h fallback

    for row in params_rows:
        if len(row) < 4:
            continue
        name, _value, _fit, bounds = row[0], row[1], row[2], row[3]
        bnds = _parse_uniform(bounds)
        if bnds is None:
            continue
        lo, hi = bnds

        if name.startswith(LN_ERR_PREFIX):
            sigma_hi = math.exp(hi)
            if sigma_hi > _LN_ERR_REL_FLUX_LIMIT:
                msgs.append(_w(
                    f"{name}: upper bound exp({hi}) ≈ {sigma_hi:.3f} "
                    f"> {_LN_ERR_REL_FLUX_LIMIT:.0%} relative flux — review noise prior."
                ))
            continue

        if name.startswith(GP_LNSIGMA_PREFIX):
            inst = name[len(GP_LNSIGMA_PREFIX):]
            sigma_hi = math.exp(hi)
            if sigma_hi > _AMPL_REL_FLUX_LIMIT:
                msgs.append(_w(
                    f"{name}: upper bound exp({hi}) ≈ {sigma_hi:.3f} > "
                    f"{_AMPL_REL_FLUX_LIMIT:.0%} relative flux — GP may absorb the transit."
                ))
            _t, f = data_cache.get(inst, (None, None))
            if f is not None:
                rms = _per_cadence_rms(f)
                if math.isfinite(rms) and sigma_hi > _AMPL_VS_RMS_RATIO * rms:
                    msgs.append(_w(
                        f"{name}: upper bound exp({hi}) ≈ {sigma_hi:.3f} > "
                        f"{_AMPL_VS_RMS_RATIO:.0f}× per-cadence RMS ({rms:.1e}) "
                        f"— GP can dominate the noise floor."
                    ))
            continue

        if name.startswith(GP_LNRHO_PREFIX):
            inst = name[len(GP_LNRHO_PREFIX):]
            rho_lo, rho_hi = math.exp(lo), math.exp(hi)
            t, _f = data_cache.get(inst, (None, None))
            if t is not None:
                cad = _cadence(t)
                base = _baseline_span(t)
                if math.isfinite(cad) and rho_lo < _RHO_MIN_CADENCES * cad:
                    msgs.append(_w(
                        f"{name}: lower bound exp({lo}) ≈ {rho_lo:.4f} d < "
                        f"{_RHO_MIN_CADENCES:.0f}× cadence ({cad:.4f} d) "
                        f"— GP could fit individual cadences."
                    ))
                if math.isfinite(base) and rho_hi > base:
                    msgs.append(_w(
                        f"{name}: upper bound exp({hi}) ≈ {rho_hi:.3f} d > "
                        f"observation baseline ({base:.3f} d) — degenerate "
                        f"with baseline slope."
                    ))
            if rho_lo < _RHO_VS_TDUR_RATIO * tdur_days:
                msgs.append(_w(
                    f"{name}: lower bound exp({lo}) ≈ {rho_lo:.4f} d < "
                    f"{_RHO_VS_TDUR_RATIO:.0%}× transit duration "
                    f"(~{tdur_days:.4f} d) — GP could fit the transit shape."
                ))
            continue

    if log is not None:
        for m in msgs:
            log(m)
    return msgs
