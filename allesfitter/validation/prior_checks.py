"""Heuristic prior-bound checks for GP and noise rows in ``params.csv``.

The warning-only sibling of :mod:`allesfitter.validation.config_checks`: where
that module *raises* on unambiguous structural errors, these checks emit
warnings when bounds let parameters take physically implausible values for the
dataset at hand. They never raise; deliberate users can ignore the warning. The
goal is to catch the common foot-guns where a too-loose GP prior lets the kernel
absorb the transit, or a too-tight noise prior pegs the chain against the wall.

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

``<companion>_sbratio_<inst>`` vs ``secondary_eclipse`` / ``phase_curve``
    1. ``secondary_eclipse`` on but the ratio is fixed at 0 / its default —
       no secondary eclipse is modelled (a fixed *non-zero* ratio is allowed).
    2. the ratio is fitted but ``secondary_eclipse`` is off — a free parameter
       with no secondary eclipse model.

Public API
----------
:func:`validate_gp_priors` — read the datadir, return a list of warning
strings, optionally piped to a logger.
:func:`transit_duration_days`, :func:`transit_duration_hours_by_companion` —
transit-duration helpers also reused by output/visualization code.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Callable

import numpy as np

from .parsing import parse_uniform_bounds, read_csv_rows, read_settings

GP_LNSIGMA_PREFIX = "baseline_gp_matern32_lnsigma_flux_"
GP_LNRHO_PREFIX = "baseline_gp_matern32_lnrho_flux_"
LN_ERR_PREFIX = "ln_err_flux_"

_AMPL_REL_FLUX_LIMIT = 0.05            # σ_GP > 5% is transit-territory
_AMPL_VS_RMS_RATIO = 100.0             # σ_GP > 100× RMS dominates the noise
_LN_ERR_REL_FLUX_LIMIT = 0.10          # noise > 10% rel flux is suspicious
_RHO_MIN_CADENCES = 2.0                # ρ should span at least 2 cadences
_RHO_VS_TDUR_RATIO = 0.5               # ρ should exceed half tdur


def _settings_inst_phot(datadir):
    for parts in read_csv_rows(Path(datadir) / "settings.csv"):
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
    return "prior_checks: " + msg


def transit_duration_days(per, rsuma, cosi, k):
    """Transit duration from per, R/a sum, cos i, and Rp/Rs.

    Uses the standard chord-length formula:

        t_dur = (P / pi) * arcsin( rsuma * sqrt((1+k)^2 - b^2) / sin i )

    with ``b = cos(i) * (1 + k) / rsuma`` (impact parameter from R/a sum).
    Returns NaN when inputs are non-positive, when sin(i) collapses to 0,
    or when the chord is imaginary (grazing / non-transiting geometries).
    """
    per = float(per); rsuma = float(rsuma); cosi = float(cosi); k = float(k)
    if not (per > 0 and rsuma > 0 and 0.0 <= abs(cosi) <= 1.0 and k > 0):
        return float("nan")
    sini = math.sqrt(max(1.0 - cosi * cosi, 0.0))
    if sini == 0.0:
        return float("nan")
    b = cosi * (1.0 + k) / rsuma
    chord_sq = (1.0 + k) ** 2 - b * b
    if chord_sq <= 0.0:
        return float("nan")
    arg = rsuma * math.sqrt(chord_sq) / sini
    if arg <= 0.0 or arg > 1.0:
        return float("nan")
    return per / math.pi * math.asin(arg)


def transit_duration_hours_by_companion(datadir):
    """Build ``{companion: tdur_hours}`` from initial-guess values in params.csv.

    For each companion ``<c>`` (single-letter, e.g. ``b``, ``c``, ...),
    looks up ``<c>_period``, ``<c>_rsuma``, ``<c>_cosi`` and a radius-ratio
    row (``<c>_rr`` or the first ``<c>_rr_<bandpass>`` match for chromatic
    fits). When all four are present and yield a finite duration, the
    companion is included in the returned dict. Returns ``None`` when no
    companion could be resolved (caller falls back to the legacy default).
    """
    rows = read_csv_rows(Path(datadir) / "params.csv")
    if not rows:
        return None
    # name -> initial value (skip rows that don't parse as float)
    values = {}
    for row in rows:
        if len(row) < 2:
            continue
        try:
            values[row[0]] = float(row[1])
        except (TypeError, ValueError):
            continue

    by_companion: dict[str, float] = {}
    for name in values:
        # cheap heuristic: <c>_period uniquely identifies a companion
        if not name.endswith("_period"):
            continue
        companion = name[: -len("_period")]
        per = values[name]
        rsuma = values.get(f"{companion}_rsuma")
        cosi = values.get(f"{companion}_cosi")
        # Radius-ratio key: prefer <c>_rr; fall back to the first <c>_rr_*
        # (chromatic mode uses per-bandpass rr columns).
        k = values.get(f"{companion}_rr")
        if k is None:
            for key in values:
                if key.startswith(f"{companion}_rr_"):
                    k = values[key]
                    break
        if None in (per, rsuma, cosi, k):
            continue
        tdur_days = transit_duration_days(per, rsuma, cosi, k)
        if math.isfinite(tdur_days) and tdur_days > 0:
            by_companion[companion] = tdur_days * 24.0
    return by_companion or None


def _settings_is_true(value) -> bool:
    """Mirror ``basement.set_bool``: ``'true'`` / ``'1'`` (any case) → True."""
    return str(value).strip().lower() in ("true", "1")


def _sbratio_is_zero_or_empty(value: str) -> bool:
    """True when a fixed sbratio is 0 or blank (i.e. defaults to 0).

    A blank value lets allesfitter fill its default (0); an explicit 0 is the
    same. Either way no secondary eclipse is produced. A non-numeric value is
    treated as 'not zero' here (config_checks flags malformed numbers), so it
    does not trigger a spurious secondary-eclipse warning.
    """
    if value == "":
        return True
    try:
        return float(value) == 0.0
    except (TypeError, ValueError):
        return False


def check_secondary_eclipse_sbratio(datadir) -> list[str]:
    """Warn when ``secondary_eclipse`` and the surface brightness ratio disagree.

    The secondary eclipse depth is governed by ``<companion>_sbratio_<inst>``.
    This heuristic (warning-only) check flags the two inconsistent setups:

    - ``secondary_eclipse=True`` (also forced by ``phase_curve=True``) while the
      ratio is **fixed at 0 / its default** — no secondary eclipse is modelled
      despite the setting. A ratio fixed at a *known non-zero* value is allowed
      (valid "known sbratio" setup) and is **not** warned about.
    - ``fit=1`` on a ``*_sbratio_*`` row while ``secondary_eclipse`` is off — a
      free parameter is sampled with no secondary eclipse model.

    Coupled rows are skipped (they inherit ``fit`` from their partner).
    """
    settings = read_settings(datadir)
    sec = _settings_is_true(settings.get("secondary_eclipse", ""))
    pc = _settings_is_true(settings.get("phase_curve", ""))
    effective = sec or pc

    msgs: list[str] = []
    for row in read_csv_rows(Path(datadir) / "params.csv"):
        if len(row) < 3 or "sbratio" not in row[0].split("_"):
            continue
        if len(row) >= 7 and row[6].strip():  # coupled row inherits fit
            continue
        name, value, fit = row[0], row[1].strip(), row[2]
        if effective:
            if fit != "1" and _sbratio_is_zero_or_empty(value):
                trigger = (
                    "secondary_eclipse=True"
                    if sec
                    else "phase_curve=True (forces secondary_eclipse)"
                )
                msgs.append(_w(
                    f"{name}: {trigger} but the surface brightness ratio is fixed "
                    f"at {value or '0 (default)'} (fit=0) — no secondary eclipse "
                    f"will be modelled. Set fit=1, or fix it at a known non-zero value."
                ))
        elif fit == "1":
            msgs.append(_w(
                f"{name}: fitted (fit=1) but secondary_eclipse is off — the surface "
                f"brightness ratio is sampled with no secondary eclipse model. "
                f"Enable secondary_eclipse, or fix it (fit=0)."
            ))
    return msgs


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
        timescale lower bounds shorter than the transit. When ``None``
        (default), the validator computes a duration for each companion
        from its initial-guess orbital row in ``params.csv``
        (``<c>_period``, ``<c>_rsuma``, ``<c>_cosi``, ``<c>_rr`` or
        ``<c>_rr_<bandpass>``) via the chord-length formula in
        :func:`transit_duration_days`. Falls back to a permissive
        0.1 d (~2.4 h) only when no companion can be resolved.
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

    params_rows = read_csv_rows(Path(datadir) / "params.csv")
    if not params_rows:
        return msgs

    insts = _settings_inst_phot(datadir)
    data_cache = {inst: _load_inst_data(datadir, inst) for inst in insts}

    # Resolve transit-duration source (caller > derived from params.csv > fallback).
    if tdur_hours_by_companion is None:
        tdur_hours_by_companion = transit_duration_hours_by_companion(datadir)
    if tdur_hours_by_companion:
        tdur_days = float(np.median(
            [t / 24.0 for t in tdur_hours_by_companion.values() if t]
        ))
    else:
        tdur_days = 0.1   # ~2.4 h fallback when no orbital row could be parsed

    for row in params_rows:
        if len(row) < 4:
            continue
        name, _value, _fit, bounds = row[0], row[1], row[2], row[3]
        bnds = parse_uniform_bounds(bounds)
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

    # Non-GP heuristic: secondary_eclipse ⇔ surface-brightness-ratio consistency.
    msgs += check_secondary_eclipse_sbratio(datadir)

    if log is not None:
        for m in msgs:
            log(m)
    return msgs
