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

from ..orbits import circular_transit_duration
from .parsing import parse_uniform_bounds, read_csv_rows, read_settings

GP_LNSIGMA_PREFIX = "baseline_gp_matern32_lnsigma_flux_"
GP_LNRHO_PREFIX = "baseline_gp_matern32_lnrho_flux_"
LN_ERR_PREFIX = "ln_err_flux_"

_AMPL_REL_FLUX_LIMIT = 0.05  # σ_GP > 5% is transit-territory
_AMPL_VS_RMS_RATIO = 100.0  # σ_GP > 100× RMS dominates the noise
_LN_ERR_REL_FLUX_LIMIT = 0.10  # noise > 10% rel flux is suspicious
_RHO_MIN_CADENCES = 2.0  # ρ should span at least 2 cadences
_RHO_VS_TDUR_RATIO = 0.5  # ρ should exceed half tdur
_BINNING_VS_CADENCE_RATIO = 2.0  # binning below this × cadence is ~a no-op
_BINNING_VS_TDUR_RATIO = 0.5  # binning above this × tdur smears the transit


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

    Delegates to the same circular-orbit equation used to initialize prepared
    fits. Returns NaN for invalid or non-transiting geometries.
    """
    return circular_transit_duration(per, rsuma, cosi, k)


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
                msgs.append(
                    _w(
                        f"{name}: {trigger} but the surface brightness ratio is fixed "
                        f"at {value or '0 (default)'} (fit=0) — no secondary eclipse "
                        f"will be modelled. Set fit=1, or fix it at a known non-zero value."
                    )
                )
        elif fit == "1":
            msgs.append(
                _w(
                    f"{name}: fitted (fit=1) but secondary_eclipse is off — the surface "
                    f"brightness ratio is sampled with no secondary eclipse model. "
                    f"Enable secondary_eclipse, or fix it (fit=0)."
                )
            )
    return msgs


def _effective_binning(settings: dict, inst: str) -> float | None:
    """Resolve the bin width (days) that applies to ``inst``.

    A per-instrument ``binning_<inst>`` key overrides the global ``binning``
    for that instrument only; an explicit empty/None override turns binning off
    even when the global default bins everything else. Returns ``None`` when no
    (valid, positive) binning applies — malformed/≤0 values are left to
    ``config_checks`` to raise.
    """
    key = "binning_" + inst
    raw = settings.get(key) if key in settings else settings.get("binning")
    raw = (raw or "").strip()
    if raw == "" or raw.lower() == "none":
        return None
    try:
        val = float(raw)
    except (TypeError, ValueError):
        return None
    return val if val > 0 else None


def check_binning(datadir) -> list[str]:
    """Warn about risky ``binning`` / ``binning_<inst>`` settings (width in days).

    Errors (non-numeric / ≤ 0 / ≥ baseline) are handled elsewhere
    (``config_checks.check_binning_value`` and ``Basement.load_data``); this
    warning-only check flags values that are *valid but risky*. The effective
    bin width is resolved per instrument (a ``binning_<inst>`` override falls
    back to the global ``binning``), so each instrument is judged against its
    own data:

    1. coarser than ``0.5 ×`` the shortest transit duration — binning may smear
       the transit shape;
    2. finer than ``2 ×`` an instrument's native cadence — little/no binning
       actually happens;
    3. set with an explicit ``t_exp_<inst>`` that does **not** match the bin
       width — binned points integrate over the bin width, so t_exp normally
       equals the binning. (A *missing* ``t_exp_<inst>`` is fine: ``Basement``
       auto-derives ``t_exp_<inst>`` = bin width, so it is not flagged here.)
    """
    settings = read_settings(datadir)

    #::: tdur is companion-level (not per-instrument); resolve once.
    tdur_days = None
    tdur_by_c = transit_duration_hours_by_companion(datadir)
    if tdur_by_c:
        finite = [t / 24.0 for t in tdur_by_c.values() if t]
        if finite:
            tdur_days = min(finite)

    msgs: list[str] = []
    seen_smear: set[float] = set()
    for inst in _settings_inst_phot(datadir):
        binning = _effective_binning(settings, inst)
        if binning is None:
            continue

        #::: 1) transit smearing (per effective width; de-duplicated across insts)
        if (
            tdur_days is not None
            and binning > _BINNING_VS_TDUR_RATIO * tdur_days
            and binning not in seen_smear
        ):
            seen_smear.add(binning)
            msgs.append(
                _w(
                    f"binning={binning:g} d > {_BINNING_VS_TDUR_RATIO:.0%} of the shortest "
                    f"transit duration (~{tdur_days:.4f} d) — binning may smear the transit."
                )
            )

        #::: 2) native cadence (no-op if finer than the data)
        t, _f = _load_inst_data(datadir, inst)
        if t is not None:
            cad = _cadence(t)
            if math.isfinite(cad) and binning < _BINNING_VS_CADENCE_RATIO * cad:
                msgs.append(
                    _w(
                        f"binning={binning:g} d < {_BINNING_VS_CADENCE_RATIO:.0f}× the native "
                        f"cadence of '{inst}' ({cad:.5f} d) — little or no binning will occur."
                    )
                )

        #::: 3) explicit t_exp that disagrees with the bin width. A missing
        #::: t_exp is fine — Basement auto-sets t_exp_<inst> = bin width — so
        #::: only a present-but-mismatched single value is flagged here.
        t_exp_raw = (settings.get("t_exp_" + inst, "") or "").strip()
        if t_exp_raw and t_exp_raw.lower() != "none":
            tokens = t_exp_raw.split()
            if len(tokens) == 1:  # single scalar; arrays are left alone
                try:
                    t_exp_val = float(tokens[0])
                except ValueError:
                    t_exp_val = None
                if t_exp_val is not None and abs(t_exp_val - binning) > 1e-9 * max(1.0, binning):
                    msgs.append(
                        _w(
                            f"binning={binning:g} d is set for '{inst}' but t_exp_{inst}"
                            f"={t_exp_val:g} d differs from the bin width; binned points "
                            f"integrate over the bin width, so t_exp_{inst} usually "
                            f"equals it. Remove t_exp_{inst} to auto-match the binning."
                        )
                    )

    return msgs


def _params_values(datadir) -> dict[str, float]:
    """Return ``{name: initial_value}`` for rows whose value parses as a float."""
    values: dict[str, float] = {}
    for row in read_csv_rows(Path(datadir) / "params.csv"):
        if len(row) < 2:
            continue
        try:
            values[row[0]] = float(row[1])
        except (TypeError, ValueError):
            continue
    return values


def check_radius_ratio(datadir) -> list[str]:
    """Warn when a radius ratio ``<c>_rr`` exceeds 1 (companion larger than host).

    ``rr`` > 1 is physical for some eclipsing binaries (a giant primary eclipsed
    by a smaller-but-brighter companion is the usual transit case; the reverse
    gives ``rr`` > 1), so this is a *warning*, not an error. Flags both the
    initial value and a uniform prior whose upper bound exceeds 1.
    """
    msgs: list[str] = []
    for row in read_csv_rows(Path(datadir) / "params.csv"):
        if not row or "_rr" not in row[0]:
            continue
        name = row[0]
        #::: match <c>_rr or chromatic <c>_rr_<bandpass>
        if not (name.endswith("_rr") or "_rr_" in name):
            continue
        if len(row) >= 2:
            try:
                if float(row[1]) > 1.0:
                    msgs.append(
                        _w(
                            f"{name}: initial value {float(row[1]):g} > 1 — companion "
                            f"radius exceeds the host's; unusual for a transiting planet."
                        )
                    )
            except (TypeError, ValueError):
                pass
        if len(row) >= 4 and row[2] == "1":
            bnds = parse_uniform_bounds(row[3])
            if bnds is not None and bnds[1] > 1.0:
                msgs.append(
                    _w(
                        f"{name}: uniform prior upper bound {bnds[1]:g} > 1 — allows "
                        f"a companion larger than the host."
                    )
                )
    return msgs


def check_spin_orbit_angle(datadir) -> list[str]:
    """Warn when a projected spin-orbit angle ``*_lambda`` leaves [-180, 180] deg.

    ``lambda`` is an angle in degrees; values outside ``[-180, 180]`` are not
    wrong (they wrap), but usually signal a units or sign mistake. Warning-only.
    """
    msgs: list[str] = []
    for row in read_csv_rows(Path(datadir) / "params.csv"):
        if not row or "lambda" not in row[0].split("_"):
            continue
        name = row[0]
        if len(row) >= 2:
            try:
                val = float(row[1])
            except (TypeError, ValueError):
                continue
            if not (-180.0 <= val <= 180.0):
                msgs.append(
                    _w(
                        f"{name}: initial value {val:g} deg is outside [-180, 180] — "
                        f"check units/sign of the projected spin-orbit angle."
                    )
                )
    return msgs


def check_transit_geometry(datadir) -> list[str]:
    """Warn when initial orbital values imply a non-transiting geometry.

    The impact parameter ``b = cos(i) * (1 + k) / rsuma`` must satisfy
    ``b <= 1 + k`` for the companion to cross the stellar disk at all. When the
    initial guess gives ``b > 1 + k`` the transit model produces no eclipse, so
    the fit starts with zero signal — almost always a mistake. Warning-only
    (a grazing/near-miss start can still be deliberate).
    """
    values = _params_values(datadir)
    msgs: list[str] = []
    for name in values:
        if not name.endswith("_period"):
            continue
        companion = name[: -len("_period")]
        rsuma = values.get(f"{companion}_rsuma")
        cosi = values.get(f"{companion}_cosi")
        k = values.get(f"{companion}_rr")
        if k is None:
            for key in values:
                if key.startswith(f"{companion}_rr_"):
                    k = values[key]
                    break
        if None in (rsuma, cosi, k) or rsuma <= 0:
            continue
        b = abs(cosi) * (1.0 + k) / rsuma
        if b > 1.0 + k:
            msgs.append(
                _w(
                    f"companion '{companion}': initial impact parameter "
                    f"b=cosi*(1+k)/rsuma={b:g} > 1+k={1.0 + k:g} — the companion "
                    f"does not transit with these initial values (no eclipse modelled)."
                )
            )
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
        valid_tdur_days = [
            float(t) / 24.0
            for t in tdur_hours_by_companion.values()
            if t is not None and math.isfinite(float(t)) and float(t) > 0.0
        ]
        tdur_days = max(valid_tdur_days) if valid_tdur_days else 0.1
    else:
        tdur_days = 0.1  # ~2.4 h fallback when no orbital row could be parsed

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
                msgs.append(
                    _w(
                        f"{name}: upper bound exp({hi}) ≈ {sigma_hi:.3f} "
                        f"> {_LN_ERR_REL_FLUX_LIMIT:.0%} relative flux — review noise prior."
                    )
                )
            continue

        if name.startswith(GP_LNSIGMA_PREFIX):
            inst = name[len(GP_LNSIGMA_PREFIX) :]
            sigma_hi = math.exp(hi)
            if sigma_hi > _AMPL_REL_FLUX_LIMIT:
                msgs.append(
                    _w(
                        f"{name}: upper bound exp({hi}) ≈ {sigma_hi:.3f} > "
                        f"{_AMPL_REL_FLUX_LIMIT:.0%} relative flux — GP may absorb the transit."
                    )
                )
            _t, f = data_cache.get(inst, (None, None))
            if f is not None:
                rms = _per_cadence_rms(f)
                if math.isfinite(rms) and sigma_hi > _AMPL_VS_RMS_RATIO * rms:
                    msgs.append(
                        _w(
                            f"{name}: upper bound exp({hi}) ≈ {sigma_hi:.3f} > "
                            f"{_AMPL_VS_RMS_RATIO:.0f}× per-cadence RMS ({rms:.1e}) "
                            f"— GP can dominate the noise floor."
                        )
                    )
            continue

        if name.startswith(GP_LNRHO_PREFIX):
            inst = name[len(GP_LNRHO_PREFIX) :]
            rho_lo, rho_hi = math.exp(lo), math.exp(hi)
            t, _f = data_cache.get(inst, (None, None))
            if t is not None:
                cad = _cadence(t)
                base = _baseline_span(t)
                if math.isfinite(cad) and rho_lo < _RHO_MIN_CADENCES * cad:
                    msgs.append(
                        _w(
                            f"{name}: lower bound exp({lo}) ≈ {rho_lo:.4f} d < "
                            f"{_RHO_MIN_CADENCES:.0f}× cadence ({cad:.4f} d) "
                            f"— GP could fit individual cadences."
                        )
                    )
                if math.isfinite(base) and rho_hi > base:
                    msgs.append(
                        _w(
                            f"{name}: upper bound exp({hi}) ≈ {rho_hi:.3f} d > "
                            f"observation baseline ({base:.3f} d) — degenerate "
                            f"with baseline slope."
                        )
                    )
            if rho_lo < _RHO_VS_TDUR_RATIO * tdur_days:
                msgs.append(
                    _w(
                        f"{name}: lower bound exp({lo}) ≈ {rho_lo:.4f} d < "
                        f"{_RHO_VS_TDUR_RATIO:.0%}× longest transit duration "
                        f"(~{tdur_days:.4f} d) — GP could fit the transit shape."
                    )
                )
            continue

    # Non-GP heuristic: secondary_eclipse ⇔ surface-brightness-ratio consistency.
    msgs += check_secondary_eclipse_sbratio(datadir)

    # Non-GP heuristic: risky input-binning widths.
    msgs += check_binning(datadir)

    # Non-GP heuristics: physically-debatable model parameters (warn, never raise).
    msgs += check_radius_ratio(datadir)
    msgs += check_spin_orbit_angle(datadir)
    msgs += check_transit_geometry(datadir)

    if log is not None:
        for m in msgs:
            log(m)
    return msgs
