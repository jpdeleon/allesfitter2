#!/usr/bin/env python
"""
Usage
$python prepare_allesfit.py -toi 1097 -s all -e 120
$python prepare_allesfit.py -tic 273586149 -s -1 -p qlp
$python prepare_allesfit.py -name 'HIP 67522' -o -i --debug

Uses parameter from TOI/CTIO/NExSci databse and
creates a directory with the files needed to run allesfitter:
1. params.csv
2. settings.csv
3. run.py
4. params_star.csv
5. mission.csv e.g. tess.csv, k2.csv, kepler.csv
======
* for precise transit transit timing, some parameters can be fixed
* limb darkening can be fixed to theoretical values derived using ~ldtk~ limbdark;
  assumes feh=(0,0.1) dex if feh is not available
* uses tess-point to check if target was observed by TESS
(useful to know even if `lightkurve.search_lightcurve` returned None)
* uses aliases (K2 name --> EPIC)
https://exoplanetarchive.ipac.caltech.edu/docs/sysaliases.html
======
"""

import math
import os
import sys
from argparse import ArgumentParser
from math import ceil
from pathlib import Path
from typing import Tuple

import astropy.units as u
import lightkurve as lk
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from loguru import logger
from scipy.stats import anderson

# from ldtk import LDPSetCreator, BoxcarFilter
from tess_stars2px import tess_stars2px_function_entry

from allesfitter import allesclass  # , config, nested_sampling_output, general_output
from allesfitter.exoworlds_rdx.lightcurves.index_transits import (
    get_tmid_observed_transits,
)
from allesfitter.orbits import (
    circular_geometry_from_transit_duration,
    circular_transit_duration,
)
from allesfitter.tars import patch_lightkurve
from allesfitter.utils.scripting import (
    a_from_rhoprs,
    as_from_rhop,
    catalog_info_TIC,
    get_ctois,
    get_name_aliases,
    get_nexsci,
    get_tfop_info,
    get_tois,
    rho_from_mr,
)

#::: lightkurve cannot read TARS products on its own, so `-p tars` would die
#::: inside download_all(); this teaches it the format before any search runs
patch_lightkurve()

logger.remove()
log_format = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{message}</level>"
logger.add(sys.stderr, format=log_format)
try:
    import limbdark as ld
except Exception as e:
    logger.error(f"Error: {e}")
    command = "pip install git+https://github.com/john-livingston/limbdark.git#egg=limbdark"
    raise ModuleNotFoundError(command) from e

assert lk.__version__[0] == "2"

filter_widths = {
    "gp": (400, 550),
    "V": (480, 600),
    "rp": (560, 700),
    "ip": (700, 820),
    "zs": (825, 920),
    "I+z": (720, 1030),
    "tess": (585, 1050),
}

# home = Path.home()
# sys.path.insert(0, f"{home}/github/research/project/young_ttvs/code")

cols = ["time", "flux", "flux_err"]


def _parse_segment_label(mission_str):
    """Last whitespace token of a lightkurve `mission` string, kept as str.

    TESS sectors and Kepler quarters are integers (e.g. 'TESS Sector 82'
    → '82'); K2 campaigns may carry an alpha suffix when split, e.g.
    'K2 Campaign 11a' → '11a' / '11b'. Returning the raw token preserves
    that distinction; downstream comparisons must use string equality
    (the helper below normalizes the other side).
    """
    return str(mission_str).split()[-1]


def _segments_match(a, b):
    """Tolerant segment-id equality (handles int vs str-with-suffix).

    Purely-numeric ids compare by value so zero-padding is irrelevant
    (``1`` == ``'01'``): lightkurve's search table carries zero-padded
    strings ('01') while a downloaded ``LightCurve.sector`` is an int (1).
    Non-numeric labels (K2 '11a'/'11b') fall back to string equality so the
    alpha suffix is still distinguished.
    """
    sa, sb = str(a).strip(), str(b).strip()
    if sa.isdigit() and sb.isdigit():
        return int(sa) == int(sb)
    return sa == sb


def _natural_segment_sort_key(s):
    """Sort key for segment labels: numeric prefix then alpha suffix."""
    s = str(s)
    n = 0
    while n < len(s) and s[n].isdigit():
        n += 1
    head = int(s[:n]) if n else float("inf")
    return (head, s[n:])


def _segment_word(mission, plural=False):
    """User-facing word for a mission's time-window unit.

    TESS uses 'sector', K2 uses 'campaign', Kepler uses 'quarter'. Used in
    log messages and plot titles so output matches the mission the user
    actually queried.
    """
    base = {
        "tess": "sector",
        "k2": "campaign",
        "kepler": "quarter",
    }.get(str(mission).lower(), "sector")
    return base + ("s" if plural else "")


def _transit_requirement_settings(companions):
    """Return explicit per-companion transit-support settings rows."""
    return "".join(f"require_{companion}_transit,True\n" for companion in companions)


def _write_spoc_contamination(lc_or_collection, outpath, mission):
    """Extract CROWDSAP from SPOC headers and convert to allesfitter dilution.

    SPOC light curves carry two crowding keywords in each FITS extension:

    - ``CROWDSAP``: ratio of target flux to total flux in the optimal
      aperture (in [0, 1]). A value of 1 means no contamination; 0.8 means
      80% of measured flux is from the target and 20% is contaminants.
    - ``FLFRCSAP``: fraction of the target's PSF flux captured by the
      aperture (recorded for completeness, not used in the dilution).

    The contamination ratio commonly reported in TESS literature is the
    contaminant-to-target flux ratio:

        contratio = (1 - CROWDSAP) / CROWDSAP

    allesfitter's ``dil_<inst>`` parameter is the fraction of the *measured*
    flux that does NOT come from the target — i.e. the contaminant fraction
    of the total. Converting:

        dilution = contratio / (1 + contratio)
                 = (1 - CROWDSAP) / CROWDSAP / ((1 - CROWDSAP)/CROWDSAP + 1)
                 = 1 - CROWDSAP

    Per-segment values are extracted from the un-stitched LightCurveCollection
    (``.stitch()`` keeps only the first segment's metadata). For a single
    LightCurve the value comes straight from ``.meta``.

    Writes ``outpath`` with a header explaining the math, one row per
    segment, and the median across segments — suitable for pasting into
    ``params.csv`` as the ``dil_<inst>`` value.

    Returns a summary dict ``{'median_dilution', 'std_dilution', 'n_segments',
    'median_crowdsap'}`` when the file was written, or ``False`` when no
    CROWDSAP keyword was found (or lightkurve is unavailable).
    """
    try:
        import lightkurve as _lk
    except Exception:
        return False

    if isinstance(lc_or_collection, _lk.LightCurveCollection):
        items = list(lc_or_collection)
    else:
        items = [lc_or_collection]

    rows = []
    for one in items:
        meta = getattr(one, "meta", None) or {}
        crowdsap = meta.get("CROWDSAP")
        if crowdsap is None:
            continue
        try:
            crowdsap = float(crowdsap)
        except (TypeError, ValueError):
            continue
        flfrcsap_raw = meta.get("FLFRCSAP")
        try:
            flfrcsap = float(flfrcsap_raw) if flfrcsap_raw is not None else None
        except (TypeError, ValueError):
            flfrcsap = None
        seg = (
            getattr(one, "sector", None)
            or getattr(one, "campaign", None)
            or getattr(one, "quarter", None)
            or "?"
        )
        if crowdsap > 0:
            contratio = (1.0 - crowdsap) / crowdsap
        else:
            contratio = float("inf")
        # dilution == 1 - CROWDSAP (algebraically equivalent to contratio/(1+contratio))
        dilution = 1.0 - crowdsap
        rows.append((seg, crowdsap, flfrcsap, contratio, dilution))

    if not rows:
        return False

    seg_word = _segment_word(mission)
    lines = [
        "# SPOC photometric contamination -> allesfitter dilution",
        "#",
        "# Header keywords (per FITS extension):",
        "#   CROWDSAP : target_flux / total_flux in the optimal aperture, in [0, 1]",
        "#   FLFRCSAP : fraction of target PSF flux captured by the aperture",
        "#",
        "# allesfitter's dil_<inst> is the contaminant fraction of the measured flux:",
        "#",
        "#   contratio = (1 - CROWDSAP) / CROWDSAP     # contaminant / target",
        "#   dilution  = contratio / (1 + contratio)   # contaminant / (contaminant + target)",
        "#             = 1 - CROWDSAP                  # algebraically identical",
        "#",
        "# Suggested use: paste the median dilution below into params.csv as",
        "#   dil_<inst>,<median_dilution>,0,uniform 0 1,...",
        "# or use a tighter prior centred on the median if you trust SPOC's crowding model.",
        "#",
        f"# {seg_word}\tCROWDSAP\tFLFRCSAP\tcontratio\tdilution",
    ]
    crowds, dils = [], []
    for seg, crowdsap, flfrcsap, contratio, dilution in rows:
        flf = f"{flfrcsap:.6f}" if flfrcsap is not None else "NA"
        lines.append(f"{seg}\t{crowdsap:.6f}\t{flf}\t{contratio:.6f}\t{dilution:.6f}")
        crowds.append(crowdsap)
        dils.append(dilution)
    lines.append("")
    lines.append(f"# Median across {len(rows)} segment(s):")
    lines.append(f"median_CROWDSAP\t{float(np.median(crowds)):.6f}")
    lines.append(f"median_dilution\t{float(np.median(dils)):.6f}")

    Path(outpath).write_text("\n".join(lines) + "\n")
    return {
        "median_dilution": float(np.median(dils)),
        "std_dilution": float(np.std(dils)),
        "median_crowdsap": float(np.median(crowds)),
        "n_segments": len(rows),
    }


def _safe_stitch(collection, logger=None):
    """Normalise + stitch a LightCurveCollection without astropy's buggy
    ``nanmedian`` path.

    Background
    ----------
    The pinned astropy 5.1 + lightkurve 2.4 stack hits a known issue inside
    ``LightCurveCollection.stitch()``: the default corrector is
    ``lambda x: x.normalize()``, which calls ``np.nanmedian(self.flux)`` on
    a ``MaskedQuantity``. If the segment's flux is all-NaN or fully masked
    (frequent for short 20-s slots wiped by ``quality_bitmask``), the
    median collapses to a scalar ``MaskedQuantity`` and astropy refuses to
    iterate it (``'MaskedQuantity' object with a scalar value is not
    iterable``). Even non-empty segments can hit this when the chosen
    flux column is all-NaN.

    Strategy
    --------
    1. Walk the collection; for each segment compute the median ourselves
       on a plain ``np.ndarray`` (we strip units/mask first).
    2. Drop any segment with fewer than 2 finite samples or a non-finite/
       zero median (these are the ones lightkurve cannot normalise either).
    3. Hand-roll the normalisation (``flux /= median``; same for
       ``flux_err``) so we never invoke lightkurve's normalize and thus
       never hit astropy's nanmedian.
    4. Stitch the pre-normalised segments with an identity corrector.

    Returns the stitched LightCurve, or ``None`` when every segment was
    unusable (caller should treat that as a hard failure).
    """
    import lightkurve as _lk

    if not isinstance(collection, _lk.LightCurveCollection):
        return collection  # already a single LightCurve

    def _to_plain_array(arr):
        # Strip astropy units and any Masked wrapper down to a numpy ndarray.
        if arr is None:
            return None
        try:
            arr = arr.value  # astropy Quantity / MaskedQuantity -> ndarray-like
        except AttributeError:
            pass
        try:
            arr = arr.filled(np.nan)  # numpy MaskedArray -> ndarray with NaN
        except AttributeError:
            pass
        return np.asarray(arr, dtype=float)

    good = []
    for one in collection:
        seg = (
            getattr(one, "sector", None)
            or getattr(one, "campaign", None)
            or getattr(one, "quarter", None)
            or "?"
        )
        try:
            flux_arr = _to_plain_array(one.flux)
        except Exception as exc:
            if logger is not None:
                logger.warning(f"Dropping segment {seg}: cannot read flux ({exc})")
            continue
        if flux_arr is None or flux_arr.size == 0:
            if logger is not None:
                logger.warning(f"Dropping empty segment {seg}")
            continue
        finite = np.isfinite(flux_arr)
        n_finite = int(finite.sum())
        if n_finite < 2:
            if logger is not None:
                logger.warning(f"Dropping segment {seg} (n_finite={n_finite})")
            continue
        med = float(np.median(flux_arr[finite]))
        if not np.isfinite(med) or med == 0.0:
            if logger is not None:
                logger.warning(f"Dropping segment {seg} (median flux = {med})")
            continue
        # Hand-roll normalize: divide flux and flux_err by their (finite)
        # median. Doing it through the LightCurve's __truediv__ preserves
        # units and metadata without ever calling astropy's nanmedian.
        try:
            norm = one.copy()
            norm.flux = one.flux / med
            if getattr(one, "flux_err", None) is not None:
                norm.flux_err = one.flux_err / med
        except Exception as exc:
            if logger is not None:
                logger.warning(f"Dropping segment {seg}: normalize failed ({exc})")
            continue
        good.append(norm)

    if not good:
        if logger is not None:
            logger.error("All segments dropped — nothing to stitch.")
        return None
    # Identity corrector — already normalised above.
    return _lk.LightCurveCollection(good).stitch(corrector_func=lambda x: x)


def _dataset_aware_gp_bounds(time, flux, tdur_days):
    """Derive sensible GP / noise prior bounds from a loaded lightcurve.

    The current hardcoded defaults (``uniform -10 -3`` for ``ln σ_GP`` and
    ``uniform -1 5`` for ``ln ρ_GP``) routinely allow the GP to absorb the
    transit signal or fit the transit shape itself. This helper grounds
    those bounds in the actual data:

    * ``ln_err_flux``  centred near the observed point-to-point RMS,
      bounded so the prior never permits >10% relative-flux jitter.
    * ``ln σ_GP``      lower bound an order of magnitude below the RMS
      (otherwise the GP is useless), upper capped at
      ``min(5% rel flux, 100× RMS)`` to keep it away from transit depth.
    * ``ln ρ_GP``      lower bound at ``max(2 × cadence, 0.5 × tdur)`` so
      the GP cannot fit transit ingress/egress; upper at the actual
      observation baseline (anything longer is degenerate with a slope).

    Returns ``None`` when the lightcurve is too short to estimate noise
    (caller should fall back to a generic default).
    """
    t = np.asarray(time, dtype=float)
    f = np.asarray(flux, dtype=float)
    mask = np.isfinite(t) & np.isfinite(f)
    if int(mask.sum()) < 20:
        return None
    t = t[mask]
    f = f[mask]
    order = np.argsort(t)
    t = t[order]
    f = f[order]

    # Per-cadence RMS via MAD of point-to-point differences.
    diff = np.diff(f)
    if diff.size == 0:
        return None
    mad = float(np.median(np.abs(diff - np.median(diff))))
    rms = 1.4826 * mad / math.sqrt(2.0)
    if not math.isfinite(rms) or rms <= 0:
        rms = float(np.nanstd(f))
    rms = max(rms, 1e-6)

    cadence = float(np.median(np.diff(t)))
    baseline = float(t[-1] - t[0])
    tdur_days = max(float(tdur_days), 1.0 / 24.0)  # floor at 1 h to avoid degenerate priors

    # ln_err_flux: centre at observed RMS, ±2 dex, but always cap upper at 10% flux.
    lnerr_init = math.log(rms)
    lnerr_lo = lnerr_init - 3.0
    lnerr_hi = min(lnerr_init + 2.0, math.log(0.10))
    if lnerr_hi <= lnerr_lo:
        lnerr_hi = lnerr_lo + 1.0

    # ln σ_GP: at least 1/10 RMS (else useless), at most min(5%, 100× RMS).
    sigma_hi_phys = min(0.05, 100.0 * rms)
    sigma_lo_phys = max(rms / 10.0, 1e-7)
    lnsigma_lo = math.log(sigma_lo_phys)
    lnsigma_hi = math.log(sigma_hi_phys)
    if lnsigma_hi <= lnsigma_lo:
        lnsigma_hi = lnsigma_lo + 2.0  # guard against rms tiny edge cases
    lnsigma_init = math.log(max(3.0 * rms, sigma_lo_phys * 2))
    lnsigma_init = min(max(lnsigma_init, lnsigma_lo + 0.1), lnsigma_hi - 0.1)

    # ln ρ_GP: must be > 2 cadences AND > 0.5 × transit duration; < baseline.
    # 2% safety margins keep the bounds strictly inside the physical limits
    # after the .3f rounding round-trip when written to params.csv.
    rho_lo_phys = 1.02 * max(2.0 * cadence, 0.5 * tdur_days)
    rho_hi_phys = 0.98 * baseline
    if rho_hi_phys <= rho_lo_phys:
        rho_hi_phys = rho_lo_phys * 4.0
    lnrho_lo = math.log(rho_lo_phys)
    lnrho_hi = math.log(rho_hi_phys)
    lnrho_init = 0.5 * (lnrho_lo + lnrho_hi)

    return {
        "lnerr_init": lnerr_init,
        "lnerr_lo": lnerr_lo,
        "lnerr_hi": lnerr_hi,
        "lnsigma_init": lnsigma_init,
        "lnsigma_lo": lnsigma_lo,
        "lnsigma_hi": lnsigma_hi,
        "lnrho_init": lnrho_init,
        "lnrho_lo": lnrho_lo,
        "lnrho_hi": lnrho_hi,
        "_rms": rms,
        "_cadence_days": cadence,
        "_baseline_days": baseline,
    }


def _gp_protective_tdur_days(tdur_hours, fallback_days=0.1):
    """Return the longest finite positive companion duration, in days.

    A single instrument GP is shared by all photometric companions. Its
    shortest allowed correlation timescale therefore has to protect the
    longest transit; using the shortest or median duration leaves longer
    transits exposed to the GP.
    """
    durations = np.asarray(list(tdur_hours), dtype=float)
    valid = durations[np.isfinite(durations) & (durations > 0.0)]
    return float(np.max(valid) / 24.0) if valid.size else float(fallback_days)


def _update_params_gp_bounds(params_csv_path, inst, bounds, logger=None):
    """Rewrite ``ln_err_flux_<inst>`` and the GP rows for ``inst`` in
    ``params.csv`` with dataset-aware values from
    :func:`_dataset_aware_gp_bounds`.

    Preserves the existing label/unit/coupled columns when present.
    Returns the number of rows replaced (0 if nothing matched).
    """
    if not bounds:
        return 0
    path = Path(params_csv_path)
    if not path.exists():
        return 0
    text = path.read_text()

    def _rebuild(name, value, fit, bounds_str, tail):
        return ",".join([name, value, str(fit), bounds_str] + tail)

    def _split_keep_tail(line):
        # name,value,fit,bounds,label,unit,coupled — keep fields 4+ verbatim.
        parts = line.split(",")
        if len(parts) < 4:
            return None
        return parts[0], parts[1], parts[2], parts[3], parts[4:]

    prefixes = {
        f"ln_err_flux_{inst}": (
            "{:.3f}".format(bounds["lnerr_init"]),
            "uniform {:.3f} {:.3f}".format(bounds["lnerr_lo"], bounds["lnerr_hi"]),
        ),
        f"baseline_gp_matern32_lnsigma_flux_{inst}": (
            "{:.3f}".format(bounds["lnsigma_init"]),
            "uniform {:.3f} {:.3f}".format(bounds["lnsigma_lo"], bounds["lnsigma_hi"]),
        ),
        f"baseline_gp_matern32_lnrho_flux_{inst}": (
            "{:.3f}".format(bounds["lnrho_init"]),
            "uniform {:.3f} {:.3f}".format(bounds["lnrho_lo"], bounds["lnrho_hi"]),
        ),
    }

    new_lines = []
    replaced = 0
    for line in text.splitlines():
        if line.startswith("#") or not line.strip():
            new_lines.append(line)
            continue
        split = _split_keep_tail(line)
        if split is None:
            new_lines.append(line)
            continue
        name, _value, fit, _bounds, tail = split
        if name in prefixes and "uniform" in _bounds:
            init_str, bounds_str = prefixes[name]
            new_lines.append(_rebuild(name, init_str, fit, bounds_str, tail))
            replaced += 1
        else:
            new_lines.append(line)
    if replaced == 0:
        return 0
    suffix = "\n" if text.endswith("\n") else ""
    path.write_text("\n".join(new_lines) + suffix)
    if logger is not None:
        logger.info(
            f"params.csv: refined GP/noise priors for {inst} from data "
            f"(RMS={bounds['_rms']:.2e}, cadence={bounds['_cadence_days'] * 86400:.0f}s, "
            f"baseline={bounds['_baseline_days']:.2f}d) — {replaced} row(s) updated"
        )
    return replaced


def _inject_dilution_normal_prior(params_csv_path, inst, median, std, sigma_floor=0.01):
    """Inject a commented normal-prior dilution row above the uniform one.

    Looks for the existing ``dil_<inst>,...,uniform ...`` line in
    ``params_csv_path`` and prepends a commented-out twin using a Gaussian
    prior centred on the SPOC-derived ``median`` dilution with width
    ``max(std, sigma_floor)``. CROWDSAP carries no formal uncertainty, so a
    1% absolute floor keeps the prior physically reasonable even when only
    one segment is available.

    The row is commented out so the original uniform row remains the active
    default — users who trust the SPOC crowding model uncomment the new
    row (and comment the uniform one) to fit dilution under the tight
    Gaussian prior.

    Returns True when a row was inserted, False otherwise.
    """
    path = Path(params_csv_path)
    if not path.exists():
        return False
    text = path.read_text()
    needle = f"dil_{inst},"
    sigma = max(float(std), float(sigma_floor))
    new_row = (
        f"#dil_{inst},{median:.6f},1,normal {median:.6f} {sigma:.6f},"
        f"$D_\\mathrm{{0; {inst}}}$ (SPOC),,"
    )
    new_lines = []
    inserted = False
    for line in text.splitlines():
        if (not inserted) and line.startswith(needle) and "uniform" in line:
            new_lines.append(new_row)
            inserted = True
        new_lines.append(line)
    if not inserted:
        return False
    suffix = "\n" if text.endswith("\n") else ""
    path.write_text("\n".join(new_lines) + suffix)
    return True


Nsamples = 10_000
planets = "b c d e f g h i j k".split()
quartiles_1sig = [16.0, 50.0, 84.0]  # 1-sigma
quartiles_2sig = [2.275, 50.0, 97.725]  # 2-sigma
quartiles_3sig = [0.135, 50.0, 99.865]  # 3-sigma


def _percentile_3sig_safe(samples, fallback, label, planet, logger):
    """3-sigma percentiles of ``samples``, resilient to an empty/all-NaN input.

    Some targets yield no usable ``(R*+Rp)/a`` samples — e.g. when the stellar
    mass/radius (hence rho_star) or the period is NaN, ``as_from_rhop`` returns
    NaNs and the ``0 < rsuma < 1`` validity mask removes every sample. Rather
    than crashing inside ``np.percentile`` ("cannot do a non-empty take from an
    empty axes"), warn and fall back to ``fallback`` so prepare_allesfit can
    still emit a (clearly-flagged, reviewable) prior.

    Parameters
    ----------
    samples : array-like
        Already validity-masked samples (may be empty / all non-finite).
    fallback : tuple(float, float, float)
        ``(lo, mid, hi)`` returned when no finite samples remain.
    label, planet : str
        Used only in the warning message.
    """
    s = np.asarray(samples, dtype=float)
    s = s[np.isfinite(s)]
    if s.size == 0:
        logger.warning(
            f"Planet {planet}: could not derive {label} (no finite samples; check stellar "
            "mass/radius/period). Falling back to a wide default prior "
            f"({fallback[0]:g}, {fallback[1]:g}, {fallback[2]:g}) — review {planet}_rsuma in params.csv before fitting."
        )
        return fallback
    return tuple(np.percentile(s, q=quartiles_3sig))


def _duration_informed_geometry(
    *,
    period_days,
    period_err_days,
    tdur_hours,
    tdur_err_hours,
    radius_ratio,
    radius_ratio_err,
    nsamples=Nsamples,
    rng=None,
):
    """Return circular orbital geometry initialized by catalog T14.

    Transit duration does not uniquely determine both scaled separation and
    inclination.  We use the transit-conditioned ``b ~ Uniform(0, 1)`` prior
    for a non-grazing planet, and initialize at its median ``b=0.5``.  Each
    Monte Carlo sample derives *both* rsuma and cos(i) from the same impact
    parameter, so every retained tuple is physically coupled and reproduces
    its sampled duration exactly.
    """
    rng = np.random if rng is None else rng

    def _finite_error(value, fallback=0.0):
        try:
            value = float(value)
        except (TypeError, ValueError):
            return fallback
        return value if math.isfinite(value) and value >= 0.0 else fallback

    period_days = float(period_days)
    tdur_hours = float(tdur_hours)
    radius_ratio = float(radius_ratio)
    period_err_days = _finite_error(period_err_days)
    tdur_err_hours = _finite_error(tdur_err_hours, fallback=1.0)
    radius_ratio_err = _finite_error(radius_ratio_err)

    period_samples = rng.normal(period_days, period_err_days, size=nsamples)
    duration_samples = rng.normal(tdur_hours, tdur_err_hours, size=nsamples) / 24.0
    radius_ratio_samples = rng.normal(radius_ratio, radius_ratio_err, size=nsamples)
    impact_parameter_samples = rng.uniform(0.0, 1.0, size=nsamples)
    rsuma_samples, cosi_samples = circular_geometry_from_transit_duration(
        duration_samples,
        period_samples,
        radius_ratio_samples,
        impact_parameter_samples,
    )
    valid = (
        np.isfinite(rsuma_samples)
        & np.isfinite(cosi_samples)
        & (rsuma_samples > 0.0)
        & (rsuma_samples < 1.0)
        & (cosi_samples >= 0.0)
        & (cosi_samples < 1.0)
    )
    if not np.any(valid):
        raise ValueError("catalog duration produced no physical circular-orbit samples")

    impact_parameter = 0.5
    rsuma, cosi = circular_geometry_from_transit_duration(
        tdur_hours / 24.0,
        period_days,
        radius_ratio,
        impact_parameter,
    )
    if not (math.isfinite(rsuma) and math.isfinite(cosi)):
        raise ValueError("catalog duration cannot initialize a physical circular orbit")

    rsuma_min, _, rsuma_max = np.percentile(rsuma_samples[valid], q=quartiles_3sig)
    _, _, cosi_max = np.percentile(cosi_samples[valid], q=quartiles_3sig)
    return {
        "rsuma": float(rsuma),
        "cosi": float(cosi),
        "impact_parameter": impact_parameter,
        "rsuma_min": float(rsuma_min),
        "rsuma_max": float(rsuma_max),
        "cosi_max": float(cosi_max),
    }


def _default_prior_bounds(rprs_max, rsuma_min, rsuma_max):
    """Compute the physics-informed default prior upper/lower bounds.

    The wide unbounded defaults of older versions wasted sampler time in
    unphysical regions (non-transiting geometries, brown-dwarf depths,
    GP correlation lengths >>1 yr, etc.). These tighter bounds reject
    those regions without biasing physical solutions:

    * ``_rr_upper``  — radius-ratio upper limit. Round literature
      ``rprs_max`` up to the nearest 0.1, add 5% safety headroom, cap at
      0.5 (brown-dwarf regime).
    * ``_rsuma_lo``  — lower bound on ``(R★+Rp)/a``. Prevents ``a→∞`` at
      the prior boundary (degenerate with f_c/f_s).
    * ``_rsuma_hi``  — upper bound. Covers ultra-short period planets at
      the wide end (``a ≈ 2 R★``) without over-extending.
    * ``_cosi_max``  — transit geometry requires ``cos i < (R★+Rp)/a =
      rsuma``. Outside that the likelihood is zero. Cap with a
      1.2× safety factor, hard-bounded at 1.

    Parameters
    ----------
    rprs_max, rsuma_min, rsuma_max : float
        3-sigma upper/lower estimates of Rp/R★ and rsuma from the
        catalog-resolved literature values (computed by the caller via
        sampling).

    Returns
    -------
    dict
        Keys: ``rr_upper``, ``rsuma_lo``, ``rsuma_hi``, ``cosi_max``.
        All floats, suitable for direct substitution into params.csv.
    """
    rr_upper = min(0.5, ceil(rprs_max * 10) / 10 + 0.05)
    rsuma_lo = max(1e-3, rsuma_min)
    rsuma_hi = min(0.5, 2.0 * rsuma_max)
    cosi_max = min(1.0, 1.2 * rsuma_max)
    return {
        "rr_upper": rr_upper,
        "rsuma_lo": rsuma_lo,
        "rsuma_hi": rsuma_hi,
        "cosi_max": cosi_max,
    }


def catalog_info_name(df) -> Tuple:
    Teff, Teff_err = (
        df["st_teff"].astype(float),
        np.sqrt(df["st_tefferr1"] ** 2 + df["st_tefferr2"] ** 2),
    )
    logg, logg_err = (
        df["st_logg"].astype(float),
        np.sqrt(df["st_loggerr1"] ** 2 + df["st_loggerr2"] ** 2),
    )
    feh, feh_err = 0, 0.1
    radius, radius_err = (
        df["st_rad"].astype(float),
        np.sqrt(df["st_raderr1"] ** 2 + df["st_raderr2"] ** 2),
    )
    mass, mass_err = (
        df["st_mass"].astype(float),
        np.sqrt(df["st_masserr1"] ** 2 + df["st_masserr2"] ** 2),
    )
    return (
        Teff,
        Teff_err,
        logg,
        logg_err,
        feh,
        feh_err,
        radius,
        radius_err,
        mass,
        mass_err,
    )


_TIC_TRANSIT_FIELDS = (
    ("period", "Period (days)", "Porb (d): ", True),
    ("period_err", "Period (days) err", "Porb err (d): ", False),
    ("epoch", "Epoch (BJD)", "Epoch (BJD): ", True),
    ("epoch_err", "Epoch (BJD) err", "Epoch err (BJD): ", False),
    ("duration", "Duration (hours)", "Tdur (h): ", True),
    ("duration_err", "Duration (hours) err", "Tdur err (h): ", False),
    ("depth", "Depth (ppm)", "Depth (ppm): ", True),
    ("depth_err", "Depth (ppm) err", "Depth err (ppm): ", False),
)


def _resolve_tic_transit_parameters(values=None, prompt=None):
    """Use supplied TIC transit parameters and prompt only for missing fields.

    Every supplied argument is validated before the first prompt, so a bad
    command line fails immediately instead of asking for unrelated ephemeris
    values first.
    """
    values = {} if values is None else dict(values)
    prompt = input if prompt is None else prompt

    for key, _column, _label, must_be_positive in _TIC_TRANSIT_FIELDS:
        value = values.get(key)
        if value is None:
            continue
        value = float(value)
        if not math.isfinite(value) or value < 0 or (must_be_positive and value == 0):
            relation = "positive" if must_be_positive else "non-negative"
            raise ValueError(f"--{key.replace('_', '-')} must be a finite {relation} number")
        values[key] = value

    resolved = []
    columns = []
    for key, column, label, _must_be_positive in _TIC_TRANSIT_FIELDS:
        value = values.get(key)
        resolved.append(float(prompt(label)) if value is None else value)
        columns.append(column)
    return resolved, columns


def parse_target_name(
    toiid=None,
    ticid=None,
    ctoiid=None,
    name=None,
    update_db=False,
    tic_parameters=None,
) -> Tuple:
    if toiid:
        df = get_tois(clobber=update_db)
        logger.info("Using parameters from TOI database (use --update_db to update).")
        logger.info(f"To use published parameters in NExSci, use -name=TOI-{toiid}")
        key = "TOI"
        id = str(toiid)
        idx = df[key].apply(lambda x: str(x).split(".")[0] == id)
        source = "tfop"
        target_name = f"TOI-{id.zfill(4)}"
    if ticid:
        logger.info("Using parameters from TIC catalog (use --update_db to update).")
        key = "TIC"
        id = str(ticid)
        target_name = f"TIC-{ticid}"
        source = "custom"
        transit_values, transit_columns = _resolve_tic_transit_parameters(tic_parameters)
        df = pd.DataFrame([transit_values], columns=transit_columns)
        idx = [True]
    if ctoiid:
        logger.info("Using parameters from CTOI database (use --update_db to update).")
        df = get_ctois(clobber=update_db)
        key = "CTOI"
        id = ctoiid
        idx = df["TIC ID"] == int(ctoiid)
        target_name = f"CTOI-{ctoiid}"
        source = "ctoi"
    if name:
        logger.info("Using parameters from NExSci database (use --update_db to update).")
        df = get_nexsci("pscomppars", clobber=update_db)
        # df = df[df['default_flag']==1]
        df["Period (days)"] = df["pl_orbper"].astype(float)
        df["Period (days) err"] = np.sqrt(df["pl_orbpererr1"] ** 2 + df["pl_orbpererr2"] ** 2)
        df["Epoch (BJD)"] = df["pl_tranmid"].astype(float)
        df["Epoch (BJD) err"] = 0.1
        df["Depth (ppm)"] = df["pl_trandep"].astype(float) / 100 * 1e6  # % to ppm
        df["Depth (ppm) err"] = 1_000
        df["Duration (hours)"] = df["pl_trandur"].astype(float)
        df["Duration (hours) err"] = np.sqrt(
            df["pl_trandurerr1"].astype(float) ** 2 + df["pl_trandurerr2"].astype(float) ** 2
        )
        key = "hostname"
        id = name.lower()
        target_name = name.strip().replace(" ", "")

        def _norm(s):
            return str(s).lower().replace(" ", "").replace("-", "").replace("_", "")

        hostnames_norm = df[key].astype(str).map(_norm)
        idx = hostnames_norm == _norm(name)
        if sum(idx) == 0:
            # Fall back to NExSci alias lookup: the hostname in pscomppars is
            # often a TIC/2MASS/HD identifier rather than the TOI/K2 alias the
            # user supplied. Resolve all known aliases and try each one.
            try:
                logger.info(f"Hostname '{name}' not found directly; resolving NExSci aliases...")
                aliases = get_name_aliases(name)
                for alias in aliases:
                    match = hostnames_norm == _norm(alias)
                    if match.any():
                        logger.info(f"Matched NExSci hostname via alias: {alias}")
                        idx = match
                        break
            except Exception as e:
                logger.error(f"Alias lookup failed: {e}")
        source = "nexsci"
    msg = f"Coulnd't find {key} {id} in {key} database."
    assert sum(idx) > 0, logger.error(msg)
    return target_name, df[idx].reset_index(drop=True), source


def get_tess_sectors(
    target_name: str, df: pd.DataFrame, toiid=None, ctoiid=None, name=None
) -> Tuple:
    if toiid or ctoiid:
        # if target is available in TOI,CTOI,or NexSci
        coord = SkyCoord(*df[["RA", "Dec"]].values[0], unit=("hourangle", "deg"))
        ra, dec = coord.ra.deg, coord.dec.deg
        ticid = df["TIC ID"].unique()[0]
    elif name:
        # for other targets
        data_json = get_tfop_info(target_name)
        ra = float(data_json["coordinates"]["ra"])
        dec = float(data_json["coordinates"]["dec"])
        ticid = int(data_json["basic_info"]["tic_id"])
    else:
        msg = "Set toiid, ctoiid, or name."
        logger.error(msg)
        sys.exit()
    try:
        (
            outID,
            outEclipLong,
            outEclipLat,
            outSec,
            outCam,
            outCcd,
            outColPix,
            outRowPix,
            scinfo,
        ) = tess_stars2px_function_entry(ticid, ra, dec)
    except Exception as e:
        logger.error(f"Error: {e}")
    return ticid, outSec


def check_if_sector_is_available(target_name: str, given_sector, all_sectors: list) -> str:
    """
    All cases for given_sector=(None, 0, 1, 'all', [1,2], -1)
    Check only if given_sector is non-negative int or list
    """
    if given_sector is None:
        return "default"
    else:
        assert isinstance(given_sector, list)
        assert isinstance(all_sectors, np.ndarray)
        if len(given_sector) == 1:
            if given_sector == ["all"]:
                return "all_sector"
            elif given_sector == ["-1"]:
                return "last"
            elif given_sector == ["0"]:
                return "first"
        # check if given_sector exists if not 'all','0',or '-1'
        idx = np.array([True if int(s) in all_sectors else False for s in given_sector])
        msg = f"{target_name} was not observed in sector={np.array(given_sector)[~idx]}\n"
        msg += f"Try sector={all_sectors}."
        assert np.all(idx), logger.error(msg)
        return "multi_sector"


#::: environment override: auto-resolve interactive/exit points in headless runs
NONINTERACTIVE_ENV_VAR = "ALLESFITTER_NONINTERACTIVE"


def _handle_multiple_exptimes(search_result, unique_exptimes, logger, context="", expected_n=None):
    """Resolve the "multiple exposure times available" situation.

    Several download paths bail out when a search returns more than one
    exposure time (e.g. TESS Sector 64 offers both 20 s and 120 s SPOC
    products) because stitching mixed cadences is not meaningful. Historically
    this exited with a hint that named a non-existent ``-exp`` flag and printed
    the whole array of choices, and exited with status 0 (looking like success
    to a scheduler).

    Default behaviour now: log an actionable message naming the real
    ``-e/--exptime`` flag and a single valid value, then ``sys.exit(1)`` so the
    user (or an orchestrator) re-runs with an explicit cadence.

    Opt-in behaviour (``ALLESFITTER_NONINTERACTIVE`` set): auto-select the
    shortest cadence, narrow ``search_result`` to it, and return
    ``(narrowed_result, chosen_exptime)`` so the run completes unattended.

    Parameters
    ----------
    search_result : lightkurve.SearchResult
        The result to narrow when auto-selecting.
    unique_exptimes : iterable of float
        The distinct exposure times present (seconds).
    logger : loguru.Logger
        Logger for the hint / warning.
    context : str
        Short human-readable scope for the message (e.g. ``"the given sector"``).
    expected_n : int or None
        If given, prefer the shortest cadence that yields exactly this many
        products (so a stitched multi-sector download stays one-per-segment);
        falls back to the shortest cadence when none matches.

    Returns
    -------
    (lightkurve.SearchResult, float)
        Narrowed result and chosen exposure time. Never returns in the default
        (exit) path.
    """
    choices = sorted(float(e) for e in unique_exptimes)
    choices_str = ", ".join(f"{c:g}" for c in choices)
    ctx = f" for {context}" if context else ""

    if not os.environ.get(NONINTERACTIVE_ENV_VAR):
        logger.error(
            f"Multiple exposure times are available{ctx} ({choices_str} sec).\n"
            f"Re-run with a single exposure time, e.g. -e {choices[0]:g} "
            "(or --exptime <sec>).\n"
            f"Set {NONINTERACTIVE_ENV_VAR}=1 to auto-select the shortest cadence."
        )
        sys.exit(1)

    exptimes = search_result.table.to_pandas().exptime.to_numpy(dtype=float)
    chosen = choices[0]
    if expected_n is not None:
        for c in choices:
            if int(np.isclose(exptimes, c, atol=0.5).sum()) == expected_n:
                chosen = c
                break
    narrowed = search_result[list(np.isclose(exptimes, chosen, atol=0.5))]
    logger.warning(
        f"Multiple exposure times available{ctx} ({choices_str} sec); "
        f"auto-selected the shortest usable cadence -e {chosen:g} "
        f"({NONINTERACTIVE_ENV_VAR} set). Pass -e/--exptime to override."
    )
    return narrowed, chosen


def main():
    ap = ArgumentParser()
    group1 = ap.add_mutually_exclusive_group(required=True)
    group1.add_argument("-toi", help="TOI ID", type=int)
    group1.add_argument("-ctoi", help="CTOI ID", type=int)
    group1.add_argument("-tic", help="TIC ID", type=int)
    group1.add_argument("-name", help="Name", type=str)
    group2 = ap.add_mutually_exclusive_group(required=True)
    group2.add_argument(
        "-s",
        "--sector",
        nargs="+",
        help="sector=-1 uses most recent TESS sector (default); try -sector=all to use all",
        default=None,
    )
    group2.add_argument(
        "-c",
        "--campaign",
        help="campaign=-1 uses most recent K2 campaign (default); try -campaign=all to use all",
        default=None,
    )
    group2.add_argument(
        "-q",
        "--quarter",
        help="quarter=-1 uses most recent Kepler quarter (default); try -quarter=all to use all",
        default=None,
    )
    # ap.add_argument("-s", "--sector", "--sector", nargs='+',
    #                 help="-sector=-1 uses most recent TESS sector (default); try -sector=all to use all",
    #                 default=None)
    ap.add_argument(
        "-e",
        "--exptime",
        help="exposure time(s) in seconds (default=None, auto-detected). Accepts "
        "multiple, one per --pipeline (or a single value shared by all pipelines), "
        "e.g. -p spoc qlp -e 120 600",
        type=float,
        nargs="+",
        default=None,
    )
    ap.add_argument(
        "-p",
        "--pipeline",
        help="TESS/Kepler data pipeline(s) (default='spoc'). Accepts multiple, one "
        "per --filename, to download each instrument from its own pipeline in a "
        "single run, e.g. -f spoc120 qlp600 -p spoc qlp -e 120 600. A single "
        "--pipeline value still only downloads the first --filename, as before.",
        type=str,
        nargs="+",
        default=["spoc"],
    )
    ap.add_argument(
        "-f",
        "--filename",
        help="filename(s) of lightcurve used as inst_phot (default='tess'). "
        "Accepts multiple, e.g. -f kepler tess",
        type=str,
        nargs="+",
        default=["tess"],
    )
    ap.add_argument(
        "-m",
        "--mission",
        help="satellite mission (default='tess')",
        type=str,
        default="tess",
        choices=["tess", "k2", "kepler"],
    )
    ap.add_argument(
        "-lc",
        "--lc_type",
        help="type of light curve (default=pdcsap)",
        choices=["pdcsap", "sap"],
        type=str,
        default="pdcsap",
    )
    ap.add_argument(
        "-sig",
        "--sigma",
        help="sigma for removing outliers in (combined) TESS lc",
        type=float,
        default=None,
    )
    ap.add_argument(
        "-qb",
        "--quality",
        choices=["none", "default", "hard", "hardest"],
        type=str,
        default="default",
    )
    ap.add_argument("-dir", help="base directory", type=str, default=".")
    for option, help_text in (
        ("--period", "orbital period for a raw TIC target (days)"),
        ("--period-err", "orbital-period uncertainty (days)"),
        ("--epoch", "transit epoch for a raw TIC target (BJD)"),
        ("--epoch-err", "transit-epoch uncertainty (days)"),
        ("--duration", "transit duration for a raw TIC target (hours)"),
        ("--duration-err", "transit-duration uncertainty (hours)"),
        ("--depth", "transit depth for a raw TIC target (ppm)"),
        ("--depth-err", "transit-depth uncertainty (ppm)"),
    ):
        ap.add_argument(option, help=help_text, type=float, default=None)
    ap.add_argument(
        "-i",
        "--interactive",
        help="manually input missing values (default=False)",
        action="store_true",
        default=False,
    )
    ap.add_argument(
        "-u",
        "--update_db",
        help="update TOI or NExSci database (default=False)",
        action="store_true",
        default=False,
    )
    ap.add_argument(
        "-r",
        "--results_dir",
        help="path to the results dir of a previous run to be used in params.csv",
        default=None,
    )
    ap.add_argument(
        "-o",
        "--overwrite",
        help="overwrite files (default=False)",
        action="store_true",
        default=False,
    )
    ap.add_argument("--debug", action="store_true", default=False)
    ap.add_argument(
        "--lc-only",
        help="only download and save lightcurve, skip generating config files",
        action="store_true",
        default=False,
    )
    ap.add_argument(
        "--ttv",
        help=(
            "emit per-transit TTV parameters in params.csv (or params2.csv "
            "when --results_dir is given). The number of rows per planet "
            "equals the count of transits with actual data coverage, "
            "computed via get_tmid_observed_transits. The index N in "
            "{pl}_ttv_transit_N is the N-th chronologically observed transit."
        ),
        action="store_true",
        default=False,
    )
    ap.add_argument(
        "-bp",
        "--bandpass",
        help=(
            "Bandpass label(s) for chromatic transit modeling, one per "
            "instrument given by --filename (space-separated). The count "
            "must equal len(--filename); repeat a label to share a band "
            "across instruments. Examples: "
            "-f tess kepler -bp tess kepler (chromatic, separate Rp/Rs "
            "per band); -f tess_pdcsap tess_qlp -bp tess tess (single-band, "
            "two pipelines sharing one rr/LDC). "
            "If omitted, params.csv is generated achromatically (a single "
            "{pl}_rr row); a warning is logged when --filename has multiple "
            "distinct instruments to make sure that's intentional."
        ),
        type=str,
        nargs="+",
        default=None,
    )

    args = ap.parse_args(None if sys.argv[1:] else ["-h"])

    basedir = args.dir
    outdir = Path(basedir)
    results_dir = args.results_dir
    debug = args.debug

    if args.lc_only:
        # Minimal path: just download lightcurve
        assert args.sector is not None, "Sector required for --lc-only mode"
        assert any([args.tic, args.toi, args.ctoi, args.name]), (
            "One of -tic/-toi/-ctoi/-name required for --lc-only mode"
        )

        mission = args.mission.lower()
        ticid = args.tic
        sector = args.sector
        quality_bitmask = args.quality

        # -p/-e each accept multiple values now. Pair them positionally so
        # `-p spoc qlp -e 120 600` downloads one file per (pipeline, exptime);
        # a single -e is broadcast across every -p, and omitting -e leaves
        # every pipeline to auto-detect its own exptime (unchanged default).
        pipelines = args.pipeline
        exptimes = args.exptime
        if exptimes is None:
            exptimes = [None] * len(pipelines)
        elif len(exptimes) == 1:
            exptimes = exptimes * len(pipelines)
        elif len(exptimes) != len(pipelines):
            logger.error(
                f"--exptime has {len(exptimes)} entries but --pipeline has "
                f"{len(pipelines)} entries; give one --exptime per --pipeline, "
                "or a single value shared by all."
            )
            sys.exit(1)

        if args.tic:
            query_name = f"TIC {ticid}"
            label = f"TIC-{ticid}"
        elif args.toi:
            query_name = f"TOI {args.toi}"
            label = f"TOI-{str(args.toi).zfill(4)}"
        elif args.ctoi:
            query_name = f"CTOI {args.ctoi}"
            label = f"CTOI-{args.ctoi}"
        else:
            query_name = args.name
            label = args.name.strip().replace(" ", "")

        for pipeline, exptime in zip(pipelines, exptimes):
            lc_type = "sap_flux" if pipeline == "qlp" else args.lc_type + "_flux"

            result = lk.search_lightcurve(
                query_name, author=pipeline, exptime=exptime, mission=mission
            )

            if not result:
                logger.error(f"No lightcurve found for pipeline={pipeline}. Check inputs.")
                sys.exit()

            # Keep segment labels as strings so K2 campaigns like '11a'/'11b'
            # (which can't cast to int) are preserved alongside TESS sectors.
            sectors = [_parse_segment_label(s) for s in result.mission]
            unique_sectors = sorted(set(sectors), key=_natural_segment_sort_key)

            # Handle sector flag (args.sector is a list like ["all"] or [10])
            if sector == ["all"]:
                sector_to_use = unique_sectors
            else:
                sector_to_use = sector if isinstance(sector, list) else [sector]

            idx = [str(s) in [str(x) for x in sector_to_use] for s in sectors]

            if sum(idx) == 0:
                _w = _segment_word(mission).capitalize()
                msg = f"{_w} {sector} not available. Available: {unique_sectors}"
                logger.error(msg)
                sys.exit()

            filtered_result = result[idx]

            unique_exptimes = filtered_result.table.to_pandas().exptime.unique()
            exptime = unique_exptimes[0] if exptime is None else exptime

            if len(sector_to_use) != len(filtered_result):
                filtered_result, exptime = _handle_multiple_exptimes(
                    filtered_result,
                    unique_exptimes,
                    logger,
                    context=f"{_segment_word(mission)}={sector_to_use}",
                    expected_n=len(sector_to_use),
                )

            lc = _safe_stitch(
                filtered_result.download_all(flux_column=lc_type, quality_bitmask=quality_bitmask),
                logger=logger,
            )
            if lc is None:
                logger.error("No usable segments downloaded.")
                sys.exit()

            df = lc.to_pandas()
            if len(df) == 0:
                logger.error("Lightcurve data is empty.")
                sys.exit()

            # Handle time (lightkurve uses time as index; add mission BJD offset)
            _bjd_offset = {"tess": 2457000, "k2": 2454833, "kepler": 2454833}.get(mission, 2457000)
            df["time"] = df.index + _bjd_offset
            df = df.reset_index(drop=True).sort_values(by="time")

            cols = ["time", "flux", "flux_err"]
            msg = f"Somehow, `flux_err` is all NaN.\n{df[cols]}\n"
            if len(df["flux_err"].dropna(axis="index")) == 0:
                df["flux_err"] = 1
                msg += "Setting flux error column = 1 (See allesfitter documentation)."
                logger.error(msg)
            df2 = df[cols].dropna(axis="index")

            # Build output filename
            sector_str = (
                "_".join(map(str, sector_to_use))
                if isinstance(sector_to_use, list)
                else str(sector_to_use)
            )
            fn = f"{label}_{pipeline}_s{sector_str}_exp{int(exptime)}s.csv"
            fp = Path(basedir, fn)
            fp.parent.mkdir(parents=True, exist_ok=True)

            df2[cols].to_csv(fp, sep=",", header=False, index=False)
            logger.info(f"Saved: {fp}")
            logger.info(f"Ndata: {len(df2):,}")
        logger.info("Lightcurve(s) saved. Exiting (--lc-only mode).")
        return

    if results_dir:
        alles = allesclass(outdir.joinpath(results_dir))
        logger.info("Updating params.csv")

        # =====Update params.csv=====
        # Iterate BASEMENT.allkeys (every row in the prior params.csv), not
        # just BASEMENT.fitkeys. Previously, fixed parameters (e.g. an LDC
        # pair pinned at a fitted value, dilution, fixed eccentricity terms)
        # were silently dropped from params2.csv — leading to None values at
        # ns_fit time and a cryptic numpy ufunc TypeError when q_to_u tried
        # to sqrt(None). Now fitted keys get the posterior-derived prior row
        # and fixed keys get a faithful `name,value,0,,label,unit,` copy.
        text = """#name,value,fit,bounds,label,unit,coupled_with\n"""
        try:
            ll = alles.posterior_params_ll.copy()  # 16th percentile
            median = alles.posterior_params_median.copy()  # 50th percentile
            ul = alles.posterior_params_ul.copy()  # 84th percentile
            _fitkeys_set = set(alles.BASEMENT.fitkeys)
            for i, name in enumerate(alles.BASEMENT.allkeys):
                label = alles.BASEMENT.labels[i]
                unit = alles.BASEMENT.units[i]
                if name in _fitkeys_set:
                    # Fitted parameter: derive prior from posterior shape.
                    norm_test = anderson(alles.posterior_params[name], dist="norm")
                    # crit vals at 15/10/5/2.5/1%; statistic < crit[0] (15%)
                    # → fail to reject normality.
                    dist = (
                        "normal"
                        if norm_test.statistic < norm_test.critical_values[0]
                        else "uniform"
                    )
                    if dist == "normal":
                        l_err, mid, u_err = ll[name], median[name], ul[name]
                        sig = np.sqrt(l_err**2 + u_err**2)
                        text += f"{name},{mid:6f},1,normal {mid:6f} {sig:6f},{label},{unit},\n"
                        if debug:
                            logger.info(f"{name}: {mid:.6f} +{u_err:.6f} -{l_err:.6f} (normal)")
                    elif dist == "uniform":
                        l_limit, mid, u_limit = np.nanpercentile(
                            alles.posterior_params[name], q=[1, 50, 99]
                        )
                        text += (
                            f"{name},{mid:6f},1,uniform {l_limit:6f} {u_limit:6f},{label},{unit},\n"
                        )
                        if debug:
                            logger.info(
                                f"{name}: {l_limit:.6f} < {mid:.6f} < {u_limit:.6f} (uniform)"
                            )
                    else:
                        msg = "distribution is not uniform or normal!"
                        logger.error(msg)
                        sys.exit()
                else:
                    # Fixed parameter: preserve from BASEMENT.params so the
                    # generated params2.csv stays a complete drop-in for
                    # params.csv. Skip None values (intentionally absent
                    # entries — e.g. LD law set to None for an instrument).
                    val = alles.BASEMENT.params.get(name)
                    if val is None:
                        if debug:
                            logger.info(f"{name}: skipping (value is None)")
                        continue
                    text += f"{name},{val},0,,{label},{unit},\n"
                    if debug:
                        logger.info(f"{name}: {val} (fixed)")
        except Exception as e:
            logger.error(f"Error: {e}")

        # Append per-transit TTV rows using the observed-transit midpoints.
        # BASEMENT only caches `{c}_tmid_observed_transits` when
        # settings['fit_ttvs']==True (basement.setup_ttv_fit). On a cache
        # miss, recompute via get_tmid_observed_transits using the per-
        # instrument time arrays and posterior-median epoch/period — same
        # approach as afplot_per_transit in general_output.py.
        if args.ttv:
            _settings = alles.BASEMENT.settings
            _data = alles.BASEMENT.data
            _fast_fit_width = float(_settings.get("fast_fit_width", 0.3333333333333333))
            _inst_list = list(_settings.get("inst_phot", []))
            # Per-instrument span / count diagnostic — surfaces cases where
            # one instrument's data is missing, empty, or falls outside the
            # epoch+period grid the cache was built against.
            for _inst in _inst_list:
                _t_arr = _data.get(_inst, {}).get("time")
                if _t_arr is None or len(_t_arr) == 0:
                    logger.warning(f"TTV: inst '{_inst}' has no time data in BASEMENT.data")
                else:
                    _t_arr = np.asarray(_t_arr, dtype=float)
                    logger.info(
                        f"TTV: inst '{_inst}' N={len(_t_arr)} "
                        f"time=[{_t_arr.min():.4f}, {_t_arr.max():.4f}] "
                        f"span={_t_arr.max() - _t_arr.min():.3f} d"
                    )
            for _c in _settings.get("companions_phot", []):
                _epoch = float(median.get(f"{_c}_epoch", alles.BASEMENT.params[f"{_c}_epoch"]))
                _period = float(median.get(f"{_c}_period", alles.BASEMENT.params[f"{_c}_period"]))
                _tmids = _data.get(f"{_c}_tmid_observed_transits")
                _used_cache = _tmids is not None and len(_tmids) > 0
                if not _used_cache:
                    _times = []
                    for _inst in _inst_list:
                        _times += list(_data[_inst]["time"])
                    _times = np.sort(np.asarray(_times, dtype=float))
                    _tmids = get_tmid_observed_transits(
                        _times,
                        _epoch,
                        _period,
                        _fast_fit_width,
                    )
                    logger.info(
                        f"TTV: {_c} cache miss; recomputed via "
                        f"get_tmid_observed_transits "
                        f"(width={_fast_fit_width:.4f} d)"
                    )
                else:
                    logger.info(
                        f"TTV: {_c} using BASEMENT cache "
                        f"(populated by setup_ttv_fit; fit_ttvs={_settings.get('fit_ttvs')})"
                    )
                # Cross-check: independently recompute the transit count from
                # the union of all inst_phot times. If this disagrees with
                # the cache (which can happen if the cache was built from
                # fast-fit-reduced data of only one instrument or with a
                # stale epoch/period), prefer the fresh union count.
                _times_union = []
                for _inst in _inst_list:
                    _times_union += list(_data[_inst]["time"])
                _times_union = np.sort(np.asarray(_times_union, dtype=float))
                _tmids_union = get_tmid_observed_transits(
                    _times_union,
                    _epoch,
                    _period,
                    _fast_fit_width,
                )
                if len(_tmids_union) != len(_tmids):
                    logger.warning(
                        f"TTV: {_c} cache_count={len(_tmids)} "
                        f"vs fresh union_count={len(_tmids_union)} "
                        f"(epoch={_epoch:.6f}, period={_period:.6f}, "
                        f"width={_fast_fit_width:.4f}). "
                        f"Using union count."
                    )
                    _tmids = _tmids_union
                _n_obs = len(_tmids)
                logger.info(f"TTV: {_c} has {_n_obs} transits with data")
                if _n_obs == 0:
                    continue
                text += f"#TTV companion {_c},,,,,\n"
                for _j in range(_n_obs):
                    text += (
                        f"{_c}_ttv_transit_{_j + 1},0,1,uniform -0.05 0.05,"
                        f"TTV$_\\mathrm{{ttv;{_j + 1}}}$,d,\n"
                    )
        if debug:
            logger.info(text)
        fp = Path(results_dir, "params2.csv")
        np.savetxt(fp, [text], fmt="%s")
        logger.info(f"Saved: {fp}")
        sys.exit()

    else:
        toiid = args.toi
        ticid = args.tic
        ctoiid = args.ctoi
        name = args.name
        mission = args.mission.lower()
        quality_bitmask = args.quality
        sigma = args.sigma
        interactive = args.interactive
        ttv = args.ttv
        fns = args.filename if isinstance(args.filename, list) else [args.filename]
        fn = fns[0]  # first instrument (used where a single filename is needed)

        # -p/-e each accept multiple values. A single -p (the default) keeps
        # the original behaviour: only fns[0] is downloaded, and any other
        # --filename entries are expected to already exist on disk (e.g. from
        # a prior --lc-only run) — see the achromatic warning below. Passing
        # one -p per -f instead downloads every instrument in this one run,
        # each from its own pipeline/exptime, e.g.
        # -f spoc120 qlp600 -p spoc qlp -e 120 600.
        pipeline_list = args.pipeline
        if len(pipeline_list) > 1 and len(pipeline_list) != len(fns):
            logger.error(
                f"--pipeline has {len(pipeline_list)} entries but --filename has "
                f"{len(fns)} entries; give one --pipeline per --filename to "
                "download each instrument, or a single --pipeline value to "
                "download just the first --filename."
            )
            sys.exit(1)
        n_download = len(pipeline_list)

        exptimes = args.exptime
        if exptimes is None:
            exptimes = [None] * n_download
        elif len(exptimes) == 1:
            exptimes = exptimes * n_download
        elif len(exptimes) != n_download:
            logger.error(
                f"--exptime has {len(exptimes)} entries but --pipeline has "
                f"{n_download} entries; give one --exptime per --pipeline, "
                "or a single value shared by all."
            )
            sys.exit(1)
        # t_exp_{inst} in settings.csv is resolved per downloaded instrument
        # below; instruments beyond n_download (declared via --filename but
        # not downloaded here) fall back to this first exptime, matching the
        # single-pipeline behaviour prior to multi-pipeline support.
        t_exp_by_inst: dict[str, float] = {}
        exptime = exptimes[0]

        # ----------------------------------------------------------------- #
        # Bandpass resolution for chromatic mode.
        # bp_list  : one bandpass label per instrument, parallel to fns.
        # rr_suffixes  : list of suffixes used to key per-band rr rows.
        #                Empty list when achromatic (no `_<suffix>`).
        # ldc_suffixes : list of suffixes used to key LDC rows.
        #                When chromatic / bandpass-aware: unique bandpass set.
        #                Otherwise: the per-inst names in fns.
        # write_bandpass_line : whether settings.csv should carry a
        #                       `bandpass,<...>` row.
        # ----------------------------------------------------------------- #
        if args.bandpass is None:
            bp_list = None
            rr_suffixes = []  # achromatic: single {pl}_rr row
            ldc_suffixes = list(fns)  # per-instrument LDCs
            write_bandpass_line = False
            # Loud heads-up when the user gave several distinct instruments
            # but didn't ask for chromatic — that combination usually means
            # they wanted a per-band rr and forgot. Stay achromatic (backward
            # compatible) but make sure they see what they're getting.
            if len(set(fns)) >= 2:
                _bp_hint = " ".join(fns)
                logger.warning(
                    f"--bandpass not set, --filename has {len(set(fns))} distinct "
                    f"instruments ({fns}). The generated params.csv will be "
                    f"ACHROMATIC: a single {{pl}}_rr row shared by all "
                    f"instruments, and per-instrument LDCs."
                )
                logger.warning(
                    f"  - For chromatic (separate Rp/Rs per band): "
                    f"add `-bp {_bp_hint}` to your command."
                )
                logger.warning(
                    f"  - To group all instruments under one shared "
                    f"bandpass: add `-bp {fns[0]} {fns[0]}` (repeat one label)."
                )
        else:
            if len(args.bandpass) != len(fns):
                logger.error(
                    f"--bandpass has {len(args.bandpass)} entries but --filename has "
                    f"{len(fns)} entries; each instrument needs an explicit bandpass "
                    f"label (repeat the same label to share). Got bandpass={args.bandpass}, "
                    f"filenames={fns}."
                )
                sys.exit(1)
            bp_list = list(args.bandpass)
            # Preserve first-seen order so the emitted rows stay deterministic.
            _seen = set()
            unique_bp = [b for b in bp_list if not (b in _seen or _seen.add(b))]
            rr_suffixes = list(unique_bp)  # one rr row per unique bandpass
            ldc_suffixes = list(unique_bp)  # one LDC set per unique bandpass
            write_bandpass_line = True
            logger.info(
                f"Chromatic config: inst_phot={fns}, bandpass={bp_list} "
                f"({len(unique_bp)} unique → "
                f"{'chromatic' if len(unique_bp) > 1 else 'shared single-band'})"
            )

        # Unify the "segment" concept across missions: `sector` is always a
        # list-of-str id — TESS sector / K2 campaign / Kepler quarter.
        # --campaign and --quarter act as scalar aliases when -s is not given.
        def _as_segment_list(x):
            if x is None:
                return None
            if isinstance(x, list):
                return [str(v) for v in x]
            return [str(x)]

        if mission == "tess":
            sector = args.sector
        elif mission == "k2":
            sector = (
                args.sector
                if args.sector is not None
                else _as_segment_list(args.campaign if args.campaign is not None else -1)
            )
        elif mission == "kepler":
            sector = (
                args.sector
                if args.sector is not None
                else _as_segment_list(args.quarter if args.quarter is not None else -1)
            )
        else:
            sector = args.sector

        # Mission-appropriate BJD offset used when reconstructing full BJD.
        bjd_offset = {"tess": 2457000, "k2": 2454833, "kepler": 2454833}[mission]

        overwrite = args.overwrite
        update_db = args.update_db

        tic_parameters = {key: getattr(args, key) for key, *_rest in _TIC_TRANSIT_FIELDS}
        target_name, target_df, source = parse_target_name(
            toiid,
            ticid,
            ctoiid,
            name,
            update_db,
            tic_parameters=tic_parameters,
        )
        if ticid:
            name = target_name.replace("-", "")
        if mission == "tess":
            tic_id, outSec = get_tess_sectors(target_name, target_df, toiid, ctoiid, name)
            sector_flag = check_if_sector_is_available(
                target_name, given_sector=sector, all_sectors=outSec
            )
        else:
            # K2/Kepler: skip TESS-point lookup. Mirror the check_if_sector_is_available
            # flag logic without validating against a known segment list (lightkurve
            # will surface missing campaigns/quarters when the search returns empty).
            tic_id = None
            outSec = np.array([])
            if sector is None:
                sector_flag = "default"
            elif sector == ["all"]:
                sector_flag = "all_sector"
            elif sector == ["-1"]:
                sector_flag = "last"
            elif sector == ["0"]:
                sector_flag = "first"
            else:
                sector_flag = "multi_sector"

        fp = Path(basedir, target_name).joinpath(f"{target_name}.log")
        logger.add(fp, format=log_format)
        if debug:
            logger.info(target_df)

        try:
            if toiid or ctoiid or ticid:
                (
                    Teff,
                    Teff_err,
                    logg,
                    logg_err,
                    feh,
                    feh_err,
                    radius,
                    radius_err,
                    mass,
                    mass_err,
                ) = catalog_info_TIC(int(tic_id))
            elif name:
                (
                    Teff,
                    Teff_err,
                    logg,
                    logg_err,
                    feh,
                    feh_err,
                    radius,
                    radius_err,
                    mass,
                    mass_err,
                ) = catalog_info_name(target_df.iloc[0])
            if debug:
                logger.info(
                    Teff,
                    Teff_err,
                    logg,
                    logg_err,
                    feh,
                    feh_err,
                    radius,
                    radius_err,
                    mass,
                    mass_err,
                )
        except Exception as e:
            logger.error(f"Error: {e}")

        ticid = tic_id if ticid is None else ticid
        rhostar_prior = True
        if str(radius) == "nan":
            if interactive:
                radius = float(input("Rstar [Rsun]:"))
                rhostar_prior = False
            else:
                msg = "use --interactive to input missing value"
                logger.error(msg)
                sys.exit()
        if str(mass) == "nan":
            if interactive:
                mass = float(input("Mstar [Msun]:"))
                rhostar_prior = False
            else:
                msg = "use --interactive to input missing value"
                raise ValueError(msg)
        if str(radius_err) == "nan":
            radius_err = 0.1
            logger.info("Rstar err is nan; setting to 0.1")
        if str(mass_err) == "nan":
            mass_err = 0.1
            logger.info("Mstar err is nan; setting to 0.1")
        if debug:
            # logger.info(f"Teff={Teff:.0f}+/-{Teff_err:.0f},
            #     logg={logg:.2f}+/-{logg_err:.2f},
            #     feh={feh:.2f}+/-{feh_err:.2f}")
            logger.info(f"Rs={radius:.2f}+/-{radius_err:.2f}, Ms={mass:.2f}+/-{mass_err:.2f}")

        # band = mission.lower()
        if np.isnan(feh):
            feh = float(input("[Fe/H]: ")) if interactive else 0  # solar metallicity
        if np.isnan(feh_err):
            feh_err = float(input("[Fe/H] err: ")) if interactive else 0.1
        logger.info(f"Using [Fe/H]=({feh},{feh_err}) dex")
        if np.isnan(logg):
            if interactive:
                logg = float(input("logg: "))
            else:
                msg = "use --interactive to input missing value"
                logger.error(msg)
                sys.exit()  # no assumption
        if np.isnan(logg_err):
            logg_err = float(input("logg err: ")) if interactive else 0.1
        logger.info(f"Using logg=({logg:.2f},{logg_err:.2f}) cm/s^2")
        if np.isnan(Teff):
            if interactive:
                Teff = float(input("Teff: "))
            else:
                msg = "use --interactive to input missing value"
                logger.error(msg)
                sys.exit()  # no assumption
        if np.isnan(Teff_err):
            Teff_err = float(input("Teff err: ")) if interactive else 500
        logger.info(f"Using Teff=({Teff:.0f},{Teff_err:.0f}) K")

        q1, q1_err, q2, q2_err = ld.claret(
            band="T",
            teff=Teff,
            uteff=Teff_err,
            logg=logg,
            ulogg=logg_err,
            feh=feh,
            ufeh=feh_err,
            law="quadratic",
        )
        # ===== Write files =====#
        outdir = Path(basedir, target_name)
        # An existing directory is only a real conflict when it already holds
        # the canonical outputs of a successful prepare_allesfit run. An empty
        # dir (or one containing only stray artefacts from an aborted attempt)
        # should not block re-running without --overwrite.
        _CANONICAL_OUTPUTS = ("params.csv", "settings.csv", "run.py")
        if outdir.exists():
            existing = {p.name for p in outdir.iterdir()} if outdir.is_dir() else set()
            collisions = existing & set(_CANONICAL_OUTPUTS)
            if collisions and not overwrite:
                raise FileExistsError(
                    f"{outdir} already contains {sorted(collisions)}. "
                    "Use --overwrite to overwrite files."
                )
        outdir.mkdir(parents=True, exist_ok=True)

        # ===== Create params.csv =====#
        text = """#name,value,fit,bounds,label,unit,coupled_with\n"""
        # Track every catalog transit duration across all companions. After
        # each lightcurve download, the longest sets the ln_ρ_GP lower bound
        # so the GP cannot fit any companion's transit shape.
        companion_tdur_hours = []
        for i, row in target_df.iterrows():
            pl = planets[i]
            if debug:
                logger.info(f"=====Planet {pl}=====")

            # tic = row['TIC ID']
            Porb = row["Period (days)"]
            Porberr = row["Period (days) err"]
            epoch = row["Epoch (BJD)"]
            epocherr = row["Epoch (BJD) err"]
            tdur = row["Duration (hours)"]
            tdurerr = row["Duration (hours) err"]

            if interactive and not np.all([Porb > 0, epoch > 0, tdur > 0]):
                Porb = float(input("Porb: "))
                Porberr = float(input("Porb err: "))
                epoch = float(input("Epoch: "))
                epocherr = float(input("Epoch err: "))
            # tdur = float(input("Tdur: "))
            # tdurerr = float(input("Tdur err: "))
            else:
                assert np.all([Porb > 0, epoch > 0, tdur > 0])
            Porb_samples = np.random.normal(Porb, Porberr, size=Nsamples)

            if debug:
                logger.info(f"P={Porb:.4f}+/-{Porberr:.4f}, T0={epoch:.4f}+/-{epocherr:.4f}")

            rprs = np.sqrt(row["Depth (ppm)"] / 1e6)
            rprserr = np.sqrt(row["Depth (ppm) err"] / 1e6)

            if str(rprs) == "nan" or str(rprserr) == "nan":
                if interactive:
                    logger.info(
                        f"rprs={row['Depth (ppm)'] / 1e3:.3f}, rprserr={row['Depth (ppm) err'] / 1e3:.3f} ppt"
                    )
                    try:
                        rprs = input(f"Planet {pl} Rp/Rs (ppt): ")
                        rprs = float(rprs) / 1e3
                        rprserr = input(f"Planet {pl} Rp/Rs err (ppt): ")
                        rprserr = float(rprserr) / 1e3
                    except Exception as e:
                        msg = f"Error in inputs.\n{e}"
                        logger.error(msg)
                elif hasattr(row, "pl_rade"):
                    try:
                        rprs = row["pl_rade"] * u.Rearth.to(u.Rsun) / row["st_rad"]
                        Rperr = np.sqrt(row["pl_radeerr1"] ** 2 + row["pl_radeerr2"] ** 2)
                        rprserr = Rperr * u.Rearth.to(u.Rsun) / radius_err
                    except Exception as e:
                        msg = f"Error in parsing rp/rs\n{e}"
                        logger.error(msg)
                else:
                    msg = "Rp/Rs is nan. Try --interactive for manual input"
                    logger.error(msg)
                    sys.exit()
            assert rprs > 0

            rprs_samples = np.random.normal(rprs, rprserr, size=Nsamples)
            rprs_min, rprs, rprs_max = np.percentile(rprs_samples, q=quartiles_3sig)

            radius_samples = np.random.normal(radius, radius_err, size=Nsamples)
            mass_samples = np.random.normal(mass, mass_err, size=Nsamples)

            # compute a/Rs from stellar density
            rho_samples = rho_from_mr(mass_samples, radius_samples)
            a_over_Rs_samples = as_from_rhop(rho_samples, Porb_samples)
            if debug:
                rhomin, rho, rhomax = np.percentile(rho_samples, q=quartiles_3sig)
                as_min, a, as_max = np.percentile(a_over_Rs_samples, q=quartiles_3sig)
                a_au_s = a_from_rhoprs(rho_samples, Porb_samples, radius_samples)
                a_au_min, a_au, a_au_max = np.percentile(a_au_s, q=quartiles_3sig)

            # FIXME: a_over_Rs_samples produces some NaNs e.g. for Kepler-51
            rsuma_samples = 1 / a_over_Rs_samples * (1 + rprs_samples)
            idx = (rsuma_samples > 0) & (rsuma_samples < 1)
            rsuma_min, rsuma, rsuma_max = _percentile_3sig_safe(
                rsuma_samples[idx],
                fallback=(1e-3, 0.1, 0.25),
                label="(R*+Rp)/a from rho_star",
                planet=pl,
                logger=logger,
            )

            # A transit duration needs an impact-parameter assumption. Use the
            # median of the transit-conditioned non-grazing prior, b~U(0,1),
            # for the representative initial tuple.  Even the rho-star
            # fallback must couple b and inclination through b=(a/R*) cos(i).
            b = 0.5
            cosi = b * rsuma / (1.0 + rprs)
            inc = np.arccos(cosi)

            if str(tdur) != "nan":
                # Compare the stellar-density geometry to the catalog T14,
                # using the exact circular equation used by the fit model.
                tdur_rhostar = circular_transit_duration(Porb, rsuma, cosi, rprs) * 24
                logger.info(
                    f"tdur={tdur:.1f}h ({source}) {tdur_rhostar:.1f}h (derived from rhostar)"
                )
                try:
                    tdurerr = 1 if str(tdurerr) == "nan" else tdurerr
                    geometry = _duration_informed_geometry(
                        period_days=Porb,
                        period_err_days=Porberr,
                        tdur_hours=tdur,
                        tdur_err_hours=tdurerr,
                        radius_ratio=rprs,
                        radius_ratio_err=rprserr,
                    )
                    rsuma = geometry["rsuma"]
                    cosi = geometry["cosi"]
                    b = geometry["impact_parameter"]
                    inc = np.arccos(cosi)
                    rsuma_min = geometry["rsuma_min"]
                    rsuma_max = geometry["rsuma_max"]
                    tdur_orbit = circular_transit_duration(Porb, rsuma, cosi, rprs) * 24
                    logger.info(
                        f"tdur={tdur:.1f}h ({source}) {tdur_orbit:.1f}h "
                        "(duration-informed circular orbit; b=0.5)"
                    )
                    logger.info(
                        "Using duration-informed (Rstar+Rp)/a and cos(i) "
                        "under e=0 and b=0.5 initialization."
                    )
                except Exception as e:
                    logger.warning(
                        f"Could not derive circular geometry from catalog duration ({e}); "
                        "using the stellar-density geometry."
                    )

            model_tdur_hours = circular_transit_duration(Porb, rsuma, cosi, rprs) * 24.0
            if math.isfinite(model_tdur_hours) and model_tdur_hours > 0.0:
                companion_tdur_hours.append(model_tdur_hours)

            if debug:
                logger.info(f"rsuma={rsuma:.4f}")
                logger.info(f"rprs={rprs:.4f}")
                logger.info(f"rho={rho:.4f}")
                logger.info(f"a_s={a:.4f}")
                logger.info(f"a_au={a_au:.4f}")
                logger.info(f"inc={np.rad2deg(inc):.2f}")
                logger.info(f"b={b:.2f}")
            text += f"#companion {pl} astrophysical params,,,,,,\n"
            # rr: single achromatic key, or one per unique bandpass when
            # --bandpass was given. rsuma stays globally shared either way.
            # Bounds computed by _default_prior_bounds (see docstring for
            # the physical rationale).
            _bounds = _default_prior_bounds(rprs_max, rsuma_min, rsuma_max)
            _rr_upper = _bounds["rr_upper"]
            _rsuma_lo = _bounds["rsuma_lo"]
            _rsuma_hi = _bounds["rsuma_hi"]
            _cosi_max = _bounds["cosi_max"]
            if not rr_suffixes:
                text += f"{pl}_rr,{rprs:.4f},1,uniform 0 {_rr_upper:.4f},$R_{pl} / R_\\star$,,\n"
            else:
                for _bp in rr_suffixes:
                    text += (
                        f"{pl}_rr_{_bp},{rprs:.4f},1,uniform 0 {_rr_upper:.4f},"
                        f"$R_{pl} / R_\\star (\\mathrm{{{_bp}}})$,,\n"
                    )
            text += f"{pl}_rsuma,{rsuma:.4f},1,uniform {_rsuma_lo:.4f} {_rsuma_hi:.4f},$(R_\\star + R_{pl}) / a_{pl}$,,\n"
            # text += f"{pl}_rsuma,{rsuma:.4f},1,uniform 0 1,$(R_\star + R_{pl}) / a_{pl}$,,\n"
            text += (
                f"{pl}_cosi,{cosi:.4f},1,uniform 0 {_cosi_max:.4f},$\\cos" + "{i_" + pl + "}$,,\n"
            )
            text += (
                f"{pl}_epoch,{epoch:.6f},1,normal {epoch:.6f} {epocherr:.6f},$T_"
                + "{0;"
                + pl
                + "}$,BJD,\n"
            )
            text += f"{pl}_period,{Porb:.6f},1,normal {Porb:.6f} {Porberr:.6f},$P_{pl}$,d,\n"
            text += f"{pl}_f_c,0,0,uniform -1 1,$\\sqrt{{e_{pl}}} \\cos{{\\omega_{pl}}}$,,\n"
            text += f"{pl}_f_s,0,0,uniform -1 1,$\\sqrt{{e_{pl}}} \\sin{{\\omega_{pl}}}$,,\n"
            for inst in fns:
                text += f"#{pl}_sbratio_{inst},0,0,uniform 0 1,$J$,,\n"
        text += "#dilution per instrument,,,,,,\n"
        for inst in fns:
            text += f"dil_{inst},0,0,uniform 0 1,$D_\\mathrm{{0; {inst}}}$,,\n"
        # Limb-darkening coefficients are keyed by bandpass in chromatic mode
        # and by instrument otherwise — `ldc_suffixes` already encodes this
        # decision (unique bandpasses when --bandpass was given, fns otherwise).
        if write_bandpass_line:
            text += "#limb darkening coefficients per bandpass,,,,,,\n"
        else:
            text += "#limb darkening coefficients per instrument,,,,,,\n"
        for suffix in ldc_suffixes:
            text += (
                f"#host_ldc_q1_{suffix},{q1:.2f},1,normal {q1:.2f} {q1_err:.2f},"
                + f"$q_{{1; \\mathrm{{{suffix}}}}}$"
                + ",,\n"
            )
            text += (
                f"#host_ldc_q2_{suffix},{q2:.2f},1,normal {q2:.2f} {q2_err:.2f},"
                + f"$q_{{2; \\mathrm{{{suffix}}}}}$"
                + ",,\n"
            )
            text += f"host_ldc_q1_{suffix},0.5,1,uniform 0 1,$q_{{1; \\mathrm{{{suffix}}}}}$,,\n"
            text += f"host_ldc_q2_{suffix},0.5,1,uniform 0 1,$q_{{2; \\mathrm{{{suffix}}}}}$,,\n"

        # ln(err) upper bound = -3 → exp(-3) ≈ 5% relative scatter, already
        # generous; the previous -1 (exp(-1) ≈ 37%) covered a regime where
        # no transit signal is recoverable, so samples there were wasted.
        text += "#errors per instrument,,,,,,\n"
        for inst in fns:
            text += (
                f"ln_err_flux_{inst},-6,1,uniform -10 -3,$\\log{{\\sigma ({inst})}}$,rel. flux,\n"
            )
        # GP Matern32 ln(rho) upper bound = 5 → exp(5) ≈ 148 days, beyond
        # any plausible TESS systematic or Kepler quarter, and short of
        # the regime where the kernel is degenerate with the baseline
        # offset (correlations >>1 yr). The previous 10 (exp(10) ≈ 60 yr)
        # was unhelpful.
        text += "#baseline per instrument,,,,,,\n"
        for inst in fns:
            text += f"#baseline_gp_offset_flux_{inst},0,1,uniform -0.1 0.1,$\\mathrm{{offset ({inst})}}$,,\n"
            text += f"baseline_gp_matern32_lnsigma_flux_{inst},-5,1,uniform -10 -3,$\\mathrm{{gp ln \\sigma ({inst})}}$,,\n"
            text += f"baseline_gp_matern32_lnrho_flux_{inst},0,1,uniform -1 5,$\\mathrm{{gp ln \\rho ({inst})}}$,,\n"
        # TTV rows: when --ttv is NOT set, keep the commented-out stub for
        # reference. When --ttv IS set, the real per-transit rows are
        # appended after the lightcurve download (below) because we need
        # the observed time series to count transits with data coverage.
        if not ttv:
            for i, _ in target_df.iterrows():
                pl = planets[i]
                text += f"#TTV companion {pl},,,,,\n"
                # ±0.05 d = ±72 min; real TTV uncertainties for TESS-class
                # transit S/N are of order minutes, so a narrower default
                # is fine and converges faster.
                for i in range(5):
                    text += f"#{pl}_ttv_transit_{i + 1},0,1,uniform -0.05 0.05,TTV$_\\mathrm{{ttv;{i + 1}}}$,d,\n"
        if debug:
            logger.info(text)
        fp = outdir.joinpath("params.csv")
        np.savetxt(fp, [text], fmt="%s")
        logger.info(f"Saved: {fp}")

        # =====Create params_star.csv===== #
        text3 = f"""#R_star,R_star_lerr,R_star_uerr,M_star,M_star_lerr,M_star_uerr,Teff_star,Teff_star_lerr,Teff_star_uerr
    #R_sun,R_sun,R_sun,M_sun,M_sun,M_sun,K,K,K
    {radius:.2f},{radius_err:.2f},{radius_err:.2f},{mass:.2f},{mass_err:.2f},{mass_err:.2f},{Teff:.0f},{Teff_err:.0f},{Teff_err:.0f}"""
        if debug:
            logger.info(text3)

        fp = outdir.joinpath("params_star.csv")
        np.savetxt(fp, [text3], fmt="%s")
        logger.info(f"Saved: {fp}")

        # =====Create run.py===== #
        # Every ns_fit / ns_output / mcmc_fit / mcmc_output call is wrapped
        # in allesfitter.log_run so the centralized log at
        # ~/.allesfitter/runs.jsonl (override via $ALLESFITTER_RUN_LOG)
        # records a start/end row with absolute datadir, pid, hostname,
        # duration, and status. Inspect with: jq . ~/.allesfitter/runs.jsonl
        # or python -c 'import allesfitter; print(allesfitter.run_log_tail(20))'.
        text4 = """#!/usr/bin/env python
import allesfitter
#import os; os.nice(19)

dir_path = '.'
### ===show fit using initial guess===
fig = allesfitter.show_initial_guess(dir_path)
#allesfitter.prepare_ttv_fit(dir_path, style='tessplot')

### ===global optimization (methods: cmaes, differential_evolution, dual_annealing)===
#allesfitter.optimize(method='differential_evolution', refine=True)

### ===mcmc sampling===
#with allesfitter.log_run('mcmc_fit', dir_path):
#    allesfitter.mcmc_fit(dir_path, append=True)
#allesfitter.mcmc_output(dir_path, overwrite=True)

### ===nested sampling for evidence / model comparison===
#with allesfitter.log_run('ns_fit', dir_path):
#    allesfitter.ns_fit(dir_path, overwrite=False)
#allesfitter.ns_output(dir_path, overwrite=True)"""

        if debug:
            logger.info(text4)

        fp = outdir.joinpath("run.py")
        np.savetxt(fp, [text4], fmt="%s")
        logger.info(f"Saved: {fp}")

        query_name = name
        if toiid or ctoiid:
            query_name = f"TIC {ticid}"
        elif name.lower()[:2] == "k2":
            # search epic name or coordinates
            try:
                logger.info("Searching for EPIC name")
                query_name = get_name_aliases(name, key="epic")
            except Exception as e:
                logger.info(f"Error: {e}")

        # One iteration per (fn, pipeline, exptime) triple. With the default
        # single -p, this loop runs once for fns[0], exactly as before.
        for fn, pipeline, exptime in zip(fns[:n_download], pipeline_list, exptimes):
            lc_type = "sap_flux" if pipeline == "qlp" else args.lc_type + "_flux"

            # search all available data for reference
            all_lcs = lk.search_lightcurve(query_name, mission=mission)
            logger.info(all_lcs)
            if len(all_lcs) > 0:
                available_pipelines = set([i.lower() for i in all_lcs.author])
                logger.info(f"Available Pipelines: {available_pipelines}")
            else:
                msg = "No light curves found."
                logger.error(msg)
                sys.exit()
            unique_exptimes = all_lcs.table.to_pandas().exptime.unique()
            logger.info(f"Available Exp. times: {unique_exptimes}")
            idx = [i == pipeline.lower() for i in available_pipelines]
            if sum(idx) == 0:
                msg = f"pipeline={pipeline} not in {available_pipelines}"
                logger.error(msg)
                sys.exit()

            # search only requested data
            result = lk.search_lightcurve(
                query_name, author=pipeline, exptime=exptime, mission=mission
            )
            if result:
                # K2 campaigns 11a/11b cannot cast to int. Keep labels as
                # strings throughout and natural-sort for display.
                sectors = [_parse_segment_label(s) for s in result.mission]
                unique_sectors = sorted(set(sectors), key=_natural_segment_sort_key)
                if sector_flag == "all_sector":
                    # case: sector='all'
                    # `sectors` mirrors `result.mission` (one row per exposure
                    # time / flux column), so it has duplicates and is in search
                    # order. Log the deduped+sorted view for human readability.
                    _w_p = _segment_word(mission, plural=True)
                    logger.info(
                        f"Using {pipeline.upper()} pipeline in {len(unique_sectors)} {_w_p}: {unique_sectors}"
                    )
                    unique_exptimes = result.table.to_pandas().exptime.unique()
                    if len(unique_exptimes) > 1:
                        result, exptime = _handle_multiple_exptimes(
                            result,
                            unique_exptimes,
                            logger,
                            context=f"all {_w_p}",
                            expected_n=len(unique_sectors),
                        )
                    exptime = unique_exptimes[0] if exptime is None else exptime
                    lc = _safe_stitch(
                        result.download_all(flux_column=lc_type, quality_bitmask=quality_bitmask),
                        logger=logger,
                    )
                    if lc is None:
                        logger.error("No usable segments downloaded.")
                        sys.exit()
                    logger.info(
                        "The lightcurves were not flattened/de-trended to avoid removing transits."
                    )
                    # Header sanity check: only meaningful when a single segment
                    # was downloaded. After stitching multiple segments the
                    # collapsed lc.{sector,campaign,quarter} can't sensibly equal
                    # any one of them (and for K2 11a/11b the alpha suffix would
                    # never match the int the LightCurve carries).
                    _seg = (
                        getattr(lc, "sector", None)
                        or getattr(lc, "campaign", None)
                        or getattr(lc, "quarter", None)
                    )
                    if _seg is not None and len(unique_sectors) == 1:
                        assert _segments_match(_seg, unique_sectors[-1])
                    if pipeline == "spoc":
                        _lc1_for_meta = result.download_all(
                            quality_bitmask=quality_bitmask, flux_column="pdcsap_flux"
                        )
                        lc1 = _safe_stitch(_lc1_for_meta, logger=logger)
                        lc2 = _safe_stitch(
                            result.download_all(
                                quality_bitmask=quality_bitmask, flux_column="sap_flux"
                            ),
                            logger=logger,
                        )
                elif sector_flag == "multi_sector":
                    # case: sector int or list
                    _w = _segment_word(mission)
                    _w_p = _segment_word(mission, plural=True)
                    idx = [any(_segments_match(i, s) for s in sector) for i in sectors]
                    msg = f"{pipeline.upper()} lightcurves for {_w}={sector} is not available. Try {_w}={unique_sectors}."
                    assert sum(idx) > 0, logger.error(msg)

                    filtered_result = result[idx]
                    unique_exptimes = filtered_result.table.to_pandas().exptime.unique()
                    msg = f"Using {pipeline.upper()} pipeline in {len(sector)} {_w_p}: {sector} (exptime={unique_exptimes} sec).\n"
                    if sector_flag != "all_sector":
                        msg += f"Otherwise use {_w}=({unique_sectors}, all))."
                    logger.info(msg)
                    if len(sector) > len(filtered_result):
                        msg = f"Not all {_w}={sector} have exptime={exptime} sec.\n"
                        msg = f"Try to limit the {_w_p}.\n"
                        logger.error(msg)
                        sys.exit()
                    elif len(sector) < len(filtered_result):
                        filtered_result, exptime = _handle_multiple_exptimes(
                            filtered_result,
                            unique_exptimes,
                            logger,
                            context=f"the given {_w_p}",
                            expected_n=len(sector),
                        )
                    assert len(sector) == len(filtered_result)
                    exptime = unique_exptimes[0] if exptime is None else exptime
                    lc = _safe_stitch(
                        filtered_result.download_all(
                            quality_bitmask=quality_bitmask, flux_column=lc_type
                        ),
                        logger=logger,
                    )
                    if lc is None:
                        logger.error("No usable segments downloaded.")
                        sys.exit()
                    logger.info(
                        "The lightcurves were not flattened/de-trended to avoid removing transits."
                    )
                    _w = _segment_word(mission)
                    _seg = (
                        getattr(lc, "sector", None)
                        or getattr(lc, "campaign", None)
                        or getattr(lc, "quarter", None)
                    )
                    msg = f"{_w}={_seg} in header not in requested {_w}={sector}"
                    # Skip the header check when multiple segments were stitched —
                    # the collapsed attribute can't represent more than one.
                    if _seg is not None and len(sector) == 1:
                        assert any(_segments_match(_seg, s) for s in sector), logger.error(msg)
                    if pipeline == "spoc":
                        _lc1_for_meta = filtered_result.download_all(
                            quality_bitmask=quality_bitmask, flux_column="pdcsap_flux"
                        )
                        lc1 = _safe_stitch(_lc1_for_meta, logger=logger)
                        lc2 = _safe_stitch(
                            filtered_result.download_all(
                                quality_bitmask=quality_bitmask, flux_column="sap_flux"
                            ),
                            logger=logger,
                        )
                else:
                    if sector_flag == "first":
                        idx = 0
                        sector = sectors[idx]
                    elif sector_flag == "last" or sector_flag == "default":
                        idx = -1
                        sector = sectors[idx]

                    filtered_result = result[idx]
                    lc = filtered_result.download(
                        quality_bitmask=quality_bitmask, flux_column=lc_type
                    ).normalize()
                    unique_exptimes = filtered_result.table.to_pandas().exptime.unique()
                    # logger.info(f"Exp times: {unique_exptimes}")
                    exptime = unique_exptimes[0] if exptime is None else exptime
                    _seg = (
                        getattr(lc, "sector", None)
                        or getattr(lc, "campaign", None)
                        or getattr(lc, "quarter", None)
                    )
                    if _seg is not None:
                        assert _segments_match(_seg, sector)
                    logger.info(
                        f"Using {pipeline.upper()} pipeline in {_segment_word(mission)} {sector}."
                    )
                    if pipeline == "spoc":
                        lc1 = filtered_result.download(
                            quality_bitmask=quality_bitmask, flux_column="pdcsap_flux"
                        ).normalize()
                        _lc1_for_meta = lc1  # single LightCurve carries its own .meta
                        lc2 = filtered_result.download(
                            quality_bitmask=quality_bitmask, flux_column="sap_flux"
                        ).normalize()
                if sigma:
                    nbefore = len(lc)
                    lc = lc.remove_outliers(sigma=sigma)
                    nafter = len(lc)
                    if nbefore > nafter:
                        diff = nbefore - nafter
                        logger.info(f"Removed {diff} outliers using sigma={sigma}.")
                    if pipeline == "spoc":
                        lc1 = lc1.remove_outliers(sigma=sigma)
                        lc2 = lc2.remove_outliers(sigma=sigma)
                if sector_flag == "all_sector":
                    secs = "s".join(map(str, unique_sectors))
                else:
                    if isinstance(sector, list):
                        secs = "s".join(map(str, sector))
                    else:
                        secs = str(sector)
                if pipeline == "spoc" and len(lc1) == len(lc2):
                    fig, axs = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
                    _ = lc1.scatter(ax=axs[0], zorder=2, label="PDCSAP", c="C0")
                    _ = lc2.scatter(ax=axs[0], zorder=1, label="SAP", c="C1")
                    axs[0].set_title(
                        f"{_segment_word(mission).capitalize()}={secs}\nexptime={int(exptime)}s"
                    )
                    (lc1 - lc2).scatter(ax=axs[1], label="difference", c="k")
                    fp = outdir.joinpath(
                        f"{target_name}_{mission}_{lc_type.split('_')[0]}_s{secs}_exp{int(exptime)}s"
                    )
                    fig.savefig(fp.with_suffix(".png"))
                    # Record SPOC CROWDSAP -> allesfitter dilution next to the lightcurve
                    # so users can fill in dil_<inst> in params.csv without guessing.
                    contam_fp = outdir.joinpath(f"{fn}_spoc_contamination.txt")
                    try:
                        summary = _write_spoc_contamination(_lc1_for_meta, contam_fp, mission)
                        if summary:
                            logger.info(f"Saved: {contam_fp}")
                            # Prepend a commented normal-prior dilution row above the
                            # existing uniform row so users can opt into the
                            # SPOC-informed prior without recomputing the value.
                            params_fp = outdir.joinpath("params.csv")
                            if _inject_dilution_normal_prior(
                                params_fp,
                                fn,
                                median=summary["median_dilution"],
                                std=summary["std_dilution"],
                            ):
                                logger.info(
                                    f"params.csv: added commented SPOC normal-prior row for dil_{fn} "
                                    f"(median={summary['median_dilution']:.4f}, "
                                    f"n_segments={summary['n_segments']})"
                                )
                        else:
                            logger.info(
                                "CROWDSAP not present in SPOC header — skipping spoc_contamination.txt."
                            )
                    except Exception as _e:
                        logger.warning(f"Could not write spoc_contamination.txt: {_e}")
                else:
                    ax = lc.scatter(label=pipeline)
                    ax.set_title(
                        f"{_segment_word(mission).capitalize()}={secs}\nexptime={int(exptime)}s"
                    )
                    fp = outdir.joinpath(
                        f"{target_name}_{mission}_{pipeline}_s{secs}_exp{int(exptime)}s"
                    )
                    ax.figure.savefig(fp.with_suffix(".png"))
                logger.info(f"Saved: {fp.with_suffix('.png')}")
                df = lc.to_pandas()
                msg = "Somehow, the lightcurve data is empty."
                assert len(df) > 0, logger.error(msg)
                df["time"] = df.index + bjd_offset
                df = df.reset_index(drop=True).sort_values(by="time")
                cols = ["time", "flux", "flux_err"]
                msg = f"Somehow, `flux_err` is all NaN.\n{df[cols]}\n"
                if len(df["flux_err"].dropna(axis="index")) == 0:
                    df["flux_err"] = 1
                    msg += "Setting flux error column = 1 (See allesfitter documentation)."
                    logger.error(msg)
                df2 = df[cols].dropna(axis="index")
                msg = "Lightcurve is all NaN. No lightcurve downloaded."
                assert len(df2) > 0, logger.error(msg)
                df2[cols].to_csv(fp.with_suffix(".csv"), sep=",", header=False, index=False)
                logger.info(f"Saved: {fp.with_suffix('.csv')}")
                # Also save under the instrument label allesfitter expects
                # (inst_phot in settings.csv). This is the file the fit and
                # allesclass load at runtime.
                inst_fp = outdir.joinpath(f"{fn}.csv")
                df2[cols].to_csv(inst_fp, sep=",", header=False, index=False)
                logger.info(f"Saved: {inst_fp}")

                # Refine the GP / noise prior bounds for this instrument using
                # the actual cadence, baseline, and per-cadence scatter of the
                # downloaded lightcurve. Replaces the generic, hardcoded
                # bounds with data-aware ones (see _dataset_aware_gp_bounds).
                try:
                    _tdur_d = _gp_protective_tdur_days(companion_tdur_hours)
                    _gp = _dataset_aware_gp_bounds(
                        df2["time"].to_numpy(),
                        df2["flux"].to_numpy(),
                        tdur_days=_tdur_d,
                    )
                    if _gp is not None:
                        _update_params_gp_bounds(
                            outdir.joinpath("params.csv"), fn, _gp, logger=logger
                        )
                except Exception as _e:
                    logger.warning(f"Could not refine GP priors for {fn}: {_e}")

                logger.info(f"Ndata: {len(df):,}")
                logger.info(df[cols].head())
                if debug:
                    logger.info(df.head())

                # t_exp_{inst} in settings.csv is resolved from the exptime
                # actually used for this instrument's download, not a single
                # global value — necessary once pipelines have different
                # cadences (e.g. SPOC 120 s + QLP 600 s).
                t_exp_by_inst[fn] = exptime

                # TTV rows are appended at the very end of main() once all
                # config files and the {fn}.csv lightcurve are on disk — that
                # lets us use allesclass(outdir).BASEMENT.data to pick up the
                # same observed-transit counts the fit will compute.
            else:
                msg = f"No lightcurve downloaded for pipeline={pipeline}. Check inputs."
                logger.error(msg)
                sys.exit()

        # =====Create settings.csv===== #
        text2 = """#name,value
###############################################################################,
# General settings,
###############################################################################,\n"""

        phot_companions = planets[: len(target_df)]
        text2 += f"companions_phot,{' '.join(phot_companions)}"
        # When --bandpass is provided, emit a real `bandpass,...` row so the
        # parser activates chromatic mode (one rr per unique bandpass). When
        # omitted, leave a commented stub so users can see where to add one.
        if write_bandpass_line:
            _bandpass_row = f"bandpass,{' '.join(bp_list)}"
        else:
            _bandpass_row = f"#bandpass,{' '.join(fns)}"
        text2 += f"""
companions_rv,
inst_phot,{" ".join(fns)}
inst_rv,
time_format,BJD_TDB
{_bandpass_row}
{_transit_requirement_settings(phot_companions).rstrip()}
#flux_min_raw,0.9
#flux_max_raw,1.1
###############################################################################,
# Fit performance settings,
###############################################################################,
multiprocess,True
multiprocess_cores,40
fast_fit,True
fast_fit_width,0.3333333333333333
#fast_fit_width,0.5
secondary_eclipse,False
phase_curve,False
shift_epoch,True\n"""
        for i, _ in target_df.iterrows():
            pl = planets[i]
            text2 += f"inst_for_{pl}_epoch,all\n"
            text2 += f"#inst_for_{pl}_epoch,{' '.join(fns)}\n"
        text2 += """###############################################################################,
# MCMC settings,
###############################################################################,
mcmc_nwalkers,100
mcmc_total_steps,2000
mcmc_burn_steps,1000
mcmc_thin_by,2
###############################################################################,
# Nested Sampling settings,
###############################################################################,
ns_backend,dynesty
ns_modus,dynamic
ns_nlive,1000
ns_bound,multi
ns_sample,auto
ns_tol,100
###############################################################################,
# Limb darkening law per object and instrument,
# if 'quad' two corresponding parameter called 'ldc_q1_inst' and 'ldc_q2_inst' have to be given in params.csv,
###############################################################################,\n"""
        for inst in fns:
            text2 += f"host_ld_law_{inst},quad\n"
        text2 += """#####################################,
# Exposure interpolation settings,
#####################################,\n"""
        # Resolve exposure time strictly from input: --exptime (in seconds)
        # when supplied, else the value lightkurve reported for the selected
        # data (also in seconds). Each downloaded instrument keeps its own
        # resolved exptime (t_exp_by_inst) — necessary once pipelines have
        # different cadences (e.g. SPOC 120 s + QLP 600 s). An instrument
        # declared via --filename but not downloaded here (the single -p
        # case, where only fns[0] is fetched) falls back to fns[0]'s
        # exptime, matching pre-multi-pipeline behaviour. We never derive
        # exptime from np.diff(time) — that conflates sampling cadence with
        # actual exposure and corrupts ellc's exposure-time integration.
        text2 += "### crucial only for long (>600s) exposure times,\n"
        _fallback_exptime = t_exp_by_inst[fns[0]]
        for inst in fns:
            _exptime_sec = int(round(t_exp_by_inst.get(inst, _fallback_exptime)))
            _t_exp_val = _exptime_sec / 86400.0
            assert _t_exp_val < 1, f"t_exp_val should be < 1 day, got {_t_exp_val}"
            text2 += f"t_exp_{inst},{_t_exp_val:.6f}\n"
            text2 += f"#t_exp_{inst},0.020833\n"
            text2 += f"#t_exp_n_int_{inst},10\n"
        text2 += """###############################################################################,
# Baseline settings per instrument: sample / hybrid,
# if 'sample_offset' one corresponding parameter called 'baseline_offset_key_inst' has to be given in params.csv,
# if 'sample_linear' two corresponding parameters called 'baseline_offset_key_inst' and 'baseline_slope_key_inst' have to be given in params.csv,
# if 'sample_GP' two corresponding parameters called 'baseline_gp1_key_inst' and 'baseline_gp2_key_inst' have to be given in params.csv,
###############################################################################,\n"""
        for inst in fns:
            text2 += f"#baseline_flux_{inst},sample_offset\n"
            text2 += f"#baseline_flux_{inst},sample_linear\n"
            text2 += f"#baseline_flux_{inst},hybrid_spline\n"
            text2 += f"#baseline_flux_{inst},hybrid_poly_2\n"
            text2 += f"baseline_flux_{inst},sample_GP_Matern32\n"
        text2 += """###############################################################################,
# Error settings (overall scaling) per instrument: sample / hybrid,
# if 'sample' one corresponding parameter called 'ln_err_key_inst' (photometry) or 'ln_jitter_key_inst' (RV) has to be given in params.csv,
###############################################################################,\n"""
        for inst in fns:
            text2 += f"error_flux_{inst},sample\n"
        text2 += """###############################################################################,
# Flares,
# if N>0 4xN corresponding parameters has to be given in params.csv,
# See https://github.com/MNGuenther/allesfitter/blob/master/paper/GJ_1243/allesfit_0/params.csv,
###############################################################################,
#N_flares,0
###############################################################################,
# Number of spots per object and instrument,
# if N>0 3xN corresponding parameters has to be given in params.csv,
###############################################################################,\n"""
        for inst in fns:
            text2 += f"#host_N_spots_{inst},0\n"
        text2 += """###############################################################################,
# Host density prior,
###############################################################################,\n"""
        text2 += f"use_host_density_prior,{rhostar_prior}"
        text2 += """
###############################################################################,
# Stellar variability: sample_GP_SHO / _real / _complex / matern32,
# if 'sample_GP_SHO' three corresponding parameters has to be given in params.csv,
# See https://github.com/MNGuenther/allesfitter/blob/master/tutorials/06_transits_and_rvs_with_stellar_variability/allesfit/params.csv,
###############################################################################,
#stellar_var_flux,sample_GP_SHO
#stellar_var_rv,sample_GP_real
###################################################,
# Fit TTV,
###################################################,
fit_ttvs,False
###############################################################################,
# Stellar grid per object and instrument,
###############################################################################,\n"""
        for inst in fns:
            text2 += f"host_grid_{inst},sparse\n"
        for i, _ in target_df.iterrows():
            pl = planets[i]
            for inst in fns:
                text2 += f"#{pl}_grid_{inst},sparse\n"
                text2 += f"#{pl}_shape_{inst},sphere\n"
                text2 += f"#{pl}_flux_weighted_{inst},False\n"

        if debug:
            logger.info(text2)

        fp = outdir.joinpath("settings.csv")
        np.savetxt(fp, [text2], fmt="%s")
        logger.info(f"Saved: {fp}")

        # ===== Append TTV rows to params.csv using allesclass ===== #
        # allesclass loads params.csv + settings.csv + {inst}.csv and,
        # during BASEMENT init, may populate
        #   BASEMENT.data[f'{c}_tmid_observed_transits']
        # for every companion in companions_phot. That list holds exactly
        # the transits the fit will see, so len(...) is the right count
        # for {c}_ttv_transit_N (N indexes the N-th observed transit).
        # If the cache is missing for a companion, replicate what
        # afplot_per_transit does in allesfitter/general_output.py
        # (lines ~887-892): build times_combined across inst_phot and
        # call get_tmid_observed_transits with a window of fast_fit_width
        # (= T_tra_tot equivalent in days).
        if ttv:
            try:
                alles = allesclass(str(outdir))
                _settings = alles.BASEMENT.settings
                _params = alles.BASEMENT.params
                _data = alles.BASEMENT.data
                _fast_fit_width = float(_settings.get("fast_fit_width", 0.3333333333333333))
                ttv_text = ""
                for _c in _settings.get("companions_phot", []):
                    _key = f"{_c}_tmid_observed_transits"
                    _tmids = _data.get(_key)
                    if _tmids is None or len(_tmids) == 0:
                        # Fallback: compute on the fly, mirroring
                        # afplot_per_transit in general_output.py.
                        _times = []
                        for _inst in _settings.get("inst_phot", []):
                            _times += list(_data[_inst]["time"])
                        _times = np.sort(np.asarray(_times, dtype=float))
                        _epoch = float(_params[f"{_c}_epoch"])
                        _period = float(_params[f"{_c}_period"])
                        _T_tra_tot = _fast_fit_width  # days
                        _tmids = get_tmid_observed_transits(
                            _times,
                            _epoch,
                            _period,
                            _T_tra_tot,
                        )
                        logger.info(
                            f"TTV: {_c} cache miss; recomputed via "
                            f"get_tmid_observed_transits (width={_T_tra_tot:.4f} d)"
                        )
                    _n_obs = len(_tmids)
                    logger.info(f"TTV: {_c} has {_n_obs} transits with data")
                    if _n_obs == 0:
                        continue
                    ttv_text += f"#TTV companion {_c},,,,,\n"
                    for _j in range(_n_obs):
                        # ±0.05 d = ±72 min — matches the comment-out
                        # default and the early-TTV emission block.
                        ttv_text += (
                            f"{_c}_ttv_transit_{_j + 1},0,1,uniform -0.05 0.05,"
                            f"TTV$_\\mathrm{{ttv;{_j + 1}}}$,d,\n"
                        )
                if ttv_text:
                    params_fp = outdir.joinpath("params.csv")
                    with open(params_fp, "a") as _f:
                        _f.write(ttv_text)
                    logger.info(f"Appended TTV rows to {params_fp}")
                else:
                    logger.error(
                        "TTV: no companion had observed transits; nothing appended to params.csv"
                    )
            except Exception as e:
                logger.error(f"TTV append via allesclass failed: {e}")

        # ===== Prior-bound sanity checks =====
        # Surface obvious problems with GP/noise priors (e.g. GP amplitude
        # large enough to swallow the transit, GP timescale longer than the
        # observation baseline) before the user kicks off a multi-hour fit.
        try:
            from allesfitter.validation import validate_gp_priors

            warnings = validate_gp_priors(outdir, log=logger.warning)
            if not warnings:
                logger.info("Prior sanity: GP / noise bounds look reasonable.")
        except Exception as e:
            logger.warning(f"Prior sanity check skipped: {e}")


if __name__ == "__main__":
    main()
