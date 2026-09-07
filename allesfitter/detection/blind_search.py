"""Blind ``transitleastsquares`` search on top of a completed allesfitter fit.

``transit_search()`` takes an MCMC or NS ``*_results`` directory and:

1. Pulls the posterior-median parameters (best-fit baseline hyperparameters
   and known-companion ephemerides) via the same
   ``get_params_from_samples`` path ``mcmc_output``/``ns_output`` use.
2. Reconstructs each photometric instrument's full raw light curve
   (``Basement.fulldata``, i.e. before any ``fast_fit`` windowing) and
   subtracts the best-fit baseline model (:func:`allesfitter.computer.
   calculate_baseline`, evaluated on that full time grid) from it.
3. Before masking anything, optionally checks each known companion's own
   recoverability: isolates it (masks every *other* known companion) and
   runs a narrow, one-shot TLS scan bracketing its own period, so a
   low-significance known planet is flagged rather than silently masked away.
4. Masks out every known companion in ``params.csv`` (period/epoch from the
   posterior, duration from the circular transit-chord equation).
5. Runs :func:`allesfitter.detection.transit_search.tls_search` iteratively
   — it already masks each new detection and loops — stopping once SDE
   drops below ``sde_min``.
6. Writes a per-candidate figure (raw light curve, flattened/detrended light
   curve with the candidate's model overlaid, TLS periodogram with harmonics
   and known-planet reference lines, and phase-folded data + TLS model) and
   a quicklook-style ``.h5`` file (readable by ``prepare_allesfit.py --h5``)
   for every candidate found above threshold — and the same kind of figure
   (without an ``.h5``) for every known-companion recovery check from step 3.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from transitleastsquares import transit_mask

from .. import config
from ..computer import calculate_baseline
from ..exoworlds_rdx.lightcurves.lightcurve_tools import rebin_err
from ..general_output import get_params_from_samples
from ..lightcurves import translate_limb_darkening_from_q_to_u as q_to_u
from ..plot_utils import broken_xaxis_subplots
from ..results import target_output_directory
from ..tls_h5 import write_h5_tls_result
from ..validation.prior_checks import transit_duration_days
from .gpu_tls import run_tls as _run_tls_engine
from .notch_locor import locor_flatten, notch_flatten
from .transit_search import mask, tls_search

#: Maps a results directory's basename to the sampler that produced it.
_SAMPLER_BY_DIRNAME = {"mcmc_results": "mcmc", "ns_results": "ns"}

#: Alternatives to the default best-fit-baseline detrending, ported from
#: quicklook's ``notch_locor`` module (Rizzuto et al. 2017).
_FLATTEN_METHODS = ("baseline", "notch", "locor")

#: Sliding-window width (days) used by "notch" when --flatten-window-length
#: isn't given; matches the upstream Notch default for short-cadence data.
_DEFAULT_NOTCH_WINDOW_LENGTH = 0.5


def _resolve_sampler(results_dir: str) -> tuple[str, Path]:
    """Return ``(sampler, datadir)`` for an ``mcmc_results``/``ns_results`` dir."""
    results_path = Path(results_dir).expanduser().resolve()
    sampler = _SAMPLER_BY_DIRNAME.get(results_path.name)
    if sampler is None:
        raise ValueError(
            f"{results_dir!r} is not a recognized results directory — expected its "
            "basename to be 'mcmc_results' or 'ns_results' (the directory "
            "'allesfitter mcmc-fit'/'ns-fit' wrote its output to)."
        )
    return sampler, results_path.parent


def _load_posterior_params(datadir: Path, sampler: str) -> dict:
    """Load posterior-median params for ``datadir``, initializing ``config.BASEMENT``."""
    if sampler == "mcmc":
        from ..mcmc_output import get_mcmc_posterior_samples

        samples = get_mcmc_posterior_samples(str(datadir), as_type="2d_array")
    else:
        from ..nested_sampling_output import get_ns_posterior_samples

        samples = get_ns_posterior_samples(str(datadir), as_type="2d_array")
    params_median, _params_ll, _params_ul = get_params_from_samples(samples)
    return params_median


def _known_companion_windows(
    params_median: dict, companions: list[str], mask_width_factor: float
) -> list[dict]:
    """Build ``{companion, period, epoch, duration}`` mask windows from the
    posterior, one per companion with a resolvable transit geometry.
    ``duration`` is already widened by ``mask_width_factor``."""
    windows = []
    for companion in companions:
        try:
            period = float(params_median[companion + "_period"])
            epoch = float(params_median[companion + "_epoch"])
            rsuma = float(params_median[companion + "_rsuma"])
            cosi = float(params_median[companion + "_cosi"])
            rr = float(params_median[companion + "_rr"])
        except (KeyError, TypeError, ValueError):
            continue
        duration = transit_duration_days(period, rsuma, cosi, rr)
        if duration is None or not np.isfinite(duration) or duration <= 0:
            continue
        windows.append(
            {
                "companion": companion,
                "period": period,
                "epoch": epoch,
                "duration": duration * mask_width_factor,
            }
        )
    return windows


def _estimate_locor_period(base, quiet: bool) -> float:
    """Auto-estimate a rotation period for LOCoR via a Lomb-Scargle
    periodogram over every instrument's concatenated raw flux.

    ``clip=False`` deliberately skips ``periodicity.estimate_period``'s
    default slide-clip pass, which requires the optional ``wotan``
    package — auto-estimating a period for LOCoR shouldn't drag in a
    dependency that Notch/LOCoR themselves don't need.
    """
    from .periodicity import estimate_period

    times = [
        np.asarray(base.fulldata[inst]["time"], dtype=float) for inst in base.settings["inst_phot"]
    ]
    fluxes = [
        np.asarray(base.fulldata[inst]["flux"], dtype=float) for inst in base.settings["inst_phot"]
    ]
    time = np.concatenate(times)
    flux = np.concatenate(fluxes)
    order = np.argsort(time)
    try:
        period, _fap = estimate_period(time[order], flux[order], None, clip=False, plot=False)
        period = float(period)
        if not np.isfinite(period) or period <= 0:
            raise ValueError(f"estimate_period returned a non-positive period ({period}).")
    except Exception as exc:
        period = 1.0
        if not quiet:
            print(
                f"[transit-search] could not auto-estimate a rotation period for LOCoR "
                f"({exc}); falling back to period={period:.2f} d. Pass "
                "--flatten-window-length to set it explicitly."
            )
    return period


def _detrend_full_lightcurve(
    base,
    params_median: dict,
    flatten_method: str = "baseline",
    flatten_window_length: float | None = None,
    quiet: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Detrend every photometric instrument's full raw light curve and
    concatenate them into one time-sorted series.

    Parameters
    ----------
    flatten_method : str
        ``"baseline"`` (default) subtracts the fit's best-fit baseline
        model (:func:`allesfitter.computer.calculate_baseline` — whatever
        GP/spline/polynomial baseline the fit itself used). ``"notch"`` and
        ``"locor"`` instead run the Notch filter / LOCoR detrending
        (Rizzuto et al. 2017), ported from quicklook's ``notch_locor``
        module, directly on each instrument's raw flux — useful when the
        fit's own baseline model doesn't track the systematics well enough
        for a clean blind search (e.g. strong stellar rotation).
    flatten_window_length : float, optional
        For ``"notch"``: the sliding window width in days (default 0.5).
        For ``"locor"``: the stellar rotation period in days; if omitted,
        it's auto-estimated once (across all instruments) via a
        Lomb-Scargle periodogram. Ignored for ``"baseline"``.
    quiet : bool
        Silence the auto-period-estimation fallback message.

    Returns
    -------
    ``(time, flux_raw, flux_detrended, flux_err)`` — ``flux_raw`` is kept
    alongside the detrended flux so plots can show both.
    """
    if flatten_method not in _FLATTEN_METHODS:
        raise ValueError(
            f"flatten_method={flatten_method!r} is not recognized; expected one of "
            f"{_FLATTEN_METHODS}."
        )

    window_length = flatten_window_length
    if flatten_method == "notch" and window_length is None:
        window_length = _DEFAULT_NOTCH_WINDOW_LENGTH
    period = flatten_window_length
    if flatten_method == "locor" and period is None:
        period = _estimate_locor_period(base, quiet)

    times, fluxes_raw, fluxes_detrended, errs = [], [], [], []
    for inst in base.settings["inst_phot"]:
        full = base.fulldata[inst]
        time_full = np.asarray(full["time"], dtype=float)
        flux_full = np.asarray(full["flux"], dtype=float)
        flux_err_full = full["err_scales_flux"] * float(params_median[f"err_flux_{inst}"])

        if flatten_method == "baseline":
            baseline_full = calculate_baseline(params_median, inst, "flux", xx=time_full)
            flux_detrended = flux_full - baseline_full
        elif flatten_method == "notch":
            flux_detrended, _trend = notch_flatten(
                time_full, flux_full, window_length=window_length
            )
        else:  # "locor"
            flux_detrended, _trend = locor_flatten(time_full, flux_full, period=period)

        times.append(time_full)
        fluxes_raw.append(flux_full)
        fluxes_detrended.append(flux_detrended)
        errs.append(flux_err_full)

    time = np.concatenate(times)
    flux_raw = np.concatenate(fluxes_raw)
    flux_detrended = np.concatenate(fluxes_detrended)
    flux_err = np.concatenate(errs)
    order = np.argsort(time)
    return time[order], flux_raw[order], flux_detrended[order], flux_err[order]


def _tls_stellar_kwargs(base, params_median: dict) -> dict:
    """Best-effort ``R_star``/``M_star``/``u`` kwargs for TLS from
    ``params_star.csv`` and the fitted limb-darkening coefficients. Silently
    omits whatever isn't available; TLS falls back to its own defaults."""
    kwargs: dict = {}
    star = getattr(base, "params_star", None)
    if star:
        try:
            r_med = float(star["R_star_median"])
            m_med = float(star["M_star_median"])
            kwargs["R_star"] = r_med
            kwargs["R_star_min"] = r_med - 3 * float(star["R_star_lerr"])
            kwargs["R_star_max"] = r_med + 3 * float(star["R_star_uerr"])
            kwargs["M_star"] = m_med
            kwargs["M_star_min"] = m_med - 3 * float(star["M_star_lerr"])
            kwargs["M_star_max"] = m_med + 3 * float(star["M_star_uerr"])
        except (KeyError, TypeError, ValueError):
            pass

    inst_phot = base.settings["inst_phot"]
    if inst_phot:
        inst = inst_phot[0]
        q1 = params_median.get(f"host_ldc_q1_{inst}")
        q2 = params_median.get(f"host_ldc_q2_{inst}")
        if q1 is not None and q2 is not None:
            kwargs["u"] = q_to_u([float(q1), float(q2)], law="quad")
    return kwargs


#: Impact parameter above which TLS's default transit_template (which
#: assumes a moderate, roughly-central impact parameter — b ~ 0.32, from
#: its DEFAULT_INC/DEFAULT_A constants) measurably underestimates T14, per
#: empirical testing against injected transits (see git history for this
#: constant's introduction): ratio true/fit duration ~0.6-0.8 for b >~ 0.9,
#: recovering to ~0.85-1.1 once transit_template="grazing" is used instead.
_GRAZING_IMPACT_PARAMETER_THRESHOLD = 0.7


def _impact_parameter(rsuma: float, cosi: float, rr: float) -> float:
    """Circular-orbit impact parameter ``b = cos(i) / (R_star/a)`` — same
    formula as :func:`allesfitter.orbits.circular_transit_duration`'s
    internal computation. Returns NaN for missing/non-physical inputs."""
    try:
        rsuma = float(rsuma)
        cosi = float(cosi)
        rr = float(rr)
    except (TypeError, ValueError):
        return float("nan")
    if not (np.isfinite(rsuma) and np.isfinite(cosi) and np.isfinite(rr)) or rr < 0:
        return float("nan")
    rstar_over_a = rsuma / (1.0 + rr)
    if rstar_over_a <= 0:
        return float("nan")
    return cosi / rstar_over_a


def _companion_impact_parameter(params_median: dict, companion: str) -> float:
    return _impact_parameter(
        params_median.get(f"{companion}_rsuma"),
        params_median.get(f"{companion}_cosi"),
        params_median.get(f"{companion}_rr"),
    )


def _resolve_transit_template(
    transit_template: str,
    params_median: dict,
    companions: list[str],
    for_companion: str | None = None,
) -> tuple[str, float]:
    """Translate ``--transit-template`` into the literal TLS preset string
    (``"default"`` or ``"grazing"`` — the only two values that change TLS's
    assumed transit shape; ``"box"`` also exists but tests as
    indistinguishable from ``"default"`` in this TLS version).

    An explicit ``"default"``/``"grazing"`` (or any other literal string —
    passed through so TLS's own validation reports a bad value) is returned
    unchanged. ``"auto"`` instead picks based on impact parameter computed
    from the posterior: the companion's own for the per-companion recovery
    check (``for_companion``), or the most grazing of all known companions
    for the (geometry-unknown) blind search — a system with one near-edge-on
    orbit plausibly has others close to the same plane. Falls back to
    ``"default"`` when no companion has a resolvable impact parameter.

    Returns ``(resolved_template, impact_parameter)`` — ``impact_parameter``
    is NaN when not resolved (including for non-``"auto"`` input), purely
    for logging.
    """
    if transit_template != "auto":
        return transit_template, float("nan")

    if for_companion is not None:
        b = _companion_impact_parameter(params_median, for_companion)
    else:
        candidates = [_companion_impact_parameter(params_median, c) for c in companions]
        candidates = [b for b in candidates if np.isfinite(b)]
        b = max(candidates, key=abs) if candidates else float("nan")

    if not np.isfinite(b):
        return "default", b
    return ("grazing" if abs(b) >= _GRAZING_IMPACT_PARAMETER_THRESHOLD else "default"), b


def _apply_known_masks(
    time: np.ndarray, arrays: list[np.ndarray], windows: list[dict]
) -> tuple[np.ndarray, ...]:
    """Remove every window's transit from ``time`` and each array in
    ``arrays`` (row-aligned with ``time``), via one shared boolean mask so
    every array loses exactly the same rows."""
    keep = np.ones_like(time, dtype=bool)
    for window in windows:
        keep &= ~transit_mask(time, window["period"], window["duration"], window["epoch"])
    return (time[keep],) + tuple(np.asarray(a)[keep] for a in arrays)


def _run_tls_once(
    time: np.ndarray, flux: np.ndarray, flux_err: np.ndarray, tls_kwargs: dict, quiet: bool = True
) -> dict:
    """One-shot TLS scan (no iterative masking loop), with ``correct_duration``
    added the same way :func:`allesfitter.detection.transit_search.tls_search`
    does. Used to check whether a known companion is independently
    recoverable, before it gets masked out of the blind search."""
    kwargs = dict(tls_kwargs)
    kwargs.setdefault("show_progress_bar", not quiet)

    results = _run_tls_engine(time, flux, flux_err, kwargs, quiet=quiet)
    in_transit = np.where(results["model_folded_model"] < 1.0)[0]
    if len(in_transit) >= 2:
        results["correct_duration"] = results["period"] * (
            results["model_folded_phase"][in_transit[-1]]
            - results["model_folded_phase"][in_transit[0]]
        )
    else:
        results["correct_duration"] = results["duration"]
    return results


#: Minimum number of raw folded data points required within the currently
#: reported duration window (around phase 0.5) for that duration to be
#: trusted at face value — see _duration_is_trustworthy.
_MIN_POINTS_FOR_TRUSTED_DURATION = 20

#: Half-width, in phase, of the window searched for a fold-based duration
#: re-fit — and of the region excluded when estimating the out-of-transit
#: baseline for it. 0.1 (20% of the period) comfortably covers real
#: transit durations, which are almost always a small fraction of period.
_DURATION_REFIT_SEARCH_HALF_PHASE = 0.1

#: Significance (in units of the out-of-transit scatter) a bin has to drop
#: below baseline to count as in-transit for the fold-based duration re-fit.
_DURATION_REFIT_SIGMA = 3.0


def _duration_is_trustworthy(results: dict, min_points_in_window: int) -> bool:
    """Whether TLS's own duration has enough phase-folded evidence behind
    it to trust at face value.

    Counts raw data points in ``results['folded_phase']`` (the *full*
    dataset folded on the reported period/T0, not the sparse per-trial-
    period statistics TLS's own duration selection is vulnerable to under
    a gappy baseline — see git history for how this was diagnosed) that
    fall within the currently reported duration window around phase 0.5.
    Fewer than ``min_points_in_window`` is too thin an evidence base.
    """
    duration = results.get("correct_duration", results.get("duration"))
    period = results.get("period")
    folded_phase = results.get("folded_phase")
    if (
        duration is None
        or period is None
        or folded_phase is None
        or not np.isfinite(duration)
        or not np.isfinite(period)
        or period <= 0
        or duration <= 0
    ):
        return False
    half_phase = (duration / period) / 2.0
    folded_phase = np.asarray(folded_phase, dtype=float)
    n_in_window = int(np.sum(np.abs(folded_phase - 0.5) < half_phase))
    return n_in_window >= min_points_in_window


def _refit_duration_from_folded_data(
    folded_phase: np.ndarray,
    folded_y: np.ndarray,
    period: float,
    exclusion_half_phase: float = _DURATION_REFIT_SEARCH_HALF_PHASE,
    sigma: float = _DURATION_REFIT_SIGMA,
    n_bins: int = 200,
) -> float | None:
    """Re-measure transit duration directly from the phase-folded, fully
    aggregated light curve, independent of TLS's own per-trial-period
    duration selection.

    Bins the folded curve within ``exclusion_half_phase`` of phase 0.5,
    estimates the out-of-transit baseline (median + robust MAD scatter)
    from points *outside* that window, and grows a contiguous run of bins
    outward from phase 0.5 while each stays below
    ``baseline - sigma * scatter``. The seed bin is allowed to be a couple
    of bins off dead-center in case the exact center bin happens to have
    no data. Returns the resulting duration in days, or ``None`` if there
    isn't enough baseline data or no dip is found near phase 0.5 — never
    fabricates a number without evidence.
    """
    folded_phase = np.asarray(folded_phase, dtype=float)
    folded_y = np.asarray(folded_y, dtype=float)
    finite = np.isfinite(folded_phase) & np.isfinite(folded_y)
    folded_phase, folded_y = folded_phase[finite], folded_y[finite]
    if folded_phase.size == 0:
        return None

    out_of_window = np.abs(folded_phase - 0.5) > exclusion_half_phase
    if out_of_window.sum() < _MIN_POINTS_FOR_TRUSTED_DURATION:
        return None
    baseline = float(np.median(folded_y[out_of_window]))
    scatter = 1.4826 * float(np.median(np.abs(folded_y[out_of_window] - baseline)))
    if not np.isfinite(scatter) or scatter <= 0:
        scatter = float(np.std(folded_y[out_of_window]))
    if not np.isfinite(scatter) or scatter <= 0:
        return None

    lo, hi = 0.5 - exclusion_half_phase, 0.5 + exclusion_half_phase
    edges = np.linspace(lo, hi, n_bins + 1)
    bin_idx = np.digitize(folded_phase, edges) - 1
    in_window = (bin_idx >= 0) & (bin_idx < n_bins)

    bin_flux = np.full(n_bins, np.nan)
    bin_count = np.zeros(n_bins, dtype=int)
    for i in range(n_bins):
        m = in_window & (bin_idx == i)
        bin_count[i] = int(m.sum())
        if bin_count[i] > 0:
            bin_flux[i] = float(folded_y[m].mean())

    threshold = baseline - sigma * scatter
    below = np.where(bin_count > 0, bin_flux < threshold, False)

    center_bin = n_bins // 2
    seed = None
    for offset in (0, 1, -1, 2, -2):
        candidate = center_bin + offset
        if 0 <= candidate < n_bins and below[candidate]:
            seed = candidate
            break
    if seed is None:
        return None

    left = right = seed
    while left > 0 and below[left - 1]:
        left -= 1
    while right < n_bins - 1 and below[right + 1]:
        right += 1

    duration_phase = edges[right + 1] - edges[left]
    return duration_phase * period


#: A fold-refit duration this many times wider than TLS's own claimed
#: duration is preferred over it even when the claimed window itself has
#: enough nearby points to nominally pass _duration_is_trustworthy — a
#: narrow window can still have decent point density purely because
#: individual transit epochs are often densely sampled internally,
#: regardless of whether the claimed *width* is right. TLS's known failure
#: mode here is underestimation (never seen it go the other way), so a
#: well-supported wider re-measurement is trusted over a narrower one.
_DURATION_REFIT_WIDEN_FACTOR = 1.3


def _refine_duration_if_untrustworthy(
    results: dict, min_points_in_window: int = _MIN_POINTS_FOR_TRUSTED_DURATION
) -> dict:
    """Replace ``results['correct_duration']`` with a fold-based re-fit
    when TLS's own value either doesn't have enough phase-folded evidence
    behind it (:func:`_duration_is_trustworthy`) or is substantially
    narrower than what the fold-based re-fit independently finds; otherwise
    leave it untouched. Adds ``results['duration_refit_from_fold']`` (bool)
    either way, so callers/output can flag when this happened. Mutates and
    returns ``results``.
    """
    current = results.get("correct_duration", results.get("duration"))
    refit_days = _refit_duration_from_folded_data(
        results.get("folded_phase"), results.get("folded_y"), results.get("period")
    )
    has_refit = refit_days is not None and np.isfinite(refit_days) and refit_days > 0

    trustworthy = _duration_is_trustworthy(results, min_points_in_window)
    substantially_wider = (
        has_refit
        and current is not None
        and np.isfinite(current)
        and current > 0
        and refit_days > _DURATION_REFIT_WIDEN_FACTOR * current
    )

    if has_refit and (not trustworthy or substantially_wider):
        results["correct_duration"] = refit_days
        results["duration_refit_from_fold"] = True
    else:
        results["duration_refit_from_fold"] = False
    return results


def _check_known_companion_recovery(
    time: np.ndarray,
    flux_raw: np.ndarray,
    flux: np.ndarray,
    flux_err: np.ndarray,
    window: dict,
    other_windows: list[dict],
    tls_kwargs: dict,
    period_frac: float,
    sde_min: float,
    quiet: bool,
) -> tuple[dict, dict, np.ndarray, np.ndarray, np.ndarray]:
    """Isolate ``window``'s companion (mask every *other* known companion out
    of the light curve) and run a narrow, one-shot TLS scan bracketing its
    own period, to check whether it's independently recoverable.

    Returns ``(tls_results, summary_row, time_iso, flux_raw_iso, flux_iso)``.
    """
    time_iso, flux_raw_iso, flux_iso, err_iso = _apply_known_masks(
        time, [flux_raw, flux, flux_err], other_windows
    )

    period = window["period"]
    period_min = max(period * (1.0 - period_frac), 0.05)
    period_max = period * (1.0 + period_frac)
    kwargs = dict(tls_kwargs)
    # A narrow bracket can have too few natural grid points for TLS's own
    # period_grid() at its default oversampling_factor=3 — which then
    # silently falls back to an *unrestricted* search ignoring period_min/
    # period_max entirely (transitleastsquares.grid.period_grid's "too few
    # values" branch recurses with R_star=M_star=1 and no period bounds at
    # all). A much higher oversampling_factor keeps the bracket dense enough
    # to avoid that fallback in the typical case; the bracket_ignored check
    # below still catches it if it happens anyway.
    kwargs.setdefault("oversampling_factor", 20)
    kwargs["period_min"] = period_min
    kwargs["period_max"] = period_max

    results = _run_tls_once(time_iso, flux_iso, err_iso, kwargs, quiet=quiet)
    results = _refine_duration_if_untrustworthy(results)

    recovered_period = float(results["period"])
    recovered_epoch = float(results["T0"])
    bracket_ignored = not (period_min < recovered_period <= period_max)

    # Period is already constrained to within period_frac of the known value
    # by the search bracket above (when TLS actually honored it), so the
    # meaningful check is whether the recovered mid-transit time lines up
    # with the known one (mod the recovered period) — i.e. it's really this
    # companion's own transit, not an unrelated blip.
    phase_diff = (recovered_epoch - window["epoch"] + 0.5 * recovered_period) % recovered_period
    phase_diff = abs(phase_diff - 0.5 * recovered_period)
    epoch_match = bool(phase_diff <= window["duration"])
    recovered = bool(not bracket_ignored and epoch_match and results["SDE"] >= sde_min)

    row = {
        "companion": window["companion"],
        "known_period": period,
        "known_epoch": window["epoch"],
        "recovered_period": recovered_period,
        "recovered_epoch": recovered_epoch,
        "duration_hours": float(results.get("correct_duration", results["duration"])) * 24.0,
        "duration_refit_from_fold": bool(results.get("duration_refit_from_fold", False)),
        "SDE": float(results["SDE"]),
        "snr": float(results["snr"]),
        "epoch_match": epoch_match,
        "bracket_ignored": bracket_ignored,
        "recovered": recovered,
    }
    return results, row, time_iso, flux_raw_iso, flux_iso


def _plot_broken_series(fig, position, time, plot_fn, gap_threshold_days, ylabel, title):
    """Fill a (possibly multi-column) GridSpec ``position`` with a
    broken-x-axis time-series panel: :func:`allesfitter.plot_utils.
    broken_xaxis_subplots` splits it into one sub-Axes per contiguous
    segment of ``time`` (same convention ``general_output.afplot`` /
    ``show_initial_guess`` use for TESS-sector-sized gaps), ``plot_fn(ax)``
    draws the full series into each (matplotlib clips to that segment's
    xlim), and the shared ylabel/title/legend are kept on the first
    sub-Axes only, with the xlabel centered on the middle one.

    Returns the list of sub-Axes (length 1 when there's no large gap, or
    when ``gap_threshold_days`` is ``None``, which disables breaking).
    """
    if gap_threshold_days is None:
        sub_axes = [fig.add_subplot(position)]
    else:
        sub_axes = broken_xaxis_subplots(fig, position, time, gap_threshold_days=gap_threshold_days)
    for ax in sub_axes:
        xlim = ax.get_xlim()
        plot_fn(ax)
        ax.set_xlim(xlim)
    sub_axes[0].set_ylabel(ylabel)
    sub_axes[0].set_title(title)
    sub_axes[0].legend(loc="upper right", fontsize=8)
    sub_axes[len(sub_axes) // 2].set_xlabel("Time (BJD)")
    return sub_axes


def _passes_transit_consistency_checks(
    results: dict,
    min_distinct_transits: int,
    min_points_per_transit: int,
    consistency_sigma: float,
) -> tuple[bool, str]:
    """Reject a candidate whose periodicity is an artifact of large data
    gaps (TESS inter-sector/campaign gaps), using TLS's own per-transit
    statistics rather than an inferred period-multiple/harmonic heuristic.

    Two checks, in increasing order of strength:

    1. ``distinct_transit_count``: too few of the period's predicted transit
       epochs have *any* data at all (the rest fall entirely in a gap) — the
       period is under-constrained, not falsified, but a long period that
       only "works" because 1-2 sparse epochs happen to line up is exactly
       the classic gap-driven false positive.
    2. Per-transit consistency: among epochs with *enough* data that their
       precision (``transit_depths_uncertainties``) could have detected a
       transit at the candidate's own average depth, at least one shows no
       dip at all — this is a direct falsification from real data, not an
       inference, and is a stronger test than (1) alone.

    Returns ``(passes, reason)`` — ``reason`` is a human-readable rejection
    cause, empty when ``passes`` is True.
    """
    distinct = results.get("distinct_transit_count")
    if distinct is not None and np.isfinite(distinct) and distinct < min_distinct_transits:
        return (
            False,
            f"only {int(distinct)} distinct transit(s) covered by data "
            f"(need >= {min_distinct_transits})",
        )

    depths = np.asarray(results.get("transit_depths", []), dtype=float)
    depth_errs = np.asarray(results.get("transit_depths_uncertainties", []), dtype=float)
    counts = np.asarray(results.get("per_transit_count", []), dtype=float)
    mean_depth_frac = 1.0 - float(results["depth"])  # fractional dimming, > 0 for a real dip

    for i in range(len(depths)):
        if i >= len(counts) or counts[i] < min_points_per_transit:
            continue  # too little data at this epoch to judge it either way
        err_i = depth_errs[i] if i < len(depth_errs) else np.nan
        if not np.isfinite(err_i) or err_i <= 0:
            continue
        if mean_depth_frac / err_i < consistency_sigma:
            continue  # this epoch's precision couldn't have caught the transit anyway
        depth_i = 1.0 - depths[i]  # measured fractional dimming at this epoch
        if depth_i < consistency_sigma * err_i:
            return (
                False,
                f"transit epoch {i} has {int(counts[i])} points, precise enough to detect "
                f"the {mean_depth_frac * 1e6:.0f} ppm depth at >= {consistency_sigma:g}sigma, "
                "but shows no dip",
            )

    return True, ""


def _passes_min_depth(results: dict, min_depth_ppt: float) -> tuple[bool, str]:
    """Reject a candidate whose transit depth is too shallow to trust.

    SDE/SNR alone can pass on noise for long periods with few transits (a
    27 d, 0.46 ppt "candidate" from a handful of marginal dips is a much
    weaker claim than the same statistics on a deep, unambiguous dip) — this
    is a direct, physically-motivated floor independent of those.
    """
    depth_ppt = (1.0 - float(results["depth"])) * 1e3
    if depth_ppt < min_depth_ppt:
        return False, f"depth {depth_ppt:.3f} ppt < min_depth_ppt={min_depth_ppt:g}"
    return True, ""


def _harmonic_periods(period: float, n_harmonics: int = 4) -> list[float]:
    """Integer-ratio harmonics of ``period`` (``P/2, P/3, ..., 2P, 3P, ...``)."""
    harmonics = set()
    for i in range(2, n_harmonics + 2):
        harmonics.add(period / i)
        harmonics.add(period * i)
    return sorted(harmonics)


def _plot_candidate(
    results: dict,
    known_windows: list[dict],
    time_raw: np.ndarray,
    flux_raw: np.ndarray,
    time_flat: np.ndarray,
    flux_flat: np.ndarray,
    gap_threshold_days: float = 5.0,
) -> plt.Figure:
    """One figure per candidate, stacked top to bottom: the full raw light
    curve with the best-fit baseline model overlaid, the full flattened
    (detrended) light curve with this candidate's TLS model overlaid, the
    TLS periodogram (harmonics + known-planet reference lines), and
    phase-folded data + TLS model.

    The raw and flattened panels break their x-axis at any gap in
    ``time_raw``/``time_flat`` wider than ``gap_threshold_days`` (e.g.
    between TESS sectors), the same convention ``show-initial-guess`` uses.
    """
    fig = plt.figure(figsize=(12, 14), tight_layout=True)
    gs = fig.add_gridspec(4, 2, height_ratios=[1, 1, 1, 1.2])

    # flux_raw - flux_flat is exactly the (noiseless, smooth) best-fit
    # baseline calculate_baseline predicted; +1 puts it back on the raw
    # flux's ~1 level instead of the ~0 level it sits at as a subtracted term.
    baseline_on_time_raw = flux_raw - flux_flat + 1.0

    def _draw_raw(ax):
        ax.plot(time_raw, flux_raw, ".", color="silver", ms=2, rasterized=True, label="Raw data")
        ax.plot(
            time_raw,
            baseline_on_time_raw,
            "-",
            color="C1",
            lw=1,
            zorder=2,
            label="Best-fit baseline",
        )

    _plot_broken_series(
        fig, gs[0, :], time_raw, _draw_raw, gap_threshold_days, "Raw flux", "Raw light curve"
    )

    model_on_time_flat = np.interp(
        time_flat, results["model_lightcurve_time"], results["model_lightcurve_model"]
    )

    def _draw_flat(ax):
        ax.plot(
            time_flat,
            flux_flat,
            ".",
            color="silver",
            ms=2,
            rasterized=True,
            zorder=1,
            label="Flattened data",
        )
        ax.plot(time_flat, model_on_time_flat, "r-", lw=1, zorder=2, label="TLS model")

    _plot_broken_series(
        fig,
        gs[1, :],
        time_flat,
        _draw_flat,
        gap_threshold_days,
        "Flattened flux",
        "Flattened (detrended) light curve",
    )

    ax0 = fig.add_subplot(gs[2, :])
    ax0.plot(results["periods"], results["power"], "k-", lw=0.8, rasterized=True)
    ax0.axvline(results["period"], color="C3", lw=1.5, label=f"P = {results['period']:.4f} d")
    periods = np.asarray(results["periods"], dtype=float)
    for harmonic in _harmonic_periods(results["period"]):
        if periods.min() <= harmonic <= periods.max():
            ax0.axvline(harmonic, color="C3", lw=0.8, ls=":", alpha=0.5)
    for window in known_windows:
        ax0.axvline(
            window["period"],
            color="C0",
            lw=1.2,
            ls="--",
            alpha=0.8,
            label=f"{window['companion']} (known) P = {window['period']:.4f} d",
        )
    ax0.set(
        xlabel="Period (days)",
        ylabel="SDE",
        xscale="log",
        title=f"TLS periodogram (SDE={results['SDE']:.2f}, SNR={results['snr']:.2f})",
    )
    ax0.legend(loc="upper right", fontsize=8)

    dur = results.get("correct_duration", results["duration"])

    ax1 = fig.add_subplot(gs[3, 0])
    bintime, binflux, _binflux_err, _ = rebin_err(
        results["folded_phase"],
        results["folded_y"],
        dt=0.001 * results["period"],
        ferr_type="medsig",
        ferr_style="sem",
    )
    ax1.plot(
        results["folded_phase"],
        results["folded_y"],
        ".",
        color="silver",
        ms=2,
        rasterized=True,
        zorder=-1,
    )
    ax1.plot(bintime, binflux, "b.", rasterized=True)
    ax1.plot(results["model_folded_phase"], results["model_folded_model"], "r-", lw=2)
    ax1.set(xlabel="Phase", ylabel="Relative flux", title="Phase-folded")

    ax2 = fig.add_subplot(gs[3, 1])
    x_hours = (results["folded_phase"] - 0.5) * results["period"] * 24
    bintime, binflux, _binflux_err, _ = rebin_err(
        x_hours,
        results["folded_y"],
        dt=0.001 * results["period"] * 24,
        ferr_type="medsig",
        ferr_style="sem",
    )
    ax2.plot(x_hours, results["folded_y"], ".", color="silver", ms=2, rasterized=True, zorder=-1)
    ax2.plot(bintime, binflux, "b.", rasterized=True)
    ax2.plot(
        (results["model_folded_phase"] - 0.5) * results["period"] * 24,
        results["model_folded_model"],
        "r-",
        lw=2,
    )
    ax2.set(
        xlim=[-3 * dur * 24, 3 * dur * 24],
        xlabel="Time from mid-transit (h)",
        ylabel="Relative flux",
        title="Zoom near transit",
    )

    fig.suptitle(
        f"P={results['period']:.5f} d   T0={results['T0']:.5f}   "
        f"depth={1e3 * (1.0 - results['depth']):.3f} ppt   duration={dur * 24:.2f} h"
    )
    return fig


def transit_search(
    results_dir: str,
    *,
    sde_min: float = 8.0,
    period_min: float | None = None,
    period_max: float | None = None,
    n_transits_min: int = 3,
    transit_template: str = "auto",
    mask_width_factor: float = 1.5,
    outdir: str | None = None,
    file_extension: str = ".pdf",
    max_candidates: int = 20,
    mission: str = "tess",
    check_known_recovery: bool = True,
    recovery_period_frac: float = 0.05,
    min_distinct_transits: int = 3,
    min_points_per_transit: int = 5,
    consistency_sigma: float = 3.0,
    min_depth_ppt: float = 1.0,
    flatten_method: str = "baseline",
    flatten_window_length: float | None = None,
    quiet: bool = False,
) -> dict:
    """Blind TLS search for un-modeled transits in a fitted target.

    Parameters
    ----------
    results_dir : str
        Path to an ``mcmc_results`` or ``ns_results`` directory produced by
        ``allesfitter mcmc-fit``/``ns-fit``.
    sde_min : float
        The iterative search keeps masking and re-running TLS as long as the
        found signal's SDE stays at or above this value; also the threshold
        a known companion's own SDE must clear to count as "recovered".
    period_min, period_max : float, optional
        Period search range in days for the blind search; left to TLS's own
        defaults when omitted. Does not affect the known-companion recovery
        check, which always searches its own narrow, per-companion bracket.
    n_transits_min : int
        Passed straight through to TLS's own period_grid(): periods needing
        more than this many transits than the data's time span can support
        are never included in the blind search's period grid at all. Unlike
        min_distinct_transits/consistency_sigma below (which reject a
        candidate after it's found), this shrinks the search itself — fewer
        wasted trial periods, and large-gap false positives that would only
        have 1-2 transits can't even be proposed in the first place.
    transit_template : str
        TLS's assumed transit shape: ``"default"`` (moderate impact
        parameter, b~0.32) or ``"grazing"`` (b=0.99). TLS fits duration by
        picking the best-matching discrete template for the trial period,
        not a free fit — with a low-b (near-central) template applied to a
        genuinely high-b (grazing, V-shaped) transit, the fitted duration
        comes out ~20-40% short regardless of SNR (confirmed by injection
        tests). ``"auto"`` (default) picks per companion from its own
        posterior impact parameter for the recovery check, or from the most
        grazing known companion for the (geometry-unknown) blind search;
        falls back to ``"default"`` when no companion has a resolvable
        impact parameter. Pass ``"default"``/``"grazing"`` directly to
        override for every companion/candidate uniformly.
    mask_width_factor : float
        Known-companion transits are masked out to this many times their
        total (T14) duration on either side of mid-transit.
    outdir : str, optional
        Where to write figures/h5 files/summaries. Defaults to
        ``<target_output_directory>/transit_search_results``.
    max_candidates : int
        Safety cap on the number of candidates kept (and written to disk).
    mission : str
        Used for the h5 files' BJD-offset convention (``tess``, ``k2``, or
        ``kepler``); does not affect the search itself.
    check_known_recovery : bool
        Before masking anything, isolate each known companion (mask every
        *other* known companion) and run a narrow TLS scan bracketing its
        own period, to check it is independently recoverable in this
        detrended light curve. Set False to skip and go straight to the
        blind search.
    recovery_period_frac : float
        Half-width of the recovery check's period-search bracket, as a
        fraction of each known companion's own period (default 0.05 = ±5%).
    min_distinct_transits : int
        Reject a blind-search candidate if fewer than this many of its
        predicted transit epochs have any data at all (the rest fall
        entirely in gaps between TESS sectors/campaigns) — the classic
        false-positive mode with large gaps, where a long period only
        "works" because 1-2 sparse epochs happen to line up.
    min_points_per_transit : int
        A predicted transit epoch counts as "well-covered" for the
        per-transit consistency check below only once it has at least this
        many in-transit data points.
    consistency_sigma : float
        Per-transit consistency check: among well-covered epochs whose
        precision (``transit_depths_uncertainties``) is good enough to have
        detected the candidate's own average depth at ``consistency_sigma``,
        reject the candidate if any shows no dip at all (depth consistent
        with zero at less than ``consistency_sigma``) — i.e. a real,
        well-sampled non-detection actively rules the period out, which is a
        stronger and more direct test than ``min_distinct_transits`` (which
        only asks "was there *any* data", not "did the data agree").
    min_depth_ppt : float
        Reject a blind-search candidate whose transit depth is below this
        many parts per thousand (1 ppt = 1000 ppm). SDE/SNR alone can clear
        the bar on noise for long periods with only a couple of marginal
        transits; this is a direct floor on the physical depth itself.
    flatten_method : str
        How to detrend the raw light curve before the blind search:
        ``"baseline"`` (default) subtracts the fit's own best-fit baseline
        model. ``"notch"`` and ``"locor"`` instead apply the Notch filter /
        LOCoR detrending (Rizzuto et al. 2017) directly to the raw flux —
        an alternative worth trying when the fit's baseline model doesn't
        track the systematics well enough (e.g. strong stellar rotation
        that the fit's own GP/spline wasn't tuned for).
    flatten_window_length : float, optional
        For ``flatten_method="notch"``: the sliding window width in days
        (default 0.5). For ``"locor"``: the stellar rotation period in
        days; if omitted, it's auto-estimated via a Lomb-Scargle
        periodogram. Ignored for ``"baseline"``.

    Returns
    -------
    dict
        ``{"candidates": [...], "known_recovery": [...]}`` — one summary dict
        per blind-search candidate (period, epoch, duration, depth, SDE,
        SNR, figure/h5 paths) and one per known companion checked (period,
        epoch, recovered period/epoch, SDE, SNR, whether it was recovered,
        and its figure path).
    """
    sampler, datadir = _resolve_sampler(results_dir)
    params_median = _load_posterior_params(datadir, sampler)
    base = config.BASEMENT

    companions = base.settings["companions_phot"]
    known_windows = _known_companion_windows(params_median, companions, mask_width_factor)

    time, flux_raw, flux, flux_err = _detrend_full_lightcurve(
        base,
        params_median,
        flatten_method=flatten_method,
        flatten_window_length=flatten_window_length,
        quiet=quiet,
    )
    full_lightcurve = {"time": time.copy(), "flux": flux.copy(), "flux_err": flux_err.copy()}

    tls_kwargs = _tls_stellar_kwargs(base, params_median)
    # Same convention as general_output.afplot / show-initial-guess: skip
    # typical TESS intra-sector gaps, break only at inter-sector scale.
    gap_threshold_days = base.settings.get("xaxis_break_gap_days", 5.0)

    if outdir is None:
        outdir = os.path.join(str(target_output_directory(datadir)), "transit_search_results")
    os.makedirs(outdir, exist_ok=True)

    known_recovery = []
    if check_known_recovery and known_windows:
        for i, window in enumerate(known_windows):
            other_windows = known_windows[:i] + known_windows[i + 1 :]
            resolved_template, template_b = _resolve_transit_template(
                transit_template, params_median, companions, for_companion=window["companion"]
            )
            companion_tls_kwargs = dict(tls_kwargs)
            companion_tls_kwargs["transit_template"] = resolved_template
            if not quiet and transit_template == "auto":
                print(
                    f"[transit-search] known companion {window['companion']}: "
                    f"auto-selected transit_template={resolved_template}"
                    + (f" (b={template_b:.2f})" if np.isfinite(template_b) else "")
                )

            results, row, time_iso, flux_raw_iso, flux_iso = _check_known_companion_recovery(
                time,
                flux_raw,
                flux,
                flux_err,
                window,
                other_windows,
                companion_tls_kwargs,
                recovery_period_frac,
                sde_min,
                quiet,
            )

            fig = _plot_candidate(
                results,
                other_windows,
                time_iso,
                flux_raw_iso,
                time_iso,
                flux_iso,
                gap_threshold_days,
            )
            fig_path = os.path.join(outdir, f"known_{window['companion']}_recovery{file_extension}")
            fig.savefig(fig_path, bbox_inches="tight")
            plt.close(fig)
            row["figure"] = fig_path
            known_recovery.append(row)

            if not quiet:
                if row["bracket_ignored"]:
                    status = (
                        "INCONCLUSIVE (TLS ignored the narrow period bracket; try a "
                        "larger --recovery-period-frac)"
                    )
                else:
                    status = "RECOVERED" if row["recovered"] else "NOT recovered"
                duration_note = (
                    " (duration re-fit from folded data: TLS's own estimate had too little "
                    "phase-folded support to trust)"
                    if row["duration_refit_from_fold"]
                    else ""
                )
                print(
                    f"[transit-search] known companion {row['companion']}: "
                    f"P={row['known_period']:.5f} d -> recovered P={row['recovered_period']:.5f} d, "
                    f"duration={row['duration_hours']:.2f} h{duration_note}, "
                    f"SDE={row['SDE']:.2f}, SNR={row['snr']:.2f} => {status} -> {fig_path}"
                )

        _write_known_recovery_csv(
            os.path.join(outdir, "known_planets_recovery.csv"), known_recovery
        )

    time_search, flux_search, err_search = time, flux, flux_err
    for window in known_windows:
        time_search, flux_search, err_search = mask(
            time_search,
            flux_search,
            err_search,
            window["period"],
            window["duration"],
            window["epoch"],
        )

    if period_min is not None:
        tls_kwargs["period_min"] = period_min
    if period_max is not None:
        tls_kwargs["period_max"] = period_max
    tls_kwargs["n_transits_min"] = n_transits_min
    resolved_template, template_b = _resolve_transit_template(
        transit_template, params_median, companions
    )
    tls_kwargs["transit_template"] = resolved_template
    if not quiet and transit_template == "auto":
        print(
            f"[transit-search] blind search: auto-selected transit_template={resolved_template}"
            + (f" (max known-companion b={template_b:.2f})" if np.isfinite(template_b) else "")
        )

    results_all = tls_search(
        time_search,
        flux_search,
        err_search,
        plot=False,
        SDE_threshold=sde_min,
        SNR_threshold=-np.inf,
        FAP_threshold=np.inf,
        show_progress_bar=not quiet,
        quiet=quiet,
        **tls_kwargs,
    )

    kept_results_all = []
    for results in results_all:
        ok, reason = _passes_min_depth(results, min_depth_ppt)
        if ok:
            ok, reason = _passes_transit_consistency_checks(
                results, min_distinct_transits, min_points_per_transit, consistency_sigma
            )
        if ok:
            kept_results_all.append(_refine_duration_if_untrustworthy(results))
        elif not quiet:
            print(
                f"[transit-search] rejected P={results['period']:.5f} d "
                f"(SDE={results['SDE']:.2f}): {reason}"
            )
    results_all = kept_results_all[:max_candidates]

    summary = []
    for i, results in enumerate(results_all, start=1):
        fig = _plot_candidate(
            results, known_windows, time, flux_raw, time, flux, gap_threshold_days
        )
        fig_path = os.path.join(outdir, f"candidate_{i}{file_extension}")
        fig.savefig(fig_path, bbox_inches="tight")
        plt.close(fig)

        h5_path = os.path.join(outdir, f"candidate_{i}_tls.h5")
        write_h5_tls_result(h5_path, results, lightcurve=full_lightcurve, mission=mission)

        summary.append(
            {
                "candidate": i,
                "period": float(results["period"]),
                "epoch": float(results["T0"]),
                "duration_hours": float(results.get("correct_duration", results["duration"]))
                * 24.0,
                "duration_refit_from_fold": bool(results.get("duration_refit_from_fold", False)),
                "depth_ppm": (1.0 - float(results["depth"])) * 1e6,
                "SDE": float(results["SDE"]),
                "snr": float(results["snr"]),
                "figure": fig_path,
                "h5": h5_path,
            }
        )
        if not quiet:
            duration_note = (
                " (duration re-fit from folded data)"
                if summary[-1]["duration_refit_from_fold"]
                else ""
            )
            print(
                f"[transit-search] candidate {i}: P={results['period']:.5f} d  "
                f"T0={results['T0']:.5f}  SDE={results['SDE']:.2f}  SNR={results['snr']:.2f}"
                f"{duration_note}  -> {fig_path}, {h5_path}"
            )

    _write_summary_csv(os.path.join(outdir, "candidates_summary.csv"), summary)

    if not quiet:
        if summary:
            print(f"[transit-search] {len(summary)} candidate(s) written to {outdir}")
        else:
            print(f"[transit-search] no candidates found above SDE={sde_min}")

    return {"candidates": summary, "known_recovery": known_recovery}


def _write_known_recovery_csv(path: str, known_recovery: list[dict]) -> None:
    fieldnames = [
        "companion",
        "known_period",
        "known_epoch",
        "recovered_period",
        "recovered_epoch",
        "duration_hours",
        "duration_refit_from_fold",
        "SDE",
        "snr",
        "epoch_match",
        "bracket_ignored",
        "recovered",
        "figure",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in known_recovery:
            writer.writerow(row)


def _write_summary_csv(path: str, summary: list[dict]) -> None:
    fieldnames = [
        "candidate",
        "period",
        "epoch",
        "duration_hours",
        "duration_refit_from_fold",
        "depth_ppm",
        "SDE",
        "snr",
        "figure",
        "h5",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary:
            writer.writerow(row)
