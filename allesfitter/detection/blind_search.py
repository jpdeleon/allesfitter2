"""Blind ``transitleastsquares`` search on top of a completed allesfitter fit.

``transit_search()`` takes an MCMC or NS ``*_results`` directory and:

1. Pulls the posterior-median parameters (best-fit baseline hyperparameters
   and known-companion ephemerides) via the same
   ``get_params_from_samples`` path ``mcmc_output``/``ns_output`` use.
2. Reconstructs each photometric instrument's full raw light curve
   (``Basement.fulldata``, i.e. before any ``fast_fit`` windowing) and
   subtracts the best-fit baseline model (:func:`allesfitter.computer.
   calculate_baseline`, evaluated on that full time grid) from it.
3. Masks out every known companion in ``params.csv`` (period/epoch from the
   posterior, duration from the circular transit-chord equation).
4. Runs :func:`allesfitter.detection.transit_search.tls_search` iteratively
   — it already masks each new detection and loops — stopping once SDE
   drops below ``sde_threshold``.
5. Writes a per-candidate figure (raw light curve, flattened/detrended light
   curve with the candidate's model overlaid, TLS periodogram with harmonics
   and known-planet reference lines, and phase-folded data + TLS model) and
   a quicklook-style ``.h5`` file (readable by ``prepare_allesfit.py --h5``)
   for every candidate found above threshold.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .. import config
from ..computer import calculate_baseline
from ..exoworlds_rdx.lightcurves.lightcurve_tools import rebin_err
from ..general_output import get_params_from_samples
from ..lightcurves import translate_limb_darkening_from_q_to_u as q_to_u
from ..results import target_output_directory
from ..tls_h5 import write_h5_tls_result
from ..validation.prior_checks import transit_duration_days
from .transit_search import mask, tls_search

#: Maps a results directory's basename to the sampler that produced it.
_SAMPLER_BY_DIRNAME = {"mcmc_results": "mcmc", "ns_results": "ns"}


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


def _detrend_full_lightcurve(
    base, params_median: dict
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Detrend every photometric instrument's full raw light curve with the
    best-fit baseline and concatenate them into one time-sorted series.

    Returns ``(time, flux_raw, flux_detrended, flux_err)`` — ``flux_raw`` is
    kept alongside the detrended flux so plots can show both."""
    times, fluxes_raw, fluxes_detrended, errs = [], [], [], []
    for inst in base.settings["inst_phot"]:
        full = base.fulldata[inst]
        time_full = np.asarray(full["time"], dtype=float)
        flux_full = np.asarray(full["flux"], dtype=float)
        flux_err_full = full["err_scales_flux"] * float(params_median[f"err_flux_{inst}"])

        baseline_full = calculate_baseline(params_median, inst, "flux", xx=time_full)
        flux_detrended = flux_full - baseline_full

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
) -> plt.Figure:
    """One figure per candidate, stacked top to bottom: the full raw light
    curve with the best-fit baseline model overlaid, the full flattened
    (detrended) light curve with this candidate's TLS model overlaid, the
    TLS periodogram (harmonics + known-planet reference lines), and
    phase-folded data + TLS model."""
    fig = plt.figure(figsize=(12, 14), tight_layout=True)
    gs = fig.add_gridspec(4, 2, height_ratios=[1, 1, 1, 1.2])

    ax_raw = fig.add_subplot(gs[0, :])
    ax_raw.plot(time_raw, flux_raw, ".", color="silver", ms=2, rasterized=True, label="Raw data")
    # flux_raw - flux_flat is exactly the (noiseless, smooth) best-fit
    # baseline calculate_baseline predicted; +1 puts it back on the raw
    # flux's ~1 level instead of the ~0 level it sits at as a subtracted term.
    baseline_on_time_raw = flux_raw - flux_flat + 1.0
    ax_raw.plot(
        time_raw, baseline_on_time_raw, "-", color="C1", lw=1, zorder=2, label="Best-fit baseline"
    )
    ax_raw.set(xlabel="Time (BJD)", ylabel="Raw flux", title="Raw light curve")
    ax_raw.legend(loc="upper right", fontsize=8)

    ax_flat = fig.add_subplot(gs[1, :], sharex=ax_raw)
    ax_flat.plot(
        time_flat,
        flux_flat,
        ".",
        color="silver",
        ms=2,
        rasterized=True,
        zorder=1,
        label="Flattened data",
    )
    model_on_time_flat = np.interp(
        time_flat, results["model_lightcurve_time"], results["model_lightcurve_model"]
    )
    ax_flat.plot(time_flat, model_on_time_flat, "r-", lw=1, zorder=2, label="TLS model")
    ax_flat.set(
        xlabel="Time (BJD)", ylabel="Flattened flux", title="Flattened (detrended) light curve"
    )
    ax_flat.legend(loc="upper right", fontsize=8)

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
    sde_threshold: float = 5.0,
    period_min: float | None = None,
    period_max: float | None = None,
    mask_width_factor: float = 1.5,
    outdir: str | None = None,
    file_extension: str = ".pdf",
    max_candidates: int = 20,
    mission: str = "tess",
    quiet: bool = False,
) -> list[dict]:
    """Blind TLS search for un-modeled transits in a fitted target.

    Parameters
    ----------
    results_dir : str
        Path to an ``mcmc_results`` or ``ns_results`` directory produced by
        ``allesfitter mcmc-fit``/``ns-fit``.
    sde_threshold : float
        The iterative search keeps masking and re-running TLS as long as the
        found signal's SDE stays at or above this value.
    period_min, period_max : float, optional
        Period search range in days; left to TLS's own defaults when omitted.
    mask_width_factor : float
        Known-companion transits are masked out to this many times their
        total (T14) duration on either side of mid-transit.
    outdir : str, optional
        Where to write figures/h5 files/summary. Defaults to
        ``<target_output_directory>/transit_search_results``.
    max_candidates : int
        Safety cap on the number of candidates kept (and written to disk).
    mission : str
        Used for the h5 files' BJD-offset convention (``tess``, ``k2``, or
        ``kepler``); does not affect the search itself.

    Returns
    -------
    list of dict
        One summary dict per candidate (period, epoch, duration, depth, SDE,
        SNR, and the figure/h5 paths written for it).
    """
    sampler, datadir = _resolve_sampler(results_dir)
    params_median = _load_posterior_params(datadir, sampler)
    base = config.BASEMENT

    companions = base.settings["companions_phot"]
    known_windows = _known_companion_windows(params_median, companions, mask_width_factor)

    time, flux_raw, flux, flux_err = _detrend_full_lightcurve(base, params_median)
    full_lightcurve = {"time": time.copy(), "flux": flux.copy(), "flux_err": flux_err.copy()}

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

    tls_kwargs = _tls_stellar_kwargs(base, params_median)
    if period_min is not None:
        tls_kwargs["period_min"] = period_min
    if period_max is not None:
        tls_kwargs["period_max"] = period_max

    results_all = tls_search(
        time_search,
        flux_search,
        err_search,
        plot=False,
        SDE_threshold=sde_threshold,
        SNR_threshold=-np.inf,
        FAP_threshold=np.inf,
        show_progress_bar=not quiet,
        quiet=quiet,
        **tls_kwargs,
    )
    results_all = results_all[:max_candidates]

    if outdir is None:
        outdir = os.path.join(str(target_output_directory(datadir)), "transit_search_results")
    os.makedirs(outdir, exist_ok=True)

    summary = []
    for i, results in enumerate(results_all, start=1):
        fig = _plot_candidate(results, known_windows, time, flux_raw, time, flux)
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
                "depth_ppm": (1.0 - float(results["depth"])) * 1e6,
                "SDE": float(results["SDE"]),
                "snr": float(results["snr"]),
                "figure": fig_path,
                "h5": h5_path,
            }
        )
        if not quiet:
            print(
                f"[transit-search] candidate {i}: P={results['period']:.5f} d  "
                f"T0={results['T0']:.5f}  SDE={results['SDE']:.2f}  SNR={results['snr']:.2f}  "
                f"-> {fig_path}, {h5_path}"
            )

    _write_summary_csv(os.path.join(outdir, "candidates_summary.csv"), summary)

    if not quiet:
        if summary:
            print(f"[transit-search] {len(summary)} candidate(s) written to {outdir}")
        else:
            print(f"[transit-search] no candidates found above SDE={sde_threshold}")

    return summary


def _write_summary_csv(path: str, summary: list[dict]) -> None:
    fieldnames = [
        "candidate",
        "period",
        "epoch",
        "duration_hours",
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
