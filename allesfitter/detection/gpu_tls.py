"""GPU-accelerated Transit Least Squares (GTLS) support for the blind
transit search.

When a CUDA GPU and the optional ``gputls`` package (`jpdeleon/GTLS
<https://github.com/jpdeleon/GTLS>`_) are both available, :func:`run_tls`
runs the periodogram search on GPU instead of CPU ``transitleastsquares``,
which is dramatically faster for wide period searches. Any failure along
that path -- no GPU, ``gputls`` not installed, GTLS itself raising, or the
result reassembly below raising -- falls back to plain CPU TLS, so callers
never need to special-case the GPU path.

GTLS's own result (``gputls.results.gtlsResult``) only carries the found
period/T0/duration/depth/SDE and the periodogram itself; it skips the
per-transit statistics (``transit_depths``, ``folded_phase``,
``model_lightcurve_*``, ...) that the rest of this package's vetting logic
(``blind_search._passes_transit_consistency_checks``,
``blind_search._duration_is_trustworthy``, the candidate figures) reads.
:func:`_assemble_tls_style_result` recomputes those directly from
``transitleastsquares.stats``, fed with GTLS's found parameters -- the same
functions CPU TLS itself uses to assemble its own final result (see
``transitleastsquares.main.transitleastsquares.power``) -- so a GTLS-backed
result is a like-for-like ``transitleastsquares`` results dict, not a
partial approximation.
"""

from __future__ import annotations

import contextlib
import os
import warnings

import numpy as np


def _gpu_device_count() -> int:
    """Return the number of CUDA devices visible to CuPy."""
    import cupy

    return cupy.cuda.runtime.getDeviceCount()


def get_gpu_tls_engine():
    """Return the GTLS ``gtls`` class when a CUDA GPU and the optional
    ``gputls`` package are both available, else ``None``."""
    try:
        device_count = _gpu_device_count()
    except Exception:
        return None

    if device_count < 1:
        return None

    try:
        from gputls import gtls
    except Exception as exc:
        warnings.warn(
            f"CUDA GPU detected but gputls could not be loaded ({exc}); using CPU TLS",
            stacklevel=2,
        )
        return None

    return gtls


def _assemble_tls_style_result(model, gtls_result) -> dict:
    """Turn a GTLS ``(model, gtlsResult)`` pair into a ``transitleastsquares``-
    style results dict, computing every field the CPU TLS path provides but
    GTLS itself doesn't, from ``transitleastsquares.stats`` fed with GTLS's
    found period/T0/duration/depth (mirrors
    ``transitleastsquares.main.transitleastsquares.power``'s own result
    assembly).
    """
    from transitleastsquares import tls_constants
    from transitleastsquares.core import fold
    from transitleastsquares.helpers import transit_mask
    from transitleastsquares.stats import (
        FAP,
        all_transit_times,
        calculate_fill_factor,
        calculate_stretch,
        calculate_transit_duration_in_days,
        count_stats,
        intransit_stats,
        model_lightcurve,
        period_uncertainty,
        rp_rs_from_depth,
        snr_stats,
    )
    from transitleastsquares.transit import fractional_transit

    t = np.asarray(model.t, dtype=float)
    y = np.asarray(model.y, dtype=float)
    dy = np.asarray(model.dy, dtype=float)

    period = float(gtls_result.period)
    T0 = float(gtls_result.T0)
    raw_duration = float(gtls_result.rawDuration)
    # GTLS's `depth` is the fractional transit dimming (near 0, e.g. 0.01 for
    # a 1% dip); TLS's own `depth` field convention is the in-transit
    # relative flux level (near 1, i.e. `1 - dimming`).
    dimming = float(gtls_result.depth)
    if not (np.isfinite(period) and period > 0 and np.isfinite(T0)):
        raise ValueError(f"GTLS returned a non-physical period/T0 ({period}, {T0})")

    transit_times = all_transit_times(T0, t, period)
    transit_duration_in_days = calculate_transit_duration_in_days(
        t, period, transit_times, raw_duration
    )

    # Folded *data* (not the model) -- same as TLS's own `power()` assembly.
    phases = fold(t, period, T0=T0 + period / 2)
    sort_index = np.argsort(phases)
    folded_phase = phases[sort_index]
    folded_y = y[sort_index]
    folded_dy = dy[sort_index]

    maxwidth_in_samples = int(np.max(np.asarray(gtls_result.rawDurations)) * len(t))
    if maxwidth_in_samples % 2 != 0:
        maxwidth_in_samples += 1
    fill_factor = calculate_fill_factor(t)
    fill_half = 1 - ((1 - fill_factor) * 0.5)
    stretch = calculate_stretch(t, period, transit_times)
    internal_samples = (
        int(len(y) / len(transit_times))
    ) * tls_constants.OVERSAMPLE_MODEL_LIGHT_CURVE

    model_kwargs = dict(
        per=model.per,
        rp=model.rp,
        a=model.a,
        inc=model.inc,
        ecc=model.ecc,
        w=model.w,
        u=model.u,
        limb_dark=model.limb_dark,
    )

    # Model phase, shifted by half a cadence so mid-transit is at phase=0.5.
    model_folded_phase = np.linspace(0 + 1 / len(t) / 2, 1 + 1 / len(t) / 2, len(t))
    model_folded_model = fractional_transit(
        duration=raw_duration * maxwidth_in_samples * fill_half,
        maxwidth=maxwidth_in_samples / stretch,
        depth=dimming,
        samples=int(len(t)),
        **model_kwargs,
    )
    model_transit_single = fractional_transit(
        duration=(raw_duration * maxwidth_in_samples),
        maxwidth=maxwidth_in_samples / stretch,
        depth=dimming,
        samples=internal_samples,
        **model_kwargs,
    )
    model_lightcurve_model, model_lightcurve_time = model_lightcurve(
        transit_times, period, t, model_transit_single
    )

    (
        depth_mean_odd,
        depth_mean_even,
        depth_mean_odd_std,
        depth_mean_even_std,
        all_flux_intransit_odd,
        all_flux_intransit_even,
        per_transit_count,
        transit_depths,
        transit_depths_uncertainties,
    ) = intransit_stats(t, y, transit_times, transit_duration_in_days)

    all_flux_intransit = np.concatenate([all_flux_intransit_odd, all_flux_intransit_even])
    has_intransit = len(all_flux_intransit) > 0
    depth_mean = float(np.mean(all_flux_intransit)) if has_intransit else np.nan
    depth_mean_std = (
        float(np.std(all_flux_intransit) / np.sum(per_transit_count) ** 0.5)
        if has_intransit and np.sum(per_transit_count) > 0
        else np.nan
    )

    intransit = transit_mask(t, period, 2 * transit_duration_in_days, T0)
    flux_ootr = y[~intransit]
    ootr_std = np.std(flux_ootr) if len(flux_ootr) else np.nan
    snr = (
        float(((1 - depth_mean) / ootr_std) * len(all_flux_intransit) ** 0.5)
        if has_intransit and ootr_std > 0
        else np.nan
    )
    rp_rs = float(rp_rs_from_depth(depth=dimming, law=model.limb_dark, params=model.u))

    odd_even_difference = abs(depth_mean_odd - depth_mean_even)
    odd_even_std_sum = depth_mean_odd_std + depth_mean_even_std
    odd_even_mismatch = (
        float(odd_even_difference / odd_even_std_sum)
        if np.isfinite(odd_even_std_sum) and odd_even_std_sum > 0
        else np.nan
    )

    in_transit_count, after_transit_count, before_transit_count = count_stats(
        t, y, transit_times, transit_duration_in_days
    )
    snr_per_transit, snr_pink_per_transit = snr_stats(
        t=t,
        y=y,
        period=period,
        duration=raw_duration,
        T0=T0,
        transit_times=transit_times,
        transit_duration_in_days=transit_duration_in_days,
        per_transit_count=per_transit_count,
    )

    transit_count = len(transit_times)
    empty_transit_count = int(np.count_nonzero(per_transit_count == 0))
    distinct_transit_count = transit_count - empty_transit_count

    periods = np.asarray(gtls_result.periods, dtype=float)
    power = np.asarray(gtls_result.power, dtype=float)
    SDE = float(gtls_result.SDE)

    return {
        "SDE": SDE,
        "SDE_raw": np.nan,
        "chi2_min": np.nan,
        "chi2red_min": np.nan,
        "period": period,
        "period_uncertainty": float(period_uncertainty(periods, power)),
        "T0": T0,
        "duration": transit_duration_in_days,
        "depth": 1.0 - dimming,
        "depth_mean": (depth_mean, depth_mean_std),
        "depth_mean_even": (float(depth_mean_even), float(depth_mean_even_std)),
        "depth_mean_odd": (float(depth_mean_odd), float(depth_mean_odd_std)),
        "transit_depths": transit_depths,
        "transit_depths_uncertainties": transit_depths_uncertainties,
        "rp_rs": rp_rs,
        "snr": snr,
        "snr_per_transit": snr_per_transit,
        "snr_pink_per_transit": snr_pink_per_transit,
        "odd_even_mismatch": odd_even_mismatch,
        "transit_times": transit_times,
        "per_transit_count": per_transit_count,
        "transit_count": transit_count,
        "distinct_transit_count": distinct_transit_count,
        "empty_transit_count": empty_transit_count,
        "FAP": float(FAP(SDE)),
        "in_transit_count": in_transit_count,
        "after_transit_count": after_transit_count,
        "before_transit_count": before_transit_count,
        "periods": periods,
        "power": power,
        "power_raw": np.asarray(getattr(gtls_result, "raw_power", []), dtype=float),
        "SR": np.full_like(power, np.nan),
        "chi2": np.asarray(getattr(gtls_result, "chi2", []), dtype=float),
        "chi2red": np.full_like(power, np.nan),
        "model_lightcurve_time": model_lightcurve_time,
        "model_lightcurve_model": model_lightcurve_model,
        "model_folded_phase": model_folded_phase,
        "folded_y": folded_y,
        "folded_dy": folded_dy,
        "folded_phase": folded_phase,
        "model_folded_model": model_folded_model,
    }


def run_tls(
    time, flux, flux_err, tls_kwargs, *, quiet: bool = True, verbose: bool | None = None
) -> dict:
    """Run TLS on ``(time, flux, flux_err)`` with ``tls_kwargs`` passed to
    ``power()``, on GPU (GTLS) when available, else CPU
    (``transitleastsquares``). Always returns a plain dict with the same
    fields a CPU TLS results object has -- see module docstring for how the
    GPU path fills in the fields GTLS itself doesn't compute.

    Falls back to CPU TLS (with a ``warnings.warn``, always emitted even
    under ``quiet=True``) if no GPU/``gputls`` is available, or if the GPU
    path raises for any reason.

    ``verbose`` controls GTLS's own progress bar/status prints (its
    ``show_progress_bar`` kwarg is accepted but never read by the installed
    ``gputls`` package -- only its constructor's ``verbose`` gates output).
    Defaults to ``tls_kwargs["show_progress_bar"]`` (falling back to
    ``not quiet``) so the GPU path is exactly as chatty as the CPU path
    would have been for the same call, instead of always running silent.
    """

    def _quiet_call(fn):
        if not quiet:
            return fn()
        with open(os.devnull, "w") as devnull:
            with contextlib.redirect_stdout(devnull):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    return fn()

    gtls_verbose = (
        bool(tls_kwargs.get("show_progress_bar", not quiet)) if verbose is None else verbose
    )

    gtls_engine = get_gpu_tls_engine()
    if gtls_engine is not None:
        try:

            def _run_gpu():
                model = gtls_engine(time, flux, flux_err, verbose=gtls_verbose)
                gtls_result = model.power(**tls_kwargs)
                return _assemble_tls_style_result(model, gtls_result)

            return _quiet_call(_run_gpu)
        except Exception as exc:
            warnings.warn(f"GTLS failed ({exc}); falling back to CPU TLS", stacklevel=2)

    def _run_cpu():
        from transitleastsquares import transitleastsquares as _cpu_tls

        model = _cpu_tls(time, flux, flux_err)
        results = model.power(**tls_kwargs)
        return dict(results)

    return _quiet_call(_run_cpu)
