"""Regenerate params.csv from a completed fit's posterior.

Migrated out of ``prepare_allesfit.py``'s ``-r/--results_dir`` flag into its
own command (``uv run allesfitter update-params <dir>``), since it's a
distinct step from data download/preparation: given a directory with a
completed MCMC/NS fit, write ``params2.csv`` alongside it as a drop-in
warm-start prior for a follow-up fit (e.g. after adding another sector's
data) — every fitted parameter gets a data-driven prior (normal or uniform,
picked via an Anderson-Darling normality test against its posterior
samples) centered on the posterior, and every fixed parameter is carried
over unchanged.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from loguru import logger
from scipy.stats import anderson

from allesfitter import allesclass
from allesfitter.exoworlds_rdx.lightcurves.index_transits import get_tmid_observed_transits


def update_params(dir_path: str, ttv: bool = False, debug: bool = False) -> str:
    """Write ``<dir_path>/params2.csv`` from ``dir_path``'s posterior.

    Parameters
    ----------
    dir_path : str
        Directory with a completed MCMC/NS fit (``params.csv``/
        ``settings.csv`` plus ``mcmc_results/`` or ``ns_results/``), as
        loaded by :class:`allesfitter.allesclass`.
    ttv : bool
        Append per-transit TTV rows (one per companion, one row per
        transit with actual data coverage) using the posterior-median
        epoch/period.
    debug : bool
        Log each row's derivation as it's built.

    Returns
    -------
    str
        Path to the written ``params2.csv``.
    """
    dir_path = Path(dir_path)
    alles = allesclass(str(dir_path))
    logger.info("Updating params.csv")

    # Iterate BASEMENT.allkeys (every row in the prior params.csv), not just
    # BASEMENT.fitkeys — a fixed parameter (e.g. an LDC pair pinned at a
    # fitted value, dilution, fixed eccentricity terms) that got dropped
    # here would leave None values at the next fit's config.init time and a
    # cryptic numpy ufunc TypeError when q_to_u tries to sqrt(None). Fitted
    # keys get the posterior-derived prior row; fixed keys get a faithful
    # `name,value,0,,label,unit,` copy.
    text = "#name,value,fit,bounds,label,unit,coupled_with\n"
    try:
        ll = alles.posterior_params_ll.copy()  # 16th percentile
        median = alles.posterior_params_median.copy()  # 50th percentile
        ul = alles.posterior_params_ul.copy()  # 84th percentile
        fitkeys_set = set(alles.BASEMENT.fitkeys)
        for i, name in enumerate(alles.BASEMENT.allkeys):
            label = alles.BASEMENT.labels[i]
            unit = alles.BASEMENT.units[i]
            if name in fitkeys_set:
                # Fitted parameter: derive prior from posterior shape.
                norm_test = anderson(alles.posterior_params[name], dist="norm")
                # crit vals at 15/10/5/2.5/1%; statistic < crit[0] (15%)
                # -> fail to reject normality.
                dist = "normal" if norm_test.statistic < norm_test.critical_values[0] else "uniform"
                if dist == "normal":
                    l_err, mid, u_err = ll[name], median[name], ul[name]
                    sig = np.sqrt(l_err**2 + u_err**2)
                    text += f"{name},{mid:6f},1,normal {mid:6f} {sig:6f},{label},{unit},\n"
                    if debug:
                        logger.info(f"{name}: {mid:.6f} +{u_err:.6f} -{l_err:.6f} (normal)")
                else:
                    l_limit, mid, u_limit = np.nanpercentile(
                        alles.posterior_params[name], q=[1, 50, 99]
                    )
                    text += f"{name},{mid:6f},1,uniform {l_limit:6f} {u_limit:6f},{label},{unit},\n"
                    if debug:
                        logger.info(f"{name}: {l_limit:.6f} < {mid:.6f} < {u_limit:.6f} (uniform)")
            else:
                # Fixed parameter: preserve from BASEMENT.params so params2.csv
                # stays a complete drop-in for params.csv. Skip None values
                # (intentionally absent entries, e.g. LD law set to None).
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

    if ttv:
        text += _ttv_rows(alles)

    if debug:
        logger.info(text)
    fp = dir_path / "params2.csv"
    np.savetxt(fp, [text], fmt="%s")
    logger.info(f"Saved: {fp}")
    return str(fp)


def _ttv_rows(alles) -> str:
    """Per-transit TTV rows using observed-transit midpoints (same source
    as ``afplot_per_transit`` in ``general_output.py``).

    ``BASEMENT`` only caches ``{c}_tmid_observed_transits`` when
    ``settings['fit_ttvs']==True`` (``basement.setup_ttv_fit``); on a cache
    miss this recomputes via ``get_tmid_observed_transits`` from the
    per-instrument time arrays and posterior-median epoch/period, and
    cross-checks the cached count (when present) against a fresh
    recomputation from the full time union, preferring the fresh count on
    disagreement (can happen if the cache was built from fast-fit-reduced
    data of only one instrument, or with a stale epoch/period).
    """
    text = ""
    settings = alles.BASEMENT.settings
    data = alles.BASEMENT.data
    median = alles.posterior_params_median
    fast_fit_width = float(settings.get("fast_fit_width", 0.3333333333333333))
    inst_list = list(settings.get("inst_phot", []))
    for inst in inst_list:
        t_arr = data.get(inst, {}).get("time")
        if t_arr is None or len(t_arr) == 0:
            logger.warning(f"TTV: inst '{inst}' has no time data in BASEMENT.data")
        else:
            t_arr = np.asarray(t_arr, dtype=float)
            logger.info(
                f"TTV: inst '{inst}' N={len(t_arr)} "
                f"time=[{t_arr.min():.4f}, {t_arr.max():.4f}] "
                f"span={t_arr.max() - t_arr.min():.3f} d"
            )
    for c in settings.get("companions_phot", []):
        epoch = float(median.get(f"{c}_epoch", alles.BASEMENT.params[f"{c}_epoch"]))
        period = float(median.get(f"{c}_period", alles.BASEMENT.params[f"{c}_period"]))
        tmids = data.get(f"{c}_tmid_observed_transits")
        used_cache = tmids is not None and len(tmids) > 0
        if not used_cache:
            times = []
            for inst in inst_list:
                times += list(data[inst]["time"])
            times = np.sort(np.asarray(times, dtype=float))
            tmids = get_tmid_observed_transits(times, epoch, period, fast_fit_width)
            logger.info(
                f"TTV: {c} cache miss; recomputed via get_tmid_observed_transits "
                f"(width={fast_fit_width:.4f} d)"
            )
        else:
            logger.info(
                f"TTV: {c} using BASEMENT cache "
                f"(populated by setup_ttv_fit; fit_ttvs={settings.get('fit_ttvs')})"
            )
        times_union = []
        for inst in inst_list:
            times_union += list(data[inst]["time"])
        times_union = np.sort(np.asarray(times_union, dtype=float))
        tmids_union = get_tmid_observed_transits(times_union, epoch, period, fast_fit_width)
        if len(tmids_union) != len(tmids):
            logger.warning(
                f"TTV: {c} cache_count={len(tmids)} vs fresh union_count={len(tmids_union)} "
                f"(epoch={epoch:.6f}, period={period:.6f}, width={fast_fit_width:.4f}). "
                "Using union count."
            )
            tmids = tmids_union
        n_obs = len(tmids)
        logger.info(f"TTV: {c} has {n_obs} transits with data")
        if n_obs == 0:
            continue
        text += f"#TTV companion {c},,,,,\n"
        for j in range(n_obs):
            text += (
                f"{c}_ttv_transit_{j + 1},0,1,uniform -0.05 0.05,TTV$_\\mathrm{{ttv;{j + 1}}}$,d,\n"
            )
    return text
