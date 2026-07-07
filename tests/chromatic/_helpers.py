"""Shared row-builders for chromatic fixtures and error tests.

Kept separate from conftest.py so test modules can import them with a
plain ``from tests.chromatic._helpers import ...`` style without depending
on pytest's conftest auto-discovery.
"""

from __future__ import annotations

import numpy as np

TRUE_PERIOD = 3.14159
TRUE_EPOCH = 2459000.5
TRUE_COSI = 0.01
TRUE_RSUMA = 0.12
TRUE_RR_TESS = 0.100
TRUE_RR_KEPLER = 0.105
TRUE_LDC = [0.30, 0.20]
NOISE_SIGMA = 5.0e-4
N_POINTS_PER_INST = 200
RNG_SEED = 20260526


def simulate_lightcurve(time, rr, noise_sigma, rng):
    """Direct ellc invocation — equivalent to v2.simulate.transit_model but
    without that module's broken import of allesfitter.v2.flares."""
    import ellc

    radius_1 = TRUE_RSUMA / (1.0 + rr)
    radius_2 = radius_1 * rr
    incl = np.degrees(np.arccos(TRUE_COSI))
    flux = ellc.lc(
        t_obs=time,
        radius_1=radius_1,
        radius_2=radius_2,
        sbratio=0.0,
        incl=incl,
        t_zero=TRUE_EPOCH,
        period=TRUE_PERIOD,
        ldc_1=TRUE_LDC,
        ld_1="quad",
        verbose=False,
    )
    flux = flux + rng.normal(0.0, noise_sigma, size=time.size)
    err = np.full_like(time, noise_sigma)
    return flux, err


def phase_sampled_time(n_points, period, epoch, span_periods=5, rng=None):
    rng = rng or np.random.default_rng(RNG_SEED)
    base = epoch - period * (span_periods / 2.0)
    times = base + np.sort(rng.uniform(0.0, period * span_periods, size=n_points))
    return times


def write_data_csv(path, time, flux, err):
    np.savetxt(path, np.column_stack([time, flux, err]), delimiter=",")


def write_settings(datadir, inst_phot, bandpass=None, extra=None):
    """Write a minimal datadir/settings.csv."""
    path = datadir / "settings.csv"
    rows = [
        "#name,value",
        "companions_phot,b",
        "companions_rv,",
        f"inst_phot,{' '.join(inst_phot)}",
        "inst_rv,",
        "time_format,BJD_TDB",
        "multiprocess,False",
        "fast_fit,False",
        "secondary_eclipse,False",
        "phase_curve,False",
        "shift_epoch,False",
        "inst_for_b_epoch,all",
        "mcmc_nwalkers,20",
        "mcmc_total_steps,200",
        "mcmc_burn_steps,100",
        "mcmc_thin_by,1",
        "ns_modus,static",
        "ns_nlive,50",
        "ns_bound,single",
        "ns_sample,unif",
        "ns_tol,1.0",
        "fit_ttvs,False",
        "use_host_density_prior,False",
        "use_tidal_eccentricity_prior,False",
    ]
    if bandpass is not None:
        rows.append(f"bandpass,{bandpass}")
    for inst in inst_phot:
        rows.append(f"host_ld_law_{inst},quad")
        rows.append(f"host_grid_{inst},default")
        rows.append(f"b_ld_law_{inst},none")
        rows.append(f"b_grid_{inst},default")
        rows.append(f"host_shape_{inst},sphere")
        rows.append(f"b_shape_{inst},sphere")
        rows.append(f"host_flux_weighted_{inst},False")
        rows.append(f"b_flux_weighted_{inst},False")
        rows.append(f"baseline_flux_{inst},sample_offset")
        rows.append(f"error_flux_{inst},hybrid_inv_sigma2")
        rows.append(f"host_ld_space_{inst},q")
        rows.append(f"b_ld_space_{inst},q")
    if extra:
        rows.extend(extra)
    path.write_text("\n".join(rows) + "\n")


def write_params(path, *, rows):
    """Write params.csv. ``rows`` is a list of dicts with name/value/fit/bounds/label/unit."""
    header = "#name,value,fit,bounds,label,unit,coupled_with"
    lines = [header]
    for r in rows:
        lines.append(
            ",".join(
                [
                    r["name"],
                    str(r["value"]),
                    str(r.get("fit", 0)),
                    r.get("bounds", "uniform 0 1"),
                    r.get("label", r["name"]),
                    r.get("unit", ""),
                    r.get("coupled_with", ""),
                ]
            )
        )
    path.write_text("\n".join(lines) + "\n")


def common_orbital_rows(fit_orbital=True):
    """fit_orbital=False fixes period/epoch/cosi/rsuma at truth (used by
    fast end-to-end tests that only need to recover the per-band rr)."""
    fit = 1 if fit_orbital else 0
    return [
        {
            "name": "b_rsuma",
            "value": TRUE_RSUMA,
            "fit": fit,
            "bounds": "uniform 0.01 0.5",
            "label": "rsuma",
        },
        {
            "name": "b_cosi",
            "value": TRUE_COSI,
            "fit": fit,
            "bounds": "uniform 0 1",
            "label": "cosi",
        },
        {
            "name": "b_epoch",
            "value": TRUE_EPOCH,
            "fit": fit,
            "bounds": f"uniform {TRUE_EPOCH - 0.05} {TRUE_EPOCH + 0.05}",
            "label": "epoch",
            "unit": "BJD",
        },
        {
            "name": "b_period",
            "value": TRUE_PERIOD,
            "fit": fit,
            "bounds": f"uniform {TRUE_PERIOD - 0.01} {TRUE_PERIOD + 0.01}",
            "label": "period",
            "unit": "d",
        },
        {"name": "b_f_c", "value": 0.0, "fit": 0, "bounds": "uniform -1 1", "label": "f_c"},
        {"name": "b_f_s", "value": 0.0, "fit": 0, "bounds": "uniform -1 1", "label": "f_s"},
    ]


def dilution_rows(inst_phot):
    return [
        {
            "name": f"dil_{inst}",
            "value": 0.0,
            "fit": 0,
            "bounds": "uniform -1 1",
            "label": f"dil_{inst}",
        }
        for inst in inst_phot
    ]


def err_baseline_rows(inst_phot):
    """ln_err and baseline_offset rows required by update_params.

    update_params unconditionally evaluates np.exp(ln_err_flux_<inst>) and
    the offset baseline mode requires baseline_offset_flux_<inst>.
    """
    rows = []
    for inst in inst_phot:
        rows.append(
            {
                "name": f"ln_err_flux_{inst}",
                "value": -7.0,
                "fit": 0,
                "bounds": "uniform -15 0",
                "label": f"ln_err_{inst}",
            }
        )
        rows.append(
            {
                "name": f"baseline_offset_flux_{inst}",
                "value": 0.0,
                "fit": 0,
                "bounds": "uniform -0.05 0.05",
                "label": f"offset_{inst}",
            }
        )
    return rows


def ldc_rows(suffix):
    return [
        {
            "name": f"host_ldc_q1_{suffix}",
            "value": 0.5,
            "fit": 1,
            "bounds": "uniform 0 1",
            "label": f"q1_{suffix}",
        },
        {
            "name": f"host_ldc_q2_{suffix}",
            "value": 0.5,
            "fit": 1,
            "bounds": "uniform 0 1",
            "label": f"q2_{suffix}",
        },
    ]
