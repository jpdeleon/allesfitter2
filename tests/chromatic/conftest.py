"""Fixtures for chromatic transit modeling tests.

Each datadir factory writes a minimal but valid allesfitter project layout
(settings.csv + params.csv + per-instrument flux CSVs) into a pytest
``tmp_path`` and returns the directory.

``reset_basement`` is autouse: it nulls the global ``allesfitter.config.BASEMENT``
between tests because ``config.init`` mutates that singleton.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.chromatic._helpers import (
    N_POINTS_PER_INST,
    NOISE_SIGMA,
    RNG_SEED,
    TRUE_COSI,
    TRUE_EPOCH,
    TRUE_PERIOD,
    TRUE_RR_KEPLER,
    TRUE_RR_TESS,
    TRUE_RSUMA,
    common_orbital_rows,
    dilution_rows,
    err_baseline_rows,
    ldc_rows,
    phase_sampled_time,
    simulate_lightcurve,
    write_data_csv,
    write_params,
    write_settings,
)


@pytest.fixture(autouse=True)
def reset_basement():
    """Null the BASEMENT singleton before and after each test."""
    from allesfitter import config

    config.BASEMENT = None
    yield
    config.BASEMENT = None


@pytest.fixture
def two_band_two_inst_datadir(tmp_path):
    """Chromatic config: inst_phot='tess kepler', bandpass='tess kepler'."""
    datadir = tmp_path / "two_band_two_inst"
    datadir.mkdir()
    rng = np.random.default_rng(RNG_SEED)
    for inst, rr in [("tess", TRUE_RR_TESS), ("kepler", TRUE_RR_KEPLER)]:
        time = phase_sampled_time(N_POINTS_PER_INST, TRUE_PERIOD, TRUE_EPOCH, rng=rng)
        flux, err = simulate_lightcurve(time, rr, NOISE_SIGMA, rng)
        write_data_csv(datadir / f"{inst}.csv", time, flux, err)
    write_settings(datadir, inst_phot=["tess", "kepler"], bandpass="tess kepler")
    rows = (
        [
            {"name": "b_rr_tess", "value": TRUE_RR_TESS, "fit": 1,
             "bounds": "uniform 0.0 0.3", "label": "rr_tess"},
            {"name": "b_rr_kepler", "value": TRUE_RR_KEPLER, "fit": 1,
             "bounds": "uniform 0.0 0.3", "label": "rr_kepler"},
        ]
        + common_orbital_rows()
        + dilution_rows(["tess", "kepler"])
        + err_baseline_rows(["tess", "kepler"])
        + ldc_rows("tess")
        + ldc_rows("kepler")
    )
    write_params(datadir / "params.csv", rows=rows)
    return datadir


@pytest.fixture
def one_band_two_inst_datadir(tmp_path):
    """Two instruments share the bandpass 'tess' → chromatic=False."""
    datadir = tmp_path / "one_band_two_inst"
    datadir.mkdir()
    rng = np.random.default_rng(RNG_SEED + 1)
    for inst in ("tess_pdcsap", "tess_qlp"):
        time = phase_sampled_time(N_POINTS_PER_INST, TRUE_PERIOD, TRUE_EPOCH, rng=rng)
        flux, err = simulate_lightcurve(time, TRUE_RR_TESS, NOISE_SIGMA, rng)
        write_data_csv(datadir / f"{inst}.csv", time, flux, err)
    write_settings(
        datadir,
        inst_phot=["tess_pdcsap", "tess_qlp"],
        bandpass="tess tess",
    )
    # Single unique bandpass → chromatic=False, but the suffix used by the
    # parser is the bandpass name ('tess'), not each instrument name. Both
    # instruments therefore share one rr scalar keyed b_rr_tess.
    rows = (
        [
            {"name": "b_rr_tess", "value": TRUE_RR_TESS, "fit": 1,
             "bounds": "uniform 0.0 0.3", "label": "rr_tess"},
        ]
        + common_orbital_rows()
        + dilution_rows(["tess_pdcsap", "tess_qlp"])
        + err_baseline_rows(["tess_pdcsap", "tess_qlp"])
        + ldc_rows("tess")
    )
    write_params(datadir / "params.csv", rows=rows)
    return datadir


@pytest.fixture
def achromatic_datadir(tmp_path):
    """No 'bandpass' setting → achromatic backcompat baseline."""
    datadir = tmp_path / "achromatic"
    datadir.mkdir()
    rng = np.random.default_rng(RNG_SEED + 2)
    time = phase_sampled_time(N_POINTS_PER_INST, TRUE_PERIOD, TRUE_EPOCH, rng=rng)
    flux, err = simulate_lightcurve(time, TRUE_RR_TESS, NOISE_SIGMA, rng)
    write_data_csv(datadir / "tess.csv", time, flux, err)
    write_settings(datadir, inst_phot=["tess"], bandpass=None)
    rows = (
        [
            {"name": "b_rr", "value": TRUE_RR_TESS, "fit": 1,
             "bounds": "uniform 0.0 0.3", "label": "rr"},
        ]
        + common_orbital_rows()
        + dilution_rows(["tess"])
        + err_baseline_rows(["tess"])
        + ldc_rows("tess")
    )
    write_params(datadir / "params.csv", rows=rows)
    return datadir


@pytest.fixture
def make_datadir(tmp_path):
    """Generic factory for negative tests: caller supplies settings/params rows."""

    def _factory(
        name,
        *,
        inst_phot,
        bandpass=None,
        params_rows,
        extra_settings=None,
        write_data=True,
    ):
        datadir = tmp_path / name
        datadir.mkdir()
        if write_data:
            rng = np.random.default_rng(RNG_SEED + 99)
            for inst in inst_phot:
                time = phase_sampled_time(N_POINTS_PER_INST, TRUE_PERIOD, TRUE_EPOCH, rng=rng)
                flux, err = simulate_lightcurve(time, TRUE_RR_TESS, NOISE_SIGMA, rng)
                write_data_csv(datadir / f"{inst}.csv", time, flux, err)
        write_settings(datadir, inst_phot=inst_phot, bandpass=bandpass, extra=extra_settings)
        write_params(datadir / "params.csv", rows=params_rows)
        return datadir

    return _factory


@pytest.fixture
def two_band_e2e_fast_datadir(tmp_path):
    """Same chromatic shape as two_band_two_inst, but with orbital params
    fixed at truth so an NS fit converges in seconds — only the per-band rr
    is free."""
    datadir = tmp_path / "two_band_e2e_fast"
    datadir.mkdir()
    rng = np.random.default_rng(RNG_SEED + 5)
    n_pts = 80  # smaller dataset → faster likelihood
    for inst, rr in [("tess", TRUE_RR_TESS), ("kepler", TRUE_RR_KEPLER)]:
        time = phase_sampled_time(n_pts, TRUE_PERIOD, TRUE_EPOCH, rng=rng)
        flux, err = simulate_lightcurve(time, rr, NOISE_SIGMA, rng)
        write_data_csv(datadir / f"{inst}.csv", time, flux, err)
    # Fix LDCs in addition to orbitals — only rr_tess/rr_kepler stay free.
    write_settings(datadir, inst_phot=["tess", "kepler"], bandpass="tess kepler",
                   extra=["ns_nlive,20", "ns_tol,5.0"])
    rows = (
        [
            {"name": "b_rr_tess", "value": 0.08, "fit": 1,
             "bounds": "uniform 0.0 0.3", "label": "rr_tess"},
            {"name": "b_rr_kepler", "value": 0.08, "fit": 1,
             "bounds": "uniform 0.0 0.3", "label": "rr_kepler"},
        ]
        + common_orbital_rows(fit_orbital=False)
        + dilution_rows(["tess", "kepler"])
        + err_baseline_rows(["tess", "kepler"])
        + [
            {"name": "host_ldc_q1_tess", "value": 0.5, "fit": 0,
             "bounds": "uniform 0 1", "label": "q1_tess"},
            {"name": "host_ldc_q2_tess", "value": 0.5, "fit": 0,
             "bounds": "uniform 0 1", "label": "q2_tess"},
            {"name": "host_ldc_q1_kepler", "value": 0.5, "fit": 0,
             "bounds": "uniform 0 1", "label": "q1_kepler"},
            {"name": "host_ldc_q2_kepler", "value": 0.5, "fit": 0,
             "bounds": "uniform 0 1", "label": "q2_kepler"},
        ]
    )
    write_params(datadir / "params.csv", rows=rows)
    return datadir


@pytest.fixture
def achromatic_e2e_fast_datadir(tmp_path):
    """Achromatic counterpart — only b_rr free; ns settings tuned for speed."""
    datadir = tmp_path / "achromatic_e2e_fast"
    datadir.mkdir()
    rng = np.random.default_rng(RNG_SEED + 6)
    n_pts = 80
    time = phase_sampled_time(n_pts, TRUE_PERIOD, TRUE_EPOCH, rng=rng)
    flux, err = simulate_lightcurve(time, TRUE_RR_TESS, NOISE_SIGMA, rng)
    write_data_csv(datadir / "tess.csv", time, flux, err)
    write_settings(datadir, inst_phot=["tess"], bandpass=None,
                   extra=["ns_nlive,20", "ns_tol,5.0"])
    rows = (
        [
            {"name": "b_rr", "value": 0.08, "fit": 1,
             "bounds": "uniform 0.0 0.3", "label": "rr"},
        ]
        + common_orbital_rows(fit_orbital=False)
        + dilution_rows(["tess"])
        + err_baseline_rows(["tess"])
        + [
            {"name": "host_ldc_q1_tess", "value": 0.5, "fit": 0,
             "bounds": "uniform 0 1", "label": "q1_tess"},
            {"name": "host_ldc_q2_tess", "value": 0.5, "fit": 0,
             "bounds": "uniform 0 1", "label": "q2_tess"},
        ]
    )
    write_params(datadir / "params.csv", rows=rows)
    return datadir


@pytest.fixture
def truth():
    return {
        "period": TRUE_PERIOD,
        "epoch": TRUE_EPOCH,
        "cosi": TRUE_COSI,
        "rsuma": TRUE_RSUMA,
        "rr_tess": TRUE_RR_TESS,
        "rr_kepler": TRUE_RR_KEPLER,
        "noise_sigma": NOISE_SIGMA,
    }
