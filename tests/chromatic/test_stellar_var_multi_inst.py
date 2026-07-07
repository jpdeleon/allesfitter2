"""Regression test for the shared stellar-variability GP across multiple
photometric instruments.

Bug: ``computer.calculate_stellar_var`` looped with ``for inst in insts:``,
shadowing its own ``inst`` argument. After the loop ``inst`` held the last
instrument, so a *single-instrument* call still set ``x`` to the merged phot
time grid (all instruments) while ``y``/``yerr_w`` held only that one
instrument's points → ``celerite``'s ``gp.compute`` raised
``could not broadcast input array from shape (n_inst,) into shape (n_total,)``.

Only bites with 2+ photometric instruments (with one, merged time == that
instrument's time). These tests use two instruments so the per-instrument
length (N) differs from the merged length (2N).
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.chromatic._helpers import (
    N_POINTS_PER_INST,
    TRUE_RR_TESS,
    common_orbital_rows,
    dilution_rows,
    err_baseline_rows,
    ldc_rows,
)

# celerite SHOTerm params for the shared flux GP (fixed; values just need to
# yield a valid positive-definite kernel).
SHO_PARAM_ROWS = [
    {
        "name": "stellar_var_gp_sho_lnS0_flux",
        "value": -12.0,
        "fit": 0,
        "bounds": "uniform -30 0",
        "label": "lnS0",
    },
    {
        "name": "stellar_var_gp_sho_lnQ_flux",
        "value": 0.0,
        "fit": 0,
        "bounds": "uniform -5 5",
        "label": "lnQ",
    },
    {
        "name": "stellar_var_gp_sho_lnomega0_flux",
        "value": 0.0,
        "fit": 0,
        "bounds": "uniform -5 5",
        "label": "lnomega0",
    },
]


@pytest.fixture
def two_inst_stellar_var_datadir(make_datadir):
    """Achromatic config with two photometric instruments sharing a SHO GP."""
    insts = ["tess", "kepler"]
    rows = (
        [
            {
                "name": "b_rr",
                "value": TRUE_RR_TESS,
                "fit": 1,
                "bounds": "uniform 0.0 0.3",
                "label": "rr",
            }
        ]
        + common_orbital_rows(fit_orbital=False)
        + dilution_rows(insts)
        + err_baseline_rows(insts)
        + ldc_rows("tess")
        + ldc_rows("kepler")
        + SHO_PARAM_ROWS
    )
    return make_datadir(
        "two_inst_stellar_var",
        inst_phot=insts,
        bandpass=None,
        params_rows=rows,
        extra_settings=["stellar_var_flux,sample_GP_SHO"],
    )


def _init_params(datadir):
    """Init config and return (config, params-dict from the initial guess)."""
    from allesfitter import config
    from allesfitter.computer import update_params

    config.init(datadir)
    return config, update_params(config.BASEMENT.theta_0)


def test_per_instrument_stellar_var_matches_that_instrument_length(
    two_inst_stellar_var_datadir,
):
    """A single-instrument call must return that instrument's own length,
    not the merged length — this is the exact call mcmc_output/ns_output make."""
    config, params = _init_params(two_inst_stellar_var_datadir)
    from allesfitter.computer import calculate_stellar_var

    for inst in config.BASEMENT.settings["inst_phot"]:
        n_inst = len(config.BASEMENT.data[inst]["time"])
        sv = np.asarray(calculate_stellar_var(params, inst, "flux"))
        assert sv.shape == (n_inst,), f"{inst}: expected ({n_inst},), got {sv.shape}"
        assert n_inst == N_POINTS_PER_INST


def test_all_instruments_stellar_var_matches_merged_length(
    two_inst_stellar_var_datadir,
):
    """The shared-GP ('all') path returns the merged, time-sorted length and
    does not raise — guards the ind_sort branch that the same bug disabled."""
    config, params = _init_params(two_inst_stellar_var_datadir)
    from allesfitter.computer import calculate_stellar_var

    n_merged = len(config.BASEMENT.data["inst_phot"]["time"])
    sv = np.asarray(calculate_stellar_var(params, "all", "flux"))
    assert sv.shape == (n_merged,)
    assert n_merged == sum(
        len(config.BASEMENT.data[i]["time"]) for i in config.BASEMENT.settings["inst_phot"]
    )
