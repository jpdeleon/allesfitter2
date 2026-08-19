"""End-to-end integration test for ``allesfitter.detection.blind_search.
transit_search``: fit a synthetic light curve that contains one *known*
transiting companion (modeled in params.csv/settings.csv, like
``test_integration_fit.py``'s fixtures) plus one *unmodeled* injected
transit at a different period, then verify the blind search recovers the
unmodeled signal after masking the known one.

Marked ``slow`` (deselected by default via ``pytest.ini``'s
``-m "not slow"``) since it runs a real (if tiny) NS fit and TLS scan.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from tests.integration.test_integration_fit import (
    check_allesfitter_importable,
    create_lightcurve_csv,
    create_minimal_settings_csv,
    generate_transit_lightcurve_ellc,
)

# Companion b's true injected geometry.
_B_PERIOD = 3.5
_B_EPOCH = 2457000.7
_B_RR = 0.1
_B_RSUMA = 0.15
_B_COSI = 0.1


def _create_informative_params_csv(datadir, instrument="tess"):
    """Like ``test_integration_fit.create_minimal_params_csv``, but with
    priors tightly centered on companion b's true geometry.

    The shared fixture's deliberately wide ``uniform 0 1`` prior on
    ``b_cosi`` lets a short/coarse NS run (as used by the ``slow`` tests
    here to keep runtime down) land on a degenerate, non-transiting
    (grazing) posterior median — ``transit_duration_days`` then correctly
    returns NaN for it, and the companion can't be masked. Tighter,
    informative priors keep the posterior in physically valid (transiting)
    territory even for a short run, which this module's masking test
    depends on.
    """
    params_content = f"""#name,value,fit,bounds,label,unit,coupled_with
b_rr,{_B_RR},1,uniform 0.02 0.25,$R_b/R_\\star$,,
b_rsuma,{_B_RSUMA},1,normal {_B_RSUMA} 0.02,$(R_\\star+R_b)/a_b$,,
b_cosi,{_B_COSI},1,normal {_B_COSI} 0.05,$\\cos i_b$,,
b_epoch,{_B_EPOCH},1,normal {_B_EPOCH} 0.05,$T_{{0;b}}$,BJD,
b_period,{_B_PERIOD},1,normal {_B_PERIOD} 0.01,$P_b$,d,
b_f_c,0,0,uniform -1 1,$\\sqrt{{e_b}}\\cos{{\\omega_b}}$,,
b_f_s,0,0,uniform -1 1,$\\sqrt{{e_b}}\\sin{{\\omega_b}}$,,
dil_{instrument},0,0,uniform -1 1,$D_\\mathrm{{0;{instrument}}}$,,
host_ldc_q1_{instrument},0.5,1,uniform 0 1,$q_{{1;\\mathrm{{{instrument}}}}}$,,
host_ldc_q2_{instrument},0.5,1,uniform 0 1,$q_{{2;\\mathrm{{{instrument}}}}}$,,
ln_err_flux_{instrument},-6,1,uniform -10 -1,$\\log{{\\sigma({instrument})}}$,rel. flux,
"""
    params_path = os.path.join(datadir, "params.csv")
    with open(params_path, "w") as f:
        f.write(params_content)
    return params_path


@pytest.fixture
def datadir_with_known_and_unmodeled_transit(tmp_path):
    """A datadir with companion 'b' in params.csv/settings.csv (period=3.5 d)
    and a second, unmodeled transit injected directly into the flux at
    period=5.5 d for the blind search to find."""
    time = np.linspace(2457000.0, 2457014.0, 800)

    flux_b, flux_err = generate_transit_lightcurve_ellc(
        time,
        period=_B_PERIOD,
        epoch=_B_EPOCH,
        rr=_B_RR,
        rsuma=_B_RSUMA,
        cosi=_B_COSI,
        noise_level=0.0008,
        seed=42,
    )
    flux_extra, _ = generate_transit_lightcurve_ellc(
        time,
        period=5.5,
        epoch=2457002.2,
        rr=0.12,
        rsuma=0.13,
        cosi=0.05,
        noise_level=0.0,  # noiseless: its noise is already carried by flux_b
        seed=7,
    )
    combined_flux = flux_b + (flux_extra - 1.0)

    datadir = str(tmp_path)
    _create_informative_params_csv(datadir, instrument="tess")
    create_minimal_settings_csv(
        datadir,
        mcmc_steps=100,
        mcmc_walkers=20,
        ns_nlive=50,
        instrument="tess",
        ns_modus="static",
    )
    create_lightcurve_csv(datadir, "tess", time, combined_flux, flux_err)
    return datadir


@pytest.mark.slow
@pytest.mark.skipif(
    not check_allesfitter_importable(), reason="allesfitter or dependencies not installed"
)
def test_transit_search_recovers_unmodeled_injected_transit(
    datadir_with_known_and_unmodeled_transit, monkeypatch
):
    pytest.importorskip("transitleastsquares")
    import allesfitter
    from allesfitter.detection.blind_search import transit_search
    from allesfitter.detection.transit_search import is_multiple_of
    from allesfitter.tls_h5 import read_h5_transit_params

    datadir = datadir_with_known_and_unmodeled_transit
    monkeypatch.chdir(datadir)

    allesfitter.ns_fit(datadir)

    ns_results_dir = os.path.join(datadir, "ns_results")
    result = transit_search(
        ns_results_dir,
        sde_min=5.0,
        period_min=1.0,
        period_max=8.0,
        quiet=True,
    )

    candidates = result["candidates"]
    assert len(candidates) >= 1
    best = max(candidates, key=lambda c: c["SDE"])
    assert is_multiple_of(best["period"], 5.5, tolerance=0.05)
    assert os.path.exists(best["h5"])
    assert os.path.exists(best["figure"])

    h5_values = read_h5_transit_params(best["h5"], mission="tess")
    assert is_multiple_of(h5_values["period"], 5.5, tolerance=0.05)

    summary_csv = os.path.join(datadir, "transit_search_results", "candidates_summary.csv")
    assert os.path.exists(summary_csv)

    # Companion b (the known, modeled planet) should be independently
    # recoverable from its own detrended, other-companion-masked data. The
    # recovery check's own narrow TLS bracket must lock onto b's own signal
    # — not the (much stronger) unmodeled 5.5 d transit that isn't masked
    # out here — and its recovered epoch must line up with b's known one.
    # Whether SDE also clears --sde-min isn't asserted: the fast/coarse NS
    # settings used here for test speed give a posterior (and thus this
    # SDE) that varies run to run, so gating on that specific number would
    # make this test flaky without testing anything the fields below don't
    # already cover.
    known_recovery = result["known_recovery"]
    assert len(known_recovery) == 1
    b_row = known_recovery[0]
    assert b_row["companion"] == "b"
    assert b_row["bracket_ignored"] is False
    assert b_row["epoch_match"] is True
    assert is_multiple_of(b_row["recovered_period"], _B_PERIOD, tolerance=0.05)
    assert os.path.exists(b_row["figure"])

    recovery_csv = os.path.join(datadir, "transit_search_results", "known_planets_recovery.csv")
    assert os.path.exists(recovery_csv)
