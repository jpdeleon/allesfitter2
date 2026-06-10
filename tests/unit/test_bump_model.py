"""Tests for the Gaussian "bump" model (starspot-crossing events).

The bump is the additive twin of the flare model: a time-localized Gaussian
added to the relative flux. The crossing depth is wavelength-dependent, so the
count and amplitude are keyed by *bandpass* (``N_bumps_<bp>`` /
``bump_ampl_<bp>_<i>``) — exactly like limb darkening — falling back to the
instrument name when no bandpass row is present. The geometry
(``bump_tpeak_<i>`` / ``bump_width_<i>``) is shared across bands.

    bump(t) = ampl * exp(-(t - tpeak)**2 / (2 * width**2))

Covered here:

* ``bumps.bump.bump_model`` — the pure Gaussian (peak, symmetry, decay).
* ``computer.flux_subfct_bumps`` — assembly with no bumps (no-op), a single
  bump, bandpass-shared amplitude across instruments, instrument fallback,
  multiple summed bumps, and the ``(1 - dil)`` dilution scaling — all driven by
  plain params/settings dicts.
* End-to-end through ``computer.flux_fct``: a config-initialised datadir with
  ``N_bumps_tess,1`` validates/loads, and toggling it recovers exactly the
  additive Gaussian (regression guard that the wiring is purely additive).
"""

from __future__ import annotations

import numpy as np
import pytest

from allesfitter.bumps.bump import bump_model
from tests.chromatic._helpers import (
    TRUE_EPOCH,
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


# --------------------------------------------------------------------------- #
# bump_model — the pure Gaussian
# --------------------------------------------------------------------------- #
def test_bump_model_peaks_at_tpeak():
    # Arrange
    tpeak, width, ampl = 100.0, 0.05, 0.003
    t = np.linspace(tpeak - 1.0, tpeak + 1.0, 4001)

    # Act
    bump = bump_model(t, tpeak, width, ampl)

    # Assert: maximum equals the amplitude and sits at tpeak
    assert bump.max() == pytest.approx(ampl)
    assert t[np.argmax(bump)] == pytest.approx(tpeak, abs=1e-3)


def test_bump_model_is_symmetric_and_decays_to_zero():
    # Arrange
    tpeak, width, ampl = 0.0, 0.1, 1.0

    # Act / Assert: symmetric about tpeak and ~0 many widths away
    assert bump_model(np.array([-0.3]), tpeak, width, ampl)[0] == pytest.approx(
        bump_model(np.array([0.3]), tpeak, width, ampl)[0]
    )
    assert bump_model(np.array([2.0]), tpeak, width, ampl)[0] == pytest.approx(0.0, abs=1e-12)
    # one sigma away → exp(-1/2) of the amplitude
    assert bump_model(np.array([width]), tpeak, width, ampl)[0] == pytest.approx(
        ampl * np.exp(-0.5)
    )


def test_bump_model_zero_amplitude_is_flat():
    t = np.linspace(-1.0, 1.0, 101)
    assert np.all(bump_model(t, 0.0, 0.1, 0.0) == 0.0)


# --------------------------------------------------------------------------- #
# flux_subfct_bumps — assembly (no config needed; dicts passed explicitly)
# --------------------------------------------------------------------------- #
def _grid(tpeak=100.0):
    return np.linspace(tpeak - 0.5, tpeak + 0.5, 1001)


def test_flux_subfct_bumps_no_bumps_returns_ones():
    from allesfitter import computer

    xx = _grid()
    out = computer.flux_subfct_bumps(
        {"dil_tess": 0.0}, "tess", "b", xx=xx, settings={"N_bumps_tess": 0}
    )
    assert np.array_equal(out, np.ones_like(xx))


def test_flux_subfct_bumps_single_bump_adds_amplitude():
    from allesfitter import computer

    tpeak, width, ampl = 100.0, 0.03, 0.004
    xx = _grid(tpeak)
    params = {
        "dil_tess": 0.0,
        "bump_tpeak_1": tpeak,
        "bump_width_1": width,
        "bump_ampl_tess_1": ampl,
    }
    out = computer.flux_subfct_bumps(params, "tess", "b", xx=xx, settings={"N_bumps_tess": 1})

    expected = 1.0 + bump_model(xx, tpeak, width, ampl)
    assert np.allclose(out, expected)
    assert out.max() == pytest.approx(1.0 + ampl)


def test_flux_subfct_bumps_amplitude_keyed_by_bandpass():
    """Two instruments sharing a bandpass share one amplitude (bump_ampl_<bp>)."""
    from allesfitter import computer

    tpeak, width, ampl = 100.0, 0.03, 0.004
    xx = _grid(tpeak)
    # muscat_g and lco_g both observe in bandpass 'g'
    settings = {"bandpass": {"muscat_g": "g", "lco_g": "g"}, "N_bumps_g": 1}
    params = {
        "dil_muscat_g": 0.0,
        "dil_lco_g": 0.0,
        "bump_tpeak_1": tpeak,
        "bump_width_1": width,
        "bump_ampl_g_1": ampl,  # single shared, bandpass-keyed amplitude
    }

    out_muscat = computer.flux_subfct_bumps(params, "muscat_g", "b", xx=xx, settings=settings)
    out_lco = computer.flux_subfct_bumps(params, "lco_g", "b", xx=xx, settings=settings)

    assert np.allclose(out_muscat, out_lco)
    assert out_muscat.max() == pytest.approx(1.0 + ampl)


def test_flux_subfct_bumps_amplitude_falls_back_to_instrument():
    """Without a bandpass row the amplitude key falls back to the instrument."""
    from allesfitter import computer

    tpeak, width = 100.0, 0.03
    xx = _grid(tpeak)
    params = {
        "dil_tess": 0.0,
        "dil_kepler": 0.0,
        "bump_tpeak_1": tpeak,
        "bump_width_1": width,
        "bump_ampl_tess_1": 0.004,
        "bump_ampl_kepler_1": 0.001,
    }
    settings = {"N_bumps_tess": 1, "N_bumps_kepler": 1}

    tess = computer.flux_subfct_bumps(params, "tess", "b", xx=xx, settings=settings)
    kepler = computer.flux_subfct_bumps(params, "kepler", "b", xx=xx, settings=settings)

    assert tess.max() == pytest.approx(1.0 + 0.004)
    assert kepler.max() == pytest.approx(1.0 + 0.001)


def test_flux_subfct_bumps_multiple_bumps_sum():
    from allesfitter import computer

    xx = np.linspace(99.0, 101.0, 4001)
    params = {
        "dil_tess": 0.0,
        "bump_tpeak_1": 99.5, "bump_width_1": 0.02, "bump_ampl_tess_1": 0.003,
        "bump_tpeak_2": 100.5, "bump_width_2": 0.02, "bump_ampl_tess_2": 0.005,
    }
    out = computer.flux_subfct_bumps(params, "tess", "b", xx=xx, settings={"N_bumps_tess": 2})

    expected = (
        1.0
        + bump_model(xx, 99.5, 0.02, 0.003)
        + bump_model(xx, 100.5, 0.02, 0.005)
    )
    assert np.allclose(out, expected)


def test_flux_subfct_bumps_scaled_by_dilution():
    """The bump is diluted by (1 - dil), exactly like the flare model."""
    from allesfitter import computer

    tpeak, width, ampl, dil = 100.0, 0.03, 0.01, 0.25
    xx = _grid(tpeak)
    params = {
        "dil_tess": dil,
        "bump_tpeak_1": tpeak,
        "bump_width_1": width,
        "bump_ampl_tess_1": ampl,
    }
    out = computer.flux_subfct_bumps(params, "tess", "b", xx=xx, settings={"N_bumps_tess": 1})

    assert (out.max() - 1.0) == pytest.approx((1.0 - dil) * ampl)


# --------------------------------------------------------------------------- #
# bumps_persistent — a long-lived spot crossed every transit (tpeak + n*period)
# --------------------------------------------------------------------------- #
def test_flux_subfct_bumps_persistent_repeats_each_period():
    """With bumps_persistent the bump recurs at tpeak + n*period across xx."""
    from allesfitter import computer

    tpeak, width, ampl, period = 100.0, 0.02, 0.005, 1.0
    xx = np.linspace(99.5, 102.5, 4001)  # spans transits at 100, 101, 102
    params = {
        "dil_tess": 0.0,
        "b_period": period,
        "bump_tpeak_1": tpeak,
        "bump_width_1": width,
        "bump_ampl_tess_1": ampl,
    }
    settings = {"N_bumps_tess": 1, "bumps_persistent": True}

    out = computer.flux_subfct_bumps(params, "tess", "b", xx=xx, settings=settings)

    # a bump of height ~ampl sits at every transit center inside the span
    for center in (100.0, 101.0, 102.0):
        near = np.abs(xx - center) < 0.1
        assert (out[near].max() - 1.0) == pytest.approx(ampl, rel=1e-2)

    # matches the explicit per-epoch sum
    expected = np.ones_like(xx)
    for n in range(-1, 4):
        expected += bump_model(xx, tpeak + n * period, width, ampl)
    assert np.allclose(out, expected)


def test_flux_subfct_bumps_non_persistent_is_single_event():
    """Without the flag (default), only the bump at tpeak appears."""
    from allesfitter import computer

    tpeak, width, ampl, period = 100.0, 0.02, 0.005, 1.0
    xx = np.linspace(99.5, 102.5, 4001)
    params = {
        "dil_tess": 0.0,
        "b_period": period,
        "bump_tpeak_1": tpeak,
        "bump_width_1": width,
        "bump_ampl_tess_1": ampl,
    }
    # bumps_persistent absent -> defaults to False via .get
    out = computer.flux_subfct_bumps(params, "tess", "b", xx=xx, settings={"N_bumps_tess": 1})

    near100 = np.abs(xx - 100.0) < 0.1
    near101 = np.abs(xx - 101.0) < 0.1
    assert (out[near100].max() - 1.0) == pytest.approx(ampl, rel=1e-2)
    assert (out[near101].max() - 1.0) == pytest.approx(0.0, abs=1e-6)


# --------------------------------------------------------------------------- #
# End-to-end through config + flux_fct
# --------------------------------------------------------------------------- #
@pytest.fixture(autouse=True)
def reset_basement():
    from allesfitter import config

    config.BASEMENT = None
    yield
    config.BASEMENT = None


_BUMP_TPEAK = TRUE_EPOCH
_BUMP_WIDTH = 0.02
_BUMP_AMPL = 0.005


def _bump_datadir(tmp_path):
    d = tmp_path / "bump_dd"
    d.mkdir()
    rng = np.random.default_rng(0)
    t = phase_sampled_time(400, period=3.14159, epoch=TRUE_EPOCH, rng=rng)
    flux, err = simulate_lightcurve(t, rr=0.1, noise_sigma=5e-4, rng=rng)
    write_data_csv(d / "tess.csv", t, flux, err)

    write_settings(d, inst_phot=["tess"], bandpass=None, extra=["N_bumps_tess,1"])
    rows = (
        [{"name": "b_rr", "value": 0.1, "fit": 0, "bounds": "uniform 0 0.3", "label": "rr"}]
        + common_orbital_rows(fit_orbital=False)
        + dilution_rows(["tess"])
        + err_baseline_rows(["tess"])
        + ldc_rows("tess")
        + [
            {"name": "bump_tpeak_1", "value": _BUMP_TPEAK, "fit": 0,
             "bounds": f"uniform {_BUMP_TPEAK - 0.1} {_BUMP_TPEAK + 0.1}",
             "label": "bump_tpeak_1", "unit": "BJD"},
            {"name": "bump_width_1", "value": _BUMP_WIDTH, "fit": 0,
             "bounds": "uniform 0.001 0.1", "label": "bump_width_1", "unit": "d"},
            {"name": "bump_ampl_tess_1", "value": _BUMP_AMPL, "fit": 0,
             "bounds": "uniform 0 0.05", "label": "bump_ampl_tess_1"},
        ]
    )
    write_params(d / "params.csv", rows=rows)
    return d


def test_bump_params_validate_and_load(tmp_path):
    from allesfitter import config

    # config.init must accept the N_bumps_<inst> setting and the bump_* param keys
    config.init(str(_bump_datadir(tmp_path)))
    assert config.BASEMENT.settings["N_bumps_tess"] == 1
    # bumps_persistent defaults to False when absent from settings.csv
    assert config.BASEMENT.settings["bumps_persistent"] is False


def test_flux_fct_adds_exactly_the_bump(tmp_path):
    from allesfitter import computer, config

    config.init(str(_bump_datadir(tmp_path)))
    theta = [config.BASEMENT.params[k] for k in config.BASEMENT.fitkeys]
    params = computer.update_params(np.array(theta, dtype=float))

    grid = np.linspace(_BUMP_TPEAK - 0.2, _BUMP_TPEAK + 0.2, 401)

    with_bump = computer.flux_fct(params, "tess", "b", xx=grid)
    config.BASEMENT.settings["N_bumps_tess"] = 0
    without_bump = computer.flux_fct(params, "tess", "b", xx=grid)
    config.BASEMENT.settings["N_bumps_tess"] = 1

    # the difference is purely the additive (1 - dil) * Gaussian bump (dil = 0)
    expected = bump_model(grid, _BUMP_TPEAK, _BUMP_WIDTH, _BUMP_AMPL)
    assert np.allclose(with_bump - without_bump, expected, atol=1e-10)
