"""Likelihood-assembly regression tests for chromatic mode.

We monkeypatch ``ellc.fluxes`` so the C extension never runs — instead the
mock captures the kwargs that ``flux_subfct_ellc`` would have passed to ellc.
Each test asserts a scope contract on those kwargs:

- per-band rr appears as ``radius_2/radius_1`` for its instrument's band
- per-instrument LDC lists are assembled from bandpass-scoped scalars
- shared orbital params (incl, t_zero, period, a, q, f_c, f_s) are bit-equal
  across all instruments
- achromatic fallback fires when a chromatic key is missing from params
"""

from __future__ import annotations

import numpy as np
import pytest

from allesfitter import computer, config

# Orbital kwargs that ellc receives — must be equal across all insts/bands
SHARED_ORBITAL_KWARGS = ("incl", "t_zero", "period", "a", "q", "f_c", "f_s")


@pytest.fixture
def captured_fluxes_calls(monkeypatch):
    """Patch ellc.fluxes; each call returns a benign (1.0, 0.0) flux tuple and
    is recorded in ``calls`` as the kwargs dict. Same for ellc.lc as fallback."""
    import ellc

    calls = []

    def fake_fluxes(**kwargs):
        calls.append(("fluxes", dict(kwargs)))
        n = len(np.atleast_1d(kwargs.get("t_obs", [0])))
        return np.ones(n), np.zeros(n)

    def fake_lc(**kwargs):
        calls.append(("lc", dict(kwargs)))
        n = len(np.atleast_1d(kwargs.get("t_obs", [0])))
        return np.ones(n)

    monkeypatch.setattr(ellc, "fluxes", fake_fluxes)
    monkeypatch.setattr(ellc, "lc", fake_lc)
    return calls


def _prepared_params():
    """Run update_params with the injected truth (fitkey values from params.csv)."""
    theta = [config.BASEMENT.params[k] for k in config.BASEMENT.fitkeys]
    return computer.update_params(np.array(theta, dtype=float))


# --------------------------------------------------------------------------- #
# Per-band rr
# --------------------------------------------------------------------------- #
class TestPerBandRR:
    def test_radius_2_over_radius_1_matches_per_band_rr(
        self, two_band_two_inst_datadir, captured_fluxes_calls, truth
    ):
        captured_fluxes_calls.clear()
        config.init(str(two_band_two_inst_datadir), quiet=True)
        params = _prepared_params()

        # Call flux_subfct_ellc once per inst with a tiny xx to keep ellc fast.
        xx = np.linspace(truth["epoch"] - 0.05, truth["epoch"] + 0.05, 5)
        for inst in ("tess", "kepler"):
            computer.flux_subfct_ellc(params, inst=inst, companion="b", xx=xx)

        {kw["t_obs"][0]: kw for tag, kw in captured_fluxes_calls if tag == "lc"}
        # We made 2 ellc.fluxes calls, one per inst, with identical xx[0] —
        # so re-key by inst via call order.
        inst_calls = [kw for tag, kw in captured_fluxes_calls if tag == "lc"]
        assert len(inst_calls) == 2

        expected_rr = {0: truth["rr_tess"], 1: truth["rr_kepler"]}
        for i, kw in enumerate(inst_calls):
            rr = kw["radius_2"] / kw["radius_1"]
            assert rr == pytest.approx(expected_rr[i], rel=1e-10), (
                f"call {i}: expected rr={expected_rr[i]}, got {rr} "
                f"(radius_1={kw['radius_1']}, radius_2={kw['radius_2']})"
            )


# --------------------------------------------------------------------------- #
# Per-inst LDC assembly from bandpass-scoped scalars
# --------------------------------------------------------------------------- #
class TestPerInstLDC:
    def test_ldc_lists_match_per_band_scalars(
        self, two_band_two_inst_datadir, captured_fluxes_calls
    ):
        captured_fluxes_calls.clear()
        config.init(str(two_band_two_inst_datadir), quiet=True)
        params = _prepared_params()
        xx = np.linspace(2459000.5 - 0.05, 2459000.5 + 0.05, 5)
        for inst in ("tess", "kepler"):
            computer.flux_subfct_ellc(params, inst=inst, companion="b", xx=xx)

        inst_calls = [kw for tag, kw in captured_fluxes_calls if tag == "lc"]
        # ellc receives ldc_1 = the per-inst assembled list, sourced from
        # bandpass-suffixed q1/q2 → converted to u-space.
        ldc_tess = inst_calls[0]["ldc_1"]
        ldc_kepler = inst_calls[1]["ldc_1"]
        # quad law → list of two
        assert len(ldc_tess) == 2
        assert len(ldc_kepler) == 2
        # The scalars are q1=q2=0.5 for both bandpasses (per fixture), so
        # both assembled LDC lists must be identical.
        np.testing.assert_allclose(ldc_tess, ldc_kepler, rtol=0, atol=0)

    def test_shared_bandpass_yields_identical_ldc_across_instruments(
        self, one_band_two_inst_datadir, captured_fluxes_calls
    ):
        captured_fluxes_calls.clear()
        config.init(str(one_band_two_inst_datadir), quiet=True)
        params = _prepared_params()
        xx = np.linspace(2459000.5 - 0.05, 2459000.5 + 0.05, 5)
        for inst in ("tess_pdcsap", "tess_qlp"):
            computer.flux_subfct_ellc(params, inst=inst, companion="b", xx=xx)

        inst_calls = [kw for tag, kw in captured_fluxes_calls if tag == "lc"]
        assert len(inst_calls) == 2
        np.testing.assert_allclose(inst_calls[0]["ldc_1"], inst_calls[1]["ldc_1"], rtol=0, atol=0)


# --------------------------------------------------------------------------- #
# Globally shared orbital params
# --------------------------------------------------------------------------- #
class TestSharedOrbitalParams:
    def test_orbital_kwargs_bit_equal_across_insts(
        self, two_band_two_inst_datadir, captured_fluxes_calls
    ):
        captured_fluxes_calls.clear()
        config.init(str(two_band_two_inst_datadir), quiet=True)
        params = _prepared_params()
        xx = np.linspace(2459000.5 - 0.05, 2459000.5 + 0.05, 5)
        for inst in ("tess", "kepler"):
            computer.flux_subfct_ellc(params, inst=inst, companion="b", xx=xx)

        inst_calls = [kw for tag, kw in captured_fluxes_calls if tag == "lc"]
        for key in SHARED_ORBITAL_KWARGS:
            v_tess = inst_calls[0].get(key)
            v_kepler = inst_calls[1].get(key)
            assert v_tess == v_kepler, (
                f"shared kwarg {key!r} differs across instruments: tess={v_tess}, kepler={v_kepler}"
            )

    def test_rsuma_implied_consistency_per_band(
        self, two_band_two_inst_datadir, captured_fluxes_calls, truth
    ):
        # rsuma is globally shared; per-band rr means radius_1+radius_2 may
        # equal rsuma exactly per band. Pin: radius_1+radius_2 == rsuma for each
        # call (the chromatic invariant from computer.py:538-542).
        captured_fluxes_calls.clear()
        config.init(str(two_band_two_inst_datadir), quiet=True)
        params = _prepared_params()
        xx = np.linspace(2459000.5 - 0.05, 2459000.5 + 0.05, 5)
        for inst in ("tess", "kepler"):
            computer.flux_subfct_ellc(params, inst=inst, companion="b", xx=xx)

        inst_calls = [kw for tag, kw in captured_fluxes_calls if tag == "lc"]
        for kw in inst_calls:
            assert kw["radius_1"] + kw["radius_2"] == pytest.approx(truth["rsuma"], rel=1e-12)


# --------------------------------------------------------------------------- #
# Shared-bandpass q-space LDC propagation
# --------------------------------------------------------------------------- #
class TestSharedBandpassLDCPropagation:
    """Regression: ``bandpass,tess tess`` with q-space ``host_ldc_q1_tess`` /
    ``host_ldc_q2_tess`` must (a) reach ``q_to_u`` correctly and (b) actually
    *change* what is passed to ellc when the value column is edited. The
    user reported show_initial_guess looked unchanged after editing q1; this
    test pins that the LDC pipeline does honor the edit (a visually small
    transit-shape change for a 0.03 q1 delta is expected, but the LDC
    vector passed to ellc must differ)."""

    @staticmethod
    def _build_shared_bp_datadir(tmp_path, name, *, q1, q2):
        from tests.chromatic._helpers import (
            N_POINTS_PER_INST,
            NOISE_SIGMA,
            RNG_SEED,
            TRUE_EPOCH,
            TRUE_PERIOD,
            TRUE_RR_TESS,
            common_orbital_rows,
            dilution_rows,
            err_baseline_rows,
            phase_sampled_time,
            simulate_lightcurve,
            write_data_csv,
            write_params,
            write_settings,
        )

        datadir = tmp_path / name
        datadir.mkdir()
        rng = np.random.default_rng(RNG_SEED + 11)
        for inst in ("tess_pdcsap", "tess_qlp"):
            time = phase_sampled_time(N_POINTS_PER_INST, TRUE_PERIOD, TRUE_EPOCH, rng=rng)
            flux, err = simulate_lightcurve(time, TRUE_RR_TESS, NOISE_SIGMA, rng)
            write_data_csv(datadir / f"{inst}.csv", time, flux, err)
        write_settings(
            datadir,
            inst_phot=["tess_pdcsap", "tess_qlp"],
            bandpass="tess tess",
        )
        rows = (
            [
                {
                    "name": "b_rr_tess",
                    "value": TRUE_RR_TESS,
                    "fit": 1,
                    "bounds": "uniform 0.0 0.3",
                    "label": "rr_tess",
                }
            ]
            + common_orbital_rows()
            + dilution_rows(["tess_pdcsap", "tess_qlp"])
            + err_baseline_rows(["tess_pdcsap", "tess_qlp"])
            + [
                {
                    "name": "host_ldc_q1_tess",
                    "value": q1,
                    "fit": 1,
                    "bounds": "uniform 0 1",
                    "label": "q1_tess",
                },
                {
                    "name": "host_ldc_q2_tess",
                    "value": q2,
                    "fit": 1,
                    "bounds": "uniform 0 1",
                    "label": "q2_tess",
                },
            ]
        )
        write_params(datadir / "params.csv", rows=rows)
        return datadir

    def test_q_space_ldc_reaches_ellc_with_correct_q_to_u_mapping(
        self, tmp_path, captured_fluxes_calls
    ):
        from allesfitter.lightcurves import (
            translate_limb_darkening_from_q_to_u as q_to_u,
        )

        q1, q2 = 0.47, 0.30
        datadir = self._build_shared_bp_datadir(
            tmp_path,
            "shared_bp_q47",
            q1=q1,
            q2=q2,
        )
        # Clear any ellc calls from simulate_lightcurve during datadir construction
        captured_fluxes_calls.clear()
        config.init(str(datadir), quiet=True)
        params = _prepared_params()
        xx = np.linspace(2459000.5 - 0.05, 2459000.5 + 0.05, 5)
        for inst in ("tess_pdcsap", "tess_qlp"):
            computer.flux_subfct_ellc(params, inst=inst, companion="b", xx=xx)

        inst_calls = [kw for tag, kw in captured_fluxes_calls if tag == "lc"]
        assert len(inst_calls) == 2
        expected_u = q_to_u([q1, q2], law="quad")
        for kw in inst_calls:
            np.testing.assert_allclose(kw["ldc_1"], expected_u, rtol=0, atol=1e-12)

    def test_editing_q1_value_changes_ldc_passed_to_ellc(self, tmp_path, captured_fluxes_calls):
        from allesfitter.lightcurves import (
            translate_limb_darkening_from_q_to_u as q_to_u,
        )

        # Run 1: q1=0.50
        datadir_a = self._build_shared_bp_datadir(
            tmp_path,
            "shared_bp_q50",
            q1=0.50,
            q2=0.30,
        )
        captured_fluxes_calls.clear()
        config.init(str(datadir_a), quiet=True)
        params_a = _prepared_params()
        xx = np.linspace(2459000.5 - 0.05, 2459000.5 + 0.05, 5)
        computer.flux_subfct_ellc(params_a, inst="tess_pdcsap", companion="b", xx=xx)
        ldc_a = [kw for tag, kw in captured_fluxes_calls if tag == "lc"][-1]["ldc_1"]

        # Reset BASEMENT and rebuild with q1=0.47
        config.BASEMENT = None
        datadir_b = self._build_shared_bp_datadir(
            tmp_path,
            "shared_bp_q47b",
            q1=0.47,
            q2=0.30,
        )
        captured_fluxes_calls.clear()
        config.init(str(datadir_b), quiet=True)
        params_b = _prepared_params()
        computer.flux_subfct_ellc(params_b, inst="tess_pdcsap", companion="b", xx=xx)
        ldc_b = [kw for tag, kw in captured_fluxes_calls if tag == "lc"][-1]["ldc_1"]

        # Sanity: the two ldc lists must differ — editing the value column
        # in params.csv must actually propagate through to ellc.
        assert not np.allclose(ldc_a, ldc_b), (
            f"editing host_ldc_q1_tess from 0.50 to 0.47 did NOT change the "
            f"LDC passed to ellc: ldc_a={ldc_a}, ldc_b={ldc_b}"
        )
        # And both must match the analytic q_to_u of their respective q1.
        np.testing.assert_allclose(ldc_a, q_to_u([0.50, 0.30], law="quad"), rtol=0, atol=1e-12)
        np.testing.assert_allclose(ldc_b, q_to_u([0.47, 0.30], law="quad"), rtol=0, atol=1e-12)


# --------------------------------------------------------------------------- #
# Achromatic fallback when chromatic key is missing
# --------------------------------------------------------------------------- #
class TestAchromaticFallback:
    def test_missing_chromatic_rr_falls_back_to_unsuffixed(
        self, two_band_two_inst_datadir, captured_fluxes_calls, truth
    ):
        # Remove the chromatic key and inject an achromatic b_rr → the
        # fallback at computer.py:529-530 must kick in and use b_rr.
        captured_fluxes_calls.clear()
        config.init(str(two_band_two_inst_datadir), quiet=True)
        params = _prepared_params()
        params.pop("b_rr_tess", None)
        params["b_rr"] = 0.123
        xx = np.linspace(truth["epoch"] - 0.05, truth["epoch"] + 0.05, 5)
        computer.flux_subfct_ellc(params, inst="tess", companion="b", xx=xx)

        kw = [kw for tag, kw in captured_fluxes_calls if tag == "lc"][0]
        rr = kw["radius_2"] / kw["radius_1"]
        assert rr == pytest.approx(0.123, rel=1e-10)

    def test_update_params_backfills_achromatic_rr_in_chromatic_mode(
        self, two_band_two_inst_datadir
    ):
        # Regression: legacy sites in computer.update_params (host-density
        # prior, phase-curve hack) read params['b_rr'] directly. In chromatic
        # mode this key is never created by the parser → KeyError unless
        # update_params backfills a representative value.
        config.init(str(two_band_two_inst_datadir), quiet=True)
        params = _prepared_params()
        assert "b_rr" in params, (
            "update_params must backfill b_rr in chromatic mode so legacy code paths don't KeyError"
        )
        # The backfilled value should match the first bandpass's rr.
        config.BASEMENT.settings["bandpass"]
        first_inst = config.BASEMENT.settings["inst_phot"][0]
        first_band_key = config.BASEMENT.get_rr_key("b", first_inst)
        assert params["b_rr"] == params[first_band_key], (
            f"backfilled b_rr={params['b_rr']} != {first_band_key}={params[first_band_key]}"
        )
