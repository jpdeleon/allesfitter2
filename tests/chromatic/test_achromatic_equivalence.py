"""Validate that chromatic mode reduces *exactly* to achromatic mode when
only a single unique bandpass is configured.

The chromatic refactor introduced per-bandpass `Rp/Rs` and per-bandpass LDC
keys, with a `get_rr_key` / `get_ldc_key` indirection that resolves to
either the bandpass-suffix or the inst-suffix depending on whether
``bandpass`` is set in settings.csv. The contract we pin here:

  1. Configuring `bandpass,<single label>` (chromatic infrastructure on
     but `chromatic=False`) must produce a flux model **bit-identical** to
     the equivalent achromatic config with no `bandpass` row at all.
  2. Shared internal state (orbital params, host_ldc_<inst>, host_incl,
     companion radius_1/radius_2, etc.) must be numerically identical for
     the two paths, modulo expected key-shape differences (e.g. the
     achromatic path has `b_rr`; the chromatic-single-band path has
     `b_rr_tess` AND a backfilled `b_rr`).
  3. The achromatic baseline still loads, fits, and behaves like
     pre-chromatic master — no accidental chromatic-mode artifacts.

Together with the existing scope / parser / likelihood / e2e tests, this
locks the equivalence: any future refactor that drifts the chromatic
single-band path off the achromatic baseline will fail here.
"""

from __future__ import annotations

import numpy as np
import pytest

from allesfitter import computer, config

from tests.chromatic._helpers import (
    NOISE_SIGMA, N_POINTS_PER_INST, RNG_SEED, TRUE_COSI, TRUE_EPOCH,
    TRUE_PERIOD, TRUE_RR_TESS, TRUE_RSUMA, common_orbital_rows, dilution_rows,
    err_baseline_rows, ldc_rows, phase_sampled_time, simulate_lightcurve,
    write_data_csv, write_params, write_settings,
)


# Orbital + per-inst keys that must be numerically equal in both modes.
SHARED_KEYS_NUMERIC = (
    "b_rsuma", "b_cosi", "b_epoch", "b_period", "b_f_c", "b_f_s",
    "b_incl", "b_radius_1",
    "host_ldc_tess",        # assembled per-inst LDC list (from q-space scalars)
    "host_lambda", "host_vsini",
    "b_lambda", "b_vsini",
    "host_rotfac_tess", "b_rotfac_tess",
    "host_hf_tess", "b_hf_tess",
)


@pytest.fixture
def synthetic_truth_data(tmp_path):
    """Generate one synthetic lightcurve at TRUE values; both configs share
    the same data so any difference must come from the config path."""
    rng = np.random.default_rng(RNG_SEED + 200)
    time = phase_sampled_time(N_POINTS_PER_INST, TRUE_PERIOD, TRUE_EPOCH, rng=rng)
    flux, err = simulate_lightcurve(time, TRUE_RR_TESS, NOISE_SIGMA, rng)
    return time, flux, err


def _build_achromatic(tmp_path, time, flux, err):
    datadir = tmp_path / "achromatic"
    datadir.mkdir()
    write_data_csv(datadir / "tess.csv", time, flux, err)
    write_settings(datadir, inst_phot=["tess"], bandpass=None)
    rows = (
        [{"name": "b_rr", "value": TRUE_RR_TESS, "fit": 1,
          "bounds": "uniform 0 0.3", "label": "rr"}]
        + common_orbital_rows()
        + dilution_rows(["tess"])
        + err_baseline_rows(["tess"])
        + ldc_rows("tess")
    )
    write_params(datadir / "params.csv", rows=rows)
    return datadir


def _build_chromatic_single_band(tmp_path, time, flux, err):
    datadir = tmp_path / "chromatic_single_band"
    datadir.mkdir()
    write_data_csv(datadir / "tess.csv", time, flux, err)
    # bandpass='tess' (a single label) → infrastructure active, chromatic=False
    write_settings(datadir, inst_phot=["tess"], bandpass="tess")
    rows = (
        # Per-bandpass rr (the chromatic shape), same value as achromatic.
        [{"name": "b_rr_tess", "value": TRUE_RR_TESS, "fit": 1,
          "bounds": "uniform 0 0.3", "label": "rr_tess"}]
        + common_orbital_rows()
        + dilution_rows(["tess"])
        + err_baseline_rows(["tess"])
        + ldc_rows("tess")
    )
    write_params(datadir / "params.csv", rows=rows)
    return datadir


def _prepared_params():
    theta = np.array([config.BASEMENT.params[k] for k in config.BASEMENT.fitkeys],
                     dtype=float)
    return computer.update_params(theta)


class TestSingleBandEqualsAchromatic:
    """A single-band chromatic config must reduce *exactly* to the
    achromatic config — same params, same flux model."""

    def test_flux_subfct_ellc_bit_identical(self, tmp_path, synthetic_truth_data):
        time, flux, err = synthetic_truth_data

        # Achromatic path
        d_a = _build_achromatic(tmp_path, time, flux, err)
        config.init(str(d_a), quiet=True)
        params_a = _prepared_params()
        xx = np.linspace(TRUE_EPOCH - 0.05, TRUE_EPOCH + 0.05, 201)
        flux_a = computer.flux_subfct_ellc(params_a, inst="tess", companion="b",
                                            xx=xx)

        # Chromatic-single-band path
        config.BASEMENT = None
        d_c = _build_chromatic_single_band(tmp_path, time, flux, err)
        config.init(str(d_c), quiet=True)
        params_c = _prepared_params()
        flux_c = computer.flux_subfct_ellc(params_c, inst="tess", companion="b",
                                            xx=xx)

        # Bit-identical: same call to ellc with the same numeric inputs.
        np.testing.assert_array_equal(flux_a, flux_c)

    def test_shared_params_numerically_equal(self, tmp_path, synthetic_truth_data):
        time, flux, err = synthetic_truth_data

        d_a = _build_achromatic(tmp_path, time, flux, err)
        config.init(str(d_a), quiet=True)
        params_a = _prepared_params()

        config.BASEMENT = None
        d_c = _build_chromatic_single_band(tmp_path, time, flux, err)
        config.init(str(d_c), quiet=True)
        params_c = _prepared_params()

        for key in SHARED_KEYS_NUMERIC:
            va, vc = params_a.get(key), params_c.get(key)
            if isinstance(va, list):
                np.testing.assert_array_equal(
                    va, vc, err_msg=f"shared key {key!r} differs"
                )
            else:
                assert va == vc, (
                    f"shared key {key!r}: achromatic={va!r}, "
                    f"chromatic-single-band={vc!r}"
                )

    def test_chromatic_single_band_backfills_achromatic_rr(
        self, tmp_path, synthetic_truth_data
    ):
        # The chromatic path stores `b_rr_tess` but also backfills `b_rr`
        # so legacy code paths (host-density prior, phase-curve hack) still
        # work. Both must equal the injected TRUE_RR_TESS.
        time, flux, err = synthetic_truth_data
        d_c = _build_chromatic_single_band(tmp_path, time, flux, err)
        config.init(str(d_c), quiet=True)
        params_c = _prepared_params()
        assert params_c["b_rr_tess"] == TRUE_RR_TESS
        assert params_c["b_rr"] == TRUE_RR_TESS

    def test_get_rr_key_returns_bandpass_suffix_even_when_chromatic_false(
        self, tmp_path, synthetic_truth_data
    ):
        # Document the three-state edge: chromatic=False (1 unique band)
        # but get_rr_key still returns the bandpass-suffixed key because
        # the parser stores rr under that key. The fallback at
        # computer.py:535-537 makes this transparent to flux_subfct_ellc.
        time, flux, err = synthetic_truth_data
        d_c = _build_chromatic_single_band(tmp_path, time, flux, err)
        config.init(str(d_c), quiet=True)
        assert config.BASEMENT.settings["chromatic"] is False
        assert config.BASEMENT.get_rr_key("b", "tess") == "b_rr_tess"


class TestPerInstScopePreserved:
    """Per-instrument parameters (dil, baseline, error) keep the inst
    suffix in both modes. They are NOT affected by the chromatic refactor."""

    def test_per_inst_keys_present_in_both_modes(self, tmp_path,
                                                  synthetic_truth_data):
        time, flux, err = synthetic_truth_data
        for builder in (_build_achromatic, _build_chromatic_single_band):
            config.BASEMENT = None
            d = builder(tmp_path / builder.__name__, time, flux, err) \
                if False else builder(tmp_path, time, flux, err)
            config.init(str(d), quiet=True)
            b = config.BASEMENT
            assert "dil_tess" in b.params
            assert "ln_err_flux_tess" in b.params
            assert "baseline_offset_flux_tess" in b.params


class TestGloballySharedOrbitalsNeverDuplicate:
    """The achromatic path has always kept orbital params single-keyed.
    The chromatic single-band path must not accidentally introduce a
    bandpass-suffixed duplicate (would silently shadow the global scope)."""

    def test_no_bandpass_suffixed_orbital_keys(self, tmp_path,
                                                synthetic_truth_data):
        time, flux, err = synthetic_truth_data
        d_c = _build_chromatic_single_band(tmp_path, time, flux, err)
        config.init(str(d_c), quiet=True)
        b = config.BASEMENT
        for fam in ("b_period_", "b_epoch_", "b_cosi_", "b_rsuma_",
                    "b_f_c_", "b_f_s_", "b_K_"):
            for key in b.params:
                assert not key.startswith(fam), (
                    f"unexpected per-bandpass leak of orbital param: {key!r}"
                )


class TestAchromaticBackwardCompat:
    """Sanity baseline: the achromatic config still loads with the exact
    pre-chromatic key shape — single b_rr, no `bandpass` setting, no
    per-band rr."""

    def test_achromatic_shape_unchanged(self, tmp_path, synthetic_truth_data):
        time, flux, err = synthetic_truth_data
        d_a = _build_achromatic(tmp_path, time, flux, err)
        config.init(str(d_a), quiet=True)
        b = config.BASEMENT
        assert b.settings["bandpass"] == {}
        assert b.settings["chromatic"] is False
        assert "b_rr" in b.params
        assert "host_ldc_q1_tess" in b.params
        # No chromatic-style keys leak in.
        chromatic_keys = [k for k in b.params if k.startswith("b_rr_")]
        assert chromatic_keys == [], (
            f"achromatic config leaked chromatic keys: {chromatic_keys}"
        )

    def test_achromatic_flux_model_invariant(self, tmp_path,
                                              synthetic_truth_data):
        # Run the achromatic flux model twice and assert reproducibility;
        # a guard against any global state mutation between calls.
        time, flux, err = synthetic_truth_data
        d_a = _build_achromatic(tmp_path, time, flux, err)
        xx = np.linspace(TRUE_EPOCH - 0.05, TRUE_EPOCH + 0.05, 201)

        config.init(str(d_a), quiet=True)
        params_1 = _prepared_params()
        flux_1 = computer.flux_subfct_ellc(params_1, inst="tess", companion="b",
                                            xx=xx)

        config.BASEMENT = None
        config.init(str(d_a), quiet=True)
        params_2 = _prepared_params()
        flux_2 = computer.flux_subfct_ellc(params_2, inst="tess", companion="b",
                                            xx=xx)

        np.testing.assert_array_equal(flux_1, flux_2)
