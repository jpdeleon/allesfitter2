"""Runtime physics-invariant tests for computer.py.

Verifies that ``update_params``, ``calculate_model`` / ``flux_fct``, and
``calculate_lnlike_total`` satisfy geometric, physical, and numerical
invariants that config-time validation cannot reach.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.chromatic._helpers import (
    NOISE_SIGMA,
    TRUE_EPOCH,
    TRUE_PERIOD,
    TRUE_RR_TESS,
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

RNG_SEED = 20260526


@pytest.fixture(autouse=True)
def reset_basement():
    from allesfitter import config

    config.BASEMENT = None
    yield
    config.BASEMENT = None


# ============================================================================
# update_params physics invariants
# ============================================================================
class TestUpdateParamsPhysicsInvariants:
    """update_params is the central parameter propagator — verify its output
    dict satisfies every geometric invariant that the likelihood evaluation
    depends on."""

    @pytest.fixture
    def datadir(self, tmp_path):
        d = tmp_path / "update_params_invariants"
        d.mkdir()
        rng = np.random.default_rng(RNG_SEED)
        time = phase_sampled_time(100, TRUE_PERIOD, TRUE_EPOCH, rng=rng)
        flux, err = simulate_lightcurve(time, TRUE_RR_TESS, NOISE_SIGMA, rng)
        write_data_csv(d / "tess.csv", time, flux, err)
        write_settings(d, inst_phot=["tess"], bandpass=None)
        rows = (
            [
                {
                    "name": "b_rr",
                    "value": TRUE_RR_TESS,
                    "fit": 1,
                    "bounds": "uniform 0 0.3",
                    "label": "rr",
                },
            ]
            + common_orbital_rows()
            + dilution_rows(["tess"])
            + err_baseline_rows(["tess"])
            + ldc_rows("tess")
        )
        write_params(d / "params.csv", rows=rows)
        return d

    def _get_params(self, datadir):
        from allesfitter import computer, config

        config.init(str(datadir), quiet=True)
        theta = np.array([config.BASEMENT.params[k] for k in config.BASEMENT.fitkeys], dtype=float)
        return computer.update_params(theta)

    def test_rsuma_preserved(self, datadir):
        """radius_1 == rsuma / (1 + rr)"""
        params = self._get_params(datadir)
        c = "b"
        r1 = params[c + "_radius_1"]
        rsuma = params[c + "_rsuma"]
        rr = params.get(c + "_rr", params.get(c + "_rr_tess"))
        assert r1 is not None and rsuma is not None and rr is not None
        assert r1 > 0
        expected = rsuma / (1.0 + rr)
        assert r1 == pytest.approx(expected, rel=1e-12)

    def test_rr_preserved(self, datadir):
        """rr is stored and positive"""
        params = self._get_params(datadir)
        c = "b"
        r1 = params[c + "_radius_1"]
        rr = params.get(c + "_rr", params.get(c + "_rr_tess"))
        assert r1 is not None and rr is not None
        assert r1 > 0 and rr > 0

    def test_incl_from_cosi(self, datadir):
        """incl = arccos(cosi) * 180/pi"""
        params = self._get_params(datadir)
        c = "b"
        incl = params[c + "_incl"]
        cosi = params[c + "_cosi"]
        assert incl is not None and cosi is not None
        expected = np.degrees(np.arccos(cosi))
        assert incl == pytest.approx(expected, rel=1e-12)

    def test_ecc_from_fc_fs(self, datadir):
        """ecc = f_s^2 + f_c^2"""
        params = self._get_params(datadir)
        c = "b"
        ecc = params[c + "_ecc"]
        fc = params[c + "_f_c"]
        fs = params[c + "_f_s"]
        if fc is not None and fs is not None:
            expected = fc**2 + fs**2
            assert ecc == pytest.approx(expected, rel=1e-12)
        assert ecc is None or ecc < 1.0

    def test_semi_major_axis_positive(self, datadir):
        """a > 0 when computed (may be None without params_star.csv)"""
        params = self._get_params(datadir)
        c = "b"
        a = params[c + "_a"]
        if a is not None:
            assert a > 0

    def test_period_preserved(self, datadir):
        """period is passed through unchanged"""
        params = self._get_params(datadir)
        c = "b"
        p = params[c + "_period"]
        assert p == pytest.approx(TRUE_PERIOD, rel=1e-12)

    def test_err_jitter_positivity(self, datadir):
        """ln_err is exponentiated to err, so err > 0"""
        params = self._get_params(datadir)
        for inst in ("tess",):
            err_key = "err_flux_" + inst
            if err_key in params and params[err_key] is not None:
                assert params[err_key] > 0

    def test_ldc_quadratic_length(self, datadir):
        """quadratic LDC produces a list of length 2"""

        params = self._get_params(datadir)
        for inst in ("tess",):
            ldc = params.get("host_ldc_" + inst)
            if ldc is not None:
                assert len(ldc) == 2
                assert all(np.isfinite(v) for v in ldc)

    def test_radius_1_positive(self, datadir):
        """radius_1 > 0 (fractional radius of the star, units of a)"""
        params = self._get_params(datadir)
        c = "b"
        r1 = params[c + "_radius_1"]
        assert r1 is not None and r1 > 0

    def test_rsuma_between_zero_and_one(self, datadir):
        """rsuma (aka (R1+R2)/a) ∈ (0, 1]"""
        params = self._get_params(datadir)
        c = "b"
        rsuma = params[c + "_rsuma"]
        assert rsuma is not None
        assert 0 < rsuma <= 1.0

    def test_all_keys_no_nan(self, datadir):
        """All numeric entries in the params dict are finite"""
        params = self._get_params(datadir)
        for k, v in params.items():
            if isinstance(v, (float, np.floating)):
                assert np.isfinite(v), f"non-finite {k}={v}"
            elif isinstance(v, np.ndarray):
                assert np.all(np.isfinite(v)), f"non-finite entries in {k}"


# ============================================================================
# Flux model physical bounds
# ============================================================================
class TestFluxModelPhysicalBounds:
    """The computed flux model must stay within physically plausible bounds
    and respond correctly to parameter changes like dilution."""

    @pytest.fixture
    def datadir(self, tmp_path):
        d = tmp_path / "flux_bounds"
        d.mkdir()
        rng = np.random.default_rng(RNG_SEED + 1)
        time = phase_sampled_time(500, TRUE_PERIOD, TRUE_EPOCH, rng=rng)
        flux, err = simulate_lightcurve(time, TRUE_RR_TESS, NOISE_SIGMA, rng)
        write_data_csv(d / "tess.csv", time, flux, err)
        write_settings(
            d,
            inst_phot=["tess"],
            bandpass=None,
            extra=["use_host_density_prior,True"],
        )
        rows = (
            [
                {
                    "name": "b_rr",
                    "value": TRUE_RR_TESS,
                    "fit": 1,
                    "bounds": "uniform 0.05 0.15",
                    "label": "rr",
                },
            ]
            + common_orbital_rows(fit_orbital=True)
            + dilution_rows(["tess"])
            + err_baseline_rows(["tess"])
            + ldc_rows("tess")
        )
        write_params(d / "params.csv", rows=rows)
        # Provide stellar parameters so update_params computes a > 0.
        star_csv = (
            "R_star,R_star_lerr,R_star_uerr,M_star,M_star_lerr,M_star_uerr\n"
            "1.0,0.05,0.05,1.0,0.05,0.05\n"
        )
        (d / "params_star.csv").write_text(star_csv)
        return d

    @pytest.fixture
    def params_and_time(self, datadir):
        from allesfitter import computer, config

        config.init(str(datadir), quiet=True)
        theta = np.array([config.BASEMENT.params[k] for k in config.BASEMENT.fitkeys], dtype=float)
        params = computer.update_params(theta)
        t = phase_sampled_time(
            2000, TRUE_PERIOD, TRUE_EPOCH, rng=np.random.default_rng(RNG_SEED + 2)
        )
        return params, t

    def test_flux_stays_positive(self, params_and_time):
        """model_flux > 0.5 everywhere — never drops below 50% baseline"""
        params, t = params_and_time
        from allesfitter import computer

        flux = computer.flux_fct(params, inst="tess", companion="b", xx=t)
        assert np.all(flux > 0.5), f"flux minimum = {flux.min()}"

    def test_flux_stays_below_one_point_five(self, params_and_time):
        """model_flux < 1.5 everywhere — never exceeds 150% baseline"""
        params, t = params_and_time
        from allesfitter import computer

        flux = computer.flux_fct(params, inst="tess", companion="b", xx=t)
        assert np.all(flux < 1.5), f"flux maximum = {flux.max()}"

    def test_out_of_transit_flux_near_unity(self, params_and_time):
        """Flux is ≈1.0 well outside transit (±0.5% tolerance)"""
        params, t = params_and_time
        from allesfitter import computer

        duration = 0.1 * TRUE_PERIOD
        phase = (t - TRUE_EPOCH) / TRUE_PERIOD
        phase = phase - np.round(phase)
        oot = np.abs(phase) > duration / TRUE_PERIOD
        if oot.sum() < 10:
            pytest.skip("too few out-of-transit points")
        flux = computer.flux_fct(params, inst="tess", companion="b", xx=t)
        assert np.nanmean(flux[oot]) == pytest.approx(1.0, abs=0.005)

    def test_dilution_reduces_transit_depth(self, params_and_time):
        """Setting dil=0.5 reduces the transit depth vs dil=0.0"""
        params, t = params_and_time
        from allesfitter import computer

        params_no_dil = dict(params)
        params_no_dil["dil_tess"] = 0.0
        flux_no_dil = computer.flux_fct(params_no_dil, inst="tess", companion="b", xx=t)

        params_dil = dict(params)
        params_dil["dil_tess"] = 0.5
        flux_dil = computer.flux_fct(params_dil, inst="tess", companion="b", xx=t)

        depth_no_dil = 1.0 - flux_no_dil.min()
        depth_dil = 1.0 - flux_dil.min()
        assert depth_dil < depth_no_dil, (
            f"dil=0.5 transit depth ({depth_dil:.6f}) should be shallower "
            f"than dil=0.0 depth ({depth_no_dil:.6f})"
        )

    def test_transit_depth_consistent_with_rr(self, params_and_time):
        """Transit depth scales approximately as rr^2 for small planets

        Strong limb darkening can amplify the depth up to ~30% above the
        geometric rr² (the uniform-disk value). This test uses a relaxed
        50% relative bound to catch gross errors while acknowledging LD."""
        params, t = params_and_time
        from allesfitter import computer

        flux = computer.flux_fct(params, inst="tess", companion="b", xx=t)
        depth = 1.0 - flux.min()
        rr = params.get("b_rr", params.get("b_rr_tess"))
        assert depth == pytest.approx(rr**2, rel=0.50), f"depth={depth:.6f}, rr^2={rr**2:.6f}"

    def test_flux_finite(self, params_and_time):
        """Flux contains no NaN or Inf values"""
        params, t = params_and_time
        from allesfitter import computer

        flux = computer.flux_fct(params, inst="tess", companion="b", xx=t)
        assert np.all(np.isfinite(flux)), f"{np.isnan(flux).sum()} NaN, {np.isinf(flux).sum()} Inf"

    def test_flux_mean_near_unity(self, params_and_time):
        """Mean flux over a full orbit is within 1% of unity"""
        params, t = params_and_time
        from allesfitter import computer

        flux = computer.flux_fct(params, inst="tess", companion="b", xx=t)
        assert np.nanmean(flux) == pytest.approx(1.0, abs=0.01)


# ============================================================================
# Log-likelihood numerical correctness
# ============================================================================
class TestLnlikeTotalNumericalCorrectness:
    """For a trivial case (no baseline, no GP, known noise) the log-likelihood
    must match the hand-computed Gaussian chi-squared value."""

    @pytest.fixture
    def datadir(self, tmp_path):
        d = tmp_path / "lnlike_correctness"
        d.mkdir()
        rng = np.random.default_rng(RNG_SEED + 3)
        time = phase_sampled_time(100, TRUE_PERIOD, TRUE_EPOCH, rng=rng)
        flux, err = simulate_lightcurve(time, TRUE_RR_TESS, NOISE_SIGMA, rng)
        write_data_csv(d / "tess.csv", time, flux, err)
        write_settings(
            d,
            inst_phot=["tess"],
            bandpass=None,
            extra=[
                "error_flux_tess,sample",
                "baseline_flux_tess,sample_offset",
            ],
        )
        rows = (
            [
                {
                    "name": "b_rr",
                    "value": TRUE_RR_TESS,
                    "fit": 0,
                    "bounds": "uniform 0.05 0.15",
                    "label": "rr",
                },
            ]
            + common_orbital_rows(fit_orbital=False)
            + dilution_rows(["tess"])
            + err_baseline_rows(["tess"])
            + ldc_rows("tess")
        )
        write_params(d / "params.csv", rows=rows)
        return d

    def test_lnlike_matches_hand_computed(self, datadir):
        from allesfitter import computer, config

        config.init(str(datadir), quiet=True)

        # Use the truth theta (all params fixed, so theta_0 = truth).
        theta = np.array(config.BASEMENT.theta_0, dtype=float)
        params = computer.update_params(theta)

        # Compute model and residuals.
        inst = "tess"
        model = computer.calculate_model(params, inst, "flux")
        data = config.BASEMENT.data[inst]["flux"]
        residuals = data - model

        # The baseline offset is fitted analytically — we need the baseline
        # contribution to include it in the hand computation. For a simple
        # "sample_offset" baseline this is params["baseline_offset_flux_tess"].
        offset = params.get("baseline_offset_flux_" + inst, 0.0)
        residuals -= offset

        # Expected chi-squared.
        N = len(residuals)
        yerr_w = computer.calculate_yerr_w(params, inst, "flux")
        expected_chisq = np.sum((residuals / yerr_w) ** 2)
        expected_lnlike = (
            -0.5 * expected_chisq - 0.5 * N * np.log(2 * np.pi) - np.sum(np.log(yerr_w))
        )

        lnlike = computer.calculate_lnlike_total(params)
        assert np.isfinite(lnlike), "lnlike is not finite"
        assert lnlike == pytest.approx(
            expected_lnlike, rel=1e-10
        ), f"lnlike={lnlike}, expected={expected_lnlike}"
