"""Unit tests for the pure-logic helpers in
:mod:`allesfitter.detection.blind_search`. The end-to-end pipeline (config
init, real TLS run) is covered by the ``slow`` integration test in
``tests/integration/test_transit_search_integration.py``."""

from __future__ import annotations

import csv

import numpy as np
import pytest

from allesfitter.detection.blind_search import (
    _apply_known_masks,
    _companion_impact_parameter,
    _detrend_full_lightcurve,
    _duration_is_trustworthy,
    _harmonic_periods,
    _impact_parameter,
    _known_companion_windows,
    _passes_min_depth,
    _passes_transit_consistency_checks,
    _refine_duration_if_untrustworthy,
    _refit_duration_from_folded_data,
    _resolve_sampler,
    _resolve_transit_template,
    _tls_stellar_kwargs,
    _write_known_recovery_csv,
    _write_summary_csv,
)


class TestResolveSampler:
    def test_ns_results_dir_resolves_to_ns_sampler(self, tmp_path):
        results_dir = tmp_path / "TOI-1234" / "ns_results"
        results_dir.mkdir(parents=True)

        sampler, datadir = _resolve_sampler(str(results_dir))

        assert sampler == "ns"
        assert datadir == (tmp_path / "TOI-1234").resolve()

    def test_mcmc_results_dir_resolves_to_mcmc_sampler(self, tmp_path):
        results_dir = tmp_path / "TOI-1234" / "mcmc_results"
        results_dir.mkdir(parents=True)

        sampler, datadir = _resolve_sampler(str(results_dir))

        assert sampler == "mcmc"
        assert datadir == (tmp_path / "TOI-1234").resolve()

    def test_unrecognized_directory_name_raises(self, tmp_path):
        results_dir = tmp_path / "TOI-1234" / "results"
        results_dir.mkdir(parents=True)

        with pytest.raises(ValueError, match="mcmc_results.*ns_results"):
            _resolve_sampler(str(results_dir))


class TestKnownCompanionWindows:
    def test_builds_one_window_per_transiting_companion(self):
        params_median = {
            "b_period": 3.5,
            "b_epoch": 2457000.0,
            "b_rsuma": 0.15,
            "b_cosi": 0.05,
            "b_rr": 0.1,
            "c_period": 10.0,
            "c_epoch": 2457005.0,
            "c_rsuma": 0.08,
            "c_cosi": 0.02,
            "c_rr": 0.05,
        }

        windows = _known_companion_windows(params_median, ["b", "c"], mask_width_factor=1.5)

        assert {w["companion"] for w in windows} == {"b", "c"}
        b_window = next(w for w in windows if w["companion"] == "b")
        assert b_window["period"] == pytest.approx(3.5)
        assert b_window["epoch"] == pytest.approx(2457000.0)
        assert b_window["duration"] > 0

    def test_mask_width_factor_scales_the_duration_linearly(self):
        params_median = {
            "b_period": 3.5,
            "b_epoch": 2457000.0,
            "b_rsuma": 0.15,
            "b_cosi": 0.05,
            "b_rr": 0.1,
        }

        narrow = _known_companion_windows(params_median, ["b"], mask_width_factor=1.0)[0]
        wide = _known_companion_windows(params_median, ["b"], mask_width_factor=3.0)[0]

        assert wide["duration"] == pytest.approx(3.0 * narrow["duration"])

    def test_skips_companion_missing_required_params(self):
        # No "_rr" for companion b -> geometry can't be resolved.
        params_median = {"b_period": 3.5, "b_epoch": 2457000.0, "b_rsuma": 0.15, "b_cosi": 0.05}

        windows = _known_companion_windows(params_median, ["b"], mask_width_factor=1.5)

        assert windows == []

    def test_skips_companion_with_non_transiting_geometry(self):
        # cosi > 1 is unphysical -> transit_duration_days returns NaN.
        params_median = {
            "b_period": 3.5,
            "b_epoch": 2457000.0,
            "b_rsuma": 0.15,
            "b_cosi": 5.0,
            "b_rr": 0.1,
        }

        windows = _known_companion_windows(params_median, ["b"], mask_width_factor=1.5)

        assert windows == []


class TestHarmonicPeriods:
    def test_includes_integer_ratio_and_multiple_harmonics(self):
        harmonics = _harmonic_periods(12.0, n_harmonics=2)

        assert 6.0 in harmonics  # P/2
        assert 4.0 in harmonics  # P/3
        assert 24.0 in harmonics  # 2P
        assert 36.0 in harmonics  # 3P

    def test_returns_sorted_unique_values(self):
        harmonics = _harmonic_periods(10.0, n_harmonics=3)

        assert harmonics == sorted(set(harmonics))


class TestTlsStellarKwargs:
    class _FakeBase:
        def __init__(self, params_star=None, inst_phot=("tess",)):
            self.params_star = params_star
            self.settings = {"inst_phot": list(inst_phot)}

    def test_returns_empty_when_no_star_params_or_ldc(self):
        base = self._FakeBase(params_star=None, inst_phot=())

        kwargs = _tls_stellar_kwargs(base, {})

        assert kwargs == {}

    def test_includes_stellar_radius_and_mass_bounds(self):
        base = self._FakeBase(
            params_star={
                "R_star_median": 0.79,
                "R_star_lerr": 0.05,
                "R_star_uerr": 0.05,
                "M_star_median": 0.76,
                "M_star_lerr": 0.09,
                "M_star_uerr": 0.09,
            }
        )

        kwargs = _tls_stellar_kwargs(base, {})

        assert kwargs["R_star"] == pytest.approx(0.79)
        assert kwargs["R_star_min"] == pytest.approx(0.79 - 3 * 0.05)
        assert kwargs["R_star_max"] == pytest.approx(0.79 + 3 * 0.05)
        assert kwargs["M_star"] == pytest.approx(0.76)

    def test_converts_ldc_q_to_u_for_first_photometric_instrument(self):
        base = self._FakeBase(params_star=None, inst_phot=("tess",))
        params_median = {"host_ldc_q1_tess": 0.5, "host_ldc_q2_tess": 0.3}

        kwargs = _tls_stellar_kwargs(base, params_median)

        assert "u" in kwargs
        assert len(kwargs["u"]) == 2

    def test_omits_u_when_ldc_missing(self):
        base = self._FakeBase(params_star=None, inst_phot=("tess",))

        kwargs = _tls_stellar_kwargs(base, {})

        assert "u" not in kwargs


class TestDetrendFullLightcurve:
    class _FakeBase:
        def __init__(self, fulldata, inst_phot):
            self.fulldata = fulldata
            self.settings = {"inst_phot": inst_phot}

    def test_concatenates_and_time_sorts_multiple_instruments(self, monkeypatch):
        import allesfitter.detection.blind_search as blind_search

        # Baseline that just returns zeros keeps the test independent of the
        # real GP machinery (covered by the slow integration test instead).
        monkeypatch.setattr(
            blind_search, "calculate_baseline", lambda *a, **kw: np.zeros_like(kw["xx"])
        )

        fulldata = {
            "tess": {
                "time": np.array([2.0, 0.0]),
                "flux": np.array([1.1, 1.0]),
                "err_scales_flux": np.array([1.0, 1.0]),
            },
            "qlp": {
                "time": np.array([1.0]),
                "flux": np.array([0.9]),
                "err_scales_flux": np.array([2.0]),
            },
        }
        base = self._FakeBase(fulldata, ["tess", "qlp"])
        params_median = {"err_flux_tess": 0.01, "err_flux_qlp": 0.02}

        time, flux_raw, flux, flux_err = _detrend_full_lightcurve(base, params_median)

        np.testing.assert_allclose(time, [0.0, 1.0, 2.0])
        np.testing.assert_allclose(flux_raw, [1.0, 0.9, 1.1])
        np.testing.assert_allclose(flux, [1.0, 0.9, 1.1])  # zero baseline -> unchanged
        np.testing.assert_allclose(flux_err, [0.01, 0.04, 0.01])

    def test_rejects_unrecognized_flatten_method(self):
        base = self._FakeBase({"tess": {}}, ["tess"])

        with pytest.raises(ValueError, match="flatten_method"):
            _detrend_full_lightcurve(base, {}, flatten_method="wotan")

    def test_notch_flatten_method_calls_notch_flatten_per_instrument(self, monkeypatch):
        import allesfitter.detection.blind_search as blind_search

        calls = []

        def fake_notch_flatten(time, flux, window_length):
            calls.append((tuple(time), window_length))
            return flux * 0.0 + 1.0, flux  # trivial, deterministic flat/trend

        monkeypatch.setattr(blind_search, "notch_flatten", fake_notch_flatten)

        fulldata = {
            "tess": {
                "time": np.array([0.0, 1.0]),
                "flux": np.array([1.0, 1.1]),
                "err_scales_flux": np.array([1.0, 1.0]),
            }
        }
        base = self._FakeBase(fulldata, ["tess"])
        params_median = {"err_flux_tess": 0.01}

        time, flux_raw, flux, flux_err = _detrend_full_lightcurve(
            base, params_median, flatten_method="notch", flatten_window_length=0.25
        )

        assert calls == [((0.0, 1.0), 0.25)]
        np.testing.assert_allclose(flux, [1.0, 1.0])

    def test_notch_flatten_method_defaults_window_length(self, monkeypatch):
        import allesfitter.detection.blind_search as blind_search

        received = {}

        def fake_notch_flatten(time, flux, window_length):
            received["window_length"] = window_length
            return flux, flux

        monkeypatch.setattr(blind_search, "notch_flatten", fake_notch_flatten)

        fulldata = {
            "tess": {
                "time": np.array([0.0, 1.0]),
                "flux": np.array([1.0, 1.1]),
                "err_scales_flux": np.array([1.0, 1.0]),
            }
        }
        base = self._FakeBase(fulldata, ["tess"])

        _detrend_full_lightcurve(base, {"err_flux_tess": 0.01}, flatten_method="notch")

        assert received["window_length"] == blind_search._DEFAULT_NOTCH_WINDOW_LENGTH

    def test_locor_flatten_method_uses_explicit_period(self, monkeypatch):
        import allesfitter.detection.blind_search as blind_search

        received = {}

        def fake_locor_flatten(time, flux, period):
            received["period"] = period
            return flux, flux

        monkeypatch.setattr(blind_search, "locor_flatten", fake_locor_flatten)

        fulldata = {
            "tess": {
                "time": np.array([0.0, 1.0]),
                "flux": np.array([1.0, 1.1]),
                "err_scales_flux": np.array([1.0, 1.0]),
            }
        }
        base = self._FakeBase(fulldata, ["tess"])

        _detrend_full_lightcurve(
            base, {"err_flux_tess": 0.01}, flatten_method="locor", flatten_window_length=3.3
        )

        assert received["period"] == 3.3

    def test_locor_flatten_method_auto_estimates_period_when_omitted(self, monkeypatch):
        import allesfitter.detection.blind_search as blind_search

        monkeypatch.setattr(blind_search, "_estimate_locor_period", lambda base, quiet: 7.5)
        received = {}

        def fake_locor_flatten(time, flux, period):
            received["period"] = period
            return flux, flux

        monkeypatch.setattr(blind_search, "locor_flatten", fake_locor_flatten)

        fulldata = {
            "tess": {
                "time": np.array([0.0, 1.0]),
                "flux": np.array([1.0, 1.1]),
                "err_scales_flux": np.array([1.0, 1.0]),
            }
        }
        base = self._FakeBase(fulldata, ["tess"])

        _detrend_full_lightcurve(base, {"err_flux_tess": 0.01}, flatten_method="locor")

        assert received["period"] == 7.5


def test_write_summary_csv_writes_header_and_rows(tmp_path):
    path = tmp_path / "candidates_summary.csv"
    summary = [
        {
            "candidate": 1,
            "period": 4.5,
            "epoch": 2459000.1,
            "duration_hours": 2.5,
            "depth_ppm": 1500.0,
            "SDE": 12.3,
            "snr": 8.1,
            "figure": "candidate_1.pdf",
            "h5": "candidate_1_tls.h5",
        }
    ]

    _write_summary_csv(str(path), summary)

    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert rows[0]["candidate"] == "1"
    assert rows[0]["period"] == "4.5"
    assert rows[0]["h5"] == "candidate_1_tls.h5"


def test_write_summary_csv_handles_no_candidates(tmp_path):
    path = tmp_path / "candidates_summary.csv"

    _write_summary_csv(str(path), [])

    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows == []


class TestApplyKnownMasks:
    def test_removes_in_transit_rows_from_every_array(self):
        time = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        flux = np.array([1.0, 0.9, 1.0, 0.9, 1.0])
        flux_err = np.array([0.01, 0.02, 0.03, 0.04, 0.05])
        # period=2, epoch=1, duration=0.5 -> masks t=1 and t=3 (both in-transit).
        windows = [{"period": 2.0, "epoch": 1.0, "duration": 0.5}]

        out_time, out_flux, out_err = _apply_known_masks(time, [flux, flux_err], windows)

        np.testing.assert_allclose(out_time, [0.0, 2.0, 4.0])
        np.testing.assert_allclose(out_flux, [1.0, 1.0, 1.0])
        np.testing.assert_allclose(out_err, [0.01, 0.03, 0.05])

    def test_no_windows_leaves_arrays_unchanged(self):
        time = np.array([0.0, 1.0, 2.0])
        flux = np.array([1.0, 0.95, 1.0])

        out_time, out_flux = _apply_known_masks(time, [flux], [])

        np.testing.assert_allclose(out_time, time)
        np.testing.assert_allclose(out_flux, flux)

    def test_multiple_windows_combine(self):
        time = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        flux = np.arange(6, dtype=float)
        windows = [
            {"period": 100.0, "epoch": 1.0, "duration": 0.5},  # masks only t=1
            {"period": 100.0, "epoch": 4.0, "duration": 0.5},  # masks only t=4
        ]

        out_time, out_flux = _apply_known_masks(time, [flux], windows)

        np.testing.assert_allclose(out_time, [0.0, 2.0, 3.0, 5.0])
        np.testing.assert_allclose(out_flux, [0.0, 2.0, 3.0, 5.0])


def test_write_known_recovery_csv_writes_header_and_rows(tmp_path):
    path = tmp_path / "known_planets_recovery.csv"
    rows = [
        {
            "companion": "b",
            "known_period": 3.5,
            "known_epoch": 2457000.0,
            "recovered_period": 3.5001,
            "recovered_epoch": 2457000.01,
            "SDE": 25.0,
            "snr": 40.0,
            "epoch_match": True,
            "bracket_ignored": False,
            "recovered": True,
            "figure": "known_b_recovery.pdf",
        }
    ]

    _write_known_recovery_csv(str(path), rows)

    with open(path, newline="") as f:
        read_rows = list(csv.DictReader(f))
    assert len(read_rows) == 1
    assert read_rows[0]["companion"] == "b"
    assert read_rows[0]["recovered"] == "True"


def test_write_known_recovery_csv_handles_no_rows(tmp_path):
    path = tmp_path / "known_planets_recovery.csv"

    _write_known_recovery_csv(str(path), [])

    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows == []


def _fake_tls_results(**overrides):
    # A "clean" 5-transit candidate: every epoch well-covered (20 points),
    # precise (small uncertainty), and consistently showing the ~1% dip.
    results = dict(
        depth=0.99,  # mean_depth_frac = 0.01
        distinct_transit_count=5,
        per_transit_count=np.array([20, 20, 20, 20, 20]),
        transit_depths=np.array([0.99, 0.99, 0.99, 0.99, 0.99]),
        transit_depths_uncertainties=np.array([0.0005] * 5),
    )
    results.update(overrides)
    return results


class TestPassesTransitConsistencyChecks:
    def test_passes_a_clean_well_supported_candidate(self):
        ok, reason = _passes_transit_consistency_checks(
            _fake_tls_results(),
            min_distinct_transits=3,
            min_points_per_transit=5,
            consistency_sigma=3.0,
        )
        assert ok is True
        assert reason == ""

    def test_rejects_too_few_distinct_transits(self):
        results = _fake_tls_results(distinct_transit_count=2)

        ok, reason = _passes_transit_consistency_checks(
            results, min_distinct_transits=3, min_points_per_transit=5, consistency_sigma=3.0
        )

        assert ok is False
        assert "distinct transit" in reason

    def test_rejects_well_covered_epoch_with_no_dip(self):
        # Epoch 2 has plenty of points, tiny uncertainty, but flux ~1.0
        # (no dip) even though the candidate's average depth is ~1%.
        results = _fake_tls_results(
            transit_depths=np.array([0.99, 0.99, 1.0, 0.99, 0.99]),
        )

        ok, reason = _passes_transit_consistency_checks(
            results, min_distinct_transits=3, min_points_per_transit=5, consistency_sigma=3.0
        )

        assert ok is False
        assert "shows no dip" in reason

    def test_ignores_poorly_covered_epoch_showing_no_dip(self):
        # Same "no dip" epoch as above, but it only has 2 points -> below
        # min_points_per_transit=5, so it can't be used to rule anything out.
        results = _fake_tls_results(
            per_transit_count=np.array([20, 20, 2, 20, 20]),
            transit_depths=np.array([0.99, 0.99, 1.0, 0.99, 0.99]),
        )

        ok, reason = _passes_transit_consistency_checks(
            results, min_distinct_transits=3, min_points_per_transit=5, consistency_sigma=3.0
        )

        assert ok is True

    def test_ignores_imprecise_epoch_showing_no_dip(self):
        # Well-covered by point count, but its uncertainty is too large to
        # have caught a 1% dip at 3 sigma -> can't be used to rule out.
        results = _fake_tls_results(
            transit_depths=np.array([0.99, 0.99, 1.0, 0.99, 0.99]),
            transit_depths_uncertainties=np.array([0.0005, 0.0005, 0.01, 0.0005, 0.0005]),
        )

        ok, reason = _passes_transit_consistency_checks(
            results, min_distinct_transits=3, min_points_per_transit=5, consistency_sigma=3.0
        )

        assert ok is True

    def test_missing_distinct_transit_count_key_is_skipped_not_crashed(self):
        results = _fake_tls_results()
        del results["distinct_transit_count"]

        ok, reason = _passes_transit_consistency_checks(
            results, min_distinct_transits=3, min_points_per_transit=5, consistency_sigma=3.0
        )

        assert ok is True


class TestPassesMinDepth:
    def test_passes_a_deep_candidate(self):
        ok, reason = _passes_min_depth(_fake_tls_results(), min_depth_ppt=1.0)

        assert ok is True
        assert reason == ""

    def test_rejects_a_shallower_than_threshold_candidate(self):
        # depth=0.9995 -> 0.5 ppt, below the 1.0 ppt default floor.
        results = _fake_tls_results(depth=0.9995)

        ok, reason = _passes_min_depth(results, min_depth_ppt=1.0)

        assert ok is False
        assert "ppt" in reason

    def test_boundary_depth_passes(self):
        # depth=0.999 -> exactly 1.0 ppt: the floor itself should pass.
        results = _fake_tls_results(depth=0.999)

        ok, reason = _passes_min_depth(results, min_depth_ppt=1.0)

        assert ok is True


class TestImpactParameter:
    def test_computes_known_grazing_geometry(self):
        # rsuma=0.15, cosi=0.14, rr=0.05 -> b=0.14*1.05/0.15=0.98 (matches
        # the ellc-injection test used to derive the grazing threshold).
        b = _impact_parameter(rsuma=0.15, cosi=0.14, rr=0.05)

        assert b == pytest.approx(0.98, abs=1e-6)

    def test_computes_known_low_impact_geometry(self):
        b = _impact_parameter(rsuma=0.15, cosi=0.02, rr=0.1)

        assert b == pytest.approx(0.02 * 1.1 / 0.15, abs=1e-9)

    def test_returns_nan_for_missing_input(self):
        assert np.isnan(_impact_parameter(rsuma=None, cosi=0.1, rr=0.05))

    def test_returns_nan_for_non_numeric_input(self):
        assert np.isnan(_impact_parameter(rsuma="oops", cosi=0.1, rr=0.05))

    def test_returns_nan_for_zero_rsuma(self):
        assert np.isnan(_impact_parameter(rsuma=0.0, cosi=0.1, rr=0.05))

    def test_returns_nan_for_negative_rr(self):
        assert np.isnan(_impact_parameter(rsuma=0.15, cosi=0.1, rr=-0.5))


class TestCompanionImpactParameter:
    def test_reads_companion_prefixed_keys(self):
        params_median = {"b_rsuma": 0.15, "b_cosi": 0.14, "b_rr": 0.05}

        b = _companion_impact_parameter(params_median, "b")

        assert b == pytest.approx(0.98, abs=1e-6)

    def test_missing_companion_keys_return_nan(self):
        assert np.isnan(_companion_impact_parameter({}, "c"))


class TestResolveTransitTemplate:
    def test_explicit_choice_passes_through_unchanged(self):
        template, b = _resolve_transit_template("grazing", {}, ["b"])

        assert template == "grazing"
        assert np.isnan(b)

    def test_auto_selects_grazing_for_high_impact_companion(self):
        params_median = {"b_rsuma": 0.15, "b_cosi": 0.14, "b_rr": 0.05}  # b=0.98

        template, b = _resolve_transit_template("auto", params_median, ["b"], for_companion="b")

        assert template == "grazing"
        assert b == pytest.approx(0.98, abs=1e-6)

    def test_auto_selects_default_for_low_impact_companion(self):
        params_median = {"b_rsuma": 0.15, "b_cosi": 0.02, "b_rr": 0.05}  # b~0.14

        template, b = _resolve_transit_template("auto", params_median, ["b"], for_companion="b")

        assert template == "default"
        assert b == pytest.approx(0.02 * 1.05 / 0.15, abs=1e-9)

    def test_auto_falls_back_to_default_when_unresolvable(self):
        template, b = _resolve_transit_template("auto", {}, ["b"], for_companion="b")

        assert template == "default"
        assert np.isnan(b)

    def test_auto_system_level_picks_most_grazing_companion(self):
        params_median = {
            "b_rsuma": 0.15,
            "b_cosi": 0.02,
            "b_rr": 0.05,  # low impact, b~0.14
            "c_rsuma": 0.15,
            "c_cosi": 0.14,
            "c_rr": 0.05,  # high impact, b~0.98
        }

        template, b = _resolve_transit_template("auto", params_median, ["b", "c"])

        assert template == "grazing"
        assert b == pytest.approx(0.98, abs=1e-6)

    def test_auto_system_level_falls_back_to_default_with_no_companions(self):
        template, b = _resolve_transit_template("auto", {}, [])

        assert template == "default"
        assert np.isnan(b)


def _synthetic_folded_data(
    period=4.0, true_duration_days=0.5, depth=0.02, n_points=5000, noise=0.0008, seed=0
):
    """Uniformly-sampled folded phase/flux with a known box dip, for testing
    the fold-based duration re-fit independent of any real TLS run."""
    rng = np.random.default_rng(seed)
    folded_phase = rng.uniform(0, 1, n_points)
    half_phase = (true_duration_days / period) / 2.0
    in_transit = np.abs(folded_phase - 0.5) < half_phase
    folded_y = np.ones(n_points)
    folded_y[in_transit] -= depth
    folded_y = folded_y + rng.normal(0, noise, n_points)
    return folded_phase, folded_y


class TestDurationIsTrustworthy:
    def test_true_when_enough_points_support_the_window(self):
        folded_phase, folded_y = _synthetic_folded_data()
        results = {
            "correct_duration": 0.5,
            "period": 4.0,
            "folded_phase": folded_phase,
        }

        assert _duration_is_trustworthy(results, min_points_in_window=20) is True

    def test_false_when_window_has_too_few_points(self):
        # Only 2 points anywhere near phase 0.5, everything else far away.
        folded_phase = np.concatenate([np.array([0.499, 0.501]), np.linspace(0.0, 0.4, 500)])
        results = {"correct_duration": 0.1, "period": 4.0, "folded_phase": folded_phase}

        assert _duration_is_trustworthy(results, min_points_in_window=20) is False

    def test_false_when_duration_missing_or_invalid(self):
        folded_phase, _ = _synthetic_folded_data()
        assert _duration_is_trustworthy({"period": 4.0, "folded_phase": folded_phase}, 20) is False
        assert (
            _duration_is_trustworthy(
                {"correct_duration": -1.0, "period": 4.0, "folded_phase": folded_phase}, 20
            )
            is False
        )


class TestRefitDurationFromFoldedData:
    def test_recovers_known_duration_from_a_clean_dip(self):
        folded_phase, folded_y = _synthetic_folded_data(
            period=4.0, true_duration_days=0.5, depth=0.02, n_points=6000, noise=0.0006
        )

        refit_days = _refit_duration_from_folded_data(folded_phase, folded_y, period=4.0)

        assert refit_days is not None
        assert refit_days == pytest.approx(0.5, rel=0.15)

    def test_returns_none_when_no_dip_is_present(self):
        rng = np.random.default_rng(1)
        folded_phase = rng.uniform(0, 1, 2000)
        folded_y = np.ones(2000) + rng.normal(0, 0.0005, 2000)

        assert _refit_duration_from_folded_data(folded_phase, folded_y, period=4.0) is None

    def test_returns_none_with_too_little_baseline_data(self):
        folded_phase = np.array([0.5, 0.501, 0.499])
        folded_y = np.array([0.98, 0.98, 0.98])

        assert _refit_duration_from_folded_data(folded_phase, folded_y, period=4.0) is None

    def test_returns_none_for_empty_input(self):
        assert _refit_duration_from_folded_data(np.array([]), np.array([]), period=4.0) is None


class TestRefineDurationIfUntrustworthy:
    def test_leaves_trustworthy_duration_untouched(self):
        folded_phase, folded_y = _synthetic_folded_data(
            period=4.0, true_duration_days=0.5, n_points=6000
        )
        results = {
            "correct_duration": 0.5,
            "duration": 0.5,
            "period": 4.0,
            "folded_phase": folded_phase,
            "folded_y": folded_y,
        }

        out = _refine_duration_if_untrustworthy(results)

        assert out["correct_duration"] == 0.5
        assert out["duration_refit_from_fold"] is False

    def test_replaces_untrustworthy_duration_with_fold_refit(self):
        # TLS's own (bad) duration is a tiny window with almost no support,
        # even though the folded data has a clear, well-supported 0.5 d dip.
        folded_phase, folded_y = _synthetic_folded_data(
            period=4.0, true_duration_days=0.5, depth=0.02, n_points=6000, noise=0.0006
        )
        results = {
            "correct_duration": 0.002,  # ~3 min -> far too few points in window
            "duration": 0.002,
            "period": 4.0,
            "folded_phase": folded_phase,
            "folded_y": folded_y,
        }

        out = _refine_duration_if_untrustworthy(results)

        assert out["duration_refit_from_fold"] is True
        assert out["correct_duration"] == pytest.approx(0.5, rel=0.15)

    def test_replaces_a_narrow_but_technically_trustworthy_duration_when_refit_is_much_wider(
        self,
    ):
        # Reproduces the real TOI-179 b case: TLS's claimed duration has
        # *some* nearby points (passes the naive point-count trust check)
        # because individual transit epochs are densely sampled internally,
        # but it's still much narrower than the true, well-supported dip.
        folded_phase, folded_y = _synthetic_folded_data(
            period=4.0, true_duration_days=0.5, depth=0.02, n_points=6000, noise=0.0006
        )
        narrow_but_populated = 0.15  # < true 0.5 d, but still inside the dense dip region
        results = {
            "correct_duration": narrow_but_populated,
            "duration": narrow_but_populated,
            "period": 4.0,
            "folded_phase": folded_phase,
            "folded_y": folded_y,
        }
        assert _duration_is_trustworthy(results, 20) is True  # confirms the setup is realistic

        out = _refine_duration_if_untrustworthy(results)

        assert out["duration_refit_from_fold"] is True
        assert out["correct_duration"] == pytest.approx(0.5, rel=0.15)

    def test_keeps_trustworthy_duration_close_to_the_refit(self):
        # Claimed duration already matches the fold-refit closely -> not
        # "substantially wider", so the original (trustworthy) value stays.
        folded_phase, folded_y = _synthetic_folded_data(
            period=4.0, true_duration_days=0.5, depth=0.02, n_points=6000, noise=0.0006
        )
        results = {
            "correct_duration": 0.45,
            "duration": 0.45,
            "period": 4.0,
            "folded_phase": folded_phase,
            "folded_y": folded_y,
        }

        out = _refine_duration_if_untrustworthy(results)

        assert out["duration_refit_from_fold"] is False
        assert out["correct_duration"] == 0.45

    def test_keeps_original_when_untrustworthy_and_refit_also_fails(self):
        rng = np.random.default_rng(2)
        folded_phase = rng.uniform(0, 1, 2000)
        folded_y = np.ones(2000) + rng.normal(0, 0.0005, 2000)  # no dip anywhere
        results = {
            "correct_duration": 0.002,
            "duration": 0.002,
            "period": 4.0,
            "folded_phase": folded_phase,
            "folded_y": folded_y,
        }

        out = _refine_duration_if_untrustworthy(results)

        assert out["duration_refit_from_fold"] is False
        assert out["correct_duration"] == 0.002
