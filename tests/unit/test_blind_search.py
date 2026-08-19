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
    _detrend_full_lightcurve,
    _harmonic_periods,
    _known_companion_windows,
    _passes_transit_consistency_checks,
    _resolve_sampler,
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
