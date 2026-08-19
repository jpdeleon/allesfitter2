"""Unit tests for the pure-logic helpers in
:mod:`allesfitter.detection.blind_search`. The end-to-end pipeline (config
init, real TLS run) is covered by the ``slow`` integration test in
``tests/integration/test_transit_search_integration.py``."""

from __future__ import annotations

import csv

import numpy as np
import pytest

from allesfitter.detection.blind_search import (
    _detrend_full_lightcurve,
    _harmonic_periods,
    _known_companion_windows,
    _resolve_sampler,
    _tls_stellar_kwargs,
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
