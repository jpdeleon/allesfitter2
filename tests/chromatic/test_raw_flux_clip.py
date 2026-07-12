"""Regression tests for flux_min_raw / flux_max_raw clipping.

The clip is applied in ``basement.load_data``. Clipped rows must:
- be removed from ``data[inst]['time'/'flux']`` (i.e., not seen by the fit), and
- be preserved in ``data[inst]['raw_clipped_time'/'raw_clipped_flux']`` so
  that ``initial_guess.pdf`` can overlay them in red.

These tests pin both halves of the contract.
"""

from __future__ import annotations

import numpy as np
import pytest

from allesfitter import config
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
    ldc_rows,
    phase_sampled_time,
    simulate_lightcurve,
    write_data_csv,
    write_params,
    write_settings,
)


def _achromatic_datadir_with_outliers(tmp_path, settings_extra):
    """Achromatic datadir whose tess.csv contains 4 injected out-of-range points
    (2 below 0.85, 2 above 1.15) on top of the usual transit signal."""
    datadir = tmp_path / "raw_clip_case"
    datadir.mkdir()
    rng = np.random.default_rng(RNG_SEED + 7)
    time = phase_sampled_time(N_POINTS_PER_INST, TRUE_PERIOD, TRUE_EPOCH, rng=rng)
    flux, err = simulate_lightcurve(time, TRUE_RR_TESS, NOISE_SIGMA, rng)
    # Inject 4 obvious outliers (2 below the floor, 2 above the ceiling).
    flux = flux.copy()
    outlier_idx = [10, 50, 90, 150]
    flux[outlier_idx[0]] = 0.5
    flux[outlier_idx[1]] = 0.7
    flux[outlier_idx[2]] = 1.3
    flux[outlier_idx[3]] = 1.5
    expected_clipped_times = sorted(time[outlier_idx].tolist())
    write_data_csv(datadir / "tess.csv", time, flux, err)
    write_settings(datadir, inst_phot=["tess"], bandpass=None, extra=settings_extra)
    rows = (
        [
            {
                "name": "b_rr",
                "value": TRUE_RR_TESS,
                "fit": 1,
                "bounds": "uniform 0.0 0.3",
                "label": "rr",
            },
        ]
        + common_orbital_rows()
        + dilution_rows(["tess"])
        + err_baseline_rows(["tess"])
        + ldc_rows("tess")
    )
    write_params(datadir / "params.csv", rows=rows)
    return datadir, expected_clipped_times


class TestRawFluxClip:
    def test_clipped_rows_dropped_from_fit_data(self, tmp_path):
        datadir, expected = _achromatic_datadir_with_outliers(
            tmp_path,
            settings_extra=["flux_min_raw,0.85", "flux_max_raw,1.15"],
        )
        config.init(str(datadir), quiet=True)

        flux = config.BASEMENT.data["tess"]["flux"]
        # No surviving row should fall outside [0.85, 1.15].
        assert np.all(flux >= 0.85), f"min(flux)={flux.min()} not >= 0.85"
        assert np.all(flux <= 1.15), f"max(flux)={flux.max()} not <= 1.15"

    def test_clipped_rows_preserved_for_overlay(self, tmp_path):
        datadir, expected = _achromatic_datadir_with_outliers(
            tmp_path,
            settings_extra=["flux_min_raw,0.85", "flux_max_raw,1.15"],
        )
        config.init(str(datadir), quiet=True)

        clipped_time = config.BASEMENT.data["tess"]["raw_clipped_time"]
        clipped_flux = config.BASEMENT.data["tess"]["raw_clipped_flux"]
        clipped_err = config.BASEMENT.data["tess"]["raw_clipped_flux_err"]

        assert (
            len(clipped_time) == 4
        ), f"expected 4 clipped rows, got {len(clipped_time)} (times: {clipped_time.tolist()})"
        assert len(clipped_flux) == 4 and len(clipped_err) == 4
        assert sorted(clipped_time.tolist()) == pytest.approx(expected)
        # Each clipped flux value is one of the injected outlier amplitudes.
        assert set(round(float(v), 2) for v in clipped_flux) == {0.5, 0.7, 1.3, 1.5}

    def test_one_sided_min_clip(self, tmp_path):
        datadir, _ = _achromatic_datadir_with_outliers(
            tmp_path,
            settings_extra=["flux_min_raw,0.85"],  # no upper bound
        )
        config.init(str(datadir), quiet=True)

        # Only the two low outliers (0.5, 0.7) should be clipped; the highs stay.
        clipped_flux = config.BASEMENT.data["tess"]["raw_clipped_flux"]
        assert len(clipped_flux) == 2
        assert all(v < 0.85 for v in clipped_flux)

    def test_no_clip_when_settings_absent(self, tmp_path):
        datadir, _ = _achromatic_datadir_with_outliers(
            tmp_path,
            settings_extra=[],  # neither flux_min_raw nor flux_max_raw
        )
        config.init(str(datadir), quiet=True)

        # Without clipping, the outliers survive and raw_clipped_* is empty.
        assert len(config.BASEMENT.data["tess"]["raw_clipped_time"]) == 0
        assert len(config.BASEMENT.data["tess"]["raw_clipped_flux"]) == 0

    def test_invalid_bounds_raise(self, tmp_path):
        with pytest.raises(ValueError, match="flux_min_raw.*<.*flux_max_raw"):
            datadir, _ = _achromatic_datadir_with_outliers(
                tmp_path,
                settings_extra=["flux_min_raw,1.2", "flux_max_raw,0.9"],
            )
            config.init(str(datadir), quiet=True)

    def test_clip_does_not_break_fast_fit_reduction(self, tmp_path):
        # fast_fit reduces the kept data further. The clipped points must
        # survive that reduction (they're cached on `data` directly, not
        # passed through reduce_phot_data).
        datadir, expected = _achromatic_datadir_with_outliers(
            tmp_path,
            settings_extra=[
                "flux_min_raw,0.85",
                "flux_max_raw,1.15",
                "fast_fit,True",
                "fast_fit_width,0.3333333333333333",
            ],
        )
        config.init(str(datadir), quiet=True)
        clipped_time = config.BASEMENT.data["tess"]["raw_clipped_time"]
        assert len(clipped_time) == 4
        assert sorted(clipped_time.tolist()) == pytest.approx(expected)
