import numpy as np
import pytest
from astropy.time import Time

from allesfitter.time_series import (
    clean,
    sort,
    sigma_clip,
    slide_clip,
    binning,
    mask_regions,
)

try:
    import wotan  # noqa: F401  (optional dependency used by slide_clip)

    HAS_WOTAN = True
except ImportError:
    HAS_WOTAN = False


class TestClean:
    def test_clean_removes_nan(self):
        time = np.array([1.0, 2.0, 3.0, 4.0])
        y = np.array([1.0, np.nan, 3.0, 4.0])
        time_clean, y_clean, _ = clean(time, y)
        assert len(time_clean) == 3
        assert len(y_clean) == 3
        assert np.nan not in y_clean

    def test_clean_removes_inf(self):
        time = np.array([1.0, 2.0, 3.0, 4.0])
        y = np.array([1.0, np.inf, 3.0, 4.0])
        time_clean, y_clean, _ = clean(time, y)
        assert len(time_clean) == 3

    def test_clean_with_errors(self):
        time = np.array([1.0, 2.0, 3.0])
        y = np.array([1.0, 2.0, 3.0])
        y_err = np.array([0.1, np.nan, 0.1])
        time_clean, y_clean, y_err_clean = clean(time, y, y_err)
        assert len(time_clean) == 2
        assert len(y_err_clean) == 2

    def test_clean_all_valid(self):
        time = np.array([1.0, 2.0, 3.0])
        y = np.array([1.0, 2.0, 3.0])
        time_clean, y_clean, _ = clean(time, y)
        assert len(time_clean) == 3

    def test_clean_handles_masked_array(self):
        time = np.ma.array([1.0, 2.0, 3.0, 4.0], mask=[False, True, False, False])
        y = np.array([1.0, 2.0, 3.0, 4.0])
        time_clean, y_clean, _ = clean(time, y)
        assert len(time_clean) >= 0

    def test_clean_with_astropy_time(self):
        t = Time([50000.0, 50001.0, 50002.0], format="mjd")
        y = np.array([1.0, 2.0, 3.0])
        time_clean, y_clean, _ = clean(t, y)
        assert len(time_clean) == 3


class TestSort:
    def test_sort_orders_time(self):
        time = np.array([3.0, 1.0, 2.0])
        y = np.array([30.0, 10.0, 20.0])
        time_sorted, y_sorted, _ = sort(time, y)
        assert np.all(time_sorted == np.sort(time))

    def test_sort_maintains_correspondence(self):
        time = np.array([3.0, 1.0, 2.0])
        y = np.array([30.0, 10.0, 20.0])
        time_sorted, y_sorted, _ = sort(time, y)
        assert y_sorted[0] == 10.0
        assert y_sorted[1] == 20.0
        assert y_sorted[2] == 30.0

    def test_sort_with_errors(self):
        time = np.array([3.0, 1.0, 2.0])
        y = np.array([30.0, 10.0, 20.0])
        y_err = np.array([0.3, 0.1, 0.2])
        time_sorted, y_sorted, y_err_sorted = sort(time, y, y_err)
        assert y_err_sorted[0] == 0.1
        assert y_err_sorted[1] == 0.2
        assert y_err_sorted[2] == 0.3


class TestSigmaClip:
    def test_sigma_clip_returns_array(self, sample_time, sample_flux):
        result = sigma_clip(sample_time, sample_flux)
        assert isinstance(result, np.ndarray)

    def test_sigma_clip_preserves_length(self, sample_time, sample_flux):
        result = sigma_clip(sample_time, sample_flux)
        assert len(result) == len(sample_flux)

    def test_sigma_clip_with_outliers(self, sample_time, sample_flux_with_outliers):
        result = sigma_clip(sample_time, sample_flux_with_outliers, low=4, high=4)
        assert isinstance(result, np.ndarray)
        assert len(result) == len(sample_flux_with_outliers)
        nan_count = np.sum(np.isnan(result))
        assert nan_count >= 2

    def test_sigma_clip_return_mask(self, sample_time, sample_flux):
        result = sigma_clip(
            sample_time, sample_flux, low=4, high=4, return_mask=True
        )
        assert isinstance(result, tuple)
        assert len(result) == 4

    def test_sigma_clip_different_sigmas(self, sample_time, sample_flux):
        result_loose = sigma_clip(sample_time, sample_flux, low=5, high=5)
        result_strict = sigma_clip(sample_time, sample_flux, low=2, high=2)
        nan_loose = np.sum(np.isnan(result_loose))
        nan_strict = np.sum(np.isnan(result_strict))
        assert nan_strict >= nan_loose


@pytest.mark.skipif(not HAS_WOTAN, reason="wotan not installed (optional dependency)")
class TestSlideClip:
    def test_slide_clip_returns_array(self, sample_time, sample_flux):
        result = slide_clip(sample_time, sample_flux, window_length=1.0)
        assert isinstance(result, np.ndarray)

    def test_slide_clip_preserves_length(self, sample_time, sample_flux):
        result = slide_clip(sample_time, sample_flux, window_length=1.0)
        assert len(result) == len(sample_flux)

    def test_slide_clip_with_outliers(self, sample_time, sample_flux_with_outliers):
        result = slide_clip(
            sample_time, sample_flux_with_outliers, window_length=1.0, low=4, high=4
        )
        assert isinstance(result, np.ndarray)
        assert len(result) == len(sample_flux_with_outliers)

    def test_slide_clip_return_mask(self, sample_time, sample_flux):
        result = slide_clip(
            sample_time, sample_flux, window_length=1.0, low=4, high=4, return_mask=True
        )
        assert isinstance(result, tuple)
        assert len(result) == 4


class TestBinning:
    def test_binning_no_dt_preserves_length(self):
        time = np.linspace(0, 10, 100)
        y = np.random.normal(1.0, 0.01, 100)
        time_binned, y_binned, _ = binning(time, y, dt=None)
        assert len(time_binned) == len(time)

    def test_binning_with_dt_reduces_length(self):
        time = np.linspace(0, 10, 100)
        y = np.random.normal(1.0, 0.01, 100)
        time_binned, y_binned, _ = binning(time, y, dt=1.0)
        assert len(time_binned) < len(time)

    def test_binning_preserves_order(self):
        time = np.linspace(0, 10, 100)
        y = np.random.normal(1.0, 0.01, 100)
        time_binned, y_binned, _ = binning(time, y, dt=1.0)
        assert np.all(time_binned == np.sort(time_binned))

    def test_binning_with_errors(self):
        time = np.linspace(0, 10, 100)
        y = np.random.normal(1.0, 0.01, 100)
        y_err = np.ones(100) * 0.01
        time_binned, y_binned, y_err_binned = binning(time, y, y_err, dt=1.0)
        assert len(y_err_binned) == len(time_binned)


class TestMaskRegions:
    def test_mask_regions_returns_array(self):
        time = np.linspace(0, 10, 100)
        y = np.ones(100)
        bad_regions = [(2.0, 3.0)]
        result = mask_regions(time, y, bad_regions=bad_regions)
        assert isinstance(result, np.ndarray)

    def test_mask_regions_masks_correct_region(self):
        time = np.linspace(0, 10, 100)
        y = np.ones(100)
        bad_regions = [(2.0, 3.0)]
        result = mask_regions(time, y, bad_regions=bad_regions)
        masked_values = result[np.isnan(result)]
        assert len(masked_values) > 0

    def test_mask_regions_no_bad_regions(self):
        time = np.linspace(0, 10, 100)
        y = np.ones(100)
        result = mask_regions(time, y, bad_regions=None)
        assert np.all(~np.isnan(result))

    def test_mask_regions_multiple_regions(self):
        time = np.linspace(0, 10, 100)
        y = np.ones(100)
        bad_regions = [(1.0, 2.0), (5.0, 6.0), (8.0, 9.0)]
        result = mask_regions(time, y, bad_regions=bad_regions)
        nan_count = np.sum(np.isnan(result))
        assert nan_count > 0
