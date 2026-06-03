"""
Tests for allesfitter.priors.simulate_PDF module.

Note: These tests require matplotlib to be properly installed.
Run with: pytest tests/test_simulate_PDF.py
"""

import numpy as np
import pytest
import importlib.util
import os
import sys
from unittest.mock import MagicMock

sys.modules['matplotlib'] = MagicMock()
sys.modules['matplotlib.pyplot'] = MagicMock()
sys.modules['seaborn'] = MagicMock()

from scipy.stats import skewnorm


def import_module_from_path(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# repo root = three levels up from tests/unit/<this file>
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

simulate_PDF_module = import_module_from_path('simulate_PDF', 
    os.path.join(BASE_DIR, 'allesfitter', 'priors', 'simulate_PDF.py'))
simulate_PDF = simulate_PDF_module.simulate_PDF
calculate_skewed_normal_params = simulate_PDF_module.calculate_skewed_normal_params


class TestSimulatePDF:
    def test_simulate_pdf_returns_array(self):
        median, lower_err, upper_err = 1.0, 0.1, 0.1
        result = simulate_PDF(median, lower_err, upper_err, size=100, plot=False)
        assert isinstance(result, np.ndarray)
        assert len(result) == 100

    def test_simulate_pdf_with_plot_false_returns_samples_only(self):
        median, lower_err, upper_err = 1.0, 0.1, 0.1
        result = simulate_PDF(median, lower_err, upper_err, size=50, plot=False)
        assert isinstance(result, np.ndarray)
        assert len(result) == 50

    def test_simulate_pdf_with_plot_true_returns_tuple(self):
        median, lower_err, upper_err = 1.0, 0.1, 0.1
        result = simulate_PDF(median, lower_err, upper_err, size=50, plot=True)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], np.ndarray)

    def test_simulate_pdf_asymmetric_errors(self):
        median, lower_err, upper_err = 1.0, 0.1, 0.2
        result = simulate_PDF(median, lower_err, upper_err, size=100, plot=False)
        assert isinstance(result, np.ndarray)
        assert len(result) == 100

    def test_simulate_pdf_negative_lower_err(self):
        result = simulate_PDF(1.0, -0.1, 0.1, size=50, plot=False)
        assert isinstance(result, np.ndarray)
        assert len(result) == 50

    def test_simulate_pdf_zero_errors(self):
        result = simulate_PDF(1.0, 0.0, 0.0, size=50, plot=False)
        assert isinstance(result, np.ndarray)
        assert len(result) == 50

    def test_simulate_pdf_single_sample(self):
        median, lower_err, upper_err = 1.0, 0.1, 0.1
        result = simulate_PDF(median, lower_err, upper_err, size=1, plot=False)
        assert isinstance(result, np.ndarray)
        assert len(result) == 1

    def test_simulate_pdf_large_sample(self):
        median, lower_err, upper_err = 1.0, 0.1, 0.1
        result = simulate_PDF(median, lower_err, upper_err, size=10000, plot=False)
        assert isinstance(result, np.ndarray)
        assert len(result) == 10000

    def test_simulate_pdf_mean_approximately_median(self):
        np.random.seed(42)
        median, lower_err, upper_err = 1.0, 0.1, 0.1
        result = simulate_PDF(median, lower_err, upper_err, size=10000, plot=False)
        sample_median = np.median(result)
        assert abs(sample_median - median) < 0.1

    def test_simulate_pdf_std_approximately_error(self):
        np.random.seed(42)
        median, lower_err, upper_err = 1.0, 0.1, 0.1
        result = simulate_PDF(median, lower_err, upper_err, size=10000, plot=False)
        expected_std = np.mean([abs(lower_err), abs(upper_err)])
        sample_std = np.std(result)
        assert abs(sample_std - expected_std) / expected_std < 0.3

    def test_simulate_pdf_extreme_values(self):
        result = simulate_PDF(100.0, 1.0, 2.0, size=100, plot=False)
        assert isinstance(result, np.ndarray)
        assert len(result) == 100
        assert np.all(result > 0)


class TestCalculateSkewedNormalParams:
    def test_returns_three_values(self):
        median, lower_err, upper_err = 1.0, 0.1, 0.1
        result = calculate_skewed_normal_params(median, lower_err, upper_err)
        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_params_are_finite(self):
        median, lower_err, upper_err = 1.0, 0.1, 0.1
        alpha, loc, scale = calculate_skewed_normal_params(median, lower_err, upper_err)
        assert np.isfinite(alpha)
        assert np.isfinite(loc)
        assert np.isfinite(scale)

    def test_scale_is_positive(self):
        median, lower_err, upper_err = 1.0, 0.1, 0.1
        alpha, loc, scale = calculate_skewed_normal_params(median, lower_err, upper_err)
        assert scale > 0

    def test_asymmetric_errors(self):
        median, lower_err, upper_err = 1.0, 0.1, 0.15
        alpha, loc, scale = calculate_skewed_normal_params(median, lower_err, upper_err)
        assert np.isfinite(alpha)
        assert np.isfinite(loc)
        assert scale > 0

    def test_skewed_right(self):
        alpha, loc, scale = calculate_skewed_normal_params(1.0, 0.1, 0.5)
        assert np.isfinite(alpha)
        assert scale > 0

    def test_skewed_left(self):
        alpha, loc, scale = calculate_skewed_normal_params(1.0, 0.5, 0.1)
        assert np.isfinite(alpha)
        assert scale > 0

    def test_negative_errors_handled(self):
        alpha, loc, scale = calculate_skewed_normal_params(1.0, -0.1, -0.2)
        assert np.isfinite(alpha)
        assert scale > 0

    def test_output_percentiles_match_input(self):
        median, lower_err, upper_err = 1.0, 0.1, 0.15
        alpha, loc, scale = calculate_skewed_normal_params(median, lower_err, upper_err)
        percentiles = skewnorm.ppf([0.15865, 0.5, 0.84135], alpha, loc=loc, scale=scale)
        expected_lower = median - lower_err
        expected_upper = median + upper_err
        assert abs(percentiles[0] - expected_lower) < 0.15
        assert abs(percentiles[1] - median) < 0.1
        assert abs(percentiles[2] - expected_upper) < 0.15

    def test_zero_errors(self):
        alpha, loc, scale = calculate_skewed_normal_params(1.0, 0.0, 0.0)
        assert np.isfinite(alpha)
        assert np.isfinite(loc)
        assert scale >= 0
