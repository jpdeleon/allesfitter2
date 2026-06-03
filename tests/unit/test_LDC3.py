import numpy as np
import pytest
import math
import importlib.util
import os


def import_module_from_path(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# repo root = three levels up from tests/unit/<this file>
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

LDC3 = import_module_from_path('LDC3', os.path.join(BASE_DIR, 'allesfitter', 'limb_darkening', 'LDC3.py'))
forward = LDC3.forward
inverse = LDC3.inverse
criteriatest = LDC3.criteriatest


class TestForward:
    def test_forward_returns_three_values(self):
        alphas = [0.5, 0.5, 0.5]
        result = forward(alphas)
        assert isinstance(result, list)
        assert len(result) == 3

    def test_forward_returns_floats(self):
        alphas = [0.5, 0.5, 0.5]
        result = forward(alphas)
        assert all(isinstance(x, float) for x in result)

    def test_forward_valid_alphas(self):
        alphas = [0.1, 0.2, 0.3]
        result = forward(alphas)
        assert all(np.isfinite(x) for x in result)

    def test_forward_boundary_alphas(self):
        alphas = [0.0, 0.0, 0.0]
        result = forward(alphas)
        assert all(np.isfinite(x) for x in result)

    def test_forward_near_boundary(self):
        alphas = [0.999, 0.999, 0.999]
        result = forward(alphas)
        assert all(np.isfinite(x) for x in result)

    def test_forward_c_values_bound(self):
        alphas = [0.5, 0.5, 0.5]
        c = forward(alphas)
        c_2, c_3, c_4 = c
        assert c_2 + c_3 + c_4 <= 1.0


class TestInverse:
    def test_inverse_returns_three_values(self):
        c = [0.2, -0.1, 0.3]
        result = inverse(c)
        assert isinstance(result, list)
        assert len(result) == 3

    def test_inverse_returns_floats(self):
        c = [0.2, -0.1, 0.3]
        result = inverse(c)
        assert all(isinstance(x, float) for x in result)

    def test_inverse_valid_c_values(self):
        c = [0.2, -0.1, 0.3]
        result = inverse(c)
        assert all(0 <= x <= 1 for x in result)

    def test_inverse_round_trip(self):
        original_alphas = [0.5, 0.5, 0.5]
        c = forward(original_alphas)
        recovered_alphas = inverse(c)
        for orig, recov in zip(original_alphas, recovered_alphas):
            assert abs(orig - recov) < 1e-10

    def test_inverse_round_trip_various_inputs(self):
        test_cases = [
            [0.1, 0.2, 0.3],
            [0.8, 0.9, 0.1],
            [0.3, 0.6, 0.7],
        ]
        for alphas in test_cases:
            c = forward(alphas)
            recovered_alphas = inverse(c)
            for orig, recov in zip(alphas, recovered_alphas):
                assert abs(orig - recov) < 1e-6


class TestCriteriaTest:
    def test_criteriatest_unmodified_valid(self):
        c = [0.2, -0.1, 0.3]
        result = criteriatest(0, c)
        assert result in [0, 1]

    def test_criteriatest_modified_valid(self):
        c = [0.2, -0.1, 0.3]
        result = criteriatest(1, c)
        assert result in [0, 1]

    def test_criteriatest_criterion_a_fail(self):
        c = [0.5, 0.5, 0.5]
        result = criteriatest(0, c)
        assert result == 0

    def test_criteriatest_criterion_b_fail(self):
        c = [0.1, -0.3, 0.1]
        result = criteriatest(0, c)
        assert result == 0

    def test_criteriatest_criterion_d_fail(self):
        c = [-0.5, -0.5, -0.5]
        result = criteriatest(0, c)
        assert result == 0

    def test_criteriatest_criterion_f_modified_fail(self):
        c = [0.2, -0.1, -0.1]
        result = criteriatest(1, c)
        assert result == 0

    def test_criteriatest_criterion_g_fail(self):
        c = [0.5, -0.8, 0.2]
        result = criteriatest(0, c)
        assert result == 0

    def test_criteriatest_physically_valid(self):
        c = forward([0.5, 0.5, 0.5])
        result = criteriatest(0, c)
        assert result == 1

    def test_criteriatest_modified_more_strict(self):
        c = [0.3, -0.2, 0.1]
        unmodified_result = criteriatest(0, c)
        modified_result = criteriatest(1, c)
        assert modified_result <= unmodified_result


class TestLDC3RoundTrip:
    def test_forward_inverse_roundtrip(self):
        original = [0.3, 0.5, 0.7]
        c = forward(original)
        recovered = inverse(c)
        for orig, recov in zip(original, recovered):
            assert abs(orig - recov) < 1e-10

    def test_inverse_forward_roundtrip(self):
        original = [0.2, -0.1, 0.3]
        alphas = inverse(original)
        recovered = forward(alphas)
        for orig, recov in zip(original, recovered):
            assert abs(orig - recov) < 1e-10

    def test_multiple_roundtrips(self):
        test_cases = [
            [0.1, 0.2, 0.3],
            [0.2, 0.4, 0.6],
            [0.8, 0.9, 0.1],
            [0.5, 0.5, 0.5],
        ]
        for alphas in test_cases:
            c = forward(alphas)
            recovered_alphas = inverse(c)
            for orig, recov in zip(alphas, recovered_alphas):
                assert abs(orig - recov) < 1e-6


class TestLDC3PhysicalValidity:
    def test_forward_output_passes_criteria(self):
        alphas = [0.5, 0.5, 0.5]
        c = forward(alphas)
        assert criteriatest(0, c) == 1

    def test_all_valid_criteria(self):
        alphas_list = [
            [0.1, 0.2, 0.3],
            [0.3, 0.4, 0.5],
            [0.5, 0.6, 0.7],
            [0.7, 0.8, 0.9],
        ]
        for alphas in alphas_list:
            c = forward(alphas)
            assert criteriatest(0, c) == 1, f"Failed for alphas={alphas}, c={c}"


class TestLDC3EdgeCases:
    def test_zero_alpha_h(self):
        alphas = [0.0, 0.5, 0.5]
        c = forward(alphas)
        assert all(np.isfinite(x) for x in c)
        assert criteriatest(1, c) == 1

    def test_zero_alpha_r(self):
        alphas = [0.5, 0.0, 0.5]
        c = forward(alphas)
        assert all(np.isfinite(x) for x in c)

    def test_alpha_t_boundary(self):
        alphas = [0.5, 0.5, 0.0]
        c = forward(alphas)
        assert all(np.isfinite(x) for x in c)

    def test_alpha_t_wrapped(self):
        alphas = [0.5, 0.5, 0.999]
        c = forward(alphas)
        assert all(np.isfinite(x) for x in c)
