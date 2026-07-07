import numpy as np

from allesfitter.priors.transform_priors import (
    get_cosi_from_b,
    get_cosi_from_i,
    get_q1q2_from_u1u2,
    get_Rsuma_from_a_over_Rstar,
    get_Rsuma_from_Rstar_over_a,
    get_sqrtecosw,
    get_sqrtesinw,
    get_u1u2_from_q1q2,
)


class TestGetCosiFromI:
    def test_returns_three_values(self):
        result = get_cosi_from_i([82.5, 0.2, 0.3], Nsamples=1000)
        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_cosi_is_between_0_and_1(self):
        median, lower_err, upper_err = get_cosi_from_i([82.5, 0.2, 0.3], Nsamples=1000)
        assert 0 <= median <= 1
        assert lower_err >= 0
        assert upper_err >= 0

    def test_high_inclination_gives_small_cosi(self):
        cosi_result = get_cosi_from_i([89.0, 0.5, 0.5], Nsamples=1000)
        assert cosi_result[0] < 0.5

    def test_low_inclination_gives_large_cosi(self):
        cosi_result = get_cosi_from_i([30.0, 1.0, 1.0], Nsamples=1000)
        assert cosi_result[0] > 0.5

    def test_error_propagation(self):
        result = get_cosi_from_i([82.5, 0.2, 0.3], Nsamples=1000)
        assert all(isinstance(x, (int, float)) for x in result)
        assert all(np.isfinite(x) for x in result)


class TestGetCosiFromB:
    def test_returns_three_values(self):
        result = get_cosi_from_b([0.5, 0.1, 0.1], [10.0, 0.5, 0.5], Nsamples=1000)
        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_cosi_less_than_b_over_a(self):
        result = get_cosi_from_b([0.5, 0.1, 0.1], [10.0, 0.5, 0.5], Nsamples=1000)
        median, _, _ = result
        b_median = 0.5
        a_over_r = 10.0
        assert median < b_median / a_over_r + 0.2

    def test_valid_inputs(self):
        result = get_cosi_from_b([0.3, 0.05, 0.05], [15.0, 1.0, 1.0], Nsamples=1000)
        assert all(np.isfinite(x) for x in result)


class TestGetRsumaFromAOverRstar:
    def test_returns_three_values(self):
        result = get_Rsuma_from_a_over_Rstar([10.0, 0.5, 0.5], [0.1, 0.01, 0.01], Nsamples=1000)
        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_rsuma_positive(self):
        result = get_Rsuma_from_a_over_Rstar([10.0, 0.5, 0.5], [0.1, 0.01, 0.01], Nsamples=1000)
        median, _, _ = result
        assert median > 0

    def test_rsuma_less_than_one_over_a_over_r(self):
        result = get_Rsuma_from_a_over_Rstar([10.0, 0.5, 0.5], [0.1, 0.01, 0.01], Nsamples=1000)
        median, _, _ = result
        assert median < 1.0 / 10.0 + 0.05


class TestGetRsumaFromRstarOverA:
    def test_returns_three_values(self):
        result = get_Rsuma_from_Rstar_over_a([0.1, 0.005, 0.005], [0.1, 0.01, 0.01], Nsamples=1000)
        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_rsuma_positive(self):
        result = get_Rsuma_from_Rstar_over_a([0.1, 0.005, 0.005], [0.1, 0.01, 0.01], Nsamples=1000)
        median, _, _ = result
        assert median > 0


class TestGetSqrtesinw:
    def test_returns_three_values(self):
        result = get_sqrtesinw([0.3, 0.05, 0.05], [90.0, 5.0, 5.0], Nsamples=1000)
        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_sqrtesinw_in_valid_range(self):
        result = get_sqrtesinw([0.3, 0.05, 0.05], [90.0, 5.0, 5.0], Nsamples=1000)
        median, _, _ = result
        assert -1 <= median <= 1


class TestGetSqrtecosw:
    def test_returns_three_values(self):
        result = get_sqrtecosw([0.3, 0.05, 0.05], [0.0, 10.0, 10.0], Nsamples=1000)
        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_sqrtecosw_in_valid_range(self):
        result = get_sqrtecosw([0.3, 0.05, 0.05], [0.0, 10.0, 10.0], Nsamples=1000)
        median, _, _ = result
        assert -1 <= median <= 1


class TestGetU1U2FromQ1Q2:
    def test_float_inputs(self):
        u1, u2 = get_u1u2_from_q1q2(0.5, 0.5)
        assert isinstance(u1, float)
        assert isinstance(u2, float)
        assert u1 >= 0
        assert u2 >= -1

    def test_float_inputs_quadratic_relation(self):
        q1, q2 = 0.25, 0.5
        u1, u2 = get_u1u2_from_q1q2(q1, q2)
        expected_u1 = 2.0 * np.sqrt(q1) * q2
        expected_u2 = np.sqrt(q1) * (1.0 - 2.0 * q2)
        assert abs(u1 - expected_u1) < 1e-10
        assert abs(u2 - expected_u2) < 1e-10

    def test_list_inputs(self):
        result = get_u1u2_from_q1q2([0.5, 0.1, 0.1], [0.5, 0.1, 0.1], Nsamples=1000)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert all(isinstance(x, tuple) for x in result)

    def test_u_values_bounded(self):
        result = get_u1u2_from_q1q2([0.5, 0.1, 0.1], [0.5, 0.1, 0.1], Nsamples=1000)
        u1_result, u2_result = result
        assert u1_result[0] >= 0
        assert u2_result[0] >= -1

    def test_special_q_values(self):
        u1, u2 = get_u1u2_from_q1q2(1.0, 0.5)
        assert np.isfinite(u1)
        assert np.isfinite(u2)


class TestGetQ1Q2FromU1U2:
    def test_float_inputs(self):
        q1, q2 = get_q1q2_from_u1u2(0.5, 0.3)
        assert isinstance(q1, float)
        assert isinstance(q2, float)
        assert 0 <= q1 <= 1
        assert 0 <= q2 <= 1

    def test_float_inputs_inverse_relation(self):
        u1, u2 = 0.5, 0.3
        q1, q2 = get_q1q2_from_u1u2(u1, u2)
        expected_q1 = (u1 + u2) ** 2
        expected_q2 = 0.5 * u1 / (u1 + u2)
        assert abs(q1 - expected_q1) < 1e-10
        assert abs(q2 - expected_q2) < 1e-10

    def test_list_inputs(self):
        result = get_q1q2_from_u1u2([0.5, 0.1, 0.1], [0.3, 0.1, 0.1], Nsamples=1000)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert all(isinstance(x, tuple) for x in result)

    def test_round_trip_consistency(self):
        original_u1, original_u2 = 0.4, 0.25
        q1, q2 = get_q1q2_from_u1u2(original_u1, original_u2)
        recovered_u1, recovered_u2 = get_u1u2_from_q1q2(q1, q2)
        assert abs(recovered_u1 - original_u1) < 1e-6
        assert abs(recovered_u2 - original_u2) < 1e-6


class TestRoundTripConversions:
    def test_q_to_u_to_q(self):
        original_q1, original_q2 = 0.3, 0.6
        u1, u2 = get_u1u2_from_q1q2(original_q1, original_q2)
        recovered_q1, recovered_q2 = get_q1q2_from_u1u2(u1, u2)
        assert abs(recovered_q1 - original_q1) < 1e-6
        assert abs(recovered_q2 - original_q2) < 1e-6

    def test_multiple_round_trips(self):
        test_cases = [
            (0.1, 0.2),
            (0.5, 0.5),
            (0.8, 0.3),
        ]
        for q1, q2 in test_cases:
            u1, u2 = get_u1u2_from_q1q2(q1, q2)
            r_q1, r_q2 = get_q1q2_from_u1u2(u1, u2)
            assert abs(r_q1 - q1) < 1e-6
            assert abs(r_q2 - q2) < 1e-6
