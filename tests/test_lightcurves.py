import numpy as np
import pytest

from allesfitter.lightcurves import (
    mask_ranges,
    get_first_epoch,
    get_epoch_occ,
    get_Rhost_over_a,
    get_ecc_esinw_ecosw,
    impact_parameters_smart,
    eclipse_width_smart,
    index_eclipses_smart,
    translate_limb_darkening_from_u_to_q,
    translate_limb_darkening_from_q_to_u,
)


class TestMaskRanges:
    def test_basic_masking(self):
        x = np.arange(200)
        x_min = [5, 25, 90]
        x_max = [10, 35, 110]
        x_masked, ind_mask, mask = mask_ranges(x, x_min, x_max)
        assert len(x_masked) == len(x_min) + len(x_max)
        assert len(ind_mask) == len(x_masked)
        assert mask.sum() == len(x_masked)

    def test_no_masking(self):
        x = np.arange(200)
        x_min = [500]
        x_max = [600]
        x_masked, ind_mask, mask = mask_ranges(x, x_min, x_max)
        assert len(x_masked) == 0

    def test_full_mask(self):
        x = np.arange(200)
        x_min = [-10]
        x_max = [210]
        x_masked, ind_mask, mask = mask_ranges(x, x_min, x_max)
        assert len(x_masked) == 200


class TestGetFirstEpoch:
    def test_epoch_after_start(self):
        time = np.linspace(0, 10, 100)
        epoch = 2.0
        period = 1.0
        first_epoch = get_first_epoch(time, epoch, period)
        assert first_epoch <= np.nanmin(time)

    def test_epoch_before_start(self):
        time = np.linspace(5, 15, 100)
        epoch = 2.0
        period = 1.0
        first_epoch = get_first_epoch(time, epoch, period)
        assert first_epoch >= np.nanmin(time) - period

    def test_with_width(self):
        time = np.linspace(0, 10, 100)
        epoch = 2.0
        period = 1.0
        width = 0.5
        first_epoch = get_first_epoch(time, epoch, period, width=width)
        assert isinstance(first_epoch, (int, float))

    def test_single_value_input(self):
        time = np.linspace(0, 10, 100)
        epoch = 2457000.0
        period = 3.5
        first_epoch = get_first_epoch(time, epoch, period)
        assert isinstance(first_epoch, (int, float, np.floating))


class TestGetEpochOcc:
    def test_circular_orbit(self):
        epoch = 2457000.0
        period = 3.5
        f_s = 0.0
        f_c = 0.0
        epoch_occ = get_epoch_occ(epoch, period, f_s, f_c)
        expected = epoch + period / 2.0
        assert abs(epoch_occ - expected) < 1e-10

    def test_eccentric_orbit(self):
        epoch = 2457000.0
        period = 3.5
        f_s = 0.1
        f_c = 0.1
        epoch_occ = get_epoch_occ(epoch, period, f_s, f_c)
        assert epoch_occ != epoch + period / 2.0


class TestGetRhostOverA:
    def test_basic_calculation(self):
        rr = 0.1
        rsuma = 0.11
        result = get_Rhost_over_a(rr, rsuma)
        expected = rsuma / (1.0 + rr)
        assert abs(result - expected) < 1e-10

    def test_zero_rr(self):
        rr = 0.0
        rsuma = 0.1
        result = get_Rhost_over_a(rr, rsuma)
        assert abs(result - rsuma) < 1e-10


class TestGetEccEsinwEcosw:
    def test_circular_orbit(self):
        f_s = 0.0
        f_c = 0.0
        ecc, esinw, ecosw = get_ecc_esinw_ecosw(f_s, f_c)
        assert ecc == 0.0
        assert esinw == 0.0
        assert ecosw == 0.0

    def test_eccentric_orbit(self):
        f_s = 0.3
        f_c = 0.4
        ecc, esinw, ecosw = get_ecc_esinw_ecosw(f_s, f_c)
        expected_ecc = f_s ** 2 + f_c ** 2
        assert abs(ecc - expected_ecc) < 1e-10
        assert esinw > 0
        assert ecosw > 0

    def test_ecc_less_than_one(self):
        f_s = 0.5
        f_c = 0.5
        ecc, _, _ = get_ecc_esinw_ecosw(f_s, f_c)
        assert ecc < 1.0


class TestImpactParametersSmart:
    def test_returns_two_values(self):
        rr = 0.1
        rsuma = 0.11
        cosi = 0.1
        f_s = 0.0
        f_c = 0.0
        b_1, b_2 = impact_parameters_smart(rr, rsuma, cosi, f_s, f_c)
        assert isinstance(b_1, (int, float, np.floating))
        assert isinstance(b_2, (int, float, np.floating))

    def test_circular_orbit_b1_equals_b2(self):
        rr = 0.1
        rsuma = 0.11
        cosi = 0.5
        f_s = 0.0
        f_c = 0.0
        b_1, b_2 = impact_parameters_smart(rr, rsuma, cosi, f_s, f_c)
        assert abs(b_1 - b_2) < 1e-10


class TestEclipseWidthSmart:
    def test_returns_two_values(self):
        result = eclipse_width_smart(
            period=3.5, rr=0.1, rsuma=0.11, cosi=0.1, f_s=0.0, f_c=0.0
        )
        width_1, width_2 = result
        assert isinstance(width_1, (int, float, np.floating))
        assert isinstance(width_2, (int, float, np.floating))

    def test_widths_positive(self):
        width_1, width_2 = eclipse_width_smart(
            period=3.5, rr=0.1, rsuma=0.11, cosi=0.1, f_s=0.0, f_c=0.0
        )
        assert width_1 > 0
        assert width_2 > 0

    def test_widths_less_than_period(self):
        period = 3.5
        width_1, width_2 = eclipse_width_smart(
            period=period, rr=0.1, rsuma=0.11, cosi=0.1, f_s=0.0, f_c=0.0
        )
        assert width_1 < period
        assert width_2 < period


class TestIndexEclipsesSmart:
    def test_returns_three_arrays(self):
        time = np.linspace(0, 10, 1000)
        result = index_eclipses_smart(
            time,
            epoch=0.0,
            period=3.5,
            rr=0.1,
            rsuma=0.11,
            cosi=0.1,
            f_s=0.0,
            f_c=0.0,
        )
        assert isinstance(result, tuple)
        assert len(result) == 3
        ind_ecl1, ind_ecl2, ind_out = result
        assert isinstance(ind_ecl1, np.ndarray)
        assert isinstance(ind_ecl2, np.ndarray)
        assert isinstance(ind_out, np.ndarray)

    def test_indices_disjoint(self):
        time = np.linspace(0, 10, 1000)
        ind_ecl1, ind_ecl2, ind_out = index_eclipses_smart(
            time,
            epoch=0.0,
            period=3.5,
            rr=0.1,
            rsuma=0.11,
            cosi=0.1,
            f_s=0.0,
            f_c=0.0,
        )
        overlap_1_2 = np.intersect1d(ind_ecl1, ind_ecl2)
        assert len(overlap_1_2) == 0 or len(ind_ecl1) == 0 or len(ind_ecl2) == 0


class TestTranslateLimbDarkeningFromUToQ:
    def test_none_law_returns_none(self):
        result = translate_limb_darkening_from_u_to_q(None, law=None)
        assert result is None

    def test_linear_law(self):
        u = 0.5
        result = translate_limb_darkening_from_u_to_q(u, law="lin")
        assert result == u

    def test_quadratic_law(self):
        u = [0.3, 0.1]
        result = translate_limb_darkening_from_u_to_q(u, law="quad")
        expected_q1 = (u[0] + u[1]) ** 2
        expected_q2 = 0.5 * u[0] / (u[0] + u[1])
        assert abs(result[0] - expected_q1) < 1e-10
        assert abs(result[1] - expected_q2) < 1e-10


class TestTranslateLimbDarkeningFromQToU:
    def test_none_law_returns_none(self):
        result = translate_limb_darkening_from_q_to_u(None, law=None)
        assert result is None

    def test_linear_law(self):
        q = 0.5
        result = translate_limb_darkening_from_q_to_u(q, law="lin")
        assert result == q

    def test_quadratic_law(self):
        q = [0.2, 0.6]
        result = translate_limb_darkening_from_q_to_u(q, law="quad")
        expected_u1 = 2.0 * np.sqrt(q[0]) * q[1]
        expected_u2 = np.sqrt(q[0]) * (1.0 - 2.0 * q[1])
        assert abs(result[0] - expected_u1) < 1e-10
        assert abs(result[1] - expected_u2) < 1e-10


class TestLimbDarkeningRoundTrip:
    def test_quadratic_round_trip(self):
        original_u = [0.3, 0.1]
        q = translate_limb_darkening_from_u_to_q(original_u, law="quad")
        recovered_u = translate_limb_darkening_from_q_to_u(q, law="quad")
        assert abs(recovered_u[0] - original_u[0]) < 1e-10
        assert abs(recovered_u[1] - original_u[1]) < 1e-10

    def test_multiple_quadratic_round_trips(self):
        test_cases = [
            [0.1, 0.2],
            [0.5, 0.3],
            [0.8, 0.1],
        ]
        for u in test_cases:
            q = translate_limb_darkening_from_u_to_q(u, law="quad")
            recovered_u = translate_limb_darkening_from_q_to_u(q, law="quad")
            assert abs(recovered_u[0] - u[0]) < 1e-10
            assert abs(recovered_u[1] - u[1]) < 1e-10
