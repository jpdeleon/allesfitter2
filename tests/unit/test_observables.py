"""Scientific regression tests for observable conversions."""

from __future__ import annotations

import numpy as np
from astropy import units as u

from allesfitter.observables import calc_M_comp_from_RV, calc_M_comp_from_RV_astropy


def test_rv_mass_approximation_matches_exact_solution_for_planet():
    inputs = {"K": 0.01, "P": 10.0, "incl": 87.0, "ecc": 0.1, "M_host": 1.0}

    exact = calc_M_comp_from_RV(**inputs, return_unit="M_earth")
    approximate = calc_M_comp_from_RV(**inputs, return_unit="M_earth", approx=True)

    np.testing.assert_allclose(approximate, exact, rtol=1e-3)


def test_rv_mass_approximation_float_and_astropy_implementations_agree():
    inputs = {"K": 0.01, "P": 10.0, "incl": 60.0, "ecc": 0.2, "M_host": 0.8}

    float_result = calc_M_comp_from_RV(**inputs, return_unit="M_jup", approx=True)
    astropy_result = calc_M_comp_from_RV_astropy(**inputs, return_unit=u.Mjup, approx=True)

    np.testing.assert_allclose(float_result, astropy_result, rtol=1e-12)
