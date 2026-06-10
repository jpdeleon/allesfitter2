"""Unit tests for ``allesfitter.validation.physical_limits``.

The registry and its helpers are pure (string/number in, string/None out), so
the tests assert directly on the returned error strings without constructing a
:class:`~allesfitter.basement.Basement`.
"""

from __future__ import annotations

import math

import pytest

from allesfitter.validation.physical_limits import (
    Limit,
    check_value,
    eccentricity_error,
    lookup_limit,
)

# ---------------------------------------------------------------------------
# Limit
# ---------------------------------------------------------------------------


def test_limit_contains_respects_inclusivity():
    half_open = Limit(0.0, 1.0, lo_inclusive=False, hi_inclusive=True)
    assert not half_open.contains(0.0)  # lower exclusive
    assert half_open.contains(1.0)  # upper inclusive
    assert half_open.contains(0.5)
    assert not half_open.contains(1.5)


def test_limit_describe_brackets():
    assert Limit(0.0, 1.0, lo_inclusive=False).describe() == "(0, 1]"
    assert Limit(0.0, 1.0, hi_inclusive=False).describe() == "[0, 1)"


# ---------------------------------------------------------------------------
# lookup_limit
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name,label",
    [
        ("b_rsuma", "(Rs+Rp)/a"),
        ("c_rsuma", "(Rs+Rp)/a"),
        ("dil_Leonardo", "dilution fraction"),
        ("host_vsini", "vsini"),
        ("b_vsini", "vsini"),
    ],
)
def test_lookup_limit_matches_registered_patterns(name, label):
    limit = lookup_limit(name)
    assert limit is not None and limit.label == label


@pytest.mark.parametrize("name", ["b_period", "b_cosi", "b_rr", "host_ldc_q1_Leonardo"])
def test_lookup_limit_returns_none_for_unregistered(name):
    assert lookup_limit(name) is None


# ---------------------------------------------------------------------------
# check_value
# ---------------------------------------------------------------------------


def test_rsuma_above_one_is_flagged():
    err = check_value("b_rsuma", 1.5)
    assert err is not None and "physical range (0, 1]" in err


def test_rsuma_at_one_is_allowed():
    assert check_value("b_rsuma", 1.0) is None


def test_rsuma_zero_is_flagged_lower_exclusive():
    assert check_value("b_rsuma", 0.0) is not None


def test_dilution_must_stay_below_one():
    assert check_value("dil_Leonardo", 1.0) is not None  # upper exclusive
    assert check_value("dil_Leonardo", -0.1) is not None
    assert check_value("dil_Leonardo", 0.0) is None
    assert check_value("dil_Leonardo", 0.3) is None


def test_vsini_must_be_nonnegative():
    assert check_value("host_vsini", -1.0) is not None
    assert check_value("host_vsini", 0.0) is None
    assert check_value("b_vsini", 5.0) is None


def test_check_value_ignores_unregistered_and_nonfinite():
    assert check_value("b_period", 1e6) is None  # no registered limit
    assert check_value("b_rsuma", math.nan) is None
    assert check_value("b_rsuma", math.inf) is None


# ---------------------------------------------------------------------------
# eccentricity_error
# ---------------------------------------------------------------------------


def test_eccentricity_below_one_passes():
    assert eccentricity_error("b", 0.1, 0.1) is None
    assert eccentricity_error("b", 0.0, 0.0) is None


def test_eccentricity_at_or_above_one_flagged():
    # f_s=0.8, f_c=0.8 -> e = 1.28 >= 1
    err = eccentricity_error("b", 0.8, 0.8)
    assert err is not None and "e=f_s^2+f_c^2" in err
    # boundary e == 1 is unbound (parabolic), so flagged
    assert eccentricity_error("b", 1.0, 0.0) is not None


def test_eccentricity_ignores_nonfinite():
    assert eccentricity_error("b", math.nan, 0.1) is None
    assert eccentricity_error("b", 0.1, math.inf) is None
