"""Tests for shared transit-orbit geometry helpers."""

from __future__ import annotations

import numpy as np
import pytest


def test_circular_duration_geometry_round_trip():
    from allesfitter.orbits import (
        circular_geometry_from_transit_duration,
        circular_transit_duration,
    )

    duration = np.array([0.08, 0.20, 0.35])
    period = np.array([2.0, 6.959473, 14.334892])
    radius_ratio = np.array([0.03, 0.0653, 0.0524])
    impact_parameter = np.array([0.0, 0.5, 0.95])

    rsuma, cosi = circular_geometry_from_transit_duration(
        duration, period, radius_ratio, impact_parameter
    )
    recovered = circular_transit_duration(period, rsuma, cosi, radius_ratio)

    assert np.all(np.isfinite(rsuma))
    assert np.all(np.isfinite(cosi))
    assert np.allclose(recovered, duration, rtol=0, atol=1e-12)


def test_legacy_scripting_helpers_use_the_same_standard_geometry():
    from allesfitter.orbits import circular_geometry_from_transit_duration
    from allesfitter.utils.scripting import get_rsuma, get_tdur

    duration = 4.8 / 24.0
    period = 6.959473
    radius_ratio = 0.0653
    impact_parameter = 0.5
    rsuma, cosi = circular_geometry_from_transit_duration(
        duration, period, radius_ratio, impact_parameter
    )
    inclination = np.arccos(cosi)

    assert get_tdur(period, rsuma, inclination, radius_ratio, impact_parameter) == pytest.approx(
        duration, abs=1e-12
    )
    assert get_rsuma(
        duration, period, inclination, radius_ratio, impact_parameter
    ) == pytest.approx(rsuma, abs=1e-12)


@pytest.mark.parametrize(
    ("period", "radius_ratio", "rsuma", "cosi"),
    [
        (3.0, 0.10, 0.10, 0.00),
        (6.959473, 0.0653, 0.1019879535, 0.0478681843),
        (14.334892, 0.0524, 0.0591022279, 0.0280797358),
    ],
)
def test_circular_duration_matches_allesfitter_model(period, radius_ratio, rsuma, cosi):
    from allesfitter.lightcurves import eclipse_width_smart
    from allesfitter.orbits import circular_transit_duration

    expected, _ = eclipse_width_smart(period, radius_ratio, rsuma, cosi, 0.0, 0.0)
    actual = circular_transit_duration(period, rsuma, cosi, radius_ratio)

    assert actual == pytest.approx(float(expected), rel=1e-12)


def test_circular_geometry_rejects_unphysical_inputs():
    from allesfitter.orbits import circular_geometry_from_transit_duration

    rsuma, cosi = circular_geometry_from_transit_duration(
        duration=np.array([-1.0, 2.0, 0.1]),
        period=np.array([3.0, 3.0, 3.0]),
        radius_ratio=np.array([0.1, 0.1, 0.1]),
        impact_parameter=np.array([0.5, 0.5, 1.2]),
    )

    assert np.all(np.isnan(rsuma))
    assert np.all(np.isnan(cosi))
