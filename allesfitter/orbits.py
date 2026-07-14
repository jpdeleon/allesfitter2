"""Lightweight orbital-geometry transformations shared across allesfitter.

The preparation pipeline fixes eccentricity to zero by default.  These helpers
therefore implement the exact circular-orbit geometry used by the light-curve
model, with ``rsuma = (R_star + R_companion) / a`` and the usual impact
parameter in units of the stellar radius.
"""

from __future__ import annotations

import numpy as np


def _scalarize(value):
    """Return a Python float for scalar input and an ndarray otherwise."""
    value = np.asarray(value, dtype=float)
    return float(value) if value.ndim == 0 else value


def circular_transit_duration(period, rsuma, cosi, radius_ratio):
    """Return total transit duration ``T14`` for a circular orbit, in period units.

    Parameters may be scalars or broadcastable arrays. Invalid or
    non-transiting geometries return ``NaN``.
    """
    period, rsuma, cosi, radius_ratio = np.broadcast_arrays(
        np.asarray(period, dtype=float),
        np.asarray(rsuma, dtype=float),
        np.asarray(cosi, dtype=float),
        np.asarray(radius_ratio, dtype=float),
    )
    duration = np.full(period.shape, np.nan, dtype=float)

    valid = (
        np.isfinite(period)
        & np.isfinite(rsuma)
        & np.isfinite(cosi)
        & np.isfinite(radius_ratio)
        & (period > 0.0)
        & (rsuma > 0.0)
        & (rsuma < 1.0)
        & (np.abs(cosi) < 1.0)
        & (radius_ratio >= 0.0)
    )

    with np.errstate(invalid="ignore", divide="ignore"):
        radius_sum = 1.0 + radius_ratio
        rstar_over_a = rsuma / radius_sum
        impact_parameter = cosi / rstar_over_a
        chord_squared = radius_sum**2 - impact_parameter**2
        sin_i = np.sqrt(1.0 - cosi**2)
        argument = rstar_over_a * np.sqrt(chord_squared) / sin_i
        physical = valid & (chord_squared > 0.0) & (argument > 0.0) & (argument <= 1.0)
        duration[physical] = period[physical] / np.pi * np.arcsin(argument[physical])

    return _scalarize(duration)


def circular_geometry_from_transit_duration(
    duration,
    period,
    radius_ratio,
    impact_parameter,
):
    """Derive a duration-consistent ``(rsuma, cosi)`` pair.

    This is the analytic inverse of :func:`circular_transit_duration` for an
    explicitly assumed impact parameter.  ``impact_parameter`` is restricted
    to ``0 <= b <= 1``: the preparation pipeline uses the transit-conditioned
    uniform prior for a non-grazing planet and initializes at its median,
    ``b=0.5``.

    Invalid inputs return ``(NaN, NaN)`` element-wise.
    """
    duration, period, radius_ratio, impact_parameter = np.broadcast_arrays(
        np.asarray(duration, dtype=float),
        np.asarray(period, dtype=float),
        np.asarray(radius_ratio, dtype=float),
        np.asarray(impact_parameter, dtype=float),
    )
    rsuma = np.full(duration.shape, np.nan, dtype=float)
    cosi = np.full(duration.shape, np.nan, dtype=float)

    valid = (
        np.isfinite(duration)
        & np.isfinite(period)
        & np.isfinite(radius_ratio)
        & np.isfinite(impact_parameter)
        & (duration > 0.0)
        & (period > 0.0)
        & (duration < 0.5 * period)
        & (radius_ratio >= 0.0)
        & (impact_parameter >= 0.0)
        & (impact_parameter <= 1.0)
    )

    with np.errstate(invalid="ignore", divide="ignore"):
        radius_sum = 1.0 + radius_ratio
        sine_duration = np.sin(np.pi * duration / period)
        denominator_squared = radius_sum**2 - impact_parameter**2 * (1.0 - sine_duration**2)
        candidate_rsuma = sine_duration * radius_sum / np.sqrt(denominator_squared)
        candidate_cosi = impact_parameter * candidate_rsuma / radius_sum
        physical = (
            valid
            & (denominator_squared > 0.0)
            & (candidate_rsuma > 0.0)
            & (candidate_rsuma < 1.0)
            & (candidate_cosi >= 0.0)
            & (candidate_cosi < 1.0)
        )
        rsuma[physical] = candidate_rsuma[physical]
        cosi[physical] = candidate_cosi[physical]

    return _scalarize(rsuma), _scalarize(cosi)
