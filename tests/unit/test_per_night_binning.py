"""Regression tests for gap-aware light-curve binning."""

from __future__ import annotations

import numpy as np
import pytest

from allesfitter.exoworlds_rdx.lightcurves.binning import (
    bin_edge_indices,
    binning1D_per_night,
    binning1D_per_night_list,
    binning2D_per_night,
)


def test_bin_edges_use_exclusive_ends_for_partial_bins_and_nights():
    time = np.array([0.0, 1.0, 2.0, 10.0, 11.0])

    starts, ends = bin_edge_indices(time, bin_width=2, timegap=3, N_time=len(time))

    assert starts == [0, 2, 3]
    assert ends == [2, 3, 5]


def test_1d_per_night_keeps_final_observation_and_singleton_night():
    time = np.array([0.0, 1.0, 2.0, 10.0])
    values = np.array([10.0, 20.0, 30.0, 80.0])

    bintime, binned, errors = binning1D_per_night(time, values, 3, timegap=3)

    np.testing.assert_allclose(bintime, [1.0, 10.0])
    np.testing.assert_allclose(binned, [20.0, 80.0])
    np.testing.assert_allclose(errors, [np.std(values[:3]), 0.0])


@pytest.mark.parametrize("setting", ["mean", "median"])
def test_list_variant_matches_array_variant(setting):
    time = np.array([0.0, 1.0, 2.0, 10.0, 11.0])
    values = np.array([10.0, 20.0, 30.0, 80.0, 100.0])

    expected = binning1D_per_night(time, values, 2, timegap=3, setting=setting)
    actual = binning1D_per_night_list(time, values, 2, timegap=3, setting=setting)

    for expected_array, actual_array in zip(expected, actual):
        np.testing.assert_allclose(actual_array, expected_array)


@pytest.mark.parametrize("setting", ["mean", "median"])
def test_2d_variant_keeps_boundary_and_partial_bin_observations(setting):
    time = np.array([[0.0, 1.0, 2.0, 10.0], [0.1, 1.1, 2.1, 10.1]])
    values = np.array([[10.0, 20.0, 30.0, 80.0], [20.0, 40.0, 60.0, 160.0]])

    _, binned, _ = binning2D_per_night(time, values, 3, timegap=3, setting=setting)

    np.testing.assert_allclose(binned, [[20.0, 80.0], [40.0, 160.0]])
