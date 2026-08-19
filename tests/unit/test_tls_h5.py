"""Tests for :mod:`allesfitter.tls_h5`, the quicklook TLS ``.h5`` reader used
by ``prepare_allesfit.py --h5`` to seed a raw ``-tic`` target's transit
parameters."""

from __future__ import annotations

import json

import h5py
import numpy as np
import pytest

from allesfitter.tls_h5 import (
    read_h5_lightcurve,
    read_h5_transit_params,
    write_h5_tls_result,
)


def _write_current_format(path, **scalars):
    """Write an h5 file the way quicklook's current ``h5io.save`` does:
    scalars JSON-encoded into a single ``_scalar_data`` file attribute."""
    with h5py.File(path, "w") as f:
        f.attrs["_scalar_data"] = json.dumps(scalars)


def _write_current_format_with_arrays(path, arrays, **scalars):
    """Like ``_write_current_format`` but also writes ndarray datasets
    (``time_raw``/``flux_raw``/``err_raw``), matching how quicklook's
    ``h5io.save`` splits ndarrays into datasets vs. the ``_scalar_data``
    JSON attribute for everything else."""
    with h5py.File(path, "w") as f:
        for key, arr in arrays.items():
            f.create_dataset(key, data=arr)
        f.attrs["_scalar_data"] = json.dumps(scalars)


def _write_legacy_format(path, depth_mean=None, **scalars):
    """Write an h5 file the way the legacy deepdish/flammkuchen backend did:
    scalars as file-level attrs, tuples as a Group with i0/i1/... attrs."""
    with h5py.File(path, "w") as f:
        f.attrs["CLASS"] = "GROUP"
        f.attrs["TITLE"] = ""
        for key, value in scalars.items():
            f.attrs[key] = value
        if depth_mean is not None:
            group = f.create_group("depth_mean")
            group.attrs["TITLE"] = f"tuple:{len(depth_mean)}"
            for i, item in enumerate(depth_mean):
                group.attrs[f"i{i}"] = item


_COMMON_TLS_FIELDS = dict(
    period=4.319534613356017,
    period_uncertainty=0.02513504297496638,
    T0=3561.1449146762434,
    duration=0.12827804892073855,  # days
    depth=0.9883249937702906,  # relative in-transit flux
)


def test_reads_current_h5io_format(tmp_path):
    path = tmp_path / "tic123_tls.h5"
    _write_current_format(path, **_COMMON_TLS_FIELDS, depth_mean=[0.9901, 0.000121])

    values = read_h5_transit_params(str(path), mission="tess")

    assert values["period"] == pytest.approx(4.319534613356017)
    assert values["period_err"] == pytest.approx(0.02513504297496638)
    assert values["epoch"] == pytest.approx(3561.1449146762434 + 2_457_000.0)
    assert values["duration"] == pytest.approx(0.12827804892073855 * 24.0)
    assert values["depth"] == pytest.approx((1.0 - 0.9883249937702906) * 1e6)
    assert values["depth_err"] == pytest.approx(0.000121 * 1e6)
    assert values["epoch_err"] is None
    assert values["duration_err"] is None


def test_reads_legacy_deepdish_format(tmp_path):
    path = tmp_path / "tic123_tls_legacy.h5"
    _write_legacy_format(path, depth_mean=(0.9901, 0.000121), **_COMMON_TLS_FIELDS)

    values = read_h5_transit_params(str(path), mission="tess")

    assert values["period"] == pytest.approx(4.319534613356017)
    assert values["period_err"] == pytest.approx(0.02513504297496638)
    assert values["epoch"] == pytest.approx(3561.1449146762434 + 2_457_000.0)
    assert values["duration"] == pytest.approx(0.12827804892073855 * 24.0)
    assert values["depth"] == pytest.approx((1.0 - 0.9883249937702906) * 1e6)
    assert values["depth_err"] == pytest.approx(121.0, abs=1.0)


def test_k2_mission_uses_k2_bjd_offset(tmp_path):
    path = tmp_path / "epic123_tls.h5"
    _write_current_format(path, **_COMMON_TLS_FIELDS)

    values = read_h5_transit_params(str(path), mission="k2")

    assert values["epoch"] == pytest.approx(3561.1449146762434 + 2_454_833.0)


def test_missing_fields_are_left_as_none(tmp_path):
    path = tmp_path / "sparse_tls.h5"
    _write_current_format(path, period=3.0)

    values = read_h5_transit_params(str(path), mission="tess")

    assert values["period"] == pytest.approx(3.0)
    assert values["period_err"] is None
    assert values["epoch"] is None
    assert values["duration"] is None
    assert values["depth"] is None
    assert values["depth_err"] is None


def test_depth_below_one_half_is_used_directly(tmp_path):
    # A depth already expressed as fractional dimming (< 0.5) is returned
    # unchanged rather than reinterpreted as "1 - depth".
    path = tmp_path / "frac_depth_tls.h5"
    _write_current_format(path, depth=0.02)

    values = read_h5_transit_params(str(path), mission="tess")

    assert values["depth"] == pytest.approx(0.02 * 1e6)


def test_numpy_scalar_types_are_handled(tmp_path):
    # Legacy files store numpy scalar dtypes (np.float64/np.int64) as h5py
    # attrs; the reader must coerce these to plain Python floats.
    path = tmp_path / "np_scalars_tls.h5"
    with h5py.File(path, "w") as f:
        f.attrs["period"] = np.float64(4.32)
        f.attrs["period_uncertainty"] = np.float64(0.01)
        f.attrs["T0"] = np.float64(3561.14)
        f.attrs["duration"] = np.float64(0.128)
        f.attrs["depth"] = np.float64(0.99)

    values = read_h5_transit_params(str(path), mission="tess")

    assert isinstance(values["period"], float)
    assert values["period"] == pytest.approx(4.32)


def test_read_h5_lightcurve_returns_raw_arrays_and_metadata(tmp_path):
    path = tmp_path / "tic123_with_lc_tls.h5"
    time_raw = np.array([3561.10, 3561.12, 3561.14])
    flux_raw = np.array([1.001, 0.998, 1.000])
    err_raw = np.array([0.001, 0.001, 0.001])
    _write_current_format_with_arrays(
        path,
        {"time_raw": time_raw, "flux_raw": flux_raw, "err_raw": err_raw},
        pipeline="tess-spoc",
        sector=37,
        exptime=600.0,
        cadence="long",
    )

    lc = read_h5_lightcurve(str(path))

    assert lc is not None
    np.testing.assert_allclose(lc["time"], time_raw)
    np.testing.assert_allclose(lc["flux"], flux_raw)
    np.testing.assert_allclose(lc["flux_err"], err_raw)
    assert lc["pipeline"] == "tess-spoc"
    assert lc["sector"] == 37
    assert lc["exptime"] == pytest.approx(600.0)
    assert lc["cadence"] == "long"


def test_read_h5_lightcurve_returns_none_without_raw_arrays(tmp_path):
    # A file that only has TLS results (no time_raw/flux_raw/err_raw), e.g.
    # an iterative-search companion h5 saved from the bare TLS results dict.
    path = tmp_path / "tic123_results_only_tls.h5"
    _write_current_format(path, period=3.5, T0=3561.14)

    assert read_h5_lightcurve(str(path)) is None


def _tls_results_dict(**overrides):
    results = dict(
        period=4.319534613356017,
        period_uncertainty=0.02513504297496638,
        T0=3561.1449146762434 + 2_457_000.0,  # absolute BJD, as blind_search feeds it in
        duration=0.1,
        correct_duration=0.12827804892073855,
        depth=0.9883249937702906,
        depth_mean=(0.9901, 0.000121),
    )
    results.update(overrides)
    return results


def test_write_h5_tls_result_round_trips_through_read_h5_transit_params(tmp_path):
    path = tmp_path / "candidate_1_tls.h5"
    write_h5_tls_result(str(path), _tls_results_dict(), mission="tess")

    values = read_h5_transit_params(str(path), mission="tess")

    assert values["period"] == pytest.approx(4.319534613356017)
    assert values["period_err"] == pytest.approx(0.02513504297496638)
    assert values["epoch"] == pytest.approx(3561.1449146762434 + 2_457_000.0)
    # correct_duration (not the raw TLS duration) is what gets written, in days.
    assert values["duration"] == pytest.approx(0.12827804892073855 * 24.0)
    assert values["depth"] == pytest.approx((1.0 - 0.9883249937702906) * 1e6)
    assert values["depth_err"] == pytest.approx(0.000121 * 1e6)


def test_write_h5_tls_result_falls_back_to_duration_without_correct_duration(tmp_path):
    path = tmp_path / "candidate_2_tls.h5"
    results = _tls_results_dict()
    del results["correct_duration"]
    write_h5_tls_result(str(path), results, mission="tess")

    values = read_h5_transit_params(str(path), mission="tess")

    assert values["duration"] == pytest.approx(0.1 * 24.0)


def test_write_h5_tls_result_uses_mission_bjd_offset(tmp_path):
    path = tmp_path / "candidate_k2_tls.h5"
    results = _tls_results_dict(T0=3561.14 + 2_454_833.0)
    write_h5_tls_result(str(path), results, mission="k2")

    values = read_h5_transit_params(str(path), mission="k2")

    assert values["epoch"] == pytest.approx(3561.14 + 2_454_833.0)


def test_write_h5_tls_result_embeds_lightcurve_readable_by_read_h5_lightcurve(tmp_path):
    path = tmp_path / "candidate_1_tls.h5"
    time = np.array([2459307.10, 2459307.12, 2459307.14])
    flux = np.array([1.001, 0.998, 1.000])
    flux_err = np.array([0.001, 0.001, 0.001])
    write_h5_tls_result(
        str(path),
        _tls_results_dict(),
        lightcurve={"time": time, "flux": flux, "flux_err": flux_err},
        mission="tess",
    )

    lc = read_h5_lightcurve(str(path))

    assert lc is not None
    np.testing.assert_allclose(lc["time"], time - 2_457_000.0)
    np.testing.assert_allclose(lc["flux"], flux)
    np.testing.assert_allclose(lc["flux_err"], flux_err)


def test_write_h5_tls_result_without_lightcurve_has_no_raw_arrays(tmp_path):
    path = tmp_path / "candidate_no_lc_tls.h5"
    write_h5_tls_result(str(path), _tls_results_dict(), mission="tess")

    assert read_h5_lightcurve(str(path)) is None


def test_write_h5_tls_result_records_optional_segment_metadata(tmp_path):
    path = tmp_path / "candidate_meta_tls.h5"
    time = np.array([2459307.10, 2459307.12])
    flux = np.array([1.001, 0.998])
    flux_err = np.array([0.001, 0.001])
    write_h5_tls_result(
        str(path),
        _tls_results_dict(),
        lightcurve={"time": time, "flux": flux, "flux_err": flux_err},
        mission="tess",
        pipeline="spoc",
        sector=90,
        exptime=120.0,
        cadence="short",
    )

    lc = read_h5_lightcurve(str(path))

    assert lc["pipeline"] == "spoc"
    assert lc["sector"] == 90
    assert lc["exptime"] == pytest.approx(120.0)
    assert lc["cadence"] == "short"
