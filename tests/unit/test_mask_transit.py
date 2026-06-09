"""Tests for the ``mask_transit`` setting (Basement.load_settings/params/data).

``mask_transit,True`` removes the in-transit points from each photometric
light curve at load time — the inverse of ``fast_fit`` — so the user can model
``stellar_var`` or spots on the out-of-transit data alone. These tests build a
datadir spanning several transits, run ``config.init``, and assert:

* in-transit points are dropped and the surviving grid avoids every transit
  window (covariates stay row-aligned),
* the default (setting absent) keeps all points,
* a free transit parameter (fit=1) is rejected, and
* ``mask_transit`` and ``fast_fit`` together are rejected.

Reuses the chromatic fixtures' row/CSV builders.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.chromatic._helpers import (
    TRUE_EPOCH,
    TRUE_PERIOD,
    TRUE_RR_TESS,
    common_orbital_rows,
    dilution_rows,
    err_baseline_rows,
    ldc_rows,
    write_data_csv,
    write_params,
    write_settings,
)

_FAST_FIT_WIDTH = 8.0 / 24.0   # Basement default transit half-window source
_SPAN_PERIODS = 4
_N_RAW = 2000


@pytest.fixture(autouse=True)
def reset_basement():
    from allesfitter import config

    config.BASEMENT = None
    yield
    config.BASEMENT = None


def _mask_datadir(tmp_path, *, mask_transit=None, fit_rr=False,
                  with_covariate=False, extra_settings=None):
    d = tmp_path / "mask_dd"
    d.mkdir()
    # dense, uniform cadence spanning several transits centred on TRUE_EPOCH
    t = TRUE_EPOCH - 0.5 * _SPAN_PERIODS * TRUE_PERIOD + np.linspace(
        0.0, _SPAN_PERIODS * TRUE_PERIOD, _N_RAW)
    rng = np.random.default_rng(0)
    flux = 1.0 + rng.normal(0.0, 5e-4, _N_RAW)
    err = np.full(_N_RAW, 5e-4)
    if with_covariate:
        airmass = np.linspace(1.0, 2.0, _N_RAW)
        (d / "tess.csv").write_text(
            "#time,flux,flux_err,airmass\n"
            + "\n".join(f"{a},{b},{c},{g}" for a, b, c, g in zip(t, flux, err, airmass))
            + "\n"
        )
    else:
        write_data_csv(d / "tess.csv", t, flux, err)

    extra = list(extra_settings) if extra_settings else []
    if mask_transit is not None:
        extra.append(f"mask_transit,{mask_transit}")
    write_settings(d, inst_phot=["tess"], bandpass=None, extra=extra)
    rows = (
        [{"name": "b_rr", "value": TRUE_RR_TESS, "fit": 1 if fit_rr else 0,
          "bounds": "uniform 0 0.3", "label": "rr"}]
        + common_orbital_rows(fit_orbital=False)  # all transit params fixed
        + dilution_rows(["tess"])
        + err_baseline_rows(["tess"])
        + ldc_rows("tess")
    )
    write_params(d / "params.csv", rows=rows)
    return d


def _in_transit_mask(time):
    """Phase-fold and mark points within +/- fast_fit_width/2 of a transit."""
    phase = ((time - TRUE_EPOCH + 0.5 * TRUE_PERIOD) % TRUE_PERIOD) - 0.5 * TRUE_PERIOD
    return np.abs(phase) <= _FAST_FIT_WIDTH / 2.0


def test_mask_transit_removes_in_transit_points(tmp_path):
    from allesfitter import config

    config.init(str(_mask_datadir(tmp_path, mask_transit="True")))
    t = config.BASEMENT.data["tess"]["time"]
    # something was removed, but most of the (out-of-transit) data survives
    assert 0 < len(t) < _N_RAW
    # no surviving point falls inside a transit window
    assert not _in_transit_mask(t).any()
    # the dropped count matches the in-transit count of the raw grid
    raw = config.BASEMENT.fulldata["tess"]["time"]
    assert len(raw) == _N_RAW
    assert len(t) == int(np.sum(~_in_transit_mask(raw)))


def test_mask_transit_keeps_covariate_aligned(tmp_path):
    from allesfitter import config

    config.init(str(_mask_datadir(tmp_path, mask_transit="True", with_covariate=True)))
    data = config.BASEMENT.data["tess"]
    n = len(data["time"])
    assert len(data["covariates"]["airmass"]) == n
    assert len(data["err_scales_flux"]) == n
    assert len(data["custom_series"]) == n


def test_no_mask_transit_keeps_all_points(tmp_path):
    from allesfitter import config

    config.init(str(_mask_datadir(tmp_path, mask_transit=None)))
    assert config.BASEMENT.settings["mask_transit"] is False
    assert len(config.BASEMENT.data["tess"]["time"]) == _N_RAW


def test_mask_transit_false_keeps_all_points(tmp_path):
    from allesfitter import config

    config.init(str(_mask_datadir(tmp_path, mask_transit="False")))
    assert config.BASEMENT.settings["mask_transit"] is False
    assert len(config.BASEMENT.data["tess"]["time"]) == _N_RAW


def test_mask_transit_with_free_transit_param_raises(tmp_path):
    from allesfitter import config

    d = _mask_datadir(tmp_path, mask_transit="True", fit_rr=True)
    with pytest.raises(ValueError) as exc:
        config.init(str(d))
    msg = str(exc.value)
    assert "mask_transit" in msg
    assert "b_rr" in msg


def test_mask_transit_with_fast_fit_raises(tmp_path):
    from allesfitter import config

    d = _mask_datadir(tmp_path, mask_transit="True",
                      extra_settings=["fast_fit,True", "fast_fit_width,0.3"])
    # write_settings already emits fast_fit,False; the extra row appears later
    # and wins (later keys overwrite earlier ones in the settings dict).
    with pytest.raises(ValueError) as exc:
        config.init(str(d))
    assert "mutually exclusive" in str(exc.value)
