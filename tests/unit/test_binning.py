"""Integration tests for the ``binning`` setting (Basement.load_data).

A positive ``binning`` value (bin width in days) bins each photometric light
curve at load time. These tests build a dense, uniform-cadence datadir, run
``config.init``, and assert the loaded ``self.data[inst]`` arrays are binned and
row-aligned (covariates included), plus the ``binning >= baseline`` guard.

Reuses the chromatic fixtures' row/CSV builders.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.chromatic._helpers import (
    TRUE_EPOCH,
    TRUE_RR_TESS,
    common_orbital_rows,
    dilution_rows,
    err_baseline_rows,
    ldc_rows,
    write_data_csv,
    write_params,
    write_settings,
)

_BIN_30MIN = 0.0208333  # days
_N_RAW = 1440           # 2-min cadence over 2 days
_SPAN_DAYS = 2.0


@pytest.fixture(autouse=True)
def reset_basement():
    from allesfitter import config

    config.BASEMENT = None
    yield
    config.BASEMENT = None


def _dense_datadir(tmp_path, *, binning=None, with_covariate=False):
    d = tmp_path / "binning_dd"
    d.mkdir()
    t = TRUE_EPOCH - 1.0 + np.linspace(0.0, _SPAN_DAYS, _N_RAW)
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

    extra = [f"binning,{binning}"] if binning is not None else None
    write_settings(d, inst_phot=["tess"], bandpass=None, extra=extra)
    rows = (
        [{"name": "b_rr", "value": TRUE_RR_TESS, "fit": 1,
          "bounds": "uniform 0 0.3", "label": "rr"}]
        + common_orbital_rows(fit_orbital=False)
        + dilution_rows(["tess"])
        + err_baseline_rows(["tess"])
        + ldc_rows("tess")
    )
    write_params(d / "params.csv", rows=rows)
    return d


def test_binning_reduces_point_count(tmp_path):
    from allesfitter import config

    config.init(str(_dense_datadir(tmp_path, binning=_BIN_30MIN)))
    t = config.BASEMENT.data["tess"]["time"]
    # 2-day span / 30 min ≈ 96 bins, far fewer than 1440 raw points.
    assert 50 < len(t) < 150
    assert abs(np.median(np.diff(t)) - _BIN_30MIN) < 0.01
    # every per-instrument array stays aligned with the binned time grid
    data = config.BASEMENT.data["tess"]
    assert len(data["err_scales_flux"]) == len(t)
    assert len(data["custom_series"]) == len(t)


def test_binning_keeps_covariate_aligned(tmp_path):
    from allesfitter import config

    config.init(str(_dense_datadir(tmp_path, binning=_BIN_30MIN, with_covariate=True)))
    data = config.BASEMENT.data["tess"]
    n = len(data["time"])
    assert len(data["covariates"]["airmass"]) == n
    am = data["covariates"]["airmass"]
    assert 1.0 <= am.min() <= am.max() <= 2.0  # mean-binned stays in range


def test_no_binning_keeps_all_points(tmp_path):
    from allesfitter import config

    config.init(str(_dense_datadir(tmp_path, binning=None)))
    assert len(config.BASEMENT.data["tess"]["time"]) == _N_RAW


def test_binning_ge_baseline_raises(tmp_path):
    from allesfitter import config

    d = _dense_datadir(tmp_path, binning=5.0)  # > 2-day baseline
    with pytest.raises(ValueError) as exc:
        config.init(str(d))
    assert "baseline" in str(exc.value)


def _two_inst_datadir(tmp_path, *, extra):
    """A datadir with two dense, identical-cadence photometric instruments."""
    d = tmp_path / "binning_two"
    d.mkdir()
    t = TRUE_EPOCH - 1.0 + np.linspace(0.0, _SPAN_DAYS, _N_RAW)
    rng = np.random.default_rng(0)
    insts = ["tess", "muscat"]
    for inst in insts:
        flux = 1.0 + rng.normal(0.0, 5e-4, _N_RAW)
        write_data_csv(d / f"{inst}.csv", t, flux, np.full(_N_RAW, 5e-4))
    write_settings(d, inst_phot=insts, bandpass=None, extra=extra)
    rows = (
        [{"name": "b_rr", "value": TRUE_RR_TESS, "fit": 1,
          "bounds": "uniform 0 0.3", "label": "rr"}]
        + common_orbital_rows(fit_orbital=False)
        + dilution_rows(insts)
        + err_baseline_rows(insts)
        + ldc_rows("tess")
        + ldc_rows("muscat")
    )
    write_params(d / "params.csv", rows=rows)
    return d


def test_per_instrument_binning_only_bins_target(tmp_path):
    from allesfitter import config

    # No global binning; bin TESS only via binning_tess.
    d = _two_inst_datadir(tmp_path, extra=[f"binning_tess,{_BIN_30MIN}"])
    config.init(str(d))
    assert 50 < len(config.BASEMENT.data["tess"]["time"]) < 150  # binned
    assert len(config.BASEMENT.data["muscat"]["time"]) == _N_RAW  # untouched


def test_per_instrument_override_disables_when_global_set(tmp_path):
    from allesfitter import config

    # Global binning bins everything; binning_muscat (empty) turns it off there.
    d = _two_inst_datadir(
        tmp_path, extra=[f"binning,{_BIN_30MIN}", "binning_muscat,"]
    )
    config.init(str(d))
    assert 50 < len(config.BASEMENT.data["tess"]["time"]) < 150  # global applies
    assert len(config.BASEMENT.data["muscat"]["time"]) == _N_RAW  # override off


def test_binning_auto_sets_t_exp_and_n_int(tmp_path):
    from allesfitter import config

    config.init(str(_dense_datadir(tmp_path, binning=_BIN_30MIN)))
    s = config.BASEMENT.settings
    assert s["t_exp_tess"] == _BIN_30MIN     # auto-derived from binning
    assert s["t_exp_n_int_tess"] == 10       # seeded default


def test_binning_does_not_override_explicit_t_exp(tmp_path):
    from allesfitter import config

    # Global binning bins both. tess has an explicit (different) t_exp → kept;
    # muscat has none → t_exp auto-set to the bin width.
    d = _two_inst_datadir(
        tmp_path, extra=[f"binning,{_BIN_30MIN}", "t_exp_tess,0.005"]
    )
    config.init(str(d))
    s = config.BASEMENT.settings
    assert s["t_exp_tess"] == 0.005          # explicit value preserved
    assert s["t_exp_muscat"] == _BIN_30MIN   # auto-derived


def test_binning_model_evaluates_with_supersampling(tmp_path):
    """Regression: auto-set n_int>1 must not overrun ellc's output buffer.

    Binning auto-sets t_exp_n_int=10, so the initial-guess model goes through
    ellc.fluxes with supersampling — which used to raise IndexError because the
    C output buffer was sized to n_obs instead of the n_int-times-longer t_calc.
    """
    from allesfitter import config
    from allesfitter.computer import calculate_model, update_params

    config.init(str(_dense_datadir(tmp_path, binning=_BIN_30MIN)))
    assert config.BASEMENT.settings["t_exp_n_int_tess"] == 10
    p = update_params(config.BASEMENT.theta_0)
    model = calculate_model(p, "tess", "flux", xx=None)  # ellc.fluxes, n_int=10
    assert len(model) == len(config.BASEMENT.data["tess"]["time"])
    assert np.isfinite(model).all()
