"""Tests for per-companion primary-transit prior support."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from allesfitter import computer, config
from tests.chromatic._helpers import (
    TRUE_RR_TESS,
    common_orbital_rows,
    dilution_rows,
    err_baseline_rows,
    ldc_rows,
    write_data_csv,
    write_params,
    write_settings,
)


@pytest.fixture(autouse=True)
def reset_basement():
    config.BASEMENT = None
    yield
    config.BASEMENT = None


def _prior_basement(*, required):
    return SimpleNamespace(
        settings={
            "companions_phot": ["b"],
            "companions_all": ["b"],
            "inst_all": [],
            "require_b_transit": required,
            "use_host_density_prior": False,
            "use_tidal_eccentricity_prior": False,
        },
        external_priors={},
    )


def _params(*, rsuma=0.1, cosi=0.05, f_s=0.0, f_c=0.0):
    return {
        "b_rr": 0.1,
        "b_rsuma": rsuma,
        "b_cosi": cosi,
        "b_f_s": f_s,
        "b_f_c": f_c,
        "b_ecc": f_s**2 + f_c**2,
        "b_radius_1": rsuma / 1.1,
    }


def test_required_transit_accepts_transiting_circular_geometry():
    config.BASEMENT = _prior_basement(required=True)
    assert computer.calculate_external_priors(_params(rsuma=0.1, cosi=0.05)) == 0.0


def test_required_transit_rejects_non_transiting_circular_geometry():
    config.BASEMENT = _prior_basement(required=True)
    assert computer.calculate_external_priors(_params(rsuma=0.1, cosi=0.2)) == -np.inf


def test_optional_transit_preserves_non_transiting_geometry():
    config.BASEMENT = _prior_basement(required=False)
    assert computer.calculate_external_priors(_params(rsuma=0.1, cosi=0.2)) == 0.0


def test_required_transit_uses_eccentric_primary_impact_parameter():
    config.BASEMENT = _prior_basement(required=True)
    # e=0.25 and omega=90 deg give a primary-transit correction of 0.75.
    # Thus cos(i)=0.12 transits for rsuma=0.10 even though cos(i)>rsuma.
    assert (
        computer.calculate_external_priors(
            _params(rsuma=0.1, cosi=0.12, f_s=np.sqrt(0.25), f_c=0.0)
        )
        == 0.0
    )


@pytest.mark.parametrize(("raw", "expected"), [("True", True), ("False", False)])
def test_requirement_setting_is_parsed_per_photometric_companion(tmp_path, raw, expected):
    d = tmp_path / "required_transit"
    d.mkdir()
    time = np.linspace(0.0, 10.0, 100)
    write_data_csv(d / "tess.csv", time, np.ones_like(time), np.full_like(time, 1e-3))
    write_settings(
        d,
        inst_phot=["tess"],
        extra=[f"require_b_transit,{raw}"],
    )
    rows = (
        [{"name": "b_rr", "value": TRUE_RR_TESS, "fit": 0, "bounds": "uniform 0 0.3"}]
        + common_orbital_rows(fit_orbital=False)
        + dilution_rows(["tess"])
        + err_baseline_rows(["tess"])
        + ldc_rows("tess")
    )
    write_params(d / "params.csv", rows=rows)

    config.init(str(d))
    assert config.BASEMENT.settings["require_b_transit"] is expected


def test_requirement_defaults_false_for_legacy_settings(tmp_path):
    d = tmp_path / "legacy_transit"
    d.mkdir()
    time = np.linspace(0.0, 10.0, 100)
    write_data_csv(d / "tess.csv", time, np.ones_like(time), np.full_like(time, 1e-3))
    write_settings(d, inst_phot=["tess"])
    rows = (
        [{"name": "b_rr", "value": TRUE_RR_TESS, "fit": 0, "bounds": "uniform 0 0.3"}]
        + common_orbital_rows(fit_orbital=False)
        + dilution_rows(["tess"])
        + err_baseline_rows(["tess"])
        + ldc_rows("tess")
    )
    write_params(d / "params.csv", rows=rows)

    config.init(str(d))
    assert config.BASEMENT.settings["require_b_transit"] is False
