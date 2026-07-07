"""Tests for the JSON-payload -> FitConfig builder."""

from __future__ import annotations

import pytest

from allesfitter.webgui import formparse


def test_builds_config_and_staging_items():
    payload = {
        "target": "TOI-6715",
        "companions": [{"name": "b", "period": 2.86, "epoch": 2457000.0, "period_err": 1e-5}],
        "instruments": [
            {
                "label": "m4g",
                "band": "g",
                "data_file": "/d/m4g.csv",
                "baseline": "sample_GP_Matern32",
            },
            {
                "label": "cpt_z",
                "band": "z",
                "data_file": "/d/cpt_z.csv",
                "baseline": "sample_linear",
            },
        ],
        "share_groups": [["m4g", "m4r"]],
        "chromatic": "auto",
    }
    cfg, staging = formparse.build_fit_config(payload)
    assert cfg.target == "TOI-6715"
    assert [i.label for i in cfg.instruments] == ["m4g", "cpt_z"]
    assert cfg.chromatic is None  # "auto" -> None
    assert cfg.share_groups == (("m4g", "m4r"),)
    assert staging == [("m4g", "/d/m4g.csv"), ("cpt_z", "/d/cpt_z.csv")]
    # period uncertainty threaded into a normal prior
    assert cfg.companions[0].period.bounds.startswith("normal")


def test_band_inferred_when_missing():
    payload = {"target": "T", "instruments": [{"label": "qlp600", "data_file": ""}]}
    cfg, _ = formparse.build_fit_config(payload)
    assert cfg.instruments[0].band == "tess"


def test_missing_target_raises():
    with pytest.raises(ValueError, match="target is required"):
        formparse.build_fit_config({"instruments": [{"label": "x", "band": "g"}]})


def test_missing_instruments_raises():
    with pytest.raises(ValueError, match="instrument"):
        formparse.build_fit_config({"target": "T", "instruments": []})


def test_instrument_without_label_raises():
    with pytest.raises(ValueError, match="label"):
        formparse.build_fit_config({"target": "T", "instruments": [{"band": "g"}]})


def test_chromatic_explicit_bool_passthrough():
    cfg, _ = formparse.build_fit_config(
        {"target": "T", "instruments": [{"label": "m4g", "band": "g"}], "chromatic": "false"}
    )
    assert cfg.chromatic is False
