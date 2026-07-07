"""Round-trip tests for the web-GUI config writer.

The strongest assertion available is that a generated ``settings.csv`` /
``params.csv`` loads into the engine's own :class:`allesfitter.basement.Basement`
without raising — that reuses the exact validator (unknown-key rejection,
share-group validation, bandpass/companion parsing) the real fit uses. On top of
that we assert the TOI-6715-shaped structure the writer must produce.
"""

from __future__ import annotations

import numpy as np
import pytest

from allesfitter.webgui import config_writer as cw
from allesfitter.webgui import models as m


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _write_lc(path, *, n=240, t0=2457000.0, span=8.0, cov_cols=None, seed=0):
    """Write a minimal valid instrument CSV (headered, optional covariates)."""
    rng = np.random.default_rng(seed)
    t = np.linspace(t0 - 1.0, t0 - 1.0 + span, n)
    flux = 1.0 + rng.normal(0, 1e-3, n)
    ferr = np.full(n, 1e-3)
    header_cols = ["#time", "flux", "flux_err"]
    columns = [t, flux, ferr]
    for c in cov_cols or []:
        header_cols.append(c)
        columns.append(1.0 + rng.normal(0, 0.1, n))
    lines = [",".join(header_cols)]
    for row in np.column_stack(columns):
        lines.append(",".join(str(x) for x in row))
    path.write_text("\n".join(lines) + "\n")


def _toi6715_like_config():
    """A chromatic multi-band joint fit mirroring TOI-6715's structure:
    TESS per-inst GP, MuSCAT4 joint-GP share group, LCO linear, bands shared
    across instruments (two z instruments)."""
    insts = [
        m.InstrumentSpec("qlp1800", "tess", baseline="sample_GP_Matern32"),
        m.InstrumentSpec("qlp600", "tess", baseline="sample_GP_Matern32"),
        m.InstrumentSpec("m4g", "g", baseline="sample_GP_Matern32"),
        m.InstrumentSpec("m4r", "r", baseline="sample_GP_Matern32"),
        m.InstrumentSpec("m4i", "i", baseline="sample_GP_Matern32"),
        m.InstrumentSpec("m4z", "z", baseline="sample_GP_Matern32"),
        m.InstrumentSpec("cpt_g", "g", baseline="sample_linear"),
        m.InstrumentSpec("lsc_i", "i", baseline="sample_linear"),
        m.InstrumentSpec("cpt_z", "z", baseline="sample_linear"),
        m.InstrumentSpec("coj_z", "z", baseline="sample_linear"),
    ]
    return m.FitConfig(
        target="TOI-6715",
        companions=[m.default_companion("b", period=2.862577, epoch=2457000.0, rr=0.05)],
        instruments=insts,
        share_groups=(("m4g", "m4r", "m4i", "m4z"),),
        use_host_density_prior=False,  # keep round-trip free of params_star.csv
        fast_fit=False,  # synthetic data has no real transit to window on
    )


def _stage_data(cfg, datadir):
    for k, inst in enumerate(cfg.instruments):
        _write_lc(datadir / f"{inst.label}.csv", seed=k + 1)


# --------------------------------------------------------------------------
# structural assertions (no engine)
# --------------------------------------------------------------------------
def test_settings_bandpass_parallels_inst_phot():
    cfg = _toi6715_like_config()
    settings = cw.build_settings(cfg)
    inst_line = next(line for line in settings.splitlines() if line.startswith("inst_phot,"))
    bp_line = next(line for line in settings.splitlines() if line.startswith("bandpass,"))
    insts = inst_line.split(",", 1)[1].split()
    bands = bp_line.split(",", 1)[1].split()
    assert insts == [i.label for i in cfg.instruments]
    assert len(bands) == len(insts)
    assert bands[insts.index("m4g")] == "g"
    assert bands[insts.index("coj_z")] == "z"


def test_chromatic_emits_rr_per_band():
    cfg = _toi6715_like_config()
    params = cw.build_params(cfg)
    names = {line.split(",", 1)[0] for line in params.splitlines() if not line.startswith("#")}
    for band in ("g", "r", "i", "z", "tess"):
        assert f"b_rr_{band}" in names
        assert f"host_ldc_q1_{band}" in names
        assert f"host_ldc_q2_{band}" in names
    assert "b_rr" not in names  # per-band only in chromatic mode


def test_share_group_line_and_leader_only_gp_params():
    cfg = _toi6715_like_config()
    settings = cw.build_settings(cfg)
    params = cw.build_params(cfg)
    assert "baseline_share_flux,m4g:m4r:m4i:m4z" in settings
    names = {line.split(",", 1)[0] for line in params.splitlines() if not line.startswith("#")}
    # leader gets GP params ...
    assert "baseline_gp_matern32_lnsigma_flux_m4g" in names
    assert "baseline_gp_matern32_lnrho_flux_m4g" in names
    # ... followers do not
    for follower in ("m4r", "m4i", "m4z"):
        assert f"baseline_gp_matern32_lnsigma_flux_{follower}" not in names


def test_linear_instruments_get_offset_and_slope():
    cfg = _toi6715_like_config()
    params = cw.build_params(cfg)
    names = {line.split(",", 1)[0] for line in params.splitlines() if not line.startswith("#")}
    for inst in ("cpt_g", "lsc_i", "cpt_z", "coj_z"):
        assert f"baseline_offset_flux_{inst}" in names
        assert f"baseline_slope_flux_{inst}" in names


def test_explicit_achromatic_collapses_rr():
    cfg = _toi6715_like_config()
    cfg.chromatic = False
    params = cw.build_params(cfg)
    names = {line.split(",", 1)[0] for line in params.splitlines() if not line.startswith("#")}
    assert "b_rr" in names
    assert "b_rr_g" not in names


def test_hybrid_linear_multi_emits_cols_and_no_params():
    cfg = m.FitConfig(
        target="T",
        companions=[m.default_companion("b")],
        instruments=[
            m.InstrumentSpec(
                "m4g",
                "g",
                baseline="hybrid_linear_multi",
                baseline_cols=("Airmass", "FWHM(pix)"),
            )
        ],
        use_host_density_prior=False,
        fast_fit=False,
    )
    settings = cw.build_settings(cfg)
    params = cw.build_params(cfg)
    cols_line = next(
        line for line in settings.splitlines() if line.startswith("baseline_flux_m4g_cols,")
    )
    assert "Airmass" in cols_line and "FWHM(pix)" in cols_line and "bias" in cols_line
    names = {line.split(",", 1)[0] for line in params.splitlines() if not line.startswith("#")}
    assert not any(n.startswith("baseline_") for n in names)


def test_fitted_param_requires_bounds():
    bad = m.Prior(0.1, fit=True, bounds="")
    with pytest.raises(ValueError, match="empty bounds"):
        cw._param_row("b_rr", bad)


def test_unsupported_baseline_raises():
    cfg = m.FitConfig(
        target="T",
        companions=[m.default_companion("b")],
        instruments=[m.InstrumentSpec("x", "g", baseline="sample_GP_SHO")],
    )
    with pytest.raises(NotImplementedError, match="sample_GP_SHO"):
        cw.build_params(cfg)


# --------------------------------------------------------------------------
# round-trip through the real engine validator
# --------------------------------------------------------------------------
def test_toi6715_roundtrip_loads_in_basement(tmp_path, basement):
    cfg = _toi6715_like_config()
    cw.write_config(cfg, tmp_path)
    _stage_data(cfg, tmp_path)

    b = basement(str(tmp_path), quiet=True)

    # chromatic auto-detected True (5 unique bands)
    assert b.settings["chromatic"] is True
    # bandpass mapping parsed
    assert b.settings["bandpass"]["m4g"] == "g"
    assert b.settings["bandpass"]["coj_z"] == "z"
    # rr keyed by bandpass
    assert b.get_rr_key("b", "m4g") == "b_rr_g"
    assert b.get_rr_key("b", "coj_z") == "b_rr_z"
    # share-group followers inherited the leader's GP baseline type
    assert b.settings["baseline_flux_m4r"] == "sample_GP_Matern32"


def test_hybrid_linear_multi_roundtrip(tmp_path, basement):
    cfg = m.FitConfig(
        target="T",
        companions=[m.default_companion("b", period=2.0, epoch=2457000.0)],
        instruments=[
            m.InstrumentSpec(
                "m4g",
                "g",
                baseline="hybrid_linear_multi",
                baseline_cols=("Airmass", "FWHM(pix)"),
            )
        ],
        use_host_density_prior=False,
        fast_fit=False,
    )
    cw.write_config(cfg, tmp_path)
    _write_lc(tmp_path / "m4g.csv", cov_cols=["Airmass", "FWHM(pix)"])

    b = basement(str(tmp_path), quiet=True)
    assert list(b.data["m4g"]["design_matrix_cols"]) == ["Airmass", "FWHM(pix)", "bias"]
