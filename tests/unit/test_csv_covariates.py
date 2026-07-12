"""Tests for named ancillary covariate columns in per-instrument CSVs.

Covers:
  1. Legacy 3-column CSV → covariates dict empty, custom_series = zeros.
  2. Legacy 4-column CSV → custom_series = column 3, covariates empty.
  3. Headered N-column CSV → covariates keyed by header names, first
     ancillary column aliased to custom_series.
  4. `baseline_<key>_<inst>_against=<unknown>` raises with actionable msg.
  5. hybrid_poly_1 + _against=<cov> recovers the right slope (end-to-end
     proof that the dispatch reaches the baseline routine with the
     covariate axis).
  6. GP baseline against a non-monotone covariate sorts internally and
     returns a finite log-likelihood (no celerite assertion fires).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

try:
    import allesfitter  # noqa: F401
    from allesfitter import config
    from allesfitter.basement import Basement, _load_inst_csv
    from allesfitter.mcmc import mcmc_lnlike
except Exception:
    pytest.skip(
        "allesfitter or its dependencies are not importable",
        allow_module_level=True,
    )


# ---------------------------------------------------------------------------
# CSV-loader unit tests (no Basement, no settings)
# ---------------------------------------------------------------------------


def test_loader_legacy_3col(tmp_path):
    p = tmp_path / "lc.csv"
    p.write_text("0.0,1.0,0.001\n1.0,1.001,0.001\n2.0,0.999,0.001\n")
    time, primary, primary_err, custom_series, covs = _load_inst_csv(str(p))
    assert len(time) == 3
    assert covs == {}
    assert np.all(custom_series == 0)


def test_loader_legacy_4col(tmp_path):
    p = tmp_path / "lc.csv"
    p.write_text("0.0,1.0,0.001,1.21\n1.0,1.001,0.001,1.22\n2.0,0.999,0.001,1.23\n")
    time, primary, primary_err, custom_series, covs = _load_inst_csv(str(p))
    # Legacy positional layout: 4th column → custom_series alias only,
    # no named-covariates dict (user never declared a name).
    assert "custom_series" in covs
    np.testing.assert_array_equal(covs["custom_series"], [1.21, 1.22, 1.23])
    np.testing.assert_array_equal(custom_series, [1.21, 1.22, 1.23])


def test_loader_header_named_covariates(tmp_path):
    p = tmp_path / "lc.csv"
    p.write_text(
        "#time,flux,flux_err,airmass,fwhm,sky\n"
        "0.0,1.000,0.001,1.21,1.45,250.0\n"
        "1.0,1.001,0.001,1.22,1.46,251.0\n"
        "2.0,0.999,0.001,1.23,1.47,252.0\n"
    )
    time, primary, primary_err, custom_series, covs = _load_inst_csv(str(p))
    assert set(covs.keys()) == {"airmass", "fwhm", "sky", "custom_series"}
    np.testing.assert_array_equal(covs["airmass"], [1.21, 1.22, 1.23])
    np.testing.assert_array_equal(covs["fwhm"], [1.45, 1.46, 1.47])
    np.testing.assert_array_equal(covs["sky"], [250.0, 251.0, 252.0])
    # Backward-compat: first ancillary column is also reachable as
    # 'custom_series' so existing `_against,custom_series` still works.
    np.testing.assert_array_equal(covs["custom_series"], [1.21, 1.22, 1.23])
    np.testing.assert_array_equal(custom_series, [1.21, 1.22, 1.23])


def test_loader_ignores_unrelated_hash_comment(tmp_path):
    """An ordinary `# This is a comment` line MUST NOT be misparsed as a
    header (no `time,flux,flux_err` schema tokens)."""
    p = tmp_path / "lc.csv"
    p.write_text(
        "# This is a comment, not a header line\n0.0,1.0,0.001,1.21\n1.0,1.001,0.001,1.22\n"
    )
    time, primary, primary_err, custom_series, covs = _load_inst_csv(str(p))
    # No header → legacy 4-col positional behaviour.
    assert "airmass" not in covs
    np.testing.assert_array_equal(covs["custom_series"], [1.21, 1.22])


# ---------------------------------------------------------------------------
# Basement / settings validation
# ---------------------------------------------------------------------------


def _params_csv() -> str:
    return (
        "#name,value,fit,bounds,label,unit,coupled_with\n"
        "b_rr,0.10,1,uniform 0.05 0.30,$R_b/R_*$,,\n"
        "b_rsuma,0.15,1,uniform 0.05 0.30,$R/a$,,\n"
        "b_cosi,0.10,1,uniform 0 1,$\\cos i$,,\n"
        "b_epoch,0.5,1,uniform 0.0 1.0,$T_0$,d,\n"
        "b_period,3.5,1,normal 3.5 0.1,$P$,d,\n"
        "b_f_c,0,0,uniform -1 1,,,\n"
        "b_f_s,0,0,uniform -1 1,,,\n"
        "dil_lco,0,0,uniform -1 1,,,\n"
        "host_ldc_q1_lco,0.5,1,uniform 0 1,,,\n"
        "host_ldc_q2_lco,0.5,1,uniform 0 1,,,\n"
        "ln_err_flux_lco,-6,1,uniform -10 -1,,,\n"
        "baseline_offset_flux_lco,0,1,uniform -0.01 0.01,,,\n"
        "baseline_slope_flux_lco,0,1,uniform -0.1 0.1,,,\n"
    )


def _settings_csv(against="time") -> str:
    return (
        "#name,value\n"
        "companions_phot,b\n"
        "companions_rv,\n"
        "inst_phot,lco\n"
        "inst_rv,\n"
        "time_format,BJD_TDB\n"
        "multiprocess,False\n"
        "print_progress,False\n"
        "fast_fit,False\n"
        "shift_epoch,False\n"
        "host_ld_law_lco,quad\n"
        "error_flux_lco,sample\n"
        "baseline_flux_lco,sample_linear\n"
        f"baseline_flux_lco_against,{against}\n"
    )


def _write_lc(path: Path, with_airmass: bool, with_header: bool):
    n = 200
    rng = np.random.default_rng(0)
    t = np.linspace(0.0, 1.0, n)
    airmass = 1.0 + 0.5 * np.cos(np.pi * t)
    # Synthetic flux with a small airmass-linear baseline and noise.
    flux = 1.0 - 0.01 * (airmass - 1.0) + rng.normal(0, 1e-3, n)
    err = np.full(n, 1e-3)
    lines = []
    if with_header:
        if with_airmass:
            lines.append("#time,flux,flux_err,airmass")
        else:
            lines.append("#time,flux,flux_err")
    for i in range(n):
        if with_airmass:
            lines.append(f"{t[i]:.10f},{flux[i]:.10f},{err[i]:.10f},{airmass[i]:.6f}")
        else:
            lines.append(f"{t[i]:.10f},{flux[i]:.10f},{err[i]:.10f}")
    path.write_text("\n".join(lines) + "\n")
    return t, flux, err, airmass


def _build_datadir(tmp_path, against="time", with_airmass=True, with_header=True):
    d = tmp_path / "data"
    d.mkdir()
    (d / "results").mkdir()
    (d / "params.csv").write_text(_params_csv())
    (d / "settings.csv").write_text(_settings_csv(against=against))
    _write_lc(d / "lco.csv", with_airmass=with_airmass, with_header=with_header)
    return d


def test_unknown_covariate_raises_actionable_message(tmp_path):
    d = _build_datadir(tmp_path, against="airmass", with_airmass=False, with_header=False)
    with pytest.raises(ValueError, match="has no column named 'airmass'"):
        Basement(str(d), quiet=True)


def test_basement_rejects_inconsistent_companion(tmp_path):
    """Cross-file consistency wired into Basement.__init__.

    A companion declared in settings.csv with no matching rows in params.csv
    is something allesfitter's own loaders silently tolerate (they auto-fill
    defaults for the missing body). The validator wired into Basement after
    load_params catches the mismatch and raises ConfigError. This exercises
    the wiring on a check that is unique to the validator (allesfitter already
    range-checks individual values, so value-outside-prior is caught earlier
    and would not isolate the wiring).
    """
    from allesfitter.validation import ConfigError

    d = _build_datadir(tmp_path)
    # Declare a phantom companion 'c' with no c_* parameter rows.
    settings = _settings_csv().replace("companions_phot,b\n", "companions_phot,b c\n")
    (d / "settings.csv").write_text(settings)
    with pytest.raises(ConfigError, match="companion 'c'"):
        Basement(str(d), quiet=True)


def test_basement_loads_covariate_axis_into_data(tmp_path):
    d = _build_datadir(tmp_path, against="airmass", with_airmass=True, with_header=True)
    b = Basement(str(d), quiet=True)
    assert "covariates" in b.data["lco"]
    assert "airmass" in b.data["lco"]["covariates"]
    assert len(b.data["lco"]["covariates"]["airmass"]) == len(b.data["lco"]["time"])


def test_stellar_var_setting_is_exact_not_instrument_suffix(tmp_path):
    d = _build_datadir(tmp_path)
    settings_path = d / "settings.csv"
    settings_path.write_text(settings_path.read_text() + "stellar_var_flux_lco,sample_linear\n")

    with pytest.raises(ValueError, match="stellar_var_flux_lco"):
        Basement(str(d), quiet=True)


def test_stellar_var_rv2_exact_setting_is_recognized(tmp_path):
    d = _build_datadir(tmp_path)
    settings_path = d / "settings.csv"
    settings_path.write_text(settings_path.read_text() + "stellar_var_rv2,none\n")

    b = Basement(str(d), quiet=True)
    assert b.settings["stellar_var_rv2"] == "none"


def test_hybrid_poly_against_airmass_matches_polyfit(tmp_path):
    """End-to-end proof that hybrid_poly_1 with _against=airmass receives
    the covariate as its abscissa and recovers a sensible linear fit."""
    d = tmp_path / "data"
    d.mkdir()
    (d / "results").mkdir()
    # Use hybrid_poly_1 (a one-line replacement for sample_linear) which
    # internally fits a polynomial to (x, y) where x is the regression
    # axis. Then assert the fitted baseline matches np.polyfit on the
    # covariate.
    (d / "params.csv").write_text(_params_csv())
    settings = _settings_csv(against="airmass").replace(
        "baseline_flux_lco,sample_linear", "baseline_flux_lco,hybrid_poly_1"
    )
    (d / "settings.csv").write_text(settings)
    t, flux, err, airmass = _write_lc(d / "lco.csv", with_airmass=True, with_header=True)

    b = Basement(str(d), quiet=True)
    from allesfitter.computer import calculate_baseline, update_params

    config.BASEMENT = b
    params = update_params(np.array(b.theta_0, dtype=float))
    bl = calculate_baseline(params, "lco", "flux", model=np.ones_like(flux), yerr_w=err)
    # Expected baseline = linear polyfit of (airmass, flux - 1) ≈ -0.01*(airmass-1)
    pf = np.polyfit(airmass, flux - 1.0, 1)
    expected = np.polyval(pf, airmass)
    # hybrid_poly internally rescales x → [0,1]; check that the fitted
    # baseline tracks the polyfit shape within a few mmag.
    assert np.allclose(bl - bl.mean(), expected - expected.mean(), atol=2e-3), (
        bl[:5],
        expected[:5],
    )


def test_gp_against_unsorted_covariate_runs(tmp_path):
    """GP baseline against an unsorted covariate must cosort inside the
    code path and produce a finite log-likelihood (no celerite assert)."""
    pytest.importorskip("celerite")
    d = tmp_path / "data"
    d.mkdir()
    (d / "results").mkdir()
    p = _params_csv() + (
        "baseline_gp_matern32_lnsigma_flux_lco,-5,1,uniform -10 -3,,,\n"
        "baseline_gp_matern32_lnrho_flux_lco, 0,1,uniform -1 3,,,\n"
    )
    (d / "params.csv").write_text(p)
    settings = _settings_csv(against="airmass").replace(
        "baseline_flux_lco,sample_linear", "baseline_flux_lco,sample_GP_Matern32"
    )
    (d / "settings.csv").write_text(settings)
    # Write a CSV whose airmass column is intentionally non-monotone
    # (oscillates). Time is sorted; airmass is not.
    n = 200
    rng = np.random.default_rng(2)
    t = np.linspace(0.0, 1.0, n)
    airmass = 1.3 + 0.2 * np.sin(2 * np.pi * 3 * t)  # 3-cycle wiggle
    flux = 1.0 + rng.normal(0, 1e-3, n)
    err = np.full(n, 1e-3)
    lines = ["#time,flux,flux_err,airmass"]
    for i in range(n):
        lines.append(f"{t[i]:.10f},{flux[i]:.10f},{err[i]:.10f},{airmass[i]:.6f}")
    (d / "lco.csv").write_text("\n".join(lines) + "\n")

    b = Basement(str(d), quiet=True)
    config.BASEMENT = b
    ll = mcmc_lnlike(np.array(b.theta_0, dtype=float))
    assert np.isfinite(ll), ll
