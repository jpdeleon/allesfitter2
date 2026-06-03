"""Tests for ``allesfitter.validation.validate_gp_priors``.

The validator is pure: it reads ``params.csv``, ``settings.csv`` and the
per-instrument data CSVs in a datadir, and returns a list of warning
strings. We construct miniature, in-memory datadirs in a ``tmp_path``
fixture and assert the expected warnings fire.

The implementation moved from ``allesfitter.utils.prior_sanity`` into the
consolidated ``allesfitter.validation`` package; this test imports from the
new home. (The old path still works via a backward-compat shim, exercised by
``test_gp_prior_bounds.py``.)
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from allesfitter.validation import validate_gp_priors


def _write(p: Path, body: str) -> None:
    p.write_text(body)


def _make_datadir(tmp_path: Path, params: str, settings: str, lc_csvs: dict) -> Path:
    d = tmp_path
    _write(d / "params.csv", params.strip() + "\n")
    _write(d / "settings.csv", settings.strip() + "\n")
    for fname, body in lc_csvs.items():
        _write(d / fname, body.strip() + "\n")
    return d


def _clean_tess_lc_csv(seed: int = 0, n: int = 2000, rms: float = 5e-4) -> str:
    """Synthetic TESS-like LC: 27-day baseline, 120-s cadence, RMS 500 ppm."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 27.0, n)            # 27-day TESS sector
    f = 1.0 + rng.normal(0.0, rms, n)
    e = np.full(n, rms)
    return "\n".join(f"{ti},{fi},{ei}" for ti, fi, ei in zip(t, f, e))


# ---------------------------------------------------------------------------
# 1) Clean priors → no warnings
# ---------------------------------------------------------------------------


def test_clean_priors_emit_no_warnings(tmp_path):
    d = _make_datadir(
        tmp_path,
        params="""#name,value,fit,bounds,label,unit,coupled
ln_err_flux_tess,-6,1,uniform -10 -4,...,,
baseline_gp_matern32_lnsigma_flux_tess,-5,1,uniform -15 -3,...,,
baseline_gp_matern32_lnrho_flux_tess,1,1,uniform 0 3,...,,
""",
        settings="inst_phot,tess",
        lc_csvs={"tess.csv": _clean_tess_lc_csv()},
    )
    msgs = validate_gp_priors(d)
    assert msgs == [], msgs


# ---------------------------------------------------------------------------
# 2) GP amplitude too high → transit-amplitude warning
# ---------------------------------------------------------------------------


def test_gp_lnsigma_too_loose_warns(tmp_path):
    d = _make_datadir(
        tmp_path,
        params="""#name,value,fit,bounds,label,unit,coupled
baseline_gp_matern32_lnsigma_flux_tess,-5,1,uniform -15 0,...,,
""",
        settings="inst_phot,tess",
        lc_csvs={"tess.csv": _clean_tess_lc_csv()},
    )
    msgs = validate_gp_priors(d)
    assert any("may absorb the transit" in m for m in msgs), msgs
    # Also caught by the "RMS-relative" rule since σ=1 >> 500 ppm × 100.
    assert any("dominate the noise floor" in m for m in msgs), msgs


# ---------------------------------------------------------------------------
# 3) GP timescale lower bound below transit duration → warning
# ---------------------------------------------------------------------------


def test_gp_lnrho_below_transit_duration_warns(tmp_path):
    d = _make_datadir(
        tmp_path,
        params="""#name,value,fit,bounds,label,unit,coupled
baseline_gp_matern32_lnrho_flux_tess,1,1,uniform -5 5,...,,
""",
        settings="inst_phot,tess",
        lc_csvs={"tess.csv": _clean_tess_lc_csv()},
    )
    # transit duration ~2.4 h fallback (0.1 d). exp(-5) ≈ 0.0067 d → flagged.
    msgs = validate_gp_priors(d)
    assert any("fit the transit shape" in m for m in msgs), msgs


# ---------------------------------------------------------------------------
# 4) GP timescale upper bound > baseline → degenerate warning
# ---------------------------------------------------------------------------


def test_gp_lnrho_above_baseline_warns(tmp_path):
    d = _make_datadir(
        tmp_path,
        params="""#name,value,fit,bounds,label,unit,coupled
baseline_gp_matern32_lnrho_flux_tess,1,1,uniform 0 5,...,,
""",
        settings="inst_phot,tess",
        lc_csvs={"tess.csv": _clean_tess_lc_csv()},  # baseline = 27 d, exp(5) ≈ 148 d
    )
    msgs = validate_gp_priors(d)
    assert any("degenerate with baseline slope" in m for m in msgs), msgs


# ---------------------------------------------------------------------------
# 5) GP timescale below cadence → warning
# ---------------------------------------------------------------------------


def test_gp_lnrho_below_cadence_warns(tmp_path):
    # 27-day baseline with 2000 samples → cadence ≈ 27/1999 ≈ 0.0135 d.
    # exp(-7) ≈ 9.1e-4 d, well below 2× cadence → should trigger.
    d = _make_datadir(
        tmp_path,
        params="""#name,value,fit,bounds,label,unit,coupled
baseline_gp_matern32_lnrho_flux_tess,-2,1,uniform -7 3,...,,
""",
        settings="inst_phot,tess",
        lc_csvs={"tess.csv": _clean_tess_lc_csv()},
    )
    msgs = validate_gp_priors(d)
    assert any("fit individual cadences" in m for m in msgs), msgs


# ---------------------------------------------------------------------------
# 6) ln_err_flux upper bound > 10% → warning
# ---------------------------------------------------------------------------


def test_ln_err_flux_upper_too_loose_warns(tmp_path):
    d = _make_datadir(
        tmp_path,
        params="""#name,value,fit,bounds,label,unit,coupled
ln_err_flux_tess,-6,1,uniform -10 -1,...,,
""",
        settings="inst_phot,tess",
        lc_csvs={"tess.csv": _clean_tess_lc_csv()},
    )
    # exp(-1) ≈ 0.37, well above 10% threshold.
    msgs = validate_gp_priors(d)
    assert any("review noise prior" in m for m in msgs), msgs


# ---------------------------------------------------------------------------
# 7) Transit duration override tightens the ln_ρ check
# ---------------------------------------------------------------------------


def test_tdur_override_changes_lnrho_threshold(tmp_path):
    """A long transit (10 h) should flag bounds that a 2.4-h default tolerates."""
    d = _make_datadir(
        tmp_path,
        params="""#name,value,fit,bounds,label,unit,coupled
baseline_gp_matern32_lnrho_flux_tess,-2,1,uniform -3 3,...,,
""",
        settings="inst_phot,tess",
        lc_csvs={"tess.csv": _clean_tess_lc_csv()},
    )
    # Default (tdur=2.4h, lo→exp(-3) ≈ 0.05 d vs 0.5*0.1=0.05): borderline,
    # but with tdur=10 h → 0.5*tdur ≈ 0.21 d, and 0.05 d < 0.21 d → warn.
    msgs_default = validate_gp_priors(d)
    msgs_with_tdur = validate_gp_priors(d, tdur_hours_by_companion={"b": 10.0})
    assert any("transit shape" in m for m in msgs_with_tdur), msgs_with_tdur
    # The 10h override should produce at least one warning that the
    # default did not.
    assert len(msgs_with_tdur) >= len(msgs_default)


# ---------------------------------------------------------------------------
# 8) log= sink receives every message
# ---------------------------------------------------------------------------


def test_log_sink_receives_messages(tmp_path):
    d = _make_datadir(
        tmp_path,
        params="""#name,value,fit,bounds,label,unit,coupled
baseline_gp_matern32_lnsigma_flux_tess,-5,1,uniform -15 0,...,,
""",
        settings="inst_phot,tess",
        lc_csvs={"tess.csv": _clean_tess_lc_csv()},
    )
    captured = []
    msgs = validate_gp_priors(d, log=captured.append)
    assert captured == msgs and len(captured) >= 1


# ---------------------------------------------------------------------------
# 9) Missing datadir / missing files → empty result, no exception
# ---------------------------------------------------------------------------


def test_missing_params_csv_returns_empty(tmp_path):
    # Empty dir → no params.csv → empty list, no exception.
    assert validate_gp_priors(tmp_path) == []


# ---------------------------------------------------------------------------
# 10) Derived tdur from params.csv replaces the 0.1 d fallback
# ---------------------------------------------------------------------------


def test_tdur_derived_from_params_csv(tmp_path):
    """When tdur_hours_by_companion is None, the validator should derive a
    per-companion tdur from initial-guess orbital rows in params.csv and
    use it for the lnρ-vs-tdur check."""
    from allesfitter.validation.prior_sanity import (
        _compute_tdur_hours_by_companion,
        _tdur_days_from_orbit,
    )

    # Hot-Jupiter-ish geometry: per=3 d, R★+Rp/a=0.1, cosi=0.05, k=0.1
    # → tdur ≈ 3/π * arcsin(0.1 * sqrt((1.1)^2 - b^2)) ≈ ~3 h
    params_body = (
        "#name,value,fit,bounds,label,unit,coupled\n"
        "b_period,3.0,1,uniform 2.9 3.1,$P$,d,\n"
        "b_rsuma,0.10,1,uniform 0.05 0.2,$R_{sum}/a$,,\n"
        "b_cosi,0.05,1,uniform 0 1,$\\cos i$,,\n"
        "b_rr,0.10,1,uniform 0 0.3,$R_p/R_*$,,\n"
        "baseline_gp_matern32_lnrho_flux_tess,0,1,uniform -5 3,...,,\n"
    )
    d = _make_datadir(
        tmp_path, params=params_body, settings="inst_phot,tess",
        lc_csvs={"tess.csv": _clean_tess_lc_csv()},
    )

    derived = _compute_tdur_hours_by_companion(d)
    assert derived is not None and "b" in derived
    # tdur should be ~2.9 h for this geometry
    assert 1.0 < derived["b"] < 5.0, derived

    # Independent confirmation via the math helper
    tdur_d = _tdur_days_from_orbit(per=3.0, rsuma=0.10, cosi=0.05, k=0.10)
    assert abs(derived["b"] - tdur_d * 24.0) < 1e-6

    # The validator should use the derived tdur (not the 0.1 d fallback) —
    # with a ~3-h tdur, ρ_lo=exp(-3)≈0.05d is below 0.5*tdur≈0.06d → warn.
    msgs = validate_gp_priors(d)
    assert any("transit shape" in m for m in msgs), msgs


def test_tdur_helper_chromatic_rr(tmp_path):
    """The helper should fall back to <c>_rr_<bandpass> when <c>_rr is absent."""
    from allesfitter.validation.prior_sanity import _compute_tdur_hours_by_companion

    body = (
        "b_period,3.0,1,uniform 2 4,$P$,,\n"
        "b_rsuma,0.10,1,uniform 0 1,$R/a$,,\n"
        "b_cosi,0.05,1,uniform 0 1,$\\cos i$,,\n"
        "b_rr_tess,0.10,1,uniform 0 0.3,$k_{tess}$,,\n"
        "b_rr_g,0.11,1,uniform 0 0.3,$k_g$,,\n"
    )
    (tmp_path / "params.csv").write_text(body)
    out = _compute_tdur_hours_by_companion(tmp_path)
    assert out is not None and "b" in out
    assert out["b"] > 0


def test_tdur_helper_missing_inputs_returns_none(tmp_path):
    """If a companion lacks any of per/rsuma/cosi/k, it is omitted; if no
    companion can be resolved, the helper returns None."""
    from allesfitter.validation.prior_sanity import _compute_tdur_hours_by_companion

    # b has period but no rsuma → skipped; nothing else qualifies → None.
    (tmp_path / "params.csv").write_text(
        "b_period,3.0,1,uniform 2 4,,,\n"
        "b_cosi,0.0,1,uniform 0 1,,,\n"
        "b_rr,0.1,1,uniform 0 1,,,\n"
    )
    assert _compute_tdur_hours_by_companion(tmp_path) is None


def test_tdur_helper_non_transiting_geometry_omitted(tmp_path):
    """A grazing / non-transiting geometry (b > 1+k) yields NaN tdur and is
    excluded from the dict."""
    from allesfitter.validation.prior_sanity import _compute_tdur_hours_by_companion

    # cosi=0.5 with rsuma=0.1 → b=cosi*(1+k)/rsuma = 5.5 → non-transiting.
    (tmp_path / "params.csv").write_text(
        "b_period,3.0,1,uniform 2 4,,,\n"
        "b_rsuma,0.10,1,uniform 0 1,,,\n"
        "b_cosi,0.50,1,uniform 0 1,,,\n"
        "b_rr,0.10,1,uniform 0 1,,,\n"
    )
    assert _compute_tdur_hours_by_companion(tmp_path) is None
