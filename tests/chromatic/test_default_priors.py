"""Pin the physics-informed default prior bounds emitted by
prepare_allesfit.py.

The helper ``_default_prior_bounds(rprs_max, rsuma_min, rsuma_max)``
encapsulates the four computed bounds (rr_upper, rsuma_lo, rsuma_hi,
cosi_max). Each test asserts the bound is correct for a representative
target class and that the boundary clamps fire when expected.

Static priors (ln_err_flux, lnrho, ttv) are pinned with a string-search
regression so a future re-edit of those literals doesn't silently revert.
"""

from __future__ import annotations

import importlib.util
import pathlib

import pytest


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "prepare_allesfit.py"


@pytest.fixture(scope="module")
def prep():
    """Import prepare_allesfit.py as a module (it's a script, not a
    package), exposing _default_prior_bounds for direct unit tests."""
    spec = importlib.util.spec_from_file_location("prepare_allesfit", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestDefaultPriorBounds:
    def test_typical_tess_hot_jupiter(self, prep):
        # Typical hot Jupiter: depth ~1.2%, rprs ~0.11, rsuma ~0.15
        b = prep._default_prior_bounds(
            rprs_max=0.12, rsuma_min=0.10, rsuma_max=0.18
        )
        # ceil(0.12*10)/10 = 0.2, + 0.05 = 0.25, < 0.5 cap
        assert b["rr_upper"] == pytest.approx(0.25)
        assert b["rsuma_lo"] == pytest.approx(0.10)
        # 2 * 0.18 = 0.36, < 0.5 cap
        assert b["rsuma_hi"] == pytest.approx(0.36)
        # 1.2 * 0.18 * 1.12 = 0.24192, < 1 cap
        assert b["cosi_max"] == pytest.approx(0.24192, rel=1e-5)

    def test_small_planet_long_period(self, prep):
        # Earth-size on a 50-day orbit (Kepler-like): depth tiny, rsuma small.
        b = prep._default_prior_bounds(
            rprs_max=0.012, rsuma_min=0.005, rsuma_max=0.015
        )
        # ceil(0.012*10)/10 = 0.1, + 0.05 = 0.15
        assert b["rr_upper"] == pytest.approx(0.15)
        # rsuma_min stays > 1e-3 floor
        assert b["rsuma_lo"] == pytest.approx(0.005)
        # 2 * 0.015 = 0.030
        assert b["rsuma_hi"] == pytest.approx(0.030)
        # 1.2 * 0.015 * 1.012 = 0.0182
        assert b["cosi_max"] == pytest.approx(0.0182, rel=1e-3)

    def test_substellar_companion_hits_caps(self, prep):
        # Brown dwarf / M-dwarf eclipsing binary: rr huge, rsuma huge.
        b = prep._default_prior_bounds(
            rprs_max=0.55, rsuma_min=0.05, rsuma_max=0.40
        )
        # rprs upper ceil(0.55*10)/10 + 0.05 = 0.6 + 0.05 = 0.65 → clipped to 0.5
        assert b["rr_upper"] == 0.5
        # 2 * 0.40 = 0.8 → clipped to 0.5
        assert b["rsuma_hi"] == 0.5
        # 1.2 * 0.40 * 1.55 = 0.744, < 1
        assert b["cosi_max"] == pytest.approx(0.744)

    def test_rsuma_lo_floor_prevents_divergence(self, prep):
        # Pathologically small rsuma_min — must clamp to 1e-3 to avoid
        # a→∞ at the prior boundary.
        b = prep._default_prior_bounds(
            rprs_max=0.08, rsuma_min=1e-6, rsuma_max=0.10
        )
        assert b["rsuma_lo"] == pytest.approx(1e-3)

    def test_cosi_clipped_when_rsuma_rr_exceed_unity(self, prep):
        # Very large rsuma + deep companion → 1.2·rsuma·(1+rr) > 1.
        b = prep._default_prior_bounds(
            rprs_max=0.40, rsuma_min=0.10, rsuma_max=0.80
        )
        # raw cosi_max would be 1.2 * 0.80 * 1.40 = 1.344 → clamped to 1.
        assert b["cosi_max"] == 1.0

    def test_returns_floats(self, prep):
        b = prep._default_prior_bounds(0.1, 0.05, 0.2)
        for k, v in b.items():
            assert isinstance(v, float), f"{k!r} is {type(v).__name__}, not float"


class TestStaticPriorChanges:
    """String-level regression: the tightened ln_err / lnrho / ttv priors
    must not silently revert in a future cleanup pass."""

    @pytest.fixture(scope="class")
    def source(self):
        return SCRIPT.read_text()

    def test_ln_err_flux_upper_tightened_to_minus_3(self, source):
        # Old: uniform -10 -1 (37% scatter upper). New: uniform -10 -3.
        assert 'uniform -10 -3' in source
        # The old loose bound must be gone for ln_err rows.
        for line in source.splitlines():
            if 'ln_err_flux_' in line and 'uniform' in line:
                assert 'uniform -10 -1' not in line, (
                    f"ln_err_flux still has old wide prior: {line}"
                )

    def test_lnrho_upper_tightened_to_5(self, source):
        # Old: uniform -5 10 (60-year correlation). New: uniform -5 5.
        for line in source.splitlines():
            if 'baseline_gp_matern32_lnrho_flux_' in line and 'uniform' in line:
                assert 'uniform -5 5' in line, (
                    f"lnrho row missing tightened bound: {line}"
                )
                assert 'uniform -5 10' not in line

    def test_ttv_prior_tightened_to_pm_005(self, source):
        # Old: uniform -0.1 0.1. New: uniform -0.05 0.05.
        ttv_lines = [
            line for line in source.splitlines()
            if 'ttv_transit_' in line and 'uniform' in line
        ]
        assert ttv_lines, "no TTV emission lines found"
        for line in ttv_lines:
            assert 'uniform -0.05 0.05' in line, (
                f"TTV row missing tightened bound: {line}"
            )
            assert 'uniform -0.1 0.1' not in line, (
                f"TTV row still has wide prior: {line}"
            )
