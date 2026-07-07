"""Tests for the dataset-aware GP / noise prior helpers in
``scripts/prepare_allesfit.py``: ``_dataset_aware_gp_bounds`` and
``_update_params_gp_bounds``.

The helpers are private to the script; we load them via ``runpy.run_path``
with a guard that prevents ``main()`` from running.

We exercise three realistic datasets:

* a clean 27-day TESS sector at 120 s cadence (500 ppm RMS)
* a single-transit 6-h ground-based MuSCAT-style night (2 mmag RMS)
* a degenerate / too-short series (the helper should return ``None``)

and verify the refined bounds satisfy the four physical invariants:

1. ``exp(ln_err_hi) ≤ 0.10`` (noise prior never exceeds 10% rel flux)
2. ``exp(ln_σ_hi) ≤ 0.05`` (GP amplitude never reaches transit-depth scale)
3. ``exp(ln_ρ_lo) ≥ 0.5 × transit duration`` (GP cannot fit transit shape)
4. ``exp(ln_ρ_hi) ≤ observation baseline`` (GP not degenerate with slope)
"""

from __future__ import annotations

import math
import types
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Load the private helpers without executing main().
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def helpers():
    """Import _dataset_aware_gp_bounds + _update_params_gp_bounds.

    Reads the source and exec's only the functions we need so the import
    does not run main(). Avoids depending on argparse / lightkurve etc.
    """
    # Walk up from this test file to the repo root (the first ancestor that
    # contains scripts/prepare_allesfit.py). Robust to the tests/unit/ nesting.
    here = Path(__file__).resolve()
    src_path = None
    for ancestor in here.parents:
        candidate = ancestor / "scripts" / "prepare_allesfit.py"
        if candidate.exists():
            src_path = candidate
            break
    assert src_path is not None, "could not locate scripts/prepare_allesfit.py from " + str(here)
    src = src_path.read_text()
    # Parse, then exec only the function-def AST nodes we need.
    import ast

    tree = ast.parse(src)
    wanted = {"_dataset_aware_gp_bounds", "_update_params_gp_bounds"}
    ns = {"np": np, "math": math, "Path": Path}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in wanted:
            mod = ast.Module(body=[node], type_ignores=[])
            exec(compile(mod, str(src_path), "exec"), ns)
    return types.SimpleNamespace(
        bounds=ns["_dataset_aware_gp_bounds"],
        update=ns["_update_params_gp_bounds"],
    )


# ---------------------------------------------------------------------------
# Synthetic datasets.
# ---------------------------------------------------------------------------


def _tess_120s(rng=None, n=2000, rms=5e-4, baseline_days=27.0):
    rng = rng or np.random.default_rng(0)
    t = np.linspace(0.0, baseline_days, n)
    f = 1.0 + rng.normal(0.0, rms, n)
    return t, f


def _muscat_single_transit(rng=None, n=600, rms=2e-3, baseline_hours=6.0):
    rng = rng or np.random.default_rng(1)
    t = np.linspace(0.0, baseline_hours / 24.0, n)
    f = 1.0 + rng.normal(0.0, rms, n)
    return t, f


# ---------------------------------------------------------------------------
# bounds helper.
# ---------------------------------------------------------------------------


def test_bounds_tess_120s_clean(helpers):
    t, f = _tess_120s()
    b = helpers.bounds(t, f, tdur_days=0.1)  # ~2.4 h transit
    assert b is not None
    # ln_err_hi ≤ 10% rel flux
    assert math.exp(b["lnerr_hi"]) <= 0.10 + 1e-9
    # σ_GP upper ≤ 5%
    assert math.exp(b["lnsigma_hi"]) <= 0.05 + 1e-9
    # ρ lower ≥ 0.5 × tdur (transit-shape guard)
    assert math.exp(b["lnrho_lo"]) >= 0.5 * 0.1 - 1e-9
    # ρ upper ≤ baseline (no degeneracy with slope)
    assert math.exp(b["lnrho_hi"]) <= b["_baseline_days"] + 1e-9
    # ρ lower ≥ 2× cadence
    assert math.exp(b["lnrho_lo"]) >= 2.0 * b["_cadence_days"] - 1e-9


def test_bounds_muscat_single_transit(helpers):
    """Bounds must remain valid even for a 6-hour ground-based night."""
    t, f = _muscat_single_transit()
    tdur_d = 2.0 / 24.0
    b = helpers.bounds(t, f, tdur_days=tdur_d)
    assert b is not None
    # ρ upper must not exceed the 6-h baseline
    assert math.exp(b["lnrho_hi"]) <= b["_baseline_days"] + 1e-9
    # ρ lower respects transit duration
    assert math.exp(b["lnrho_lo"]) >= 0.5 * tdur_d - 1e-9
    # σ upper at most 5% (would otherwise swallow a 2% MuSCAT transit)
    assert math.exp(b["lnsigma_hi"]) <= 0.05 + 1e-9
    # init values strictly inside their bounds
    for k_init, k_lo, k_hi in [
        ("lnerr_init", "lnerr_lo", "lnerr_hi"),
        ("lnsigma_init", "lnsigma_lo", "lnsigma_hi"),
        ("lnrho_init", "lnrho_lo", "lnrho_hi"),
    ]:
        assert b[k_lo] <= b[k_init] <= b[k_hi]


def test_bounds_too_short_returns_none(helpers):
    t = np.linspace(0.0, 0.01, 5)
    f = np.array([1.0, 1.0001, 0.9999, 1.0002, 0.9998])
    assert helpers.bounds(t, f, tdur_days=0.05) is None


def test_bounds_anchor_on_rms(helpers):
    """Increasing the per-cadence RMS by 10× should widen the noise prior
    upward and lift the GP amplitude lower bound."""
    t, f1 = _tess_120s(rms=5e-4)
    _, f2 = _tess_120s(rms=5e-3)  # 10× noisier
    b1 = helpers.bounds(t, f1, tdur_days=0.1)
    b2 = helpers.bounds(t, f2, tdur_days=0.1)
    assert b2["lnerr_init"] > b1["lnerr_init"]
    assert b2["lnsigma_lo"] > b1["lnsigma_lo"]


# ---------------------------------------------------------------------------
# updater helper.
# ---------------------------------------------------------------------------


_DUMMY_PARAMS_CSV = """#name,value,fit,bounds,label,unit,coupled_with
b_rr_tess,0.1,1,uniform 0 0.3,$R_p/R_\\star$,,
ln_err_flux_tess,-6,1,uniform -10 -3,$\\log{\\sigma (tess)}$,rel. flux,
#baseline per instrument,,,,,,
#baseline_gp_offset_flux_tess,0,1,uniform -0.1 0.1,$\\mathrm{offset (tess)}$,,
baseline_gp_matern32_lnsigma_flux_tess,-5,1,uniform -15 0,$\\mathrm{gp ln \\sigma (tess)}$,,
baseline_gp_matern32_lnrho_flux_tess,0,1,uniform -5 5,$\\mathrm{gp ln \\rho (tess)}$,,
"""


def test_update_rewrites_three_rows(helpers, tmp_path):
    p = tmp_path / "params.csv"
    p.write_text(_DUMMY_PARAMS_CSV)
    t, f = _tess_120s()
    b = helpers.bounds(t, f, tdur_days=0.1)
    n = helpers.update(p, "tess", b)
    assert n == 3
    out = p.read_text()
    # the original "uniform -15 0" / "uniform -5 5" must be gone
    assert "uniform -15 0" not in out
    assert "uniform -5 5" not in out
    # rebuilt bounds match the helper's output
    assert "uniform {:.3f}".format(b["lnerr_lo"]) in out
    assert "uniform {:.3f}".format(b["lnsigma_lo"]) in out
    assert "uniform {:.3f}".format(b["lnrho_lo"]) in out


def test_update_preserves_tail_columns(helpers, tmp_path):
    p = tmp_path / "params.csv"
    p.write_text(_DUMMY_PARAMS_CSV)
    t, f = _tess_120s()
    b = helpers.bounds(t, f, tdur_days=0.1)
    helpers.update(p, "tess", b)
    out = p.read_text()
    # original label/unit fields preserved
    assert "$\\log{\\sigma (tess)}$" in out
    assert "rel. flux" in out
    assert "$\\mathrm{gp ln \\sigma (tess)}$" in out


def test_update_skips_when_inst_absent(helpers, tmp_path):
    p = tmp_path / "params.csv"
    p.write_text(_DUMMY_PARAMS_CSV)
    t, f = _tess_120s()
    b = helpers.bounds(t, f, tdur_days=0.1)
    # ask to update an instrument that's not in the file
    assert helpers.update(p, "muscat_g", b) == 0
    # file unchanged
    assert p.read_text() == _DUMMY_PARAMS_CSV


def test_update_returns_zero_on_missing_file(helpers, tmp_path):
    t, f = _tess_120s()
    b = helpers.bounds(t, f, tdur_days=0.1)
    assert helpers.update(tmp_path / "does_not_exist.csv", "tess", b) == 0


# ---------------------------------------------------------------------------
# End-to-end: refined bounds + sanity validator agree (no warnings).
# ---------------------------------------------------------------------------


def test_refined_bounds_pass_prior_checks(helpers, tmp_path):
    """After the updater rewrites the bounds with dataset-aware values, the
    independent prior_checks validator should emit zero warnings."""
    from allesfitter.validation import validate_gp_priors

    p_params = tmp_path / "params.csv"
    p_params.write_text(_DUMMY_PARAMS_CSV)
    (tmp_path / "settings.csv").write_text("inst_phot,tess\n")
    t, f = _tess_120s()
    np.savetxt(
        tmp_path / "tess.csv",
        np.column_stack([t, f, np.full_like(f, 5e-4)]),
        delimiter=",",
    )
    b = helpers.bounds(t, f, tdur_days=0.1)
    n = helpers.update(p_params, "tess", b)
    assert n == 3
    msgs = validate_gp_priors(tmp_path, tdur_hours_by_companion={"b": 2.4})
    assert msgs == [], msgs
