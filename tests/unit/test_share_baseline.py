"""Tests for the `baseline_share_<key>` settings.csv feature.

This feature lets the user declare that several instruments fit a single
joint celerite GP baseline (one realization shared across all members)
rather than N independent GPs. Tests cover:

  1. settings.csv parsing into groups / leader_of / followers_of dicts
  2. Followers inherit the leader's `baseline_<key>_<inst>` type
  3. Follower GP hyperparameters are aliased to the leader's row
  4. Conflicting follower rows raise ValueError
  5. The `_stack_and_sort` helper concatenates and tie-breaks
  6. Legacy behaviour is preserved when `baseline_share_<key>` is absent

The tests build a minimal 4-instrument datadir on tmp_path and call
`config.init(datadir)`. They do not run a fit, only assertions on the
fully-loaded Basement state and on a one-shot `update_params(theta_0)`
to confirm the per-iteration alias hook fires.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Skip the entire module if allesfitter (or required deps) cannot be imported
# in this environment; mirrors the guard used in test_integration_fit.py.
# ---------------------------------------------------------------------------
try:
    import allesfitter  # noqa: F401
    from allesfitter import config
    from allesfitter.basement import Basement, BASELINE_GP_HYPER_PREFIXES
except Exception:
    pytest.skip(
        "allesfitter or its dependencies are not importable",
        allow_module_level=True,
    )


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

_INSTS = ("muscat_g", "muscat_r", "muscat_i", "muscat_z")


def _lc_csv(seed: int, n: int = 200) -> str:
    """Tiny synthetic flat-flux CSV with `n` samples on a 1-day window."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 1.0, n)
    f = 1.0 + rng.normal(0.0, 1e-3, n)
    e = np.full(n, 1e-3)
    lines = ["#time,flux,flux_err"]
    lines += [f"{ti:.10f},{fi:.10f},{ei:.10f}" for ti, fi, ei in zip(t, f, e)]
    return "\n".join(lines) + "\n"


def _params_csv(insts, gp_insts=None) -> str:
    """Minimal params.csv with a transiting companion `b`, per-inst LDC and
    err. GP hyperparameter rows are emitted only for instruments in
    `gp_insts` (defaults to all of `insts` — the legacy / conflict-test
    layout; pass `[leader]` for the share-group happy path so followers
    have no params.csv row to conflict with).
    """
    if gp_insts is None:
        gp_insts = insts
    head = "#name,value,fit,bounds,label,unit,coupled_with\n"
    body = (
        "b_rr,0.10,1,uniform 0.05 0.30,$R_b/R_*$,,\n"
        "b_rsuma,0.15,1,uniform 0.05 0.30,$R/a$,,\n"
        "b_cosi,0.10,1,uniform 0 1,$\\cos i$,,\n"
        "b_epoch,0.5,1,uniform 0.0 1.0,$T_0$,d,\n"
        "b_period,3.5,1,normal 3.5 0.1,$P$,d,\n"
        "b_f_c,0,0,uniform -1 1,,,\n"
        "b_f_s,0,0,uniform -1 1,,,\n"
    )
    for i in insts:
        body += (
            f"dil_{i},0,0,uniform -1 1,,,\n"
            f"host_ldc_q1_{i},0.5,1,uniform 0 1,,,\n"
            f"host_ldc_q2_{i},0.5,1,uniform 0 1,,,\n"
            f"ln_err_flux_{i},-6,1,uniform -10 -1,,,\n"
        )
        if i in gp_insts:
            body += (
                f"baseline_gp_matern32_lnsigma_flux_{i},-5,1,uniform -10 -3,,,\n"
                f"baseline_gp_matern32_lnrho_flux_{i},0,1,uniform -1 3,,,\n"
            )
    return head + body


def _settings_csv(insts, share_value: str | None = None) -> str:
    inst_str = " ".join(insts)
    settings = [
        "#name,value",
        "companions_phot,b",
        "companions_rv,",
        f"inst_phot,{inst_str}",
        "inst_rv,",
        "time_format,BJD_TDB",
        "multiprocess,False",
        "print_progress,False",
        "fast_fit,False",
        "shift_epoch,False",
    ]
    for i in insts:
        settings += [
            f"host_ld_law_{i},quad",
            f"error_flux_{i},sample",
        ]
    # leader gets the GP baseline; followers default to 'none' and inherit
    # via the share-group machinery if share_value is provided.
    settings.append(f"baseline_flux_{insts[0]},sample_GP_Matern32")
    if share_value is not None:
        settings.append(f"baseline_share_flux,{share_value}")
    return "\n".join(settings) + "\n"


def _make_datadir(
    tmp_path: Path,
    insts=_INSTS,
    share_value=None,
    gp_insts=None,
) -> Path:
    d = tmp_path
    (d / "results").mkdir(exist_ok=True)
    (d / "params.csv").write_text(_params_csv(insts, gp_insts=gp_insts))
    (d / "settings.csv").write_text(_settings_csv(insts, share_value))
    for k, inst in enumerate(insts):
        (d / f"{inst}.csv").write_text(_lc_csv(seed=k))
    return d


def _make_basement(tmp_path: Path, **kwargs) -> Basement:
    datadir = _make_datadir(tmp_path, **kwargs)
    return Basement(str(datadir), quiet=True)


# ---------------------------------------------------------------------------
# 1) settings parsing
# ---------------------------------------------------------------------------


def test_parse_share_groups_basic(tmp_path):
    b = _make_basement(
        tmp_path,
        share_value=":".join(_INSTS),
        gp_insts=[_INSTS[0]],
    )
    s = b.settings
    assert s["baseline_share_flux_groups"] == [list(_INSTS)]
    assert s["baseline_share_flux_leader_of"] == {
        "muscat_g": "muscat_g",
        "muscat_r": "muscat_g",
        "muscat_i": "muscat_g",
        "muscat_z": "muscat_g",
    }
    assert s["baseline_share_flux_followers_of"] == {
        "muscat_g": ["muscat_r", "muscat_i", "muscat_z"],
    }


def test_parse_share_groups_multiple(tmp_path):
    insts = ("m1_g", "m1_r", "m2_g", "m2_r")
    share = "m1_g:m1_r m2_g:m2_r"
    # The settings template only sets the GP for insts[0]; we need both
    # leaders to be sample_GP_Matern32, so extend the template inline.
    d = tmp_path
    (d / "results").mkdir(exist_ok=True)
    (d / "params.csv").write_text(_params_csv(insts, gp_insts=["m1_g", "m2_g"]))
    s = _settings_csv(insts, share)
    # second leader also gets sample_GP_Matern32
    s = s.replace(
        "baseline_flux_m1_g,sample_GP_Matern32",
        "baseline_flux_m1_g,sample_GP_Matern32\nbaseline_flux_m2_g,sample_GP_Matern32",
    )
    (d / "settings.csv").write_text(s)
    for k, inst in enumerate(insts):
        (d / f"{inst}.csv").write_text(_lc_csv(seed=k))
    b = Basement(str(d), quiet=True)
    assert b.settings["baseline_share_flux_groups"] == [
        ["m1_g", "m1_r"], ["m2_g", "m2_r"]
    ]
    assert b.settings["baseline_share_flux_followers_of"] == {
        "m1_g": ["m1_r"], "m2_g": ["m2_r"]
    }


# ---------------------------------------------------------------------------
# 2) followers inherit the leader's baseline type
# ---------------------------------------------------------------------------


def test_followers_inherit_baseline_type(tmp_path):
    b = _make_basement(
        tmp_path,
        share_value=":".join(_INSTS),
        gp_insts=[_INSTS[0]],
    )
    for f in _INSTS[1:]:
        assert b.settings[f"baseline_flux_{f}"] == "sample_GP_Matern32"
        assert b.settings[f"baseline_flux_{f}_against"] == "time"


# ---------------------------------------------------------------------------
# 3) follower GP hyperparameters are aliased to the leader at load time
#    AND re-aliased on every update_params call
# ---------------------------------------------------------------------------


def test_follower_params_aliased_to_leader_at_load(tmp_path):
    b = _make_basement(
        tmp_path,
        share_value=":".join(_INSTS),
        gp_insts=[_INSTS[0]],
    )
    leader_sig = b.params["baseline_gp_matern32_lnsigma_flux_muscat_g"]
    leader_rho = b.params["baseline_gp_matern32_lnrho_flux_muscat_g"]
    for f in _INSTS[1:]:
        assert b.params[f"baseline_gp_matern32_lnsigma_flux_{f}"] == leader_sig
        assert b.params[f"baseline_gp_matern32_lnrho_flux_{f}"] == leader_rho


def test_explicit_follower_rows_marked_fit_zero(tmp_path):
    """If the user explicitly declared follower GP rows with coupled_with,
    they remain in allkeys but never enter fitkeys. (This variant uses
    fit=1 + coupled_with, which the share-group code force-couples too.)"""
    # Write a params.csv where follower rows are explicitly coupled to the
    # leader (legal user input that should NOT raise).
    d = tmp_path
    (d / "results").mkdir(exist_ok=True)
    body = (
        _params_csv(_INSTS, gp_insts=[_INSTS[0]])
        + f"baseline_gp_matern32_lnsigma_flux_muscat_r,-5,0,uniform -10 -3,,,baseline_gp_matern32_lnsigma_flux_muscat_g\n"
        + f"baseline_gp_matern32_lnrho_flux_muscat_r,0,0,uniform -1 3,,,baseline_gp_matern32_lnrho_flux_muscat_g\n"
    )
    (d / "params.csv").write_text(body)
    (d / "settings.csv").write_text(_settings_csv(_INSTS, share_value=":".join(_INSTS)))
    for k, inst in enumerate(_INSTS):
        (d / f"{inst}.csv").write_text(_lc_csv(seed=k))
    b = Basement(str(d), quiet=True)
    follower_keys = [
        "baseline_gp_matern32_lnsigma_flux_muscat_r",
        "baseline_gp_matern32_lnrho_flux_muscat_r",
    ]
    for fk in follower_keys:
        assert fk not in list(b.fitkeys), fk


def test_update_params_aliases_follower_to_leader(tmp_path):
    """update_params() must propagate the leader's *current* theta value
    into every follower entry — not the stale value from load time."""
    from allesfitter.computer import update_params
    b = _make_basement(
        tmp_path,
        share_value=":".join(_INSTS),
        gp_insts=[_INSTS[0]],
    )
    config.BASEMENT = b
    # tweak the leader's lnsigma to something distinctive
    fitkeys = list(b.fitkeys)
    target_key = "baseline_gp_matern32_lnsigma_flux_muscat_g"
    assert target_key in fitkeys, fitkeys
    theta = np.array(b.theta_0, dtype=float)
    theta[fitkeys.index(target_key)] = -7.5
    params = update_params(theta)
    assert params[target_key] == -7.5
    for f in _INSTS[1:]:
        assert params[f"baseline_gp_matern32_lnsigma_flux_{f}"] == -7.5


# ---------------------------------------------------------------------------
# 4) conflicting follower row raises ValueError
# ---------------------------------------------------------------------------


def test_conflicting_follower_row_raises(tmp_path):
    """If a follower has its own fit=1 GP-hyper row in params.csv (no
    coupled_with), Basement must refuse to load — that row would conflict
    with the share-group's leader-only sampling."""
    d = tmp_path
    (d / "results").mkdir(exist_ok=True)
    # follower 'muscat_r' is explicitly fit=1 for lnsigma — should raise
    params = _params_csv(_INSTS)
    # the helper already writes fit=1 for every inst; that is exactly the
    # conflict we want to detect.
    (d / "params.csv").write_text(params)
    (d / "settings.csv").write_text(_settings_csv(_INSTS, share_value=":".join(_INSTS)))
    for k, inst in enumerate(_INSTS):
        (d / f"{inst}.csv").write_text(_lc_csv(seed=k))
    with pytest.raises(ValueError, match="baseline_share_flux"):
        # Force the conflict by editing the params.csv to mark the follower
        # row as fit=1 explicitly (already so via _params_csv helper); the
        # validator should refuse without an explicit coupled_with.
        # First mutate: ensure the follower row stays fit=1 and uncoupled.
        Basement(str(d), quiet=True)


# ---------------------------------------------------------------------------
# 5) _stack_and_sort helper
# ---------------------------------------------------------------------------


def test_stack_and_sort_singleton_is_identity():
    from allesfitter.computer import _stack_and_sort
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([10.0, 20.0, 30.0])
    e = np.array([0.1, 0.2, 0.3])
    out_x, out_y, out_e = _stack_and_sort([(x, y, e)])
    # Identity: returns the original arrays unchanged
    assert out_x is x and out_y is y and out_e is e


def test_stack_and_sort_breaks_ties():
    from allesfitter.computer import _stack_and_sort
    # Four bands with identical time grids (MuSCAT-style simultaneous
    # photometry) — every t in [0,1] appears 4 times.
    t = np.linspace(0.0, 1.0, 10)
    parts = [
        (t.copy(), np.full_like(t, k + 1.0), np.full_like(t, 0.1))
        for k in range(4)
    ]
    x, y, e = _stack_and_sort(parts)
    # Strictly sorted, no duplicates
    assert np.all(np.diff(x) > 0), "stack_and_sort did not produce strictly sorted x"
    # Length preserved
    assert len(x) == 4 * len(t)
    # Values are within one ULP of the original timestamp cluster
    # (we nudged duplicates by np.nextafter, which is <1e-12 for t ~1)
    assert np.allclose(x, np.sort(np.concatenate([p[0] for p in parts])),
                       atol=1e-9), x


# ---------------------------------------------------------------------------
# 6) backward compatibility: no share key → no share state, legacy lnlike
# ---------------------------------------------------------------------------


def test_singleton_group_warns(tmp_path):
    """A share group with only one member shares nothing and almost
    certainly indicates a typo. The loader should warn so the user
    notices, while still loading successfully (legacy fallback)."""
    with pytest.warns(UserWarning, match="only one member"):
        b = _make_basement(
            tmp_path,
            share_value="muscat_g",
            gp_insts=list(_INSTS),
        )
    assert b.settings["baseline_share_flux_groups"] == [["muscat_g"]]


def test_duplicate_member_in_group_raises(tmp_path):
    with pytest.raises(ValueError, match="duplicate members"):
        _make_basement(
            tmp_path,
            share_value="muscat_g:muscat_g:muscat_r",
            gp_insts=[_INSTS[0]],
        )


def test_duplicate_leader_across_groups_raises(tmp_path):
    insts = ("m1_g", "m1_r", "m1_i")
    d = tmp_path
    (d / "results").mkdir(exist_ok=True)
    (d / "params.csv").write_text(_params_csv(insts, gp_insts=["m1_g"]))
    s = _settings_csv(insts, share_value="m1_g:m1_r m1_g:m1_i")
    (d / "settings.csv").write_text(s)
    for k, inst in enumerate(insts):
        (d / f"{inst}.csv").write_text(_lc_csv(seed=k))
    with pytest.raises(ValueError, match="more than one group"):
        Basement(str(d), quiet=True)


def test_leader_missing_gp_hypers_raises(tmp_path):
    """settings.csv declares the leader uses sample_GP_Matern32, but the
    user forgot to add baseline_gp_matern32_lnsigma_flux_<leader> to
    params.csv. We must catch this at load time, not at the first lnlike
    call."""
    # Pass gp_insts=[] so NO inst gets GP hyper rows.
    with pytest.raises(ValueError, match="missing the required row"):
        _make_basement(
            tmp_path,
            share_value=":".join(_INSTS),
            gp_insts=[],
        )


def test_follower_coupled_to_wrong_target_raises(tmp_path):
    """If a follower row uses coupled_with=<not-the-leader>, the share
    machinery cannot reconcile it — refuse to load."""
    d = tmp_path
    (d / "results").mkdir(exist_ok=True)
    body = (
        _params_csv(_INSTS, gp_insts=[_INSTS[0]])
        # coupled to ln_err_flux_muscat_g (a wrong target)
        + "baseline_gp_matern32_lnsigma_flux_muscat_r,-5,0,uniform -10 -3,,,ln_err_flux_muscat_g\n"
    )
    (d / "params.csv").write_text(body)
    (d / "settings.csv").write_text(_settings_csv(_INSTS, share_value=":".join(_INSTS)))
    for k, inst in enumerate(_INSTS):
        (d / f"{inst}.csv").write_text(_lc_csv(seed=k))
    with pytest.raises(ValueError, match="coupled_with"):
        Basement(str(d), quiet=True)


def test_follower_against_mismatch_raises(tmp_path):
    """If the user explicitly set baseline_<key>_<follower>_against to
    something other than 'time', the share group cannot honour it (the
    joint GP uses time) — error rather than silently override."""
    insts = _INSTS
    d = tmp_path
    (d / "results").mkdir(exist_ok=True)
    (d / "params.csv").write_text(_params_csv(insts, gp_insts=[insts[0]]))
    s = _settings_csv(insts, share_value=":".join(insts))
    # Splice in a non-time _against for a follower; the leader's stays at
    # the implicit default ('time').
    s = s.rstrip() + "\nbaseline_flux_muscat_r_against,custom_series\n"
    (d / "settings.csv").write_text(s)
    for k, inst in enumerate(insts):
        (d / f"{inst}.csv").write_text(_lc_csv(seed=k))
    with pytest.raises(ValueError, match="_against"):
        Basement(str(d), quiet=True)


def test_no_share_key_preserves_legacy(tmp_path):
    """When `baseline_share_flux` is absent, the share-state dicts are
    empty and the lnlike path collapses to today's behaviour."""
    b = _make_basement(tmp_path, share_value=None)
    assert b.settings["baseline_share_flux_groups"] == []
    assert b.settings["baseline_share_flux_leader_of"] == {}
    assert b.settings["baseline_share_flux_followers_of"] == {}


# ---------------------------------------------------------------------------
# 7) chromatic_rr posterior plot — emitted for both NS and MCMC pipelines
# ---------------------------------------------------------------------------


def _chromatic_params_csv(bandpasses) -> str:
    """params.csv for chromatic mode: per-bandpass b_rr + per-bandpass LDC."""
    head = "#name,value,fit,bounds,label,unit,coupled_with\n"
    body = (
        "b_rsuma,0.15,1,uniform 0.05 0.30,$R/a$,,\n"
        "b_cosi,0.10,1,uniform 0 1,$\\cos i$,,\n"
        "b_epoch,0.5,1,uniform 0.0 1.0,$T_0$,d,\n"
        "b_period,3.5,1,normal 3.5 0.1,$P$,d,\n"
        "b_f_c,0,0,uniform -1 1,,,\n"
        "b_f_s,0,0,uniform -1 1,,,\n"
    )
    for bp in bandpasses:
        body += f"b_rr_{bp},0.10,1,uniform 0.05 0.30,$R_b/R_*$,,\n"
        body += f"host_ldc_q1_{bp},0.5,1,uniform 0 1,,,\n"
        body += f"host_ldc_q2_{bp},0.5,1,uniform 0 1,,,\n"
    return head + body


def _chromatic_settings_csv(insts, bandpasses) -> str:
    inst_str = " ".join(insts)
    bp_str = " ".join(bandpasses)
    settings = [
        "#name,value",
        "companions_phot,b",
        "companions_rv,",
        f"inst_phot,{inst_str}",
        f"bandpass,{bp_str}",
        "inst_rv,",
        "time_format,BJD_TDB",
        "multiprocess,False",
        "print_progress,False",
        "fast_fit,False",
        "shift_epoch,False",
    ]
    for i in insts:
        settings += [
            f"host_ld_law_{i},quad",
            f"error_flux_{i},sample",
        ]
    # Note: dil_<inst> and ln_err_flux_<inst> are *parameters* (params.csv),
    # not settings — dil_<inst> defaults to 0 and ln_err rows are added to
    # params.csv in _make_chromatic_datadir.
    return "\n".join(settings) + "\n"


def _make_chromatic_datadir(tmp_path: Path) -> Path:
    """Build a 4-band chromatic datadir under tmp_path/'data'."""
    d = tmp_path / "data"
    d.mkdir()
    (d / "results").mkdir()
    insts = ("muscat_g", "muscat_r", "muscat_i", "muscat_z")
    bandpasses = ("g", "r", "i", "z")
    # ln_err is the only per-inst param needed (chromatic LDC/rr live under
    # the bandpass suffix). Append ln_err rows to the bandpass-keyed params.
    params = _chromatic_params_csv(bandpasses)
    for i in insts:
        params += f"ln_err_flux_{i},-6,1,uniform -10 -1,,,\n"
    (d / "params.csv").write_text(params)
    (d / "settings.csv").write_text(_chromatic_settings_csv(insts, bandpasses))
    for k, inst in enumerate(insts):
        (d / f"{inst}.csv").write_text(_lc_csv(seed=k))
    return d


def test_mcmc_chromatic_rr_pdf_emitted(tmp_path):
    """plot_chromatic_rr_histogram with prefix='mcmc' must write
    `<outdir>/mcmc_chromatic_rr_<companion>.pdf`."""
    from allesfitter.nested_sampling_output import plot_chromatic_rr_histogram
    datadir = _make_chromatic_datadir(tmp_path)
    config.init(str(datadir))
    b = config.BASEMENT
    assert b.settings["chromatic"] is True, "fixture should be chromatic"

    # Build a synthetic (Nsamples, ndim) posterior around theta_0.
    rng = np.random.default_rng(0)
    samples = np.array(b.theta_0)[None, :] + rng.normal(
        0, 1e-3, size=(200, b.ndim)
    )

    plot_chromatic_rr_histogram(samples, prefix="mcmc")
    out = Path(b.outdir) / "mcmc_chromatic_rr_b.pdf"
    assert out.exists(), f"missing {out}"
    assert out.stat().st_size > 0, f"empty {out}"


def test_ns_chromatic_rr_default_prefix_unchanged(tmp_path):
    """Calling without an explicit prefix must keep emitting the legacy
    `ns_chromatic_rr_<companion>.pdf` filename (NS backward compat)."""
    from allesfitter.nested_sampling_output import plot_chromatic_rr_histogram
    datadir = _make_chromatic_datadir(tmp_path)
    config.init(str(datadir))
    b = config.BASEMENT

    rng = np.random.default_rng(1)
    samples = np.array(b.theta_0)[None, :] + rng.normal(
        0, 1e-3, size=(200, b.ndim)
    )

    plot_chromatic_rr_histogram(samples)  # default prefix='ns'
    out = Path(b.outdir) / "ns_chromatic_rr_b.pdf"
    assert out.exists(), f"missing {out}"
    assert out.stat().st_size > 0


# ---------------------------------------------------------------------------
# 8) corner-plot nuisance filter (drops baseline / err / stellar-var GP rows
#    when ndim exceeds the threshold)
# ---------------------------------------------------------------------------


def test_filter_nuisance_keeps_when_ndim_small():
    from allesfitter.nested_sampling_output import _filter_nuisance_for_corner
    fitkeys = np.array(['b_rr', 'b_cosi', 'baseline_offset_flux_a',
                        'ln_err_flux_a'])
    samples = np.random.default_rng(0).normal(0, 1, size=(50, 4))
    labels = list(fitkeys)
    truths = np.zeros(4)
    # threshold=10 > ndim=4 → no filtering
    out_keys, out_s, out_lab, out_t, dropped = _filter_nuisance_for_corner(
        fitkeys, samples, labels, truths, threshold=10,
    )
    assert dropped == []
    assert out_s.shape == samples.shape
    assert list(out_keys) == list(fitkeys)


def test_filter_nuisance_drops_baseline_and_error():
    from allesfitter.nested_sampling_output import _filter_nuisance_for_corner
    fitkeys = np.array([
        'b_rr_g', 'b_rr_r', 'b_rr_i', 'b_rr_z',
        'b_rsuma', 'b_cosi', 'b_epoch', 'b_period',
        'host_ldc_q1_g', 'host_ldc_q1_r',
        'baseline_gp_matern32_lnsigma_flux_a',
        'baseline_gp_matern32_lnrho_flux_a',
        'baseline_offset_flux_a',
        'ln_err_flux_a',
        'ln_jitter_rv_a',
        'stellar_var_gp_matern32_lnsigma_flux',
    ])
    samples = np.random.default_rng(1).normal(0, 1, size=(50, len(fitkeys)))
    labels = list(fitkeys)
    truths = np.arange(len(fitkeys), dtype=float)
    out_keys, out_s, out_lab, out_t, dropped = _filter_nuisance_for_corner(
        fitkeys, samples, labels, truths, threshold=5,
    )
    # 6 nuisance rows expected to be dropped
    assert len(dropped) == 6
    assert all(k.startswith(('baseline_', 'ln_err_', 'ln_jitter_',
                              'stellar_var_gp_')) for k in dropped), dropped
    # 10 science rows kept (b_rr_*, b_rsuma, b_cosi, b_epoch, b_period,
    # host_ldc_q1_g, host_ldc_q1_r)
    assert out_s.shape == (50, 10)
    assert len(out_lab) == 10
    assert out_t.shape == (10,)


def test_corner_nuisance_mask_matches_documented_prefixes():
    from allesfitter.nested_sampling_output import _corner_nuisance_mask
    keys = np.array(['b_rr', 'baseline_offset_flux_a',
                     'ln_err_flux_a', 'ln_jitter_rv_a',
                     'stellar_var_gp_sho_lnS0_flux', 'dil_a', 'host_ldc_q1_a'])
    mask = _corner_nuisance_mask(keys)
    # b_rr, dil_a, host_ldc_q1_a should be kept (dil is physical-ish)
    assert mask.tolist() == [True, False, False, False, False, True, True]


def test_single_member_group_matches_legacy_lnlike(tmp_path):
    """A share group with one member must give identical lnlike to the
    no-share case (within numerical noise). This is the regression test
    that proves the joint-GP refactor is a pure-identity transformation
    when no actual sharing is configured."""
    from allesfitter.computer import calculate_lnlike_total

    # Build two datadirs: one with no share, one with a singleton group.
    (tmp_path / "legacy").mkdir()
    (tmp_path / "share").mkdir()
    b_legacy = _make_basement(tmp_path / "legacy", share_value=None)
    b_share = _make_basement(tmp_path / "share", share_value="muscat_g")

    # Use the singleton-group leader's theta_0 in both cases.
    config.BASEMENT = b_legacy
    from allesfitter.computer import update_params as up
    params_legacy = up(np.array(b_legacy.theta_0, dtype=float))
    ll_legacy = calculate_lnlike_total(params_legacy)

    config.BASEMENT = b_share
    params_share = up(np.array(b_share.theta_0, dtype=float))
    ll_share = calculate_lnlike_total(params_share)

    assert np.isfinite(ll_legacy)
    assert np.isfinite(ll_share)
    assert abs(ll_share - ll_legacy) < 1e-8, (ll_share, ll_legacy)
