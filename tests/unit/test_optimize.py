"""Tests for ``allesfitter.optimize``.

Coverage:
  1. ``_extract_bounds`` returns finite ``[lo, hi]`` for uniform / normal /
     trunc_normal priors.
  2. ``_on_bounds`` flags points within margin of the prior edge.
  3. L-BFGS-B from theta_0 finds a strict improvement on a chromatic 4-band
     datadir with stub data, and BASEMENT.theta_0 is updated when accepted.
  4. Acceptance gate: when the optimizer regresses (synthetic objective),
     OptimizeResult.accepted is False and BASEMENT.theta_0 is untouched.
  5. ``mutate_basement=False`` leaves BASEMENT.theta_0 alone even on a
     successful accept.
  6. JSON persistence: ``results/optimize_save.json`` exists with the
     expected schema.
  7. Multistart consistency: when restarts disagree wildly, the run is
     rejected with the consistency-spread reason.
  8. CMA-ES path (skipped if `cma` is not installed) — strict improvement
     on the same fixture.
  9. Accepted optima with ``shift_epoch=True`` persist in the original
     params.csv epoch frame and reload with the correctly shifted prior.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pytest

# Module-level guard — mirrors the share-baseline test file.
try:
    import allesfitter  # noqa: F401
    from allesfitter import config, optimize
    from allesfitter.optimize import (
        OptimizeResult,
        _extract_bounds,
        _on_bounds,
        logprint,
    )
except Exception:
    pytest.skip(
        "allesfitter or its dependencies are not importable",
        allow_module_level=True,
    )


# ---------------------------------------------------------------------------
# fixture: minimal chromatic 4-band datadir (same recipe as
# tests/test_share_baseline.py::_make_chromatic_datadir but self-contained
# so this file can be run in isolation)
# ---------------------------------------------------------------------------

_INSTS = ("muscat_g", "muscat_r", "muscat_i", "muscat_z")
_BANDS = ("g", "r", "i", "z")


def _lc_csv(seed: int, n: int = 200, time_offset: float = 0.0) -> str:
    rng = np.random.default_rng(seed)
    t = time_offset + np.linspace(0.0, 1.0, n)
    f = 1.0 + rng.normal(0.0, 1e-3, n)
    e = np.full(n, 1e-3)
    lines = ["#time,flux,flux_err"]
    lines += [f"{ti:.10f},{fi:.10f},{ei:.10f}" for ti, fi, ei in zip(t, f, e)]
    return "\n".join(lines) + "\n"


def _params_csv() -> str:
    head = "#name,value,fit,bounds,label,unit,coupled_with\n"
    body = (
        "b_rsuma,0.15,1,uniform 0.05 0.30,$R/a$,,\n"
        "b_cosi,0.10,1,uniform 0 1,$\\cos i$,,\n"
        "b_epoch,0.5,1,uniform 0.0 1.0,$T_0$,d,\n"
        "b_period,3.5,1,normal 3.5 0.1,$P$,d,\n"
        "b_f_c,0,0,uniform -1 1,,,\n"
        "b_f_s,0,0,uniform -1 1,,,\n"
    )
    for bp in _BANDS:
        body += f"b_rr_{bp},0.10,1,uniform 0.05 0.30,$R_b/R_*$,,\n"
        body += f"host_ldc_q1_{bp},0.5,1,uniform 0 1,,,\n"
        body += f"host_ldc_q2_{bp},0.5,1,uniform 0 1,,,\n"
    for i in _INSTS:
        body += f"ln_err_flux_{i},-6,1,uniform -10 -1,,,\n"
    return head + body


def _settings_csv() -> str:
    inst_str = " ".join(_INSTS)
    bp_str = " ".join(_BANDS)
    lines = [
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
    for i in _INSTS:
        lines += [
            f"host_ld_law_{i},quad",
            f"error_flux_{i},sample",
        ]
    return "\n".join(lines) + "\n"


def _make_datadir(tmp_path: Path) -> Path:
    d = tmp_path / "data"
    d.mkdir()
    (d / "results").mkdir()
    (d / "params.csv").write_text(_params_csv())
    (d / "settings.csv").write_text(_settings_csv())
    for k, inst in enumerate(_INSTS):
        (d / f"{inst}.csv").write_text(_lc_csv(seed=k))
    return d


def _make_shift_epoch_datadir(tmp_path: Path) -> Path:
    d = _make_datadir(tmp_path)
    settings = (d / "settings.csv").read_text().replace("shift_epoch,False", "shift_epoch,True")
    (d / "settings.csv").write_text(settings)
    params = (
        (d / "params.csv")
        .read_text()
        .replace("b_epoch,0.5,1,uniform 0.0 1.0", "b_epoch,0.5,1,normal 0.5 0.1")
    )
    (d / "params.csv").write_text(params)
    for k, inst in enumerate(_INSTS):
        (d / f"{inst}.csv").write_text(_lc_csv(seed=k, time_offset=100.0))
    return d


@pytest.fixture
def datadir(tmp_path):
    d = _make_datadir(tmp_path)
    config.init(str(d))
    return d


# ---------------------------------------------------------------------------
# 1) bounds extraction
# ---------------------------------------------------------------------------


def test_extract_bounds_returns_finite_pair_per_fitkey(datadir):
    bounds = _extract_bounds(config.BASEMENT)
    assert len(bounds) == config.BASEMENT.ndim
    for lo, hi in bounds:
        assert np.isfinite(lo) and np.isfinite(hi)
        assert lo < hi


def test_extract_bounds_normal_prior_uses_5sigma_window():
    # b_period was declared as normal 3.5 0.1 in _params_csv → ±5σ box
    # should be [3.0, 4.0].
    class _FakeBasement:
        bounds = [["normal", 3.5, 0.1]]
        ndim = 1

    b = _FakeBasement()
    out = _extract_bounds(b)
    assert out == [(3.0, 4.0)]


# ---------------------------------------------------------------------------
# 2) _on_bounds helper
# ---------------------------------------------------------------------------


def test_on_bounds_detects_edge():
    bounds = [(0.0, 1.0), (-5.0, 5.0)]
    assert _on_bounds([0.0, 0.0], bounds) is True  # left edge of dim 0
    assert _on_bounds([1.0, 0.0], bounds) is True  # right edge of dim 0
    assert _on_bounds([0.5, 4.9999], bounds) is True  # within margin of dim 1
    assert _on_bounds([0.5, 0.0], bounds) is False


# ---------------------------------------------------------------------------
# 3) L-BFGS-B finds an improvement and updates theta_0
# ---------------------------------------------------------------------------


def test_lbfgsb_improves_and_mutates_basement(datadir):
    theta_0_before = np.array(config.BASEMENT.theta_0, dtype=float)
    res = optimize(
        str(datadir),
        method="L-BFGS-B",
        refine=False,
        n_restarts=1,
        save=False,
        quiet=True,
        improvement_threshold=0.0,
        skip_bounds_check=True,
    )
    assert isinstance(res, OptimizeResult)
    assert res.success
    assert res.accepted
    assert res.lnprob_opt >= res.lnprob_initial
    # BASEMENT.theta_0 has been overwritten with the optimum
    assert not np.allclose(config.BASEMENT.theta_0, theta_0_before), (
        "theta_0 should have been mutated"
    )


def test_accepted_optimize_persists_to_params_csv(datadir):
    """The in-memory theta_0 mutation alone isn't enough: mcmc_fit / ns_fit /
    show_initial_guess all call config.init(datadir) again, which rebuilds
    BASEMENT straight from params.csv. Only a disk write survives that."""
    res = optimize(
        str(datadir),
        method="L-BFGS-B",
        refine=False,
        n_restarts=1,
        save=False,
        quiet=True,
        improvement_threshold=0.0,
        skip_bounds_check=True,
    )
    assert res.accepted

    # A fresh config.init (what every subsequent engine call does) must see
    # the optimized values, not the original guess.
    config.init(str(datadir))
    reloaded = dict(zip(config.BASEMENT.fitkeys, config.BASEMENT.theta_0))
    for key, value in zip(res.fitkeys, res.theta_opt):
        assert reloaded[key] == pytest.approx(value)


def test_accepted_optimize_persists_shifted_epoch_in_input_frame(tmp_path, monkeypatch):
    """Optimizer coordinates are data-centered, while params.csv stores the
    reference epoch from which Basement derives that centered coordinate.

    Persisting the centered value directly makes the next config.init infer a
    zero-cycle shift, leaving the normal prior about 100 days away. The writer
    must undo the integer epoch shift using the optimized period.
    """
    d = _make_shift_epoch_datadir(tmp_path)
    config.init(str(d))
    fitkeys = [str(key) for key in config.BASEMENT.fitkeys]
    epoch_idx = fitkeys.index("b_epoch")
    period_idx = fitkeys.index("b_period")
    theta_initial = np.asarray(config.BASEMENT.theta_0, dtype=float)
    assert theta_initial[epoch_idx] > 90.0  # shifted into the data window

    theta_opt = theta_initial.copy()
    theta_opt[epoch_idx] += 0.02
    theta_opt[period_idx] += 0.005

    def fake_run_local(method, x0, bounds, maxiter=None):
        return theta_opt.copy(), 1, True

    def fake_lnprob(theta):
        return 1.0 if np.allclose(theta, theta_opt) else 0.0

    monkeypatch.setitem(optimize.__globals__, "_run_local", fake_run_local)
    monkeypatch.setitem(optimize.__globals__, "mcmc_lnprob", fake_lnprob)

    res = optimize(
        str(d),
        method="L-BFGS-B",
        refine=False,
        save=False,
        quiet=True,
        improvement_threshold=0.0,
        skip_bounds_check=True,
    )
    assert res.accepted

    params_line = next(
        line for line in (d / "params.csv").read_text().splitlines() if line.startswith("b_epoch,")
    )
    persisted_epoch = float(params_line.split(",")[1])
    assert persisted_epoch < 10.0  # original params.csv reference frame
    assert "normal 0.5 0.1" in params_line  # prior definition is unchanged

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        config.init(str(d))
    assert not any("more than 3 sigma" in str(w.message) for w in caught)

    reloaded = {
        str(key): value for key, value in zip(config.BASEMENT.fitkeys, config.BASEMENT.theta_0)
    }
    assert reloaded["b_epoch"] == pytest.approx(theta_opt[epoch_idx])
    assert reloaded["b_period"] == pytest.approx(theta_opt[period_idx])
    epoch_prior = config.BASEMENT.bounds[epoch_idx]
    assert abs(reloaded["b_epoch"] - epoch_prior[1]) < 3.0 * epoch_prior[2]


# ---------------------------------------------------------------------------
# 4) acceptance gate rejects when delta < threshold (synthetic case)
# ---------------------------------------------------------------------------


def test_acceptance_gate_rejects_insufficient_improvement(datadir, monkeypatch):
    """Set an absurdly large improvement_threshold so even a real
    improvement is rejected — verifies BASEMENT.theta_0 stays put."""
    theta_0_before = np.array(config.BASEMENT.theta_0, dtype=float)
    res = optimize(
        str(datadir),
        method="L-BFGS-B",
        refine=False,
        n_restarts=1,
        save=False,
        quiet=True,
        improvement_threshold=1e30,
    )
    assert res.accepted is False
    assert "improvement_threshold" in res.reject_reason
    assert np.allclose(config.BASEMENT.theta_0, theta_0_before), (
        "theta_0 must NOT mutate when the run is rejected"
    )


# ---------------------------------------------------------------------------
# 5) mutate_basement=False never touches BASEMENT.theta_0
# ---------------------------------------------------------------------------


def test_mutate_basement_false_leaves_theta0(datadir):
    theta_0_before = np.array(config.BASEMENT.theta_0, dtype=float)
    res = optimize(
        str(datadir),
        method="L-BFGS-B",
        refine=False,
        n_restarts=1,
        save=False,
        quiet=True,
        improvement_threshold=0.0,
        mutate_basement=False,
        skip_bounds_check=True,
    )
    assert res.accepted
    assert np.allclose(config.BASEMENT.theta_0, theta_0_before)


def test_mutate_basement_false_leaves_params_csv(datadir):
    params_csv_before = (datadir / "params.csv").read_text()
    res = optimize(
        str(datadir),
        method="L-BFGS-B",
        refine=False,
        n_restarts=1,
        save=False,
        quiet=True,
        improvement_threshold=0.0,
        mutate_basement=False,
        skip_bounds_check=True,
    )
    assert res.accepted
    assert (datadir / "params.csv").read_text() == params_csv_before


def test_rejected_optimize_leaves_params_csv(datadir):
    params_csv_before = (datadir / "params.csv").read_text()
    res = optimize(
        str(datadir),
        method="L-BFGS-B",
        refine=False,
        n_restarts=1,
        save=False,
        quiet=True,
        improvement_threshold=1e30,
    )
    assert res.accepted is False
    assert (datadir / "params.csv").read_text() == params_csv_before


# ---------------------------------------------------------------------------
# 6) JSON persistence
# ---------------------------------------------------------------------------


def test_save_writes_json(datadir):
    optimize(
        str(datadir),
        method="L-BFGS-B",
        refine=False,
        n_restarts=1,
        save=True,
        quiet=True,
        improvement_threshold=0.0,
    )
    out = datadir / "results" / "optimize_save.json"
    assert out.exists()
    payload = json.loads(out.read_text())
    for key in (
        "method",
        "accepted",
        "lnprob_initial",
        "lnprob_opt",
        "theta_initial",
        "theta_opt",
        "fitkeys",
        "nfev",
        "wallclock_s",
        "restart_lnprobs",
    ):
        assert key in payload, key
    assert payload["method"] == "L-BFGS-B"
    assert len(payload["theta_opt"]) == config.BASEMENT.ndim


# ---------------------------------------------------------------------------
# 7) Multistart consistency gate
# ---------------------------------------------------------------------------


def test_multistart_consistency_rejection(datadir):
    """With 3 restarts and a tiny consistency_threshold, the run gets
    rejected on multistart spread (since Latin-hypercube draws inside the
    big uniform priors land on radically different lnprobs)."""
    theta_0_before = np.array(config.BASEMENT.theta_0, dtype=float)
    res = optimize(
        str(datadir),
        method="L-BFGS-B",
        refine=False,
        n_restarts=3,
        save=False,
        quiet=True,
        improvement_threshold=0.0,
        consistency_threshold=0.01,
        skip_bounds_check=True,
    )
    # one of: rejected for consistency, or accepted (if by luck all 3
    # land in the same basin within 0.01); the deterministic check we
    # want is "if rejected, theta_0 stays put".
    if not res.accepted:
        assert "consistency_threshold" in res.reject_reason
        assert np.allclose(config.BASEMENT.theta_0, theta_0_before)


# ---------------------------------------------------------------------------
# 8) CMA-ES path (skipped if cma not installed)
# ---------------------------------------------------------------------------


def test_cmaes_improves(datadir):
    pytest.importorskip("cma")
    res = optimize(
        str(datadir),
        method="cmaes",
        refine=True,
        n_restarts=1,
        save=False,
        quiet=True,
        improvement_threshold=0.0,
        maxfevals=200,
        seed=7,
    )
    assert res.success
    assert res.lnprob_opt >= res.lnprob_initial
    assert res.refine is True


# ---------------------------------------------------------------------------
# 9) CMA-ES warm-resume via pickled state
# ---------------------------------------------------------------------------


def test_cmaes_saves_state_on_every_call(datadir):
    pytest.importorskip("cma")
    pkl = datadir / "results" / "optimize_cma_state.pkl"
    assert not pkl.exists()
    optimize(
        str(datadir),
        method="cmaes",
        refine=False,
        n_restarts=1,
        save=False,
        quiet=True,
        improvement_threshold=0.0,
        skip_bounds_check=True,
        maxfevals=120,
        seed=11,
    )
    assert pkl.exists() and pkl.stat().st_size > 0


def test_cmaes_resume_continues_from_saved_state(datadir):
    pytest.importorskip("cma")
    # First call seeds the pickle.
    optimize(
        str(datadir),
        method="cmaes",
        refine=False,
        n_restarts=1,
        save=False,
        quiet=True,
        improvement_threshold=0.0,
        skip_bounds_check=True,
        maxfevals=120,
        seed=11,
    )
    # Second call with resume=True must load the saved strategy and
    # record that fact in OptimizeResult.resumed_from_pickle.
    res = optimize(
        str(datadir),
        method="cmaes",
        refine=False,
        n_restarts=1,
        save=False,
        quiet=True,
        improvement_threshold=0.0,
        skip_bounds_check=True,
        maxfevals=60,
        seed=11,
        resume=True,
    )
    assert res.resumed_from_pickle is True


def test_cmaes_resume_missing_pickle_warns_and_runs_fresh(datadir):
    pytest.importorskip("cma")
    pkl = datadir / "results" / "optimize_cma_state.pkl"
    assert not pkl.exists()
    with pytest.warns(UserWarning, match="resume=True but"):
        res = optimize(
            str(datadir),
            method="cmaes",
            refine=False,
            n_restarts=1,
            save=False,
            quiet=True,
            improvement_threshold=0.0,
            skip_bounds_check=True,
            maxfevals=60,
            seed=11,
            resume=True,
        )
    assert res.resumed_from_pickle is False
    # Even though we couldn't resume, a fresh run still seeds the pickle.
    assert pkl.exists()


def test_cmaes_resume_ndim_mismatch_raises(datadir):
    """If a stale pickle from a different fit (different ndim) is on
    disk, resume=True must refuse with a clear error."""
    pytest.importorskip("cma")
    import pickle as _pickle

    pkl = datadir / "results" / "optimize_cma_state.pkl"
    pkl.parent.mkdir(exist_ok=True)
    # Forge a (stub, meta) tuple with the wrong ndim. The strategy object
    # is never touched — _load_cma_state validates meta first and bails.
    with open(pkl, "wb") as f:
        _pickle.dump((object(), {"ndim": 99, "bounds": []}), f)
    with pytest.raises(ValueError, match="ndim="):
        optimize(
            str(datadir),
            method="cmaes",
            refine=False,
            n_restarts=1,
            save=False,
            quiet=True,
            improvement_threshold=0.0,
            skip_bounds_check=True,
            maxfevals=60,
            seed=11,
            resume=True,
        )


def test_resume_with_non_cmaes_method_raises(datadir):
    with pytest.raises(ValueError, match="resume=True is only meaningful"):
        optimize(str(datadir), method="L-BFGS-B", n_restarts=1, save=False, quiet=True, resume=True)


def test_resume_with_multiple_restarts_raises(datadir):
    pytest.importorskip("cma")
    with pytest.raises(ValueError, match="incompatible with n_restarts"):
        optimize(str(datadir), method="cmaes", n_restarts=3, save=False, quiet=True, resume=True)


def test_typo_kwarg_actionable(tmp_path):
    """Common kwarg typos must raise TypeError with a Did-you-mean hint."""
    import allesfitter

    cases = [
        ("maxfeval", "maxfevals"),
        ("mutate_baseline", "mutate_basement"),
        ("n_restart", "n_restarts"),
        ("improvement_thresh", "improvement_threshold"),
    ]
    for bad, expected in cases:
        with pytest.raises(TypeError) as ei:
            allesfitter.optimize(str(tmp_path), **{bad: 1})
        msg = str(ei.value)
        assert bad in msg, msg
        assert expected in msg, (bad, msg)
        assert "did you mean" in msg.lower()


# ---------------------------------------------------------------------------
# 10) logprint mirrors output into the run's timestamped logfile
# ---------------------------------------------------------------------------


def _logfile_path(datadir) -> Path:
    b = config.BASEMENT
    return Path(b.outdir) / ("logfile_" + b.now + ".log")


def test_logprint_writes_to_logfile_and_console(datadir, capsys):
    logprint("hello", "optimize")
    captured = capsys.readouterr()
    assert "hello optimize" in captured.out
    logf = _logfile_path(datadir)
    assert logf.exists()
    assert "hello optimize" in logf.read_text()


def test_logprint_quiet_suppresses_console_but_logs_to_file(datadir, capsys):
    logprint("quiet-line", quiet=True)
    captured = capsys.readouterr()
    assert "quiet-line" not in captured.out
    logf = _logfile_path(datadir)
    assert logf.exists()
    assert "quiet-line" in logf.read_text()


def test_logprint_without_basement_is_noop(monkeypatch, capsys):
    monkeypatch.setattr(config, "BASEMENT", None, raising=False)
    # Must not raise even though there is no BASEMENT to resolve a logfile.
    logprint("orphan", quiet=True)
    assert "orphan" not in capsys.readouterr().out


def test_optimize_summary_lands_in_logfile(datadir):
    optimize(
        str(datadir),
        method="L-BFGS-B",
        refine=False,
        n_restarts=1,
        save=False,
        quiet=True,
        improvement_threshold=0.0,
        skip_bounds_check=True,
    )
    logf = _logfile_path(datadir)
    assert logf.exists()
    assert "optimize[L-BFGS-B]" in logf.read_text()


def test_optimize_logs_start_and_per_restart_progress(datadir):
    optimize(
        str(datadir),
        method="L-BFGS-B",
        refine=False,
        n_restarts=2,
        save=False,
        quiet=True,
        improvement_threshold=0.0,
        consistency_threshold=1e30,
        skip_bounds_check=True,
    )
    text = _logfile_path(datadir).read_text()
    # start marker
    assert "starting: ndim=" in text
    # one progress line per restart
    assert "restart 1/2:" in text
    assert "restart 2/2:" in text


def test_optimize_verbose_accepted_for_cmaes(datadir):
    pytest.importorskip("cma")
    # verbose=True must be a valid kwarg and not raise; it streams the cma
    # table to stdout but the run result is unaffected.
    res = optimize(
        str(datadir),
        method="cmaes",
        refine=False,
        n_restarts=1,
        save=False,
        quiet=True,
        verbose=True,
        improvement_threshold=0.0,
        skip_bounds_check=True,
        maxfevals=120,
        seed=13,
    )
    assert res.success


def test_cmaes_missing_dependency_raises_clear_error(datadir, monkeypatch):
    """If `import cma` fails, optimize() should raise an ImportError with
    a guidance message rather than crash deep inside the helper."""
    import sys

    # Hide the real cma module (if any) and inject an importer that fails.
    monkeypatch.setitem(sys.modules, "cma", None)
    with pytest.raises(ImportError, match="pip install cma"):
        optimize(
            str(datadir),
            method="cmaes",
            refine=False,
            n_restarts=1,
            save=False,
            quiet=True,
            improvement_threshold=0.0,
            maxfevals=50,
        )
