"""Tests for the nested-sampling backend dispatch and the two backends.

Two layers:

* **Unit** (always run): dispatch + schema. Validate that ``get_backend``
  returns the right module, raises on unknown names, and that
  ``NSResults`` behaves as a dict + namespace and survives pickle.

* **Slow** (``-m slow``): actually run both samplers on a small 2D
  Gaussian and compare posterior means and ``logZ`` for **consistency**,
  plus record wall time for a **speed** snapshot. UltraNest tests are
  ``importorskip``'d so the suite is green even when the optional extra
  is not installed.

The slow tests deliberately sidestep allesfitter's full ``config`` /
``Basement`` setup — they call the backend ``run()`` functions directly
with a synthetic loglike. That keeps the test self-contained, fast, and
focused on the backend contract.
"""

from __future__ import annotations

import json
import os
import pickle
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from allesfitter.utils.ns_backends import (
    BACKEND_SETTINGS,
    KNOWN_BACKENDS,
    NSResults,
    build_results,
    get_backend,
    validate_settings_for_backend,
)


# ---------------------------------------------------------------------------
# Synthetic 2D Gaussian — used by the slow consistency/speed tests.
# ---------------------------------------------------------------------------

_MU = np.array([0.3, -0.7])
_SIGMA = np.array([0.5, 0.4])
_LO, _HI = -5.0, 5.0
_TRUE_LOGZ = float(
    # ln of integral of Gaussian over uniform prior box, analytic for product of 1D Gaussians.
    # ln Z = sum_i [ ln( sqrt(2 pi) sigma_i ) - ln(HI-LO) ]
    np.sum(0.5 * np.log(2 * np.pi) + np.log(_SIGMA) - np.log(_HI - _LO))
)


def _loglike(theta):
    return float(-0.5 * np.sum(((theta - _MU) / _SIGMA) ** 2))


def _prior_transform(u):
    return _LO + u * (_HI - _LO)


def _baseline_settings(nlive=200, tol=0.5, modus="static"):
    return {
        "ns_nlive": nlive,
        "ns_bound": "single",
        "ns_sample": "rwalk",
        "ns_tol": tol,
        "ns_modus": modus,
        "multiprocess": False,
        "multiprocess_cores": 1,
        "print_progress": False,
        "un_min_ess": 200,
        "un_max_iters": None,
    }


def _final(arr):
    return float(np.asarray(arr)[-1])


# ---------------------------------------------------------------------------
# Unit: dispatch + schema (no sampling).
# ---------------------------------------------------------------------------


def test_ultranest_safe_loglike_floors_nonfinite():
    """UltraNest aborts on -inf / NaN log-likelihoods; the safe wrapper
    must replace them with a large negative finite floor while passing
    finite values through untouched."""
    pytest.importorskip("ultranest")
    from allesfitter.utils.ns_backends.ultranest_backend import (
        _safe_loglike, _NONFINITE_LOGL_FLOOR,
    )
    raw_neginf = _safe_loglike(lambda t: float("-inf"))
    raw_posinf = _safe_loglike(lambda t: float("inf"))
    raw_nan = _safe_loglike(lambda t: float("nan"))
    raw_finite = _safe_loglike(lambda t: -123.5)
    assert raw_neginf(None) == _NONFINITE_LOGL_FLOOR
    assert raw_posinf(None) == _NONFINITE_LOGL_FLOOR
    assert raw_nan(None) == _NONFINITE_LOGL_FLOOR
    assert raw_finite(None) == -123.5


@pytest.mark.parametrize("name", KNOWN_BACKENDS)
def test_get_backend_known(name):
    be = get_backend(name)
    assert be.name == name
    assert callable(be.run)


def test_get_backend_unknown_raises():
    with pytest.raises(ValueError, match="Unknown ns backend"):
        get_backend("nope")


def test_get_backend_default_when_empty():
    # None or '' falls back to dynesty
    assert get_backend(None).name == "dynesty"
    assert get_backend("").name == "dynesty"


def test_NSResults_dict_and_attr_access():
    r = NSResults(samples=np.zeros((3, 2)), logwt=np.zeros(3), logz=np.array([1.0]))
    assert r["samples"].shape == (3, 2)
    assert r.samples.shape == (3, 2)
    assert r.logz[-1] == 1.0
    r.foo = "bar"
    assert r["foo"] == "bar"
    with pytest.raises(AttributeError):
        _ = r.does_not_exist


def test_NSResults_pickleable_and_deepcopy():
    r = build_results(
        backend="dynesty",
        samples=np.arange(6).reshape(3, 2),
        logwt=np.array([0.0, -1.0, -2.0]),
        logz=np.array([-1.5, -1.2, -1.0]),
        logzerr=np.array([0.5, 0.3, 0.2]),
        fitkeys=["a", "b"],
        wall_time_sec=1.23,
    )
    # round-trip via pickle
    blob = pickle.dumps(r)
    r2 = pickle.loads(blob)
    assert isinstance(r2, NSResults)
    assert r2["backend"] == "dynesty"
    assert _final(r2["logz"]) == -1.0
    # deep copy + in-place mutation (ns_output relies on this)
    r3 = deepcopy(r)
    r3["samples"][:, 0] -= 1
    assert r["samples"][0, 0] == 0  # original unchanged


def test_validator_flags_ignored_keys_for_dynesty():
    logs = []
    raw = {"ns_nlive", "ns_bound", "ns_sample", "ns_tol", "ns_modus",
           "un_min_ess"}  # un_min_ess is irrelevant under dynesty
    validate_settings_for_backend("dynesty", raw, logprint=logs.append)
    assert any("IGNORED" in m and "un_min_ess" in m for m in logs), logs


def test_validator_flags_ignored_keys_for_ultranest():
    logs = []
    raw = {"ns_nlive", "ns_tol", "ns_modus", "ns_bound", "ns_sample",
           "un_min_ess", "un_max_iters"}
    validate_settings_for_backend("ultranest", raw, logprint=logs.append)
    # dynesty-only keys are ignored under ultranest
    assert any("IGNORED" in m and "ns_modus" in m for m in logs), logs
    assert any("ns_bound" in m for m in logs), logs


def test_validator_flags_missing_relevant_keys():
    logs = []
    # user set ns_nlive but forgot the rest
    raw = {"ns_nlive"}
    validate_settings_for_backend("dynesty", raw, logprint=logs.append)
    msg = "\n".join(logs)
    for k in ("ns_bound", "ns_sample", "ns_tol", "ns_modus"):
        assert k in msg, "missing-key warning should mention {}".format(k)


def test_validator_quiet_when_fully_explicit():
    logs = []
    raw = set(BACKEND_SETTINGS["dynesty"]["relevant"])
    validate_settings_for_backend("dynesty", raw, logprint=logs.append)
    # the "all keys set" green-check line is the only output
    joined = "\n".join(logs)
    assert "IGNORED" not in joined
    assert "NOT set" not in joined
    assert "explicitly set" in joined


def test_build_results_required_keys():
    r = build_results(
        backend="ultranest",
        samples=np.zeros((4, 2)),
        logwt=np.zeros(4),
        logz=np.array([0.0]),
        logzerr=np.array([0.1]),
        fitkeys=["x", "y"],
        wall_time_sec=0.0,
    )
    required = {"backend", "samples", "logwt", "logz", "logzerr",
                "fitkeys", "wall_time_sec", "raw"}
    assert required.issubset(set(r.keys()))


# ---------------------------------------------------------------------------
# Slow: real sampling. Compare dynesty vs ultranest on the 2D Gaussian.
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_dynesty_backend_recovers_2d_gaussian(tmp_path):
    be = get_backend("dynesty")
    r = be.run(
        _loglike, _prior_transform, ndim=2,
        settings=_baseline_settings(),
        param_names=["x", "y"],
        outdir=str(tmp_path),
    )
    _assert_schema(r, backend="dynesty", ndim=2, fitkeys=["x", "y"])
    means = _posterior_means(r)
    assert np.allclose(means, _MU, atol=0.1), means
    assert abs(_final(r["logz"]) - _TRUE_LOGZ) < 0.5


@pytest.mark.slow
def test_ultranest_backend_recovers_2d_gaussian(tmp_path):
    pytest.importorskip("ultranest")
    be = get_backend("ultranest")
    r = be.run(
        _loglike, _prior_transform, ndim=2,
        settings=_baseline_settings(),
        param_names=["x", "y"],
        outdir=str(tmp_path),
    )
    _assert_schema(r, backend="ultranest", ndim=2, fitkeys=["x", "y"])
    means = _posterior_means(r)
    assert np.allclose(means, _MU, atol=0.1), means
    assert abs(_final(r["logz"]) - _TRUE_LOGZ) < 0.5


@pytest.mark.slow
def test_ultranest_clamps_huge_dlogz_and_warns(tmp_path):
    pytest.importorskip("ultranest")
    logs = []
    settings = _baseline_settings()
    settings["ns_tol"] = 100.0  # dynesty-static-mode default, nonsensical for ultranest
    r = get_backend("ultranest").run(
        _loglike, _prior_transform, ndim=2,
        settings=settings,
        param_names=["x", "y"],
        outdir=str(tmp_path),
        logprint=logs.append,
    )
    assert any("clamping to dlogz=0.5" in m for m in logs), logs
    # And still converges
    assert np.allclose(_posterior_means(r), _MU, atol=0.15)


@pytest.mark.slow
def test_ultranest_recovers_from_truncated_h5_store(tmp_path):
    pytest.importorskip("ultranest")
    # Simulate a prior aborted run: create the log_dir with a corrupt h5 file.
    log_dir = tmp_path / "ultranest_logs"
    log_dir.mkdir()
    chains = log_dir / "chains"
    chains.mkdir()
    # Truncated/garbage HDF5 — what an aborted run typically leaves behind.
    (chains / "weighted_post_untransformed.txt").write_text("not h5\n")
    points_h5 = chains / "run1" / "points.hdf5"
    points_h5.parent.mkdir(parents=True, exist_ok=True)
    points_h5.write_bytes(b"\x89HDF\r\n\x1a\n" + b"\x00" * 88)  # truncated header

    logs = []
    r = get_backend("ultranest").run(
        _loglike, _prior_transform, ndim=2,
        settings=_baseline_settings(),
        param_names=["x", "y"],
        outdir=str(tmp_path),
        logprint=logs.append,
    )
    # Either the broken store was detected and rotated, or ultranest happened
    # not to need it (newer versions probe differently). The contract we care
    # about is: the run completed and we got valid results.
    _assert_schema(r, backend="ultranest", ndim=2, fitkeys=["x", "y"])


@pytest.mark.slow
def test_dynesty_vs_ultranest_consistency_and_speed(tmp_path):
    """Cross-backend consistency: posteriors within 3σ, |Δlog Z| within 5σ.

    Also records wall time for both into a JSON artifact so CI / humans
    can review the speed snapshot without it gating the suite.
    """
    pytest.importorskip("ultranest")

    out_dyn = tmp_path / "dyn"
    out_un = tmp_path / "un"
    out_dyn.mkdir()
    out_un.mkdir()

    t0 = time.perf_counter()
    r_dyn = get_backend("dynesty").run(
        _loglike, _prior_transform, ndim=2,
        settings=_baseline_settings(),
        param_names=["x", "y"],
        outdir=str(out_dyn),
    )
    t_dyn = time.perf_counter() - t0

    t0 = time.perf_counter()
    r_un = get_backend("ultranest").run(
        _loglike, _prior_transform, ndim=2,
        settings=_baseline_settings(),
        param_names=["x", "y"],
        outdir=str(out_un),
    )
    t_un = time.perf_counter() - t0

    mu_dyn = _posterior_means(r_dyn)
    mu_un = _posterior_means(r_un)
    sigma = np.maximum(_posterior_stds(r_dyn), _posterior_stds(r_un))
    diff = np.abs(mu_dyn - mu_un)
    assert np.all(diff < 3 * sigma), (
        "Posterior means disagree: |Δ|={} > 3σ={}".format(diff, 3 * sigma)
    )

    logz_dyn = _final(r_dyn["logz"])
    logz_un = _final(r_un["logz"])
    logzerr_max = max(_final(r_dyn["logzerr"]), _final(r_un["logzerr"]))
    assert abs(logz_dyn - logz_un) < 5 * logzerr_max, (
        "logZ disagrees beyond combined uncertainty: "
        "dyn={}, un={}, max_err={}".format(logz_dyn, logz_un, logzerr_max)
    )

    artifact_dir = Path(__file__).parent / "_artifacts"
    artifact_dir.mkdir(exist_ok=True)
    (artifact_dir / "ns_backend_speed.json").write_text(json.dumps({
        "dynesty_wall_time_sec": t_dyn,
        "ultranest_wall_time_sec": t_un,
        "dynesty_logz": logz_dyn,
        "ultranest_logz": logz_un,
        "delta_logz": logz_dyn - logz_un,
        "posterior_means_dyn": mu_dyn.tolist(),
        "posterior_means_un": mu_un.tolist(),
    }, indent=2))


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _assert_schema(r, *, backend, ndim, fitkeys):
    assert isinstance(r, NSResults)
    assert r["backend"] == backend
    assert r["samples"].ndim == 2 and r["samples"].shape[1] == ndim
    assert r["logwt"].shape[0] == r["samples"].shape[0]
    assert np.asarray(r["logz"]).ndim >= 1
    assert np.asarray(r["logzerr"]).ndim >= 1
    assert list(r["fitkeys"]) == list(fitkeys)
    assert r["wall_time_sec"] >= 0.0


def _posterior_means(r):
    samples = np.asarray(r["samples"])
    logwt = np.asarray(r["logwt"])
    logz_final = _final(r["logz"])
    w = np.exp(logwt - logz_final)
    w = w / w.sum()
    return np.average(samples, axis=0, weights=w)


def _posterior_stds(r):
    samples = np.asarray(r["samples"])
    logwt = np.asarray(r["logwt"])
    logz_final = _final(r["logz"])
    w = np.exp(logwt - logz_final)
    w = w / w.sum()
    mean = np.average(samples, axis=0, weights=w)
    var = np.average((samples - mean) ** 2, axis=0, weights=w)
    return np.sqrt(var)
