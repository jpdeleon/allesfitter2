"""Global + local optimization of the joint posterior, intended as a
warm-start for mcmc_fit / ns_fit.

Default stack: CMA-ES global search → L-BFGS-B refine → acceptance gates →
optional update of ``config.BASEMENT.theta_0`` *and* the ``value`` column of
``params.csv`` on disk, so that the next sampler call starts from the
optimized point. The disk write matters because ``mcmc_fit`` / ``ns_fit`` /
``show_initial_guess`` each call ``config.init(datadir)``, which rebuilds
BASEMENT from ``params.csv`` — an in-memory-only ``theta_0`` update would be
silently discarded before the next call ever sees it.

Public entry point:

    >>> import allesfitter
    >>> res = allesfitter.optimize('.', method='cmaes', refine=True)
    >>> if res.accepted:
    ...     allesfitter.mcmc_fit('.')   # warm-started

The acceptance gates (improvement, bounds, multistart consistency) make
``optimize()`` safe to call unconditionally — if the optimizer fails to
beat the user's hand-tuned ``theta_0``, BASEMENT is left untouched.
"""

import json
import os
import pickle
import shutil
import sys
import tempfile
import warnings
from dataclasses import asdict, dataclass, field
from time import time as _now
from typing import Any

import numpy as np
from scipy import optimize as scipy_opt

# Filename for the pickled CMA-ES strategy state used by resume=True.
_CMA_STATE_FILENAME = "optimize_cma_state.pkl"

from . import config
from .mcmc import mcmc_lnprob


###############################################################################
#::: print function that prints into console and logfile at the same time
###############################################################################
def logprint(*text: Any, quiet: bool = False) -> None:
    """Print to console and append to the run's timestamped logfile.

    Mirrors ``general_output.logprint`` so that ``optimize()`` output is
    recorded in the same ``logfile_<now>.log`` as the sampler runs. The
    console echo is suppressed when ``quiet=True``; the logfile is written
    regardless (best-effort — silently skipped if BASEMENT or the output
    directory is unavailable).
    """
    if not quiet:
        print(*text)
    b = getattr(config, "BASEMENT", None)
    if b is None:
        return
    original = sys.stdout
    try:
        logpath = os.path.join(b.outdir, "logfile_" + b.now + ".log")
        with open(logpath, "a") as f:
            sys.stdout = f
            print(*text)
    except OSError:
        # Some Windows versions choke on the open()/os.path.join() combo;
        # also covers a missing/unwritable outdir.
        pass
    finally:
        sys.stdout = original


# Methods that scipy.optimize.minimize accepts a `bounds` keyword for.
_LOCAL_METHODS = ("L-BFGS-B", "Powell", "TNC", "SLSQP", "trust-constr")
# Global methods we wrap explicitly.
_GLOBAL_METHODS = ("cmaes", "dual_annealing", "differential_evolution")


@dataclass
class OptimizeResult:
    """Lightweight, JSON-serialisable summary of an optimize() run."""

    method: str
    accepted: bool
    success: bool
    lnprob_initial: float
    lnprob_opt: float
    delta_lnprob: float
    theta_initial: list
    theta_opt: list
    fitkeys: list
    n_restarts: int
    restart_lnprobs: list
    nfev: int
    wallclock_s: float
    reject_reason: str = ""
    refine: bool = False
    bounds: list = field(default_factory=list)
    resumed_from_pickle: bool = False
    baseline_first: bool = False
    baseline_stage_accepted: bool = False
    baseline_stage_fitkeys: list = field(default_factory=list)
    baseline_stage_nfev: int = 0
    baseline_stage_wallclock_s: float = 0.0


def _extract_bounds(basement, normal_sigma_clip: float = 5.0):
    """Return per-fit-parameter ``[lo, hi]`` for optimizer ``bounds`` kwargs.

    Uniform / trunc_normal priors give their explicit limits. Normal priors
    fall back to ``mu ± normal_sigma_clip * sigma`` since the optimizers
    we use require finite bounds.
    """
    bounds = []
    for b in basement.bounds:
        kind = b[0]
        if kind == "uniform":
            bounds.append((float(b[1]), float(b[2])))
        elif kind == "trunc_normal":
            bounds.append((float(b[3]), float(b[4])))
        elif kind == "normal":
            mu, sigma = float(b[1]), float(b[2])
            bounds.append((mu - normal_sigma_clip * sigma, mu + normal_sigma_clip * sigma))
        else:
            raise ValueError(f"optimize: unsupported prior kind '{kind}' for fitkey")
    return bounds


def _clip_to_bounds(theta, bounds):
    lo = np.array([b[0] for b in bounds], dtype=float)
    hi = np.array([b[1] for b in bounds], dtype=float)
    return np.clip(np.asarray(theta, dtype=float), lo, hi)


def _on_bounds(theta, bounds, margin_frac: float = 1e-4):
    """True iff any component sits on (within `margin_frac`*range of) a
    bound edge."""
    theta = np.asarray(theta, dtype=float)
    for x, (lo, hi) in zip(theta, bounds):
        span = max(hi - lo, 1e-30)
        if (x - lo) < margin_frac * span or (hi - x) < margin_frac * span:
            return True
    return False


def _objective(theta):
    """Scalar to minimise: ``-mcmc_lnprob``. ``mcmc_lnprob`` returns
    ``-np.inf`` outside the prior box; we map that to a large finite
    positive value so gradient-using optimizers don't explode."""
    lp = mcmc_lnprob(np.asarray(theta, dtype=float))
    if not np.isfinite(lp):
        return 1e30
    return -lp


def _default_sigma0(theta_0, bounds, basement):
    """CMA-ES asks for one scalar initial std. Use the geometric mean of
    ``0.1 * (hi - lo)`` (10 % of the prior width per axis) as the
    isotropic σ₀ — CMA-ES will adapt per-axis scaling internally."""
    spans = np.array([hi - lo for lo, hi in bounds], dtype=float)
    spans = spans[spans > 0]
    if len(spans) == 0:
        # Fallback to init_err if every bound collapsed (shouldn't happen).
        ie = np.atleast_1d(basement.init_err)
        return float(np.max(ie)) if ie.size else 1e-2
    return float(0.1 * np.exp(np.mean(np.log(spans))))


# --------------------------------------------------------------------------
# per-method runners — each returns (theta_best, nfev, success)
# --------------------------------------------------------------------------


def _run_local(method, x0, bounds, maxiter=None, options=None):
    opts = dict(options or {})
    if maxiter is not None:
        opts.setdefault("maxiter", int(maxiter))
    res = scipy_opt.minimize(_objective, x0, method=method, bounds=bounds, options=opts)
    return np.asarray(res.x, dtype=float), int(res.nfev), bool(res.success)


def _run_differential_evolution(x0, bounds, maxiter=None, seed=None, workers=1):
    res = scipy_opt.differential_evolution(
        _objective,
        bounds=bounds,
        init="sobol",
        x0=x0,
        maxiter=int(maxiter) if maxiter else 100,
        seed=seed,
        polish=False,
        workers=workers,
        updating="deferred",
    )
    return np.asarray(res.x, dtype=float), int(res.nfev), bool(res.success)


def _run_dual_annealing(x0, bounds, maxiter=None, seed=None):
    res = scipy_opt.dual_annealing(
        _objective,
        bounds=bounds,
        x0=x0,
        maxiter=int(maxiter) if maxiter else 1000,
        seed=seed,
        no_local_search=False,
    )
    return np.asarray(res.x, dtype=float), int(res.nfev), bool(res.success)


def _load_cma_state(path, ndim, bounds):
    """Load a pickled (strategy, metadata) tuple from `path` and validate
    it matches the current problem dimensions. Returns the strategy or
    raises ValueError on mismatch.
    """
    with open(path, "rb") as f:
        es, meta = pickle.load(f)
    if int(meta.get("ndim", -1)) != int(ndim):
        raise ValueError(
            "optimize: resume=True but saved CMA-ES state at '{p}' was for "
            "ndim={saved}, current fit has ndim={now}. Delete the file or "
            "set resume=False.".format(p=path, saved=meta.get("ndim"), now=ndim)
        )
    saved_bounds = meta.get("bounds")
    if saved_bounds is not None:
        # Tolerate tiny float-repr drift but flag real changes.
        try:
            sb = np.asarray(saved_bounds, dtype=float)
            cb = np.asarray(bounds, dtype=float)
            if sb.shape != cb.shape or not np.allclose(sb, cb, atol=1e-12):
                raise ValueError(
                    "optimize: resume=True but bounds in saved state differ "
                    f"from current bounds. Delete '{path}' or set resume=False."
                )
        except ValueError:
            raise
        except Exception:
            pass
    return es


def _save_cma_state(path, es, ndim, bounds):
    """Pickle (strategy, metadata) to `path` for a future resume=True."""
    meta = {"ndim": int(ndim), "bounds": [list(b) for b in bounds]}
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump((es, meta), f, protocol=pickle.HIGHEST_PROTOCOL)


def _run_cmaes(
    x0, bounds, sigma0, maxfevals=None, seed=None, verbose=False, resume_path=None, save_path=None
):
    """Run CMA-ES; optionally resume from a pickled state.

    Parameters
    ----------
    resume_path : str | None
        If given AND the file exists, unpickle and continue from that
        strategy. The function ignores `x0` and `sigma0` in that case
        (the saved state already carries its own mean / sigma / C).
    save_path : str | None
        If given, pickle the final strategy (after optimization) for a
        future ``resume=True`` call.
    """
    try:
        import cma
    except ImportError as e:
        raise ImportError(
            "optimize(method='cmaes') requires the 'cma' package. "
            "Install it with `pip install cma` or pick a scipy-only "
            "method such as 'L-BFGS-B' / 'dual_annealing'."
        ) from e

    es = None
    resumed = False
    if resume_path and os.path.exists(resume_path):
        # Caller already validated ndim/bounds via _load_cma_state when it
        # decided whether to resume; we just unpickle here.
        with open(resume_path, "rb") as f:
            es, _meta = pickle.load(f)
        resumed = True
        # If the user passed a maxfevals budget for THIS call, extend the
        # strategy's option so es.optimize() runs that many more evals.
        if maxfevals is not None:
            try:
                es.opts.set({"maxfevals": int(es.countevals) + int(maxfevals)})
            except Exception:
                pass

    if es is None:
        lo = [b[0] for b in bounds]
        hi = [b[1] for b in bounds]
        opts = {
            "bounds": [lo, hi],
            "verbose": 0 if not verbose else 1,
            "seed": seed if seed is not None else np.random.randint(1, 2**31 - 1),
        }
        if maxfevals is not None:
            opts["maxfevals"] = int(maxfevals)
        es = cma.CMAEvolutionStrategy(list(x0), float(sigma0), opts)

    es.optimize(_objective)

    if save_path:
        _save_cma_state(save_path, es, ndim=len(bounds), bounds=bounds)

    return (
        np.asarray(es.result.xbest, dtype=float),
        int(es.result.evaluations),
        bool(es.result.xbest is not None),
        resumed,
    )


def _write_theta_to_params_csv(datadir, fitkeys, theta) -> None:
    """Overwrite the ``value`` column of ``params.csv`` for rows in *fitkeys*.

    Every other line — comments, section headers, other columns — is left
    byte-for-byte untouched. This is what makes ``mutate_basement=True``'s
    documented promise ("the next sampler call starts from the optimized
    point") actually true: ``mcmc_fit`` / ``ns_fit`` / ``show_initial_guess``
    all call ``config.init(datadir)``, which reconstructs BASEMENT from disk
    and would otherwise silently discard an in-memory-only ``theta_0`` update.
    """
    path = os.path.join(datadir, "params.csv")
    updates = {str(k): float(v) for k, v in zip(fitkeys, theta)}
    with open(path, encoding="utf-8") as f:
        lines = f.readlines()

    out = []
    for line in lines:
        stripped = line.strip()
        if not updates or not stripped or stripped.startswith("#"):
            out.append(line)
            continue
        parts = line.split(",", 2)
        name = parts[0].strip() if parts else ""
        if len(parts) >= 2 and name in updates:
            rest = parts[2] if len(parts) > 2 else "\n" if line.endswith("\n") else ""
            out.append(f"{parts[0]},{updates.pop(name)!r},{rest}")
        else:
            out.append(line)

    with open(path, "w", encoding="utf-8") as f:
        f.writelines(out)


def _theta_in_params_frame(basement, theta):
    """Convert optimized internal coordinates back to the params.csv frame.

    With ``shift_epoch=True``, :meth:`Basement.change_epoch` moves every
    fitted epoch into the data window and shifts its prior by the same number
    of periods. The CSV remains in the original reference frame so that a
    future ``config.init`` can repeat both transformations. Persisting the
    internal epoch directly would make that future shift zero and leave the
    prior behind in the original frame.

    Undo each recorded integer shift with the *optimized* period. This keeps
    the optimized ephemeris exactly cycle-equivalent after a fresh reload,
    including when both epoch and period changed during optimization.
    """
    theta_params = np.asarray(theta, dtype=float).copy()
    if not basement.settings.get("shift_epoch", False):
        return theta_params

    fitkey_index = {str(key): i for i, key in enumerate(basement.fitkeys)}
    for companion, n_shift in getattr(basement, "epoch_shift_periods", {}).items():
        epoch_key = f"{companion}_epoch"
        epoch_idx = fitkey_index.get(epoch_key)
        if epoch_idx is None or n_shift == 0:
            continue

        period_key = f"{companion}_period"
        period_idx = fitkey_index.get(period_key)
        if period_idx is None:
            period = float(basement.params[period_key])
        else:
            period = float(theta_params[period_idx])
        theta_params[epoch_idx] -= int(n_shift) * period

    return theta_params


def _is_photometric_baseline_key(key: Any) -> bool:
    """Return whether *key* is a sampled photometric-baseline parameter."""
    key = str(key)
    return key.startswith("baseline_") and "_flux_" in key


def _set_setting(path: str, name: str, value: Any) -> None:
    """Set one row in a two-column settings.csv, appending it if absent."""
    with open(path, encoding="utf-8") as f:
        lines = f.readlines()

    replacement = f"{name},{value}\n"
    found = False
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.split(",", 1)[0].strip() == name:
            lines[i] = replacement
            found = True
            break
    if not found:
        if lines and not lines[-1].endswith("\n"):
            lines[-1] += "\n"
        lines.append(replacement)

    with open(path, "w", encoding="utf-8") as f:
        f.writelines(lines)


def _freeze_nonbaseline_params(path: str) -> None:
    """Freeze every fitted CSV row except photometric baseline parameters.

    Existing ``fit=0`` baseline rows remain fixed. Parameters synthesized by
    the loader (currently ``sample_linear_multi`` weights) are unaffected and
    are therefore still available to the baseline-only fit.
    """
    with open(path, encoding="utf-8") as f:
        lines = f.readlines()

    out = []
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            out.append(line)
            continue
        parts = line.split(",", 3)
        if len(parts) >= 3 and not _is_photometric_baseline_key(parts[0].strip()):
            parts[2] = "0"
            line = ",".join(parts)
        out.append(line)

    with open(path, "w", encoding="utf-8") as f:
        f.writelines(out)


def _make_baseline_stage_datadir(datadir, stage_dir, basement, mask_width=None) -> None:
    """Build an isolated datadir for the out-of-transit baseline fit.

    Root-level inputs are symlinked where possible so large light curves are
    not copied. ``settings.csv`` and ``params.csv`` are real copies because the
    stage changes them. The caller's current in-memory theta is written into
    the copied params file before non-baseline rows are frozen.
    """
    for entry in os.scandir(datadir):
        if not entry.is_file():
            continue
        dst = os.path.join(stage_dir, entry.name)
        if entry.name in ("settings.csv", "params.csv"):
            shutil.copy2(entry.path, dst)
            continue
        try:
            os.symlink(os.path.abspath(entry.path), dst)
        except OSError:
            shutil.copy2(entry.path, dst)

    params_path = os.path.join(stage_dir, "params.csv")
    theta_params = _theta_in_params_frame(basement, basement.theta_0)
    _write_theta_to_params_csv(stage_dir, basement.fitkeys, theta_params)
    _freeze_nonbaseline_params(params_path)

    settings_path = os.path.join(stage_dir, "settings.csv")
    _set_setting(settings_path, "fast_fit", "False")
    _set_setting(settings_path, "mask_transit", "True")
    if mask_width is not None:
        if not np.isfinite(mask_width) or float(mask_width) <= 0:
            raise ValueError("optimize: baseline_mask_width must be a positive finite number")
        _set_setting(settings_path, "fast_fit_width", repr(float(mask_width)))


def _baseline_warm_start(
    datadir,
    basement,
    *,
    method,
    refine,
    n_restarts,
    sigma0,
    maxfevals,
    seed,
    consistency_threshold,
    skip_bounds_check,
    workers,
    quiet,
    verbose,
    mask_width,
):
    """Fit baseline parameters on OOT data and restore the full Basement.

    Returns the baseline-stage result. Its accepted values are injected into
    the restored full-data ``theta_0`` in memory; no original input file is
    changed here. The public optimize call remains responsible for persistence
    after its full-data acceptance gates pass.
    """
    theta_before = {str(k): float(v) for k, v in zip(basement.fitkeys, basement.theta_0)}
    stage_result = None
    try:
        with tempfile.TemporaryDirectory(prefix="allesfitter-baseline-") as stage_dir:
            _make_baseline_stage_datadir(datadir, stage_dir, basement, mask_width=mask_width)
            stage_result = optimize(
                stage_dir,
                method=method,
                refine=refine,
                n_restarts=n_restarts,
                sigma0=sigma0,
                maxfevals=maxfevals,
                seed=seed,
                save=False,
                mutate_basement=False,
                improvement_threshold=0.0,
                consistency_threshold=consistency_threshold,
                skip_bounds_check=skip_bounds_check,
                workers=workers,
                quiet=quiet,
                verbose=verbose,
                resume=False,
                baseline_first=False,
            )
    finally:
        # The nested optimize points config.BASEMENT at the temporary stage.
        # Always restore the caller's full-data configuration, even if the
        # stage raises, and preserve any in-memory starting values it supplied.
        config.init(datadir)
        restored = config.BASEMENT
        restored.theta_0 = np.asarray(
            [theta_before.get(str(k), v) for k, v in zip(restored.fitkeys, restored.theta_0)],
            dtype=float,
        )

    if stage_result is not None and stage_result.accepted:
        baseline_values = {
            str(k): float(v)
            for k, v in zip(stage_result.fitkeys, stage_result.theta_opt)
            if _is_photometric_baseline_key(k)
        }
        restored.theta_0 = np.asarray(
            [baseline_values.get(str(k), v) for k, v in zip(restored.fitkeys, restored.theta_0)],
            dtype=float,
        )
    return stage_result


def _refine(theta, bounds, maxiter=30):
    """L-BFGS-B finite-diff refinement within `maxiter` steps. Catches and
    returns the input theta unchanged if the refinement fails."""
    try:
        x, nfev, _ = _run_local("L-BFGS-B", theta, bounds, maxiter=maxiter)
        # only accept the refinement if it didn't regress
        if _objective(x) < _objective(theta):
            return x, nfev
    except Exception:
        pass
    return np.asarray(theta, dtype=float), 0


# --------------------------------------------------------------------------
# public entry point
# --------------------------------------------------------------------------


def optimize(
    datadir: str,
    method: str = "differential_evolution",
    refine: bool = True,
    n_restarts: int = 1,
    sigma0: float = None,
    maxfevals: int = None,
    seed: int = 42,
    save: bool = True,
    mutate_basement: bool = True,
    improvement_threshold: float = None,
    consistency_threshold: float = 1.0,
    skip_bounds_check: bool = False,
    workers: int = 1,
    quiet: bool = False,
    verbose: bool = False,
    resume: bool = False,
    baseline_first: bool = True,
    baseline_mask_width: float = None,
    **_extra_kwargs,
) -> OptimizeResult:
    """Find a posterior maximum and (optionally) push it into BASEMENT.theta_0.

    By default, sampled photometric baseline parameters are initialized in a
    separate out-of-transit fit before the joint fit. The stage masks every
    primary transit (and secondary eclipse when enabled), freezes all science
    parameters, and changes no user file. The accepted baseline point then
    warm-starts the regular full-data optimization below.

    Parameters
    ----------
    datadir : str
        Path passed to ``config.init(datadir)`` if BASEMENT is not already
        populated.
    method : str
        One of ``'differential_evolution'`` (default), ``'cmaes'`` (requires
        ``pip install cma``), ``'dual_annealing'``, ``'L-BFGS-B'``,
        ``'Powell'``, ``'TNC'``, ``'SLSQP'``, ``'trust-constr'``.
    refine : bool
        If True and ``method`` is global, run a short L-BFGS-B from the
        global optimum to refine within the basin.
    baseline_first : bool
        If True (default), first optimize sampled ``baseline_*_flux_*``
        parameters using only out-of-transit photometry. This stage is skipped
        when no such free parameters exist, when ``mask_transit=True`` already,
        or when resuming a saved CMA-ES trajectory.
    baseline_mask_width : float
        Full transit-mask width in days for the baseline-only stage. ``None``
        reuses ``fast_fit_width`` from settings.csv. A value around 1.2--1.5
        times the expected T14 leaves a useful safety margin.
    n_restarts : int
        Number of optimizer runs (1 = single run from theta_0; >1 adds
        Latin-hypercube draws inside ``bounds``). Best of N is kept.
    sigma0 : float
        CMA-ES initial isotropic step. Defaults to 10 % of the geometric
        mean prior width.
    maxfevals : int
        Per-restart budget. ``None`` lets the chosen optimizer pick its
        default.
    seed : int
        Reproducibility seed for multistart draws + stochastic methods.
    save : bool
        Persist a JSON summary to ``<datadir>/results/optimize_save.json``.
    mutate_basement : bool
        If accepted, overwrite ``config.BASEMENT.theta_0`` in place *and*
        rewrite the ``value`` column of the fitted rows in ``params.csv`` on
        disk, so the next ``mcmc_fit`` / ``ns_fit`` / ``show_initial_guess``
        call — in this process or a fresh one — warm-starts from the optimum.
    improvement_threshold : float
        Required lnprob improvement before the result is accepted. Default
        ``0.5 * ndim`` (loose AIC-style margin).
    consistency_threshold : float
        When ``n_restarts > 1``, max allowed spread of restart lnprobs
        before the run is flagged as multimodal-suspect.
    workers : int
        Parallel-process workers (currently honoured by
        ``differential_evolution`` only).
    quiet : bool
        Suppress the console echo of the progress / summary lines. The
        run logfile (``<outdir>/logfile_<now>.log``) is still written.
    verbose : bool
        CMA-ES only. Stream the ``cma`` library's own per-generation
        convergence table to the console (off by default — the
        start/per-restart/summary ``logprint`` lines give progress
        without the full table). Independent of ``quiet``.
    resume : bool
        CMA-ES only. When True, load the pickled strategy state from
        ``<outdir>/optimize_cma_state.pkl`` (saved by a previous call) and
        continue evolution with its adapted covariance / step size intact.
        Falls back to a fresh start with a warning if the pickle is
        missing; raises ``ValueError`` if it's incompatible with the
        current ``ndim``/``bounds``. Requires ``n_restarts=1`` (resume of
        multiple restarts is ambiguous). Independently of this flag,
        every CMA-ES call writes the final state to the same path so a
        future ``resume=True`` is always possible.
    """
    if _extra_kwargs:
        import difflib as _difflib

        _valid_keys = sorted(
            [
                "datadir",
                "method",
                "refine",
                "baseline_first",
                "baseline_mask_width",
                "n_restarts",
                "sigma0",
                "maxfevals",
                "seed",
                "save",
                "mutate_basement",
                "improvement_threshold",
                "consistency_threshold",
                "skip_bounds_check",
                "workers",
                "quiet",
                "verbose",
                "resume",
            ]
        )
        _bits = []
        for _bad in sorted(_extra_kwargs):
            _close = _difflib.get_close_matches(_bad, _valid_keys, n=2, cutoff=0.6)
            _bits.append(
                "{!r} (did you mean {}?)".format(_bad, " or ".join(repr(c) for c in _close))
                if _close
                else f"{_bad!r}"
            )
        raise TypeError(
            "optimize() got unexpected keyword argument(s): "
            + ", ".join(_bits)
            + f".  Valid keyword arguments: {_valid_keys}."
        )

    # Bootstrap BASEMENT if the caller hasn't already done so.
    if getattr(config, "BASEMENT", None) is None or os.path.abspath(
        config.BASEMENT.datadir
    ) != os.path.abspath(datadir):
        config.init(datadir)
    b = config.BASEMENT

    baseline_result = None
    baseline_keys = [str(k) for k in b.fitkeys if _is_photometric_baseline_key(k)]
    if (
        baseline_first
        and baseline_keys
        and not b.settings.get("mask_transit", False)
        and not resume
    ):
        logprint(
            "optimize[baseline-first]  masking transits and fitting {} baseline parameter(s): {}".format(
                len(baseline_keys), ", ".join(baseline_keys)
            ),
            quiet=quiet,
        )
        baseline_result = _baseline_warm_start(
            datadir,
            b,
            method=method,
            refine=refine,
            n_restarts=n_restarts,
            sigma0=sigma0,
            maxfevals=maxfevals,
            seed=seed,
            consistency_threshold=consistency_threshold,
            skip_bounds_check=skip_bounds_check,
            workers=workers,
            quiet=quiet,
            verbose=verbose,
            mask_width=baseline_mask_width,
        )
        b = config.BASEMENT
        state = (
            "accepted"
            if baseline_result.accepted
            else ("rejected: " + baseline_result.reject_reason)
        )
        logprint(
            f"optimize[baseline-first]  stage {state}; restoring all transit points for the joint fit",
            quiet=quiet,
        )
    elif baseline_first and baseline_keys and b.settings.get("mask_transit", False):
        logprint(
            "optimize[baseline-first]  skipped because settings.csv already has mask_transit=True",
            quiet=quiet,
        )
    elif baseline_first and baseline_keys and resume:
        logprint(
            "optimize[baseline-first]  skipped because resume=True continues the saved CMA-ES state",
            quiet=quiet,
        )

    theta_0 = np.array(b.theta_0, dtype=float)
    bounds = _extract_bounds(b)
    if improvement_threshold is None:
        improvement_threshold = 0.5 * b.ndim

    # Compute pickle path early so we can validate on resume and write on save.
    outdir = getattr(b, "outdir", os.path.join(datadir, "results"))
    cma_state_path = os.path.join(outdir, _CMA_STATE_FILENAME)

    if resume:
        if method != "cmaes":
            raise ValueError(
                "optimize: resume=True is only meaningful for method='cmaes' "
                "(scipy minimize / DE / dual_annealing don't expose resumable "
                f"state). Got method={method!r}."
            )
        if n_restarts != 1:
            raise ValueError(
                f"optimize: resume=True is incompatible with n_restarts={n_restarts} "
                "(resume continues one trajectory; restarts mean N "
                "trajectories). Use n_restarts=1 or resume=False."
            )
        if not os.path.exists(cma_state_path):
            warnings.warn(
                f"optimize: resume=True but '{cma_state_path}' does not exist; falling "
                "back to a fresh CMA-ES start.",
                stacklevel=2,
            )
        else:
            # Validate metadata up-front so we fail before burning CPU.
            _load_cma_state(cma_state_path, ndim=b.ndim, bounds=bounds)

    lnp_0 = float(mcmc_lnprob(theta_0))
    if not np.isfinite(lnp_0):
        raise ValueError(
            "optimize: initial theta_0 has non-finite lnprob "
            f"({lnp_0}); fix params.csv before optimizing."
        )

    rng = np.random.default_rng(seed)

    # Build the multistart x0 list — first restart always uses theta_0.
    x0_list = [theta_0]
    for _ in range(max(0, n_restarts - 1)):
        u = rng.random(b.ndim)
        x0 = np.array([lo + u_i * (hi - lo) for u_i, (lo, hi) in zip(u, bounds)])
        x0_list.append(x0)

    if sigma0 is None and method == "cmaes":
        sigma0 = _default_sigma0(theta_0, bounds, b)

    # Dispatch + per-restart accumulation.
    t0 = _now()
    logprint(
        "optimize[{}]  starting: ndim={}  n_restarts={}  maxfevals={}  "
        "lnprob_initial={:.2f}".format(
            method, b.ndim, n_restarts, maxfevals if maxfevals is not None else "default", lnp_0
        ),
        quiet=quiet,
    )
    restart_results = []
    total_nfev = 0
    any_success = False
    resumed_from_pickle = False
    for r, x0 in enumerate(x0_list):
        if method in _LOCAL_METHODS:
            x_r, nfev_r, ok_r = _run_local(method, x0, bounds, maxiter=maxfevals)
        elif method == "differential_evolution":
            x_r, nfev_r, ok_r = _run_differential_evolution(
                x0, bounds, maxiter=maxfevals, seed=seed + r, workers=workers
            )
        elif method == "dual_annealing":
            x_r, nfev_r, ok_r = _run_dual_annealing(x0, bounds, maxiter=maxfevals, seed=seed + r)
        elif method == "cmaes":
            # Resume only for the first restart (and only if requested);
            # always overwrite the saved state with the final strategy.
            _resume_path = (
                cma_state_path if (resume and r == 0 and os.path.exists(cma_state_path)) else None
            )
            x_r, nfev_r, ok_r, _resumed = _run_cmaes(
                x0,
                bounds,
                sigma0=sigma0,
                maxfevals=maxfevals,
                seed=seed + r,
                verbose=verbose,
                resume_path=_resume_path,
                save_path=cma_state_path,
            )
            resumed_from_pickle = resumed_from_pickle or _resumed
        else:
            raise ValueError(
                f"optimize: unknown method '{method}'. Choose from {_LOCAL_METHODS} or {_GLOBAL_METHODS}."
            )
        any_success = any_success or ok_r

        if refine and method in _GLOBAL_METHODS:
            x_r, nfev_refine = _refine(x_r, bounds)
            nfev_r += nfev_refine

        # Clip into bounds to guard against tiny numerical overshoot.
        x_r = _clip_to_bounds(x_r, bounds)
        lnp_r = float(mcmc_lnprob(x_r))
        restart_results.append((x_r, lnp_r))
        total_nfev += nfev_r
        logprint(
            f"optimize[{method}]  restart {r + 1}/{len(x0_list)}: lnprob={lnp_r:.2f}  nfev={nfev_r}  "
            f"ok={ok_r}",
            quiet=quiet,
        )

    # Best across restarts.
    restart_lnprobs = [lp for _, lp in restart_results]
    best_idx = int(np.argmax(restart_lnprobs))
    theta_best, lnp_best = restart_results[best_idx]
    delta = lnp_best - lnp_0
    wallclock = _now() - t0

    # Acceptance gates.
    reject = ""
    if not np.isfinite(lnp_best):
        reject = "lnprob_opt is non-finite"
    elif delta < improvement_threshold:
        reject = f"delta_lnprob={delta:.2f} < improvement_threshold={improvement_threshold:.2f}"
    elif (not skip_bounds_check) and _on_bounds(theta_best, bounds):
        reject = (
            "theta_opt sits on a prior bound — pass "
            "skip_bounds_check=True if this is physically expected"
        )
    elif n_restarts > 1:
        finite_lps = [lp for lp in restart_lnprobs if np.isfinite(lp)]
        if len(finite_lps) >= 2:
            spread = max(finite_lps) - min(finite_lps)
            if spread > consistency_threshold:
                reject = f"multistart spread {spread:.2f} > consistency_threshold {consistency_threshold:.2f}"

    accepted = (reject == "") and any_success

    if accepted and mutate_basement:
        theta_params = _theta_in_params_frame(b, theta_best)
        b.theta_0 = np.asarray(theta_best, dtype=float)
        _write_theta_to_params_csv(datadir, b.fitkeys, theta_params)

    result = OptimizeResult(
        method=method,
        accepted=accepted,
        success=any_success,
        lnprob_initial=lnp_0,
        lnprob_opt=float(lnp_best),
        delta_lnprob=float(delta),
        theta_initial=theta_0.tolist(),
        theta_opt=np.asarray(theta_best, dtype=float).tolist(),
        fitkeys=list(b.fitkeys),
        n_restarts=int(n_restarts),
        restart_lnprobs=[float(lp) for lp in restart_lnprobs],
        nfev=int(total_nfev),
        wallclock_s=float(wallclock),
        reject_reason=reject,
        refine=bool(refine and method in _GLOBAL_METHODS),
        bounds=[list(bb) for bb in bounds],
        resumed_from_pickle=bool(resumed_from_pickle),
        baseline_first=bool(baseline_result is not None),
        baseline_stage_accepted=bool(baseline_result is not None and baseline_result.accepted),
        baseline_stage_fitkeys=(list(baseline_result.fitkeys) if baseline_result else []),
        baseline_stage_nfev=(int(baseline_result.nfev) if baseline_result else 0),
        baseline_stage_wallclock_s=(float(baseline_result.wallclock_s) if baseline_result else 0.0),
    )

    if save:
        os.makedirs(outdir, exist_ok=True)
        with open(os.path.join(outdir, "optimize_save.json"), "w") as f:
            json.dump(asdict(result), f, indent=2)

    status = "accepted" if accepted else f"rejected ({reject})"
    logprint(
        f"optimize[{method}]  lnprob: {lnp_0:.2f} -> {lnp_best:.2f}  (Δ={delta:+.2f})  "
        f"nfev={total_nfev}  {wallclock:.1f}s  [{status}]",
        quiet=quiet,
    )

    return result


import types


class _CallableModule(types.ModuleType):
    def __init__(self, module, func):
        super().__init__(module.__name__)
        self.__dict__.update(module.__dict__)
        self._func = func

    def __call__(self, *args, **kwargs):
        return self._func(*args, **kwargs)


sys.modules[__name__] = _CallableModule(sys.modules[__name__], optimize)
