"""Global + local optimization of the joint posterior, intended as a
warm-start for mcmc_fit / ns_fit.

Default stack: CMA-ES global search → L-BFGS-B refine → acceptance gates →
optional in-place update of ``config.BASEMENT.theta_0`` so that the next
sampler call starts from the optimized point.

Public entry point:

    >>> import allesfitter
    >>> res = allesfitter.optimize('.', method='cmaes', refine=True)
    >>> if res.accepted:
    ...     allesfitter.mcmc_fit('.')   # warm-started

The acceptance gates (improvement, bounds, multistart consistency) make
``optimize()`` safe to call unconditionally — if the optimizer fails to
beat the user's hand-tuned ``theta_0``, BASEMENT is left untouched.
"""

from __future__ import print_function, division, absolute_import

import json
import os
import pickle
import sys
import warnings
from dataclasses import dataclass, field, asdict
from time import time as _now
from typing import Any

import numpy as np
from scipy import optimize as scipy_opt

# Filename for the pickled CMA-ES strategy state used by resume=True.
_CMA_STATE_FILENAME = 'optimize_cma_state.pkl'

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
    b = getattr(config, 'BASEMENT', None)
    if b is None:
        return
    original = sys.stdout
    try:
        logpath = os.path.join(b.outdir, 'logfile_' + b.now + '.log')
        with open(logpath, 'a') as f:
            sys.stdout = f
            print(*text)
    except OSError:
        # Some Windows versions choke on the open()/os.path.join() combo;
        # also covers a missing/unwritable outdir.
        pass
    finally:
        sys.stdout = original


# Methods that scipy.optimize.minimize accepts a `bounds` keyword for.
_LOCAL_METHODS = ('L-BFGS-B', 'Powell', 'TNC', 'SLSQP', 'trust-constr')
# Global methods we wrap explicitly.
_GLOBAL_METHODS = ('cmaes', 'dual_annealing', 'differential_evolution')


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


def _extract_bounds(basement, normal_sigma_clip: float = 5.0):
    """Return per-fit-parameter ``[lo, hi]`` for optimizer ``bounds`` kwargs.

    Uniform / trunc_normal priors give their explicit limits. Normal priors
    fall back to ``mu ± normal_sigma_clip * sigma`` since the optimizers
    we use require finite bounds.
    """
    bounds = []
    for b in basement.bounds:
        kind = b[0]
        if kind == 'uniform':
            bounds.append((float(b[1]), float(b[2])))
        elif kind == 'trunc_normal':
            bounds.append((float(b[3]), float(b[4])))
        elif kind == 'normal':
            mu, sigma = float(b[1]), float(b[2])
            bounds.append((mu - normal_sigma_clip * sigma,
                           mu + normal_sigma_clip * sigma))
        else:
            raise ValueError(
                "optimize: unsupported prior kind '{}' for fitkey".format(kind))
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
        opts.setdefault('maxiter', int(maxiter))
    res = scipy_opt.minimize(_objective, x0, method=method,
                             bounds=bounds, options=opts)
    return np.asarray(res.x, dtype=float), int(res.nfev), bool(res.success)


def _run_differential_evolution(x0, bounds, maxiter=None, seed=None, workers=1):
    res = scipy_opt.differential_evolution(
        _objective, bounds=bounds, init='sobol', x0=x0,
        maxiter=int(maxiter) if maxiter else 100,
        seed=seed, polish=False, workers=workers, updating='deferred',
    )
    return np.asarray(res.x, dtype=float), int(res.nfev), bool(res.success)


def _run_dual_annealing(x0, bounds, maxiter=None, seed=None):
    res = scipy_opt.dual_annealing(
        _objective, bounds=bounds, x0=x0,
        maxiter=int(maxiter) if maxiter else 1000,
        seed=seed, no_local_search=False,
    )
    return np.asarray(res.x, dtype=float), int(res.nfev), bool(res.success)


def _load_cma_state(path, ndim, bounds):
    """Load a pickled (strategy, metadata) tuple from `path` and validate
    it matches the current problem dimensions. Returns the strategy or
    raises ValueError on mismatch.
    """
    with open(path, 'rb') as f:
        es, meta = pickle.load(f)
    if int(meta.get('ndim', -1)) != int(ndim):
        raise ValueError(
            "optimize: resume=True but saved CMA-ES state at '{p}' was for "
            "ndim={saved}, current fit has ndim={now}. Delete the file or "
            "set resume=False.".format(p=path, saved=meta.get('ndim'), now=ndim)
        )
    saved_bounds = meta.get('bounds')
    if saved_bounds is not None:
        # Tolerate tiny float-repr drift but flag real changes.
        try:
            sb = np.asarray(saved_bounds, dtype=float)
            cb = np.asarray(bounds, dtype=float)
            if sb.shape != cb.shape or not np.allclose(sb, cb, atol=1e-12):
                raise ValueError(
                    "optimize: resume=True but bounds in saved state differ "
                    "from current bounds. Delete '{p}' or set resume=False.".format(p=path)
                )
        except ValueError:
            raise
        except Exception:
            pass
    return es


def _save_cma_state(path, es, ndim, bounds):
    """Pickle (strategy, metadata) to `path` for a future resume=True."""
    meta = {'ndim': int(ndim), 'bounds': [list(b) for b in bounds]}
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump((es, meta), f, protocol=pickle.HIGHEST_PROTOCOL)


def _run_cmaes(x0, bounds, sigma0, maxfevals=None, seed=None, verbose=False,
               resume_path=None, save_path=None):
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
        with open(resume_path, 'rb') as f:
            es, _meta = pickle.load(f)
        resumed = True
        # If the user passed a maxfevals budget for THIS call, extend the
        # strategy's option so es.optimize() runs that many more evals.
        if maxfevals is not None:
            try:
                es.opts.set({'maxfevals': int(es.countevals) + int(maxfevals)})
            except Exception:
                pass

    if es is None:
        lo = [b[0] for b in bounds]
        hi = [b[1] for b in bounds]
        opts = {
            'bounds': [lo, hi],
            'verbose': 0 if not verbose else 1,
            'seed': seed if seed is not None else np.random.randint(1, 2**31 - 1),
        }
        if maxfevals is not None:
            opts['maxfevals'] = int(maxfevals)
        es = cma.CMAEvolutionStrategy(list(x0), float(sigma0), opts)

    es.optimize(_objective)

    if save_path:
        _save_cma_state(save_path, es, ndim=len(bounds), bounds=bounds)

    return (np.asarray(es.result.xbest, dtype=float),
            int(es.result.evaluations),
            bool(es.result.xbest is not None),
            resumed)


def _refine(theta, bounds, maxiter=30):
    """L-BFGS-B finite-diff refinement within `maxiter` steps. Catches and
    returns the input theta unchanged if the refinement fails."""
    try:
        x, nfev, _ = _run_local('L-BFGS-B', theta, bounds, maxiter=maxiter)
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
    method: str = 'cmaes',
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
    **_extra_kwargs,
) -> OptimizeResult:
    """Find a posterior maximum and (optionally) push it into BASEMENT.theta_0.

    Parameters
    ----------
    datadir : str
        Path passed to ``config.init(datadir)`` if BASEMENT is not already
        populated.
    method : str
        One of ``'cmaes'`` (default, requires ``pip install cma``),
        ``'dual_annealing'``, ``'differential_evolution'``, ``'L-BFGS-B'``,
        ``'Powell'``, ``'TNC'``, ``'SLSQP'``, ``'trust-constr'``.
    refine : bool
        If True and ``method`` is global, run a short L-BFGS-B from the
        global optimum to refine within the basin.
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
        If accepted, overwrite ``config.BASEMENT.theta_0`` in place so the
        next ``mcmc_fit`` / ``ns_fit`` warm-starts from the optimum.
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
        _valid_keys = sorted([
            'datadir', 'method', 'refine', 'n_restarts', 'sigma0', 'maxfevals',
            'seed', 'save', 'mutate_basement', 'improvement_threshold',
            'consistency_threshold', 'skip_bounds_check', 'workers', 'quiet',
            'verbose', 'resume',
        ])
        _bits = []
        for _bad in sorted(_extra_kwargs):
            _close = _difflib.get_close_matches(
                _bad, _valid_keys, n=2, cutoff=0.6)
            _bits.append(
                "{!r} (did you mean {}?)".format(
                    _bad, ' or '.join(repr(c) for c in _close))
                if _close else "{!r}".format(_bad)
            )
        raise TypeError(
            "optimize() got unexpected keyword argument(s): "
            + ", ".join(_bits)
            + ".  Valid keyword arguments: {}.".format(_valid_keys)
        )

    # Bootstrap BASEMENT if the caller hasn't already done so.
    if getattr(config, 'BASEMENT', None) is None \
            or os.path.abspath(config.BASEMENT.datadir) != os.path.abspath(datadir):
        config.init(datadir)
    b = config.BASEMENT
    theta_0 = np.array(b.theta_0, dtype=float)
    bounds = _extract_bounds(b)
    if improvement_threshold is None:
        improvement_threshold = 0.5 * b.ndim

    # Compute pickle path early so we can validate on resume and write on save.
    outdir = getattr(b, 'outdir', os.path.join(datadir, 'results'))
    cma_state_path = os.path.join(outdir, _CMA_STATE_FILENAME)

    if resume:
        if method != 'cmaes':
            raise ValueError(
                "optimize: resume=True is only meaningful for method='cmaes' "
                "(scipy minimize / DE / dual_annealing don't expose resumable "
                "state). Got method={!r}.".format(method))
        if n_restarts != 1:
            raise ValueError(
                "optimize: resume=True is incompatible with n_restarts={n} "
                "(resume continues one trajectory; restarts mean N "
                "trajectories). Use n_restarts=1 or resume=False.".format(
                    n=n_restarts))
        if not os.path.exists(cma_state_path):
            warnings.warn(
                "optimize: resume=True but '{p}' does not exist; falling "
                "back to a fresh CMA-ES start.".format(p=cma_state_path))
        else:
            # Validate metadata up-front so we fail before burning CPU.
            _load_cma_state(cma_state_path, ndim=b.ndim, bounds=bounds)

    lnp_0 = float(mcmc_lnprob(theta_0))
    if not np.isfinite(lnp_0):
        raise ValueError(
            "optimize: initial theta_0 has non-finite lnprob "
            "({}); fix params.csv before optimizing.".format(lnp_0))

    rng = np.random.default_rng(seed)

    # Build the multistart x0 list — first restart always uses theta_0.
    x0_list = [theta_0]
    for _ in range(max(0, n_restarts - 1)):
        u = rng.random(b.ndim)
        x0 = np.array([lo + u_i * (hi - lo)
                       for u_i, (lo, hi) in zip(u, bounds)])
        x0_list.append(x0)

    if sigma0 is None and method == 'cmaes':
        sigma0 = _default_sigma0(theta_0, bounds, b)

    # Dispatch + per-restart accumulation.
    t0 = _now()
    logprint(
        "optimize[{}]  starting: ndim={}  n_restarts={}  maxfevals={}  "
        "lnprob_initial={:.2f}".format(
            method, b.ndim, n_restarts,
            maxfevals if maxfevals is not None else 'default', lnp_0),
        quiet=quiet)
    restart_results = []
    total_nfev = 0
    any_success = False
    resumed_from_pickle = False
    for r, x0 in enumerate(x0_list):
        if method in _LOCAL_METHODS:
            x_r, nfev_r, ok_r = _run_local(method, x0, bounds, maxiter=maxfevals)
        elif method == 'differential_evolution':
            x_r, nfev_r, ok_r = _run_differential_evolution(
                x0, bounds, maxiter=maxfevals, seed=seed + r, workers=workers)
        elif method == 'dual_annealing':
            x_r, nfev_r, ok_r = _run_dual_annealing(
                x0, bounds, maxiter=maxfevals, seed=seed + r)
        elif method == 'cmaes':
            # Resume only for the first restart (and only if requested);
            # always overwrite the saved state with the final strategy.
            _resume_path = (cma_state_path
                            if (resume and r == 0
                                and os.path.exists(cma_state_path))
                            else None)
            x_r, nfev_r, ok_r, _resumed = _run_cmaes(
                x0, bounds, sigma0=sigma0, maxfevals=maxfevals,
                seed=seed + r, verbose=verbose, resume_path=_resume_path,
                save_path=cma_state_path)
            resumed_from_pickle = resumed_from_pickle or _resumed
        else:
            raise ValueError(
                "optimize: unknown method '{}'. Choose from {} or {}.".format(
                    method, _LOCAL_METHODS, _GLOBAL_METHODS))
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
            "optimize[{}]  restart {}/{}: lnprob={:.2f}  nfev={}  "
            "ok={}".format(method, r + 1, len(x0_list), lnp_r, nfev_r, ok_r),
            quiet=quiet)

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
        reject = (
            "delta_lnprob={:.2f} < improvement_threshold={:.2f}"
            .format(delta, improvement_threshold))
    elif (not skip_bounds_check) and _on_bounds(theta_best, bounds):
        reject = ("theta_opt sits on a prior bound — pass "
                  "skip_bounds_check=True if this is physically expected")
    elif n_restarts > 1:
        finite_lps = [lp for lp in restart_lnprobs if np.isfinite(lp)]
        if len(finite_lps) >= 2:
            spread = max(finite_lps) - min(finite_lps)
            if spread > consistency_threshold:
                reject = ("multistart spread {:.2f} > consistency_threshold {:.2f}"
                          .format(spread, consistency_threshold))

    accepted = (reject == "") and any_success

    if accepted and mutate_basement:
        b.theta_0 = np.asarray(theta_best, dtype=float)

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
    )

    if save:
        os.makedirs(outdir, exist_ok=True)
        with open(os.path.join(outdir, 'optimize_save.json'), 'w') as f:
            json.dump(asdict(result), f, indent=2)

    status = "accepted" if accepted else "rejected ({})".format(reject)
    logprint(
        "optimize[{}]  lnprob: {:.2f} -> {:.2f}  (Δ={:+.2f})  "
        "nfev={}  {:.1f}s  [{}]".format(
            method, lnp_0, lnp_best, delta,
            total_nfev, wallclock, status),
        quiet=quiet)

    return result
