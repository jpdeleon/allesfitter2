"""UltraNest backend for allesfitter nested sampling.

Wraps :class:`ultranest.ReactiveNestedSampler` behind the unified backend
interface. ``ultranest`` is an optional dependency — install with the
``[ultranest]`` extra (``pip install allesfitter[ultranest]``).

Multiprocessing
---------------
UltraNest parallelises via MPI (``mpi4py``), not via ``multiprocessing``.
When ``settings['multiprocess']`` is True we simply log a notice and run
single-process; we do **not** try to wrap the sampler in a ``Pool``
(its sampler object is not safe to pickle the way dynesty's is).
"""

from __future__ import annotations

import os
from time import time as timer
from typing import Callable

import numpy as np

from . import build_results


name = "ultranest"


_NONFINITE_LOGL_FLOOR = -1e300


def _safe_loglike(scalar_loglike: Callable):
    """Wrap allesfitter's scalar ``ns_lnlike`` to satisfy UltraNest.

    UltraNest aborts when the log-likelihood returns ``-inf`` or ``NaN``
    (its strict check in ``evaluate_likelihood``). allesfitter's
    ``calculate_lnlike_total`` legitimately produces those values for
    unphysical configurations (cos i > 1, negative semi-major axes,
    eccentricities outside [0, 1), etc.) — dynesty tolerates them, but
    UltraNest does not.

    Replace any non-finite return with ``-1e300`` so:
      * the sampler treats the point as overwhelmingly improbable
        (the intended semantics of ``-inf``),
      * the value remains finite so logaddexp / logsumexp stay defined,
      * the floor is far enough below any plausible posterior that it
        has no effect on the inferred distribution.
    """
    floor = _NONFINITE_LOGL_FLOOR

    def wrapped(theta):
        v = scalar_loglike(theta)
        if not np.isfinite(v):
            return floor
        return float(v)
    return wrapped


def _vectorised_loglike(scalar_loglike: Callable):
    """Backwards-compatible alias for the previous helper name."""
    return _safe_loglike(scalar_loglike)


def run(
    loglike: Callable,
    prior_transform: Callable,
    ndim: int,
    settings: dict,
    param_names,
    outdir: str,
    logprint=print,
):
    """Run UltraNest and return an ``NSResults`` in the unified schema."""
    try:
        from ultranest import ReactiveNestedSampler
    except ImportError as exc:
        raise ImportError(
            "UltraNest is not installed. Install with: pip install 'allesfitter[ultranest]' "
            "or pip install ultranest. See https://github.com/JohannesBuchner/UltraNest."
        ) from exc

    nlive = int(settings.get("ns_nlive", 500))
    # UltraNest's dlogz default is 0.5; dynesty's static-mode ns_tol can be
    # huge (e.g. 100 = effectively no stopping). Such values are nonsensical
    # for UltraNest's reactive scheme — warn and clamp to a sane value.
    dlogz = float(settings.get("ns_tol", 0.5))
    if dlogz > 5.0:
        logprint(
            "\n! ns_tol={} is dynesty-scale; clamping to dlogz=0.5 for ultranest "
            "(typical range 0.1–1.0). Set ns_tol explicitly in settings.csv to silence.".format(dlogz)
        )
        dlogz = 0.5
    min_ess = int(settings.get("un_min_ess", 400))
    max_iters = settings.get("un_max_iters")
    if max_iters is not None:
        max_iters = int(max_iters)
    print_progress = bool(settings.get("print_progress", True))

    if settings.get("multiprocess", False):
        logprint(
            "\n! Note: UltraNest uses MPI for parallelism, not multiprocessing.Pool. "
            "Running single-process. To parallelise, launch with `mpiexec -n N python run.py`."
        )

    # Convert param_names to plain Python strings — ultranest stores them as paramnames
    # and uses them for plot labels / output files. numpy strings can confuse it.
    pnames = [str(p) for p in param_names]

    # Detect MPI rank when running under mpiexec / mpirun / srun.
    #
    # Priority: mpi4py (preferred — gives a real communicator) > environment
    # variables set by the launcher (works even when mpi4py is not
    # installed). Without env-var fallback, each rank's mpi4py import would
    # silently fail and every rank would think it was rank 0 — they would
    # then race to take an exclusive lock on the same HDF5 point store
    # and all but one would crash with BlockingIOError.
    _mpi_size, _mpi_rank = 1, 0
    try:
        from mpi4py import MPI
        _mpi_size = MPI.COMM_WORLD.Get_size()
        _mpi_rank = MPI.COMM_WORLD.Get_rank()
    except Exception:
        # Common launcher env-vars: OpenMPI, MPICH/Slurm PMI, PMIx.
        for size_var, rank_var in (
            ("OMPI_COMM_WORLD_SIZE", "OMPI_COMM_WORLD_RANK"),
            ("PMI_SIZE", "PMI_RANK"),
            ("PMIX_SIZE", "PMIX_RANK"),
            ("SLURM_NTASKS", "SLURM_PROCID"),
        ):
            if size_var in os.environ and rank_var in os.environ:
                try:
                    _mpi_size = int(os.environ[size_var])
                    _mpi_rank = int(os.environ[rank_var])
                    break
                except ValueError:
                    pass
        if _mpi_size > 1:
            logprint(
                "\n! Running under mpiexec but mpi4py is NOT installed. "
                "UltraNest cannot parallelise likelihood evaluations — each "
                "rank will run an independent copy. Install mpi4py with "
                "`pip install mpi4py` to get true MPI parallelism, "
                "or launch without mpiexec for a single-process run."
            )

    # UltraNest's log_dir backend uses h5py. If h5py is missing — or we are
    # on a non-zero MPI rank — run with log_dir=None.
    if _mpi_size > 1 and _mpi_rank > 0:
        log_dir = None
        sampler_kwargs = dict(log_dir=None)
        logprint(
            "\n! MPI rank {}/{}: log_dir disabled (only rank 0 writes the HDF5 store).".format(
                _mpi_rank, _mpi_size
            )
        )
    else:
        try:
            import h5py  # noqa: F401
            log_dir = os.path.join(outdir, "ultranest_logs")
            os.makedirs(log_dir, exist_ok=True)
            sampler_kwargs = dict(log_dir=log_dir, resume="resume")
        except ImportError:
            log_dir = None
            sampler_kwargs = dict(log_dir=None)
            logprint("\n! h5py not installed — running ultranest without log_dir/resume.")

    logprint("\nRunning Reactive Nested Sampler (ultranest)...")
    logprint("--------------------------")
    # `outdir` is <datadir>/results; print its parent so log lines from
    # multiple concurrent fits are unambiguous when grepped or tailed.
    logprint("datadir: {}".format(os.path.abspath(os.path.dirname(outdir))))
    logprint("  log_dir   = {}".format(log_dir))
    logprint("  min_nlive = {}, dlogz = {}, min_ess = {}".format(nlive, dlogz, min_ess))

    # Resilient resume: if a prior aborted run left a truncated/corrupt HDF5
    # store, ultranest's resume='resume' path raises OSError when it tries to
    # open the file. Detect that, rename the broken store, and retry with a
    # fresh log_dir so the user does not have to manually wipe files.
    def _make_sampler(kwargs):
        return ReactiveNestedSampler(
            pnames,
            _vectorised_loglike(loglike),
            transform=prior_transform,
            **kwargs,
        )

    try:
        sampler = _make_sampler(sampler_kwargs)
    except OSError as exc:
        if log_dir is None:
            raise
        import shutil
        import datetime as _dt
        ts = _dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        broken = log_dir + ".broken." + ts
        logprint(
            "\n! UltraNest could not resume from {} ({}). "
            "Renaming to {} and starting fresh.".format(log_dir, exc, broken)
        )
        shutil.move(log_dir, broken)
        os.makedirs(log_dir, exist_ok=True)
        sampler_kwargs["resume"] = "overwrite"
        sampler = _make_sampler(sampler_kwargs)

    t0 = timer()
    run_kwargs = dict(
        min_num_live_points=nlive,
        dlogz=dlogz,
        min_ess=min_ess,
        show_status=print_progress,
    )
    if max_iters is not None:
        run_kwargs["max_iters"] = max_iters
    result = sampler.run(**run_kwargs)
    elapsed = timer() - t0

    logprint("\nTime taken to run 'ultranest': {:.2f} hours".format(elapsed / 3600.0))

    # Map UltraNest result → unified schema.
    #
    # UltraNest's ``weighted_samples`` exposes ``points`` (the raw samples)
    # and ``weights`` (already-normalised importance weights summing to 1).
    # Its ``logw`` key is the log of the per-bin prior-volume increment
    # (log dX) — NOT the importance log-weight — so it is *not* the
    # dynesty-equivalent ``logwt``.
    #
    # Downstream callers (``draw_ns_posterior_samples`` and
    # ``_corner_from_results``) reconstruct importance weights via
    # ``w = exp(logwt - logz_final)``. To make that identity recover
    # UltraNest's normalised weights, we synthesise
    # ``logwt = log(weights) + logz_final`` — algebraically equivalent and
    # backend-portable.
    ws = result["weighted_samples"]
    samples_raw = np.asarray(ws["points"])
    weights_norm = np.asarray(ws["weights"], dtype=float)
    weights_norm = weights_norm / weights_norm.sum()
    logz_final = float(result["logz"])
    logzerr_final = float(result["logzerr"])
    # Avoid -inf when a weight is exactly zero (rare but possible).
    logwt = np.log(np.maximum(weights_norm, 1e-300)) + logz_final

    return build_results(
        backend="ultranest",
        samples=samples_raw,
        logwt=logwt,
        # Store as length-1 arrays so [-1] indexing matches the dynesty path
        # (dynesty exposes cumulative arrays). The final value is what callers care about.
        logz=np.array([logz_final]),
        logzerr=np.array([logzerr_final]),
        fitkeys=pnames,
        wall_time_sec=elapsed,
        raw=None,
    )
