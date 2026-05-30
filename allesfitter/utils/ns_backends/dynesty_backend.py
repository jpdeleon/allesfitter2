"""Dynesty backend for allesfitter nested sampling.

Wraps ``dynesty.NestedSampler`` and ``dynesty.DynamicNestedSampler``
behind the unified backend interface. Behavior matches the original
in-line implementation that lived in ``nested_sampling.py`` so existing
users see no change when ``backend='dynesty'`` (the default).
"""

from __future__ import annotations

import os
from contextlib import closing
from multiprocessing import Pool
from time import time as timer
from typing import Callable

import numpy as np
import dynesty

from . import build_results


name = "dynesty"


def run(
    loglike: Callable,
    prior_transform: Callable,
    ndim: int,
    settings: dict,
    param_names,
    outdir: str,
    logprint=print,
):
    """Run dynesty and return an ``NSResults`` in the unified schema."""
    nlive = int(settings["ns_nlive"])
    bound = settings["ns_bound"]
    sample = settings["ns_sample"]
    tol = float(settings["ns_tol"])
    modus = settings.get("ns_modus", "static")
    use_mp = bool(settings.get("multiprocess", False))
    cores = int(settings.get("multiprocess_cores", 1)) if use_mp else 1
    print_progress = bool(settings.get("print_progress", True))

    # `outdir` is <datadir>/results; print its parent so log lines from
    # multiple concurrent fits are unambiguous when grepped or tailed.
    _datadir = os.path.abspath(os.path.dirname(outdir))

    t0 = timer()
    if modus == "static":
        logprint("\nRunning Static Nested Sampler (dynesty)...")
        logprint("--------------------------")
        logprint("datadir: {}".format(_datadir))
        if use_mp:
            with closing(Pool(processes=cores)) as pool:
                logprint("\nRunning on {} CPUs.".format(cores))
                sampler = dynesty.NestedSampler(
                    loglike, prior_transform, ndim,
                    pool=pool, queue_size=cores,
                    bound=bound, sample=sample, nlive=nlive,
                )
                sampler.run_nested(dlogz=tol, print_progress=print_progress)
        else:
            sampler = dynesty.NestedSampler(
                loglike, prior_transform, ndim,
                bound=bound, sample=sample, nlive=nlive,
            )
            sampler.run_nested(dlogz=tol, print_progress=print_progress)

    elif modus == "dynamic":
        logprint("\nRunning Dynamic Nested Sampler (dynesty)...")
        logprint("--------------------------")
        logprint("datadir: {}".format(_datadir))
        if use_mp:
            with closing(Pool(processes=cores)) as pool:
                logprint("\nRunning on {} CPUs.".format(cores))
                sampler = dynesty.DynamicNestedSampler(
                    loglike, prior_transform, ndim,
                    pool=pool, queue_size=cores,
                    bound=bound, sample=sample,
                )
                sampler.run_nested(
                    nlive_init=nlive, dlogz_init=tol,
                    print_progress=print_progress,
                )
        else:
            sampler = dynesty.DynamicNestedSampler(
                loglike, prior_transform, ndim,
                bound=bound, sample=sample,
            )
            sampler.run_nested(nlive_init=nlive, print_progress=print_progress)
    else:
        raise ValueError("ns_modus must be 'static' or 'dynamic', got {!r}".format(modus))

    elapsed = timer() - t0
    logprint("\nTime taken to run 'dynesty' ({}): {:.2f} hours".format(modus, elapsed / 3600.0))

    r = sampler.results
    # dynesty Results behaves like a dict; access via subscript for portability.
    out = build_results(
        backend="dynesty",
        samples=r["samples"],
        logwt=r["logwt"],
        logz=r["logz"],
        logzerr=r["logzerr"],
        fitkeys=param_names,
        wall_time_sec=elapsed,
        raw=None,
    )
    # Pass-through all native dynesty fields so dyplot.traceplot / cornerplot
    # can still consume the unified dict (they need logvol, niter, logl, nlive,
    # samples_id, etc.). Unified-schema keys (set above) take precedence.
    try:
        native_keys = list(r.keys())
    except Exception:
        native_keys = []
    for k in native_keys:
        if k not in out:
            try:
                out[k] = r[k]
            except Exception:
                pass
    return out
