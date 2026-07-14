#!/usr/bin/env python2
"""
Nested Sampling inference module.

This module provides Dynamic Nested Sampling using dynesty for Bayesian
parameter estimation. It computes the Bayesian evidence (log Z) which enables
model comparison via Bayes factors. Supports both static and dynamic sampling
modes for efficient evidence estimation.

Functions:
    ns_fit: Run nested sampling for parameter inference and evidence calculation.
    ns_lnlike: Compute log-likelihood for nested sampling.
    ns_lnprior: Compute log-prior for nested sampling.

Notes
-----
    - Uses dynesty for nested sampling
    - Supports static and dynamic sampling modes
    - Computes Bayesian evidence for model comparison
    - Saves results as compressed pickle files
"""

#::: modules
import multiprocessing
import os

import numpy as np
from scipy.special import ndtri
from scipy.stats import truncnorm

from ._numpy_compat import RankWarning, VisibleDeprecationWarning

multiprocessing.set_start_method("fork", force=True)
# solves python>=3.8 issues, see https://stackoverflow.com/questions/60518386/error-with-module-multiprocessing-under-python3-8
import gzip
import shutil

try:
    import cPickle as pickle
except Exception:
    import pickle

#::: warnings
import warnings

warnings.filterwarnings("ignore", category=VisibleDeprecationWarning)
warnings.filterwarnings("ignore", category=RankWarning)

#::: allesfitter modules
from . import config
from .computer import calculate_lnlike_total, update_params
from .general_output import logprint, resolve_overwrite


###############################################################################
#::: Nested Sampling log likelihood
###############################################################################
def ns_lnlike(theta):
    params = update_params(theta)
    lnlike = calculate_lnlike_total(params)

    #    lnlike = 0
    #
    #    for inst in config.BASEMENT.settings['inst_phot']:
    #        lnlike += calculate_lnlike(params, inst, 'flux')
    #
    #    for inst in config.BASEMENT.settings['inst_rv']:
    #        lnlike += calculate_lnlike(params, inst, 'rv')
    #
    #    if np.isnan(lnlike) or np.isinf(lnlike):
    #        lnlike = -np.inf

    return lnlike


###############################################################################
#::: Nested Sampling prior transform
###############################################################################
def ns_prior_transform(utheta):
    #    global config.BASEMENT
    theta = np.zeros_like(utheta) * np.nan
    for i in range(len(theta)):
        if config.BASEMENT.bounds[i][0] == "uniform":
            theta[i] = (
                utheta[i] * (config.BASEMENT.bounds[i][2] - config.BASEMENT.bounds[i][1])
                + config.BASEMENT.bounds[i][1]
            )
        elif config.BASEMENT.bounds[i][0] == "normal":
            theta[i] = config.BASEMENT.bounds[i][1] + config.BASEMENT.bounds[i][2] * ndtri(
                utheta[i]
            )
        elif config.BASEMENT.bounds[i][0] == "trunc_normal":
            theta[i] = my_truncnorm_isf(
                utheta[i],
                config.BASEMENT.bounds[i][1],
                config.BASEMENT.bounds[i][2],
                config.BASEMENT.bounds[i][3],
                config.BASEMENT.bounds[i][4],
            )
        else:
            raise ValueError(
                'Bounds have to be "uniform", "normal" and "trunc_normal". Input from "params.csv" was "'
                + config.BASEMENT.bounds[i][0]
                + '".'
            )
    return theta


def my_truncnorm_isf(q, a, b, mean, std):
    a_scipy = 1.0 * (a - mean) / std
    b_scipy = 1.0 * (b - mean) / std
    return truncnorm.isf(q, a_scipy, b_scipy, loc=mean, scale=std)


###############################################################################
#::: Nested Sampling fitter class
###############################################################################
def ns_fit(datadir, backend=None, overwrite=None):
    """Run nested sampling and persist the result.

    Parameters
    ----------
    datadir : str
        Working directory containing ``settings.csv`` / ``params.csv``.
    backend : {'dynesty', 'ultranest', None}, optional
        Which sampler to use. Resolution order:
        1. ``settings.csv`` ``ns_backend`` row (if set);
        2. this kwarg (if provided);
        3. ``'dynesty'`` (default).
    overwrite : bool or None, optional
        If ``True``, overwrite an existing ``save_ns.pickle.gz`` and start a
        genuinely fresh run (also wiping ultranest's ``ultranest_logs`` resume
        store). If ``False``, abort when a save file exists. If ``None``
        (default) and a save file exists, the user is prompted interactively
        (overwrite / abort); a non-interactive stdin proceeds by overwriting.

    Notes
    -----
    Saves an ``NSResults`` (unified schema) to
    ``<outdir>/save_ns.pickle.gz``. ``ns_output`` auto-detects the backend
    from the saved object.

    There is no ``append`` option: nested sampling has no chain to extend, the
    results pickle is rewritten wholesale every run, and ``ultranest`` already
    auto-resumes from its ``ultranest_logs`` store unless ``overwrite=True``
    clears it (``dynesty`` cannot resume at all).
    """
    #::: init
    config.init(datadir)
    from .results import use_results_directory

    use_results_directory(config.BASEMENT, "ns", for_write=True)

    #::: resolve backend (settings.csv wins, then kwarg, then default)
    resolved = config.BASEMENT.settings.get("ns_backend") or backend or "dynesty"
    resolved = str(resolved).lower()

    #::: resolve overwrite for an existing save file (prompts if overwrite is
    #::: None; EOFError-safe so non-interactive/batch runs proceed by
    #::: overwriting). Raises ValueError if the user (or overwrite=False) aborts.
    save_file = os.path.join(config.BASEMENT.outdir, "save_ns.pickle.gz")
    resolve_overwrite(save_file, overwrite=overwrite, label="Nested Sampling save")

    #::: an explicit overwrite=True also clears ultranest's resume store so the
    #::: run starts genuinely fresh (otherwise ultranest auto-resumes from it).
    if overwrite:
        ultranest_logs = os.path.join(config.BASEMENT.outdir, "ultranest_logs")
        if os.path.isdir(ultranest_logs):
            shutil.rmtree(ultranest_logs)

    #::: dispatch (imports happen lazily, so ultranest stays optional)
    from .utils.ns_backends import get_backend, validate_settings_for_backend

    be = get_backend(resolved)

    #::: tell the user which settings.csv keys are unused / implicit for this backend
    raw_keys = getattr(config.BASEMENT, "_settings_raw_keys", set())
    # Also surface ns_backend itself when defaulted
    if "ns_backend" not in raw_keys:
        logprint(
            f"\n! settings.csv: 'ns_backend' not set; defaulting to '{resolved}'. "
            "Add a row to settings.csv for reproducibility."
        )
    validate_settings_for_backend(resolved, raw_keys, logprint=logprint)

    results = be.run(
        ns_lnlike,
        ns_prior_transform,
        config.BASEMENT.ndim,
        config.BASEMENT.settings,
        list(config.BASEMENT.fitkeys),
        config.BASEMENT.outdir,
        logprint=logprint,
    )

    #::: pickle-save the unified-schema results
    with gzip.GzipFile(os.path.join(config.BASEMENT.outdir, "save_ns.pickle.gz"), "wb") as f:
        pickle.dump(results, f)

    #::: return a German saying
    try:
        with open(os.path.join(os.path.dirname(__file__), "utils", "quotes2.txt")) as dataset:
            return np.random.choice([l for l in dataset])
    except Exception:
        return "42"
