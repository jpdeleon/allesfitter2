#!/usr/bin/env python2
"""
Created on Tue Oct  2 10:54:58 2018

@author:
Dr. Maximilian N. Günther
European Space Agency (ESA)
European Space Research and Technology Centre (ESTEC)
Keplerlaan 1, 2201 AZ Noordwijk, The Netherlands
Email: maximilian.guenther@esa.int
GitHub: mnguenther
Twitter: m_n_guenther
Web: www.mnguenther.com
"""

#::: modules
import json
import os
import tempfile
import warnings

import matplotlib.pyplot as plt
import numpy as np

#::: plotting settings
import seaborn as sns
from scipy.optimize import minimize
from scipy.stats import skewnorm

sns.set(
    context="paper",
    style="ticks",
    palette="deep",
    font="sans-serif",
    font_scale=1.5,
    color_codes=True,
)
sns.set_style({"xtick.direction": "in", "ytick.direction": "in"})
sns.set_context(rc={"lines.markeredgewidth": 1})

#::: random seed
np.random.seed(42)


def simulate_PDF(median, lower_err, upper_err, size=1, plot=True):
    """
    Simulates a draw of posterior samples from a value and asymmetric errorbars
    by assuming the underlying distribution is a skewed normal distribution.

    Developed to estimate PDFs from literature exoplanet parameters that did not report their MCMC chains.

    Inputs:
    -------
    median : float
        the median value that was reported
    lower_err : float
        the lower errorbar that was reported
    upper_err : float
        the upper errorbar that was reported
    size : int
        the number of samples to be drawn

    Returns:
    --------
    samples : array of float
        the samples drawn from the simulated skrewed normal distribution
    """

    alpha, loc, scale = calculate_skewed_normal_params(median, lower_err, upper_err)
    samples = skewnorm.rvs(alpha, loc=loc, scale=scale, size=size)

    if not plot:
        return samples

    else:
        lower_err = np.abs(lower_err)
        upper_err = np.abs(upper_err)
        x = np.linspace(
            np.min(samples) - 0.1 * np.ptp(samples), np.max(samples) + 0.1 * np.ptp(samples), 1000
        )
        # x = np.arange(median-4*lower_err, median+4*upper_err, 0.001)
        fig = plt.figure()
        for i in range(3):
            plt.axvline([median - lower_err, median, median + upper_err][i], color="k", lw=2)
        plt.plot(x, skewnorm.pdf(x, alpha, loc=loc, scale=scale), "r-", lw=2)
        fit_percentiles = skewnorm.ppf([0.15865, 0.5, 0.84135], alpha, loc=loc, scale=scale)
        for i in range(3):
            plt.axvline(fit_percentiles[i], color="r", ls="--", lw=2)
        plt.hist(samples, density=True, color="r", alpha=0.5, bins=100)
        print("\n")
        print("Input", median - lower_err, median, median + upper_err)
        print("Output", fit_percentiles)
        plt.text(
            0.95,
            0.95,
            "Black: Input\nRed: Output",
            ha="right",
            va="top",
            transform=plt.gca().transAxes,
        )
        return samples, fig


###############################################################################
#::: disk cache for the expensive (alpha, loc, scale) solve
###############################################################################
#
# `calculate_skewed_normal_params` solves an inverse percentile problem via
# scipy.stats.skewnorm.ppf, which internally numerically integrates the
# skewed-normal CDF. For typical stellar inputs this takes ~10–15 s per call;
# in pipelines that re-init Basement many times (e.g. per-target loops or
# successive `config.init(datadir)` calls) the cost adds up to tens of
# minutes for inputs that never change.
#
# Cache the (median, lower_err, upper_err) → (alpha, loc, scale) mapping to
# a JSON file on disk. Same inputs in any future process get the result
# instantly. Override the path with the ALLESFITTER_SIMULATE_PDF_CACHE
# environment variable; set ALLESFITTER_SIMULATE_PDF_NO_CACHE=1 to bypass.

_CACHE_ENV_PATH = "ALLESFITTER_SIMULATE_PDF_CACHE"
_CACHE_ENV_DISABLE = "ALLESFITTER_SIMULATE_PDF_NO_CACHE"
_DEFAULT_CACHE_PATH = os.path.join(
    os.path.expanduser("~"), ".allesfitter", "simulate_PDF_cache.json"
)


def _cache_path():
    return os.environ.get(_CACHE_ENV_PATH, _DEFAULT_CACHE_PATH)


def _cache_disabled():
    return os.environ.get(_CACHE_ENV_DISABLE, "").strip() in ("1", "true", "True")


def _cache_key(median, lower_err, upper_err):
    """Stable string key from three floats. ``repr(float)`` round-trips
    losslessly in Python 3, so equal inputs always produce equal keys."""
    return f"{repr(float(median))}|{repr(float(lower_err))}|{repr(float(upper_err))}"


def _cache_load():
    path = _cache_path()
    try:
        with open(path) as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return {}
        return data
    except (OSError, ValueError):
        return {}


def _cache_save(cache_dict):
    """Atomic write: dump to a sibling temp file then os.replace into place.
    On any I/O failure we warn but never raise — the cache is an
    optimisation, not a correctness requirement.
    """
    path = _cache_path()
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        fd, tmp = tempfile.mkstemp(
            prefix=".simulate_PDF_cache_",
            dir=os.path.dirname(path) or ".",
            suffix=".json",
        )
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(cache_dict, f, indent=2, sort_keys=True)
            os.replace(tmp, path)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)
    except OSError as exc:
        warnings.warn(
            f"simulate_PDF cache write to {path} failed ({exc}); next call will recompute.",
            stacklevel=2,
        )


def _calculate_skewed_normal_params_uncached(median, lower_err, upper_err):
    """
    Fits a screwed normal distribution via its CDF to the [16,50,84]-percentiles

    Inputs:
    -------
    median : float
        the median value that was reported
    lower_err : float
        the lower errorbar that was reported
    upper_err : float
        the upper errorbar that was reported
    size : int
        the number of samples to be drawn

    Returns:
    --------
    alpha : float
        the skewness parameter
    loc : float
        the mean of the fitted skewed normal distribution
    scale : float
        the std of the fitted skewed normal distribution
    """

    lower_err = np.abs(lower_err)
    upper_err = np.abs(upper_err)
    reference = np.array([(median - lower_err), median, (median + upper_err)])

    def fake_lnlike(p):
        alpha, loc, scale = p

        #::: way 1: easier to read, but slower; fake log likelihood
        # eq1 = skewnorm.ppf(0.5, alpha, loc=loc, scale=scale) - median
        # eq2 = skewnorm.ppf(0.15865, alpha, loc=loc, scale=scale) - (median-lower_err)
        # eq3 = skewnorm.ppf(0.84135, alpha, loc=loc, scale=scale) - (median+upper_err)
        # fake_lnlike = 0.5 * np.log( eq1**2 + eq2**2 + eq3**2 ) #fake log likelihod

        #::: way 2: pythonic; simple chi squared
        ppf = skewnorm.ppf([0.15865, 0.5, 0.84135], alpha, loc=loc, scale=scale)  # np array
        fake_lnlike = np.sum((ppf - reference) ** 2)  # simple chi squared

        #::: way 2: pythonic; fake log likehood, just 'cause we're feeling fancy
        # ppf = skewnorm.ppf([0.15865, 0.5, 0.84135], alpha, loc=loc, scale=scale) #np array
        # fake_lnlike = 0.5 * np.log( np.sum( (ppf - reference)**2 ) ) #fake log likelihod

        if np.isnan(fake_lnlike):
            return np.inf
        else:
            return fake_lnlike

    # TODO:
    # scipy minimize is really bad because it depends so strongly on the initial guess
    # and likes to get stuck in a local minima, which is the worst possible outcome
    # for our purpose here. We only use it because it is fast. Think about replacing
    # it with something more robust in the future though, maybe a short MCMC chain or the like.
    # The alpha_guess hack below seems to get around its weakness in finding the right alpha for now.

    #::: initial guess for loc and scale
    loc_guess = median
    scale_guess = np.mean([lower_err, upper_err])
    # print('\n')

    #::: way 1: choose alpha_guess depending on the errors and hope for the best
    # if lower_err == upper_err:
    #     alpha_guess = 0
    # elif lower_err < upper_err:
    #     alpha_guess = 1
    # elif lower_err > upper_err:
    #     alpha_guess = -1
    # initial_guess = (median, sigma_guess, alpha_guess) #sigma, omega, alpha
    # sol = minimize(fake_lnlike, initial_guess, bounds=[(None,None), (0,None), (None,None)])
    # sigma, omega, alpha = sol.x

    #::: way 2: choose a few different alpha_guesses and compare (similar to basinhopping optimization)
    # initial_guess1 = None #just for printing
    sol = None
    for alpha_guess in [-10, -1, 0, 1, 10]:
        initial_guess1 = (alpha_guess, loc_guess, scale_guess)
        sol1 = minimize(fake_lnlike, initial_guess1, bounds=[(None, None), (None, None), (0, None)])
        # print('sol1.fun', sol1.fun)
        if (sol is None) or (sol1.fun < sol.fun):
            # initial_guess = initial_guess1 #just for printing
            sol = sol1

    # print('best initial_guess:', initial_guess)
    # print('best solution:', sol)

    alpha, loc, scale = sol.x
    return alpha, loc, scale


def calculate_skewed_normal_params(median, lower_err, upper_err):
    """Cached public entry point. See
    :func:`_calculate_skewed_normal_params_uncached` for the math.

    On a cache hit the JSON read is ~1 ms; on a miss the underlying
    scipy.optimize.minimize + skewnorm.ppf loop takes ~10–15 s (no change
    from before this wrapper existed). The result is persisted so future
    process invocations skip the solve entirely.

    Bypass the cache via the ``ALLESFITTER_SIMULATE_PDF_NO_CACHE=1``
    environment variable; redirect it via ``ALLESFITTER_SIMULATE_PDF_CACHE``.
    """
    if _cache_disabled():
        return _calculate_skewed_normal_params_uncached(median, lower_err, upper_err)
    key = _cache_key(median, lower_err, upper_err)
    cache = _cache_load()
    if key in cache:
        try:
            alpha, loc, scale = cache[key]
            return float(alpha), float(loc), float(scale)
        except (TypeError, ValueError):
            # corrupt entry — fall through and recompute
            pass
    alpha, loc, scale = _calculate_skewed_normal_params_uncached(median, lower_err, upper_err)
    cache[key] = [float(alpha), float(loc), float(scale)]
    _cache_save(cache)
    return alpha, loc, scale


if __name__ == "__main__":
    """
    For testing, simulate a skewed normal distribution with parameters
    (alpha_0, loc_0, scale_0), perform the fit, and compare the resulting PDF.
    """

    ###########################################################################
    # ::: three test cases
    ###########################################################################
    # simulate_PDF(1., 0.15, 0.15, size=10000) #R_star

    # simulate_PDF(1.895, 0.077, 0.1, size=10000) #M_star 1

    # simulate_PDF(1.895, 0.07, 0.15, size=10000) #M_star 2

    # simulate_PDF(84.3, -2.8, 1.3, size=10000) #inclination

    ###########################################################################
    # ::: simulate a skewed normal PDF, measure median and errors, and fit
    ###########################################################################
#    sigma_0 = 0.
#    omega_0 = 5.
#    alpha_0 = 2
#
#    x = np.arange(-20+sigma_0,20+sigma_0,0.01)
#    y = skewnorm.pdf(x, alpha_0, loc=sigma_0, scale=omega_0)
#    plt.figure()
#    plt.plot( x, y, 'b-', lw=4 )
#
#    median = skewnorm.ppf(0.5, alpha_0, loc=sigma_0, scale=omega_0)
#    lower_err = median - skewnorm.ppf(0.16, alpha_0, loc=sigma_0, scale=omega_0)
#    upper_err = skewnorm.ppf(0.84, alpha_0, loc=sigma_0, scale=omega_0) - median
#    plt.axvline(median)
#    plt.axvline(median-lower_err)
#    plt.axvline(median+upper_err)
#
#    sigma, omega, alpha = calculate_skewed_normal_params(median, lower_err, upper_err)
#    plt.plot( x, skewnorm.pdf(x, alpha, loc=sigma, scale=omega), 'r--', lw=2 )


###########################################################################
# ::: example inclination
###########################################################################
#    (median, lower_err, upper_err) = (84.3, -2.0, 1.3)
#    samples, fig = simulate_posterior_samples(median, lower_err, upper_err, size=1, plot=True)
#    print(np.percentile(samples, [16,50,84]))


#    lower_err = np.abs(lower_err)
#    upper_err = np.abs(upper_err)
#
#    x = np.arange(median-4*lower_err, median+4*upper_err, 0.01)
#    plt.figure()
#    plt.axvline(median, color='k', lw=2)
#    plt.axvline(median-lower_err, color='k', lw=2)
#    plt.axvline(median+upper_err, color='k', lw=2)
#
#    sigma, omega, alpha = calculate_skewed_normal_params(median, lower_err, upper_err)
#    plt.plot( x, skewnorm.pdf(x, alpha, loc=sigma, scale=omega), 'r-', lw=2 )
#
#    fit_percentiles = skewnorm.ppf([0.16, 0.5, 0.84], alpha, loc=sigma, scale=omega)
#    for i in range(3): plt.axvline(fit_percentiles[i], color='r', ls='--', lw=2)
#
#    rvs = simulate_posterior_samples(median, lower_err, upper_err, size=1000)
#    plt.hist(rvs, density=True, color='lightblue')
