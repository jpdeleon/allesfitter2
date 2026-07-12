#!/usr/bin/env python3
"""
Created on Fri Oct  5 14:28:55 2018

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
import gzip
import os
from copy import deepcopy

import numpy as np

_HAS_PLOTTING = False


def _init_plotting():
    global _HAS_PLOTTING
    if _HAS_PLOTTING:
        return
    _HAS_PLOTTING = True
    import seaborn as sns

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


#::: allesfitter modules
from . import config
from ._output_shared import save_per_transit_plots, write_priors_latex_table
from .computer import calculate_baseline, calculate_model, calculate_stellar_var
from .general_output import (
    afplot,
    get_params_from_samples,
    logprint,
    plot_ttv_results,
    resolve_overwrite,
    save_latex_table,
    save_table,
)
from .plot_top_down_view import plot_top_down_view
from .statistics import residual_stats
from .utils.colormaputil import truncate_colormap
from .utils.latex_printer import round_tex


###############################################################################
#::: draw samples from the ns results (internally in the code)
###############################################################################
def draw_ns_posterior_samples(results, Nsamples=None, as_type="2d_array"):
    """
    ! posterior samples are drawn as resampled weighted samples !
    ! do not confuse posterior_samples (weighted, resampled) with results['samples'] (unweighted) !
    """
    from dynesty import utils as dyutils

    weights = np.exp(results["logwt"] - results["logz"][-1])
    np.random.seed(42)
    posterior_samples = dyutils.resample_equal(results["samples"], weights)
    if Nsamples:
        posterior_samples = posterior_samples[
            np.random.randint(len(posterior_samples), size=Nsamples)
        ]

    if as_type == "2d_array":
        return posterior_samples

    elif as_type == "dic":
        posterior_samples_dic = {}
        for key in config.BASEMENT.fitkeys:
            ind = np.where(config.BASEMENT.fitkeys == key)[0]
            posterior_samples_dic[key] = posterior_samples[:, ind].flatten()
        return posterior_samples_dic


###############################################################################
#::: backend-agnostic plotting helpers (used when backend != 'dynesty')
###############################################################################
# Memory caps for the fallback plotting helpers. For very high-dim fits
# (chromatic + many baseline GPs + per-band err_flux + LDCs) the default
# matplotlib canvas balloons to ~1500 megapixels at 100 dpi and the OOM
# killer terminates the post-processing run. These caps keep the plots
# legible *and* small enough to render on a workstation.
_MAX_TRACE_INCHES = 60.0  # cap _simple_traceplot height
_MAX_CORNER_INCHES = 40.0  # cap corner.corner figsize per side
_MAX_TRACE_PLOT_SAMPLES = 5000  # subsample raw trace per panel
_MAX_CORNER_SAMPLES = 10000  # subsample posterior for corner
_HARD_NDIM_CAP = 60  # above this, skip the corner plot entirely


def _simple_traceplot(results, labels, truths):
    """Lightweight traceplot when dynesty's runplot/traceplot is unavailable.

    Mirrors the (ndim, 2) layout dyplot.traceplot returns so the downstream
    title-setting code (``taxes[i,1].set_title(...)`` etc.) keeps working.
    Left column: index vs. parameter value (raw samples).
    Right column: weighted-posterior histogram.

    Memory-bounded: the figure height is capped at ``_MAX_TRACE_INCHES``
    inches regardless of ndim, and the raw-sample trace is subsampled to
    ``_MAX_TRACE_PLOT_SAMPLES`` points per panel (the weighted histogram
    uses all samples).
    """
    _init_plotting()
    import matplotlib.pyplot as plt

    samples = np.asarray(results["samples"])
    logwt = np.asarray(results["logwt"])
    logz_final = float(np.asarray(results["logz"])[-1])
    weights = np.exp(logwt - logz_final)
    weights = weights / weights.sum()
    ndim = samples.shape[1]
    # Cap the figure height so it stays renderable for ndim >> 20.
    height = min(2.5 * ndim, _MAX_TRACE_INCHES)
    fig, axes = plt.subplots(ndim, 2, figsize=(12, height))
    if ndim == 1:
        axes = np.array([axes])
    # Subsample the trace for plotting only — the histogram still uses
    # the full posterior.
    n = samples.shape[0]
    if n > _MAX_TRACE_PLOT_SAMPLES:
        step = max(1, n // _MAX_TRACE_PLOT_SAMPLES)
        trace_idx = np.arange(0, n, step)
    else:
        trace_idx = slice(None)
    for i in range(ndim):
        axes[i, 0].plot(samples[trace_idx, i], lw=0.5, color="grey", rasterized=True)
        axes[i, 0].set_ylabel(labels[i] if i < len(labels) else "")
        axes[i, 1].hist(
            samples[:, i], bins=60, weights=weights, color="grey", histtype="stepfilled", alpha=0.6
        )
        if truths is not None and i < len(truths) and truths[i] is not None:
            axes[i, 0].axhline(truths[i], color="C3", lw=0.8)
            axes[i, 1].axvline(truths[i], color="C3", lw=0.8)
    axes[-1, 0].set_xlabel("sample index")
    axes[-1, 1].set_xlabel("value")
    return fig, axes


# ---------------------------------------------------------------------------
# Nuisance-parameter filtering for the corner plot.
#
# When ndim is large, the corner plot becomes unreadable AND its matplotlib
# canvas approaches OOM territory. The baseline-model rows (GP hypers,
# offsets/slopes, per-band white-noise jitters, stellar-variability GP
# hypers) are almost always nuisance for inspection — the science is in
# the orbital + planet-radius + LDC parameters. When ndim exceeds the
# threshold below, drop the nuisance rows from the corner-plot only; the
# full posterior is still saved to disk and used everywhere else.
# ---------------------------------------------------------------------------
_NUISANCE_CORNER_PREFIXES = (
    "baseline_",  # offsets/slopes + every baseline_gp_* hyper
    "ln_err_flux_",
    "ln_jitter_rv_",
    "stellar_var_gp_",
)
_CORNER_HIDE_NUISANCE_NDIM_THRESHOLD = 25


def _corner_nuisance_mask(fitkeys):
    """Return a boolean keep-mask over ``fitkeys``: True for science
    parameters, False for nuisance baseline / error / stellar-var-GP rows.
    """
    keys = np.atleast_1d(np.asarray(fitkeys, dtype=object))
    keep = np.ones(len(keys), dtype=bool)
    for i, k in enumerate(keys):
        ks = str(k)
        if any(ks.startswith(p) for p in _NUISANCE_CORNER_PREFIXES):
            keep[i] = False
    return keep


def _filter_nuisance_for_corner(
    fitkeys,
    samples,
    labels,
    truths,
    threshold=_CORNER_HIDE_NUISANCE_NDIM_THRESHOLD,
):
    """Drop nuisance rows iff ``len(fitkeys) > threshold``.

    Returns ``(fitkeys_kept, samples_kept, labels_kept, truths_kept,
    dropped_names)`` — when no filtering happens (ndim small, or no
    nuisance rows found) the inputs are returned unchanged and
    ``dropped_names`` is an empty list.
    """
    n = len(fitkeys)
    if n <= threshold:
        return fitkeys, samples, labels, truths, []
    keep = _corner_nuisance_mask(fitkeys)
    if keep.all():
        return fitkeys, samples, labels, truths, []
    dropped = [str(k) for k, kk in zip(np.atleast_1d(fitkeys), keep) if not kk]
    s = np.asarray(samples)
    s_kept = s[:, keep] if s.ndim == 2 else s
    keys_kept = np.asarray(fitkeys)[keep]
    labels_kept = [labels[i] for i in range(len(labels)) if keep[i]]
    if truths is not None:
        truths_kept = np.asarray(truths)[keep]
    else:
        truths_kept = None
    return keys_kept, s_kept, labels_kept, truths_kept, dropped


def _corner_from_samples(eq_samples, labels, truths, fontsize, ndim):
    """Draw a corner plot from an already equal-weight posterior.

    Memory-bounded: subsamples to ``_MAX_CORNER_SAMPLES`` rows, caps
    figsize at ``_MAX_CORNER_INCHES`` per side, and short-circuits to an
    empty grid when ``ndim > _HARD_NDIM_CAP``.
    """
    _init_plotting()
    import corner as _corner
    import matplotlib.pyplot as plt

    eq = np.asarray(eq_samples)
    if eq.shape[0] > _MAX_CORNER_SAMPLES:
        rng = np.random.default_rng(42)
        idx = rng.choice(eq.shape[0], size=_MAX_CORNER_SAMPLES, replace=False)
        eq = eq[idx]
    side_inches = min(2.0 * ndim, _MAX_CORNER_INCHES)
    if ndim > _HARD_NDIM_CAP or _corner is None:
        cfig, caxes = plt.subplots(ndim, ndim, figsize=(side_inches, side_inches))
        return cfig, caxes
    try:
        cfig = _corner.corner(
            eq,
            labels=labels,
            truths=truths,
            quantiles=[0.16, 0.5, 0.84],
            show_titles=False,
            fig=plt.figure(figsize=(side_inches, side_inches)),
            label_kwargs={"fontsize": fontsize, "rotation": 45, "horizontalalignment": "right"},
            hist_kwargs={"alpha": 0.25, "linewidth": 0, "histtype": "stepfilled"},
        )
        caxes = np.array(cfig.axes).reshape((ndim, ndim))
    except (MemoryError, ValueError, RuntimeError):
        plt.close("all")
        cfig, caxes = plt.subplots(ndim, ndim, figsize=(side_inches, side_inches))
    return cfig, caxes


def _corner_from_results(results, labels, truths, fontsize, ndim):
    """Backend-agnostic cornerplot via ``corner.corner``.

    Resamples to equal weight using the same ``logwt``/``logz`` contract
    as :func:`draw_ns_posterior_samples`, then hands off to
    :func:`_corner_from_samples`. Returns ``(fig, axes_2d)`` where
    ``axes_2d`` matches dyplot.cornerplot's shape so existing title-
    setting code is unchanged.
    """
    from dynesty import utils as dyutils

    samples = np.asarray(results["samples"])
    logwt = np.asarray(results["logwt"])
    logz_final = float(np.asarray(results["logz"])[-1])
    weights = np.exp(logwt - logz_final)
    weights = weights / weights.sum()
    eq = dyutils.resample_equal(samples, weights)
    return _corner_from_samples(eq, labels, truths, fontsize, ndim)


###############################################################################
#::: chromatic Rp/Rs posterior histogram
###############################################################################
def _load_R_star_samples(n_samples, seed=42):
    """Load R_star posterior from params_star.csv and draw n_samples
    from an asymmetric normal. Returns (R_star_samples_in_Rsun, R_star_med)
    or (None, None) if params_star.csv is absent or malformed.

    Honors both columns ``R_star_lerr`` (lower 1-sigma) and ``R_star_uerr``
    (upper 1-sigma); a sample's draw uses the side appropriate to its sign.
    """
    path = os.path.join(config.BASEMENT.datadir, "params_star.csv")
    if not os.path.exists(path):
        return None, None
    try:
        buf = np.genfromtxt(
            path,
            delimiter=",",
            names=True,
            dtype=None,
            encoding="utf-8",
            comments="#",
        )
        R_star = float(np.atleast_1d(buf["R_star"])[0])
        R_lerr = float(np.atleast_1d(buf["R_star_lerr"])[0])
        R_uerr = float(np.atleast_1d(buf["R_star_uerr"])[0])
    except (KeyError, ValueError, TypeError, IndexError):
        return None, None
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n_samples)
    samples = np.where(z < 0, R_star + z * R_lerr, R_star + z * R_uerr)
    samples = np.clip(samples, 1e-4, None)  # guard against negatives
    return samples, R_star


def plot_chromatic_rr_histogram(posterior_samples, prefix="ns"):
    """
    Overlay posterior histograms of per-bandpass Rp/Rs for chromatic fits.

    Only runs when settings.csv defines ``bandpass`` with >=2 unique labels
    (``config.BASEMENT.settings['chromatic'] is True``). One PDF is written
    per photometric companion at
    ``<outdir>/<prefix>_chromatic_rr_<companion>.pdf`` (``prefix`` defaults
    to ``'ns'`` to preserve the legacy NS filename; ``mcmc_output`` passes
    ``prefix='mcmc'``).

    When ``params_star.csv`` is present in the datadir, a second panel is
    added below showing the implied planet radius posterior with twin
    x-axes: the bottom axis in Earth radii and the top axis in Jupiter
    radii (sharing the same data). R_star uncertainty is propagated by
    sampling from the asymmetric normal described in params_star.csv.

    Inputs
    ------
    posterior_samples : 2d ndarray
        Weighted posterior samples with shape (Nsamples, ndim) matching
        ``config.BASEMENT.fitkeys`` ordering.
    prefix : str
        Output filename prefix (``'ns'`` or ``'mcmc'``).
    """
    _init_plotting()

    import matplotlib.pyplot as plt

    if not config.BASEMENT.settings.get("chromatic", False):
        return

    bandpass_map = config.BASEMENT.settings.get("bandpass", {}) or {}
    unique_bandpasses = sorted(set(bandpass_map.values()))
    if len(unique_bandpasses) < 2:
        return

    fitkeys = list(config.BASEMENT.fitkeys)

    # Solar / planetary radius constants (CODATA-consistent with astropy).
    # Local constants keep this function importable without astropy at top
    # level; matches the conversions used in deriver.py:329-330.
    R_SUN_KM = 6.957e5
    R_EARTH_KM = 6.3781e3
    R_JUP_KM = 7.1492e4
    SUN_TO_EARTH = R_SUN_KM / R_EARTH_KM  # ~109.08
    EARTH_TO_JUP = R_EARTH_KM / R_JUP_KM  # ~0.0892

    for companion in config.BASEMENT.settings["companions_phot"]:
        per_band = []
        for bp in unique_bandpasses:
            key = f"{companion}_rr_{bp}"
            if key in fitkeys:
                ind = fitkeys.index(key)
                per_band.append((bp, posterior_samples[:, ind]))

        if len(per_band) < 2:
            continue

        # Decide layout: 2 panels if params_star.csv is available, else 1.
        n_post = len(per_band[0][1])
        R_star_samples, R_star_med = _load_R_star_samples(n_post)
        two_panel = R_star_samples is not None

        # Canonical per-bandpass color map; unknown bandpasses fall back to
        # the viridis ramp so the plot still works for arbitrary labels.
        color_map = {"tess": "k", "g": "C0", "r": "C2", "i": "C8", "z": "C3"}
        fallback = plt.cm.viridis(np.linspace(0.15, 0.85, len(per_band)))
        colors = [color_map.get(bp, fallback[i]) for i, (bp, _) in enumerate(per_band)]

        if two_panel:
            fig, (ax, ax_r) = plt.subplots(2, 1, figsize=(7, 9))
        else:
            fig, ax = plt.subplots(figsize=(7, 5))
            ax_r = None

        # Top panel: Rp/Rs histograms (unchanged).
        for (bp, samples), color in zip(per_band, colors):
            med = np.median(samples)
            lo, hi = np.percentile(samples, [16, 84])
            label = f"{bp}: ${med:.4f}^{{+{hi - med:.4f}}}_{{-{med - lo:.4f}}}$"
            ax.hist(
                samples,
                bins=40,
                density=True,
                alpha=0.5,
                color=color,
                label=label,
                histtype="stepfilled",
                edgecolor=color,
                linewidth=1.2,
            )
            ax.axvline(med, color=color, linestyle="--", linewidth=1.0, alpha=0.9)

        ax.set_xlabel(r"$R_p / R_\star$ (companion " + companion + ")")
        ax.set_ylabel("Posterior density")
        ax.set_title("Chromatic transit depth posterior (companion " + companion + ")")
        ax.legend(loc="best", fontsize=10)

        # Bottom panel: implied R_p in Earth radii (bottom x-axis) with a
        # twin top x-axis in Jupiter radii. Only when R_star is available.
        if two_panel:
            for (bp, rr_samples), color in zip(per_band, colors):
                # Propagate R_star uncertainty by drawing fresh R_star
                # samples per band so the two posteriors are independent
                # (they share the same R_star prior, so any correlated
                # offset would be a stellar-systematic, not band).
                R_p_earth = R_star_samples * rr_samples * SUN_TO_EARTH
                med = np.median(R_p_earth)
                lo, hi = np.percentile(R_p_earth, [16, 84])
                label = f"{bp}: ${med:.2f}^{{+{hi - med:.2f}}}_{{-{med - lo:.2f}}}\\,R_\\oplus$"
                ax_r.hist(
                    R_p_earth,
                    bins=40,
                    density=True,
                    alpha=0.5,
                    color=color,
                    label=label,
                    histtype="stepfilled",
                    edgecolor=color,
                    linewidth=1.2,
                )
                ax_r.axvline(med, color=color, linestyle="--", linewidth=1.0, alpha=0.9)

            ax_r.set_xlabel(
                r"$R_p$ ($R_\oplus$)  " + rf"(assuming $R_\star = {R_star_med:.2f}\,R_\odot$)"
            )
            ax_r.set_ylabel("Posterior density")
            ax_r.legend(loc="best", fontsize=10)

            # Twin top x-axis in Jupiter radii.
            ax_r_jup = ax_r.secondary_xaxis(
                "top",
                functions=(lambda r: r * EARTH_TO_JUP, lambda r: r / EARTH_TO_JUP),
            )
            ax_r_jup.set_xlabel(r"$R_p$ ($R_\mathrm{Jup}$)")

        fig.tight_layout()
        fig.savefig(
            os.path.join(config.BASEMENT.outdir, prefix + "_chromatic_rr_" + companion + ".pdf"),
            bbox_inches="tight",
        )
        plt.close(fig)


###############################################################################
#::: analyse the output from save_ns.pickle file
###############################################################################
###############################################################################
#::: convert params.csv into a LaTeX prior table
###############################################################################
def plot_linear_baseline_components(posterior_samples=None, prefix="ns"):
    """timex-style diagnostic of every linear-multi baseline in the fit.

    For each instrument whose ``baseline_<key>_<inst>`` is
    ``sample_linear_multi`` or ``hybrid_linear_multi`` this writes
    ``<outdir>/<prefix>_linear_baseline_<inst>.pdf`` with two stacked
    panels:

      1. Each column of the (standardized) design matrix overlaid on the
         same time axis, with the fitted weight in the legend.
      2. The resulting linear combination ``X @ w`` — the actual
         systematics curve subtracted from the residuals.

    Weights come from:

      * sample_linear_multi → posterior median of the
        ``baseline_linmulti_<col>_<key>_<inst>`` fit params.
      * hybrid_linear_multi → analytic MAP solve at the posterior-
        median transit/LDC/err params via
        :func:`computer._hybrid_linear_multi_solve` (the marginalised
        weights aren't in the posterior table by construction).

    No-op when no inst uses a linear-multi baseline. Wrapped at the
    caller in ``ns_output`` / ``mcmc_output`` with a MemoryError-safe
    try/except so a plot failure never aborts the pipeline.
    """
    # Local imports to avoid pulling computer at module load.
    _init_plotting()
    import matplotlib.pyplot as plt

    from .computer import (
        _hybrid_linear_multi_solve,
        calculate_model,
        calculate_stellar_var,
        calculate_yerr_w,
        update_params,
    )

    settings = config.BASEMENT.settings
    linmulti_kinds = ("sample_linear_multi", "hybrid_linear_multi")

    # Build a per-inst (key, baseline_type) work list.
    targets = []
    for key, key2 in zip(("flux", "rv", "rv2"), ("inst_phot", "inst_rv", "inst_rv2")):
        for inst in settings.get(key2, []):
            btype = settings.get("baseline_" + key + "_" + inst, "none")
            if btype in linmulti_kinds:
                targets.append((inst, key, btype))
    if not targets:
        return

    # Median params from the posterior (used by both Tier 1 weight lookup
    # and Tier 2 analytic solve).
    if posterior_samples is None:
        params_median = config.BASEMENT.params
        theta_med = np.array(config.BASEMENT.theta_0, dtype=float)
    else:
        # Median of the posterior; same shape contract as elsewhere.
        theta_med = np.median(np.asarray(posterior_samples), axis=0)
        params_median = update_params(theta_med)

    for inst, key, btype in targets:
        X = config.BASEMENT.data[inst].get("design_matrix")
        cols = config.BASEMENT.data[inst].get("design_matrix_cols", [])
        if X is None or len(cols) == 0:
            continue
        t = config.BASEMENT.data[inst]["time"]

        if btype == "sample_linear_multi":
            w = np.array(
                [params_median["baseline_linmulti_" + c + "_" + key + "_" + inst] for c in cols],
                dtype=float,
            )
            w_provenance = "posterior median"
        else:  # hybrid_linear_multi — analytic MAP
            model = calculate_model(params_median, inst, key)
            yerr = calculate_yerr_w(params_median, inst, key)
            try:
                stellar_var = calculate_stellar_var(
                    params_median, inst, key, model=model, baseline=0.0, yerr_w=yerr
                )
            except Exception:
                stellar_var = 0.0
            y_resid = config.BASEMENT.data[inst][key] - model - stellar_var
            w, _corr = _hybrid_linear_multi_solve(inst, key, y_resid, yerr)
            w_provenance = "analytic MAP"

        n_cols = X.shape[1]
        fig_h = max(5.0, 2.0 + 0.8 * n_cols)
        fig, axes = plt.subplots(
            2,
            1,
            figsize=(10, fig_h),
            gridspec_kw={"height_ratios": [n_cols, 2]},
            sharex=True,
        )

        # Row 0: each design-matrix column with its weight in the legend.
        offset = 0.0
        step = 1.2 * float(np.max(np.abs(X))) if np.any(X) else 1.0
        for j, name in enumerate(cols):
            col = X[:, j]
            axes[0].plot(t, col + offset, lw=0.8, label=f"{name:s}  w={w[j]:+.4f}")
            offset += step
        axes[0].set_ylabel("standardized covariate (offset)")
        axes[0].legend(loc="upper right", fontsize="small", framealpha=0.85)
        axes[0].set_title(f"{inst}  ({btype};  weights: {w_provenance})")

        # Row 1: the linear combination X @ w (the actual baseline curve).
        axes[1].plot(t, X @ w, color="k", lw=1.0)
        axes[1].set_ylabel(r"$X \cdot w$  (rel. flux)")
        axes[1].set_xlabel("time")
        axes[1].axhline(0, color="grey", lw=0.5, ls="--")

        fig.tight_layout()
        out = os.path.join(config.BASEMENT.outdir, prefix + "_linear_baseline_" + inst + ".pdf")
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)


#::: write_priors_latex_table lives in allesfitter._output_shared (it is shared
#::: with the MCMC path); it is re-exported at the top of this module so the
#::: public import path `from allesfitter.nested_sampling_output import
#::: write_priors_latex_table` keeps working unchanged.


def ns_output(datadir, backend=None, overwrite=None):
    """
    Inputs:
    -------
    datadir : str
        the working directory for allesfitter
        must contain all the data files
        output directories and files will also be created inside datadir
    backend : {'dynesty', 'ultranest', None}, optional
        Which sampler produced the saved results. If ``None`` (default),
        auto-detect from the ``backend`` key on the unified-schema results
        dict; legacy dynesty pickles (no ``backend`` key) are treated as
        ``'dynesty'``. Affects only the trace plot (which uses
        ``dynesty.plotting.traceplot`` when available); the corner plot
        and downstream analysis are backend-agnostic.
    overwrite : bool or None, optional
        If ``True``, overwrite existing Nested Sampling output files without
        prompting; if ``False``, abort. If ``None`` (default) and output files
        already exist, the user is prompted interactively (overwrite / abort).

    Outputs:
    --------
    This will output information into the console, and create a output files
    into datadir/results/ (or datadir/QL/ if QL==True)
    """
    _init_plotting()
    import pickle

    import matplotlib.pyplot as plt
    from dynesty import plotting as dyplot
    from dynesty import utils as dyutils
    from matplotlib.ticker import MaxNLocator, ScalarFormatter

    from . import deriver

    config.init(datadir)

    #::: security check
    resolve_overwrite(
        os.path.join(config.BASEMENT.outdir, "ns_table.csv"),
        overwrite=overwrite,
        label="Nested Sampling output",
    )

    #::: load the save_ns.pickle
    f = gzip.GzipFile(os.path.join(config.BASEMENT.outdir, "save_ns.pickle.gz"), "rb")
    results = pickle.load(f)
    f.close()

    #::: resolve which sampler produced this file
    #    - new unified-schema files carry results['backend']
    #    - legacy dynesty pickles do not; default to 'dynesty'
    try:
        detected = results.get("backend") if isinstance(results, dict) else None
    except Exception:
        detected = None
    resolved_backend = (detected or backend or "dynesty").lower()

    #::: plot the fit
    posterior_samples_for_plot = draw_ns_posterior_samples(
        results, Nsamples=10
    )  # only a few samples for plotting (fit-plot speed; was 20)

    for companion in config.BASEMENT.settings["companions_all"]:
        fig, axes = afplot(posterior_samples_for_plot, companion)
        if fig is not None:
            fig.savefig(
                os.path.join(config.BASEMENT.outdir, "ns_fit_" + companion + ".pdf"),
                bbox_inches="tight",
            )
            plt.close(fig)

    #::: per-transit fit plots (shared with mcmc_output; differs only in the
    #::: posterior-sample array and the 'ns' filename prefix)
    save_per_transit_plots(posterior_samples_for_plot, "ns")

    #::: retrieve the results
    posterior_samples = draw_ns_posterior_samples(results)  # all weighted posterior_samples
    params_median, params_ll, params_ul = get_params_from_samples(
        posterior_samples
    )  # params drawn form these posterior_samples

    #::: chromatic Rp/Rs posterior histograms (no-op when achromatic)
    try:
        plot_chromatic_rr_histogram(posterior_samples)
    except Exception as _e:
        logprint("\n! WARNING: chromatic Rp/Rs histogram could not be produced: " + str(_e))

    #::: linear-multi baseline component diagnostic (timex-style; no-op
    #::: when no inst uses sample_linear_multi / hybrid_linear_multi)
    try:
        plot_linear_baseline_components(posterior_samples, prefix="ns")
    except (MemoryError, Exception) as _e:
        logprint("\n! WARNING: linear-multi baseline components plot failed: " + str(_e))

    #::: output the results
    logprint("\nResults:")
    logprint("----------")
    #    print(results.summary())
    logZ = float(np.asarray(results["logz"])[-1])  # value of logZ
    logZerr = float(np.asarray(results["logzerr"])[-1])  # uncertainty on logZ
    logprint(f"Backend: {resolved_backend}")
    logprint(f"log(Z) = {logZ} +- {logZerr}")
    logprint(f"Nr. of posterior samples: {len(posterior_samples)}")

    #::: make pretty titles for the plots
    labels, units = [], []
    for i, _l in enumerate(config.BASEMENT.fitlabels):
        labels.append(str(config.BASEMENT.fitlabels[i]))
        units.append(str(config.BASEMENT.fitunits[i]))

    results2 = deepcopy(results)  # results.copy() does not work anymore since dynesty 1.2
    params_median2, params_ll2, params_ul2 = (
        params_median.copy(),
        params_ll.copy(),
        params_ul.copy(),
    )  # params drawn form these posterior_samples; only needed for plots (subtract epoch offset)
    fittruths2 = config.BASEMENT.fittruths.copy()
    for companion in config.BASEMENT.settings["companions_all"]:
        if companion + "_epoch" in config.BASEMENT.fitkeys:
            ind = np.where(config.BASEMENT.fitkeys == companion + "_epoch")[0][0]
            results2["samples"][:, ind] -= int(
                params_median[companion + "_epoch"]
            )  # np.round(params_median[companion+'_epoch'],decimals=0)
            units[ind] = str(
                units[ind] + "-" + str(int(params_median[companion + "_epoch"])) + "d"
            )  # np.format_float_positional(params_median[companion+'_epoch'],0)+'d')
            fittruths2[ind] -= int(params_median[companion + "_epoch"])
            params_median2[companion + "_epoch"] -= int(params_median[companion + "_epoch"])

    for i, _l in enumerate(labels):
        if len(units[i].strip(" ")) > 0:
            labels[i] = str(labels[i] + " (" + units[i] + ")")

    #::: traceplot
    # dyplot.traceplot needs the dynesty-native fields ('logvol', 'niter', ...).
    # New unified-schema saves carry them through; legacy saves and non-dynesty
    # backends do not — fall back to the minimal in-house traceplot.
    cmap = truncate_colormap("Greys", minval=0.2, maxval=0.8, n=256)
    _dyplot_keys = ("logvol", "niter", "logl", "nlive")
    _can_dyplot = resolved_backend == "dynesty" and all(k in results2 for k in _dyplot_keys)

    def _safe_simple_trace():
        try:
            return _simple_traceplot(results2, labels=labels, truths=fittruths2)
        except (MemoryError, Exception) as _exc:
            logprint(f"! Fallback traceplot failed ({_exc}); skipping.")
            plt.close("all")
            return plt.subplots(1, 2, figsize=(6, 3))

    def _safe_simple_corner():
        try:
            return _corner_from_results(
                results2,
                labels=labels,
                truths=fittruths2,
                fontsize=fontsize,
                ndim=config.BASEMENT.ndim,
            )
        except (MemoryError, Exception) as _exc:
            logprint(f"! Fallback cornerplot failed ({_exc}); skipping.")
            plt.close("all")
            return plt.subplots(
                config.BASEMENT.ndim,
                config.BASEMENT.ndim,
                figsize=(min(2 * config.BASEMENT.ndim, 40), min(2 * config.BASEMENT.ndim, 40)),
            )

    if _can_dyplot:
        try:
            tfig, taxes = dyplot.traceplot(
                results2,
                labels=labels,
                quantiles=[0.16, 0.5, 0.84],
                truths=fittruths2,
                post_color="grey",
                trace_cmap=[cmap] * config.BASEMENT.ndim,
                trace_kwargs={"rasterized": True},
            )
        except (KeyError, ValueError) as _exc:
            logprint(f"! dyplot.traceplot failed ({_exc}); using fallback.")
            tfig, taxes = _safe_simple_trace()
    else:
        if resolved_backend == "dynesty":
            logprint(
                f"! Legacy save_ns.pickle.gz missing dynesty fields {list(_dyplot_keys)} — "
                "using fallback traceplot. Re-run ns_fit to restore "
                "native dynesty trace plots."
            )
        tfig, taxes = _safe_simple_trace()
    plt.tight_layout()

    #::: cornerplot
    # When ndim is large, drop nuisance baseline / error / stellar-var-GP
    # rows from the corner plot only — they make the plot unreadable AND
    # are almost never inspected. The full posterior is still in
    # results2 / posterior_samples for tables, deriver, etc.
    (corner_fitkeys, corner_samples, corner_labels, corner_truths, _dropped) = (
        _filter_nuisance_for_corner(
            config.BASEMENT.fitkeys,
            results2["samples"],
            labels,
            fittruths2,
        )
    )
    fontsize = np.min((24.0 + 0.5 * len(corner_fitkeys), 40))
    if _dropped:
        logprint(
            "! corner: hiding {n} nuisance params for readability "
            "(ndim {full} > {th}). Example: {ex}{more}".format(
                n=len(_dropped),
                full=config.BASEMENT.ndim,
                th=_CORNER_HIDE_NUISANCE_NDIM_THRESHOLD,
                ex=", ".join(_dropped[:3]),
                more=", ..." if len(_dropped) > 3 else "",
            )
        )
        # Filtered samples are equal-weight already (results2['samples']
        # are weighted, but the cost of resampling here is small and
        # corner.corner expects equal-weight rows). Use the post-filter
        # samples directly with _corner_from_samples.
        try:
            # Reweight if results2 has logwt (NS path); otherwise treat
            # as equal-weight (MCMC path is never here).
            try:
                logwt = np.asarray(results2["logwt"])
                logz_final = float(np.asarray(results2["logz"])[-1])
                w = np.exp(logwt - logz_final)
                w = w / w.sum()
                eq_samples = dyutils.resample_equal(corner_samples, w)
            except (KeyError, ValueError):
                eq_samples = corner_samples
            cfig, caxes = _corner_from_samples(
                eq_samples,
                corner_labels,
                corner_truths,
                fontsize=fontsize,
                ndim=len(corner_fitkeys),
            )
        except (MemoryError, Exception) as _exc:
            logprint(f"! filtered corner failed ({_exc}); skipping.")
            plt.close("all")
            cfig, caxes = plt.subplots(
                len(corner_fitkeys),
                len(corner_fitkeys),
                figsize=(
                    min(2 * len(corner_fitkeys), _MAX_CORNER_INCHES),
                    min(2 * len(corner_fitkeys), _MAX_CORNER_INCHES),
                ),
            )
    elif _can_dyplot:
        try:
            cfig, caxes = dyplot.cornerplot(
                results2,
                labels=labels,
                span=[0.997 for i in range(config.BASEMENT.ndim)],
                quantiles=[0.16, 0.5, 0.84],
                truths=fittruths2,
                hist_kwargs={"alpha": 0.25, "linewidth": 0, "histtype": "stepfilled"},
                label_kwargs={"fontsize": fontsize, "rotation": 45, "horizontalalignment": "right"},
            )
        except Exception as _exc:
            logprint(f"! dyplot.cornerplot failed ({_exc}); using corner.corner fallback.")
            cfig, caxes = _safe_simple_corner()
    else:
        cfig, caxes = _safe_simple_corner()

    #::: runplot
    #    rfig, raxes = dyplot.runplot(results)
    #    rfig.savefig( os.path.join(config.BASEMENT.outdir,'ns_run.jpg'), dpi=100, bbox_inches='tight' )
    #    plt.close(rfig)

    #::: set allesfitter titles and labels — trace iterates the FULL
    #::: fitkeys, corner iterates only the filtered subset (may be the
    #::: same when ndim is small enough that no nuisance was dropped).
    if len(config.BASEMENT.fitkeys) > 1:
        # trace titles — always one per full fitkey
        for i, key in enumerate(config.BASEMENT.fitkeys):
            value = round_tex(params_median2[key], params_ll2[key], params_ul2[key])
            ttitle = r"" + labels[i] + r"$=" + value + "$"
            taxes[i, 1].set_title(ttitle)
        # corner titles — only the kept (corner_fitkeys) rows
        for ci, key in enumerate(corner_fitkeys):
            value = round_tex(params_median2[str(key)], params_ll2[str(key)], params_ul2[str(key)])
            ctitle = r"" + corner_labels[ci] + "\n" + r"$=" + value + "$"
            caxes[ci, ci].set_title(
                ctitle, fontsize=fontsize, rotation=45, horizontalalignment="left"
            )
        # corner axis label / tick formatting
        for i in range(caxes.shape[0]):
            for j in range(caxes.shape[1]):
                caxes[i, j].xaxis.set_label_coords(0.5, -0.5)
                caxes[i, j].yaxis.set_label_coords(-0.5, 0.5)
                if i == (caxes.shape[0] - 1):
                    fmt = ScalarFormatter(useOffset=False)
                    caxes[i, j].xaxis.set_major_locator(MaxNLocator(nbins=3))
                    caxes[i, j].xaxis.set_major_formatter(fmt)
                if (i > 0) and (j == 0):
                    fmt = ScalarFormatter(useOffset=False)
                    caxes[i, j].yaxis.set_major_locator(MaxNLocator(nbins=3))
                    caxes[i, j].yaxis.set_major_formatter(fmt)
    else:
        # single-parameter fits: full == filtered, trace + corner both 1-D
        key = config.BASEMENT.fitkeys[0]
        value = round_tex(params_median2[key], params_ll2[key], params_ul2[key])
        ctitle = r"" + labels[0] + "\n" + r"$=" + value + "$"
        ttitle = r"" + labels[0] + r"$=" + value + "$"
        caxes[0, 0].set_title(ctitle)
        taxes[1].set_title(ttitle)
        caxes[0, 0].xaxis.set_label_coords(0.5, -0.5)
        caxes[0, 0].yaxis.set_label_coords(-0.5, 0.5)

    #::: save and close the trace- and cornerplot
    tfig.savefig(os.path.join(config.BASEMENT.outdir, "ns_trace.pdf"), bbox_inches="tight")
    plt.close(tfig)
    cfig.savefig(os.path.join(config.BASEMENT.outdir, "ns_corner.pdf"), bbox_inches="tight")
    plt.close(cfig)

    #::: save the tables
    save_table(posterior_samples, "ns")
    save_latex_table(posterior_samples, "ns")

    #::: write a LaTeX-formatted prior table derived from params.csv
    try:
        _priors_fp = write_priors_latex_table()
        if _priors_fp:
            logprint(f"\nWrote prior LaTeX table: {_priors_fp}")
    except Exception as _exc:
        logprint(f"\n! Could not write prior LaTeX table: {_exc}")

    #::: derive values (using stellar parameters from params_star.csv)
    deriver.derive(posterior_samples, "ns")

    #::: check the residuals
    for inst in config.BASEMENT.settings["inst_all"]:
        if inst in config.BASEMENT.settings["inst_phot"]:
            key = "flux"
        elif inst in config.BASEMENT.settings["inst_rv"]:
            key = "rv"
        elif inst in config.BASEMENT.settings["inst_rv2"]:
            key = "rv2"
        model = calculate_model(params_median, inst, key)
        baseline = calculate_baseline(params_median, inst, key)
        stellar_var = calculate_stellar_var(params_median, inst, key)
        residuals = config.BASEMENT.data[inst][key] - model - baseline - stellar_var
        residual_stats(residuals)

    #::: make top-down orbit plot (using stellar parameters from params_star.csv)
    try:
        params_star = np.genfromtxt(
            os.path.join(config.BASEMENT.datadir, "params_star.csv"),
            delimiter=",",
            names=True,
            dtype=None,
            encoding="utf-8",
            comments="#",
        )
        fig, ax = plot_top_down_view(params_median, params_star)
        fig.savefig(os.path.join(config.BASEMENT.outdir, "top_down_view.pdf"), bbox_inches="tight")
        plt.close(fig)
    except Exception as e:
        logprint(f"\nOrbital plots could not be produced: {e}")

    #::: plot TTV results (if wished for)
    if config.BASEMENT.settings["fit_ttvs"]:
        plot_ttv_results(params_median, params_ll, params_ul)

    #::: clean up
    logprint("\nDone. For all outputs, see", config.BASEMENT.outdir)

    #::: return a nerdy quote
    try:
        with open(os.path.join(os.path.dirname(__file__), "utils", "quotes.txt")) as dataset:
            return np.random.choice([l for l in dataset])
    except Exception:
        return "42"


def ns_derive(datadir):  # emergency function if matplotlib and Mac OSX crash
    from . import deriver

    posterior_samples = get_ns_posterior_samples(datadir, as_type="2d_array")
    deriver.derive(posterior_samples, "ns")


###############################################################################
#::: get NS samples (for top-level user)
###############################################################################
def get_ns_posterior_samples(datadir, Nsamples=None, as_type="dic"):
    import pickle

    config.init(datadir)

    try:
        f = gzip.GzipFile(os.path.join(datadir, "results", "save_ns.pickle.gz"), "rb")
        results = pickle.load(f)
        f.close()

    except Exception:
        with open(os.path.join(datadir, "results", "save_ns.pickle"), "rb") as f:
            results = pickle.load(f)

    return draw_ns_posterior_samples(results, Nsamples=Nsamples, as_type=as_type)


###############################################################################
#::: get NS params (for top-level user)
###############################################################################
def get_ns_params(datadir):
    posterior_samples = get_ns_posterior_samples(
        datadir, Nsamples=None, as_type="2d_array"
    )  # all weighted posterior_samples
    params_median, params_ll, params_ul = get_params_from_samples(
        posterior_samples
    )  # params drawn form these posterior_samples
    return params_median, params_ll, params_ul
