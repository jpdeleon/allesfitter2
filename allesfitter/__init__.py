"""
allesfitter - A global inference framework for photometry and radial velocity.

This package provides tools for joint modeling of photometric light curves
and radial velocity measurements of exoplanetary and stellar systems.
It supports various observational techniques including transit photometry,
radial velocity, phase curves, and TTV (transit timing variations) analysis.

Key Features:
    - MCMC and Nested Sampling parameter estimation
    - Gaussian Process regression for stellar variability
    - Limb darkening modeling
    - Flare and spot detection
    - Chromatic transit modeling
    - Injection-recovery simulations for transit detection

Typical Usage:
    >>> import allesfitter
    >>> allesfitter.GUI()  # Launch Jupyter-based GUI
    >>> # Or use command line:
    >>> # prepare_allesfit /path/to/datadir

Module Structure:
    - config: Configuration and initialization
    - basement: Core data and settings container
    - computer: Model computation engine
    - mcmc / nested_sampling: Bayesian inference methods
    - priors: Prior transformations and noise estimation
    - lightcurves: Limb darkening and light curve models
    - plotting: Visualization utilities
    - detection: Transit search and injection-recovery

For detailed documentation, visit https://www.allesfitter.com
"""

from __future__ import annotations

import gzip
import os
import pickle
import warnings
from shutil import copyfile
from typing import TYPE_CHECKING, Any, Optional, Union

warnings.filterwarnings("ignore", message=".*pkg_resources is deprecated.*")

# All third-party imports and submodule imports are lazy via __getattr__ below.
# This makes `import allesfitter` (~0.1s) instead of ~17s.
import importlib as _importlib

if TYPE_CHECKING:
    import emcee
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns

    from allesfitter import config, general_output, nested_sampling_output
    from allesfitter.computer import (
        calculate_baseline,
        calculate_model,
        calculate_stellar_var,
        calculate_yerr_w,
        update_params,
    )
    from allesfitter.general_output import draw_initial_guess_samples, get_labels
    from allesfitter.mcmc_output import (
        draw_mcmc_posterior_samples,
        draw_mcmc_posterior_samples_at_maximum_likelihood,
    )

_LAZY_NAMES: dict[str, tuple[str, str | None]] = {
    "np": ("numpy", None),
    "plt": ("matplotlib.pyplot", None),
    "sns": ("seaborn", None),
    "calculate_baseline": ("allesfitter.computer", "calculate_baseline"),
    "calculate_model": ("allesfitter.computer", "calculate_model"),
    "calculate_stellar_var": ("allesfitter.computer", "calculate_stellar_var"),
    "calculate_yerr_w": ("allesfitter.computer", "calculate_yerr_w"),
    "update_params": ("allesfitter.computer", "update_params"),
    "draw_initial_guess_samples": (
        "allesfitter.general_output",
        "draw_initial_guess_samples",
    ),
    "get_data": ("allesfitter.general_output", "get_data"),
    "get_labels": ("allesfitter.general_output", "get_labels"),
    "get_settings": ("allesfitter.general_output", "get_settings"),
    "show_initial_guess": ("allesfitter.general_output", "show_initial_guess"),
    "q_to_u": (
        "allesfitter.lightcurves",
        "translate_limb_darkening_from_q_to_u",
    ),
    "u_to_q": (
        "allesfitter.lightcurves",
        "translate_limb_darkening_from_u_to_q",
    ),
    "mcmc_fit": ("allesfitter.mcmc", "mcmc_fit"),
    "draw_mcmc_posterior_samples": (
        "allesfitter.mcmc_output",
        "draw_mcmc_posterior_samples",
    ),
    "draw_mcmc_posterior_samples_at_maximum_likelihood": (
        "allesfitter.mcmc_output",
        "draw_mcmc_posterior_samples_at_maximum_likelihood",
    ),
    "get_mcmc_posterior_samples": (
        "allesfitter.mcmc_output",
        "get_mcmc_posterior_samples",
    ),
    "mcmc_output": ("allesfitter.mcmc_output", "mcmc_output"),
    "ns_fit": ("allesfitter.nested_sampling", "ns_fit"),
    "get_ns_params": ("allesfitter.nested_sampling_output", "get_ns_params"),
    "get_ns_posterior_samples": (
        "allesfitter.nested_sampling_output",
        "get_ns_posterior_samples",
    ),
    "ns_derive": ("allesfitter.nested_sampling_output", "ns_derive"),
    "ns_output": ("allesfitter.nested_sampling_output", "ns_output"),
    "OptimizeResult": ("allesfitter.optimize", "OptimizeResult"),
    "optimize": ("allesfitter.optimize", "optimize"),
    "broken_xaxis_subplots": ("allesfitter.plot_utils", "broken_xaxis_subplots"),
    "detect_time_gaps": ("allesfitter.plot_utils", "detect_time_gaps"),
    "brokenplot": ("allesfitter.plotting", "brokenplot"),
    "brokenplot_csv": ("allesfitter.plotting", "brokenplot_csv"),
    "fullplot": ("allesfitter.plotting", "fullplot"),
    "fullplot_csv": ("allesfitter.plotting", "fullplot_csv"),
    "tessplot": ("allesfitter.plotting", "tessplot"),
    "tessplot_csv": ("allesfitter.plotting", "tessplot_csv"),
    "get_logZ": (
        "allesfitter.postprocessing.nested_sampling_compare_logZ",
        "get_logZ",
    ),
    "compare_logz": (
        "allesfitter.postprocessing.nested_sampling_compare_logZ",
        "compare_logz",
    ),
    "ns_plot_bayes_factors": (
        "allesfitter.postprocessing.nested_sampling_compare_logZ",
        "ns_plot_bayes_factors",
    ),
    "plot_histograms": ("allesfitter.postprocessing.plot_histograms", "plot_histograms"),
    "mcmc_plot_violins": ("allesfitter.postprocessing.plot_violins", "mcmc_plot_violins"),
    "ns_plot_violins": ("allesfitter.postprocessing.plot_violins", "ns_plot_violins"),
    "prepare_ttv_fit": ("allesfitter.prepare_ttv_fit", "prepare_ttv_fit"),
    "transform_priors": ("allesfitter.priors", "transform_priors"),
    "estimate_noise": ("allesfitter.priors.estimate_noise", "estimate_noise"),
    "estimate_noise_out_of_transit": (
        "allesfitter.priors.estimate_noise",
        "estimate_noise_out_of_transit",
    ),
    "get_run_log_path": ("allesfitter.run_logger", "get_log_path"),
    "log_event": ("allesfitter.run_logger", "log_event"),
    "log_run": ("allesfitter.run_logger", "log_run"),
    "run_log_tail": ("allesfitter.run_logger", "tail"),
}

_SEABORN_CONFIGURED = False


def __getattr__(name: str) -> Any:
    if name == "emcee":
        try:
            mod = _importlib.import_module("emcee")
        except ImportError:
            mod = None
        globals()["emcee"] = mod
        return mod

    if name in _LAZY_NAMES:
        module_path, attr = _LAZY_NAMES[name]
        mod = _importlib.import_module(module_path)

        if name == "sns":
            global _SEABORN_CONFIGURED
            if not _SEABORN_CONFIGURED:
                mod.set(
                    context="paper",
                    style="ticks",
                    palette="deep",
                    font="sans-serif",
                    font_scale=1.5,
                    color_codes=True,
                )
                mod.set_style({"xtick.direction": "in", "ytick.direction": "in"})
                mod.set_context(rc={"lines.markeredgewidth": 1})
                _SEABORN_CONFIGURED = True

        result = getattr(mod, attr) if attr else mod
        globals()[name] = result
        return result

    try:
        mod = _importlib.import_module(f"allesfitter.{name}")
        globals()[name] = mod
        return mod
    except ImportError:
        pass

    msg = f"module 'allesfitter' has no attribute {name!r}"
    raise AttributeError(msg)


def __dir__() -> list[str]:
    names = set(globals().keys())
    names.update(_LAZY_NAMES)
    names.add("emcee")
    return sorted(names)


# ::::: allesclass


class allesclass:
    def __init__(self, datadir, quiet=True):
        # ``allesfitter`` deliberately defers heavyweight imports.  Names
        # looked up by a function body do not go through this module's
        # ``__getattr__``, though, so load the dependencies used by the
        # constructor explicitly when an instance is requested.
        from . import config, general_output
        from .general_output import draw_initial_guess_samples, get_labels

        config.init(datadir, quiet=quiet)
        self.BASEMENT = config.BASEMENT
        self.fulldata = config.BASEMENT.fulldata
        self.data = config.BASEMENT.data
        self.settings = config.BASEMENT.settings
        self.labels = get_labels(datadir, as_type="dic")

        self.initial_guess_samples = draw_initial_guess_samples()
        self.initial_guess_params_median = general_output.get_params_from_samples(
            self.initial_guess_samples
        )[0]

        try:
            self.params_star = config.BASEMENT.params_star
        except Exception:
            pass

        try:
            self.external_priors = config.BASEMENT.external_priors
        except Exception:
            pass

        from .results import results_directory

        ns_outdir = results_directory(datadir, "ns")
        mcmc_outdir = results_directory(datadir, "mcmc")
        ns_save_path = os.path.join(ns_outdir, "save_ns.pickle.gz")
        mcmc_save_path = os.path.join(mcmc_outdir, "mcmc_save.h5")
        ns_available = os.path.exists(ns_save_path)
        mcmc_available = os.path.exists(mcmc_save_path)

        if ns_available and mcmc_available:
            print(
                f"Both NS and MCMC results found; using NS (priority): {ns_outdir}\n"
                f"  ignoring MCMC results at: {mcmc_outdir}"
            )
        elif ns_available:
            print(f"Using NS results: {ns_outdir}")
        elif mcmc_available:
            print(f"Using MCMC results: {mcmc_outdir}")

        if ns_available:
            from . import nested_sampling_output

            config.BASEMENT.outdir = ns_outdir
            f = gzip.GzipFile(os.path.join(config.BASEMENT.outdir, "save_ns.pickle.gz"), "rb")
            results = pickle.load(f)
            f.close()
            self.posterior_samples = nested_sampling_output.draw_ns_posterior_samples(results)
            self.posterior_params = nested_sampling_output.draw_ns_posterior_samples(
                results, as_type="dic"
            )
            self.posterior_params_median, self.posterior_params_ll, self.posterior_params_ul = (
                general_output.get_params_from_samples(self.posterior_samples)
            )

        elif mcmc_available:
            import emcee

            from .mcmc_output import (
                draw_mcmc_posterior_samples,
                draw_mcmc_posterior_samples_at_maximum_likelihood,
            )

            config.BASEMENT.outdir = mcmc_outdir
            copyfile(
                os.path.join(config.BASEMENT.outdir, "mcmc_save.h5"),
                os.path.join(config.BASEMENT.outdir, "mcmc_save_tmp.h5"),
            )
            reader = emcee.backends.HDFBackend(
                os.path.join(config.BASEMENT.outdir, "mcmc_save_tmp.h5"), read_only=True
            )
            self.posterior_samples = draw_mcmc_posterior_samples(reader)
            self.posterior_params = draw_mcmc_posterior_samples(reader, as_type="dic")
            self.posterior_samples_at_maximum_likelihood = (
                draw_mcmc_posterior_samples_at_maximum_likelihood(reader)
            )
            self.posterior_params_at_maximum_likelihood = (
                draw_mcmc_posterior_samples_at_maximum_likelihood(reader, as_type="dic")
            )
            self.posterior_params_median, self.posterior_params_ll, self.posterior_params_ul = (
                general_output.get_params_from_samples(self.posterior_samples)
            )
            os.remove(os.path.join(config.BASEMENT.outdir, "mcmc_save_tmp.h5"))

        else:
            warnings.warn("No NS nor MCMC save file detected.", stacklevel=2)

        if os.path.exists(os.path.join(config.BASEMENT.outdir, "ns_derived_samples.pickle")):
            self.posterior_derived_params = pickle.load(
                open(os.path.join(config.BASEMENT.outdir, "ns_derived_samples.pickle"), "rb")
            )

        elif os.path.exists(os.path.join(config.BASEMENT.outdir, "mcmc_derived_samples.pickle")):
            self.posterior_derived_params = pickle.load(
                open(os.path.join(config.BASEMENT.outdir, "mcmc_derived_samples.pickle"), "rb")
            )

        else:
            warnings.warn("No NS nor MCMC derived file detected.", stacklevel=2)

    def plot(
        self,
        inst,
        companion,
        style,
        fig=None,
        ax=None,
        mode="posterior",
        Nsamples=20,
        samples=None,
        dt=None,
        zoomwindow=8.0,
        force_binning=False,
        kwargs_data=None,
        kwargs_model=None,
        kwargs_ax=None,
    ):
        if ax is None:
            fig, ax = plt.subplots(1, 1)
        if (samples is None) and (Nsamples > 0) and (mode != "data"):
            if mode == "posterior":
                samples = self.posterior_samples[
                    np.random.randint(len(self.posterior_samples), size=Nsamples)
                ]
            elif mode == "initial_guess":
                samples = self.initial_guess_samples
            else:
                raise ValueError('Variable "mode" has to be "posterior" or "initial_guess".')
        general_output.plot_1(
            ax,
            samples,
            inst,
            companion,
            style,
            base=self,
            dt=dt,
            zoomwindow=zoomwindow,
            force_binning=force_binning,
            kwargs_data=kwargs_data,
            kwargs_ax=kwargs_ax,
        )
        return fig, ax

    def get_posterior_median_model(self, inst, key, xx=None, phased=False, settings=None):
        if not phased:
            return calculate_model(
                self.posterior_params_median, inst, key, xx=xx, settings=settings
            )
        elif phased:
            p = update_params(self.posterior_params_median, phased=True)
            return calculate_model(p, inst, key, xx=xx)

    def get_posterior_median_baseline(self, inst, key, xx=None, model=None, phased=False):
        if not phased:
            return calculate_baseline(self.posterior_params_median, inst, key, xx=xx, model=model)
        elif phased:
            raise ValueError("Not yet implemented.")

    def get_posterior_median_stellar_var(self, inst, key, xx=None, phased=False):
        if not phased:
            return calculate_stellar_var(self.posterior_params_median, inst, key, xx=xx)
        elif phased:
            raise ValueError("Not yet implemented.")

    def get_posterior_median_residuals(self, inst, key):
        model = self.get_posterior_median_model(inst, key)
        baseline = self.get_posterior_median_baseline(inst, key, model=model)
        stellar_var = self.get_posterior_median_stellar_var(inst, key)
        return self.data[inst][key] - model - baseline - stellar_var

    def get_posterior_median_yerr(self, inst, key):
        return calculate_yerr_w(self.posterior_params_median, inst, key)

    def get_posterior_at_maximum_likelihood_model(
        self, inst, key, xx=None, phased=False, settings=None
    ):
        if not phased:
            return calculate_model(
                self.posterior_params_at_maximum_likelihood, inst, key, xx=xx, settings=settings
            )
        elif phased:
            p = update_params(self.posterior_params_at_maximum_likelihood, phased=True)
            return calculate_model(p, inst, key, xx=xx)

    def get_posterior_at_maximum_likelihood_baseline(
        self, inst, key, xx=None, model=None, phased=False
    ):
        if not phased:
            return calculate_baseline(
                self.posterior_params_at_maximum_likelihood, inst, key, xx=xx, model=model
            )
        elif phased:
            raise ValueError("Not yet implemented.")

    def get_posterior_at_maximum_likelihood_stellar_var(self, inst, key, xx=None, phased=False):
        if not phased:
            return calculate_stellar_var(
                self.posterior_params_at_maximum_likelihood, inst, key, xx=xx
            )
        elif phased:
            raise ValueError("Not yet implemented.")

    def get_posterior_at_maximum_likelihood_residuals(self, inst, key):
        model = self.get_posterior_at_maximum_likelihood_model(inst, key)
        baseline = self.get_posterior_at_maximum_likelihood_baseline(inst, key, model=model)
        stellar_var = self.get_posterior_at_maximum_likelihood_stellar_var(inst, key)
        return self.data[inst][key] - model - baseline - stellar_var

    def get_posterior_at_maximum_likelihood_yerr(self, inst, key):
        return calculate_yerr_w(self.posterior_params_median, inst, key)

    def get_initial_guess_model(self, inst, key, xx=None, phased=False):
        if not phased:
            return calculate_model(self.initial_guess_params_median, inst, key, xx=xx)
        elif phased:
            raise ValueError("Not yet implemented.")

    def get_initial_guess_baseline(self, inst, key, xx=None, model=None, phased=False):
        if not phased:
            return calculate_baseline(
                self.initial_guess_params_median, inst, key, xx=xx, model=model
            )
        elif phased:
            raise ValueError("Not yet implemented.")

    def get_initial_guess_stellar_var(self, inst, key, xx=None, phased=False):
        if not phased:
            return calculate_stellar_var(self.initial_guess_params_median, inst, key, xx=xx)
        elif phased:
            raise ValueError("Not yet implemented.")

    def get_one_posterior_curve_set(self, inst, key, xx=None, sample_id=None, phased=False):
        if sample_id is None:
            sample_id = np.random.randint(self.posterior_samples.shape[0])
        buf = self.posterior_params_median.copy()
        for k in self.posterior_params:
            if not phased:
                buf[k] = self.posterior_params[k][sample_id]
            elif phased:
                p = update_params(self.posterior_params[k][sample_id], phased=True)
                buf[k] = p
        return (
            calculate_model(buf, inst, key, xx=xx),
            calculate_baseline(buf, inst, key, xx=xx),
            calculate_stellar_var(buf, inst, key, xx=xx),
        )

    def get_one_posterior_model(self, inst, key, xx=None, sample_id=None, phased=False):
        if sample_id is None:
            sample_id = np.random.randint(self.posterior_samples.shape[0])
        buf = self.posterior_params_median.copy()
        for k in self.posterior_params:
            if not phased:
                buf[k] = self.posterior_params[k][sample_id]
            elif phased:
                p = update_params(self.posterior_params[k][sample_id], phased=True)
                buf[k] = p
        return calculate_model(buf, inst, key, xx=xx)

    def get_one_posterior_baseline(self, inst, key, xx=None, sample_id=None, phased=False):
        if sample_id is None:
            sample_id = np.random.randint(self.posterior_samples.shape[0])
        buf = self.posterior_params_median.copy()
        for k in self.posterior_params:
            if not phased:
                buf[k] = self.posterior_params[k][sample_id]
            elif phased:
                p = update_params(self.posterior_params[k][sample_id], phased=True)
                buf[k] = p
        return calculate_baseline(buf, inst, key, xx=xx)

    def get_one_posterior_stellar_var(self, inst, key, xx=None, sample_id=None, phased=False):
        if sample_id is None:
            sample_id = np.random.randint(self.posterior_samples.shape[0])
        buf = self.posterior_params_median.copy()
        for k in self.posterior_params:
            if not phased:
                buf[k] = self.posterior_params[k][sample_id]
            elif phased:
                p = update_params(self.posterior_params[k][sample_id], phased=True)
                buf[k] = p
        return calculate_stellar_var(buf, key, xx=xx)


def GUI() -> None:
    allesfitter_path = os.path.dirname(os.path.realpath(__file__))
    os.system('jupyter notebook "' + os.path.join(allesfitter_path, "GUI.ipynb") + '"')


from ._version import __version__
