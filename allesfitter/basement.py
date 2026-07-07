#!/usr/bin/env python3
"""
Core data and settings container for everything.

The Basement class serves as the central data structure for allesfitter,
containing all observational data, model parameters, fitting configuration,
and derived quantities. It handles loading from CSV files, validation,
and initialization of all components needed for Bayesian inference.

Classes:
    Basement: Main container class for all fitting data and settings.

Module-Level Constants:
    DEFAULT_LD_CODES: Mapping of limb darkening law integer codes to strings.
"""


import collections
import fnmatch
import os
import sys
import warnings
from datetime import datetime
from multiprocessing import cpu_count
from typing import Any

import numpy as np

warnings.formatwarning = (
    lambda msg, *args, **kwargs: f"\n! WARNING:\n {msg}\ntype: {args[0]}, file: {args[1]}, line: {args[2]}\n"
)
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
warnings.filterwarnings("ignore", category=np.RankWarning)
#::: plotting settings
import seaborn as sns
from scipy.stats import truncnorm

#::: allesfitter modules
from ._version import __version__
from .exoworlds_rdx.lightcurves.index_transits import (
    get_first_epoch,
    get_tmid_observed_transits,
    index_eclipses,
    index_transits,
)
from .priors.simulate_PDF import simulate_PDF
from .utils.mcmc_move_translator import translate_str_to_move
from .validation import validate_params_settings
from .validation.physical_limits import eccentricity_error, lookup_limit

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


# Per-hyperparameter prefixes for all celerite baseline kernels supported by
# baseline_get_gp() in computer.py. Used to enumerate which params.csv rows are
# GP hyperparameters when aliasing share-group followers to their leader.
BASELINE_GP_HYPER_PREFIXES = (
    "baseline_gp_matern32_lnsigma_",
    "baseline_gp_matern32_lnrho_",
    "baseline_gp_sho_lnS0_",
    "baseline_gp_sho_lnQ_",
    "baseline_gp_sho_lnomega0_",
    "baseline_gp_real_lna_",
    "baseline_gp_real_lnc_",
    "baseline_gp_complex_lna_",
    "baseline_gp_complex_lnb_",
    "baseline_gp_complex_lnc_",
    "baseline_gp_complex_lnd_",
    "baseline_gp_offset_",
)

# REQUIRED hyperparameter row prefixes per supported baseline GP kernel.
# Used by load_params() to verify the share-group leader actually has the
# rows celerite will need before sampling starts. (baseline_gp_offset_ is
# optional and intentionally not listed here.)
BASELINE_GP_REQUIRED_HYPERS = {
    "sample_GP_Matern32": (
        "baseline_gp_matern32_lnsigma_",
        "baseline_gp_matern32_lnrho_",
    ),
    "sample_GP_SHO": (
        "baseline_gp_sho_lnS0_",
        "baseline_gp_sho_lnQ_",
        "baseline_gp_sho_lnomega0_",
    ),
    "sample_GP_real": (
        "baseline_gp_real_lna_",
        "baseline_gp_real_lnc_",
    ),
    "sample_GP_complex": (
        "baseline_gp_complex_lna_",
        "baseline_gp_complex_lnb_",
        "baseline_gp_complex_lnc_",
        "baseline_gp_complex_lnd_",
    ),
}


def _parse_inst_csv_header(path):
    """Return ``(column_names, is_hash_prefixed)`` from the first non-blank
    line of an instrument CSV, or ``(None, False)`` when the file has no
    recognizable header (legacy positional layout).

    Two header styles are accepted:

      1. **`#`-prefixed schema header** — the legacy "documenting"
         convention; column 0 must start with ``time`` (case-insensitive)
         and column 2 must end with ``_err``. Guards against ordinary
         comments like ``# my notes``.

      2. **Plain (pandas-style) header on the first row** — accepted
         when the line is NOT a `#`-comment AND its first token cannot
         be parsed as a float. The caller then knows to ``skip_header=1``
         in ``np.genfromtxt`` because the header isn't a `#`-comment.
         Column naming is lenient (no time/err token check) since users
         often name them ``BJD_TDB,Flux,Err,Airmass`` etc.

    Returns
    -------
    column_names : list[str] | None
    is_hash_prefixed : bool
        True for style (1) — genfromtxt skips it automatically.
        False for style (2) — caller must skip it explicitly.
    """
    try:
        with open(path) as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                if s.startswith("#"):
                    toks = [t.strip() for t in s.lstrip("#").strip().split(",")]
                    if (
                        len(toks) >= 3
                        and toks[0].lower().startswith("time")
                        and toks[2].lower().endswith("_err")
                    ):
                        return toks, True
                    return None, False
                # Non-`#` line — could be a plain header or a data row.
                toks = [t.strip() for t in s.split(",")]
                if len(toks) < 3:
                    return None, False
                try:
                    float(toks[0])
                    return None, False  # data row — no header
                except ValueError:
                    return toks, False  # plain string header
    except OSError:
        return None, False
    return None, False


def _load_inst_csv(path):
    """Load an instrument CSV with optional named covariate columns.

    Returns
    -------
    time, primary, primary_err : ndarray
        Required columns 0–2 (time, flux/rv, flux_err/rv_err).
    custom_series : ndarray
        Backward-compat alias: positional column-3 in legacy layout, or
        the first ancillary column in headered layout, or zeros otherwise.
    covariates : dict[str, ndarray]
        Named ancillary regressors. Empty in legacy layouts; populated
        in headered layouts with one entry per column past index 2.

    Supports three layouts:

      A. Headered: first non-blank line is e.g.
         ``#time,flux,flux_err,airmass,fwhm``. Column 3+ are stored in
         ``covariates`` keyed by header name. The first ancillary column
         is also aliased to ``custom_series``.
      B. Legacy 4-col: no header. ``custom_series`` = column 3;
         ``covariates`` is empty.
      C. Legacy 3-col: no header. ``custom_series`` = zeros;
         ``covariates`` is empty.
    """
    header, hash_prefixed = _parse_inst_csv_header(path)
    skip_header = 0 if (header is None or hash_prefixed) else 1
    arr = np.genfromtxt(
        path,
        delimiter=",",
        dtype=float,
        comments="#",
        skip_header=skip_header,
    ).T
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    n_cols = arr.shape[0]
    if n_cols < 3:
        raise ValueError(
            f"'{path}' has only {n_cols} numeric columns; need at least 3 "
            "(time, primary, primary_err)."
        )
    time, primary, primary_err = arr[0], arr[1], arr[2]
    covariates = {}
    if header is not None and len(header) >= 4 and n_cols >= 4:
        for i, name in enumerate(header[3:], start=3):
            if i < n_cols:
                covariates[name] = arr[i]
        # Legacy alias for users still selecting `_against,custom_series`.
        if covariates and "custom_series" not in covariates:
            first_name = header[3]
            if first_name in covariates:
                covariates["custom_series"] = covariates[first_name]
    elif header is None and n_cols >= 4:
        # Legacy positional 4th column.
        covariates["custom_series"] = arr[3]
    custom_series = covariates.get("custom_series", np.zeros_like(time))
    return time, primary, primary_err, custom_series, covariates


# Default number of integration sub-samples used when t_exp is auto-derived
# from `binning`. A binned point is the time-average over the bin width, so the
# transit model must be integrated over that window; t_exp alone (n_int == 1) is
# inert, so we seed a sensible n_int when the user left it unset.
_BINNING_DEFAULT_N_INT = 10


def _bin_phot_arrays(time, flux, flux_err, custom_series, covariates, dt):
    """Bin a photometric light curve to a fixed bin width ``dt`` (days).

    ``time``/``flux``/``flux_err`` are combined with an inverse-variance
    (error-weighted) mean, with the formal binned uncertainty
    ``1/sqrt(sum(1/err**2))``. ``custom_series`` and every covariate are
    mean-binned on the **same** bin grid so all per-instrument arrays stay
    row-aligned. Bins are half-open ``[t0 + k*dt, t0 + (k+1)*dt)``; empty bins
    are dropped.

    Returns ``(time, flux, flux_err, custom_series, covariates)`` with the same
    types as the inputs (``covariates`` is a new dict of binned arrays).
    """
    time = np.asarray(time, dtype=float)
    flux = np.asarray(flux, dtype=float)
    flux_err = np.asarray(flux_err, dtype=float)
    custom_series = np.asarray(custom_series, dtype=float)
    cov_items = [(k, np.asarray(v, dtype=float)) for k, v in covariates.items()]

    #::: group points by bin index, sorted so equal bins are contiguous
    bin_id = np.floor((time - time.min()) / dt).astype(np.int64)
    order = np.argsort(bin_id, kind="stable")
    bin_id = bin_id[order]
    time, flux, flux_err = time[order], flux[order], flux_err[order]
    custom_series = custom_series[order]
    cov_items = [(k, v[order]) for k, v in cov_items]
    groups = np.split(np.arange(len(bin_id)), np.flatnonzero(np.diff(bin_id)) + 1)

    bt, bf, bferr, bcustom = [], [], [], []
    bcov = {k: [] for k, _ in cov_items}
    for g in groups:
        w = 1.0 / np.square(flux_err[g])
        sw = np.sum(w)
        bt.append(np.mean(time[g]))
        bf.append(np.sum(flux[g] * w) / sw)
        bferr.append(1.0 / np.sqrt(sw))
        bcustom.append(np.mean(custom_series[g]))
        for k, v in cov_items:
            bcov[k].append(np.mean(v[g]))

    binned_covariates = {k: np.asarray(vals, dtype=float) for k, vals in bcov.items()}
    return (
        np.asarray(bt),
        np.asarray(bf),
        np.asarray(bferr),
        np.asarray(bcustom),
        binned_covariates,
    )


def _build_linear_design_matrix(data_inst, col_tokens, time_axis):
    """Build the per-instrument design matrix for ``sample_linear_multi``.

    Parameters
    ----------
    data_inst : dict
        Per-instrument data dict (``self.data[inst]``) — must contain
        a populated ``'covariates'`` dict.
    col_tokens : list[str]
        Column names declared by ``baseline_<key>_<inst>_cols``.
        Special tokens:

        - ``bias``    -> column of ones (NOT standardized)
        - any other   -> looked up in ``data_inst['covariates']``;
                         standardized to zero-mean, unit-variance
    time_axis : ndarray
        Used only for the length of the ``bias`` column.

    Returns
    -------
    X : ndarray of shape (n_samples, n_cols)
        Design matrix, columns ordered as in ``col_tokens``.
    cols_resolved : list[str]
        The same token list, returned for explicit traceability when
        the caller stores it on ``data_inst['design_matrix_cols']``.
    """
    covs = data_inst.get("covariates", {}) or {}
    cols = []
    for tok in col_tokens:
        t = tok.strip()
        if not t:
            continue
        if t == "bias":
            cols.append(np.ones_like(time_axis, dtype=float))
        elif t in covs:
            v = np.asarray(covs[t], dtype=float)
            mu = float(np.nanmean(v))
            sd = float(np.nanstd(v))
            if sd == 0.0:
                # constant column — keep zero-mean form so it contributes
                # nothing; user gets a polyfit-degenerate weight that
                # the prior pulls to zero.
                cols.append(v - mu)
            else:
                cols.append((v - mu) / sd)
        else:
            raise ValueError(
                f"baseline_<...>_cols: unknown token '{t}'. Known options: "
                f"'bias' or any of {sorted(covs.keys())}."
            )
    X = np.column_stack(cols) if cols else np.zeros((len(time_axis), 0))
    return X, [t.strip() for t in col_tokens if t.strip()]


###############################################################################
#::: 'Basement' class, which contains all the data, settings, etc.
###############################################################################
class Basement:
    """The 'Basement' class contains all the data, settings, etc.

    This is the core data container for everything, holding:
        - All observational data (photometry, radial velocity)
        - Model parameters and their priors
        - Fitting configuration and settings
        - Derived stellar parameters
        - External priors (e.g., stellar density)

    Attributes
    ----------
    datadir : str
        Path to the data directory.
    outdir : str
        Path to the output directory where results are saved.
    settings : dict
        Configuration settings loaded from settings.csv.
    params : OrderedDict
        Model parameters loaded from params.csv.
    data : dict
        Nested dict of observational data by instrument and type.
    fulldata : dict
        Complete data including all metadata.
    labels : dict
        Parameter labels for plotting and output.
    external_priors : dict
        External priors such as stellar density constraints.

    Examples
    --------
    >>> from allesfitter import config
    >>> config.init('/path/to/datadir')
    >>> base = config.BASEMENT
    >>> print(base.settings['inst_phot'])
    """

    ###############################################################################
    #::: init
    ###############################################################################
    def __init__(self, datadir: str, quiet: bool = False) -> None:
        """Initialize the Basement with data from a directory.

        Parameters
        ----------
        datadir : str
            The working directory for allesfitter.
            Must contain all the data files:
            - settings.csv: Fitting configuration
            - params.csv: Initial parameter guesses
            - Data files: Light curves and/or RV measurements
            Output directories and files will also be created inside datadir.
        quiet : bool, optional
            If True, suppress verbose output during initialization (default: False).

        Returns
        -------
        None

        Raises
        ------
        FileNotFoundError
            If required input files are missing.
        ValueError
            If settings contain invalid values or conflicts.

        Notes
        -----
        This method:
            1. Creates output directory structure
            2. Loads and validates settings from settings.csv
            3. Loads and validates parameters from params.csv
            4. Loads observational data from CSV files
            5. Applies epoch shifting if configured
            6. Sets up TTV fitting if enabled
            7. Loads stellar priors if available
        """

        print("Filling the Basement")

        self.quiet = quiet
        self.now = f"{datetime.now():%Y-%m-%d_%H-%M-%S}"
        self.datadir = datadir
        self.outdir = os.path.join(datadir, "results")
        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir)

        print("")
        self.logprint("\nallesfitter version")
        self.logprint("---------------------")
        self.logprint("v" + __version__)

        self.load_settings()
        self.load_params()
        #::: structural sanity + cross-file consistency for params.csv/settings.csv.
        #::: Raises ConfigError (a ValueError) on unambiguous mistakes before any
        #::: data is loaded. See allesfitter.validation.config_checks.
        validate_params_settings(self.datadir)
        self.load_data()
        self.validate_baseline_against_covariates()
        self.synthesize_linear_multi_params()

        if self.settings["shift_epoch"]:
            try:
                self.change_epoch()
            except Exception as err:
                #::: If the user explicitly asked for shift_epoch in settings.csv,
                #::: a failed calculation is a real problem they need to see, not
                #::: something to silently swallow. Only downgrade to a warning
                #::: when shift_epoch was defaulted (e.g. a single transit with no
                #::: period given, where shifting is meaningless anyway).
                if "shift_epoch" in self._settings_raw_keys:
                    raise ValueError(
                        "shift_epoch=True was set in settings.csv, but the epoch "
                        "shift failed: " + str(err) + "\nCheck that a valid period and "
                        "epoch are given for every companion, or set shift_epoch "
                        "to False."
                    ) from err
                warnings.warn(
                    "\nCould not shift epoch (you can peacefully ignore this warning if no period was given)\n",
                    stacklevel=2,
                )

        if self.settings["fit_ttvs"]:
            self.setup_ttv_fit()

        #::: external priors (e.g. stellar density)
        self.external_priors = {}
        self.load_stellar_priors()

        #::: if baseline model == sample_GP, set up a GP object for photometric data
        #        self.setup_GPs()

        #::: translate limb darkening codes from params.csv (int) into str for ellc
        self.ldcode_to_ldstr = [
            "none",  #   :  0,
            "lin",  #    :  1,
            "quad",  #   :  2,
            "sing",  #   :  3,
            "claret",  # :  4,
            "log",  #  :  5,
            "sqrt",  #  :  6,
            "exp",  #    :  7,
            "power-2",  #:  8,
            "mugrid",
        ]  # : -1

        #::: check if the input is consistent
        for inst in self.settings["inst_phot"]:
            key = "flux"
            if (
                self.settings["baseline_" + key + "_" + inst]
                in ["sample_GP_Matern32", "sample_GP_SHO"]
            ) & (self.settings["error_" + key + "_" + inst] != "sample"):
                raise ValueError(
                    "If you want to use "
                    + self.settings["baseline_" + key + "_" + inst]
                    + ", you will want to sample the jitters, too!"
                )

    ###############################################################################
    #::: print function that prints into console and logfile at the same time
    ###############################################################################
    def logprint(self, *text: Any) -> None:
        """Print to both console and logfile.

        Outputs text to stdout and appends to a timestamped log file
        in the output directory.

        Parameters
        ----------
        *text : Any
            Any objects to be printed (like the built-in print function).

        Returns
        -------
        None

        Notes
        -----
        If quiet=True was set during initialization, this method does nothing.
        """
        if not self.quiet:
            print(*text)
            original = sys.stdout
            with open(os.path.join(self.outdir, "logfile_" + self.now + ".log"), "a") as f:
                sys.stdout = f
                print(*text)
            sys.stdout = original
        else:
            pass

    ###############################################################################
    #::: helper to get bandpass for an instrument
    ###############################################################################
    def get_bandpass(self, inst):
        """
        Return bandpass for an instrument, or None if achromatic.

        Respects an explicit ``chromatic,False`` override in settings.csv:
        when the user forces achromatic mode (even with a multi-label
        bandpass row), this returns ``None`` for every instrument so
        downstream callers fall through to achromatic parameter naming
        (``b_rr`` instead of ``b_rr_<bp>``, per-inst LDC keys instead
        of per-bandpass).

        Parameters
        ----------
        inst : str
            Instrument name (e.g., 'tess', 'kepler')

        Returns
        -------
        str or None
            Bandpass name if chromatic, None if achromatic.
        """
        if not self.settings.get("chromatic", False):
            return None
        return self.settings.get("bandpass", {}).get(inst)

    ###############################################################################
    #::: get_rr_key: helper to get the correct rr key for a companion/instrument
    ###############################################################################
    def get_rr_key(self, companion, inst):
        """
        Return the parameter key for radius ratio (rr) for a given companion and instrument.

        Parameters
        ----------
        companion : str
            Companion name (e.g., 'b', 'c')
        inst : str
            Instrument name (e.g., 'tess', 'kepler')

        Returns
        -------
        str
            Parameter key: 'b_rr' for achromatic, 'b_rr_tess' for chromatic.

        Notes
        -----
        The suffix follows the parser's naming, which keys rr by bandpass as
        soon as a ``bandpass`` row is present — even when ``chromatic`` is
        auto-detected False because all instruments share one bandpass (two
        instruments on 'tess' both resolve to the shared scalar
        ``b_rr_tess``). This deliberately does NOT go through
        :meth:`get_bandpass`, which suppresses the suffix whenever the
        chromatic flag is False. The single exception is an *explicit*
        ``chromatic,False`` override in settings.csv, where the parser
        collapses rr to a single achromatic ``b_rr`` (LDC keys still stay
        per-bandpass); that case is honoured here.
        """
        bandpass_dict = self.settings.get("bandpass") or {}
        if not bandpass_dict:
            return f"{companion}_rr"
        explicit_achromatic = self.settings.get(
            "chromatic_explicit", False
        ) and not self.settings.get("chromatic", False)
        if explicit_achromatic:
            return f"{companion}_rr"
        bandpass = bandpass_dict.get(inst)
        if bandpass:
            return f"{companion}_rr_{bandpass}"
        return f"{companion}_rr"

    ###############################################################################
    #::: get_ldc_key: helper to get the correct LDC scalar key for a role/n/inst
    ###############################################################################
    def get_ldc_bandpass(self, inst):
        """
        Return the LDC suffix bandpass for an instrument.

        UNLIKE :meth:`get_bandpass`, this method consults the raw
        ``settings['bandpass']`` dict directly and **ignores the
        chromatic flag**. Limb darkening depends only on wavelength, so
        LDC keys are always keyed by bandpass when a bandpass row is
        present — even under an explicit ``chromatic,False`` override.

        Returns the bandpass label when known, else ``None``.
        """
        return (self.settings.get("bandpass") or {}).get(inst)

    def get_ldc_key(self, role, n, inst, space="u"):
        """
        Return the per-coefficient LDC key for a role, coefficient index,
        and instrument.

        The suffix is the instrument's bandpass when settings.csv carries
        a bandpass row (so multiple instruments sharing a bandpass share
        a single LDC scalar) — independent of the ``chromatic`` flag,
        because limb darkening is a function of wavelength only. Falls
        back to the instrument name when no bandpass row is provided.

        Parameters
        ----------
        role : str
            'host' or a companion identifier ('b', 'c', ...).
        n : int
            Coefficient index (1..4).
        inst : str
            Instrument name (e.g., 'tess', 'kepler', 'tess_pdcsap').
        space : str, optional
            'u' (default) or 'q'.

        Returns
        -------
        str
            For example ``host_ldc_u1_tess`` (bandpass='tess') or
            ``host_ldc_u1_tess_pdcsap`` (no bandpass row, fallback).
        """
        if space not in ("u", "q"):
            raise ValueError(f"space must be 'u' or 'q', got {space!r}")
        bandpass = self.get_ldc_bandpass(inst)
        suffix = bandpass if bandpass else inst
        return f"{role}_ldc_{space}{n}_{suffix}"

    ###############################################################################
    #::: load settings
    ###############################################################################
    def load_settings(self):
        """
        For the full list of options see www.allesfitter.com
        """

        def set_bool(text):
            if text.lower() in ["true", "1"]:
                return True
            else:
                return False

        def is_empty_or_none(key):
            return (
                (key not in self.settings)
                or (str(self.settings[key]).lower() == "none")
                or (len(self.settings[key]) == 0)
            )

        def unique(array):
            uniq, index = np.unique(array, return_index=True)
            return uniq[index.argsort()]

        rows = np.genfromtxt(
            os.path.join(self.datadir, "settings.csv"), dtype=None, encoding="utf-8", delimiter=","
        )

        #::: make backwards compatible
        for i, row in enumerate(rows):
            #            print(row)
            name = row[0]
            if name[:7] == "planets":
                rows[i][0] = "companions" + name[7:]
                warnings.warn(
                    "You are using outdated keywords. Automatically renaming "
                    + name
                    + " ---> "
                    + rows[i][0]
                    + ". Please fix this before the Duolingo owl comes to get you.",
                    stacklevel=2,
                )  # , category=DeprecationWarning)
            if name[:6] == "ld_law":
                rows[i][0] = "host_ld_law" + name[6:]
                warnings.warn(
                    "You are using outdated keywords. Automatically renaming "
                    + name
                    + " ---> "
                    + rows[i][0]
                    + ". Please fix this before the Duolingo owl comes to get you.",
                    stacklevel=2,
                )  # , category=DeprecationWarning)

        #        self.settings = {r[0]:r[1] for r in rows}
        self.settings = collections.OrderedDict(
            [("user-given:", "")] + [(r[0], r[1]) for r in rows] + [("automatically set:", "")]
        )

        # Snapshot the set of keys that came from the user's settings.csv,
        # BEFORE any defaults are filled in below. Used by the nested-sampling
        # dispatcher to tell the user when a backend-relevant knob was left
        # implicit (defaulted) vs. explicitly set in the CSV.
        self._settings_raw_keys = {r[0] for r in rows}

        #::: check for unrecognized settings keys
        valid_settings_keys = {
            "companions_phot",
            "companions_rv",
            "companions_all",
            "inst_phot",
            "inst_rv",
            "inst_rv2",
            "inst_all",
            "time_format",
            "multiprocess",
            "multiprocess_cores",
            "fast_fit",
            "fast_fit_width",
            "secondary_eclipse",
            "phase_curve",
            "phase_curve_style",
            "shift_epoch",
            "mask_transit",
            "inst_for_b_epoch",
            "inst_for_c_epoch",
            "inst_for_d_epoch",
            "inst_for_e_epoch",
            "inst_for_f_epoch",
            "inst_for_g_epoch",
            "mcmc_nwalkers",
            "mcmc_total_steps",
            "mcmc_burn_steps",
            "mcmc_thin_by",
            "mcmc_pre_run_loops",
            "mcmc_pre_run_steps",
            "mcmc_moves",
            "ns_modus",
            "ns_nlive",
            "ns_bound",
            "ns_sample",
            "ns_tol",
            "ns_backend",
            "un_min_ess",
            "un_max_iters",
            "bandpass",
            "chromatic",
            "fit_ttvs",
            "exact_grav",
            "use_host_density_prior",
            "use_tidal_eccentricity_prior",
            "N_flares",
            "N_spots",
            "bumps_persistent",
            "t_exp_tess",
            "t_exp_kepler",
            "t_exp_n_int_tess",
            "t_exp_n_int_kepler",
            "print_progress",
            "quiet",
            "flux_min_raw",
            "flux_max_raw",
            "flux_min_flat",
            "flux_max_flat",
            "binning",
            "baseline_share_flux",
            "baseline_share_rv",
            "baseline_share_rv2",
        }
        valid_settings_prefixes = [
            "host_ld_law_",
            "host_ld_space_",
            "host_grid_",
            "host_shape_",
            "host_flux_weighted_",
            "host_rotfac_",
            "host_hf_",
            "host_bfac_",
            "host_heat_",
            "host_lambda_",
            "host_N_spots_",
            "b_ld_law_",
            "b_ld_space_",
            "b_grid_",
            "b_shape_",
            "b_flux_weighted_",
            "b_N_spots_",
            "c_ld_law_",
            "c_ld_space_",
            "c_grid_",
            "c_shape_",
            "c_flux_weighted_",
            "c_N_spots_",
            "d_ld_law_",
            "d_ld_space_",
            "d_grid_",
            "d_shape_",
            "d_flux_weighted_",
            "d_N_spots_",
            "e_ld_law_",
            "e_ld_space_",
            "e_grid_",
            "e_shape_",
            "e_flux_weighted_",
            "e_N_spots_",
            "f_ld_law_",
            "f_ld_space_",
            "f_grid_",
            "f_shape_",
            "f_flux_weighted_",
            "f_N_spots_",
            "g_ld_law_",
            "g_ld_space_",
            "g_grid_",
            "g_shape_",
            "g_flux_weighted_",
            "g_N_spots_",
            "baseline_flux_",
            "baseline_rv_",
            "baseline_rv2_",
            "error_flux_",
            "error_rv_",
            "error_rv2_",
            "binning_",
            "N_bumps_",
            "t_exp_",
            "stellar_var_flux",
            "stellar_var_rv",
        ]

        #::: Every key in settings.csv is an explicit user choice. An
        #::: unrecognized key (typo or deprecated keyword) is otherwise silently
        #::: ignored, so the fit quietly runs with a default instead of the
        #::: user's intent. Collect all offenders and fail loudly with one
        #::: comprehensive message rather than warning and continuing.
        unrecognized_settings = [
            key
            for key in self.settings
            if key not in ["user-given:", "automatically set:"]
            and key not in valid_settings_keys
            and not any(key.startswith(prefix) for prefix in valid_settings_prefixes)
        ]
        if unrecognized_settings:
            msg = (
                "The following setting keys in settings.csv are not "
                "recognized, likely due to a typo or a deprecated keyword:\n"
            )
            for key in unrecognized_settings:
                msg += "  - " + key + "\n"
            msg += (
                "Fix or remove these keys. Refusing to silently ignore "
                "settings that would otherwise fall back to defaults."
            )
            raise ValueError(msg)

        #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        #::: Main settings
        #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        if "time_format" not in self.settings:
            self.settings["time_format"] = "BJD_TDB"

        for key in ["companions_phot", "companions_rv", "inst_phot", "inst_rv", "inst_rv2"]:
            if key not in self.settings:
                self.settings[key] = []
            elif len(self.settings[key]):
                self.settings[key] = str(self.settings[key]).split(" ")
            else:
                self.settings[key] = []

        self.settings["companions_all"] = list(
            np.unique(self.settings["companions_phot"] + self.settings["companions_rv"])
        )  # sorted by b, c, d...
        self.settings["inst_all"] = list(
            unique(
                self.settings["inst_phot"] + self.settings["inst_rv"] + self.settings["inst_rv2"]
            )
        )  # sorted like user input

        if len(self.settings["inst_phot"]) == 0 and len(self.settings["companions_phot"]) > 0:
            raise ValueError(
                "No photometric instrument is selected, but photometric companions are given."
            )
        if len(self.settings["inst_rv"]) == 0 and len(self.settings["companions_rv"]) > 0:
            raise ValueError("No RV instrument is selected, but RV companions are given.")

        #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        #::: Bandpass settings (for chromatic transit modeling)
        #::: If not specified → achromatic (all instruments share same rr)
        #::: If specified with multiple unique values → chromatic
        #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        if "bandpass" not in self.settings or is_empty_or_none("bandpass"):
            self.settings["bandpass"] = {}  # empty = achromatic
        else:
            bp_list = str(self.settings["bandpass"]).split()
            n_inst = len(self.settings["inst_phot"])
            if len(bp_list) != n_inst:
                raise ValueError(
                    "settings.csv 'bandpass' has {n_bp} entries but inst_phot has "
                    "{n_inst} entries; each photometric instrument needs an explicit "
                    "bandpass label (repeat the same label to keep instruments achromatic). "
                    "Got bandpass={bp_list!r}, inst_phot={inst_phot!r}.".format(
                        n_bp=len(bp_list),
                        n_inst=n_inst,
                        bp_list=bp_list,
                        inst_phot=self.settings["inst_phot"],
                    )
                )
            self.settings["bandpass"] = {
                inst: bp_list[i] for i, inst in enumerate(self.settings["inst_phot"])
            }

        # Determine if chromatic (multiple unique bandpasses) or achromatic
        unique_bandpasses = set(self.settings["bandpass"].values())
        # Honour an explicit user-set `chromatic,True/False` in settings.csv;
        # otherwise auto-detect from the number of unique bandpass labels
        # (legacy behaviour). The override lets users force a single shared
        # b_rr across multiple bandpasses (e.g. low-S/N MuSCAT 4-band fits)
        # while keeping the bandpass row for plot labels / per-band LDCs.
        if "chromatic" in self._settings_raw_keys:
            self.settings["chromatic"] = set_bool(str(self.settings["chromatic"]))
            self.settings["chromatic_explicit"] = True
        else:
            self.settings["chromatic"] = len(unique_bandpasses) > 1
            self.settings["chromatic_explicit"] = False

        #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        #::: Validate per-instrument settings suffixes
        #::: Catches typos like ``host_ld_law_tess,quad`` when no instrument is
        #::: named "tess" (e.g. inst_phot=['tglc120_s90', ...] + bandpass='tess').
        #::: Without this guard, the default at ~685 silently sets the LD law to
        #::: None for every real instrument, ldc_1=None reaches ellc, and the
        #::: q1/q2 values in params.csv have zero effect on the transit shape.
        #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        _per_inst_prefixes = (
            "host_ld_law_",
            "host_ld_space_",
            "host_grid_",
            "host_shape_",
            "host_flux_weighted_",
            "host_rotfac_",
            "host_hf_",
            "host_bfac_",
            "host_heat_",
            "host_N_spots_",
        )
        for _comp in ("b", "c", "d", "e", "f", "g"):
            _per_inst_prefixes = _per_inst_prefixes + (
                _comp + "_ld_law_",
                _comp + "_ld_space_",
                _comp + "_grid_",
                _comp + "_shape_",
                _comp + "_flux_weighted_",
                _comp + "_N_spots_",
            )
        _per_inst_prefixes = _per_inst_prefixes + (
            "baseline_flux_",
            "baseline_rv_",
            "baseline_rv2_",
            "error_flux_",
            "error_rv_",
            "error_rv2_",
            "t_exp_",
            "t_exp_n_int_",
            "stellar_var_flux_",
            "stellar_var_rv_",
        )
        _known_insts = set(self.settings["inst_all"])
        _known_bands = unique_bandpasses
        #::: host_ld_law is keyed by BANDPASS, not instrument: it carries the
        #::: q1/q2 limb-darkening coefficients, which depend on the photometric
        #::: band (true even when chromatic=False). It is resolved per-bandpass
        #::: downstream (see the host_ld_law defaulting below), so its suffix may
        #::: be a bandpass label; instruments are still accepted for back-compat.
        #::: Other per-instrument prefixes (ld_space, companion LD, baseline,
        #::: error, t_exp, ...) are NOT resolved per-bandpass, so their suffix must
        #::: still be an instrument to avoid an accepted-but-ignored setting.
        _bandpass_keyed_prefixes = ("host_ld_law_",)
        #::: Match the most specific (longest) prefix first, so that e.g.
        #::: ``t_exp_n_int_<inst>`` is not mis-parsed as ``t_exp_`` + ``n_int_<inst>``.
        _per_inst_prefixes = tuple(sorted(_per_inst_prefixes, key=len, reverse=True))
        _orphans = []
        for _key in list(self.settings.keys()):
            if _key in ("user-given:", "automatically set:"):
                continue
            for _pref in _per_inst_prefixes:
                if _key.startswith(_pref):
                    _suffix = _key[len(_pref) :]
                    # Strip trailing modifier tokens that turn an inst-keyed
                    # setting into an inst-keyed sub-setting:
                    #   baseline_flux_<inst>_against → covariate axis name
                    #   baseline_flux_<inst>_args    → extra args (e.g. spline s)
                    _stripped = _suffix
                    for _mod in ("_against", "_args", "_cols"):
                        if _stripped.endswith(_mod):
                            _stripped = _stripped[: -len(_mod)]
                            break
                    if _pref in _bandpass_keyed_prefixes:
                        _valid = _known_insts | set(_known_bands)
                    else:
                        _valid = _known_insts
                    if _stripped and _stripped not in _valid:
                        _orphans.append((_key, _pref, _suffix))
                    break
        if _orphans:
            _hint_lines = []
            for _k, _p, _s in _orphans:
                _msg = f"  '{_k}': suffix '{_s}' is not in inst_phot+inst_rv+inst_rv2 ({sorted(_known_insts)})"
                if _s in _known_bands:
                    _affected = sorted(i for i, b in self.settings["bandpass"].items() if b == _s)
                    _msg += (
                        f"  [hint: '{_s}' is a BANDPASS label, not an instrument. "
                        f"Repeat this row once per instrument using that bandpass: {_affected}]"
                    )
                _hint_lines.append(_msg)
            raise ValueError(
                "settings.csv contains per-instrument keys whose suffix is "
                "not a known instrument name. The suffix must match an entry "
                "of inst_phot/inst_rv/inst_rv2 (NOT a bandpass label).\n"
                "Offending rows:\n" + "\n".join(_hint_lines)
            )

        #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        #::: General settings
        #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        if "print_progress" in self.settings:
            self.settings["print_progress"] = set_bool(self.settings["print_progress"])
        else:
            self.settings["print_progress"] = True

        #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        #::: Epoch settings
        #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        if "shift_epoch" in self.settings:
            self.settings["shift_epoch"] = set_bool(self.settings["shift_epoch"])
        else:
            self.settings["shift_epoch"] = True

        for companion in self.settings["companions_all"]:
            if "inst_for_" + companion + "_epoch" not in self.settings:
                self.settings["inst_for_" + companion + "_epoch"] = "all"

            if self.settings["inst_for_" + companion + "_epoch"] in ["all", "none"]:
                self.settings["inst_for_" + companion + "_epoch"] = self.settings["inst_all"]
            else:
                if len(self.settings["inst_for_" + companion + "_epoch"]):
                    self.settings["inst_for_" + companion + "_epoch"] = str(
                        self.settings["inst_for_" + companion + "_epoch"]
                    ).split(" ")
                else:
                    self.settings["inst_for_" + companion + "_epoch"] = []

        #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        #::: Multiprocess settings
        #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        self.settings["multiprocess"] = set_bool(self.settings["multiprocess"])

        if "multiprocess_cores" not in self.settings.keys():
            self.settings["multiprocess_cores"] = cpu_count() - 1
        elif self.settings["multiprocess_cores"] == "all":
            self.settings["multiprocess_cores"] = cpu_count() - 1
        else:
            self.settings["multiprocess_cores"] = int(self.settings["multiprocess_cores"])
            if self.settings["multiprocess_cores"] == cpu_count():
                string = (
                    "You are pushing your luck: you want to run on "
                    + str(self.settings["multiprocess_cores"])
                    + " cores, but your computer has only "
                    + str(cpu_count())
                    + ". I will let you go through with it this time..."
                )
                warnings.warn(string, stacklevel=2)
            if self.settings["multiprocess_cores"] > cpu_count():
                string = (
                    "Oops, you want to run on "
                    + str(self.settings["multiprocess_cores"])
                    + " cores, but your computer has only "
                    + str(cpu_count())
                    + ". Maybe try running on "
                    + str(cpu_count() - 1)
                    + "?"
                )

        #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        #::: Phase variations
        #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        if ("phase_variations" in self.settings.keys()) and len(self.settings["phase_variations"]):
            warnings.warn(
                'You are using outdated keywords. Automatically renaming "phase_variations" ---> "phase_curve".'
                + ". Please fix this before the Duolingo owl comes to get you.",
                stacklevel=2,
            )
            self.settings["phase_curve"] = self.settings["phase_variations"]

        if ("phase_curve" in self.settings.keys()) and len(self.settings["phase_curve"]):
            self.settings["phase_curve"] = set_bool(self.settings["phase_curve"])
            if self.settings["phase_curve"]:
                # self.logprint('The user set phase_curve==True. Automatically set fast_fit=False and secondary_eclispe=True, and overwrite other settings.')
                self.settings["fast_fit"] = "False"
                self.settings["secondary_eclipse"] = "True"
        else:
            self.settings["phase_curve"] = False

        #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        #::: Fast fit
        #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        if ("fast_fit" in self.settings.keys()) and len(self.settings["fast_fit"]):
            self.settings["fast_fit"] = set_bool(self.settings["fast_fit"])
        else:
            self.settings["fast_fit"] = False

        if ("fast_fit_width" in self.settings.keys()) and len(self.settings["fast_fit_width"]):
            self.settings["fast_fit_width"] = float(self.settings["fast_fit_width"])
        else:
            self.settings["fast_fit_width"] = 8.0 / 24.0

        #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        #::: Mask transit
        #::: When True, the in-transit points are removed from each photometric
        #::: light curve at load time (the inverse of fast_fit). This lets the
        #::: user model `stellar_var` or spots on the out-of-transit data alone,
        #::: without the transit biasing the fit. The transit window half-width
        #::: is `fast_fit_width` (the same window fast_fit uses). All transit
        #::: parameters must be fixed (fit=0) in params.csv; load_params()
        #::: enforces this, since fitting a transit that is no longer in the data
        #::: is meaningless.
        #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        if ("mask_transit" in self.settings.keys()) and len(self.settings["mask_transit"]):
            self.settings["mask_transit"] = set_bool(self.settings["mask_transit"])
        else:
            self.settings["mask_transit"] = False

        #::: fast_fit (keep only in-transit) and mask_transit (drop in-transit)
        #::: are exact opposites; enabling both at once is a contradiction.
        if self.settings["mask_transit"] and self.settings["fast_fit"]:
            raise ValueError(
                "mask_transit==True and fast_fit==True are mutually exclusive in "
                "settings.csv: fast_fit keeps only the in-transit data, while "
                "mask_transit removes it. Set one of them to False."
            )

        #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        #::: Raw-flux outlier clip bounds (applied at load time in load_data())
        #::: If `flux_min_raw` / `flux_max_raw` are present, rows with flux
        #::: outside [flux_min_raw, flux_max_raw] are dropped from each
        #::: photometric instrument's data before any further reduction.
        #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        if ("flux_min_raw" in self.settings.keys()) and not is_empty_or_none("flux_min_raw"):
            self.settings["flux_min_raw"] = float(self.settings["flux_min_raw"])
        else:
            self.settings["flux_min_raw"] = None

        if ("flux_max_raw" in self.settings.keys()) and not is_empty_or_none("flux_max_raw"):
            self.settings["flux_max_raw"] = float(self.settings["flux_max_raw"])
        else:
            self.settings["flux_max_raw"] = None

        if (
            self.settings["flux_min_raw"] is not None
            and self.settings["flux_max_raw"] is not None
            and self.settings["flux_min_raw"] >= self.settings["flux_max_raw"]
        ):
            raise ValueError(
                "flux_min_raw ({}) must be < flux_max_raw ({}) in settings.csv.".format(
                    self.settings["flux_min_raw"], self.settings["flux_max_raw"]
                )
            )

        #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        #::: Binning of the input photometric light curves (load_data()).
        #::: None (default) = no binning. A positive float gives the bin width
        #::: in DAYS (e.g. 0.0208333 ≈ 30 min). Parsed defensively here so a
        #::: malformed/<=0 value is reported by validation (config_checks) with
        #::: a clean message rather than a raw float() traceback.
        #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        if ("binning" in self.settings.keys()) and not is_empty_or_none("binning"):
            try:
                self.settings["binning"] = float(self.settings["binning"])
            except (TypeError, ValueError):
                pass  # leave the raw string; check_binning_value() raises a clean error
        else:
            self.settings["binning"] = None

        #::: Per-instrument binning overrides: `binning_<inst>` bins only that
        #::: instrument's light curve, leaving the rest at the global `binning`.
        #::: Parsed defensively (float-or-None) for every photometric instrument;
        #::: invalid values are reported by check_binning_value(). A key present
        #::: but empty/None disables binning for that instrument (overrides the
        #::: global value), so absence (fall back to global) and explicit-None
        #::: (force off) stay distinguishable.
        for inst in self.settings["inst_phot"]:
            key = "binning_" + inst
            if (key in self.settings.keys()) and not is_empty_or_none(key):
                try:
                    self.settings[key] = float(self.settings[key])
                except (TypeError, ValueError):
                    pass  # leave raw; check_binning_value() raises a clean error
            elif key in self.settings.keys():
                self.settings[key] = None

        #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        #::: Flattened-flux outlier clip bounds (applied by config.init() AFTER
        #::: BASEMENT is constructed, since computing the trend requires
        #::: calculate_baseline which reads config.BASEMENT.{settings,data}).
        #::: Bounds are interpreted on the detrended flux:
        #:::     flat = flux - baseline(initial-guess) - stellar_var(initial-guess)
        #::: Rows with `flat` outside [flux_min_flat, flux_max_flat] are dropped.
        #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        if ("flux_min_flat" in self.settings.keys()) and not is_empty_or_none("flux_min_flat"):
            self.settings["flux_min_flat"] = float(self.settings["flux_min_flat"])
        else:
            self.settings["flux_min_flat"] = None

        if ("flux_max_flat" in self.settings.keys()) and not is_empty_or_none("flux_max_flat"):
            self.settings["flux_max_flat"] = float(self.settings["flux_max_flat"])
        else:
            self.settings["flux_max_flat"] = None

        if (
            self.settings["flux_min_flat"] is not None
            and self.settings["flux_max_flat"] is not None
            and self.settings["flux_min_flat"] >= self.settings["flux_max_flat"]
        ):
            raise ValueError(
                "flux_min_flat ({}) must be < flux_max_flat ({}) in settings.csv.".format(
                    self.settings["flux_min_flat"], self.settings["flux_max_flat"]
                )
            )

        #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        #::: Host stellar density prior
        #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        if "use_host_density_prior" in self.settings:
            self.settings["use_host_density_prior"] = set_bool(
                self.settings["use_host_density_prior"]
            )
        else:
            self.settings["use_host_density_prior"] = True

        #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        #::: Host stellar density prior
        #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        if "use_tidal_eccentricity_prior" in self.settings:
            self.settings["use_tidal_eccentricity_prior"] = set_bool(
                self.settings["use_tidal_eccentricity_prior"]
            )
        else:
            self.settings["use_tidal_eccentricity_prior"] = False

        #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        #::: TTVs
        #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        if ("fit_ttvs" in self.settings.keys()) and len(self.settings["fit_ttvs"]):
            self.settings["fit_ttvs"] = set_bool(self.settings["fit_ttvs"])
            if (self.settings["fit_ttvs"]) and (not self.settings["fast_fit"]):
                raise ValueError(
                    "fit_ttvs==True, but fast_fit==False."
                    + "Currently, you can only fit for TTVs if fast_fit==True."
                    + "Please choose different settings."
                )
        else:
            self.settings["fit_ttvs"] = False

        #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        #::: Secondary eclipse
        #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        if ("secondary_eclipse" in self.settings.keys()) and len(
            self.settings["secondary_eclipse"]
        ):
            self.settings["secondary_eclipse"] = set_bool(self.settings["secondary_eclipse"])
        else:
            self.settings["secondary_eclipse"] = False

        #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        #::: MCMC settings
        #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        if "mcmc_pre_run_loops" not in self.settings:
            self.settings["mcmc_pre_run_loops"] = 0
        if "mcmc_pre_run_steps" not in self.settings:
            self.settings["mcmc_pre_run_steps"] = 0
        if "mcmc_nwalkers" not in self.settings:
            self.settings["mcmc_nwalkers"] = 100
        if "mcmc_total_steps" not in self.settings:
            self.settings["mcmc_total_steps"] = 2000
        if "mcmc_burn_steps" not in self.settings:
            self.settings["mcmc_burn_steps"] = 1000
        if "mcmc_thin_by" not in self.settings:
            self.settings["mcmc_thin_by"] = 1
        if "mcmc_moves" not in self.settings:
            self.settings["mcmc_moves"] = "DEMove"

        #::: make sure these are integers
        for key in [
            "mcmc_nwalkers",
            "mcmc_pre_run_loops",
            "mcmc_pre_run_steps",
            "mcmc_total_steps",
            "mcmc_burn_steps",
            "mcmc_thin_by",
        ]:
            self.settings[key] = int(self.settings[key])

        #::: luser proof
        if self.settings["mcmc_total_steps"] <= self.settings["mcmc_burn_steps"]:
            raise ValueError(
                "Your setting for mcmc_total_steps must be larger than mcmc_burn_steps (check your settings.csv)."
            )

        #::: translate the mcmc_move string into a list of emcee commands
        self.settings["mcmc_moves"] = translate_str_to_move(self.settings["mcmc_moves"])

        # N_evaluation_samples = int( 1. * self.settings['mcmc_nwalkers'] * (self.settings['mcmc_total_steps']-self.settings['mcmc_burn_steps']) / self.settings['mcmc_thin_by'] )
        # self.logprint('\nAnticipating ' + str(N_evaluation_samples) + 'MCMC evaluation samples.\n')
        # if N_evaluation_samples>200000:
        #     answer = input('It seems like you are asking for ' + str(N_evaluation_samples) + 'MCMC evaluation samples (calculated as mcmc_nwalkers * (mcmc_total_steps-mcmc_burn_steps) / mcmc_thin_by).'+\
        #                    'That is an aweful lot of samples.'+\
        #                    'What do you want to do?\n'+\
        #                    '1 : continue at any sacrifice\n'+\
        #                    '2 : abort and increase the mcmc_thin_by parameter in settings.csv (do not do this if you continued an old run!)\n')
        #     if answer==1:
        #         pass
        #     else:
        #         raise ValueError('User aborted the run.')

        #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        #::: Nested Sampling settings
        #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        if "ns_modus" not in self.settings:
            self.settings["ns_modus"] = "static"
        if "ns_nlive" not in self.settings:
            self.settings["ns_nlive"] = 500
        if "ns_bound" not in self.settings:
            self.settings["ns_bound"] = "single"
        if "ns_sample" not in self.settings:
            self.settings["ns_sample"] = "rwalk"
        if "ns_tol" not in self.settings:
            self.settings["ns_tol"] = 0.01
        if "ns_backend" not in self.settings:
            self.settings["ns_backend"] = "dynesty"
        # UltraNest-specific knobs (ignored when ns_backend != 'ultranest')
        if "un_min_ess" not in self.settings:
            self.settings["un_min_ess"] = 400
        if "un_max_iters" not in self.settings:
            self.settings["un_max_iters"] = None

        self.settings["ns_nlive"] = int(self.settings["ns_nlive"])
        self.settings["ns_tol"] = float(self.settings["ns_tol"])
        self.settings["ns_backend"] = str(self.settings["ns_backend"]).strip().lower()
        self.settings["un_min_ess"] = int(self.settings["un_min_ess"])
        if self.settings["un_max_iters"] in (None, "", "None", "none"):
            self.settings["un_max_iters"] = None
        else:
            self.settings["un_max_iters"] = int(self.settings["un_max_iters"])

        #        if self.settings['ns_sample'] == 'auto':
        #            if self.ndim < 10:
        #                self.settings['ns_sample'] = 'unif'
        #                print('Using ns_sample=="unif".')
        #            elif 10 <= self.ndim <= 20:
        #                self.settings['ns_sample'] = 'rwalk'
        #                print('Using ns_sample=="rwalk".')
        #            else:
        #                self.settings['ns_sample'] = 'slice'
        #                print('Using ns_sample=="slice".')

        #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        #::: host & companion grids, limb darkening laws, shapes, etc.
        #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        for companion in self.settings["companions_all"]:
            for inst in self.settings["inst_all"]:

                if "host_grid_" + inst not in self.settings:
                    self.settings["host_grid_" + inst] = "default"

                if companion + "_grid_" + inst not in self.settings:
                    self.settings[companion + "_grid_" + inst] = "default"

                # host_ld_law is keyed by BANDPASS: q1/q2 depend on the photometric
                # band, not the detector (true even when chromatic=False). Resolution
                # order for each instrument: an explicit per-instrument key wins
                # (back-compat); otherwise the bandpass-keyed value; otherwise the
                # 'quad' default. A prior None default silently disabled limb
                # darkening, making host_ldc_q1/q2 in params.csv appear to have no
                # effect; write host_ld_law_<band|inst>,none to disable LD explicitly.
                _band = self.settings.get("bandpass", {}).get(inst)
                _h_key = "host_ld_law_" + inst
                _h_band_key = ("host_ld_law_" + _band) if _band else None
                if (_h_key in self.settings) and (len(str(self.settings[_h_key])) > 0):
                    pass  # explicit per-instrument value already in place
                elif (
                    _h_band_key
                    and (_h_band_key in self.settings)
                    and (len(str(self.settings[_h_band_key])) > 0)
                ):
                    self.settings[_h_key] = self.settings[_h_band_key]
                else:
                    self.settings[_h_key] = "quad"
                if str(self.settings[_h_key]).lower() == "none":
                    self.settings[_h_key] = None

                if is_empty_or_none(companion + "_ld_law_" + inst):
                    self.settings[companion + "_ld_law_" + inst] = None

                if is_empty_or_none("host_ld_space_" + inst):
                    self.settings["host_ld_space_" + inst] = "q"

                if is_empty_or_none(companion + "_ld_space_" + inst):
                    self.settings[companion + "_ld_space_" + inst] = "q"

                if "host_shape_" + inst not in self.settings:
                    self.settings["host_shape_" + inst] = "sphere"

                if companion + "_shape_" + inst not in self.settings:
                    self.settings[companion + "_shape_" + inst] = "sphere"

        for companion in self.settings["companions_rv"]:
            for inst in list(self.settings["inst_rv"]) + list(self.settings["inst_rv2"]):
                if companion + "_flux_weighted_" + inst in self.settings:
                    self.settings[companion + "_flux_weighted_" + inst] = set_bool(
                        self.settings[companion + "_flux_weighted_" + inst]
                    )
                else:
                    self.settings[companion + "_flux_weighted_" + inst] = False

        if "exact_grav" in self.settings:
            self.settings["exact_grav"] = set_bool(self.settings["exact_grav"])
        else:
            self.settings["exact_grav"] = False

        #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        #::: Phase curve styles
        #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        if is_empty_or_none("phase_curve_style"):
            self.settings["phase_curve_style"] = None
        if self.settings["phase_curve_style"] not in [
            None,
            "sine_series",
            "sine_physical",
            "ellc_physical",
            "GP",
        ]:
            raise ValueError(
                "The setting 'phase_curve_style' must be one of [None, 'sine_series', 'sine_physical', 'ellc_physical', 'GP'], but was '"
                + str(self.settings["phase_curve_style"])
                + "'."
            )
        if (self.settings["phase_curve"] is True) and (self.settings["phase_curve_style"] is None):
            raise ValueError(
                "You chose 'phase_curve=True' but did not select a 'phase_curve_style'; please select one of ['sine_series', 'sine_physical', 'ellc_physical', 'GP']."
            )
        if (self.settings["phase_curve"] is False) and (
            self.settings["phase_curve_style"]
            in ["sine_series", "sine_physical", "ellc_physical", "GP"]
        ):
            raise ValueError(
                "You chose 'phase_curve=False' but also selected a 'phase_curve_style'; please double check and set 'phase_curve_style=None' (or remove it)."
            )

        #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        #::: Stellar variability
        #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        for key in ["flux", "rv", "rv2"]:
            if (
                ("stellar_var_" + key not in self.settings)
                or (self.settings["stellar_var_" + key] is None)
                or (self.settings["stellar_var_" + key].lower() == "none")
            ):
                self.settings["stellar_var_" + key] = "none"

        #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        #::: Baselines
        #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        for inst in self.settings["inst_all"]:
            if inst in self.settings["inst_phot"]:
                key = "flux"
            elif inst in self.settings["inst_rv"]:
                key = "rv"
            elif inst in self.settings["inst_rv2"]:
                key = "rv2"

            #::: default
            #::: if the user gives no baseline, the default is 'none'
            if "baseline_" + key + "_" + inst not in self.settings:
                self.settings["baseline_" + key + "_" + inst] = "none"

            #::: hybrid_spline
            #::: the user can define the s value directly, e.g. as "hybrid_spline 0.001"
            #::: this block serves to split up this input and assign it to the right functions
            if ("hybrid_spline" in self.settings["baseline_" + key + "_" + inst]) and (
                len(self.settings["baseline_" + key + "_" + inst].split(" ")) > 1
            ):
                s = self.settings["baseline_" + key + "_" + inst].split(" ")[1]
                self.settings["baseline_" + key + "_" + inst] = "hybrid_spline_s"
                self.settings["baseline_" + key + "_" + inst + "_args"] = (
                    s  # any arguments coming with this baseline (for future expandability; for now it is simply the s-value)
                )

            #::: sample_GP
            #::: make sure the keywords are updated correctly
            elif self.settings["baseline_" + key + "_" + inst] == "sample_GP":
                warnings.warn(
                    "You are using outdated keywords. Automatically renaming sample_GP ---> sample_GP_Matern32."
                    + ". Please update your files before the Duolingo owl comes to get you.",
                    stacklevel=2,
                )  # , category=DeprecationWarning)
                self.settings["baseline_" + key + "_" + inst] = "sample_GP_Matern32"

            #::: baseline against custom series
            #::: allows the user to fit a baseline not vs. time but vs. a chosen custom series
            if "baseline_" + key + "_" + inst + "_against" not in self.settings:
                self.settings["baseline_" + key + "_" + inst + "_against"] = "time"
            # Accept 'time', the legacy 'custom_series' alias, or any
            # named covariate. The named-covariate existence check needs
            # the data CSV to be loaded first, so it runs after load_data
            # in validate_baseline_against_covariates().
            _av = str(self.settings["baseline_" + key + "_" + inst + "_against"]).strip()
            if not _av:
                raise ValueError(
                    f"The setting 'baseline_{key}_{inst}_against' must be 'time', "
                    "'custom_series', or a named covariate column from "
                    f"{inst}.csv. Got an empty string."
                )

        #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        #::: Baseline share groups (joint celerite GP across instruments)
        #::: settings.csv format:
        #:::   baseline_share_flux,muscat_g:muscat_r:muscat_i:muscat_z
        #::: or multiple groups separated by spaces:
        #:::   baseline_share_flux,g1lead:g1f1 g2lead:g2f1:g2f2
        #::: All members of a group must be in inst_<key2>, must use the same
        #::: sample_GP_* baseline type as the leader (first member), and may
        #::: appear in at most one group. Followers inherit the leader's GP
        #::: hyperparameters via the existing coupled_with mechanism (see
        #::: load_params), so plotting/output paths see consistent draws.
        #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        _supported_share_gps = {
            "sample_GP_Matern32",
            "sample_GP_SHO",
            "sample_GP_real",
            "sample_GP_complex",
        }
        for key, key2 in zip(["flux", "rv", "rv2"], ["inst_phot", "inst_rv", "inst_rv2"]):
            skey = "baseline_share_" + key
            raw = self.settings.get(skey, None)
            groups = []
            if raw is not None and str(raw).strip() not in ("", "none", "None"):
                for group_str in str(raw).split():
                    members = [m for m in group_str.split(":") if m]
                    if len(members) >= 1:
                        groups.append(members)
            leader_of = {}
            followers_of = {}
            for members in groups:
                # Duplicate members within a single group are a typo, not a
                # legal aliasing — refuse before they shadow real instruments.
                if len(set(members)) != len(members):
                    raise ValueError(
                        "baseline_share_{k}: group '{g}' contains duplicate "
                        "members.".format(k=key, g=":".join(members))
                    )
                # A singleton group shares nothing (joint-GP path collapses to
                # the legacy per-inst path). Almost always a user typo where
                # the colon-separated follower list got dropped — warn so they
                # notice instead of silently getting independent-GP behaviour.
                if len(members) == 1:
                    warnings.warn(
                        f"baseline_share_{key}: group '{members[0]}' has only one member "
                        "and shares nothing. Did you forget the colon-"
                        "separated follower list?",
                        stacklevel=2,
                    )

                leader = members[0]
                followers = members[1:]

                # The leader of group A cannot also appear (as leader OR
                # follower) in group B — that would make the alias graph
                # ambiguous. The follower-check below covers leader-as-
                # follower; this check covers leader-as-duplicate-leader.
                if leader in leader_of:
                    raise ValueError(
                        f"baseline_share_{key}: '{leader}' appears as a leader in "
                        "more than one group (or as a follower in an earlier "
                        "group)."
                    )

                if leader not in self.settings[key2]:
                    raise ValueError(
                        f"baseline_share_{key}: leader '{leader}' is not in {key2}. "
                        f"Every member of a share group must be listed in {key2}."
                    )
                leader_base = self.settings.get("baseline_" + key + "_" + leader, "none")
                if leader_base not in _supported_share_gps:
                    raise ValueError(
                        f"baseline_share_{key}: leader '{leader}' has baseline "
                        f"'{leader_base}' which is not a supported GP kernel for sharing "
                        f"(must be one of {sorted(_supported_share_gps)})."
                    )
                if (
                    self.settings.get("baseline_" + key + "_" + leader + "_against", "time")
                    != "time"
                ):
                    raise ValueError(
                        f"baseline_share_{key}: leader '{leader}' must use "
                        f"baseline_{key}_{leader}_against=time for a joint GP to be "
                        "well-defined across instruments."
                    )
                for f in followers:
                    if f not in self.settings[key2]:
                        raise ValueError(
                            f"baseline_share_{key}: follower '{f}' is not in " f"{key2}."
                        )
                    if f in leader_of:
                        raise ValueError(
                            f"baseline_share_{key}: '{f}' appears in more than " "one share group."
                        )
                    f_base = self.settings.get("baseline_" + key + "_" + f, "none")
                    if f_base not in ("none", leader_base):
                        raise ValueError(
                            f"baseline_share_{key}: follower '{f}' has "
                            f"baseline_{key}_{f}={f_base} but leader '{leader}' has "
                            f"{leader_base}. Followers must inherit the leader's GP "
                            "type (leave blank or matching)."
                        )
                    # If the follower's `_against` is explicitly set in the
                    # user's settings.csv (not just the default), it must
                    # match the leader's. Silently overriding the user would
                    # hide a real configuration mistake.
                    f_against_key = "baseline_" + key + "_" + f + "_against"
                    if f_against_key in self._settings_raw_keys:
                        f_against = self.settings.get(f_against_key, "time")
                        if f_against != "time":
                            raise ValueError(
                                f"baseline_share_{key}: follower '{f}' has "
                                f"{f_against_key}={f_against} but the share group requires "
                                "'time'. Remove the explicit setting."
                            )
                    # propagate leader's baseline settings to follower
                    self.settings["baseline_" + key + "_" + f] = leader_base
                    self.settings["baseline_" + key + "_" + f + "_against"] = "time"
                    args_key = "baseline_" + key + "_" + leader + "_args"
                    if args_key in self.settings:
                        self.settings["baseline_" + key + "_" + f + "_args"] = self.settings[
                            args_key
                        ]
                    leader_of[f] = leader
                    followers_of.setdefault(leader, []).append(f)
                leader_of.setdefault(leader, leader)
            self.settings[skey + "_groups"] = groups
            self.settings[skey + "_leader_of"] = leader_of
            self.settings[skey + "_followers_of"] = followers_of

        #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        #::: Errors
        #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        for inst in self.settings["inst_all"]:
            if inst in self.settings["inst_phot"]:
                key = "flux"
            elif inst in self.settings["inst_rv"]:
                key = "rv"
            elif inst in self.settings["inst_rv2"]:
                key = "rv2"
            if "error_" + key + "_" + inst not in self.settings:
                self.settings["error_" + key + "_" + inst] = "sample"

        # for inst in self.settings['inst_phot']:
        #     for key in ['flux']:
        #         if 'error_'+key+'_'+inst not in self.settings:
        #             self.settings['error_'+key+'_'+inst] = 'sample'

        # for inst in self.settings['inst_rv']:
        #     for key in ['rv']:
        #         if 'error_'+key+'_'+inst not in self.settings:
        #             self.settings['error_'+key+'_'+inst] = 'sample'

        #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        #::: Color plot
        #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        if "color_plot" not in self.settings.keys():
            self.settings["color_plot"] = False

        #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        #::: Companion colors
        #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        for i, companion in enumerate(self.settings["companions_all"]):
            self.settings[companion + "_color"] = sns.color_palette()[i]

        #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        #::: Plot zoom window
        #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        if "zoom_window" not in self.settings:
            self.settings["zoom_window"] = (
                8.0 / 24.0
            )  # 8h window around transit/eclipse midpoint by Default
        else:
            self.settings["zoom_window"] = float(self.settings["zoom_window"])

        #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        #::: Exposure time interpolation
        #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        for inst in self.settings["inst_all"]:
            #::: if t_exp is given
            if "t_exp_" + inst in self.settings.keys() and len(self.settings["t_exp_" + inst]):
                t_exp = self.settings["t_exp_" + inst].split(" ")
                # if float
                if len(t_exp) == 1:
                    self.settings["t_exp_" + inst] = float(t_exp[0])
                # if array
                else:
                    self.settings["t_exp_" + inst] = np.array([float(t) for t in t_exp])
            #::: if not given / given as an empty field
            else:
                self.settings["t_exp_" + inst] = None

            #::: if t_exp_n_int is given
            if (
                "t_exp_" + inst in self.settings
                and "t_exp_n_int_" + inst in self.settings
                and len(self.settings["t_exp_n_int_" + inst])
            ):

                self.settings["t_exp_n_int_" + inst] = int(self.settings["t_exp_n_int_" + inst])
                if self.settings["t_exp_n_int_" + inst] < 1:
                    raise ValueError(
                        '"t_exp_n_int_'
                        + inst
                        + '" must be >= 1, but is given as '
                        + str(self.settings["t_exp_n_int_" + inst])
                        + " in params.csv"
                    )
            else:
                self.settings["t_exp_n_int_" + inst] = None

        #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        #::: Auto-derive t_exp from binning (photometry only).
        #::: A binned point is the time-average of the flux over the bin width,
        #::: so the transit model must be integrated over that window. When an
        #::: instrument is binned (global `binning` or a per-instrument
        #::: `binning_<inst>` override) and the user did NOT set t_exp for it, we
        #::: set t_exp_<inst> = bin width and seed t_exp_n_int_<inst> (which is
        #::: inert at 1) so the supersampling actually takes effect. An explicit
        #::: user t_exp is never overwritten; a mismatch is surfaced instead.
        #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        for inst in self.settings["inst_phot"]:
            _bkey = "binning_" + inst
            _eff_binning = (
                self.settings[_bkey] if _bkey in self.settings else self.settings["binning"]
            )
            if _eff_binning is None:
                continue
            if self.settings["t_exp_" + inst] is None:
                self.settings["t_exp_" + inst] = float(_eff_binning)
                self.logprint(
                    "\nAuto-set t_exp_"
                    + inst
                    + " = "
                    + repr(float(_eff_binning))
                    + ' d to match binning of "'
                    + inst
                    + '" (binned points are '
                    + "time-averages over the bin width; the model is integrated "
                    + "over t_exp)."
                )
                if self.settings["t_exp_n_int_" + inst] is None:
                    self.settings["t_exp_n_int_" + inst] = _BINNING_DEFAULT_N_INT
                    self.logprint(
                        "Auto-set t_exp_n_int_"
                        + inst
                        + " = "
                        + str(_BINNING_DEFAULT_N_INT)
                        + " (sub-samples for the t_exp integration; set it "
                        + "explicitly in settings.csv to override)."
                    )
            elif np.isscalar(self.settings["t_exp_" + inst]) and not np.isclose(
                self.settings["t_exp_" + inst], _eff_binning
            ):
                warnings.warn(
                    'binning of "'
                    + inst
                    + '" is '
                    + repr(float(_eff_binning))
                    + " d but t_exp_"
                    + inst
                    + " is explicitly set to "
                    + repr(self.settings["t_exp_" + inst])
                    + " d. Keeping your "
                    + "t_exp_"
                    + inst
                    + "; for binned data t_exp usually equals the "
                    + "bin width. Remove t_exp_"
                    + inst
                    + " to auto-match the binning.",
                    stacklevel=2,
                )

        #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        #::: Number of spots
        #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        for inst in self.settings["inst_all"]:
            if "host_N_spots_" + inst in self.settings and len(
                self.settings["host_N_spots_" + inst]
            ):
                self.settings["host_N_spots_" + inst] = int(self.settings["host_N_spots_" + inst])
            else:
                self.settings["host_N_spots_" + inst] = 0

            for companion in self.settings["companions_all"]:
                if companion + "_N_spots_" + inst in self.settings:
                    self.settings[companion + "_N_spots_" + inst] = int(
                        self.settings[companion + "_N_spots_" + inst]
                    )
                else:
                    self.settings[companion + "_N_spots_" + inst] = 0

        #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        #::: Number of flares
        #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        if "N_flares" in self.settings and len(self.settings["N_flares"]) > 0:
            self.settings["N_flares"] = int(self.settings["N_flares"])
        else:
            self.settings["N_flares"] = 0

        #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        #::: Number of bumps (starspot-crossing events), keyed by bandpass
        #::: (falls back to per-instrument when no bandpass row is given). The
        #::: crossing depth is wavelength-dependent, exactly like limb darkening,
        #::: so two instruments sharing a bandpass share one N_bumps_<bp> count.
        #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        #::: iterate over unique suffixes so a shared bandpass is parsed once
        #::: (otherwise the second visit would len() an already-int value)
        _bump_bandpass = self.settings.get("bandpass") or {}
        _bump_suffixes = {(_bump_bandpass.get(inst) or inst) for inst in self.settings["inst_phot"]}
        for _suffix in _bump_suffixes:
            _key = "N_bumps_" + _suffix
            if _key in self.settings and len(self.settings[_key]):
                self.settings[_key] = int(self.settings[_key])
            else:
                self.settings[_key] = 0

        #::: bumps_persistent: repeat each bump at tpeak + n*period (every transit)
        if ("bumps_persistent" in self.settings.keys()) and len(self.settings["bumps_persistent"]):
            self.settings["bumps_persistent"] = set_bool(self.settings["bumps_persistent"])
        else:
            self.settings["bumps_persistent"] = False

    ###############################################################################
    #::: load params
    ###############################################################################
    def load_params(self):
        """
        For the full list of options see www.allesfitter.com
        """

        # ==========================================================================
        #::: load params.csv
        # ==========================================================================
        buf = np.genfromtxt(
            os.path.join(self.datadir, "params.csv"),
            delimiter=",",
            comments="#",
            dtype=None,
            encoding="utf-8",
            names=True,
        )

        # ==========================================================================
        #::: luser-proof: reject duplicate parameter rows
        #::: numpy.genfromtxt is happy with duplicate names but downstream
        #::: dict-assembly silently last-wins, which corrupts chromatic configs
        #::: edited by hand (e.g. two rows for b_rr_tess with different priors).
        # ==========================================================================
        _names = list(np.atleast_1d(buf["name"]))
        _stripped = [
            n.strip() for n in _names if str(n).strip() not in ("user-given:", "automatically set:")
        ]
        _seen = {}
        for _n in _stripped:
            _seen[_n] = _seen.get(_n, 0) + 1
        _dups = sorted(k for k, v in _seen.items() if v > 1)
        if _dups:
            raise ValueError(
                "params.csv contains duplicate rows for: "
                + ", ".join(_dups)
                + ". Each parameter must be defined exactly once."
            )

        # ==========================================================================
        #::: luser-proof: chromatic suffix must match a known bandpass
        #::: Catches typos (e.g. b_rr_tes vs tess) that would otherwise be
        #::: silently ignored, leaving the fit to use a default 0 for the rr.
        # ==========================================================================
        _bandpass_map = self.settings.get("bandpass", {}) or {}
        _known_bands = set(_bandpass_map.values())
        _companions = self.settings.get("companions_all", []) or []
        _is_chromatic = bool(self.settings.get("chromatic", False))
        if _known_bands and _companions:
            _bad_chromatic = []
            for _c in _companions:
                _prefix = _c + "_rr_"
                for _n in _stripped:
                    if _n.startswith(_prefix):
                        _suffix = _n[len(_prefix) :]
                        if _suffix not in _known_bands:
                            _bad_chromatic.append((_n, _suffix))
            if _bad_chromatic:
                _msg_pairs = "; ".join(f"{k} (suffix '{s}')" for k, s in _bad_chromatic)
                raise ValueError(
                    "params.csv references unknown bandpass(es): "
                    + _msg_pairs
                    + ". Known bandpasses (from settings.csv 'bandpass'): "
                    + sorted(_known_bands).__repr__()
                    + "."
                )

        # ==========================================================================
        #::: luser-proof: chromatic mode requires one rr row per (companion,
        #::: bandpass). Without that, the validator silently defaults the
        #::: missing keys to None and the likelihood falls back to the
        #::: unsuffixed b_rr — a fit that looks chromatic in settings.csv but
        #::: is achromatic in practice. Catch both half-states up front:
        #:::   - chromatic settings + plain `<c>_rr` row present (ambiguous),
        #:::   - chromatic settings + at least one expected `<c>_rr_<bp>` row missing.
        # ==========================================================================
        if _is_chromatic and _companions:
            _problems = []
            _stripped_set = set(_stripped)
            for _c in _companions:
                _achromatic_key = _c + "_rr"
                _expected = {f"{_c}_rr_{bp}" for bp in _known_bands}
                _present = _expected & _stripped_set
                _missing = sorted(_expected - _present)
                _has_achromatic = _achromatic_key in _stripped_set
                if _has_achromatic and not _present:
                    _problems.append(
                        f"companion '{_c}': params.csv has '{_achromatic_key}' but "
                        f"settings.csv 'bandpass' is chromatic. Replace it with "
                        f"one row per bandpass: " + ", ".join(sorted(_expected)) + "."
                    )
                elif _missing and not _has_achromatic:
                    _problems.append(
                        f"companion '{_c}': chromatic mode requires a row per "
                        f"bandpass; missing " + ", ".join(_missing) + "."
                    )
                elif _missing and _has_achromatic:
                    _problems.append(
                        f"companion '{_c}': params.csv mixes the achromatic key "
                        f"'{_achromatic_key}' with chromatic rows; missing "
                        + ", ".join(_missing)
                        + ". Pick one shape and remove "
                        f"'{_achromatic_key}'."
                    )
            if _problems:
                raise ValueError(
                    "Chromatic configuration mismatch between settings.csv and "
                    "params.csv:\n  - " + "\n  - ".join(_problems)
                )

        # ==========================================================================
        #::: luser-proof: explicit `chromatic,False` override forbids per-band
        #::: rr rows in params.csv. Without this check those rows would be
        #::: silently ignored (every `get_bandpass(inst)` returns None under
        #::: the override), masking the user's mistake.
        # ==========================================================================
        elif (
            (not _is_chromatic)
            and _companions
            and _known_bands
            and ("chromatic" in getattr(self, "_settings_raw_keys", set()))
        ):
            _stripped_set = set(_stripped)
            _problems = []
            for _c in _companions:
                _stray = sorted(
                    n
                    for n in _stripped_set
                    if n.startswith(_c + "_rr_") and n[len(_c + "_rr_") :] in _known_bands
                )
                if _stray:
                    _achromatic_key = _c + "_rr"
                    _problems.append(
                        f"companion '{_c}': settings.csv has 'chromatic,False' "
                        f"(achromatic override) but params.csv carries per-band "
                        f"row(s) "
                        + ", ".join(_stray)
                        + f". Remove them and add a single '{_achromatic_key}' row, "
                        f"or set 'chromatic,True' / drop the override to fit "
                        f"per-band radius ratios."
                    )
            if _problems:
                raise ValueError(
                    "Chromatic-override configuration mismatch between "
                    "settings.csv ('chromatic,False') and params.csv:\n  - "
                    + "\n  - ".join(_problems)
                )

        # ==========================================================================
        #::: function to assure backwards compability
        # ==========================================================================
        def backwards_compability(key_new, key_deprecated):
            if key_deprecated in np.atleast_1d(buf["name"]):
                warnings.warn(
                    "You are using outdated keywords. Automatically renaming "
                    + key_deprecated
                    + " ---> "
                    + key_new
                    + ". Please fix this before the Duolingo owl comes to get you.",
                    stacklevel=2,
                )  # , category=DeprecationWarning)
                ind = np.where(buf["name"] == key_deprecated)[0]
                np.atleast_1d(buf["name"])[ind] = key_new

        # ==========================================================================
        #::: luser-proof: backwards compability
        # (has to happend first thing and right inside buf['name'])
        # ==========================================================================
        for inst in self.settings["inst_all"]:
            backwards_compability(key_new="host_ldc_q1_" + inst, key_deprecated="ldc_q1_" + inst)
            backwards_compability(key_new="host_ldc_q2_" + inst, key_deprecated="ldc_q2_" + inst)
            backwards_compability(key_new="host_ldc_q3_" + inst, key_deprecated="ldc_q3_" + inst)
            backwards_compability(key_new="host_ldc_q4_" + inst, key_deprecated="ldc_q4_" + inst)
            backwards_compability(
                key_new="ln_err_flux_" + inst, key_deprecated="log_err_flux_" + inst
            )
            backwards_compability(
                key_new="ln_jitter_rv_" + inst, key_deprecated="log_jitter_rv_" + inst
            )
            backwards_compability(
                key_new="baseline_gp_matern32_lnsigma_flux_" + inst,
                key_deprecated="baseline_gp1_flux_" + inst,
            )
            backwards_compability(
                key_new="baseline_gp_matern32_lnrho_flux_" + inst,
                key_deprecated="baseline_gp2_flux_" + inst,
            )
            backwards_compability(
                key_new="baseline_gp_matern32_lnsigma_rv_" + inst,
                key_deprecated="baseline_gp1_rv_" + inst,
            )
            backwards_compability(
                key_new="baseline_gp_matern32_lnrho_rv_" + inst,
                key_deprecated="baseline_gp2_rv_" + inst,
            )

        # ==========================================================================
        #::: luser-proof: check for allowed keys to catch typos etc.
        # ==========================================================================
        def get_valid_param_patterns():
            valid_patterns = set()
            companions = self.settings.get("companions_all", [])
            inst_all = self.settings.get("inst_all", [])
            inst_phot = self.settings.get("inst_phot", [])
            inst_rv = self.settings.get("inst_rv", [])

            for companion in companions:
                valid_patterns.add(companion + "_rr")
                valid_patterns.add(companion + "_rsuma")
                valid_patterns.add(companion + "_cosi")
                valid_patterns.add(companion + "_epoch")
                valid_patterns.add(companion + "_period")
                valid_patterns.add(companion + "_f_c")
                valid_patterns.add(companion + "_f_s")
                valid_patterns.add(companion + "_sbratio")
                valid_patterns.add(companion + "_a")
                valid_patterns.add(companion + "_q")
                valid_patterns.add(companion + "_K")
                valid_patterns.add(companion + "_dil")
                valid_patterns.add(companion + "_ld_law")
                valid_patterns.add(companion + "_ld_space")
                valid_patterns.add(companion + "_shape")
                valid_patterns.add(companion + "_grid")
                valid_patterns.add(companion + "_flux_weighted")
                valid_patterns.add(companion + "_rotfac")
                valid_patterns.add(companion + "_hf")
                valid_patterns.add(companion + "_bfac")
                valid_patterns.add(companion + "_heat")
                valid_patterns.add(companion + "_lambda")
                valid_patterns.add(companion + "_vsini")
                valid_patterns.add(companion + "_N_spots")
                valid_patterns.add(companion + "_phase_curve_beaming")
                valid_patterns.add(companion + "_phase_curve_atmospheric")
                valid_patterns.add(companion + "_phase_curve_ellipsoidal")
                for i in range(1, 50):
                    valid_patterns.add(companion + "_ttv_transit_" + str(i))

                for inst in inst_all:
                    valid_patterns.add(companion + "_ldc_" + inst)
                    for j in range(1, 10):
                        valid_patterns.add(companion + "_ldc_q" + str(j) + "_" + inst)
                        valid_patterns.add(companion + "_ldc_u" + str(j) + "_" + inst)
                    valid_patterns.add(companion + "_gdc_" + inst)
                    valid_patterns.add(companion + "_spots_" + inst)
                    for k in range(1, 10):
                        valid_patterns.add(companion + "_spot_" + str(k) + "_long_" + inst)
                        valid_patterns.add(companion + "_spot_" + str(k) + "_lat_" + inst)
                        valid_patterns.add(companion + "_spot_" + str(k) + "_size_" + inst)
                        valid_patterns.add(companion + "_spot_" + str(k) + "_brightness_" + inst)

            for inst in inst_all:
                valid_patterns.add("host_ld_law_" + inst)
                valid_patterns.add("host_ld_space_" + inst)
                valid_patterns.add("host_grid_" + inst)
                valid_patterns.add("host_shape_" + inst)
                valid_patterns.add("host_flux_weighted_" + inst)
                valid_patterns.add("host_rotfac_" + inst)
                valid_patterns.add("host_hf_" + inst)
                valid_patterns.add("host_bfac_" + inst)
                valid_patterns.add("host_heat_" + inst)
                valid_patterns.add("host_lambda_" + inst)
                valid_patterns.add("host_N_spots_" + inst)
                valid_patterns.add("host_spots_" + inst)
                for j in range(1, 10):
                    valid_patterns.add("host_ldc_q" + str(j) + "_" + inst)
                    valid_patterns.add("host_ldc_u" + str(j) + "_" + inst)
                    valid_patterns.add("host_spot_" + str(j) + "_long_" + inst)
                    valid_patterns.add("host_spot_" + str(j) + "_lat_" + inst)
                    valid_patterns.add("host_spot_" + str(j) + "_size_" + inst)
                    valid_patterns.add("host_spot_" + str(j) + "_brightness_" + inst)

            for inst in inst_all:
                valid_patterns.add("dil_" + inst)
                valid_patterns.add("host_gdc_" + inst)

            for inst in inst_phot:
                valid_patterns.add("ln_err_flux_" + inst)
                valid_patterns.add("baseline_offset_flux_" + inst)
                valid_patterns.add("baseline_slope_flux_" + inst)
                valid_patterns.add("baseline_gp_matern32_lnsigma_flux_" + inst)
                valid_patterns.add("baseline_gp_matern32_lnrho_flux_" + inst)
                valid_patterns.add("baseline_gp_sho_omega_flux_" + inst)
                valid_patterns.add("baseline_gp_sho_A_flux_" + inst)
                valid_patterns.add("baseline_gp_real_omega_flux_" + inst)
                valid_patterns.add("baseline_gp_real_A_flux_" + inst)
                valid_patterns.add("baseline_gp_complex_omega_flux_" + inst)
                valid_patterns.add("baseline_gp_complex_A_flux_" + inst)
                valid_patterns.add("baseline_gp_complex_Q_flux_" + inst)

            for inst in inst_rv:
                valid_patterns.add("ln_jitter_rv_" + inst)
                valid_patterns.add("baseline_offset_rv_" + inst)
                valid_patterns.add("baseline_slope_rv_" + inst)
                valid_patterns.add("baseline_gp_matern32_lnsigma_rv_" + inst)
                valid_patterns.add("baseline_gp_matern32_lnrho_rv_" + inst)

            valid_patterns.add("R_host")
            valid_patterns.add("M_host")
            valid_patterns.add("Teff_host")
            valid_patterns.add("host_vsini")
            valid_patterns.add("host_rotfac")
            valid_patterns.add("R_host_err")
            valid_patterns.add("M_host_err")
            valid_patterns.add("Teff_host_err")

            for i in range(1, 10):
                valid_patterns.add("flare_" + str(i) + "_epoch")
                valid_patterns.add("flare_" + str(i) + "_duration")
                valid_patterns.add("flare_" + str(i) + "_amplitude")
                valid_patterns.add("flare_" + str(i) + "_beta")

            for i in range(1, 10):
                valid_patterns.add("bump_tpeak_" + str(i))
                valid_patterns.add("bump_width_" + str(i))

            #::: bump amplitude is wavelength-dependent: keyed by bandpass when a
            #::: bandpass row is present, else falls back to the instrument name
            _bump_bandpass = self.settings.get("bandpass") or {}
            for inst in inst_phot:
                _suffix = _bump_bandpass.get(inst) or inst
                for i in range(1, 10):
                    valid_patterns.add("bump_ampl_" + _suffix + "_" + str(i))

            return valid_patterns

        def is_valid_key(key, valid_patterns):
            if key in valid_patterns:
                return True
            for pattern in valid_patterns:
                if key.startswith(pattern):
                    return True
            if key.startswith("host_ldc_q") and "_" in key:
                return True
            if key.startswith("host_ldc_u") and "_" in key:
                return True
            if "_ldc_q" in key and "_" in key:
                return True
            if "_ldc_u" in key and "_" in key:
                return True
            if key.startswith("dil_") and "_" in key:
                return True
            if key.startswith("ln_err_flux_") and "_" in key:
                return True
            if key.startswith("ln_jitter_rv_") and "_" in key:
                return True
            if key.startswith("baseline_") and "_" in key:
                return True
            # Stellar-variability params (stellar_var_gp_{real,complex,matern32,sho}_*,
            # stellar_var_gp_offset_*, stellar_var_{slope,offset}_*) are keyed by data
            # type (flux/rv) and consumed by computer.stellar_var_* (e.g. the SHOTerm
            # log_S0/log_Q/log_omega0 hyperparameters). Accept them like baseline_*.
            if key.startswith("stellar_var_") and "_" in key:
                return True
            return False

        valid_patterns = get_valid_param_patterns()
        allkeys_list = list(buf["name"])
        fit_flags = buf["fit"] if "fit" in buf.dtype.names else None

        def _is_free_param(idx):
            #::: a row counts as a free parameter only if its fit flag is 1
            if fit_flags is None:
                return False
            try:
                return int(float(str(np.atleast_1d(fit_flags)[idx]).strip())) == 1
            except (ValueError, TypeError):
                return False

        unrecognized = []
        unrecognized_free = []
        for idx, key in enumerate(allkeys_list):
            key_clean = key.strip()
            if key_clean in ["user-given:", "automatically set:"]:
                continue
            if not is_valid_key(key_clean, valid_patterns):
                if _is_free_param(idx):
                    unrecognized_free.append(key_clean)
                else:
                    unrecognized.append(key_clean)

        #::: A free parameter (fit=1) that is not recognized would otherwise be
        #::: silently dropped from the fit, so the user would believe they are
        #::: sampling a dimension that is actually ignored. Fail loudly instead.
        if unrecognized_free:
            msg = (
                "The following parameters in params.csv are set as free "
                "(fit=1) but are not recognized, likely due to a typo in the "
                "parameter name or instrument suffix:\n"
            )
            for key in unrecognized_free:
                msg += "  - " + key + "\n"
            msg += (
                "Fix the parameter name(s), or set fit=0 to keep them as "
                "fixed values. Refusing to silently drop a free parameter."
            )
            raise ValueError(msg)

        if unrecognized:
            self.logprint(
                "\nWARNING: The following parameters in params.csv are not recognized and will be ignored:"
            )
            for key in unrecognized:
                self.logprint("  - " + key)
            self.logprint("")

        # ==========================================================================
        #::: luser-proof: N_spots set in settings.csv must have matching spot
        #::: parameters in params.csv, otherwise update_params() crashes later
        #::: with a cryptic KeyError. Warn and reset the count to 0 so that
        #::: plotting and fitting can still proceed (consistent with the
        #::: warn-not-raise policy used elsewhere for config mismatches).
        # ==========================================================================
        allkeys_set = set(k.strip() for k in allkeys_list)
        _spot_suffixes = ["_long_", "_lat_", "_size_", "_brightness_"]

        def _check_spot_params(count_key, prefix):
            n = self.settings.get(count_key, 0)
            if not n:
                return
            missing = [
                prefix + "spot_" + str(i) + suffix + inst
                for i in range(1, n + 1)
                for suffix in _spot_suffixes
                if prefix + "spot_" + str(i) + suffix + inst not in allkeys_set
            ]
            if missing:
                self.logprint(
                    "\nWARNING: " + count_key + " = " + str(n) + " in settings.csv, but the "
                    "following spot parameters are missing from params.csv:"
                )
                for key in missing:
                    self.logprint("  - " + key)
                self.logprint(
                    "Resetting " + count_key + " to 0. Add the spot parameters to "
                    "params.csv to model these spots.\n"
                )
                self.settings[count_key] = 0

        for inst in self.settings["inst_all"]:
            _check_spot_params("host_N_spots_" + inst, "host_")
            for companion in self.settings["companions_all"]:
                _check_spot_params(companion + "_N_spots_" + inst, companion + "_")

        # ==========================================================================
        #::: set up stuff
        # ==========================================================================
        self.allkeys = np.atleast_1d(buf["name"])  # len(all rows in params.csv)
        self.labels = np.atleast_1d(buf["label"])  # len(all rows in params.csv)
        self.units = np.atleast_1d(buf["unit"])  # len(all rows in params.csv)
        if "truth" in buf.dtype.names:
            self.truths = np.atleast_1d(buf["truth"])  # len(all rows in params.csv)
        else:
            self.truths = np.nan * np.ones(len(self.allkeys))

        self.params = collections.OrderedDict()  # len(all rows in params.csv)
        self.params["user-given:"] = ""  # just for pretty printing
        for i, key in enumerate(self.allkeys):
            #::: if it's not a "coupled parameter", then use the given value
            if np.atleast_1d(buf["value"])[i] not in list(self.allkeys):
                self.params[key] = float(np.atleast_1d(buf["value"])[i])
            #::: if it's a "coupled parameter", then write the string of the key it is coupled to
            else:
                self.params[key] = np.atleast_1d(buf["value"])[i]

        # ==========================================================================
        #::: function to automatically set default params if they were not given
        # ==========================================================================
        #::: When a key has a registered *physical* limit (see
        #::: allesfitter.validation.physical_limits) the registry is the single
        #::: source of truth for its (min, max) — so e.g. rsuma <= 1, dilution
        #::: in [0, 1) and vsini >= 0 are enforced here without duplicating the
        #::: numbers. Keys with no registered limit fall back to the explicit
        #::: default_min / default_max passed by the caller.
        def validate(key, default, default_min, default_max):
            if (key in self.params) and (self.params[key] is not None):
                limit = lookup_limit(key)
                if limit is not None and isinstance(self.params[key], (int, float)):
                    if not limit.contains(self.params[key]):
                        raise ValueError(
                            "User input for "
                            + key
                            + " is "
                            + str(self.params[key])
                            + " but must lie within "
                            + limit.describe()
                            + "."
                        )
                elif (self.params[key] < default_min) or (self.params[key] > default_max):
                    raise ValueError(
                        "User input for "
                        + key
                        + " is "
                        + str(self.params[key])
                        + " but must lie within ["
                        + str(default_min)
                        + ","
                        + str(default_max)
                        + "]."
                    )
            if key not in self.params:
                self.params[key] = default

        # ==========================================================================
        #::: luser-proof: make sure the limb darkening values are uniquely
        #::: from either the u- or q-space
        # ==========================================================================
        def check_ld(obj, inst):
            if self.settings[obj + "_ld_space_" + inst] == "q":
                matches = fnmatch.filter(self.allkeys, obj + "_ldc_u*_" + inst)
                if len(matches) > 0:
                    raise ValueError(
                        "The following user input is inconsistent:\n"
                        + "Setting: '"
                        + key
                        + "' = 'q'\n"
                        + f"Parameters: {matches}"
                    )

            elif self.settings[obj + "_ld_space_" + inst] == "u":
                matches = fnmatch.filter(self.allkeys, obj + "_ldc_q*_" + inst)
                if len(matches) > 0:
                    raise ValueError(
                        "The following user input is inconsistent:\n"
                        + "Setting: '"
                        + key
                        + "' = 'u'\n"
                        + f"Parameters: {matches}"
                    )

        for inst in self.settings["inst_all"]:
            for obj in ["host"] + self.settings["companions_all"]:
                check_ld(obj, inst)

        # ==========================================================================
        #::: validate that initial guess params have reasonable values
        # ==========================================================================
        self.params["automatically set:"] = ""  # just for pretty printing
        for companion in self.settings["companions_all"]:
            for inst in self.settings["inst_all"]:

                # rr-key bandpass: respects the chromatic flag so an
                # explicit `chromatic,False` override collapses rr keys
                # back to the achromatic `<companion>_rr`.
                bandpass = self.get_bandpass(inst)
                if bandpass:
                    bp_suffix = "_" + bandpass
                else:
                    bp_suffix = ""

                # LDC-key bandpass: always reads the raw bandpass dict
                # (independent of the chromatic flag) because limb
                # darkening is a function of wavelength, not of the
                # rr-naming convention. Falls back to the inst name only
                # when settings.csv has no bandpass row at all.
                _ldc_bp = self.get_ldc_bandpass(inst)

                #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
                #::: ellc defaults
                #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

                #::: frequently used parameters
                # rr is per-bandpass in chromatic mode, per-companion in achromatic
                if bandpass:
                    rr_key = companion + "_rr" + bp_suffix
                    validate(rr_key, None, 0.0, np.inf)
                else:
                    validate(companion + "_rr", None, 0.0, np.inf)
                validate(companion + "_rsuma", None, 0.0, np.inf)
                validate(companion + "_cosi", 0.0, 0.0, 1.0)
                validate(companion + "_epoch", 0.0, -np.inf, np.inf)
                validate(companion + "_period", 0.0, 0.0, np.inf)
                validate(companion + "_sbratio_" + inst, 0.0, 0.0, np.inf)
                validate(companion + "_K", 0.0, 0.0, np.inf)
                validate(companion + "_f_s", 0.0, -1, 1)
                validate(companion + "_f_c", 0.0, -1, 1)
                #::: joint eccentricity constraint e = f_s^2 + f_c^2 < 1
                #::: (each is bounded [-1,1] alone, which would admit e up to 2)
                f_s_val = self.params.get(companion + "_f_s")
                f_c_val = self.params.get(companion + "_f_c")
                if isinstance(f_s_val, (int, float)) and isinstance(f_c_val, (int, float)):
                    ecc_err = eccentricity_error(companion, f_s_val, f_c_val)
                    if ecc_err is not None:
                        raise ValueError("User input invalid: " + ecc_err)
                validate("dil_" + inst, 0.0, -np.inf, np.inf)

                #::: limb darkenings, u-space (per-bandpass in chromatic, per-inst in achromatic)
                ldc_suffix = ("_" + _ldc_bp) if _ldc_bp else ("_" + inst)
                validate("host_ldc_u1" + ldc_suffix, None, 0, 1)
                validate("host_ldc_u2" + ldc_suffix, None, 0, 1)
                validate("host_ldc_u3" + ldc_suffix, None, 0, 1)
                validate("host_ldc_u4" + ldc_suffix, None, 0, 1)
                validate(companion + "_ldc_u1" + ldc_suffix, None, 0, 1)
                validate(companion + "_ldc_u2" + ldc_suffix, None, 0, 1)
                validate(companion + "_ldc_u3" + ldc_suffix, None, 0, 1)
                validate(companion + "_ldc_u4" + ldc_suffix, None, 0, 1)

                #::: limb darkenings, q-space
                validate("host_ldc_q1" + ldc_suffix, None, 0, 1)
                validate("host_ldc_q2" + ldc_suffix, None, 0, 1)
                validate("host_ldc_q3" + ldc_suffix, None, 0, 1)
                validate("host_ldc_q4" + ldc_suffix, None, 0, 1)
                validate(companion + "_ldc_q1" + ldc_suffix, None, 0, 1)
                validate(companion + "_ldc_q2" + ldc_suffix, None, 0, 1)
                validate(companion + "_ldc_q3" + ldc_suffix, None, 0, 1)
                validate(companion + "_ldc_q4" + ldc_suffix, None, 0, 1)

                #::: catch exceptions
                if self.params[companion + "_period"] is None:
                    self.settings["do_not_phase_fold"] = True

                #::: advanced parameters
                validate(companion + "_a", None, 0.0, np.inf)
                validate(companion + "_q", 1.0, 0.0, np.inf)

                validate("didt_" + inst, None, -np.inf, np.inf)
                validate("domdt_" + inst, None, -np.inf, np.inf)

                validate("host_gdc_" + inst, None, 0.0, 1.0)
                validate("host_rotfac_" + inst, 1.0, 0.0, np.inf)
                validate("host_hf_" + inst, 1.5, -np.inf, np.inf)
                validate("host_bfac_" + inst, None, -np.inf, np.inf)
                validate("host_heat_" + inst, None, -np.inf, np.inf)
                validate("host_lambda", None, -np.inf, np.inf)
                validate("host_vsini", None, -np.inf, np.inf)

                validate(companion + "_gdc_" + inst, None, 0.0, 1.0)
                validate(companion + "_rotfac_" + inst, 1.0, 0.0, np.inf)
                validate(companion + "_hf_" + inst, 1.5, -np.inf, np.inf)
                validate(companion + "_bfac_" + inst, None, -np.inf, np.inf)
                validate(companion + "_heat_" + inst, None, -np.inf, np.inf)
                validate(companion + "_lambda", None, -np.inf, np.inf)
                validate(companion + "_vsini", None, -np.inf, np.inf)

                #::: special parameters (list type)
                if "host_spots_" + inst not in self.params:
                    self.params["host_spots_" + inst] = None
                if companion + "_spots_" + inst not in self.params:
                    self.params[companion + "_spots_" + inst] = None

                #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
                #::: errors and jitters
                #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
                # TODO: add validations for all errors / jitters

                #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
                #::: baselines (and backwards compability)
                #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
                # TODO: add validations for all baseline params

                #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
                #::: phase curve style: sine_series
                # all in ppt
                # A1 (beaming)
                # B1 (atmospheric), can be split in thermal and reflected
                # B2 (ellipsoidal)
                # B3 (ellipsoidal 2nd order)
                #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
                # if (self.settings['phase_curve_style'] == 'sine_series') and (inst in self.settings['inst_phot']):
                if inst in self.settings["inst_phot"]:
                    validate(companion + "_phase_curve_A1_" + inst, None, 0.0, np.inf)
                    validate(companion + "_phase_curve_B1_" + inst, None, -np.inf, 0.0)
                    validate(companion + "_phase_curve_B1_shift_" + inst, 0.0, -np.inf, np.inf)
                    validate(companion + "_phase_curve_B1t_" + inst, None, -np.inf, 0.0)
                    validate(companion + "_phase_curve_B1t_shift_" + inst, 0.0, -np.inf, np.inf)
                    validate(companion + "_phase_curve_B1r_" + inst, None, -np.inf, 0.0)
                    validate(companion + "_phase_curve_B1r_shift_" + inst, 0.0, -np.inf, np.inf)
                    validate(companion + "_phase_curve_B2_" + inst, None, -np.inf, 0.0)
                    validate(companion + "_phase_curve_B3_" + inst, None, -np.inf, 0.0)

                #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
                #::: phase curve style: sine_physical
                # A1 (beaming)
                # B1 (atmospheric), can be split in thermal and reflected
                # B2 (ellipsoidal)
                # B3 (ellipsoidal 2nd order)
                #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
                # if (self.settings['phase_curve_style'] == 'sine_physical') and (inst in self.settings['inst_phot']):
                if inst in self.settings["inst_phot"]:
                    validate(companion + "_phase_curve_beaming_" + inst, None, 0.0, np.inf)
                    validate(companion + "_phase_curve_atmospheric_" + inst, None, 0.0, np.inf)
                    validate(
                        companion + "_phase_curve_atmospheric_shift_" + inst, 0.0, -np.inf, np.inf
                    )
                    validate(
                        companion + "_phase_curve_atmospheric_thermal_" + inst, None, 0.0, np.inf
                    )
                    validate(
                        companion + "_phase_curve_atmospheric_thermal_shift_" + inst,
                        0.0,
                        -np.inf,
                        np.inf,
                    )
                    validate(
                        companion + "_phase_curve_atmospheric_reflected_" + inst, None, 0.0, np.inf
                    )
                    validate(
                        companion + "_phase_curve_atmospheric_reflected_shift_" + inst,
                        0.0,
                        -np.inf,
                        np.inf,
                    )
                    validate(companion + "_phase_curve_ellipsoidal_" + inst, None, 0.0, np.inf)
                    validate(companion + "_phase_curve_ellipsoidal_2nd_" + inst, None, 0.0, np.inf)

                #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
                #::: to avoid a bug/feature in ellc, if either property is >0, set the other to 1-15 (not 0):
                #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
                if self.params[companion + "_heat_" + inst] is not None:
                    if (self.params[companion + "_sbratio_" + inst] == 0) and (
                        self.params[companion + "_heat_" + inst] > 0
                    ):
                        self.params[companion + "_sbratio_" + inst] = (
                            1e-15  # this is to avoid a bug/feature in ellc
                        )
                    if (self.params[companion + "_sbratio_" + inst] > 0) and (
                        self.params[companion + "_heat_" + inst] == 0
                    ):
                        self.params[companion + "_heat_" + inst] = (
                            1e-15  # this is to avoid a bug/feature in ellc
                        )

                #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
                #::: luser proof: avoid conflicting/degenerate phase curve commands
                #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
                if (inst in self.settings["inst_phot"]) and (self.settings["phase_curve"]):
                    phase_curve_model_1 = (
                        self.params[companion + "_phase_curve_B1_" + inst] is not None
                    )
                    phase_curve_model_2 = (
                        self.params[companion + "_phase_curve_B1t_" + inst] is not None
                    ) or (self.params[companion + "_phase_curve_B1r_" + inst] is not None)
                    phase_curve_model_3 = (
                        self.params[companion + "_phase_curve_atmospheric_" + inst] is not None
                    )
                    phase_curve_model_4 = (
                        self.params[companion + "_phase_curve_atmospheric_thermal_" + inst]
                        is not None
                    ) or (
                        self.params[companion + "_phase_curve_atmospheric_reflected_" + inst]
                        is not None
                    )
                    phase_curve_model_5 = (
                        (self.params["host_bfac_" + inst] is not None)
                        or (self.params["host_heat_" + inst] is not None)
                        or (self.params["host_gdc_" + inst] is not None)
                        or (self.settings["host_shape_" + inst] != "sphere")
                        or (self.params[companion + "_bfac_" + inst] is not None)
                        or (self.params[companion + "_heat_" + inst] is not None)
                        or (self.params[companion + "_gdc_" + inst] is not None)
                        or (self.settings[companion + "_shape_" + inst] != "sphere")
                    )
                    if (
                        phase_curve_model_1
                        + phase_curve_model_2
                        + phase_curve_model_3
                        + phase_curve_model_4
                        + phase_curve_model_5
                    ) > 1:
                        raise ValueError(
                            "You can use either\n"
                            + '1) the sine_series phase curve model with "*_phase_curve_B1_*",\n'
                            + '2) the sine_series phase curve model with "*_phase_curve_B1t_*" and "*_phase_curve_B1r_*", or\n'
                            + '3) the sine_physical phase curve model with "*_phase_curve_atmospheric_*",\n'
                            + '4) the sine_physical phase curve model with "*_phase_curve_atmospheric_thermal_*" and "*_phase_curve_atmospheric_reflected_*", or\n'
                            + '5) the ellc_physical phase curve model with "*_bfac_*", "*_heat_*", "*_gdc_*" etc.\n'
                            + "but you shall not pass with a mix&match."
                        )

        # ==========================================================================
        #::: coupled params
        # ==========================================================================
        if "coupled_with" in buf.dtype.names:
            self.coupled_with = buf["coupled_with"]
        else:
            self.coupled_with = [None] * len(self.allkeys)

        for i, key in enumerate(self.allkeys):
            if isinstance(self.coupled_with[i], str) and (len(self.coupled_with[i]) > 0):
                self.params[key] = self.params[
                    self.coupled_with[i]
                ]  # luser proof: automatically set the values of the params coupled to another param
                buf["fit"][
                    i
                ] = 0  # luser proof: automatically set fit=0 for the params coupled to another param

        # ==========================================================================
        #::: baseline share groups: alias follower GP hyperparameters to leader
        #::: Each follower GP hyperparameter (baseline_gp_*_{key}_{follower})
        #::: gets the leader's value here so self.params is self-consistent at
        #::: load time. Per-iteration re-aliasing happens in computer.update_params
        #::: so that follower entries track the leader as theta changes. Together
        #::: this means the NS fit vector contains only the leader's hypers,
        #::: while computer.py's joint-GP code path assembles all group members'
        #::: residuals into a single celerite GP under the leader's name.
        # ==========================================================================
        _allkeys_set = set(self.allkeys)
        for k_share in ("flux", "rv", "rv2"):
            followers_of = self.settings.get("baseline_share_" + k_share + "_followers_of", {})
            if not followers_of:
                continue
            for leader, followers in followers_of.items():
                # Cross-file sanity: settings.csv says the leader uses a
                # specific GP kernel — every required hyperparameter row for
                # that kernel must exist in params.csv. Without this check,
                # the user would only see a KeyError deep inside
                # baseline_get_gp at the first likelihood call.
                leader_base = self.settings.get("baseline_" + k_share + "_" + leader, "none")
                required_prefixes = BASELINE_GP_REQUIRED_HYPERS.get(leader_base, ())
                missing = []
                for rp in required_prefixes:
                    if rp + k_share + "_" + leader not in self.params:
                        missing.append(rp + k_share + "_" + leader)
                if missing:
                    raise ValueError(
                        f"baseline_share_{k_share}: leader '{leader}' declares "
                        f"baseline_{k_share}_{leader}={leader_base} but params.csv is missing "
                        f"the required row(s) {missing}. Every share-group leader "
                        "must own all GP hyperparameter rows for its "
                        "declared kernel."
                    )

                for prefix in BASELINE_GP_HYPER_PREFIXES:
                    leader_key = prefix + k_share + "_" + leader
                    if leader_key not in self.params:
                        continue
                    for f in followers:
                        follower_key = prefix + k_share + "_" + f
                        if follower_key in _allkeys_set:
                            idx = list(self.allkeys).index(follower_key)
                            coupled = self.coupled_with[idx]
                            is_coupled = isinstance(coupled, str) and len(coupled) > 0
                            if is_coupled and coupled != leader_key:
                                # Coupled to something other than the leader's
                                # corresponding row — fundamentally
                                # inconsistent with the share-group alias.
                                raise ValueError(
                                    f"baseline_share_{k_share}: '{follower_key}' is "
                                    f"coupled_with='{coupled}' but instrument "
                                    f"'{f}' is in a share group led by '{leader}', "
                                    f"which expects coupled_with='{leader_key}'. "
                                    "Remove the explicit coupling or point "
                                    "it at the leader's key."
                                )
                            if int(np.atleast_1d(buf["fit"])[idx]) == 1 and not is_coupled:
                                raise ValueError(
                                    f"baseline_share_{k_share}: '{follower_key}' has fit=1 in "
                                    f"params.csv but instrument '{f}' is in a "
                                    f"share group led by '{leader}'. Either remove "
                                    "the follower row, set fit=0, or use "
                                    f"coupled_with={leader_key}."
                                )
                            buf["fit"][idx] = 0
                        self.params[follower_key] = self.params[leader_key]

        # ==========================================================================
        #::: mark to be fitted params
        # ==========================================================================
        self.ind_fit = buf["fit"] == 1  # len(all rows in params.csv)

        self.fitkeys = buf["name"][self.ind_fit]  # len(ndim)
        self.fitlabels = self.labels[self.ind_fit]  # len(ndim)
        self.fitunits = self.units[self.ind_fit]  # len(ndim)
        self.fittruths = self.truths[self.ind_fit]  # len(ndim)
        self.theta_0 = buf["value"][self.ind_fit]  # len(ndim)

        if "init_err" in buf.dtype.names:
            self.init_err = buf["init_err"][self.ind_fit]  # len(ndim)
        else:
            self.init_err = 1e-8

        self.bounds = [str(item).split(" ") for item in buf["bounds"][self.ind_fit]]  # len(ndim)
        for i, item in enumerate(self.bounds):
            if item[0] in ["uniform", "normal"]:
                self.bounds[i] = [item[0], float(item[1]), float(item[2])]
            elif item[0] in ["trunc_normal"]:
                self.bounds[i] = [
                    item[0],
                    float(item[1]),
                    float(item[2]),
                    float(item[3]),
                    float(item[4]),
                ]
            else:
                raise ValueError(
                    'Bounds have to be "uniform", "normal" or "trunc_normal". Input from "params.csv" was "'
                    + self.bounds[i][0]
                    + '".'
                )

        self.ndim = len(self.theta_0)  # len(ndim)

        # ==========================================================================
        #::: luser proof: mask_transit removes the transit from the data, so every
        #::: transit parameter MUST be fixed (fit=0). A free transit parameter
        #::: would be sampled against data that no longer contains the transit,
        #::: which is meaningless. Fail loudly listing the offenders so the user
        #::: either fixes them or turns mask_transit off.
        # ==========================================================================
        if self.settings.get("mask_transit", False):
            #::: per-companion transit-shape + ephemeris parameters; `rr` is
            #::: handled separately to also catch the chromatic `_rr_<band>` keys.
            _transit_suffixes = ["_rsuma", "_cosi", "_epoch", "_period", "_f_s", "_f_c"]
            _free_transit = []
            for companion in self.settings["companions_phot"]:
                for key in self.fitkeys:
                    k = str(key)
                    if k == companion + "_rr" or k.startswith(companion + "_rr_"):
                        _free_transit.append(k)
                    elif any(k == companion + suf for suf in _transit_suffixes):
                        _free_transit.append(k)
            if _free_transit:
                raise ValueError(
                    "mask_transit==True in settings.csv removes the transit from "
                    "the data, so all transit parameters must be fixed (fit=0) in "
                    "params.csv. The following transit parameters are still free "
                    "(fit=1):\n  - "
                    + "\n  - ".join(_free_transit)
                    + "\nSet fit=0 for these (you are modelling stellar_var/spots "
                    "on the out-of-transit data), or set mask_transit=False."
                )

        # ==========================================================================
        #::: luser proof: check if all initial guesses lie within their bounds
        # ==========================================================================
        # TODO: make this part of the validate() function
        for th, b, key in zip(self.theta_0, self.bounds, self.fitkeys):

            #:::: test bounds
            if (b[0] == "uniform") and not (b[1] <= th <= b[2]):
                raise ValueError("The initial guess for " + key + " lies outside of its bounds.")

            elif (b[0] == "normal") and (np.abs(th - b[1]) > 3 * b[2]):
                answer = input(
                    "The initial guess for "
                    + key
                    + " lies more than 3 sigma from its prior\n"
                    + "What do you want to do?\n"
                    + "1 : continue at any sacrifice \n"
                    + "2 : stop and let me fix the params.csv file \n"
                )
                if answer == 1:
                    pass
                else:
                    raise ValueError("User aborted the run.")

            elif (b[0] == "trunc_normal") and not (b[1] <= th <= b[2]):
                raise ValueError("The initial guess for " + key + " lies outside of its bounds.")

            elif (b[0] == "trunc_normal") and (np.abs(th - b[3]) > 3 * b[4]):
                answer = input(
                    "The initial guess for "
                    + key
                    + " lies more than 3 sigma from its prior\n"
                    + "What do you want to do?\n"
                    + "1 : continue at any sacrifice \n"
                    + "2 : stop and let me fix the params.csv file \n"
                )
                if answer == 1:
                    pass
                else:
                    raise ValueError("User aborted the run.")

    ###############################################################################
    #::: load data
    ###############################################################################
    def load_data(self):
        """
        Example:
        -------
            A lightcurve is stored as
                data['TESS']['time'], data['TESS']['flux'], etc.
            A RV curve is stored as
                data['HARPS']['time'], data['HARPS']['flux'], etc.
        """
        self.fulldata = {}
        self.data = {}

        # ======================================================================
        #::: photometry
        # ======================================================================
        for inst in self.settings["inst_phot"]:
            time, flux, flux_err, custom_series, covariates = _load_inst_csv(
                os.path.join(self.datadir, inst + ".csv")
            )
            if any(np.isnan(time * flux * flux_err * custom_series)):
                raise ValueError(
                    'There are NaN values in "'
                    + inst
                    + '.csv". Please make sure everything is fine with your data, then exclude these rows from the file and restart.'
                )
            if any(flux_err == 0):
                raise ValueError(
                    'There are uncertainties with values of 0 in "'
                    + inst
                    + '.csv". Please make sure everything is fine with your data, then exclude these rows from the file and restart.'
                )
            if any(flux_err < 0):
                raise ValueError(
                    'There are uncertainties with negative values in "'
                    + inst
                    + '.csv". Please make sure everything is fine with your data, then exclude these rows from the file and restart.'
                )
            if not all(np.diff(time) >= 0):
                raise ValueError(
                    'The time array in "'
                    + inst
                    + '.csv" is not sorted. Please make sure the file is not corrupted, then sort it by time and restart.'
                )
            elif not all(np.diff(time) > 0):
                warnings.warn(
                    'There are repeated time stamps in the time array in "'
                    + inst
                    + '.csv". Please make sure the file is not corrupted (e.g. insuffiecient precision in your time stamps).',
                    stacklevel=2,
                )
            #                overwrite = str(input('There are repeated time stamps in the time array in "'+inst+'.csv". Please make sure the file is not corrupted (e.g. insuffiecient precision in your time stamps).'+\
            #                                      'What do you want to do?\n'+\
            #                                      '1 : continue and hope for the best; no risk, no fun; #yolo\n'+\
            #                                      '2 : abort\n'))
            #                if (overwrite == '1'):
            #                    pass
            #                else:
            #                    raise ValueError('User aborted operation.')

            #::: Raw-flux outlier removal (applied before fulldata is captured
            #::: so that all downstream consumers see the clipped rows).
            #::: Drops rows outside [flux_min_raw, flux_max_raw]; either bound
            #::: may be None to make it one-sided. Clipped points are kept
            #::: aside under ``raw_clipped_*`` so initial_guess plots can
            #::: surface them in red without affecting the fit.
            _fmin = self.settings.get("flux_min_raw")
            _fmax = self.settings.get("flux_max_raw")
            _clipped_time = np.empty(0, dtype=float)
            _clipped_flux = np.empty(0, dtype=float)
            _clipped_flux_err = np.empty(0, dtype=float)
            if _fmin is not None or _fmax is not None:
                _mask = np.ones_like(flux, dtype=bool)
                if _fmin is not None:
                    _mask &= flux >= _fmin
                if _fmax is not None:
                    _mask &= flux <= _fmax
                _n_drop = int(np.sum(~_mask))
                if _n_drop > 0:
                    warnings.warn(
                        f"{_n_drop}"
                        + f'/{len(flux)} rows in "'
                        + f'{inst}.csv" dropped by flux_min_raw={_fmin}, flux_max_raw={_fmax}.',
                        stacklevel=2,
                    )
                if not np.any(_mask):
                    raise ValueError(
                        'All rows in "' + inst + '.csv" were removed by flux_min_raw/flux_max_raw. '
                        "Check that the bounds bracket your normalized flux level."
                    )
                _clipped_time = time[~_mask]
                _clipped_flux = flux[~_mask]
                _clipped_flux_err = flux_err[~_mask]
                time = time[_mask]
                flux = flux[_mask]
                flux_err = flux_err[_mask]
                custom_series = custom_series[_mask]
                # keep named covariates row-aligned with flux
                covariates = {k: v[_mask] for k, v in covariates.items()}

            #::: Optional binning of the input light curve (in days). Applied
            #::: before `fulldata` so fulldata, fast_fit and covariate detrending
            #::: all see the binned series; covariates and custom_series are
            #::: mean-binned on the same grid (row-aligned). `raw_clipped_*`
            #::: (dropped outliers) are intentionally untouched.
            #::: A per-instrument `binning_<inst>` overrides the global `binning`
            #::: for this instrument only (explicit empty/None turns it off here
            #::: even when the global default bins everything else).
            _bkey = "binning_" + inst
            if _bkey in self.settings:
                _binning = self.settings[_bkey]
            else:
                _binning = self.settings["binning"]
            if _binning is not None:
                _span = float(time.max() - time.min()) if len(time) > 1 else 0.0
                if _binning >= _span:
                    raise ValueError(
                        f"binning ({_binning:g} d) must be smaller than the observation "
                        f'baseline of "{inst}.csv" ({_span:g} d). Pick a finer bin width.'
                    )
                time, flux, flux_err, custom_series, covariates = _bin_phot_arrays(
                    time, flux, flux_err, custom_series, covariates, _binning
                )

            self.fulldata[inst] = {
                "time": time,
                "flux": flux,
                "err_scales_flux": flux_err / np.nanmean(flux_err),
                "custom_series": custom_series,
                "covariates": dict(covariates),
                "raw_clipped_time": _clipped_time,
                "raw_clipped_flux": _clipped_flux,
                "raw_clipped_flux_err": _clipped_flux_err,
            }
            if (self.settings["fast_fit"]) and (len(self.settings["inst_phot"]) > 0):
                time, flux, flux_err, custom_series = self.reduce_phot_data(
                    time, flux, flux_err, custom_series=custom_series, inst=inst
                )
                # reduce_phot_data writes the kept indices to fulldata; use
                # them to keep every covariate aligned with the new time grid.
                _ind_in = self.fulldata[inst].get("all_ind_in")
                if _ind_in is not None and len(covariates):
                    covariates = {k: v[_ind_in] for k, v in covariates.items()}
            elif (self.settings["mask_transit"]) and (len(self.settings["inst_phot"]) > 0):
                time, flux, flux_err, custom_series = self.mask_phot_transit(
                    time, flux, flux_err, custom_series=custom_series, inst=inst
                )
                # mask_phot_transit writes the surviving (out-of-transit) indices
                # to fulldata; use them to keep every covariate row-aligned.
                _ind_out = self.fulldata[inst].get("all_ind_out")
                if _ind_out is not None and len(covariates):
                    covariates = {k: v[_ind_out] for k, v in covariates.items()}
            self.data[inst] = {
                "time": time,
                "flux": flux,
                "err_scales_flux": flux_err / np.nanmean(flux_err),
                "custom_series": custom_series,
                "covariates": dict(covariates),
                "raw_clipped_time": _clipped_time,
                "raw_clipped_flux": _clipped_flux,
                "raw_clipped_flux_err": _clipped_flux_err,
            }

        # ======================================================================
        #::: detect duplicate inst_phot input files (e.g. user accidentally
        #::: copied qlp1800.csv to qlp600.csv). Pairwise compare the
        #::: pre-fast-fit `fulldata` time+flux arrays — fulldata is the raw
        #::: read-in data, so identical content here is unambiguous.
        # ======================================================================
        _inst_phot_list = list(self.settings["inst_phot"])
        for _i in range(len(_inst_phot_list)):
            for _j in range(_i + 1, len(_inst_phot_list)):
                _a_inst, _b_inst = _inst_phot_list[_i], _inst_phot_list[_j]
                _a, _b = self.fulldata[_a_inst], self.fulldata[_b_inst]
                if (
                    len(_a["time"]) == len(_b["time"])
                    and np.array_equal(_a["time"], _b["time"])
                    and np.array_equal(_a["flux"], _b["flux"])
                ):
                    raise ValueError(
                        f'"{_a_inst}" and "{_b_inst}" contain identical data '
                        + "(N={}, time=[{:.4f}, {:.4f}]). Likely a duplicated file. ".format(
                            len(_a["time"]), float(_a["time"][0]), float(_a["time"][-1])
                        )
                        + "Please verify each inst_phot points to a distinct "
                        + "lightcurve, or remove the duplicate from inst_phot."
                    )

        # ======================================================================
        #::: RV
        # ======================================================================
        for inst in self.settings["inst_rv"]:
            time, rv, rv_err, custom_series, covariates = _load_inst_csv(
                os.path.join(self.datadir, inst + ".csv")
            )
            if any(np.isnan(time * rv * rv_err * custom_series)):
                raise ValueError(
                    'There are NaN values in "'
                    + inst
                    + '.csv". Please make sure everything is fine with your data, then exclude these rows from the file and restart.'
                )
            # aCkTuaLLLyy rv_err=0 is ok, since we add a jitter term here anyway (instead of scaling)
            # if any(rv_err==0):
            #     raise ValueError('There are uncertainties with values of 0 in "'+inst+'.csv". Please make sure everything is fine with your data, then exclude these rows from the file and restart.')
            if any(rv_err < 0):
                raise ValueError(
                    'There are uncertainties with negative values in "'
                    + inst
                    + '.csv". Please make sure everything is fine with your data, then exclude these rows from the file and restart.'
                )
            if not all(np.diff(time) > 0):
                raise ValueError(
                    'Your time array in "'
                    + inst
                    + '.csv" is not sorted. You will want to check that...'
                )
            self.data[inst] = {
                "time": time,
                "rv": rv,
                "white_noise_rv": rv_err,
                "custom_series": custom_series,
                "covariates": dict(covariates),
            }

        # ======================================================================
        #::: RV2 (for detached binaries)
        # ======================================================================
        for inst in self.settings["inst_rv2"]:
            time, rv, rv_err, custom_series, covariates = _load_inst_csv(
                os.path.join(self.datadir, inst + ".csv")
            )
            if not all(np.diff(time) > 0):
                raise ValueError(
                    'Your time array in "'
                    + inst
                    + '.csv" is not sorted. You will want to check that...'
                )
            self.data[inst] = {
                "time": time,
                "rv2": rv,
                "white_noise_rv2": rv_err,
                "custom_series": custom_series,
                "covariates": dict(covariates),
            }

        # ======================================================================
        #::: also save the combined time series
        #::: for cases where all instruments are treated together
        #::: e.g. for stellar variability GPs
        # ======================================================================
        self.data["inst_phot"] = {"time": [], "flux": [], "flux_err": [], "inst": []}
        for inst in self.settings["inst_phot"]:
            self.data["inst_phot"]["time"] += list(self.data[inst]["time"])
            self.data["inst_phot"]["flux"] += list(self.data[inst]["flux"])
            self.data["inst_phot"]["flux_err"] += [inst] * len(
                self.data[inst]["time"]
            )  # errors will be sampled/derived later
            self.data["inst_phot"]["inst"] += [inst] * len(self.data[inst]["time"])
        ind_sort = np.argsort(self.data["inst_phot"]["time"])
        self.data["inst_phot"]["ind_sort"] = ind_sort
        self.data["inst_phot"]["time"] = np.array(self.data["inst_phot"]["time"])[ind_sort]
        self.data["inst_phot"]["flux"] = np.array(self.data["inst_phot"]["flux"])[ind_sort]
        self.data["inst_phot"]["flux_err"] = np.array(self.data["inst_phot"]["flux_err"])[ind_sort]
        self.data["inst_phot"]["inst"] = np.array(self.data["inst_phot"]["inst"])[ind_sort]

        self.data["inst_rv"] = {"time": [], "rv": [], "rv_err": [], "inst": []}
        for inst in self.settings["inst_rv"]:
            self.data["inst_rv"]["time"] += list(self.data[inst]["time"])
            self.data["inst_rv"]["rv"] += list(self.data[inst]["rv"])
            self.data["inst_rv"]["rv_err"] += list(
                np.nan * self.data[inst]["rv"]
            )  # errors will be sampled/derived later
            self.data["inst_rv"]["inst"] += [inst] * len(self.data[inst]["time"])
        ind_sort = np.argsort(self.data["inst_rv"]["time"])
        self.data["inst_rv"]["ind_sort"] = ind_sort
        self.data["inst_rv"]["time"] = np.array(self.data["inst_rv"]["time"])[ind_sort]
        self.data["inst_rv"]["rv"] = np.array(self.data["inst_rv"]["rv"])[ind_sort]
        self.data["inst_rv"]["rv_err"] = np.array(self.data["inst_rv"]["rv_err"])[ind_sort]
        self.data["inst_rv"]["inst"] = np.array(self.data["inst_rv"]["inst"])[ind_sort]

        self.data["inst_rv2"] = {"time": [], "rv2": [], "rv2_err": [], "inst": []}
        for inst in self.settings["inst_rv2"]:
            self.data["inst_rv2"]["time"] += list(self.data[inst]["time"])
            self.data["inst_rv2"]["rv2"] += list(self.data[inst]["rv2"])
            self.data["inst_rv2"]["rv2_err"] += list(
                np.nan * self.data[inst]["rv2"]
            )  # errors will be sampled/derived later
            self.data["inst_rv2"]["inst"] += [inst] * len(self.data[inst]["time"])
        ind_sort = np.argsort(self.data["inst_rv2"]["time"])
        self.data["inst_rv2"]["ind_sort"] = ind_sort
        self.data["inst_rv2"]["time"] = np.array(self.data["inst_rv2"]["time"])[ind_sort]
        self.data["inst_rv2"]["rv2"] = np.array(self.data["inst_rv2"]["rv2"])[ind_sort]
        self.data["inst_rv2"]["rv2_err"] = np.array(self.data["inst_rv2"]["rv2_err"])[ind_sort]
        self.data["inst_rv2"]["inst"] = np.array(self.data["inst_rv2"]["inst"])[ind_sort]

    ###############################################################################
    #::: change epoch
    ###############################################################################

    def my_truncnorm_isf(q, a, b, mean, std):
        a_scipy = 1.0 * (a - mean) / std
        b_scipy = 1.0 * (b - mean) / std
        return truncnorm.isf(q, a_scipy, b_scipy, loc=mean, scale=std)

    def change_epoch(self):
        """
        change epoch entry from params.csv to set epoch into the middle of the range
        """

        self.logprint("\nShifting epochs into the data center")
        self.logprint("------------------------------------")
        # Echo the datadir so logs from multiple concurrent fits are
        # easy to attribute when grepped or tailed.
        self.logprint("datadir: " + os.path.abspath(self.datadir))

        #::: for all companions
        for companion in self.settings["companions_all"]:

            self.logprint("Companion", companion)
            self.logprint("\tinput epoch:", self.params[companion + "_epoch"])

            #::: get data time range
            alldata = []
            for inst in self.settings["inst_for_" + companion + "_epoch"]:
                alldata += list(self.data[inst]["time"])
            start = np.nanmin(alldata)
            end = np.nanmax(alldata)

            #::: get the given values
            user_epoch = 1.0 * self.params[companion + "_epoch"]
            period = 1.0 * self.params[companion + "_period"]
            #            buf = self.bounds[ind_e].copy()

            #::: calculate the true first_epoch
            if "fast_fit_width" in self.settings and self.settings["fast_fit_width"] is not None:
                width = self.settings["fast_fit_width"]
            else:
                width = 0
            first_epoch = get_first_epoch(
                alldata,
                self.params[companion + "_epoch"],
                self.params[companion + "_period"],
                width=width,
            )

            #::: calculate the mid_epoch (in the middle of the data set)
            N = int(np.round((end - start) / 2.0 / period))
            self.settings["mid_epoch"] = first_epoch + N * period

            #::: calculate how much the user_epoch has to be shifted to get the mid_epoch
            N_shift = int(np.round((self.settings["mid_epoch"] - user_epoch) / period))

            #::: set the new initial guess (and truth)
            self.params[companion + "_epoch"] = 1.0 * self.settings["mid_epoch"]

            #::: also shift the truth (implies that the turth epoch is set where the initial guess is)
            try:
                ind_e = np.where(self.fitkeys == companion + "_epoch")[0][0]
                ind_p = np.where(self.fitkeys == companion + "_period")[0][0]
                N_truth_shift = int(
                    np.round(
                        (self.settings["mid_epoch"] - self.fittruths[ind_e]) / self.fittruths[ind_p]
                    )
                )
                self.fittruths[ind_e] += N_truth_shift * self.fittruths[ind_p]
            except Exception:
                pass

            #::: if a fit param, also update the bounds accordingly
            if (N_shift != 0) and (companion + "_epoch" in self.fitkeys):
                ind_e = np.where(self.fitkeys == companion + "_epoch")[0][0]
                ind_p = np.where(self.fitkeys == companion + "_period")[0][0]

                #                print('\n')
                #                print('############################################################################')
                #                print('user_epoch', user_epoch, self.bounds[ind_e])
                #                print('user_period', period, self.bounds[ind_p])
                #                print('----------------------------------------------------------------------------')

                #::: set the new initial guess
                self.theta_0[ind_e] = 1.0 * self.settings["mid_epoch"]

                #::: get the bounds / errors
                #::: if the epoch and period priors are both uniform
                if (self.bounds[ind_e][0] == "uniform") & (self.bounds[ind_p][0] == "uniform"):
                    if N_shift > 0:
                        self.bounds[ind_e][1] = (
                            self.bounds[ind_e][1] + N_shift * self.bounds[ind_p][1]
                        )  # lower bound
                        self.bounds[ind_e][2] = (
                            self.bounds[ind_e][2] + N_shift * self.bounds[ind_p][2]
                        )  # upper bound
                    elif N_shift < 0:
                        self.bounds[ind_e][1] = (
                            self.bounds[ind_e][1] + N_shift * self.bounds[ind_p][2]
                        )  # lower bound; period bounds switched if N_shift is negative
                        self.bounds[ind_e][2] = (
                            self.bounds[ind_e][2] + N_shift * self.bounds[ind_p][1]
                        )  # upper bound; period bounds switched if N_shift is negative

                #::: if the epoch and period priors are both normal
                elif (self.bounds[ind_e][0] == "normal") & (self.bounds[ind_p][0] == "normal"):
                    self.bounds[ind_e][1] = (
                        self.bounds[ind_e][1] + N_shift * self.bounds[ind_p][1]
                    )  # mean (in case the prior-mean is not the initial-guess-mean)
                    self.bounds[ind_e][2] = np.sqrt(
                        self.bounds[ind_e][2] ** 2 + N_shift**2 * self.bounds[ind_p][2] ** 2
                    )  # std (in case the prior-mean is not the initial-guess-mean)

                #::: if the epoch and period priors are both trunc_normal
                elif (self.bounds[ind_e][0] == "trunc_normal") & (
                    self.bounds[ind_p][0] == "trunc_normal"
                ):
                    if N_shift > 0:
                        self.bounds[ind_e][1] = (
                            self.bounds[ind_e][1] + N_shift * self.bounds[ind_p][1]
                        )  # lower bound
                        self.bounds[ind_e][2] = (
                            self.bounds[ind_e][2] + N_shift * self.bounds[ind_p][2]
                        )  # upper bound
                    elif N_shift < 0:
                        self.bounds[ind_e][1] = (
                            self.bounds[ind_e][1] + N_shift * self.bounds[ind_p][2]
                        )  # lower bound; period bounds switched if N_shift is negative
                        self.bounds[ind_e][2] = (
                            self.bounds[ind_e][2] + N_shift * self.bounds[ind_p][1]
                        )  # upper bound; period bounds switched if N_shift is negative
                    self.bounds[ind_e][3] = (
                        self.bounds[ind_e][3] + N_shift * self.bounds[ind_p][3]
                    )  # mean (in case the prior-mean is not the initial-guess-mean)
                    self.bounds[ind_e][4] = np.sqrt(
                        self.bounds[ind_e][4] ** 2 + N_shift**2 * self.bounds[ind_p][4] ** 2
                    )  # std (in case the prior-mean is not the initial-guess-mean)

                #::: if the epoch prior is uniform and period prior is normal
                elif (self.bounds[ind_e][0] == "uniform") & (self.bounds[ind_p][0] == "normal"):
                    self.bounds[ind_e][1] = self.bounds[ind_e][1] + N_shift * (
                        period + self.bounds[ind_p][2]
                    )  # lower bound epoch + Nshift * period + Nshift * std_period
                    self.bounds[ind_e][2] = self.bounds[ind_e][2] + N_shift * (
                        period + self.bounds[ind_p][2]
                    )  # upper bound + Nshift * period + Nshift * std_period

                #::: if the epoch prior is uniform and period prior is trunc_normal
                elif (self.bounds[ind_e][0] == "uniform") & (
                    self.bounds[ind_p][0] == "trunc_normal"
                ):
                    self.bounds[ind_e][1] = self.bounds[ind_e][1] + N_shift * (
                        period + self.bounds[ind_p][4]
                    )  # lower bound epoch + Nshift * period + Nshift * std_period
                    self.bounds[ind_e][2] = self.bounds[ind_e][2] + N_shift * (
                        period + self.bounds[ind_p][4]
                    )  # upper bound + Nshift * period + Nshift * std_period

                elif (self.bounds[ind_e][0] == "normal") & (self.bounds[ind_p][0] == "uniform"):
                    raise ValueError(
                        "shift_epoch with different priors for epoch and period is not yet implemented."
                    )

                elif (self.bounds[ind_e][0] == "normal") & (
                    self.bounds[ind_p][0] == "trunc_normal"
                ):
                    raise ValueError(
                        "shift_epoch with different priors for epoch and period is not yet implemented."
                    )

                elif (self.bounds[ind_e][0] == "trunc_normal") & (
                    self.bounds[ind_p][0] == "uniform"
                ):
                    raise ValueError(
                        "shift_epoch with different priors for epoch and period is not yet implemented."
                    )

                elif (self.bounds[ind_e][0] == "trunc_normal") & (
                    self.bounds[ind_p][0] == "normal"
                ):
                    raise ValueError(
                        "shift_epoch with different priors for epoch and period is not yet implemented."
                    )

                else:
                    raise ValueError(
                        'Parameters "bounds" have to be "uniform", "normal" or "trunc_normal".'
                    )

                self.logprint("\tshifted epoch:", self.params[companion + "_epoch"])
                self.logprint("\tshifted by", N_shift, "periods")

    ###############################################################################
    #::: reduce_phot_data
    ###############################################################################
    def reduce_phot_data(self, time, flux, flux_err, custom_series=None, inst=None):
        ind_in = []

        for companion in self.settings["companions_phot"]:
            epoch = self.params[companion + "_epoch"]
            period = self.params[companion + "_period"]
            width = self.settings["fast_fit_width"]
            if self.settings["secondary_eclipse"]:
                ind_ecl1x, ind_ecl2x, ind_outx = index_eclipses(
                    time, epoch, period, width, width
                )  # TODO: currently this assumes width_occ == width_tra
                ind_in += list(ind_ecl1x)
                ind_in += list(ind_ecl2x)
                self.fulldata[inst][companion + "_ind_ecl1"] = ind_ecl1x
                self.fulldata[inst][companion + "_ind_ecl2"] = ind_ecl2x
                self.fulldata[inst][companion + "_ind_out"] = ind_outx
            else:
                ind_inx, ind_outx = index_transits(time, epoch, period, width)
                ind_in += list(ind_inx)
                self.fulldata[inst][companion + "_ind_in"] = ind_inx
                self.fulldata[inst][companion + "_ind_out"] = ind_outx

        ind_in = np.sort(np.unique(ind_in))
        self.fulldata[inst]["all_ind_in"] = ind_in
        self.fulldata[inst]["all_ind_out"] = np.delete(
            np.arange(len(self.fulldata[inst]["time"])), ind_in
        )

        if len(ind_in) == 0:
            raise ValueError(
                inst
                + ".csv does not contain any in-transit data. Check that your epoch and period guess are correct."
            )

        time = time[ind_in]
        flux = flux[ind_in]
        flux_err = flux_err[ind_in]
        if custom_series is None:
            return time, flux, flux_err
        else:
            custom_series = custom_series[ind_in]
            return time, flux, flux_err, custom_series

    ###############################################################################
    #::: mask_phot_transit (inverse of reduce_phot_data)
    ###############################################################################
    def mask_phot_transit(self, time, flux, flux_err, custom_series=None, inst=None):
        """
        Remove the in-transit points, keeping only the out-of-transit data.

        This is the inverse of reduce_phot_data() and is used when
        ``mask_transit==True`` so the user can model stellar variability or
        spots on data that no longer contains the transit signal. The transit
        window half-width is taken from ``fast_fit_width`` (the same window
        fast_fit uses). The surviving (out-of-transit) indices are written to
        ``self.fulldata[inst]`` so load_data() can keep covariates row-aligned.
        """
        ind_in = []

        for companion in self.settings["companions_phot"]:
            epoch = self.params[companion + "_epoch"]
            period = self.params[companion + "_period"]
            width = self.settings["fast_fit_width"]
            if self.settings["secondary_eclipse"]:
                ind_ecl1x, ind_ecl2x, ind_outx = index_eclipses(
                    time, epoch, period, width, width
                )  # TODO: currently this assumes width_occ == width_tra
                ind_in += list(ind_ecl1x)
                ind_in += list(ind_ecl2x)
            else:
                ind_inx, ind_outx = index_transits(time, epoch, period, width)
                ind_in += list(ind_inx)

        ind_in = np.sort(np.unique(ind_in))
        ind_out = np.delete(np.arange(len(time)), ind_in)
        self.fulldata[inst]["all_ind_in"] = ind_in
        self.fulldata[inst]["all_ind_out"] = ind_out

        if len(ind_out) == 0:
            raise ValueError(
                inst
                + ".csv contains only in-transit data after masking; nothing is left to model. Check that your epoch/period guess and fast_fit_width are correct."
            )

        time = time[ind_out]
        flux = flux[ind_out]
        flux_err = flux_err[ind_out]
        if custom_series is None:
            return time, flux, flux_err
        else:
            custom_series = custom_series[ind_out]
            return time, flux, flux_err, custom_series

    ###############################################################################
    #::: cross-file validation: baseline_<key>_<inst>_against must name a real column
    ###############################################################################
    def validate_baseline_against_covariates(self):
        """Confirm every ``baseline_<key>_<inst>_against`` value names a
        loaded column. Run after :meth:`load_data` so per-inst covariate
        dicts are populated.

        Raises ``ValueError`` with an actionable message if a setting
        points to a column that isn't in ``self.data[inst]['covariates']``
        and isn't one of the legacy fixed names (``time``,
        ``custom_series``).
        """
        for key, key2 in zip(["flux", "rv", "rv2"], ["inst_phot", "inst_rv", "inst_rv2"]):
            for inst in self.settings.get(key2, []):
                skey = "baseline_" + key + "_" + inst + "_against"
                against = self.settings.get(skey, "time")
                if against in ("time", "custom_series"):
                    continue
                covs = self.data.get(inst, {}).get("covariates", {})
                if against in covs:
                    continue
                known = ["time", "custom_series"] + sorted(covs.keys())
                raise ValueError(
                    f"{skey}={against!r} but {inst}.csv has no column named "
                    f"'{against}'. Known options for this instrument: {known}. "
                    f"Add a `#time,{key}_err,...,{against},...` header line to "
                    f"{inst}.csv to expose the column."
                )

    ###############################################################################
    #::: allocate fit-vector entries for every sample_linear_multi baseline
    ###############################################################################
    def synthesize_linear_multi_params(self):
        """Build the per-instrument design matrix and inject one fit
        parameter per column for every ``sample_linear_multi`` baseline.

        Called after :meth:`load_data` so per-inst covariate arrays are
        already populated. New weights are appended to ``self.fitkeys``
        / ``self.theta_0`` / ``self.bounds`` etc. with a default
        ``normal 0 1e3`` prior (matching timex), unless the user
        pre-declared the row in ``params.csv`` — in which case the
        user's prior is honoured and only the column is registered in
        the design-matrix metadata.

        Mirrors timex's ``pm.Normal('{name}_weights', mu=0, sd=1e3,
        shape=X.shape[1])`` + ``pt.dot(X, weights)`` pattern.
        """
        new_keys, new_labels, new_units, new_truths = [], [], [], []
        new_theta0, new_bounds, new_init_err = [], [], []
        for key, key2 in zip(["flux", "rv", "rv2"], ["inst_phot", "inst_rv", "inst_rv2"]):
            for inst in self.settings.get(key2, []):
                btype = self.settings.get("baseline_" + key + "_" + inst, "none")
                if btype not in ("sample_linear_multi", "hybrid_linear_multi"):
                    continue
                cols_setting = self.settings.get("baseline_" + key + "_" + inst + "_cols", "")
                tokens = [t for t in str(cols_setting).strip().split() if t]
                if not tokens:
                    raise ValueError(
                        f"baseline_{key}_{inst}={btype} requires baseline_{key}_{inst}_cols "
                        "to list one or more covariate names (and optionally "
                        "the 'bias' token), space-separated."
                    )
                time_axis = self.data[inst]["time"]
                X, cols_resolved = _build_linear_design_matrix(self.data[inst], tokens, time_axis)
                self.data[inst]["design_matrix"] = X
                self.data[inst]["design_matrix_cols"] = cols_resolved
                # Tier 2 (hybrid_linear_multi) marginalises the weights
                # analytically — no fit-vector rows synthesized.
                if btype == "hybrid_linear_multi":
                    continue
                for col in cols_resolved:
                    pname = "baseline_linmulti_" + col + "_" + key + "_" + inst
                    if pname in self.params:
                        # User-declared row honoured — register only.
                        continue
                    self.params[pname] = 0.0
                    new_keys.append(pname)
                    new_labels.append("$w_{" + col + ";" + inst + "}$")
                    new_units.append("")
                    new_truths.append(np.nan)
                    new_theta0.append(0.0)
                    new_bounds.append(["normal", 0.0, 1e3])
                    new_init_err.append(1e-2)
        if new_keys:
            self.allkeys = np.concatenate([self.allkeys, np.array(new_keys, dtype=object)])
            self.coupled_with = list(self.coupled_with) + [None] * len(new_keys)
            self.fitkeys = np.concatenate([self.fitkeys, np.array(new_keys, dtype=object)])
            self.fitlabels = np.concatenate([self.fitlabels, np.array(new_labels, dtype=object)])
            self.fitunits = np.concatenate([self.fitunits, np.array(new_units, dtype=object)])
            self.fittruths = np.concatenate([self.fittruths, np.array(new_truths, dtype=float)])
            self.theta_0 = np.concatenate([self.theta_0, np.array(new_theta0, dtype=float)])
            if np.isscalar(self.init_err):
                self.init_err = np.full(self.ndim, float(self.init_err))
            self.init_err = np.concatenate([self.init_err, np.array(new_init_err, dtype=float)])
            self.bounds = list(self.bounds) + new_bounds
            self.ndim = len(self.theta_0)

    ###############################################################################
    #::: setup TTV fit (if chosen)
    ###############################################################################
    def setup_ttv_fit(self):
        """
        this must be run *after* reduce_phot_data()
        """

        #::: the window we choose to look for transits is determined by fast_fit_width
        window = self.settings["fast_fit_width"]

        #::: for each companion, stitch together all the time stamps observed by all photometric instruments
        #::: and check which of these times overlap with a potential transit window (determined by fast_fit_width)
        for companion in self.settings["companions_phot"]:
            times_combined = []
            for inst in self.settings["inst_phot"]:
                times_combined += list(self.data[inst]["time"])
            times_combined = np.sort(times_combined)

            self.data[companion + "_tmid_observed_transits"] = get_tmid_observed_transits(
                times_combined,
                self.params[companion + "_epoch"],
                self.params[companion + "_period"],
                window,
            )

            for inst in self.settings["inst_phot"]:
                time = self.data[inst]["time"]
                for i, t in enumerate(self.data[companion + "_tmid_observed_transits"]):
                    ind = np.where((time >= (t - window / 2.0)) & (time <= (t + window / 2.0)))[0]
                    self.data[inst][companion + "_ind_time_transit_" + str(i + 1)] = ind
                    self.data[inst][companion + "_time_transit_" + str(i + 1)] = time[ind]

            #::: THE FOLLOWING PART MOVED INTO THE SEPARATE SCRIPT "PREPARE_TTV_FIT.PY"
            #::: plots
            # if self.settings['fit_ttvs']:
            #     flux_min = np.nanmin(all_flux)
            #     flux_max = np.nanmax(all_flux)
            #     N_days = int( np.max(all_times) - np.min(all_times) )
            #     figsizex = np.min( [1, int(N_days/20.)] )*5
            #     fig, ax = plt.subplots(figsize=(figsizex, 4)) #figsize * 5 for every 20 days
            #     for inst in self.settings['inst_phot']:
            #         ax.plot(self.data[inst]['time'], self.data[inst]['flux'],ls='none',marker='.',label=inst)
            #     ax.plot( self.data[companion+'_tmid_observed_transits'], np.ones_like(self.data[companion+'_tmid_observed_transits'])*0.995*flux_min, 'k^' )
            #     for i, tmid in enumerate(self.data[companion+'_tmid_observed_transits']):
            #         ax.text( tmid, 0.9925*flux_min, str(i+1), ha='center' )
            #     ax.set(ylim=[0.99*flux_min, flux_max], xlabel='Time (BJD)', ylabel='Relative Flux')
            #     if not os.path.exists( os.path.join(self.datadir,'results') ):
            #         os.makedirs(os.path.join(self.datadir,'results'))
            #     ax.legend()
            #     fname = os.path.join(self.datadir,'results','preparation_for_TTV_fit_'+companion+'.pdf')
            #     if os.path.exists(fname):
            #         overwrite = str(input('Figure "preparation_for_TTV_fit_'+companion+'.pdf" already exists.\n'+\
            #                               'What do you want to do?\n'+\
            #                               '1 : overwrite it\n'+\
            #                               '2 : skip it and move on\n'))
            #         if (overwrite == '1'):
            #             fig.savefig(fname, bbox_inches='tight' )
            #         else:
            #             pass
            #     plt.close(fig)

    ###############################################################################
    #::: apply flattened-flux outlier clip
    ###############################################################################
    def apply_flat_clip(self):
        """
        Drops rows from each photometric instrument whose detrended flux falls
        outside [flux_min_flat, flux_max_flat]. The trend is computed from the
        initial-guess parameters using the same primitives that
        `show_initial_guess` plots:
            flat = flux - calculate_baseline(...) - calculate_stellar_var(...)
        Must be called AFTER config.BASEMENT has been assigned, because
        calculate_baseline/calculate_stellar_var read config.BASEMENT.{settings,data}.
        Re-runs setup_ttv_fit() if fit_ttvs is enabled, since per-transit
        index arrays depend on the data length.
        """
        fmin = self.settings.get("flux_min_flat")
        fmax = self.settings.get("flux_max_flat")
        if fmin is None and fmax is None:
            return

        # Local imports to avoid a circular import at module load time
        # (computer.py imports config which imports basement).
        from . import config as _config
        from .computer import (
            calculate_baseline,
            calculate_model,
            calculate_stellar_var,
        )

        if _config.BASEMENT is not self:
            # Defensive: this method needs the global BASEMENT to point at us
            # so the calculate_* helpers see the right data/settings.
            warnings.warn(
                "apply_flat_clip(): config.BASEMENT is not this Basement; " "skipping flat clip.",
                stacklevel=2,
            )
            return

        for inst in self.settings["inst_phot"]:
            try:
                model = calculate_model(self.params, inst, "flux")
                baseline = calculate_baseline(self.params, inst, "flux", model=model)
                stellar_var = calculate_stellar_var(
                    self.params,
                    inst,
                    "flux",
                    model=model,
                    baseline=baseline,
                )
            except Exception as e:
                warnings.warn(
                    f'apply_flat_clip(): could not compute trend for "{inst}" '
                    f"({e}); skipping flat clip for this instrument.",
                    stacklevel=2,
                )
                continue

            flux = self.data[inst]["flux"]
            flat = flux - baseline - stellar_var

            mask = np.ones_like(flat, dtype=bool)
            if fmin is not None:
                mask &= flat >= fmin
            if fmax is not None:
                mask &= flat <= fmax

            n_drop = int(np.sum(~mask))
            if n_drop == 0:
                continue
            if not np.any(mask):
                raise ValueError(
                    'All rows in "' + inst + '" were removed by flux_min_flat/flux_max_flat. '
                    "Check the bounds against your detrended flux level."
                )
            warnings.warn(
                f"{n_drop}"
                + f'/{len(flat)} rows in "'
                + f'{inst}" dropped by flux_min_flat={fmin}, flux_max_flat={fmax} '
                + "(applied to flux - baseline - stellar_var from initial guess).",
                stacklevel=2,
            )

            for k in ("time", "flux", "err_scales_flux", "custom_series"):
                if k in self.data[inst]:
                    self.data[inst][k] = self.data[inst][k][mask]

        if self.settings.get("fit_ttvs"):
            try:
                self.setup_ttv_fit()
            except Exception as e:
                warnings.warn(
                    f"apply_flat_clip(): setup_ttv_fit() failed after clip ({e}).", stacklevel=2
                )

    ###############################################################################
    #::: stellar priors
    ###############################################################################
    def load_stellar_priors(self, N_samples=10000):
        if os.path.exists(os.path.join(self.datadir, "params_star.csv")) and (
            self.settings["use_host_density_prior"] is True
        ):
            buf = np.genfromtxt(
                os.path.join(self.datadir, "params_star.csv"),
                delimiter=",",
                names=True,
                dtype=None,
                encoding="utf-8",
                comments="#",
            )
            radius = (
                simulate_PDF(
                    buf["R_star"],
                    buf["R_star_lerr"],
                    buf["R_star_uerr"],
                    size=N_samples,
                    plot=False,
                )
                * 6.957e10
            )  # in cgs
            mass = (
                simulate_PDF(
                    buf["M_star"],
                    buf["M_star_lerr"],
                    buf["M_star_uerr"],
                    size=N_samples,
                    plot=False,
                )
                * 1.9884754153381438e33
            )  # in cgs
            volume = (4.0 / 3.0) * np.pi * radius**3  # in cgs
            density = mass / volume  # in cgs
            self.params_star = {
                "R_star_median": buf["R_star"],
                "R_star_lerr": buf["R_star_lerr"],
                "R_star_uerr": buf["R_star_uerr"],
                "M_star_median": buf["M_star"],
                "M_star_lerr": buf["M_star_lerr"],
                "M_star_uerr": buf["M_star_uerr"],
            }
            self.external_priors["host_density"] = [
                "normal",
                np.median(density),
                np.max(
                    [
                        np.median(density) - np.percentile(density, 16),
                        np.percentile(density, 84) - np.median(density),
                    ]
                ),
            ]  # in cgs
