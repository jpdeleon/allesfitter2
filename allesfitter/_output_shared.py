#!/usr/bin/env python3
"""
Shared output helpers for the MCMC and Nested Sampling result paths.

``mcmc_output`` and ``ns_output`` historically carried byte-identical copies of
several plotting/table routines that drifted out of sync over time. This module
is the single home for the logic that genuinely does not differ between the two
samplers; the per-path modules call into it and differ only where the sampler
itself differs (e.g. the posterior-sample array and the output-filename prefix).

Currently shared here:
    - ``save_per_transit_plots`` : per-transit fit plots for photometric fits
    - ``write_priors_latex_table`` : LaTeX prior table built from ``params.csv``
"""

import csv
import os

import matplotlib.pyplot as plt
import numpy as np

from . import config
from ._figure_output import normalize_file_extension
from .general_output import afplot_per_transit


def save_per_transit_plots(posterior_samples, prefix, file_extension=".pdf"):
    """Render and save per-transit fit plots for every photometric companion.

    Shared by ``mcmc_output`` and ``ns_output``; the two paths differ only in
    the posterior-sample array passed in and the output-filename prefix.

    Parameters
    ----------
    posterior_samples : array-like
        The (subsampled) posterior draws used for plotting, exactly as passed
        to :func:`allesfitter.general_output.afplot_per_transit`.
    prefix : str
        Output-filename prefix, ``'mcmc'`` or ``'ns'``. Files are written as
        ``<outdir>/<prefix>_fit_per_transit_<inst>_<companion>_<N>th.pdf``.

    Notes
    -----
    Behaviour is preserved from the original in-line loops: any plotting error
    ends the per-transit loop for that companion/instrument pair (previously a
    bare ``except:``; narrowed to ``except Exception`` so ``KeyboardInterrupt``
    and ``SystemExit`` still propagate).
    """
    file_extension = normalize_file_extension(file_extension)
    for companion in config.BASEMENT.settings["companions_phot"]:
        for inst in config.BASEMENT.settings["inst_phot"]:
            first_transit = 0
            while first_transit >= 0:
                try:
                    fig, axes, last_transit, total_transits = afplot_per_transit(
                        posterior_samples,
                        inst,
                        companion,
                        kwargs_dict={"first_transit": first_transit},
                    )
                    fig.savefig(
                        os.path.join(
                            config.BASEMENT.outdir,
                            prefix
                            + "_fit_per_transit_"
                            + inst
                            + "_"
                            + companion
                            + "_"
                            + str(last_transit)
                            + "th"
                            + file_extension,
                        ),
                        bbox_inches="tight",
                    )
                    plt.close(fig)
                    if total_transits > 0 and last_transit < total_transits - 1:
                        first_transit = last_transit
                    else:
                        first_transit = -1
                except Exception:
                    #::: any plotting error ends the per-transit loop exactly as
                    #::: before; KeyboardInterrupt / SystemExit still propagate.
                    first_transit = -1


def save_ttv_csv(posterior_samples, outpath=None):
    """Write posterior transit times to ttv.csv.

    epoch is the integer orbital epoch relative to the configured reference
    epoch. tc_unc(BJD) is half the 16th-to-84th percentile interval of the
    full transit-time posterior, including uncertainty and covariance from
    the reference epoch, period, and fitted TTV offset.
    """
    base = config.BASEMENT
    if not base.settings.get("fit_ttvs", False):
        return False

    samples = np.asarray(posterior_samples, dtype=float)
    if samples.ndim == 1:
        samples = samples[None, :]
    if samples.ndim != 2 or samples.shape[1] != len(base.fitkeys):
        raise ValueError(
            "posterior_samples must have shape (n_samples, len(config.BASEMENT.fitkeys))"
        )

    columns = {str(key): samples[:, i] for i, key in enumerate(base.fitkeys)}

    def _draws(key):
        if key in columns:
            return columns[key]
        try:
            value = base.params[key]
        except KeyError as exc:
            raise KeyError(f"Cannot create ttv.csv: missing parameter {key!r}") from exc
        return np.full(samples.shape[0], float(value))

    rows = []
    for companion in base.settings["companions_phot"]:
        epoch_key = companion + "_epoch"
        period_key = companion + "_period"
        reference_epoch = float(base.params[epoch_key])
        reference_period = float(base.params[period_key])
        observed_midtimes = base.data[companion + "_tmid_observed_transits"]

        for i, observed_midtime in enumerate(observed_midtimes, start=1):
            epoch = int(np.rint((float(observed_midtime) - reference_epoch) / reference_period))
            ttv_key = companion + "_ttv_transit_" + str(i)
            tc_draws = _draws(epoch_key) + epoch * _draws(period_key) + _draws(ttv_key)
            p16, median, p84 = np.nanpercentile(tc_draws, [16, 50, 84])
            rows.append((companion, epoch, median, 0.5 * (p84 - p16)))

    if outpath is None:
        outpath = os.path.join(base.outdir, "ttv.csv")
    with open(outpath, "w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["planet", "epoch", "tc(BJD)", "tc_unc(BJD)"])
        writer.writerows(rows)
    return outpath


def write_priors_latex_table(datadir=None, outpath=None):
    """Convert ``params.csv`` into a LaTeX prior table.

    Walks the user's ``params.csv``, keeps only rows with ``fit==1``, and
    writes a booktabs-style ``tabular`` block to
    ``<outdir>/priors_latex_table.txt`` (or ``outpath`` when provided).

    Prior encoding
    --------------
    ``uniform LO HI``                   -> ``\\mathcal{U}(LO,\\, HI)``
    ``normal MU SIGMA``                 -> ``\\mathcal{N}(MU,\\, SIGMA)``
    ``trunc_normal LO HI MU SIGMA``     -> ``\\mathcal{N}_T(MU,\\, SIGMA;\\, LO,\\, HI)``

    Unrecognised bounds strings are written through verbatim so unusual
    priors (e.g., custom Jeffreys) still appear in the table.

    Parameters
    ----------
    datadir : str, optional
        Working directory (must contain ``params.csv``). Defaults to
        ``config.BASEMENT.datadir`` when called from inside ``ns_output``.
    outpath : str, optional
        Destination filename. Defaults to
        ``os.path.join(config.BASEMENT.outdir, 'priors_latex_table.txt')``.

    Returns
    -------
    str or False
        The output path on success; ``False`` when ``params.csv`` is
        missing or contained no fittable rows.
    """
    if datadir is None:
        datadir = config.BASEMENT.datadir
    if outpath is None:
        outpath = os.path.join(config.BASEMENT.outdir, "priors_latex_table.txt")

    params_path = os.path.join(datadir, "params.csv")
    if not os.path.exists(params_path):
        return False

    rows = []
    with open(params_path) as f:
        for raw in f:
            line = raw.rstrip("\n")
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 4:
                continue
            name = parts[0]
            try:
                fit = int(parts[2])
            except (TypeError, ValueError):
                fit = 0
            if fit != 1:
                continue
            bounds = parts[3]
            label = parts[4] if len(parts) > 4 and parts[4] else name.replace("_", r"\_")
            unit = parts[5] if len(parts) > 5 and parts[5] else ""
            rows.append((name, bounds, label, unit))

    if not rows:
        return False

    def _prior_tex(bounds_str):
        tokens = bounds_str.split()
        if len(tokens) == 3 and tokens[0] == "uniform":
            lo, hi = tokens[1], tokens[2]
            return r"$\mathcal{U}(" + lo + r",\," + hi + r")$"
        if len(tokens) == 3 and tokens[0] == "normal":
            mu, sigma = tokens[1], tokens[2]
            return r"$\mathcal{N}(" + mu + r",\," + sigma + r")$"
        if len(tokens) == 5 and tokens[0] == "trunc_normal":
            lo, hi, mu, sigma = tokens[1], tokens[2], tokens[3], tokens[4]
            return r"$\mathcal{N}_T(" + mu + r",\," + sigma + r";\," + lo + r",\," + hi + r")$"
        return bounds_str  # unknown encoding — pass through verbatim

    out_lines = [
        r"% Prior table -- auto-generated by allesfitter.ns_output",
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Priors for fitted parameters.}",
        r"\label{tab:priors}",
        r"\begin{tabular}{lll}",
        r"\hline",
        r"Parameter & Prior & Unit \\",
        r"\hline",
    ]
    for _name, bounds, label, unit in rows:
        out_lines.append(f"{label} & {_prior_tex(bounds)} & {unit} \\\\")
    out_lines += [r"\hline", r"\end{tabular}", r"\end{table}"]

    with open(outpath, "w") as f:
        f.write("\n".join(out_lines) + "\n")
    return outpath
