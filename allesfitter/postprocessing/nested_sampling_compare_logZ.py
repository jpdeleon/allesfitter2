#!/usr/bin/env python2
"""
Created on Tue Oct 23 14:11:05 2018

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
import sys
from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

try:
    import cPickle as pickle
except Exception:
    import pickle
#::: plotting settings
import seaborn as sns
from dynesty import utils as dyutils
from tqdm import tqdm

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


def ns_plot_bayes_factors(datadirs, labels=None, return_dlogZ=False, ax=None, explanation=False):
    """
    Inputs:
    -------
    datadirs : list of str (see Example 1) OR tuple of lists of str (see Example 2)
        all the directories from which
        the first run_name must be the "null hypothesis"

    labels : list of str
        all the labels for the plot


    Outputs:
    --------
    fig : matplotlib.Figure object

    ax : matplotlib.Axes object


    Example 1:
    ---------
    #::: just do a single model comparison
    datadirs = ['circular_model', 'eccentric_model']
    labels = ['circular', 'eccentric']
    fig, ax = ns_compare_logZ(datadirs, labels)


    Example 2:
    ----------
    #::: do multiple model comparisons in one plot
    datadirs_1 = ['circular_model', 'eccentric_model']
    labels_1 = ['circular', 'eccentric']

    datadirs_2 = ['no_occultation_model', 'occultation_model']
    labels_2 = ['without occultation', 'with occulation']

    collection_of_datadirs = ( datadirs_1, datadirs_2 )
    collection_of_labels = ( labels_1, labels_2 )

    fig, ax = ns_compare_logZ(datadirs, labels)
    """
    if labels is None:
        labels = datadirs

    if isinstance(datadirs, list):
        delta_logZ, delta_logZ_err, delta_labels = get_delta_logZ_and_delta_labels(datadirs, labels)
    elif isinstance(datadirs, tuple):
        delta_logZ, delta_logZ_err, delta_labels = get_collective_delta_logZ_and_delta_labels(
            datadirs, labels
        )
    else:
        raise ValueError("datadirs must be tuple or list.")

    #::: plot
    index = np.arange(len(delta_logZ))
    if ax is None:
        fig, ax = plt.subplots(figsize=(3 * len(datadirs), 4))
    else:
        fig = plt.gcf()
    ax.bar(index, delta_logZ, edgecolor="b")
    ax.errorbar(
        index,
        delta_logZ,
        yerr=delta_logZ_err,
        color="k",
        linestyle="none",
        markersize=0,
        capsize=2,
        elinewidth=5,
        zorder=10,
    )
    ax.set_xticks(index)
    ax.set_xticklabels(delta_labels)

    # Jeffreys limits
    #    ax.axhspan(np.nanmin(logZ)+np.log(10**0.5),np.nanmin(logZ)+np.log(10**1),color='g',zorder=-1,alpha=0.2)
    #    ax.axhspan(np.nanmin(logZ)+np.log(10**1),np.nanmin(logZ)+np.log(10**1.5),color='g',zorder=-1,alpha=0.4)
    #    ax.axhspan(np.nanmin(logZ)+np.log(10**1.5),np.nanmin(logZ)+np.log(10**2),color='g',zorder=-1,alpha=0.6)
    #    ax.axhspan(np.nanmin(logZ)+np.log(10**2),np.nanmin(logZ)+np.log(10**4),color='g',zorder=-1,alpha=0.8)
    #    ax2 = ax.twinx()
    #    ax2.set_yticks( [np.nanmin(logZ)+np.log(10**(i-0.25)) for i in [0.5,1.,1.5,2.,2.5]] )
    #    ax2.set_yticklabels( ['no evidence','substantial','strong','very strong','decisive'] )

    # Kass and Raftery limits
    #    ax.axhspan(np.nanmin(logZ)+1,np.nanmin(logZ)+3,color='g',zorder=-1,alpha=0.3)
    #    ax.axhspan(np.nanmin(logZ)+3,np.nanmin(logZ)+5,color='g',zorder=-1,alpha=0.55)
    #    ax.axhspan(np.nanmin(logZ)+5,np.nanmin(logZ)+20,color='g',zorder=-1,alpha=0.8)
    #    ax2 = ax.twinx()
    #    ax2.set_yticks( [np.nanmin(logZ)+i for i in [0.5,2,4,6]] )
    #    ax2.set_yticklabels( ['no evidence','positive','strong','very strong'] )

    # Kass and Raftery limits
    #    ax.axhspan(np.nanmin(logZ)+1,np.nanmin(logZ)+3,color='g',zorder=-1,alpha=0.3)

    ymax = np.nanmax(list(1.1 * delta_logZ) + [7])
    ax.axhspan(3, 5, color="g", zorder=-1, alpha=0.33)
    ax.axhspan(5, ymax, color="g", zorder=-1, alpha=0.66)
    if explanation:
        ax.text(index[-1] + 1, 1.5, "no strong\nevidence", va="center")
        ax.text(index[-1] + 1, 4, "strong\nevidence", va="center")
        ax.text(
            index[-1] + 1,
            np.max([(np.max(delta_logZ) + 5.0) / 2.0, 6.0]),
            "very strong\nevidence",
            va="center",
        )
    ax.set(ylim=[0, ymax], ylabel=r"$\Delta \ln{Z}$")

    if return_dlogZ:
        return fig, ax, delta_logZ
    else:
        return fig, ax


def get_delta_logZ_and_delta_labels(datadirs, labels):
    logZ, logZ_err = get_logZ(datadirs)

    #::: calculate delta_logZ
    delta_logZ = np.array(logZ) - logZ[0]
    delta_logZ_err = np.sqrt(np.array(logZ_err) ** 2 + np.array(logZ_err[0]) ** 2)

    #::: remove the null hypothesis from the plot
    delta_logZ = delta_logZ[1:]
    delta_logZ_err = delta_logZ_err[1:]
    delta_labels = [labels[i + 1] + "\nvs.\n" + labels[0] for i in range(len(delta_logZ))]

    return delta_logZ, delta_logZ_err, delta_labels


def get_logZ(datadirs, quiet=False):
    logZ = []
    logZ_err = []

    for dirname in np.atleast_1d(datadirs):
        fname, results = _load_results(dirname)
        if not quiet:
            print("--------------------------")
            print(fname)

        #::: get the results
        logz = _result_value(results, "logz")
        logzerr = _result_value(results, "logzerr")
        logZdynesty = logz[-1]
        logZerrdynesty = logzerr[-1]

        #::: recalculate logZ error if it was NaN (bug in dynesty 0.9.2b)
        if (
            np.isnan(logZerrdynesty)
            or np.isinf(logZerrdynesty)
            or (logZerrdynesty / logZdynesty > 1)
        ):
            if not quiet:
                print("recalculating ln(Z) error...")
            sys.stdout.flush()
            logvol = _result_value(results, "logvol")
            lnzs = np.zeros((10, len(logvol)))
            for i in tqdm(range(10), disable=quiet):
                results_s = dyutils.simulate_run(results)
                lnzs[i] = np.interp(-logvol, -results_s.logvol, results_s.logz)
            lnzerr = np.std(lnzs, axis=0)
            logZerrdynesty = lnzerr[-1]

        if not quiet:
            print(f"ln(Z) = {logZdynesty} +- {logZerrdynesty}")

        logZ.append(logZdynesty)
        logZ_err.append(logZerrdynesty)

    return logZ, logZ_err


def _result_value(results, key):
    """Read a dynesty field from modern mappings or legacy Results objects."""
    if isinstance(results, dict):
        return np.asarray(results[key])
    return np.asarray(getattr(results, key))


def _load_results(path):
    """Load a nested-sampling result from a run directory or pickle path."""
    path = Path(path)
    if path.is_file():
        candidates = [path]
    else:
        candidates = [
            path / "ns_results" / "save_ns.pickle.gz",
            path / "results" / "save_ns.pickle.gz",
            path / "ns_results" / "save_ns.pickle",
            path / "results" / "save_ns.pickle",
        ]
    fname = next((candidate for candidate in candidates if candidate.is_file()), None)
    if fname is None:
        raise FileNotFoundError(f"No nested-sampling results found for {path}")
    opener = gzip.open if fname.suffix == ".gz" else open
    with opener(fname, "rb") as stream:
        return str(fname), pickle.load(stream)


def compare_logz(datadirs, labels=None):
    """Compare final dynesty log-evidences against the first (reference) run.

    Returns a list of dictionaries containing ``logz``, ``logzerr``, and the
    propagated ``delta_logz`` uncertainty. ``datadirs`` may contain run
    directories or direct paths to dynesty pickle files.
    """
    datadirs = list(datadirs)
    if not datadirs:
        raise ValueError("At least one nested-sampling result is required")
    if labels is None:
        labels = [str(path) for path in datadirs]
    if len(labels) != len(datadirs):
        raise ValueError("labels and datadirs must have the same length")

    logz, logzerr = get_logZ(datadirs, quiet=True)
    reference_logz = logz[0]
    reference_error = logzerr[0]
    return [
        {
            "label": label,
            "logz": float(value),
            "logzerr": float(error),
            "delta_logz": float(value - reference_logz),
            "delta_logzerr": float(np.hypot(error, reference_error)),
        }
        for label, value, error in zip(labels, logz, logzerr)
    ]


def main(argv=None):
    """Print a log-evidence comparison table for nested-sampling runs."""
    parser = ArgumentParser(description="Compare dynesty log-evidences")
    parser.add_argument("datadirs", nargs="+", help="run directories or result pickle files")
    parser.add_argument("--labels", nargs="+", help="labels matching the input order")
    args = parser.parse_args(argv)
    rows = compare_logz(args.datadirs, args.labels)
    print("label\tlogz\tlogzerr\tdelta_logz\tdelta_logzerr")
    for row in rows:
        print(
            f"{row['label']}\t{row['logz']:.6g}\t{row['logzerr']:.6g}\t"
            f"{row['delta_logz']:.6g}\t{row['delta_logzerr']:.6g}"
        )
    return rows


def get_collective_delta_logZ_and_delta_labels(collection_of_datadirs, collection_of_labels):
    """
    Example:
    --------
    datadirs_1 = ['circular_model', 'eccentric_model']
    labels_1 = ['circular', 'eccentric']

    datadirs_2 = ['no_occultation_model', 'occultation_model']
    labels_2 = ['without occultation', 'with occulation']

    collection_of_datadirs = ( datadirs_1, datadirs_2 )
    collection_of_labels = ( labels_1, labels_2 )

    delta_logZ, delta_logZ_err, delta_labels = \
        get_collective_delta_logZ_and_delta_labels(collection_of_datadirs, collection_of_labels)
    """
    delta_logZ, delta_logZ_err, delta_labels = [], [], []
    for datadirs, labels in zip(collection_of_datadirs, collection_of_labels):
        a, b, c = get_delta_logZ_and_delta_labels(datadirs, labels)
        delta_logZ += list(a)
        delta_logZ_err += list(b)
        delta_labels += list(c)
    return delta_logZ, delta_logZ_err, delta_labels


if __name__ == "__main__":
    main()
