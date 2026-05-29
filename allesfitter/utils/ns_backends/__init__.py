"""Nested-sampling backend dispatch for allesfitter.

allesfitter ships with two interchangeable nested-sampling backends:

* ``dynesty`` (default) — pure-Python, the historical implementation.
* ``ultranest`` — reactive nested sampling with MLFriends; optional extra.

Both backends share a single, picklable result schema (``NSResults``) so
``ns_output`` and ``allesclass`` do not need to know which sampler ran.

Schema
------
``NSResults`` is a ``dict`` subclass that also exposes attribute-style
access (``r.logz`` ≡ ``r['logz']``). This preserves the historical
``results.logz[-1]`` ergonomics from dynesty's own ``Results`` class.

Required keys::

    backend        : 'dynesty' | 'ultranest'
    samples        : (Nsamples, ndim) raw (unweighted) parameter samples
    logwt          : (Nsamples,) log-importance weights, aligned with samples
    logz           : (Niter,) cumulative log-evidence; [-1] is the final value
    logzerr        : (Niter,) uncertainty on logz; [-1] is the final value
    fitkeys        : (ndim,) parameter names, in column order of samples
    wall_time_sec  : float, sampler wall time
    raw            : optional backend-native object (may be None)

The first four keys reproduce the dynesty ``Results`` access pattern
used throughout ``nested_sampling_output.py`` and ``allesclass``.
"""

from __future__ import annotations

from typing import Callable, Iterable

import numpy as np


__all__ = [
    "NSResults",
    "get_backend",
    "KNOWN_BACKENDS",
    "BACKEND_SETTINGS",
    "validate_settings_for_backend",
]


KNOWN_BACKENDS = ("dynesty", "ultranest")


# Settings.csv keys consumed by each backend. Keys NOT listed here are
# logged as "ignored" so users do not silently waste rows. Keys listed here
# but absent from settings.csv are logged as "missing — using default" so
# users are nudged toward explicit, reproducible configuration.
BACKEND_SETTINGS = {
    "dynesty": {
        "relevant": {"ns_nlive", "ns_bound", "ns_sample", "ns_tol", "ns_modus"},
        "ignored":  {"un_min_ess", "un_max_iters"},
    },
    "ultranest": {
        "relevant": {"ns_nlive", "ns_tol", "un_min_ess", "un_max_iters"},
        # ns_modus/ns_bound/ns_sample are dynesty concepts; ultranest is
        # reactive and chooses its own region/sampling strategy.
        "ignored":  {"ns_modus", "ns_bound", "ns_sample"},
    },
}


def validate_settings_for_backend(backend, raw_keys, logprint=print):
    """Warn about backend-irrelevant or implicit nested-sampling settings.

    Parameters
    ----------
    backend : str
        Resolved backend name (e.g. ``'dynesty'``).
    raw_keys : iterable of str
        The keys that were explicitly present in the user's settings.csv
        (before allesfitter filled in defaults).
    logprint : callable
        Logging sink — typically ``allesfitter.general_output.logprint``.

    Behaviour
    ---------
    * Emits a clear warning for any key listed in ``BACKEND_SETTINGS[backend]['ignored']``
      that the user *did* set — it has no effect under this backend.
    * Emits a clear warning for any key listed in ``BACKEND_SETTINGS[backend]['relevant']``
      that the user did *not* set — they are running on the package default
      and should add an explicit row for reproducibility.
    """
    spec = BACKEND_SETTINGS.get(backend)
    if spec is None:
        return
    raw = set(raw_keys)
    ignored_but_set = sorted(spec["ignored"] & raw)
    relevant_but_missing = sorted(spec["relevant"] - raw)

    if ignored_but_set:
        logprint(
            "\n! settings.csv: the following keys are IGNORED by ns_backend='{}': {}".format(
                backend, ", ".join(ignored_but_set)
            )
        )
        logprint("  → remove them, or switch backend, to avoid confusion.")

    if relevant_but_missing:
        logprint(
            "\n! settings.csv: ns_backend='{}' uses these keys but they are NOT set "
            "(running on defaults): {}".format(
                backend, ", ".join(relevant_but_missing)
            )
        )
        logprint("  → add explicit rows to settings.csv for reproducible fits.")

    if not ignored_but_set and not relevant_but_missing:
        logprint("\nsettings.csv: all ns_backend='{}' keys explicitly set. ✓".format(backend))


class NSResults(dict):
    """Picklable, backend-agnostic nested-sampling result container.

    Subclasses ``dict`` and forwards attribute access to item access so
    legacy ``results.logz[-1]`` callers keep working alongside the new
    ``results['logz'][-1]`` style. ``copy.deepcopy`` is supported via the
    standard dict protocol — that is critical for
    ``nested_sampling_output.py`` which deep-copies before in-place edits.
    """

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        try:
            del self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


def _coerce(arr) -> np.ndarray:
    return np.asarray(arr)


def build_results(
    *,
    backend: str,
    samples,
    logwt,
    logz,
    logzerr,
    fitkeys: Iterable[str],
    wall_time_sec: float,
    raw=None,
) -> NSResults:
    """Construct an ``NSResults`` from backend-specific arrays."""
    return NSResults(
        backend=backend,
        samples=_coerce(samples),
        logwt=_coerce(logwt),
        logz=_coerce(logz),
        logzerr=_coerce(logzerr),
        fitkeys=np.asarray(list(fitkeys)),
        wall_time_sec=float(wall_time_sec),
        raw=raw,
    )


def get_backend(name: str):
    """Lazy-import a backend module by name.

    Lazy imports keep ``ultranest`` an optional dependency: users who only
    use ``dynesty`` never trigger an ``ultranest`` import.
    """
    name = (name or "dynesty").lower()
    if name == "dynesty":
        from . import dynesty_backend as be
        return be
    if name == "ultranest":
        from . import ultranest_backend as be
        return be
    raise ValueError(
        "Unknown ns backend: {!r}. Choose one of {}.".format(name, KNOWN_BACKENDS)
    )
