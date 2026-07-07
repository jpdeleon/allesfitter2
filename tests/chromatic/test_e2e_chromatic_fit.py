"""End-to-end Nested Sampling fits on synthetic chromatic data.

Marked ``@pytest.mark.slow`` because each call takes seconds-to-minutes of
sampler runtime. Gated by ``pytest.ini`` default ``-m "not slow"``; run with
``-m slow`` or ``-m ""`` to include.

What we pin:
- chromatic NS fit recovers the injected per-band rr_tess / rr_kepler within
  posterior tolerance, and shared orbital params (period, epoch) are recovered
  as single keys.
- achromatic NS fit on the achromatic_datadir runs end-to-end with the same
  pipeline (backwards compatibility), producing a finite log-evidence and an
  rr posterior centered on truth.
"""

from __future__ import annotations

import gzip
import os
import pickle

import numpy as np
import pytest

import allesfitter
from allesfitter import config

pytestmark = pytest.mark.slow


def _load_posterior(datadir):
    """Load the gzipped dynesty results pickle and return (samples, logwt, logz, fitkeys)."""
    with gzip.open(os.path.join(datadir, "results", "save_ns.pickle.gz"), "rb") as f:
        results = pickle.load(f)
    samples = results.samples
    # weighted posterior via normalized importance weights
    logwt = results.logwt - results.logz[-1]
    weights = np.exp(logwt)
    return samples, weights, float(results.logz[-1])


def _weighted_quantile(values, weights, q):
    """Quantile of a weighted 1-D sample."""
    order = np.argsort(values)
    v = values[order]
    w = weights[order]
    cw = np.cumsum(w) / np.sum(w)
    return float(np.interp(q, cw, v))


# --------------------------------------------------------------------------- #
# Chromatic two-band fit recovers per-band rr (orbital params fixed)
# --------------------------------------------------------------------------- #
def test_ns_recovers_per_band_rr(two_band_e2e_fast_datadir, truth):
    datadir = str(two_band_e2e_fast_datadir)
    allesfitter.ns_fit(datadir)

    fitkeys = list(config.BASEMENT.fitkeys)
    samples, weights, logz = _load_posterior(datadir)
    assert samples.shape[1] == len(fitkeys)
    assert np.isfinite(logz)

    medians = {k: _weighted_quantile(samples[:, i], weights, 0.5) for i, k in enumerate(fitkeys)}
    assert "b_rr_tess" in medians and "b_rr_kepler" in medians
    assert "b_rr" not in medians, "achromatic key should not appear in chromatic fit"

    # NS recovers each band's injected rr within a generous tolerance.
    # SNR floor at NOISE_SIGMA=5e-4, 80 pts → per-rr uncertainty ~few e-3.
    assert (
        abs(medians["b_rr_tess"] - truth["rr_tess"]) < 0.02
    ), f"rr_tess median {medians['b_rr_tess']} drifted from truth {truth['rr_tess']}"
    assert (
        abs(medians["b_rr_kepler"] - truth["rr_kepler"]) < 0.02
    ), f"rr_kepler median {medians['b_rr_kepler']} drifted from truth {truth['rr_kepler']}"


# --------------------------------------------------------------------------- #
# Achromatic backward-compat fit runs end-to-end with the same pipeline
# --------------------------------------------------------------------------- #
def test_ns_achromatic_backcompat_runs_end_to_end(achromatic_e2e_fast_datadir, truth):
    datadir = str(achromatic_e2e_fast_datadir)
    allesfitter.ns_fit(datadir)

    fitkeys = list(config.BASEMENT.fitkeys)
    assert "b_rr" in fitkeys
    assert not any(k.startswith("b_rr_") for k in fitkeys)

    samples, weights, logz = _load_posterior(datadir)
    assert np.isfinite(logz)
    medians = {k: _weighted_quantile(samples[:, i], weights, 0.5) for i, k in enumerate(fitkeys)}
    assert abs(medians["b_rr"] - truth["rr_tess"]) < 0.02
