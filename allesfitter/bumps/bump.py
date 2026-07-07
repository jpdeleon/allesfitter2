"""
Hold the analytic "bump" model for starspot-crossing events.

A planet transiting over a cool starspot momentarily blocks less flux than over
the quiet photosphere, producing a small localized brightening ("bump") inside
the transit. This is modeled here as a simple additive Gaussian in time, in
relative-flux units (mirrors the approach used in timex).
"""

import numpy as np


def bump_model(t, tpeak, width, ampl):
    """
    Gaussian bump for a starspot-crossing event.

        bump(t) = ampl * exp(-(t - tpeak)**2 / (2 * width**2))

    The bump is additive and expressed in relative-flux units, so it is added on
    top of the transit/eclipse model (centered at 1.0) by the caller.

    Parameters
    ----------
    t : array-like
        Time array to evaluate the bump over (BJD).
    tpeak : float
        Time at the center of the bump (BJD).
    width : float
        Gaussian standard deviation (days).
    ampl : float
        Peak amplitude of the bump (relative flux).

    Returns
    -------
    bump : array-like
        The bump flux evaluated at each time point.
    """
    return ampl * np.exp(-((t - tpeak) ** 2) / (2.0 * width**2))
