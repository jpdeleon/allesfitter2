"""Single source of truth for *hard* physical limits on model parameters.

Some allesfitter model parameters have unambiguous physical ranges that a valid
configuration can never violate, e.g. ``<c>_rsuma`` = (Rs+Rp)/a must lie in
(0, 1] because the semi-major axis cannot be smaller than the sum of the radii.
Neither validation surface enforced these:

- :func:`allesfitter.basement.Basement.load_params` validates only the *initial
  guess* value, and most upper bounds were ``np.inf``;
- :mod:`allesfitter.validation.config_checks` validated prior *structure*
  (``lo < hi``, ``sigma > 0``, value-in-bounds) but never the *physical* range.

This module holds the limits once so both surfaces consult the same registry:

- :mod:`allesfitter.validation.config_checks` uses :func:`check_value` (initial
  value) and :func:`lookup_limit` (prior bounds) to *raise* on violations;
- :func:`allesfitter.basement.Basement.load_params` uses :func:`lookup_limit`
  to source the ``(min, max)`` for its initial-value ``validate`` calls.

Only **unambiguous** constraints live here. Debatable ones (``<c>_rr`` > 1 for
eclipsing binaries, ``<c>_lambda`` range, non-transiting geometry) are
warning-only and live in :mod:`allesfitter.validation.prior_checks`.

Every function is pure (string/number in, string/None out) so the rules are
trivially unit-testable without constructing a :class:`~allesfitter.basement.Basement`.
"""

from __future__ import annotations

import fnmatch
import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Limit:
    """A closed/half-open physical interval for a model parameter.

    ``lo_inclusive`` / ``hi_inclusive`` select ``<=`` vs ``<`` at each end so
    that e.g. dilution can be ``[0, 1)`` (1 means a fully diluted, signal-free
    light curve) while ``rsuma`` can be ``(0, 1]``.
    """

    lo: float
    hi: float
    lo_inclusive: bool = True
    hi_inclusive: bool = True
    label: str = ""

    def contains(self, value: float) -> bool:
        """True when ``value`` lies inside the interval (respecting endpoints)."""
        lo_ok = value >= self.lo if self.lo_inclusive else value > self.lo
        hi_ok = value <= self.hi if self.hi_inclusive else value < self.hi
        return lo_ok and hi_ok

    def describe(self) -> str:
        """Human-readable interval, e.g. ``(0, 1]`` or ``[0, inf)``."""
        lo_br = "[" if self.lo_inclusive else "("
        hi_br = "]" if self.hi_inclusive else ")"
        return f"{lo_br}{self.lo:g}, {self.hi:g}{hi_br}"


# Glob pattern -> Limit. Matched with fnmatch against the full parameter name;
# the suffix carries the companion letter / instrument (e.g. ``b_rsuma``,
# ``dil_Leonardo``, ``host_vsini``). Patterns are mutually exclusive — a name
# should match at most one entry.
_PHYSICAL_LIMITS: dict[str, Limit] = {
    # (Rs+Rp)/a: a >= Rs+Rp so the ratio cannot exceed 1; 0 is degenerate.
    "*_rsuma": Limit(0.0, 1.0, lo_inclusive=False, label="(Rs+Rp)/a"),
    # Dilution is a flux fraction; 1 leaves no signal to fit.
    "dil_*": Limit(0.0, 1.0, hi_inclusive=False, label="dilution fraction"),
    # Projected rotation speed cannot be negative.
    "*_vsini": Limit(0.0, math.inf, label="vsini"),
}


def lookup_limit(name: str) -> Limit | None:
    """Return the :class:`Limit` registered for ``name``, or ``None``.

    Matches the parameter ``name`` against the registry's glob patterns with
    :func:`fnmatch.fnmatch`. Returns the first match (patterns are designed to
    be mutually exclusive).
    """
    for pattern, limit in _PHYSICAL_LIMITS.items():
        if fnmatch.fnmatch(name, pattern):
            return limit
    return None


def check_value(name: str, value: float) -> str | None:
    """Return an error string if ``value`` violates ``name``'s physical limit.

    Returns ``None`` when ``name`` has no registered limit, when ``value`` is
    non-finite (NaN/inf placeholders are handled elsewhere), or when ``value``
    lies inside the limit.
    """
    limit = lookup_limit(name)
    if limit is None:
        return None
    if not math.isfinite(value):
        return None
    if limit.contains(value):
        return None
    label = f" ({limit.label})" if limit.label else ""
    return f"'{name}'{label} value {value:g} is outside its physical range {limit.describe()}."


def eccentricity_error(companion: str, f_s: float, f_c: float) -> str | None:
    """Return an error if ``f_s**2 + f_c**2 >= 1`` (eccentricity e >= 1).

    allesfitter parametrizes eccentricity as ``f_s = sqrt(e) sin(omega)`` and
    ``f_c = sqrt(e) cos(omega)``, so ``e = f_s**2 + f_c**2``. A bound orbit
    requires ``e < 1``. Each of ``f_s`` / ``f_c`` is individually bounded to
    ``[-1, 1]``, which alone would admit ``e`` up to 2 — this cross-parameter
    check closes that gap. Returns ``None`` when either input is non-finite
    (the parameter is unset / fixed to a placeholder).
    """
    if not (math.isfinite(f_s) and math.isfinite(f_c)):
        return None
    ecc = f_s * f_s + f_c * f_c
    if ecc < 1.0:
        return None
    return (
        f"companion '{companion}' has f_s={f_s:g}, f_c={f_c:g} giving "
        f"eccentricity e=f_s^2+f_c^2={ecc:g} >= 1 (unbound orbit); "
        f"require f_s^2 + f_c^2 < 1."
    )
