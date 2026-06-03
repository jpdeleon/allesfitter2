"""Backward-compatible shim for the relocated prior-sanity validator.

The implementation now lives in :mod:`allesfitter.validation.prior_sanity`
(part of the consolidated :mod:`allesfitter.validation` package). This module
re-exports the full public *and* private surface so existing imports such as::

    from allesfitter.utils.prior_sanity import validate_gp_priors
    from allesfitter.utils.prior_sanity import _tdur_days_from_orbit

keep working unchanged. Prefer importing from ``allesfitter.validation`` in new
code.
"""

from __future__ import annotations

from allesfitter.validation.prior_sanity import (  # noqa: F401
    GP_LNRHO_PREFIX,
    GP_LNSIGMA_PREFIX,
    LN_ERR_PREFIX,
    _compute_tdur_hours_by_companion,
    _tdur_days_from_orbit,
    validate_gp_priors,
)

__all__ = [
    "validate_gp_priors",
    "_compute_tdur_hours_by_companion",
    "_tdur_days_from_orbit",
    "GP_LNSIGMA_PREFIX",
    "GP_LNRHO_PREFIX",
    "LN_ERR_PREFIX",
]
