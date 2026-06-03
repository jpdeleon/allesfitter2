"""Configuration validation for allesfitter2.

Two layers, intentionally separate:

- :mod:`~allesfitter.validation.config_checks` — *structural* checks that
  catch unambiguous ``params.csv`` / ``settings.csv`` errors and raise
  :class:`ConfigError`.
- :mod:`~allesfitter.validation.prior_sanity` — *heuristic* GP / noise prior
  checks that only warn.

Import the public surface from here::

    from allesfitter.validation import validate_params_settings, validate_gp_priors
"""

from __future__ import annotations

from .config_checks import (
    ConfigError,
    check_bounds_wellformed,
    check_companions_have_params,
    check_duplicate_param_names,
    check_fit_flags,
    check_values_numeric,
    check_values_within_bounds,
    collect_config_errors,
    companions_from_settings,
    parse_bounds,
    read_param_rows,
    read_settings,
    validate_params_settings,
)
from .prior_sanity import (
    _compute_tdur_hours_by_companion,
    _tdur_days_from_orbit,
    validate_gp_priors,
)

__all__ = [
    # structural config validation
    "ConfigError",
    "validate_params_settings",
    "collect_config_errors",
    "check_duplicate_param_names",
    "check_fit_flags",
    "check_values_numeric",
    "check_bounds_wellformed",
    "check_values_within_bounds",
    "check_companions_have_params",
    "companions_from_settings",
    "parse_bounds",
    "read_param_rows",
    "read_settings",
    # heuristic prior sanity
    "validate_gp_priors",
    "_compute_tdur_hours_by_companion",
    "_tdur_days_from_orbit",
]
