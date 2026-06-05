"""Configuration validation for allesfitter2.

Two layers, intentionally separate:

- :mod:`~allesfitter.validation.config_checks` — *structural* checks that
  catch unambiguous ``params.csv`` / ``settings.csv`` errors and raise
  :class:`ConfigError`.
- :mod:`~allesfitter.validation.prior_checks` — *heuristic* GP / noise prior
  checks that only warn.

Both layers share the low-level readers and bounds parser in
:mod:`~allesfitter.validation.parsing`.

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
    check_gp_baseline_vs_stellar_var,
    check_values_numeric,
    check_values_within_bounds,
    collect_config_errors,
    companions_from_settings,
    validate_params_settings,
)
from .parsing import (
    parse_bounds,
    parse_uniform_bounds,
    read_csv_rows,
    read_param_rows,
    read_settings,
)
from .prior_checks import (
    check_secondary_eclipse_sbratio,
    transit_duration_days,
    transit_duration_hours_by_companion,
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
    "check_gp_baseline_vs_stellar_var",
    "companions_from_settings",
    # shared parsing helpers
    "parse_bounds",
    "parse_uniform_bounds",
    "read_csv_rows",
    "read_param_rows",
    "read_settings",
    # heuristic prior checks
    "validate_gp_priors",
    "check_secondary_eclipse_sbratio",
    "transit_duration_days",
    "transit_duration_hours_by_companion",
]
