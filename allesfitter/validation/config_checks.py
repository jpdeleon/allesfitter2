"""Structural sanity checks for ``params.csv`` and ``settings.csv``.

Unlike :mod:`allesfitter.validation.prior_checks` (heuristic, warning-only),
these checks catch *unambiguous* configuration errors that would otherwise
fail late, silently, or with a confusing traceback deep inside the sampler:

- duplicate parameter rows (numpy.genfromtxt silently last-wins),
- malformed ``fit`` flags (must be ``0`` or ``1``),
- non-numeric initial values,
- malformed / unrecognized prior bounds (``uniform``, ``normal``,
  ``trunc_normal``), inverted bounds, non-positive sigma,
- an initial value that falls outside its own (uniform / truncated) prior,
- a companion declared in ``settings.csv`` that has *no* parameter rows in
  ``params.csv`` — the cross-file consistency check,
- a GP baseline combined with a GP stellar-variability model on the same key
  (forbidden; ``computer.py`` would raise deep inside the likelihood).

Every function is **pure**: it takes already-parsed rows / dicts and returns
a list of human-readable error strings. :func:`validate_params_settings`
reads a datadir, runs every check, and (by default) raises
:class:`ConfigError` if any check failed. This split keeps each rule trivially
unit-testable without constructing a full :class:`~allesfitter.basement.Basement`.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

from .parsing import parse_bounds, read_param_rows, read_settings

# Rows whose ``name`` is one of these are section separators in params.csv,
# not real parameters; they carry no value/fit/bounds and must be skipped.
_SENTINEL_NAMES = {"user-given:", "automatically set:"}

# settings.csv keys that hold a space-separated list of companion letters.
_COMPANION_KEYS = ("companions_phot", "companions_rv")

# Values of ``stellar_var_<key>`` / ``baseline_<key>_<inst>`` that select a
# Gaussian Process model (mirrors ``GPs`` in :mod:`allesfitter.computer`).
_GP_VALUES = {
    "sample_GP_real",
    "sample_GP_complex",
    "sample_GP_Matern32",
    "sample_GP_SHO",
}

# ``stellar_var_<key>`` pairs with the instrument-list setting ``<key2>`` that
# enumerates the instruments whose ``baseline_<key>_<inst>`` is checked — the
# same key/key2 zip used in :func:`allesfitter.computer.calculate_lnlike_total`.
_STELLAR_VAR_KEY_TO_INST_KEY = {
    "flux": "inst_phot",
    "rv": "inst_rv",
    "rv2": "inst_rv2",
}


class ConfigError(ValueError):
    """Raised when ``params.csv`` / ``settings.csv`` fail a structural check.

    Subclasses :class:`ValueError` so existing ``except ValueError`` handlers
    around config loading keep working.
    """


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------
#
# Low-level CSV reading (``read_csv_rows``, ``read_param_rows``,
# ``read_settings``) and bounds parsing (``parse_bounds``) live in
# :mod:`allesfitter.validation.parsing` and are imported above; they are shared
# with :mod:`allesfitter.validation.prior_checks`.


def companions_from_settings(settings: Dict[str, str]) -> List[str]:
    """Union of companion letters declared across the companion keys."""
    seen: List[str] = []
    for key in _COMPANION_KEYS:
        for tok in (settings.get(key, "") or "").split():
            if tok and tok not in seen:
                seen.append(tok)
    return seen


def _is_real_param(row: Sequence[str]) -> bool:
    """A row that names an actual parameter (not a sentinel / blank name)."""
    return bool(row) and row[0] != "" and row[0] not in _SENTINEL_NAMES


def _is_coupled(row: Sequence[str]) -> bool:
    """A coupled row inherits its value/bounds from a partner parameter.

    allesfitter's 7th column (``coupled_with``) names that partner; such rows
    legitimately leave ``value`` and ``bounds`` blank, so the numeric / bounds
    checks must skip them.
    """
    return len(row) >= 7 and row[6].strip() != ""


# ---------------------------------------------------------------------------
# Individual checks — each returns a list of error strings (empty == ok)
# ---------------------------------------------------------------------------


def check_duplicate_param_names(rows: Sequence[Sequence[str]]) -> List[str]:
    """Flag parameter names that appear on more than one row."""
    counts: Dict[str, int] = {}
    for row in rows:
        if not _is_real_param(row):
            continue
        counts[row[0]] = counts.get(row[0], 0) + 1
    dups = sorted(name for name, n in counts.items() if n > 1)
    if dups:
        return [
            "params.csv contains duplicate rows for: "
            + ", ".join(dups)
            + ". Each parameter must be defined exactly once."
        ]
    return []


def check_fit_flags(rows: Sequence[Sequence[str]]) -> List[str]:
    """The ``fit`` column (3rd) must be ``0`` or ``1`` on every real row."""
    errors: List[str] = []
    for row in rows:
        if not _is_real_param(row) or len(row) < 3:
            continue
        flag = row[2]
        if flag not in ("0", "1"):
            errors.append(
                f"params.csv row '{row[0]}' has fit flag '{flag}'; "
                f"expected '0' (fixed) or '1' (fitted)."
            )
    return errors


def check_values_numeric(rows: Sequence[Sequence[str]]) -> List[str]:
    """The ``value`` column (2nd) must hold a usable initial value.

    Empty values are allowed on fixed (``fit==0``) or coupled rows — allesfitter
    fills those from defaults or the coupled partner. A fitted row, however,
    must carry a numeric starting point, and any non-empty value must parse as
    a float (catches typos like ``0..1``).
    """
    errors: List[str] = []
    for row in rows:
        if not _is_real_param(row) or len(row) < 2 or _is_coupled(row):
            continue
        value = row[1].strip()
        fitted = len(row) >= 3 and row[2] == "1"
        if value == "":
            if fitted:
                errors.append(
                    f"params.csv row '{row[0]}' is fitted (fit=1) but has an "
                    f"empty initial value."
                )
            continue
        try:
            float(value)
        except (TypeError, ValueError):
            errors.append(
                f"params.csv row '{row[0]}' has non-numeric value '{row[1]}'."
            )
    return errors


def check_bounds_wellformed(rows: Sequence[Sequence[str]]) -> List[str]:
    """Validate prior bounds for *fitted* rows.

    Fixed rows (``fit==0``) are ignored — their bounds are unused. For fitted
    rows the bounds must be present, use a recognized keyword, have the right
    arity, satisfy ``lo < hi`` (uniform / truncated) and ``sigma > 0``
    (normal / truncated).
    """
    errors: List[str] = []
    for row in rows:
        if not _is_real_param(row) or len(row) < 4 or _is_coupled(row):
            continue
        if row[2] != "1":  # only fitted rows need a usable prior
            continue
        name = row[0]
        parsed = parse_bounds(row[3])
        if parsed is None:
            errors.append(
                f"params.csv row '{name}' is fitted (fit=1) but has no prior bounds."
            )
            continue
        kind, nums = parsed
        if kind == "__unknown__":
            errors.append(
                f"params.csv row '{name}' has unrecognized prior '{row[3]}'; "
                f"expected one of uniform / normal / trunc_normal."
            )
            continue
        if kind == "__bad__":
            errors.append(
                f"params.csv row '{name}' has malformed prior bounds '{row[3]}'."
            )
            continue
        if kind == "uniform":
            lo, hi = nums
            if not lo < hi:
                errors.append(
                    f"params.csv row '{name}' uniform bounds require lo < hi, "
                    f"got lo={lo}, hi={hi}."
                )
        elif kind == "normal":
            _mean, sigma = nums
            if not sigma > 0:
                errors.append(
                    f"params.csv row '{name}' normal prior requires sigma > 0, "
                    f"got sigma={sigma}."
                )
        elif kind == "trunc_normal":
            lo, hi, _mean, sigma = nums
            if not lo < hi:
                errors.append(
                    f"params.csv row '{name}' trunc_normal bounds require lo < hi, "
                    f"got lo={lo}, hi={hi}."
                )
            if not sigma > 0:
                errors.append(
                    f"params.csv row '{name}' trunc_normal prior requires sigma > 0, "
                    f"got sigma={sigma}."
                )
    return errors


def check_values_within_bounds(rows: Sequence[Sequence[str]]) -> List[str]:
    """A fitted row's initial value must lie inside its uniform/truncated prior.

    A starting point outside the prior support has zero likelihood under the
    sampler and is almost always a typo. ``normal`` priors are unbounded, so
    they are skipped here.
    """
    errors: List[str] = []
    for row in rows:
        if not _is_real_param(row) or len(row) < 4 or row[2] != "1" or _is_coupled(row):
            continue
        try:
            value = float(row[1])
        except (TypeError, ValueError):
            continue  # caught by check_values_numeric
        parsed = parse_bounds(row[3])
        if parsed is None:
            continue
        kind, nums = parsed
        if kind == "uniform":
            lo, hi = nums
        elif kind == "trunc_normal":
            lo, hi = nums[0], nums[1]
        else:
            continue
        if not (lo <= value <= hi):
            errors.append(
                f"params.csv row '{row[0]}' initial value {value} lies outside "
                f"its prior [{lo}, {hi}]."
            )
    return errors


def check_companions_have_params(
    settings: Dict[str, str], rows: Sequence[Sequence[str]]
) -> List[str]:
    """Cross-file consistency: every declared companion must have param rows.

    A companion letter listed in ``settings.csv`` (``companions_phot`` /
    ``companions_rv``) but with no ``<c>_*`` row in ``params.csv`` means the
    two files disagree — the fit would proceed with that body entirely
    unparameterized.
    """
    names = {row[0] for row in rows if _is_real_param(row)}
    errors: List[str] = []
    for companion in companions_from_settings(settings):
        prefix = companion + "_"
        if not any(n.startswith(prefix) for n in names):
            errors.append(
                f"settings.csv declares companion '{companion}' but params.csv "
                f"has no '{prefix}*' rows for it."
            )
    return errors


def check_gp_baseline_vs_stellar_var(settings: Dict[str, str]) -> List[str]:
    """Forbid a GP baseline and a GP stellar-variability model on the same key.

    For a given key (``flux`` / ``rv`` / ``rv2``), allesfitter cannot combine a
    GP stellar-variability model with a GP baseline: ``computer.py`` raises a
    ``KeyError`` deep inside the likelihood (CASE 4). The two GPs would model
    the same additive correlated signal over the same time axis — statistically
    degenerate — and allesfitter never builds the combined kernel needed to do
    it coherently.

    The supported pattern is a GP for stellar variability **plus a non-GP
    baseline** (e.g. polynomial, ``hybrid_*`` or ``sample_linear``).
    """
    errors: List[str] = []
    for key, inst_key in _STELLAR_VAR_KEY_TO_INST_KEY.items():
        if settings.get("stellar_var_" + key, "none") not in _GP_VALUES:
            continue
        instruments = (settings.get(inst_key, "") or "").split()
        gp_baseline_insts = [
            inst
            for inst in instruments
            if settings.get(f"baseline_{key}_{inst}", "none") in _GP_VALUES
        ]
        if gp_baseline_insts:
            errors.append(
                f"settings.csv uses a GP for both stellar_var_{key} and the "
                f"baseline of instrument(s) {', '.join(gp_baseline_insts)} "
                f"(baseline_{key}_<inst> set to a sample_GP_* kernel). A GP "
                f"stellar-variability model and a GP baseline cannot be used at "
                f"the same time. Supported pattern: keep the GP for stellar "
                f"variability and switch the baseline to a non-GP model "
                f"(polynomial / hybrid_* / sample_linear)."
            )
    return errors


def check_binning_value(settings: Dict[str, str]) -> List[str]:
    """``binning`` (settings.csv) must be empty / ``none`` or a positive float.

    The value is a bin width in **days** (e.g. ``0.0208333`` for 30 min). A
    non-numeric or non-positive value is an unambiguous mistake and raises.
    The data-dependent "binning ≥ observation baseline" error is enforced in
    ``Basement.load_data`` (which has the light-curve time arrays).
    """
    raw = (settings.get("binning", "") or "").strip()
    if raw == "" or raw.lower() == "none":
        return []
    try:
        val = float(raw)
    except (TypeError, ValueError):
        return [
            f"settings.csv 'binning' = {raw!r} is not a number; leave it empty / "
            f"None to disable, or set a positive bin width in days "
            f"(e.g. 0.0208333 for 30 min)."
        ]
    if val <= 0:
        return [
            f"settings.csv 'binning' = {val} must be > 0 (bin width in days); "
            f"leave it empty / None to disable binning."
        ]
    return []


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------


def collect_config_errors(datadir: str) -> List[str]:
    """Run every structural check against a datadir and return all errors.

    Returns an empty list when ``params.csv`` is absent (nothing to validate —
    the caller's loader will report the missing file) or when every check
    passes.
    """
    rows = read_param_rows(datadir)
    if not rows:
        return []
    settings = read_settings(datadir)

    errors: List[str] = []
    errors += check_duplicate_param_names(rows)
    errors += check_fit_flags(rows)
    errors += check_values_numeric(rows)
    errors += check_bounds_wellformed(rows)
    errors += check_values_within_bounds(rows)
    errors += check_companions_have_params(settings, rows)
    errors += check_gp_baseline_vs_stellar_var(settings)
    errors += check_binning_value(settings)
    return errors


def validate_params_settings(datadir: str, *, raise_on_error: bool = True) -> List[str]:
    """Validate a datadir's ``params.csv`` / ``settings.csv`` consistency.

    Parameters
    ----------
    datadir : str
        allesfitter working directory.
    raise_on_error : bool, default True
        When ``True`` (the runtime default) a non-empty error list raises
        :class:`ConfigError`. Tests pass ``False`` to inspect the messages.

    Returns
    -------
    list[str]
        Every error message found (empty when the config is well-formed).
    """
    errors = collect_config_errors(datadir)
    if errors and raise_on_error:
        raise ConfigError(
            "Invalid allesfitter configuration in '%s':\n  - %s"
            % (datadir, "\n  - ".join(errors))
        )
    return errors
