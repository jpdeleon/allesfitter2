"""Unit tests for ``allesfitter.validation.config_checks``.

Every check is a pure function over already-parsed rows / dicts, so the tests
build tiny in-memory row lists (or tmp_path datadirs) and assert on the
returned error strings. No :class:`Basement` construction required.
"""

from __future__ import annotations

import pytest

from allesfitter.validation import ConfigError, validate_params_settings
from allesfitter.validation.config_checks import (
    check_bounds_wellformed,
    check_companions_have_params,
    check_duplicate_param_names,
    check_fit_flags,
    check_values_numeric,
    check_values_within_bounds,
    collect_config_errors,
    companions_from_settings,
    parse_bounds,
)


# ---------------------------------------------------------------------------
# parse_bounds
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text,expected",
    [
        ("uniform 0 1", ("uniform", (0.0, 1.0))),
        ("normal 0.5 0.1", ("normal", (0.5, 0.1))),
        ("trunc_normal 0 1 0.5 0.2", ("trunc_normal", (0.0, 1.0, 0.5, 0.2))),
        ("", None),
        ("   ", None),
    ],
)
def test_parse_bounds_recognized_forms(text, expected):
    assert parse_bounds(text) == expected


def test_parse_bounds_unknown_keyword_is_flagged():
    assert parse_bounds("weird 1 2") == ("__unknown__", ())


@pytest.mark.parametrize("text", ["uniform 0", "uniform 0 1 2", "normal a b"])
def test_parse_bounds_malformed_is_flagged(text):
    assert parse_bounds(text) == ("__bad__", ())


# ---------------------------------------------------------------------------
# check_duplicate_param_names
# ---------------------------------------------------------------------------


def test_duplicate_names_flagged():
    rows = [
        ["b_rr", "0.1", "1", "uniform 0 1"],
        ["b_rr", "0.2", "1", "uniform 0 1"],
        ["b_period", "3", "1", "uniform 2 4"],
    ]
    errors = check_duplicate_param_names(rows)
    assert len(errors) == 1 and "b_rr" in errors[0]


def test_sentinel_rows_are_not_duplicates():
    rows = [
        ["user-given:", "", "", ""],
        ["automatically set:", "", "", ""],
        ["b_rr", "0.1", "1", "uniform 0 1"],
    ]
    assert check_duplicate_param_names(rows) == []


# ---------------------------------------------------------------------------
# check_fit_flags
# ---------------------------------------------------------------------------


def test_fit_flag_must_be_zero_or_one():
    rows = [
        ["b_rr", "0.1", "1", "uniform 0 1"],
        ["b_period", "3", "yes", "uniform 2 4"],
        ["b_epoch", "0", "2", "uniform -1 1"],
    ]
    errors = check_fit_flags(rows)
    assert len(errors) == 2
    assert any("'yes'" in e for e in errors)
    assert any("'2'" in e for e in errors)


# ---------------------------------------------------------------------------
# check_values_numeric
# ---------------------------------------------------------------------------


def test_non_numeric_value_flagged():
    rows = [["b_rr", "0..1", "1", "uniform 0 1"]]
    errors = check_values_numeric(rows)
    assert len(errors) == 1 and "non-numeric" in errors[0]


def test_empty_value_allowed_on_fixed_row():
    rows = [["host_ldc_q1", "", "0", ""]]
    assert check_values_numeric(rows) == []


def test_empty_value_rejected_on_fitted_row():
    rows = [["b_rr", "", "1", "uniform 0 1"]]
    errors = check_values_numeric(rows)
    assert len(errors) == 1 and "empty initial value" in errors[0]


def test_coupled_row_with_empty_value_is_exempt():
    # 7-column coupled row: value & bounds blank, coupled_with set.
    rows = [
        [
            "baseline_gp_matern32_lnsigma_flux_m4r",
            "",
            "0",
            "",
            "label",
            "",
            "baseline_gp_matern32_lnsigma_flux_m4g",
        ]
    ]
    assert check_values_numeric(rows) == []
    assert check_bounds_wellformed(rows) == []


# ---------------------------------------------------------------------------
# check_bounds_wellformed
# ---------------------------------------------------------------------------


def test_fitted_row_without_bounds_flagged():
    rows = [["b_rr", "0.1", "1", ""]]
    errors = check_bounds_wellformed(rows)
    assert len(errors) == 1 and "no prior bounds" in errors[0]


def test_fixed_row_without_bounds_is_fine():
    rows = [["b_rr", "0.1", "0", ""]]
    assert check_bounds_wellformed(rows) == []


def test_inverted_uniform_bounds_flagged():
    rows = [["b_rr", "0.5", "1", "uniform 1 0"]]
    errors = check_bounds_wellformed(rows)
    assert len(errors) == 1 and "lo < hi" in errors[0]


def test_non_positive_sigma_flagged():
    rows = [
        ["a", "0", "1", "normal 0 0"],
        ["b", "0", "1", "trunc_normal 0 1 0.5 -0.2"],
    ]
    errors = check_bounds_wellformed(rows)
    assert len(errors) == 2
    assert all("sigma > 0" in e for e in errors)


def test_unknown_prior_keyword_flagged():
    rows = [["b_rr", "0.1", "1", "lognormal 0 1"]]
    errors = check_bounds_wellformed(rows)
    assert len(errors) == 1 and "unrecognized prior" in errors[0]


# ---------------------------------------------------------------------------
# check_values_within_bounds
# ---------------------------------------------------------------------------


def test_value_outside_uniform_prior_flagged():
    rows = [["b_rr", "2.0", "1", "uniform 0 1"]]
    errors = check_values_within_bounds(rows)
    assert len(errors) == 1 and "outside" in errors[0]


def test_value_inside_prior_ok():
    rows = [
        ["b_rr", "0.5", "1", "uniform 0 1"],
        ["b_p", "0.5", "1", "trunc_normal 0 1 0.5 0.1"],
    ]
    assert check_values_within_bounds(rows) == []


def test_value_within_bounds_skips_unbounded_normal():
    # normal prior is unbounded → no support check, even for a large value.
    rows = [["b_x", "999", "1", "normal 0 1"]]
    assert check_values_within_bounds(rows) == []


# ---------------------------------------------------------------------------
# check_companions_have_params (cross-file consistency)
# ---------------------------------------------------------------------------


def test_companion_without_params_flagged():
    settings = {"companions_phot": "b c"}
    rows = [["b_rr", "0.1", "1", "uniform 0 1"]]  # only b has rows
    errors = check_companions_have_params(settings, rows)
    assert len(errors) == 1 and "'c'" in errors[0]


def test_companions_all_present_ok():
    settings = {"companions_phot": "b", "companions_rv": "b"}
    rows = [["b_rr", "0.1", "1", "uniform 0 1"], ["b_period", "3", "1", "uniform 2 4"]]
    assert check_companions_have_params(settings, rows) == []


def test_companions_from_settings_unions_keys():
    settings = {"companions_phot": "b c", "companions_rv": "c d"}
    assert companions_from_settings(settings) == ["b", "c", "d"]


# ---------------------------------------------------------------------------
# Aggregator + datadir-level behavior
# ---------------------------------------------------------------------------


def _write_datadir(tmp_path, params: str, settings: str = "companions_phot,b"):
    (tmp_path / "params.csv").write_text(params.strip() + "\n")
    (tmp_path / "settings.csv").write_text(settings.strip() + "\n")
    return str(tmp_path)


def test_collect_config_errors_clean_config(tmp_path):
    d = _write_datadir(
        tmp_path,
        "#name,value,fit,bounds,label,unit,coupled\n"
        "b_rr,0.1,1,uniform 0 0.3,$R_p/R_*$,,\n"
        "b_period,3.0,1,uniform 2 4,$P$,d,\n",
    )
    assert collect_config_errors(d) == []
    # raise_on_error path stays quiet on a clean config
    assert validate_params_settings(d) == []


def test_validate_params_settings_raises_on_bad_config(tmp_path):
    d = _write_datadir(
        tmp_path,
        "#name,value,fit,bounds,label,unit,coupled\n"
        "b_rr,2.0,1,uniform 0 1,k,,\n",  # value outside prior
    )
    with pytest.raises(ConfigError) as exc:
        validate_params_settings(d)
    assert "outside" in str(exc.value)
    # non-raising mode returns the same messages
    msgs = validate_params_settings(d, raise_on_error=False)
    assert any("outside" in m for m in msgs)


def test_missing_params_csv_returns_empty(tmp_path):
    (tmp_path / "settings.csv").write_text("companions_phot,b\n")
    assert collect_config_errors(str(tmp_path)) == []
    assert validate_params_settings(str(tmp_path)) == []
