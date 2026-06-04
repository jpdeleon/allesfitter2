"""Tests for ``allesfitter.validation.parsing``.

The shared low-level helpers used by both the structural (``config_checks``)
and heuristic (``prior_checks``) layers: CSV row reading and prior-bounds
parsing. These are pure functions over files / strings.
"""

from __future__ import annotations

from allesfitter.validation.parsing import (
    parse_bounds,
    parse_uniform_bounds,
    read_csv_rows,
    read_param_rows,
    read_settings,
)


# ---------------------------------------------------------------------------
# read_csv_rows
# ---------------------------------------------------------------------------


def test_read_csv_rows_returns_empty_for_missing_file(tmp_path):
    assert read_csv_rows(tmp_path / "does_not_exist.csv") == []


def test_read_csv_rows_skips_blanks_and_comments_and_strips(tmp_path):
    p = tmp_path / "data.csv"
    p.write_text(
        "# a comment\n"
        "\n"
        "  a , 1 , uniform 0 1 \n"
        "   \n"
        "#another comment\n"
        "b,2\n"
    )

    rows = read_csv_rows(p)

    assert rows == [["a", "1", "uniform 0 1"], ["b", "2"]]


def test_read_csv_rows_accepts_str_path(tmp_path):
    p = tmp_path / "data.csv"
    p.write_text("a,1\n")
    assert read_csv_rows(str(p)) == [["a", "1"]]


# ---------------------------------------------------------------------------
# read_param_rows / read_settings
# ---------------------------------------------------------------------------


def test_read_param_rows_reads_params_csv(tmp_path):
    (tmp_path / "params.csv").write_text("# header\nb_rr,0.1,1,uniform 0 1\n")
    assert read_param_rows(str(tmp_path)) == [["b_rr", "0.1", "1", "uniform 0 1"]]


def test_read_settings_keys_first_column(tmp_path):
    (tmp_path / "settings.csv").write_text(
        "inst_phot,qlp600 qlp1800\nchromatic,False\nlonely_key\n"
    )

    settings = read_settings(str(tmp_path))

    assert settings["inst_phot"] == "qlp600 qlp1800"
    assert settings["chromatic"] == "False"
    # a key with no value column maps to the empty string, not an error
    assert settings["lonely_key"] == ""


# ---------------------------------------------------------------------------
# parse_bounds
# ---------------------------------------------------------------------------


def test_parse_bounds_uniform():
    assert parse_bounds("uniform 0 1") == ("uniform", (0.0, 1.0))


def test_parse_bounds_normal():
    assert parse_bounds("normal 0.5 0.1") == ("normal", (0.5, 0.1))


def test_parse_bounds_trunc_normal():
    assert parse_bounds("trunc_normal 0 1 0.5 0.1") == (
        "trunc_normal",
        (0.0, 1.0, 0.5, 0.1),
    )


def test_parse_bounds_none_for_empty():
    assert parse_bounds("") is None
    assert parse_bounds("   ") is None
    assert parse_bounds(None) is None


def test_parse_bounds_unknown_keyword():
    assert parse_bounds("loguniform 0 1") == ("__unknown__", ())


def test_parse_bounds_bad_arity_or_numbers():
    assert parse_bounds("uniform 0") == ("__bad__", ())          # wrong arity
    assert parse_bounds("uniform 0 abc") == ("__bad__", ())      # non-numeric


# ---------------------------------------------------------------------------
# parse_uniform_bounds
# ---------------------------------------------------------------------------


def test_parse_uniform_bounds_returns_lo_hi():
    assert parse_uniform_bounds("uniform -2 1") == (-2.0, 1.0)


def test_parse_uniform_bounds_none_for_non_uniform():
    assert parse_uniform_bounds("normal 0 1") is None
    assert parse_uniform_bounds("trunc_normal 0 1 0.5 0.1") is None


def test_parse_uniform_bounds_none_for_empty_or_malformed():
    assert parse_uniform_bounds("") is None
    assert parse_uniform_bounds("uniform 0") is None
    assert parse_uniform_bounds("uniform 0 abc") is None
