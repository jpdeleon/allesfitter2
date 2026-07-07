"""Tests for ``allesfitter.nested_sampling_output.write_priors_latex_table``.

The helper is exercised in standalone mode (explicit ``datadir`` and
``outpath`` arguments) so the tests do not need to spin up a full
``config.BASEMENT`` / allesclass.
"""

from __future__ import annotations

from pathlib import Path

from allesfitter.nested_sampling_output import write_priors_latex_table

_PARAMS_CSV = """#name,value,fit,bounds,label,unit,coupled_with
b_rr_tess,0.1,1,uniform 0 0.3,$R_p/R_\\star$,,
b_epoch,2459000.0,1,normal 2459000.0 0.001,$T_0$,d,
b_cosi,0.05,1,trunc_normal 0 1 0.05 0.05,$\\cos i$,,
ln_err_flux_tess,-6,1,uniform -10 -3,$\\log{\\sigma (tess)}$,rel. flux,
dil_tess,0,0,uniform 0 1,$D_\\mathrm{0; tess}$,,
#dil_tess,0.06,1,normal 0.06 0.01,$D$ (SPOC),,
# generic comment line
"""


def _make_params(tmp_path: Path) -> Path:
    p = tmp_path / "params.csv"
    p.write_text(_PARAMS_CSV)
    return p


def test_writes_outfile_and_returns_path(tmp_path):
    _make_params(tmp_path)
    out = tmp_path / "priors_latex_table.txt"
    result = write_priors_latex_table(datadir=str(tmp_path), outpath=str(out))
    assert result == str(out)
    assert out.exists()


def test_only_fit_rows_are_included(tmp_path):
    _make_params(tmp_path)
    out = tmp_path / "priors.txt"
    write_priors_latex_table(datadir=str(tmp_path), outpath=str(out))
    txt = out.read_text()
    # fit=1 parameters present
    assert "$R_p/R_\\star$" in txt
    assert "$T_0$" in txt
    assert "$\\cos i$" in txt
    assert "$\\log{\\sigma (tess)}$" in txt
    # fit=0 / commented dilution rows absent
    assert "D_\\mathrm{0; tess}" not in txt
    assert "(SPOC)" not in txt


def test_uniform_prior_rendering(tmp_path):
    _make_params(tmp_path)
    out = tmp_path / "priors.txt"
    write_priors_latex_table(datadir=str(tmp_path), outpath=str(out))
    txt = out.read_text()
    assert r"\mathcal{U}(0,\,0.3)" in txt
    assert r"\mathcal{U}(-10,\,-3)" in txt


def test_normal_prior_rendering(tmp_path):
    _make_params(tmp_path)
    out = tmp_path / "priors.txt"
    write_priors_latex_table(datadir=str(tmp_path), outpath=str(out))
    txt = out.read_text()
    assert r"\mathcal{N}(2459000.0,\,0.001)" in txt


def test_trunc_normal_prior_rendering(tmp_path):
    _make_params(tmp_path)
    out = tmp_path / "priors.txt"
    write_priors_latex_table(datadir=str(tmp_path), outpath=str(out))
    txt = out.read_text()
    assert r"\mathcal{N}_T(0.05,\,0.05;\,0,\,1)" in txt


def test_units_column_propagated(tmp_path):
    _make_params(tmp_path)
    out = tmp_path / "priors.txt"
    write_priors_latex_table(datadir=str(tmp_path), outpath=str(out))
    txt = out.read_text()
    assert "rel. flux" in txt  # ln_err_flux_tess unit
    assert " d \\\\" in txt  # b_epoch unit "d"


def test_unknown_bounds_passed_through(tmp_path):
    p = tmp_path / "params.csv"
    p.write_text("foo,0,1,jeffreys 1e-3 1,$f$,,\n")
    out = tmp_path / "priors.txt"
    write_priors_latex_table(datadir=str(tmp_path), outpath=str(out))
    assert "jeffreys 1e-3 1" in out.read_text()


def test_missing_params_csv_returns_false(tmp_path):
    out = tmp_path / "priors.txt"
    assert write_priors_latex_table(datadir=str(tmp_path), outpath=str(out)) is False
    assert not out.exists()


def test_no_fit_rows_returns_false(tmp_path):
    p = tmp_path / "params.csv"
    p.write_text("#name,value,fit,bounds,label,unit,coupled_with\nfoo,1,0,uniform 0 1,$f$,,\n")
    out = tmp_path / "priors.txt"
    assert write_priors_latex_table(datadir=str(tmp_path), outpath=str(out)) is False


def test_table_structure_is_complete(tmp_path):
    _make_params(tmp_path)
    out = tmp_path / "priors.txt"
    write_priors_latex_table(datadir=str(tmp_path), outpath=str(out))
    txt = out.read_text()
    for needle in (
        r"\begin{table}",
        r"\centering",
        r"\caption{Priors for fitted parameters.}",
        r"\begin{tabular}{lll}",
        r"Parameter & Prior & Unit \\",
        r"\end{tabular}",
        r"\end{table}",
    ):
        assert needle in txt, f"missing: {needle}"
