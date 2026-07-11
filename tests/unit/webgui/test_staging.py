"""Tests for light-curve staging and covariate sniffing."""

from __future__ import annotations

import os

import pytest
import yaml

from allesfitter.webgui import staging


def _write(path, text):
    path.write_text(text)
    return path


def test_sniff_hash_header(tmp_path):
    p = _write(tmp_path / "a.csv", "#time,flux,flux_err,Airmass,FWHM(pix)\n1,1,0.1,1.2,3.4\n")
    assert staging.sniff_header(p) == ["time", "flux", "flux_err", "Airmass", "FWHM(pix)"]
    assert staging.covariate_columns(p) == ["Airmass", "FWHM(pix)"]


def test_sniff_plain_header(tmp_path):
    p = _write(tmp_path / "b.csv", "BJD_TDB,Flux,Err,Airmass\n1,1,0.1,1.2\n")
    assert staging.covariate_columns(p) == ["Airmass"]


def test_sniff_legacy_positional_has_no_named_covariates(tmp_path):
    p = _write(tmp_path / "c.csv", "1,1,0.1,0.5\n2,1,0.1,0.6\n")
    assert staging.sniff_header(p) == []
    assert staging.covariate_columns(p) == []


def test_comment_line_is_not_mistaken_for_header(tmp_path):
    p = _write(tmp_path / "d.csv", "# my notes about this file\n1,1,0.1\n")
    assert staging.sniff_header(p) == []


def test_stage_file_symlink(tmp_path):
    (tmp_path / "raw").mkdir()
    src = _write(tmp_path / "raw" / "lc.csv", "#time,flux,flux_err,Airmass\n1,1,0.1,1.2\n")
    run = tmp_path / "run"
    staged = staging.stage_file(src, "m4g", run, method="symlink")
    dest = run / "m4g.csv"
    assert dest.is_symlink()
    assert os.path.realpath(dest) == os.path.realpath(src)
    assert staged.covariate_columns == ["Airmass"]
    assert staged.method == "symlink"


def test_stage_file_copy(tmp_path):
    src = _write(tmp_path / "lc.csv", "#time,flux,flux_err\n1,1,0.1\n")
    run = tmp_path / "run"
    staged = staging.stage_file(src, "cpt_z", run, method="copy")
    dest = run / "cpt_z.csv"
    assert dest.is_file() and not dest.is_symlink()
    assert staged.method == "copy"


def test_stage_file_missing_source_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        staging.stage_file(tmp_path / "nope.csv", "x", tmp_path / "run")


def test_stage_file_rejects_path_traversal_without_unlinking_destination(tmp_path):
    src = _write(tmp_path / "lc.csv", "#time,flux,flux_err\n1,1,0.1\n")
    escaped = _write(tmp_path / "escaped.csv", "must remain")

    with pytest.raises(ValueError, match="instrument label"):
        staging.stage_file(src, "../escaped", tmp_path / "run")

    assert escaped.read_text() == "must remain"


def test_stage_all_rejects_duplicate_labels(tmp_path):
    a = _write(tmp_path / "a.csv", "#time,flux,flux_err\n1,1,0.1\n")
    b = _write(tmp_path / "b.csv", "#time,flux,flux_err\n1,1,0.1\n")
    with pytest.raises(ValueError, match="duplicate instrument labels"):
        staging.stage_all([("m4g", a), ("m4g", b)], tmp_path / "run")


def test_stage_all_records_provenance_in_meta(tmp_path):
    a = _write(tmp_path / "a.csv", "#time,flux,flux_err,Airmass\n1,1,0.1,1.2\n")
    run = tmp_path / "run"
    run.mkdir()
    (run / "meta.yaml").write_text(yaml.safe_dump({"target": "T"}))
    staging.stage_all([("m4g", a)], run, method="copy")
    meta = yaml.safe_load((run / "meta.yaml").read_text())
    assert meta["target"] == "T"  # existing content preserved
    assert meta["staged"][0]["label"] == "m4g"
    assert meta["staged"][0]["covariate_columns"] == ["Airmass"]
