"""Tests for result discovery + logZ scraping."""

from __future__ import annotations

from allesfitter.webgui import results


def _make_results(tmp_path):
    rd = tmp_path / "results"
    rd.mkdir()
    (rd / "mcmc_chains.jpg").write_bytes(b"\xff\xd8jpg")
    (rd / "mcmc_fit_b.pdf").write_bytes(b"%PDF-1.4")
    (rd / "initial_guess_b.pdf").write_bytes(b"%PDF-1.4")
    (rd / "mcmc_table.csv").write_text("name,value\n")
    return tmp_path


def test_images_are_raster_only(tmp_path):
    _make_results(tmp_path)
    assert [p.name for p in results.result_images(tmp_path)] == ["mcmc_chains.jpg"]


def test_documents_are_pdfs(tmp_path):
    _make_results(tmp_path)
    assert {p.name for p in results.result_documents(tmp_path)} == {
        "mcmc_fit_b.pdf",
        "initial_guess_b.pdf",
    }


def test_find_result_file_and_traversal_guard(tmp_path):
    _make_results(tmp_path)
    (tmp_path / "secret.txt").write_text("nope")
    assert results.find_result_file(tmp_path, "mcmc_chains.jpg") is not None
    assert results.find_result_file(tmp_path, "mcmc_fit_b.pdf") is not None
    assert results.find_result_file(tmp_path, "../secret.txt") is None
    assert results.find_result_file(tmp_path, "missing.png") is None


def test_read_logz(tmp_path):
    rd = tmp_path / "results"
    rd.mkdir()
    (rd / "ns.log").write_text("iter 1\nlog(Z) = -100.0 +- 5.0\nlog(Z) = -123.4 +- 0.5\n")
    assert results.read_logz(tmp_path) == "-123.4 +- 0.5"  # last (converged) match


def test_read_logz_absent(tmp_path):
    (tmp_path / "results").mkdir()
    assert results.read_logz(tmp_path) == ""
