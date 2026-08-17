"""save_per_transit_plots: PDF concatenation and raster fallback for mcmc/ns output."""

import re
from types import SimpleNamespace

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from allesfitter import _output_shared, config, general_output  # noqa: E402


def count_pdf_pages(path):
    """Count `/Type /Page` objects; `\\b` keeps this from matching `/Pages`."""
    with open(path, "rb") as stream:
        return len(re.findall(rb"/Type\s*/Page\b", stream.read()))


@pytest.fixture
def fake_basement(monkeypatch, tmp_path):
    def install(inst_phot):
        basement = SimpleNamespace(
            settings={"inst_phot": list(inst_phot), "companions_phot": ["b"]},
            outdir=str(tmp_path),
        )
        monkeypatch.setattr(config, "BASEMENT", basement, raising=False)
        return basement

    return install


@pytest.fixture
def fake_transits(monkeypatch):
    """Stub general_output.get_observed_transits_for_inst, used by
    iter_per_transit_pages_sorted (the PDF-combining path)."""

    def install(tmid_per_inst):
        def fake(samples, inst, companion, base=None):
            return {}, 0.5, list(tmid_per_inst[inst])

        monkeypatch.setattr(general_output, "get_observed_transits_for_inst", fake)

    return install


@pytest.fixture
def fake_afplot_general(monkeypatch):
    """Stub general_output.afplot_per_transit, one transit per figure — matches
    what iter_per_transit_pages_sorted forces (max_transits=1)."""

    def install():
        calls = []

        def fake(samples, inst, companion, base=None, kwargs_dict=None):
            first_transit = kwargs_dict["first_transit"]
            calls.append((inst, first_transit))
            return plt.figure(), np.atleast_1d(plt.gca()), first_transit + 1, 99

        monkeypatch.setattr(general_output, "afplot_per_transit", fake)
        return calls

    return install


@pytest.fixture
def fake_afplot_shared(monkeypatch):
    """Stub the afplot_per_transit name _output_shared imported directly, used
    by the raster (non-PDF) fallback path."""

    def install(pages_per_inst):
        calls = []

        def fake(samples, inst, companion, kwargs_dict=None):
            first_transit = kwargs_dict["first_transit"]
            calls.append((inst, first_transit))
            page = sum(1 for call in calls if call[0] == inst) - 1
            last_transit, total_transits = pages_per_inst[inst][page]
            return plt.figure(), np.atleast_1d(plt.gca()), last_transit, total_transits

        monkeypatch.setattr(_output_shared, "afplot_per_transit", fake)
        return calls

    return install


# ==========================================================================
#::: .pdf output — one combined multi-page file per companion
# ==========================================================================
def test_pdf_extension_combines_multiple_instruments_into_one_file(
    fake_basement, fake_transits, fake_afplot_general, tmp_path
):
    # Arrange: three transits across two instruments
    fake_basement(["spoc120", "qlp600"])
    fake_transits({"spoc120": [130.0], "qlp600": [100.0, 110.0]})
    fake_afplot_general()

    # Act
    _output_shared.save_per_transit_plots(np.zeros((1, 1)), "mcmc")

    # Assert: a single combined PDF, no per-instrument/per-page files
    outpath = tmp_path / "mcmc_fit_per_transit_b.pdf"
    assert outpath.exists()
    assert count_pdf_pages(str(outpath)) == 3
    assert list(tmp_path.glob("mcmc_fit_per_transit_*_b_*th.pdf")) == []


def test_pdf_extension_combines_single_instrument_many_transits(
    fake_basement, fake_transits, fake_afplot_general, tmp_path
):
    # Arrange: one instrument, enough transits to previously paginate into
    # several files (afplot_per_transit's default max_transits=20)
    fake_basement(["qlp600"])
    fake_transits({"qlp600": [float(i) for i in range(25)]})
    fake_afplot_general()

    # Act
    _output_shared.save_per_transit_plots(np.zeros((1, 1)), "ns")

    # Assert: still just one file, one page per transit
    outpath = tmp_path / "ns_fit_per_transit_b.pdf"
    assert outpath.exists()
    assert count_pdf_pages(str(outpath)) == 25


# ==========================================================================
#::: non-.pdf output — raster formats can't hold multiple pages
# ==========================================================================
def test_non_pdf_extension_falls_back_to_per_instrument_files_and_warns(
    fake_basement, fake_afplot_shared, tmp_path
):
    # Arrange
    fake_basement(["spoc120", "qlp600"])
    fake_afplot_shared({"spoc120": [(3, 3)], "qlp600": [(20, 25), (25, 25)]})

    # Act
    with pytest.warns(UserWarning, match="cannot hold multiple pages"):
        _output_shared.save_per_transit_plots(np.zeros((1, 1)), "mcmc", file_extension=".png")

    # Assert: back to one file per instrument/page
    assert (tmp_path / "mcmc_fit_per_transit_spoc120_b_3th.png").exists()
    assert (tmp_path / "mcmc_fit_per_transit_qlp600_b_20th.png").exists()
    assert (tmp_path / "mcmc_fit_per_transit_qlp600_b_25th.png").exists()


def test_non_pdf_extension_single_instrument_does_not_warn(
    fake_basement, fake_afplot_shared, tmp_path, recwarn
):
    # Arrange
    fake_basement(["qlp600"])
    fake_afplot_shared({"qlp600": [(3, 3)]})

    # Act
    _output_shared.save_per_transit_plots(np.zeros((1, 1)), "ns", file_extension=".png")

    # Assert
    assert (tmp_path / "ns_fit_per_transit_qlp600_b_3th.png").exists()
    assert not any("cannot hold multiple pages" in str(w.message) for w in recwarn.list)
