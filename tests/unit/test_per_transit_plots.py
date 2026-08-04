"""Per-transit plot pagination, chronological ordering, and PDF concatenation."""

import re
from types import SimpleNamespace

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from allesfitter import config, general_output  # noqa: E402


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
def fake_afplot(monkeypatch):
    """Stub afplot_per_transit and record the (inst, first_transit, max_transits) calls."""

    def install(pages_per_inst=None, failing_insts=(), failing_pages=()):
        calls = []

        def fake(samples, inst, companion, base=None, kwargs_dict=None):
            first_transit = kwargs_dict["first_transit"]
            calls.append((inst, first_transit, kwargs_dict.get("max_transits")))
            if inst in failing_insts or (inst, first_transit) in failing_pages:
                raise RuntimeError("boom")
            if pages_per_inst is None:
                #::: one-transit-per-figure mode: paging values are unused
                last_transit, total_transits = first_transit + 1, 99
            else:
                page = sum(1 for call in calls if call[0] == inst) - 1
                last_transit, total_transits = pages_per_inst[inst][page]
            return plt.figure(), np.atleast_1d(plt.gca()), last_transit, total_transits

        monkeypatch.setattr(general_output, "afplot_per_transit", fake)
        return calls

    return install


@pytest.fixture
def fake_transits(monkeypatch):
    """Stub get_observed_transits_for_inst with a midtime list per instrument."""

    def install(tmid_per_inst, failing_insts=()):
        def fake(samples, inst, companion, base=None):
            if inst in failing_insts:
                raise RuntimeError("boom")
            return {}, 0.5, list(tmid_per_inst[inst])

        monkeypatch.setattr(general_output, "get_observed_transits_for_inst", fake)

    return install


def drain(pages):
    """Collect (inst, third-element) pairs, closing every figure."""
    out = []
    for inst, fig, extra in pages:
        out.append((inst, extra))
        plt.close(fig)
    return out


# ==========================================================================
#::: iter_per_transit_pages — one figure per instrument page, inst_phot order
# ==========================================================================
def test_iter_per_transit_pages_walks_every_instrument(fake_afplot, fake_basement):
    # Arrange: qlp600 fits on one page, qlp200 needs two
    fake_basement(["qlp600", "qlp200"])
    fake_afplot({"qlp600": [(3, 3)], "qlp200": [(20, 25), (25, 25)]})

    # Act
    pages = drain(general_output.iter_per_transit_pages(np.zeros((1, 1)), "b"))

    # Assert
    assert pages == [("qlp600", 3), ("qlp200", 20), ("qlp200", 25)]


def test_iter_per_transit_pages_paginates_from_previous_last_transit(fake_afplot, fake_basement):
    # Arrange
    fake_basement(["qlp200"])
    calls = fake_afplot({"qlp200": [(20, 25), (25, 25)]})

    # Act
    drain(general_output.iter_per_transit_pages(np.zeros((1, 1)), "b"))

    # Assert: page 2 resumes where page 1 stopped
    assert [(inst, first) for inst, first, _max in calls] == [("qlp200", 0), ("qlp200", 20)]


def test_iter_per_transit_pages_warns_and_continues_past_failure(fake_afplot, fake_basement):
    # Arrange: the first instrument blows up mid-plot
    fake_basement(["qlp600", "qlp200"])
    fake_afplot({"qlp200": [(3, 3)]}, failing_insts=["qlp600"])

    # Act
    with pytest.warns(UserWarning, match="afplot_per_transit failed for inst='qlp600'"):
        pages = drain(general_output.iter_per_transit_pages(np.zeros((1, 1)), "b"))

    # Assert: the healthy instrument is still plotted
    assert [inst for inst, _extra in pages] == ["qlp200"]


# ==========================================================================
#::: iter_per_transit_pages_sorted — one figure per transit, chronological
# ==========================================================================
def test_sorted_pages_interleave_instruments_by_midtime(fake_afplot, fake_transits, fake_basement):
    # Arrange: inst_phot order is NOT chronological
    fake_basement(["spoc120", "qlp600"])
    fake_transits({"spoc120": [130.0], "qlp600": [100.0, 110.0]})
    calls = fake_afplot()

    # Act
    pages = drain(general_output.iter_per_transit_pages_sorted(np.zeros((1, 1)), "b"))

    # Assert: ordered by midtime, not by inst_phot
    assert pages == [("qlp600", 100.0), ("qlp600", 110.0), ("spoc120", 130.0)]
    assert [(inst, first) for inst, first, _max in calls] == [
        ("qlp600", 0),
        ("qlp600", 1),
        ("spoc120", 0),
    ]


def test_sorted_pages_put_duplicate_coverage_back_to_back(
    fake_afplot, fake_transits, fake_basement
):
    # Arrange: both instruments cover the same transit at t=100
    fake_basement(["spoc120", "qlp200"])
    fake_transits({"spoc120": [100.0], "qlp200": [90.0, 100.0]})
    fake_afplot()

    # Act
    pages = drain(general_output.iter_per_transit_pages_sorted(np.zeros((1, 1)), "b"))

    # Assert: the shared transit yields consecutive figures, inst_phot order breaking the tie
    assert pages == [("qlp200", 90.0), ("spoc120", 100.0), ("qlp200", 100.0)]


def test_sorted_pages_force_one_transit_per_figure_without_touching_caller_kwargs(
    fake_afplot, fake_transits, fake_basement
):
    # Arrange
    fake_basement(["qlp600"])
    fake_transits({"qlp600": [100.0, 110.0]})
    calls = fake_afplot()
    kwargs_dict = {"max_transits": 20, "color": "b"}

    # Act
    drain(general_output.iter_per_transit_pages_sorted(np.zeros((1, 1)), "b", kwargs_dict))

    # Assert: interleaving needs one transit per figure, but the caller's dict is untouched
    assert [max_transits for _inst, _first, max_transits in calls] == [1, 1]
    assert kwargs_dict == {"max_transits": 20, "color": "b"}


def test_sorted_pages_warn_when_an_instrument_layout_fails(
    fake_afplot, fake_transits, fake_basement
):
    # Arrange: the transit list of one instrument cannot be determined
    fake_basement(["spoc120", "qlp600"])
    fake_transits({"qlp600": [100.0]}, failing_insts=["spoc120"])
    fake_afplot()

    # Act
    with pytest.warns(UserWarning, match="could not determine the transits of inst='spoc120'"):
        pages = drain(general_output.iter_per_transit_pages_sorted(np.zeros((1, 1)), "b"))

    # Assert: the healthy instrument still gets plotted
    assert pages == [("qlp600", 100.0)]


def test_sorted_pages_warn_and_continue_when_one_figure_fails(
    fake_afplot, fake_transits, fake_basement
):
    # Arrange: rendering the second transit of qlp600 blows up
    fake_basement(["qlp600"])
    fake_transits({"qlp600": [100.0, 110.0, 120.0]})
    fake_afplot(failing_pages=[("qlp600", 1)])

    # Act
    with pytest.warns(UserWarning, match="afplot_per_transit failed"):
        pages = drain(general_output.iter_per_transit_pages_sorted(np.zeros((1, 1)), "b"))

    # Assert: one bad transit does not cost us the rest
    assert pages == [("qlp600", 100.0), ("qlp600", 120.0)]


# ==========================================================================
#::: _save_per_transit_pdf — one multi-page PDF per companion
# ==========================================================================
def test_save_per_transit_pdf_concatenates_all_instruments(
    fake_afplot, fake_transits, fake_basement, tmp_path
):
    # Arrange: three transits across two instruments
    fake_basement(["spoc120", "qlp600"])
    fake_transits({"spoc120": [130.0], "qlp600": [100.0, 110.0]})
    fake_afplot()

    # Act
    outpath = general_output._save_per_transit_pdf(np.zeros((1, 1)), "b", {}, ".pdf")

    # Assert: one file per planet, one page per transit
    assert outpath == str(tmp_path / "initial_guess_per_transit_b.pdf")
    assert count_pdf_pages(outpath) == 3


def test_save_per_transit_pdf_writes_no_file_without_pages(
    fake_afplot, fake_transits, fake_basement, tmp_path
):
    # Arrange: no photometric instruments at all
    fake_basement([])
    fake_transits({})
    fake_afplot()

    # Act
    outpath = general_output._save_per_transit_pdf(np.zeros((1, 1)), "b", {}, ".pdf")

    # Assert: a zero-page PDF would be unreadable, so none is created
    assert outpath is None
    assert not (tmp_path / "initial_guess_per_transit_b.pdf").exists()
