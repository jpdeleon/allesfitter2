"""Per-transit plot pagination and per-companion PDF concatenation."""

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
def fake_afplot(monkeypatch):
    """Stub afplot_per_transit with a scripted (last_transit, total_transits) per page."""

    def install(pages_per_inst, failing_insts=()):
        calls = []

        def fake(samples, inst, companion, base=None, kwargs_dict=None):
            calls.append((inst, kwargs_dict["first_transit"]))
            if inst in failing_insts:
                raise RuntimeError("boom")
            page = sum(1 for call in calls if call[0] == inst) - 1
            last_transit, total_transits = pages_per_inst[inst][page]
            return plt.figure(), np.atleast_1d(plt.gca()), last_transit, total_transits

        monkeypatch.setattr(general_output, "afplot_per_transit", fake)
        return calls

    return install


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


def test_iter_per_transit_pages_walks_every_instrument(fake_afplot, fake_basement):
    # Arrange: qlp600 fits on one page, qlp200 needs two
    fake_basement(["qlp600", "qlp200"])
    fake_afplot({"qlp600": [(3, 3)], "qlp200": [(20, 25), (25, 25)]})

    # Act
    pages = list(general_output.iter_per_transit_pages(np.zeros((1, 1)), "b"))

    # Assert
    assert [(inst, last) for inst, _fig, last in pages] == [
        ("qlp600", 3),
        ("qlp200", 20),
        ("qlp200", 25),
    ]
    for _inst, fig, _last in pages:
        plt.close(fig)


def test_iter_per_transit_pages_paginates_from_previous_last_transit(fake_afplot, fake_basement):
    # Arrange
    fake_basement(["qlp200"])
    calls = fake_afplot({"qlp200": [(20, 25), (25, 25)]})

    # Act
    for _inst, fig, _last in general_output.iter_per_transit_pages(np.zeros((1, 1)), "b"):
        plt.close(fig)

    # Assert: page 2 resumes where page 1 stopped
    assert calls == [("qlp200", 0), ("qlp200", 20)]


def test_iter_per_transit_pages_warns_and_continues_past_failure(fake_afplot, fake_basement):
    # Arrange: the first instrument blows up mid-plot
    fake_basement(["qlp600", "qlp200"])
    fake_afplot({"qlp200": [(3, 3)]}, failing_insts=["qlp600"])

    # Act
    with pytest.warns(UserWarning, match="afplot_per_transit failed for inst='qlp600'"):
        pages = list(general_output.iter_per_transit_pages(np.zeros((1, 1)), "b"))

    # Assert: the healthy instrument is still plotted
    assert [inst for inst, _fig, _last in pages] == ["qlp200"]
    for _inst, fig, _last in pages:
        plt.close(fig)


def test_save_per_transit_pdf_concatenates_all_instruments(fake_afplot, fake_basement, tmp_path):
    # Arrange: three pages across two instruments
    fake_basement(["qlp600", "qlp200"])
    fake_afplot({"qlp600": [(3, 3)], "qlp200": [(20, 25), (25, 25)]})

    # Act
    outpath = general_output._save_per_transit_pdf(np.zeros((1, 1)), "b", {}, ".pdf")

    # Assert: one file per planet, one page per instrument page
    assert outpath == str(tmp_path / "initial_guess_per_transit_b.pdf")
    assert count_pdf_pages(outpath) == 3


def test_save_per_transit_pdf_writes_no_file_without_pages(fake_afplot, fake_basement, tmp_path):
    # Arrange: no photometric instruments at all
    fake_basement([])
    fake_afplot({})

    # Act
    outpath = general_output._save_per_transit_pdf(np.zeros((1, 1)), "b", {}, ".pdf")

    # Assert: a zero-page PDF would be unreadable, so none is created
    assert outpath is None
    assert not (tmp_path / "initial_guess_per_transit_b.pdf").exists()
