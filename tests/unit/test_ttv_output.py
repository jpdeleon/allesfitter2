import csv
from types import SimpleNamespace

import numpy as np
import pytest

from allesfitter import config
from allesfitter._output_shared import save_ttv_csv
from allesfitter.general_output import get_global_transit_labels

#::: global list as Basement.setup_ttv_fit would build it for TOI-6454 b: all
#::: photometric instruments stitched together, so qlp600 covers entries 3-5.
PERIOD = 22.501694
GLOBAL_TMID = [
    2458508.72600,
    2458531.22770,
    2459228.78021,
    2459251.28191,
    2459273.78360,
    2459971.33611,
]


def test_save_ttv_csv_uses_full_transit_time_posterior(monkeypatch, tmp_path):
    basement = SimpleNamespace(
        settings={"fit_ttvs": True, "companions_phot": ["b"]},
        fitkeys=["b_epoch", "b_period", "b_ttv_transit_1", "b_ttv_transit_2"],
        params={"b_epoch": 100.0, "b_period": 2.0},
        data={"b_tmid_observed_transits": np.array([100.0, 104.0])},
        outdir=str(tmp_path),
    )
    monkeypatch.setattr(config, "BASEMENT", basement, raising=False)
    samples = np.array(
        [
            [99.9, 1.99, -0.01, 0.02],
            [100.0, 2.00, 0.00, 0.03],
            [100.1, 2.01, 0.01, 0.04],
        ]
    )

    outpath = save_ttv_csv(samples)

    with open(outpath, newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert list(rows[0]) == ["planet", "epoch", "tc(BJD)", "tc_unc(BJD)"]
    assert [row["planet"] for row in rows] == ["b", "b"]
    assert [int(row["epoch"]) for row in rows] == [0, 2]

    expected_draws = samples[:, 0, None] + np.array([0, 2]) * samples[:, 1, None] + samples[:, 2:]
    for row, draws in zip(rows, expected_draws.T):
        p16, median, p84 = np.percentile(draws, [16, 50, 84])
        assert float(row["tc(BJD)"]) == pytest.approx(median)
        assert float(row["tc_unc(BJD)"]) == pytest.approx(0.5 * (p84 - p16))


def test_save_ttv_csv_is_disabled_without_ttv_fit(monkeypatch, tmp_path):
    basement = SimpleNamespace(
        settings={"fit_ttvs": False},
        outdir=str(tmp_path),
    )
    monkeypatch.setattr(config, "BASEMENT", basement, raising=False)

    assert save_ttv_csv(np.empty((0, 0))) is False
    assert not (tmp_path / "ttv.csv").exists()


def test_global_transit_labels_offset_for_later_instrument():
    # Arrange: qlp600 only sees the 3rd, 4th and 5th global transits
    base = SimpleNamespace(data={"b_tmid_observed_transits": GLOBAL_TMID})
    tmid_inst = np.array(GLOBAL_TMID[2:5])

    # Act
    labels = get_global_transit_labels(base, "b", tmid_inst, PERIOD)

    # Assert: panels are numbered 3,4,5 — not 1,2,3
    assert labels == [3, 4, 5]


def test_global_transit_labels_tolerate_posterior_drift():
    # Arrange: plot midtimes come from the posterior median, the global list
    # from the initial params, so they differ by minutes
    base = SimpleNamespace(data={"b_tmid_observed_transits": GLOBAL_TMID})
    drift = 12.0 / 60.0 / 24.0  # 12 minutes, in days
    tmid_inst = np.array(GLOBAL_TMID[3:5]) + drift

    # Act
    labels = get_global_transit_labels(base, "b", tmid_inst, PERIOD)

    # Assert
    assert labels == [4, 5]


def test_global_transit_labels_returns_none_when_no_global_match():
    # Arrange: a transit more than half a period from every global entry
    base = SimpleNamespace(data={"b_tmid_observed_transits": GLOBAL_TMID})
    tmid_inst = np.array([GLOBAL_TMID[0] + 5 * PERIOD])

    # Act
    labels = get_global_transit_labels(base, "b", tmid_inst, PERIOD)

    # Assert
    assert labels == [None]


def test_global_transit_labels_falls_back_without_ttv_fit():
    # Arrange: fit_ttvs=False, so setup_ttv_fit never populated the global list
    base = SimpleNamespace(data={})

    # Act
    labels = get_global_transit_labels(base, "b", np.array(GLOBAL_TMID[:2]), PERIOD)

    # Assert: caller falls back to per-instrument numbering
    assert labels == [None, None]
