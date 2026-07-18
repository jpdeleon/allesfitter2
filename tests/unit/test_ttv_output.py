import csv
from types import SimpleNamespace

import numpy as np
import pytest

from allesfitter import config
from allesfitter._output_shared import save_ttv_csv


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
