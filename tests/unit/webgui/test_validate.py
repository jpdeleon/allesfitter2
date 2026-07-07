"""Tests for the pre-launch dry-run validator."""

from __future__ import annotations

import numpy as np

from allesfitter.webgui import config_writer as cw
from allesfitter.webgui import models as m
from allesfitter.webgui import validate


def _tiny_config():
    return m.FitConfig(
        target="T",
        companions=[m.default_companion("b", period=2.0, epoch=2457000.0)],
        instruments=[m.InstrumentSpec("lco_g", "g", baseline="sample_linear")],
        use_host_density_prior=False,
        fast_fit=False,
    )


def _write_lc(path, seed=0):
    rng = np.random.default_rng(seed)
    t = np.linspace(2456999.0, 2457007.0, 200)
    rows = ["#time,flux,flux_err"]
    for ti, fi in zip(t, 1.0 + rng.normal(0, 1e-3, t.size)):
        rows.append(f"{ti},{fi},0.001")
    path.write_text("\n".join(rows) + "\n")


def test_dry_run_missing_files():
    res = validate.dry_run("/nonexistent/run/dir")
    assert not res.ok
    assert "settings.csv" in res.error


def test_dry_run_ok(tmp_path, basement):
    cfg = _tiny_config()
    cw.write_config(cfg, tmp_path)
    _write_lc(tmp_path / "lco_g.csv")
    res = validate.dry_run(tmp_path)
    assert res.ok, res.error
    assert res.n_instruments == 1
    assert res.n_free_params > 0
    assert bool(res) is True


def test_dry_run_reports_unknown_key(tmp_path, basement):
    cfg = _tiny_config()
    cw.write_config(cfg, tmp_path)
    _write_lc(tmp_path / "lco_g.csv")
    # Corrupt settings.csv with a bogus key the engine must reject.
    settings = (tmp_path / "settings.csv").read_text()
    (tmp_path / "settings.csv").write_text(settings + "totally_bogus_key,1\n")
    res = validate.dry_run(tmp_path)
    assert not res.ok
    assert "totally_bogus_key" in res.error or "not" in res.error.lower()
