"""Tests for the `binning_<inst>` plot-title/legend annotation.

When `settings.csv` sets a per-instrument `binning_<inst>` bin width, plots
that show that instrument's data should say so -- otherwise a reader
comparing two instruments' panels has no way to tell one was rebinned. This
covers the pure helper (`_binning_label`) plus its wiring into `plot_1`
(the shared full/phase plotting primitive behind `afplot`).
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

try:
    import allesfitter  # noqa: F401
    from allesfitter import config
    from allesfitter.basement import Basement
    from allesfitter.general_output import _binning_label, plot_1
except Exception:
    pytest.skip("allesfitter not importable", allow_module_level=True)


# ---------------------------------------------------------------------------
# _binning_label: pure function over a settings dict
# ---------------------------------------------------------------------------


def test_binning_label_empty_when_not_set():
    base = SimpleNamespace(settings={})
    assert _binning_label(base, "ts600") == ""


def test_binning_label_empty_when_explicitly_disabled():
    # binning_<inst> present in settings.csv but blank/None disables it
    base = SimpleNamespace(settings={"binning_ts600": None})
    assert _binning_label(base, "ts600") == ""


def test_binning_label_formats_minutes_to_two_decimals():
    # 0.006944 d * 1440 min/d = 9.99936 min -> "10.00 min"
    base = SimpleNamespace(settings={"binning_ts200": 0.006944})
    assert _binning_label(base, "ts200") == " (binned 10.00 min)"


def test_binning_label_ignores_other_instruments():
    base = SimpleNamespace(settings={"binning_ts200": 0.006944})
    assert _binning_label(base, "ts600") == ""


# ---------------------------------------------------------------------------
# wiring: plot_1's title actually carries the annotation
# ---------------------------------------------------------------------------


def _params_csv() -> str:
    text = "#name,value,fit,bounds,label,unit,coupled_with\n"
    text += (
        "b_rr,0.10,1,uniform 0.05 0.30,,,\n"
        "b_rsuma,0.15,1,uniform 0.05 0.30,,,\n"
        "b_cosi,0.10,1,uniform 0 1,,,\n"
        "b_epoch,0.5,1,uniform 0.0 1.0,,,\n"
        "b_period,3.5,1,normal 3.5 0.1,,,\n"
        "b_f_c,0,0,uniform -1 1,,,\n"
        "b_f_s,0,0,uniform -1 1,,,\n"
    )
    for inst in ("binned_inst", "plain_inst"):
        text += (
            f"dil_{inst},0,0,uniform -1 1,,,\n"
            f"host_ldc_q1_{inst},0.5,1,uniform 0 1,,,\n"
            f"host_ldc_q2_{inst},0.5,1,uniform 0 1,,,\n"
            f"ln_err_flux_{inst},-6,1,uniform -10 -1,,,\n"
        )
    return text


def _settings_csv() -> str:
    return (
        "#name,value\n"
        "companions_phot,b\n"
        "companions_rv,\n"
        "inst_phot,binned_inst plain_inst\n"
        "inst_rv,\n"
        "time_format,BJD_TDB\n"
        "multiprocess,False\n"
        "print_progress,False\n"
        "fast_fit,False\n"
        "shift_epoch,False\n"
        "host_ld_law_binned_inst,quad\n"
        "host_ld_law_plain_inst,quad\n"
        "error_flux_binned_inst,sample\n"
        "error_flux_plain_inst,sample\n"
        "baseline_flux_binned_inst,hybrid_offset\n"
        "baseline_flux_plain_inst,hybrid_offset\n"
        "binning_binned_inst,0.01\n"  # 0.01 d = 14.40 min
    )


def _write_lc(path: Path, n=200, seed=0):
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 2.0, n)
    flux = 1.0 + rng.normal(0, 1e-4, n)
    err = np.full(n, 1e-4)
    lines = ["#time,flux,flux_err"]
    for i in range(n):
        lines.append(f"{t[i]:.10f},{flux[i]:.10f},{err[i]:.10f}")
    path.write_text("\n".join(lines) + "\n")


@pytest.fixture
def basement(tmp_path):
    d = tmp_path / "data"
    d.mkdir()
    (d / "results").mkdir()
    (d / "params.csv").write_text(_params_csv())
    (d / "settings.csv").write_text(_settings_csv())
    _write_lc(d / "binned_inst.csv", seed=0)
    _write_lc(d / "plain_inst.csv", seed=1)

    b = Basement(str(d), quiet=True)
    config.BASEMENT = b
    return b


def test_plot_1_title_carries_binning_annotation(basement):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    samples = np.array([basement.theta_0])

    fig, ax = plt.subplots()
    plot_1(ax, samples, "binned_inst", "b", "full")
    assert ax.get_title() == "binned_inst (binned 14.40 min)"
    plt.close(fig)

    fig, ax = plt.subplots()
    plot_1(ax, samples, "plain_inst", "b", "full")
    assert ax.get_title() == "plain_inst"
    plt.close(fig)
