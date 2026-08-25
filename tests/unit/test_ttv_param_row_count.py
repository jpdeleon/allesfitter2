"""Tests for the ``{companion}_ttv_transit_N`` row-count check in
``Basement.setup_ttv_fit``.

``params.csv``'s TTV rows are written once (by ``prepare_ttv_fit`` /
``update-params --ttv``) using whatever ``fast_fit_width`` / epoch / period
were active at the time. If ``settings.csv`` changes afterwards (e.g. a wider
``fast_fit_width``) without regenerating those rows, ``setup_ttv_fit``
recomputes a different transit count than what ``params.csv`` provides, and
every model evaluation that reaches the mismatched index used to fail with a
bare ``KeyError`` deep inside ``ellc.lc()``. ``setup_ttv_fit`` now raises a
``ConfigError`` immediately instead.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

try:
    import allesfitter  # noqa: F401
    from allesfitter.basement import Basement
    from allesfitter.validation import ConfigError
except Exception:
    pytest.skip("allesfitter not importable", allow_module_level=True)

PERIOD = 2.0
EPOCH = 0.5
N_TRANSITS = 3  # transits at t=0.5, 2.5, 4.5 within the data span below


def _settings_csv(fast_fit_width=0.2) -> str:
    return (
        "#name,value\n"
        "companions_phot,b\n"
        "companions_rv,\n"
        "inst_phot,lco\n"
        "inst_rv,\n"
        "time_format,BJD_TDB\n"
        "multiprocess,False\n"
        "print_progress,False\n"
        "fast_fit,True\n"
        f"fast_fit_width,{fast_fit_width}\n"
        "fit_ttvs,True\n"
        "shift_epoch,False\n"
        "host_ld_law_lco,quad\n"
        "error_flux_lco,sample\n"
        "baseline_flux_lco,hybrid_offset\n"
    )


def _params_csv(n_ttv_rows: int) -> str:
    text = (
        "#name,value,fit,bounds,label,unit,coupled_with\n"
        "b_rr,0.10,1,uniform 0.05 0.30,,,\n"
        "b_rsuma,0.15,1,uniform 0.05 0.30,,,\n"
        "b_cosi,0.10,1,uniform 0 1,,,\n"
        f"b_epoch,{EPOCH},1,uniform 0.0 1.0,,,\n"
        f"b_period,{PERIOD},1,normal {PERIOD} 0.1,,,\n"
        "b_f_c,0,0,uniform -1 1,,,\n"
        "b_f_s,0,0,uniform -1 1,,,\n"
        "dil_lco,0,0,uniform -1 1,,,\n"
        "host_ldc_q1_lco,0.5,1,uniform 0 1,,,\n"
        "host_ldc_q2_lco,0.5,1,uniform 0 1,,,\n"
        "ln_err_flux_lco,-6,1,uniform -10 -1,,,\n"
    )
    if n_ttv_rows:
        text += "#TTV companion b,,,,,\n"
        for j in range(n_ttv_rows):
            text += (
                f"b_ttv_transit_{j + 1},0,1,uniform -0.05 0.05,TTV$_\\mathrm{{ttv;{j + 1}}}$,d,\n"
            )
    return text


def _write_lc(path: Path, span=6.0, dt=0.005, noise=1e-4, seed=0):
    """Flat light curve (transit shape is irrelevant to this check, which
    fires during Basement construction before any model is evaluated)."""
    rng = np.random.default_rng(seed)
    t = np.arange(0.0, span, dt)
    flux = 1.0 + rng.normal(0, noise, len(t))
    err = np.full(len(t), noise)
    lines = ["#time,flux,flux_err"]
    for i in range(len(t)):
        lines.append(f"{t[i]:.10f},{flux[i]:.10f},{err[i]:.10f}")
    path.write_text("\n".join(lines) + "\n")


def _build_datadir(tmp_path, n_ttv_rows: int, fast_fit_width=0.2) -> Path:
    d = tmp_path / "data"
    d.mkdir()
    (d / "results").mkdir()
    (d / "params.csv").write_text(_params_csv(n_ttv_rows))
    (d / "settings.csv").write_text(_settings_csv(fast_fit_width))
    _write_lc(d / "lco.csv")
    return d


def test_setup_ttv_fit_raises_on_stale_row_count(tmp_path):
    # Arrange: 3 transits are actually observed, but params.csv only carries
    # rows for 2 (as if fast_fit_width was widened after the rows were
    # generated with a narrower one).
    d = _build_datadir(tmp_path, n_ttv_rows=N_TRANSITS - 1)

    # Act / Assert
    with pytest.raises(ConfigError, match="b_ttv_transit_"):
        Basement(str(d))


def test_setup_ttv_fit_accepts_matching_row_count(tmp_path):
    # Arrange: params.csv carries exactly one row per observed transit.
    d = _build_datadir(tmp_path, n_ttv_rows=N_TRANSITS)

    # Act
    basement = Basement(str(d))

    # Assert
    assert len(basement.data["b_tmid_observed_transits"]) == N_TRANSITS


def test_setup_ttv_fit_tolerates_no_ttv_rows_yet(tmp_path):
    # Arrange: fit_ttvs=True but the TTV rows haven't been generated yet
    # (the initial-guess-preview case) -- not a mismatch, just not-yet-done.
    d = _build_datadir(tmp_path, n_ttv_rows=0)

    # Act
    basement = Basement(str(d))

    # Assert
    assert len(basement.data["b_tmid_observed_transits"]) == N_TRANSITS
