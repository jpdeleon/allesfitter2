"""Output tests for ``scripts/prepare_allesfit.py``.

``prepare_allesfit.main()`` is a long, network-driven entry point (it queries
the TOI/CTOI/NExSci catalogs, downloads light curves via ``lightkurve``, and
calls ``tess-point``), so it cannot be exercised end-to-end offline. What *can*
be tested deterministically is the set of pure / file-writing helpers that
actually shape the files the script emits — the prior bounds written into
``params.csv``, the data-driven GP/noise priors, the SPOC dilution rows, and
the per-mission segment-label formatting that lands in filenames and headers.

Each test asserts on the concrete output the script produces (return values,
rewritten ``params.csv`` rows, the contamination report file), not on
implementation details.
"""

from __future__ import annotations

import importlib.util
import math
import os
from types import SimpleNamespace

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Import the script by path. It pulls heavy optional deps (lightkurve,
# tess_stars2px, astropy, loguru); skip the whole module if any are missing
# so the suite stays green on a minimal install.
# ---------------------------------------------------------------------------
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_SCRIPT_PATH = os.path.join(_BASE_DIR, "scripts", "prepare_allesfit.py")


def _load_script():
    spec = importlib.util.spec_from_file_location("prepare_allesfit", _SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


try:
    prep = _load_script()
except Exception as exc:  # pragma: no cover - depends on optional deps
    pytest.skip(
        f"scripts/prepare_allesfit.py not importable in this env: {exc}",
        allow_module_level=True,
    )


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _params_csv_text() -> str:
    """A minimal params.csv carrying the rows the helpers rewrite."""
    return (
        "#name,value,fit,bounds,label,unit,coupled_with\n"
        "b_rr,0.10,1,uniform 0 1,$R_b/R_*$,,\n"
        "dil_tess,0,0,uniform 0 1,$D_{0;tess}$,,\n"
        "ln_err_flux_tess,-6,1,uniform -10 -1,$\\ln\\sigma$,,\n"
        "baseline_gp_matern32_lnsigma_flux_tess,-5,1,uniform -10 -3,$\\ln\\sigma_{GP}$,,\n"
        "baseline_gp_matern32_lnrho_flux_tess,0,1,uniform -1 5,$\\ln\\rho_{GP}$,,\n"
    )


def _synthetic_lightcurve(n=500, cadence_days=2.0 / 1440.0, rms=1e-3, seed=0):
    """Evenly sampled flat light curve with known cadence + noise level."""
    rng = np.random.default_rng(seed)
    t = np.arange(n) * cadence_days
    f = 1.0 + rng.normal(0.0, rms, n)
    return t, f


# ---------------------------------------------------------------------------
# 1) segment-label / mission-word formatting (drives filenames + headers)
# ---------------------------------------------------------------------------


def test_parse_segment_label_returns_last_token_as_str():
    assert prep._parse_segment_label("TESS Sector 82") == "82"
    assert prep._parse_segment_label("K2 Campaign 11a") == "11a"
    assert prep._parse_segment_label("Kepler Quarter 3") == "3"


def test_segments_match_is_string_tolerant():
    assert prep._segments_match(82, "82") is True
    assert prep._segments_match("11a", "11a") is True
    assert prep._segments_match(5, 6) is False


def test_segments_match_is_zero_pad_tolerant():
    # lightkurve's search table carries zero-padded strings ('01') while a
    # downloaded LightCurve.sector is an int (1); numeric ids must compare by
    # value so a user's "-s 1" survives both the availability and header gates.
    assert prep._segments_match(1, "01") is True
    assert prep._segments_match("01", "1") is True
    assert prep._segments_match("07", 7) is True
    # K2 campaign alpha suffixes stay distinguished (not purely numeric).
    assert prep._segments_match("11a", "11b") is False
    assert prep._segments_match("11a", "11") is False


def test_natural_segment_sort_orders_numeric_then_suffix():
    labels = ["11b", "2", "11a", "1", "10"]
    ordered = sorted(labels, key=prep._natural_segment_sort_key)
    assert ordered == ["1", "2", "10", "11a", "11b"]


def test_segment_word_per_mission_singular_and_plural():
    assert prep._segment_word("tess") == "sector"
    assert prep._segment_word("TESS", plural=True) == "sectors"
    assert prep._segment_word("k2") == "campaign"
    assert prep._segment_word("kepler", plural=True) == "quarters"
    # Unknown mission falls back to the TESS vocabulary.
    assert prep._segment_word(" groundbased") == "sector"


# ---------------------------------------------------------------------------
# 2) default physics-informed prior bounds (substituted into params.csv)
# ---------------------------------------------------------------------------


def test_default_prior_bounds_typical_planet():
    out = prep._default_prior_bounds(rprs_max=0.12, rsuma_min=0.02, rsuma_max=0.15)
    # rr_upper = ceil(0.12*10)/10 + 0.05 = 0.2 + 0.05 = 0.25
    assert out["rr_upper"] == pytest.approx(0.25)
    assert out["rsuma_lo"] == pytest.approx(0.02)
    assert out["rsuma_hi"] == pytest.approx(0.30)  # 2*0.15
    # cosi_max = 1.2 * 0.15 * (1 + 0.12) = 0.2016
    assert out["cosi_max"] == pytest.approx(1.2 * 0.15 * 1.12)


def test_default_prior_bounds_are_clamped():
    # Brown-dwarf-deep, grazing, ultra-short-period extreme inputs must clamp.
    out = prep._default_prior_bounds(rprs_max=0.9, rsuma_min=1e-9, rsuma_max=0.9)
    assert out["rr_upper"] == 0.5  # capped at brown-dwarf regime
    assert out["rsuma_lo"] == pytest.approx(1e-3)  # floored
    assert out["rsuma_hi"] == 0.5  # capped
    assert out["cosi_max"] == 1.0  # hard-bounded at 1


# ---------------------------------------------------------------------------
# 3) dataset-aware GP / noise bounds
# ---------------------------------------------------------------------------


def test_dataset_aware_gp_bounds_returns_none_for_short_series():
    assert prep._dataset_aware_gp_bounds([0, 1, 2], [1, 1, 1], tdur_days=0.1) is None


def test_dataset_aware_gp_bounds_invariants():
    cadence = 2.0 / 1440.0  # 2 min
    tdur = 0.1
    t, f = _synthetic_lightcurve(n=600, cadence_days=cadence, rms=1e-3)
    b = prep._dataset_aware_gp_bounds(t, f, tdur_days=tdur)
    assert b is not None

    # Every bound must be ordered lo < init < hi.
    for lo, init, hi in [
        (b["lnerr_lo"], b["lnerr_init"], b["lnerr_hi"]),
        (b["lnsigma_lo"], b["lnsigma_init"], b["lnsigma_hi"]),
        (b["lnrho_lo"], b["lnrho_init"], b["lnrho_hi"]),
    ]:
        assert lo < hi
        assert lo <= init <= hi

    # ln_err upper never permits >10% relative-flux jitter.
    assert b["lnerr_hi"] <= math.log(0.10) + 1e-9
    # GP correlation length floor keeps it away from transit ingress/egress:
    # rho_lo must exceed both 2 cadences and half the transit duration.
    assert math.exp(b["lnrho_lo"]) > 2.0 * cadence
    assert math.exp(b["lnrho_lo"]) > 0.5 * tdur
    # ...and the upper bound stays within the observation baseline.
    assert math.exp(b["lnrho_hi"]) <= (t[-1] - t[0])


# ---------------------------------------------------------------------------
# 4) _update_params_gp_bounds rewrites the right rows in params.csv
# ---------------------------------------------------------------------------


def test_update_params_gp_bounds_rewrites_matching_rows(tmp_path):
    p = tmp_path / "params.csv"
    p.write_text(_params_csv_text())
    t, f = _synthetic_lightcurve()
    bounds = prep._dataset_aware_gp_bounds(t, f, tdur_days=0.1)

    n = prep._update_params_gp_bounds(str(p), "tess", bounds)
    assert n == 3  # ln_err + lnsigma + lnrho

    lines = {
        line.split(",")[0]: line
        for line in p.read_text().splitlines()
        if line and not line.startswith("#")
    }
    # The three targeted rows now carry the data-driven uniform bounds...
    for name in (
        "ln_err_flux_tess",
        "baseline_gp_matern32_lnsigma_flux_tess",
        "baseline_gp_matern32_lnrho_flux_tess",
    ):
        assert "uniform" in lines[name].split(",")[3]
    # ...and the label/unit/coupled tail is preserved verbatim.
    assert lines["ln_err_flux_tess"].split(",")[4] == "$\\ln\\sigma$"
    # Untouched rows (b_rr) are unchanged.
    assert lines["b_rr"].split(",")[3] == "uniform 0 1"


def test_update_params_gp_bounds_noop_for_unknown_inst(tmp_path):
    p = tmp_path / "params.csv"
    original = _params_csv_text()
    p.write_text(original)
    t, f = _synthetic_lightcurve()
    bounds = prep._dataset_aware_gp_bounds(t, f, tdur_days=0.1)

    assert prep._update_params_gp_bounds(str(p), "muscat_g", bounds) == 0
    assert p.read_text() == original  # file untouched


def test_update_params_gp_bounds_handles_missing_file(tmp_path):
    bounds = {"lnerr_init": 0.0}
    assert prep._update_params_gp_bounds(str(tmp_path / "nope.csv"), "tess", bounds) == 0


# ---------------------------------------------------------------------------
# 5) _inject_dilution_normal_prior prepends a commented normal-prior twin
# ---------------------------------------------------------------------------


def test_inject_dilution_normal_prior_inserts_commented_row(tmp_path):
    p = tmp_path / "params.csv"
    p.write_text(_params_csv_text())

    ok = prep._inject_dilution_normal_prior(str(p), "tess", median=0.2, std=0.0)
    assert ok is True

    out_lines = p.read_text().splitlines()
    new_rows = [l for l in out_lines if l.startswith("#dil_tess,")]
    assert len(new_rows) == 1
    row = new_rows[0]
    assert "normal 0.200000" in row
    # std=0 -> width floored at sigma_floor (default 0.01).
    assert "0.010000" in row
    # The commented twin sits directly above the still-active uniform row.
    idx = out_lines.index(row)
    assert out_lines[idx + 1].startswith("dil_tess,") and "uniform" in out_lines[idx + 1]


def test_inject_dilution_normal_prior_returns_false_without_match(tmp_path):
    p = tmp_path / "params.csv"
    p.write_text(_params_csv_text())
    # No dil row for this instrument -> nothing inserted, file unchanged.
    before = p.read_text()
    assert prep._inject_dilution_normal_prior(str(p), "muscat_g", 0.2, 0.05) is False
    assert p.read_text() == before


# ---------------------------------------------------------------------------
# 6) _write_spoc_contamination: CROWDSAP -> dilution report file
# ---------------------------------------------------------------------------


def test_write_spoc_contamination_single_segment(tmp_path):
    out = tmp_path / "tess_contamination.txt"
    lc = SimpleNamespace(meta={"CROWDSAP": 0.8, "FLFRCSAP": 0.95}, sector=82)

    summary = prep._write_spoc_contamination(lc, str(out), "tess")
    assert summary is not False
    # dilution == 1 - CROWDSAP
    assert summary["median_dilution"] == pytest.approx(0.2)
    assert summary["median_crowdsap"] == pytest.approx(0.8)
    assert summary["n_segments"] == 1

    text = out.read_text()
    assert out.exists() and len(text) > 0
    assert "median_dilution\t0.200000" in text
    # Uses the mission-appropriate vocabulary in the table header.
    assert "sector\tCROWDSAP" in text


def test_write_spoc_contamination_returns_false_without_crowdsap(tmp_path):
    out = tmp_path / "none.txt"
    lc = SimpleNamespace(meta={}, sector=1)
    assert prep._write_spoc_contamination(lc, str(out), "tess") is False
    assert not out.exists()


def test_write_spoc_contamination_median_across_segments(tmp_path):
    lk = pytest.importorskip("lightkurve")
    # Build a real LightCurveCollection so the multi-segment median path runs.
    lcs = []
    for sec, crowd in [(1, 0.9), (2, 0.7)]:
        lc = lk.LightCurve(time=np.linspace(0, 1, 5), flux=np.ones(5))
        lc.meta["CROWDSAP"] = crowd
        lc.meta["FLFRCSAP"] = 0.9
        lc.meta["SECTOR"] = sec
        lcs.append(lc)
    coll = lk.LightCurveCollection(lcs)

    out = tmp_path / "multi.txt"
    summary = prep._write_spoc_contamination(coll, str(out), "tess")
    assert summary["n_segments"] == 2
    # dilutions = [0.1, 0.3] -> median 0.2
    assert summary["median_dilution"] == pytest.approx(0.2)
    assert summary["median_crowdsap"] == pytest.approx(0.8)


# ---------------------------------------------------------------------------
# _percentile_3sig_safe — empty/all-NaN rsuma samples must not crash
# ---------------------------------------------------------------------------


class _CapLogger:
    def __init__(self):
        self.warnings = []

    def warning(self, msg, *args):
        self.warnings.append(msg % args if args else msg)


def test_percentile_3sig_safe_empty_returns_fallback_and_warns():
    log = _CapLogger()
    fb = (1e-3, 0.1, 0.25)
    out = prep._percentile_3sig_safe(
        np.array([]), fallback=fb, label="(R*+Rp)/a", planet="b", logger=log
    )
    assert out == fb
    assert len(log.warnings) == 1 and "could not derive" in log.warnings[0]


def test_percentile_3sig_safe_all_nan_returns_fallback():
    log = _CapLogger()
    fb = (1e-3, 0.1, 0.25)
    out = prep._percentile_3sig_safe(
        np.full(100, np.nan), fallback=fb, label="x", planet="c", logger=log
    )
    assert out == fb
    assert log.warnings  # warned


def test_percentile_3sig_safe_valid_samples_compute_percentiles():
    log = _CapLogger()
    samples = np.linspace(0.05, 0.15, 1000)
    lo, mid, hi = prep._percentile_3sig_safe(
        samples, fallback=(1e-3, 0.1, 0.25), label="x", planet="b", logger=log
    )
    assert lo < mid < hi
    assert mid == pytest.approx(0.1, abs=1e-3)
    assert log.warnings == []  # no fallback used
