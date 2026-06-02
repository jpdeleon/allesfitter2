"""Tests for the explicit ``chromatic,True/False`` override in settings.csv.

Default behaviour (no `chromatic` key in settings.csv) is unchanged: the
flag is auto-detected from the number of unique bandpass labels. When
the user supplies the key explicitly, it overrides auto-detection in
both directions.

Covers:
  1. `chromatic,False` + multi-label `bandpass` → forces achromatic mode,
     `get_bandpass(inst)` returns None.
  2. `chromatic,False` → fitkeys carry `b_rr` (achromatic), NOT
     `b_rr_<bp>` (chromatic).
  3. `chromatic,False` + per-band `b_rr_g/r/i/z` rows in params.csv →
     Basement(...) raises with an actionable mismatch message.
  4. `chromatic,True` with single-band `bandpass` → flag honoured.
  5. No `chromatic` key in settings.csv → legacy bandpass-uniqueness
     auto-detect runs unchanged.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

try:
    import allesfitter  # noqa: F401
    from allesfitter import config
    from allesfitter.basement import Basement
except Exception:
    pytest.skip("allesfitter not importable", allow_module_level=True)


# ---------------------------------------------------------------------------
# helpers (fixture pattern follows tests/test_share_baseline.py)
# ---------------------------------------------------------------------------

_INSTS = ("muscat_g", "muscat_r", "muscat_i", "muscat_z")
_BANDS = ("g", "r", "i", "z")


def _lc_csv(seed: int, n: int = 200) -> str:
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 1.0, n)
    f = 1.0 + rng.normal(0.0, 1e-3, n)
    e = np.full(n, 1e-3)
    lines = ["#time,flux,flux_err"]
    lines += [f"{ti:.10f},{fi:.10f},{ei:.10f}" for ti, fi, ei in zip(t, f, e)]
    return "\n".join(lines) + "\n"


def _achromatic_params(insts) -> str:
    """params.csv with a single shared b_rr (no per-band rows)."""
    head = "#name,value,fit,bounds,label,unit,coupled_with\n"
    body = (
        "b_rr,0.10,1,uniform 0.05 0.30,$R_b/R_*$,,\n"
        "b_rsuma,0.15,1,uniform 0.05 0.30,$R/a$,,\n"
        "b_cosi,0.10,1,uniform 0 1,$\\cos i$,,\n"
        "b_epoch,0.5,1,uniform 0.0 1.0,$T_0$,d,\n"
        "b_period,3.5,1,normal 3.5 0.1,$P$,d,\n"
        "b_f_c,0,0,uniform -1 1,,,\n"
        "b_f_s,0,0,uniform -1 1,,,\n"
    )
    for i in insts:
        body += (
            f"dil_{i},0,0,uniform -1 1,,,\n"
            f"host_ldc_q1_{i},0.5,1,uniform 0 1,,,\n"
            f"host_ldc_q2_{i},0.5,1,uniform 0 1,,,\n"
            f"ln_err_flux_{i},-6,1,uniform -10 -1,,,\n"
        )
    return head + body


def _chromatic_params(insts, bandpasses) -> str:
    """params.csv with per-band b_rr_<bp> and host_ldc_q1/q2_<bp>."""
    head = "#name,value,fit,bounds,label,unit,coupled_with\n"
    body = (
        "b_rsuma,0.15,1,uniform 0.05 0.30,$R/a$,,\n"
        "b_cosi,0.10,1,uniform 0 1,$\\cos i$,,\n"
        "b_epoch,0.5,1,uniform 0.0 1.0,$T_0$,d,\n"
        "b_period,3.5,1,normal 3.5 0.1,$P$,d,\n"
        "b_f_c,0,0,uniform -1 1,,,\n"
        "b_f_s,0,0,uniform -1 1,,,\n"
    )
    for bp in bandpasses:
        body += f"b_rr_{bp},0.10,1,uniform 0.05 0.30,$R_b/R_*$,,\n"
        body += f"host_ldc_q1_{bp},0.5,1,uniform 0 1,,,\n"
        body += f"host_ldc_q2_{bp},0.5,1,uniform 0 1,,,\n"
    for i in insts:
        body += f"dil_{i},0,0,uniform -1 1,,,\n"
        body += f"ln_err_flux_{i},-6,1,uniform -10 -1,,,\n"
    return head + body


def _settings(insts, bandpasses, chromatic_value=None) -> str:
    inst_str = " ".join(insts)
    bp_str = " ".join(bandpasses)
    lines = [
        "#name,value",
        "companions_phot,b",
        "companions_rv,",
        f"inst_phot,{inst_str}",
        f"bandpass,{bp_str}",
        "inst_rv,",
        "time_format,BJD_TDB",
        "multiprocess,False",
        "print_progress,False",
        "fast_fit,False",
        "shift_epoch,False",
    ]
    for i in insts:
        lines += [
            f"host_ld_law_{i},quad",
            f"error_flux_{i},sample",
        ]
    if chromatic_value is not None:
        lines.append(f"chromatic,{chromatic_value}")
    return "\n".join(lines) + "\n"


def _make_datadir(tmp_path: Path, insts=_INSTS, bandpasses=_BANDS,
                  params_kind='achromatic', chromatic_value=None) -> Path:
    d = tmp_path / "data"
    d.mkdir()
    (d / "results").mkdir()
    if params_kind == 'achromatic':
        (d / "params.csv").write_text(_achromatic_params(insts))
    elif params_kind == 'chromatic':
        (d / "params.csv").write_text(_chromatic_params(insts, bandpasses))
    else:
        raise ValueError(params_kind)
    (d / "settings.csv").write_text(_settings(insts, bandpasses, chromatic_value))
    for k, inst in enumerate(insts):
        (d / f"{inst}.csv").write_text(_lc_csv(seed=k))
    return d


# ---------------------------------------------------------------------------
# 1) explicit chromatic=False overrides auto-detect
# ---------------------------------------------------------------------------


def test_explicit_chromatic_false_overrides_auto_detect(tmp_path):
    """Multi-label bandpass + chromatic,False → achromatic, get_bandpass
    returns None for every instrument."""
    d = _make_datadir(tmp_path, params_kind='achromatic',
                      chromatic_value='False')
    b = Basement(str(d), quiet=True)
    assert b.settings['chromatic'] is False
    for inst in _INSTS:
        assert b.get_bandpass(inst) is None, inst


# ---------------------------------------------------------------------------
# 2) achromatic key forced in fit vector
# ---------------------------------------------------------------------------


def test_chromatic_false_keeps_ldc_keys_per_bandpass(tmp_path):
    """LDC keys must remain per-bandpass under chromatic,False — limb
    darkening depends on wavelength, not on the rr-naming convention."""
    d = _make_datadir(tmp_path, params_kind='achromatic',
                      chromatic_value='False')
    # _achromatic_params() writes per-inst LDC rows; for this test we
    # need per-bandpass rows. Overwrite params.csv with a mixed layout.
    body = (
        "#name,value,fit,bounds,label,unit,coupled_with\n"
        "b_rr,0.10,1,uniform 0.05 0.30,$R_b/R_*$,,\n"
        "b_rsuma,0.15,1,uniform 0.05 0.30,$R/a$,,\n"
        "b_cosi,0.10,1,uniform 0 1,$\\cos i$,,\n"
        "b_epoch,0.5,1,uniform 0.0 1.0,$T_0$,d,\n"
        "b_period,3.5,1,normal 3.5 0.1,$P$,d,\n"
        "b_f_c,0,0,uniform -1 1,,,\n"
        "b_f_s,0,0,uniform -1 1,,,\n"
    )
    for bp in _BANDS:
        body += f"host_ldc_q1_{bp},0.5,1,uniform 0 1,,,\n"
        body += f"host_ldc_q2_{bp},0.5,1,uniform 0 1,,,\n"
    for inst in _INSTS:
        body += f"dil_{inst},0,0,uniform -1 1,,,\n"
        body += f"ln_err_flux_{inst},-6,1,uniform -10 -1,,,\n"
    (d / "params.csv").write_text(body)

    b = Basement(str(d), quiet=True)
    keys = list(map(str, b.fitkeys))
    # rr collapsed to achromatic (the chromatic,False override applies to rr)
    assert 'b_rr' in keys
    # LDC keys are still per-bandpass, not per-inst
    for bp in _BANDS:
        assert 'host_ldc_q1_'+bp in keys, (bp, [k for k in keys if 'ldc' in k])
        assert 'host_ldc_q2_'+bp in keys
    for inst in _INSTS:
        assert 'host_ldc_q1_'+inst not in keys, inst
    # get_ldc_bandpass bypasses the chromatic flag
    assert b.get_ldc_bandpass('muscat_g') == 'g'
    # get_bandpass still respects the override (rr-naming behaviour)
    assert b.get_bandpass('muscat_g') is None


def test_chromatic_false_forces_achromatic_rr_key(tmp_path):
    d = _make_datadir(tmp_path, params_kind='achromatic',
                      chromatic_value='False')
    b = Basement(str(d), quiet=True)
    keys = list(map(str, b.fitkeys))
    assert 'b_rr' in keys, keys
    for bp in _BANDS:
        assert 'b_rr_'+bp not in keys, (bp, keys)


# ---------------------------------------------------------------------------
# 3) per-band rows in params.csv are rejected when chromatic=False
# ---------------------------------------------------------------------------


def test_chromatic_false_forbids_per_band_rr_rows(tmp_path):
    d = _make_datadir(tmp_path, params_kind='chromatic',
                      chromatic_value='False')
    with pytest.raises(ValueError, match="chromatic,False"):
        Basement(str(d), quiet=True)


# ---------------------------------------------------------------------------
# 4) explicit chromatic=True honoured even with single-band bandpass
# ---------------------------------------------------------------------------


def test_chromatic_true_when_user_explicit(tmp_path):
    """Single bandpass label (would auto-detect as achromatic) + the user
    sets chromatic,True → flag forced True."""
    insts = ("muscat_g",)
    bandpasses = ("g",)
    d = tmp_path / "data"
    d.mkdir()
    (d / "results").mkdir()
    (d / "params.csv").write_text(_chromatic_params(insts, bandpasses))
    (d / "settings.csv").write_text(_settings(insts, bandpasses, chromatic_value='True'))
    (d / "muscat_g.csv").write_text(_lc_csv(seed=0))
    b = Basement(str(d), quiet=True)
    assert b.settings['chromatic'] is True
    # chromatic naming kicks in for the single inst
    assert b.get_bandpass('muscat_g') == 'g'


# ---------------------------------------------------------------------------
# 5) legacy auto-detect when chromatic is not in settings.csv
# ---------------------------------------------------------------------------


def test_chromatic_value_robust_to_lowercase_and_int(tmp_path):
    """set_bool() under the hood accepts only `'true'` / `'1'`
    (case-insensitive) as True; everything else is False. Confirm the
    override is robust to common spellings.

    For each raw value we pair the matching params layout so the
    OTHER validator (chromatic⇄params consistency) doesn't fire:
    expected=False → achromatic params, expected=True → chromatic params.
    """
    for raw_value, expected in [
        ('false', False),
        ('FALSE', False),
        ('0', False),
        ('no', False),       # set_bool defaults to False for unknown
        ('true', True),
        ('TRUE', True),
        ('1', True),
    ]:
        sub = tmp_path / raw_value
        sub.mkdir()
        kind = 'chromatic' if expected else 'achromatic'
        d = _make_datadir(sub, params_kind=kind,
                          chromatic_value=raw_value)
        b = Basement(str(d), quiet=True)
        assert b.settings['chromatic'] is expected, (raw_value, expected,
                                                      b.settings['chromatic'])


def test_no_chromatic_setting_preserves_legacy_autodetect(tmp_path):
    """Omit the chromatic key entirely → bandpass-uniqueness auto-detect
    runs as before (4-band MuSCAT → chromatic=True)."""
    d = _make_datadir(tmp_path, params_kind='chromatic',
                      chromatic_value=None)
    b = Basement(str(d), quiet=True)
    assert b.settings['chromatic'] is True
    # And the achromatic-override branch must NOT have fired:
    assert b.get_bandpass('muscat_g') == 'g'
