"""Tests for ephemeris auto-fill (local TOI CSV + graceful degradation)."""

from __future__ import annotations

from allesfitter.webgui import catalog


def _toi_csv(tmp_path):
    p = tmp_path / "TOIs.csv"
    p.write_text(
        "TOI,Period (days),Epoch (BJD),Duration (hours),Depth (ppm)\n"
        "6715.01,2.862577,2460621.83,2.4,3100\n"
        "1234.01,5.5,2459000.0,3.6,900\n"
    )
    return p


def test_from_toi_csv_matches_and_converts_duration(tmp_path):
    eph = catalog.from_toi_csv("TOI-6715", _toi_csv(tmp_path))
    assert eph.source == "toi_csv"
    assert eph.period == 2.862577
    assert eph.epoch == 2460621.83
    assert abs(eph.duration - 2.4 / 24.0) < 1e-9  # hours -> days
    assert eph.depth == 3100
    assert eph.found


def test_toi_number_parsing_variants(tmp_path):
    csv = _toi_csv(tmp_path)
    for name in ("TOI-6715", "toi 6715", "TOI6715", "6715.01"):
        assert catalog.from_toi_csv(name, csv).period == 2.862577


def test_unknown_target_returns_empty(tmp_path):
    eph = catalog.from_toi_csv("TOI-9999", _toi_csv(tmp_path))
    assert not eph.found
    assert eph.source == "none"


def test_missing_csv_is_graceful():
    eph = catalog.from_toi_csv("TOI-6715", "/no/such/file.csv")
    assert not eph.found


def test_lookup_prefers_local_csv_without_network(tmp_path):
    eph = catalog.lookup("TOI-6715", toi_csv=_toi_csv(tmp_path), allow_network=False)
    assert eph.source == "toi_csv"
    assert eph.period == 2.862577


def test_lookup_no_sources_returns_none_source(tmp_path):
    eph = catalog.lookup("TOI-6715", toi_csv=None, allow_network=False)
    assert eph.source == "none"
    assert not eph.found
