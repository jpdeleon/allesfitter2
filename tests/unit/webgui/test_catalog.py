"""Tests for ephemeris auto-fill (local TOI CSV + graceful degradation) and
the sector/campaign/quarter availability lookup used by the Prepare page."""

from __future__ import annotations

import pytest

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


# ---------------------------------------------------------------------------
# available_sectors: segment-label helpers
# ---------------------------------------------------------------------------


def test_segment_label_takes_last_token():
    assert catalog._segment_label("TESS Sector 42") == "42"
    assert catalog._segment_label("K2 Campaign 11a") == "11a"
    assert catalog._segment_label("Kepler Quarter 5") == "5"


def test_segment_sort_key_orders_numeric_then_alpha_suffix():
    labels = ["9", "27", "11b", "11a", "2"]
    assert sorted(labels, key=catalog._segment_sort_key) == ["2", "9", "11a", "11b", "27"]


@pytest.mark.parametrize(
    ("id_type", "target", "expected"),
    [
        ("tic", "123456", "TIC 123456"),
        ("toi", "6715", "TOI 6715"),
        ("ctoi", "6715.01", "CTOI 6715.01"),
        ("name", "HD 39091", "HD 39091"),
        ("", "HD 39091", "HD 39091"),
    ],
)
def test_sector_query_name_maps_id_type(id_type, target, expected):
    assert catalog._sector_query_name(target, id_type) == expected


# ---------------------------------------------------------------------------
# available_sectors: input guards (no network call needed)
# ---------------------------------------------------------------------------


def test_available_sectors_requires_target():
    result = catalog.available_sectors("")
    assert not result.ok
    assert "target" in result.error


def test_available_sectors_respects_allow_network_false():
    result = catalog.available_sectors("TOI-6715", allow_network=False)
    assert not result.ok
    assert "network" in result.error


# ---------------------------------------------------------------------------
# available_sectors: mocked lightkurve search (no real network access)
# ---------------------------------------------------------------------------


class _FakeSearchResult:
    """Minimal stand-in for ``lightkurve.SearchResult`` used by these tests."""

    def __init__(self, rows):
        self._rows = list(rows)  # list of (mission_str, author) tuples

    def __len__(self):
        return len(self._rows)

    def __getitem__(self, idx):
        return _FakeSearchResult([r for r, keep in zip(self._rows, idx) if keep])

    @property
    def mission(self):
        return [r[0] for r in self._rows]

    @property
    def author(self):
        return [r[1] for r in self._rows]


def test_available_sectors_returns_sorted_segments_and_pipelines(monkeypatch):
    import lightkurve

    rows = [
        ("TESS Sector 27", "SPOC"),
        ("TESS Sector 2", "SPOC"),
        ("TESS Sector 2", "QLP"),
        ("TESS Sector 9", "QLP"),
    ]
    monkeypatch.setattr(lightkurve, "search_lightcurve", lambda *a, **kw: _FakeSearchResult(rows))

    result = catalog.available_sectors("TOI-6715", id_type="toi", mission="tess")
    assert result.ok
    assert result.segments == ["2", "9", "27"]
    assert result.pipelines == ["qlp", "spoc"]


class _FakeSearchResultExp(_FakeSearchResult):
    """SearchResult stand-in that also carries a per-row exposure time.

    Rows are ``(mission_str, author, exptime_seconds)`` tuples; exposes the
    ``.exptime`` attribute that ``catalog._exptime_values`` falls back to.
    """

    def __getitem__(self, idx):
        #:: keep the subclass (and its exptime) after boolean filtering
        return _FakeSearchResultExp([r for r, keep in zip(self._rows, idx) if keep])

    @property
    def exptime(self):
        return [r[2] for r in self._rows]


def test_available_sectors_reports_unique_sorted_exptimes(monkeypatch):
    import lightkurve

    rows = [
        ("TESS Sector 64", "SPOC", 120),
        ("TESS Sector 64", "SPOC", 20),
        ("TESS Sector 11", "SPOC", 120),
    ]
    monkeypatch.setattr(
        lightkurve, "search_lightcurve", lambda *a, **kw: _FakeSearchResultExp(rows)
    )

    result = catalog.available_sectors("TOI-6715", id_type="toi", mission="tess")
    assert result.ok
    assert result.exptimes == [20, 120]  # deduped + ascending


def test_available_sectors_exptimes_follow_pipeline_filter(monkeypatch):
    import lightkurve

    rows = [
        ("TESS Sector 64", "SPOC", 20),
        ("TESS Sector 64", "QLP", 1800),
        ("TESS Sector 11", "QLP", 1800),
    ]
    monkeypatch.setattr(
        lightkurve, "search_lightcurve", lambda *a, **kw: _FakeSearchResultExp(rows)
    )

    result = catalog.available_sectors("TOI-6715", id_type="toi", pipeline="qlp")
    assert result.ok
    assert result.exptimes == [1800]  # only the QLP rows survive the filter


def test_available_sectors_without_exptime_column_is_graceful(monkeypatch):
    import lightkurve

    #:: the plain fake exposes no exptime -> best-effort yields an empty list
    rows = [("TESS Sector 2", "SPOC")]
    monkeypatch.setattr(lightkurve, "search_lightcurve", lambda *a, **kw: _FakeSearchResult(rows))

    result = catalog.available_sectors("TOI-6715")
    assert result.ok
    assert result.exptimes == []


def test_available_sectors_filters_by_pipeline(monkeypatch):
    import lightkurve

    rows = [
        ("TESS Sector 27", "SPOC"),
        ("TESS Sector 2", "QLP"),
        ("TESS Sector 9", "QLP"),
    ]
    monkeypatch.setattr(lightkurve, "search_lightcurve", lambda *a, **kw: _FakeSearchResult(rows))

    result = catalog.available_sectors("TOI-6715", id_type="toi", pipeline="qlp")
    assert result.ok
    assert result.segments == ["2", "9"]
    assert result.pipelines == ["qlp", "spoc"]  # unfiltered list stays informational


def test_available_sectors_pipeline_with_no_matches_reports_available(monkeypatch):
    import lightkurve

    rows = [("TESS Sector 27", "SPOC")]
    monkeypatch.setattr(lightkurve, "search_lightcurve", lambda *a, **kw: _FakeSearchResult(rows))

    result = catalog.available_sectors("TOI-6715", pipeline="qlp")
    assert not result.ok
    assert "qlp" in result.error
    assert "spoc" in result.error


def test_available_sectors_empty_result_reports_not_found(monkeypatch):
    import lightkurve

    monkeypatch.setattr(lightkurve, "search_lightcurve", lambda *a, **kw: _FakeSearchResult([]))

    result = catalog.available_sectors("TOI-6715")
    assert not result.ok
    assert "no light curves" in result.error


def test_available_sectors_search_exception_is_graceful(monkeypatch):
    import lightkurve

    def _raise(*a, **kw):
        raise RuntimeError("MAST is down")

    monkeypatch.setattr(lightkurve, "search_lightcurve", _raise)

    result = catalog.available_sectors("TOI-6715")
    assert not result.ok
    assert "MAST is down" in result.error
