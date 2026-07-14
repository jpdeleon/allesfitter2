"""Tests for the multiple-exposure-time resolver in ``scripts/prepare_allesfit.py``.

When a light-curve search returns more than one exposure time (e.g. TESS
Sector 64 offers both 20 s and 120 s SPOC products), the download path must
not stitch mixed cadences. These tests pin the resolver's two behaviours:

  * default: log an actionable hint naming the real ``-e/--exptime`` flag and a
    single valid value, then ``sys.exit(1)`` (non-zero so schedulers notice);
  * ``ALLESFITTER_NONINTERACTIVE`` set: auto-select the shortest usable cadence,
    narrow the search result to it, and return ``(narrowed, chosen)``.
"""

from __future__ import annotations

import importlib.util
import os

import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Import the script by path (it lives in scripts/, not an installed package)
# and skip if its heavy optional deps are missing, matching the sibling tests.
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
# Test doubles standing in for lightkurve.SearchResult and loguru.Logger
# ---------------------------------------------------------------------------
class _FakeTable:
    def __init__(self, exptimes):
        self._df = pd.DataFrame({"exptime": list(exptimes)})

    def to_pandas(self):
        return self._df


class _FakeSearchResult:
    """Minimal SearchResult: len(), boolean-list indexing, .table.to_pandas()."""

    def __init__(self, exptimes):
        self._exptimes = [float(e) for e in exptimes]
        self.table = _FakeTable(self._exptimes)

    def __len__(self):
        return len(self._exptimes)

    def __getitem__(self, mask):
        kept = [e for e, keep in zip(self._exptimes, list(mask)) if keep]
        return _FakeSearchResult(kept)

    def exptimes(self):
        return list(self._exptimes)


class _RecordingLogger:
    def __init__(self):
        self.errors = []
        self.warnings = []

    def error(self, msg):
        self.errors.append(msg)

    def warning(self, msg):
        self.warnings.append(msg)


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch):
    monkeypatch.delenv(prep.NONINTERACTIVE_ENV_VAR, raising=False)


# ---------------------------------------------------------------------------
# default behaviour: actionable exit, non-zero, correct flag
# ---------------------------------------------------------------------------
def test_default_exits_nonzero_with_actionable_hint():
    logger = _RecordingLogger()
    result = _FakeSearchResult([120.0, 20.0])

    with pytest.raises(SystemExit) as excinfo:
        prep._handle_multiple_exptimes(result, [120.0, 20.0], logger, context="the given sector")

    assert excinfo.value.code == 1
    assert len(logger.errors) == 1
    msg = logger.errors[0]
    #::: names the real flag and a single value; never the bogus -exp=array form
    assert "-e 20" in msg
    assert "--exptime" in msg
    assert "-exp=" not in msg
    assert "the given sector" in msg


def test_default_lists_all_choices_sorted():
    logger = _RecordingLogger()
    result = _FakeSearchResult([120.0, 20.0, 600.0])

    with pytest.raises(SystemExit):
        prep._handle_multiple_exptimes(result, [120.0, 20.0, 600.0], logger)

    msg = logger.errors[0]
    assert "20, 120, 600" in msg  # sorted, human-readable


# ---------------------------------------------------------------------------
# opt-in auto-select behaviour
# ---------------------------------------------------------------------------
def test_autoselect_picks_shortest_and_narrows(monkeypatch):
    monkeypatch.setenv(prep.NONINTERACTIVE_ENV_VAR, "1")
    logger = _RecordingLogger()
    result = _FakeSearchResult([120.0, 20.0])

    narrowed, chosen = prep._handle_multiple_exptimes(result, [120.0, 20.0], logger)

    assert chosen == 20.0
    assert narrowed.exptimes() == [20.0]
    assert len(logger.errors) == 0
    assert len(logger.warnings) == 1
    assert "20" in logger.warnings[0]


def test_autoselect_prefers_cadence_matching_expected_n(monkeypatch):
    monkeypatch.setenv(prep.NONINTERACTIVE_ENV_VAR, "1")
    logger = _RecordingLogger()
    #::: two sectors at 120 s, only one has a 20 s product -> want 120 for 2 sectors
    result = _FakeSearchResult([120.0, 120.0, 20.0])

    narrowed, chosen = prep._handle_multiple_exptimes(result, [120.0, 20.0], logger, expected_n=2)

    assert chosen == 120.0
    assert narrowed.exptimes() == [120.0, 120.0]


def test_autoselect_falls_back_to_shortest_when_no_expected_match(monkeypatch):
    monkeypatch.setenv(prep.NONINTERACTIVE_ENV_VAR, "1")
    logger = _RecordingLogger()
    result = _FakeSearchResult([120.0, 20.0])

    #::: no cadence yields 5 products -> fall back to shortest
    narrowed, chosen = prep._handle_multiple_exptimes(result, [120.0, 20.0], logger, expected_n=5)

    assert chosen == 20.0
    assert narrowed.exptimes() == [20.0]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
