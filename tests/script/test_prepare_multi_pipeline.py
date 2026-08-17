"""Tests for the multi-pipeline / multi-exptime ``-p``/``-e`` support in
``scripts/prepare_allesfit.py``.

``-p/--pipeline`` and ``-e/--exptime`` now accept multiple values (like
``-f/--filename`` already did) so a single ``prepare`` invocation can
download several instruments, each from its own pipeline/exptime, e.g.::

    prepare_allesfit.py -toi 5423 -s all -f spoc120 qlp600 \
        -p spoc qlp -e 120 600

These tests only exercise the length-validation guard, which runs (and
``sys.exit(1)``s on mismatch) before any network access — so it's safe to
call ``main()`` directly offline. The happy-path download loop itself still
requires network access and isn't covered here, matching the rest of this
test module's scope (see ``test_prepare_allesfit_output.py`` docstring).
"""

from __future__ import annotations

import importlib.util
import os

import pytest

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


@pytest.fixture
def _errors(monkeypatch):
    captured = []
    monkeypatch.setattr(prep.logger, "error", lambda msg: captured.append(msg))
    return captured


def _run(monkeypatch, argv):
    monkeypatch.setattr("sys.argv", ["prepare_allesfit.py", *argv])
    with pytest.raises(SystemExit):
        prep.main()


# ---------------------------------------------------------------------------
# main-path validation: -p count vs -f count, -e count vs -p count
# ---------------------------------------------------------------------------


def test_multi_pipeline_requires_matching_filename_count(monkeypatch, _errors, tmp_path):
    _run(
        monkeypatch,
        [
            "-toi",
            "5423",
            "-s",
            "all",
            "-dir",
            str(tmp_path),
            "-f",
            "spoc120",
            "qlp600",
            "extra",
            "-p",
            "spoc",
            "qlp",
        ],
    )
    assert any("--pipeline has 2 entries but --filename has 3 entries" in msg for msg in _errors)


def test_multi_exptime_requires_matching_pipeline_count(monkeypatch, _errors, tmp_path):
    _run(
        monkeypatch,
        [
            "-toi",
            "5423",
            "-s",
            "all",
            "-dir",
            str(tmp_path),
            "-f",
            "spoc120",
            "qlp600",
            "-p",
            "spoc",
            "qlp",
            "-e",
            "120",
            "600",
            "1800",
        ],
    )
    assert any("--exptime has 3 entries but --pipeline has 2 entries" in msg for msg in _errors)


def test_single_pipeline_with_multiple_filenames_is_unchanged(monkeypatch, tmp_path):
    """A single -p (the default) must not trip the new length check, even
    with several --filename entries — that's the pre-existing 'download
    fns[0], supply the rest manually' workflow."""
    called = {}

    def _fake_search_lightcurve(*_a, **_k):
        called["reached_network"] = True
        raise RuntimeError("network reached (expected to happen after validation passes)")

    monkeypatch.setattr(prep.lk, "search_lightcurve", _fake_search_lightcurve)
    monkeypatch.setattr(
        prep,
        "parse_target_name",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("stop before catalog lookup")),
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "prepare_allesfit.py",
            "-toi",
            "5423",
            "-s",
            "all",
            "-dir",
            str(tmp_path),
            "-f",
            "tess",
            "kepler",
        ],
    )
    with pytest.raises(RuntimeError, match="stop before catalog lookup"):
        prep.main()
    # Reaching parse_target_name (patched to raise) proves the pipeline/exptime
    # validation didn't reject a single -p with multiple -f.


def test_matching_pipeline_filename_exptime_counts_pass_validation(monkeypatch, tmp_path):
    """-p spoc qlp -e 120 600 -f spoc120 qlp600 (equal counts) must clear the
    new length checks and reach catalog resolution, same as any valid run."""
    monkeypatch.setattr(
        prep,
        "parse_target_name",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("stop before catalog lookup")),
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "prepare_allesfit.py",
            "-toi",
            "5423",
            "-s",
            "all",
            "-dir",
            str(tmp_path),
            "-f",
            "spoc120",
            "qlp600",
            "-p",
            "spoc",
            "qlp",
            "-e",
            "120",
            "600",
        ],
    )
    with pytest.raises(RuntimeError, match="stop before catalog lookup"):
        prep.main()
