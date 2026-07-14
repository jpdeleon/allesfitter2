"""Regression coverage for unavailable celerite backends."""

from __future__ import annotations

import pytest

from allesfitter import computer


def test_baseline_gp_reports_celerite_import_failure(monkeypatch):
    """A failed optional import must not become an undefined ``terms`` name."""
    cause = ImportError("CXXABI_1.3.15 not found")
    monkeypatch.setattr(computer, "celerite_version", 0)
    monkeypatch.setattr(computer, "_celerite_import_error", cause)

    with pytest.raises(ImportError, match="Cannot construct a GP baseline") as exc_info:
        computer.baseline_get_gp({}, "inst", "flux")

    assert exc_info.value.__cause__ is cause
