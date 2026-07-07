"""Test fixtures for the webgui suite.

The webgui config-generation modules (``models``, ``instruments``,
``config_writer``, ``staging``) are deliberately engine-free — they need only
the standard library, numpy, and pyyaml. But importing them normally still runs
``allesfitter/__init__.py``, which eagerly imports the compiled engine
(``ellc``/``celerite``). On platforms where those wheels are not importable
(e.g. CI's manylinux runner with the upstream ``ellc`` mac-only wheel), that
would make even the pure-logic tests un-collectable.

To keep the pure config-writer logic covered *everywhere*, we register
lightweight namespace stubs for ``allesfitter`` / ``allesfitter.webgui`` when —
and only when — the real engine import fails. When the engine imports fine we do
nothing and the real package is used, so the ``Basement`` round-trip tests run
for real. Those round-trip tests take the :func:`basement` fixture, which skips
them when the engine is unavailable.
"""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

import pytest

ENGINE_AVAILABLE = False
try:
    importlib.import_module("allesfitter")
    from allesfitter.basement import Basement as _Basement  # noqa: F401

    ENGINE_AVAILABLE = True
except Exception:
    _ALLESFITTER_ROOT = Path(__file__).resolve().parents[3] / "allesfitter"
    if "allesfitter" not in sys.modules:
        _pkg = types.ModuleType("allesfitter")
        _pkg.__path__ = [str(_ALLESFITTER_ROOT)]
        sys.modules["allesfitter"] = _pkg
    if "allesfitter.webgui" not in sys.modules:
        _web = types.ModuleType("allesfitter.webgui")
        _web.__path__ = [str(_ALLESFITTER_ROOT / "webgui")]
        sys.modules["allesfitter.webgui"] = _web


@pytest.fixture
def basement():
    """Return the real :class:`allesfitter.basement.Basement`, or skip.

    Skips (rather than errors) when the compiled engine is not importable, so
    round-trip tests that need the real validator degrade gracefully.
    """
    if not ENGINE_AVAILABLE:
        pytest.skip("allesfitter engine (ellc/celerite) not importable in this environment")
    from allesfitter.basement import Basement

    return Basement
