#!/usr/bin/env python3
"""Unit tests for the non-interactive-safe prompting helpers.

These guard against the historical failure mode where a batch/headless run
would block forever on an ``input()`` sanity check with no human at the
keyboard. The helper must:
  * prompt normally when stdin is a real TTY,
  * return the caller's default (with a warning) when stdin is not a TTY,
  * honour the ALLESFITTER_NONINTERACTIVE override even on a TTY, and
  * fall back to the default on EOF (piped, empty stdin).
"""

import io

import pytest

from allesfitter.utils import interactive
from allesfitter.utils.interactive import ask_choice, stdin_is_interactive


class _FakeStdin:
    """Minimal stdin stand-in with a controllable isatty()."""

    def __init__(self, tty):
        self._tty = tty

    def isatty(self):
        return self._tty


# ---------------------------------------------------------------------------
# stdin_is_interactive
# ---------------------------------------------------------------------------
def test_interactive_when_tty_and_env_unset(monkeypatch):
    monkeypatch.delenv(interactive.NONINTERACTIVE_ENV_VAR, raising=False)
    monkeypatch.setattr(interactive.sys, "stdin", _FakeStdin(tty=True))

    assert stdin_is_interactive() is True


def test_not_interactive_when_no_tty(monkeypatch):
    monkeypatch.delenv(interactive.NONINTERACTIVE_ENV_VAR, raising=False)
    monkeypatch.setattr(interactive.sys, "stdin", _FakeStdin(tty=False))

    assert stdin_is_interactive() is False


def test_env_var_forces_non_interactive_even_on_tty(monkeypatch):
    monkeypatch.setenv(interactive.NONINTERACTIVE_ENV_VAR, "1")
    monkeypatch.setattr(interactive.sys, "stdin", _FakeStdin(tty=True))

    assert stdin_is_interactive() is False


def test_not_interactive_when_stdin_none(monkeypatch):
    monkeypatch.delenv(interactive.NONINTERACTIVE_ENV_VAR, raising=False)
    monkeypatch.setattr(interactive.sys, "stdin", None)

    assert stdin_is_interactive() is False


def test_not_interactive_when_isatty_raises(monkeypatch):
    monkeypatch.delenv(interactive.NONINTERACTIVE_ENV_VAR, raising=False)

    class _Broken:
        def isatty(self):
            raise ValueError("I/O operation on closed file")

    monkeypatch.setattr(interactive.sys, "stdin", _Broken())

    assert stdin_is_interactive() is False


# ---------------------------------------------------------------------------
# ask_choice
# ---------------------------------------------------------------------------
def test_ask_choice_prompts_when_interactive(monkeypatch):
    monkeypatch.delenv(interactive.NONINTERACTIVE_ENV_VAR, raising=False)
    monkeypatch.setattr(interactive.sys, "stdin", _FakeStdin(tty=True))
    monkeypatch.setattr("builtins.input", lambda prompt="": "  2  ")

    #::: real answer is returned, stripped, and the default is ignored
    assert ask_choice("pick: ", default="1") == "2"


def test_ask_choice_returns_default_when_non_interactive(monkeypatch, recwarn):
    monkeypatch.delenv(interactive.NONINTERACTIVE_ENV_VAR, raising=False)
    monkeypatch.setattr(interactive.sys, "stdin", _FakeStdin(tty=False))

    def _boom(prompt=""):
        raise AssertionError("input() must not be called when non-interactive")

    monkeypatch.setattr("builtins.input", _boom)

    result = ask_choice("pick: ", default="1", warning="auto-continuing")

    assert result == "1"
    assert any("auto-continuing" in str(w.message) for w in recwarn.list)


def test_ask_choice_falls_back_on_eof(monkeypatch, recwarn):
    monkeypatch.delenv(interactive.NONINTERACTIVE_ENV_VAR, raising=False)
    monkeypatch.setattr(interactive.sys, "stdin", _FakeStdin(tty=True))

    def _eof(prompt=""):
        raise EOFError

    monkeypatch.setattr("builtins.input", _eof)

    result = ask_choice("pick: ", default="1", warning="auto-continuing")

    assert result == "1"
    assert any("auto-continuing" in str(w.message) for w in recwarn.list)


def test_ask_choice_no_warning_is_silent(monkeypatch, recwarn):
    monkeypatch.delenv(interactive.NONINTERACTIVE_ENV_VAR, raising=False)
    monkeypatch.setattr(interactive.sys, "stdin", _FakeStdin(tty=False))

    result = ask_choice("pick: ", default="2")

    assert result == "2"
    assert len(recwarn.list) == 0


# ---------------------------------------------------------------------------
# integration: the basement 3-sigma check must not hang non-interactively
# ---------------------------------------------------------------------------
def test_real_stdin_pipe_is_non_interactive(monkeypatch):
    """A piped stdin (io.StringIO) reports as non-interactive."""
    monkeypatch.delenv(interactive.NONINTERACTIVE_ENV_VAR, raising=False)
    monkeypatch.setattr(interactive.sys, "stdin", io.StringIO(""))

    assert stdin_is_interactive() is False
    #::: and ask_choice returns the default rather than raising on EOF
    assert ask_choice("pick: ", default="1") == "1"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
