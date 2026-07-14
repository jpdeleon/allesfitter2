#!/usr/bin/env python3
"""
Non-interactive-safe prompting helpers.

allesfitter historically calls the built-in :func:`input` in several
"luser proof" sanity checks (e.g. an initial guess lying more than 3 sigma
from its prior, or an output file that already exists). When a fit is launched
non-interactively -- a batch job, a scheduler, a subprocess, a notebook kernel
with no attached TTY -- those ``input()`` calls block forever because there is
no human at the keyboard to answer them.

:func:`ask_choice` centralises the decision. When stdin is a real terminal it
prompts exactly as before. When it is not (no TTY, EOF reached, or the
``ALLESFITTER_NONINTERACTIVE`` environment variable is set) it emits a warning
and returns a caller-supplied default answer so the run proceeds instead of
hanging.
"""

import os
import sys
import warnings

#::: set (to any non-empty value) to force the non-interactive default at every
#::: prompt, even when a TTY is attached. Useful for CI and headless batch runs.
NONINTERACTIVE_ENV_VAR = "ALLESFITTER_NONINTERACTIVE"


def stdin_is_interactive():
    """Return ``True`` only when it is safe to block on :func:`input`.

    Returns ``False`` if the ``ALLESFITTER_NONINTERACTIVE`` environment variable
    is set, if stdin is missing/closed, or if stdin is not a TTY (pipe, file,
    captured stream, or detached job).
    """
    if os.environ.get(NONINTERACTIVE_ENV_VAR):
        return False
    try:
        return bool(sys.stdin) and sys.stdin.isatty()
    except (AttributeError, ValueError):
        #::: stdin replaced by an object without isatty(), or already closed
        return False


def ask_choice(prompt, default, warning=None):
    """Prompt for a line of input, or auto-answer when non-interactive.

    Parameters
    ----------
    prompt : str
        Text passed to :func:`input` when a TTY is attached.
    default : str
        Answer returned (without prompting) when stdin is non-interactive or
        EOF is reached. The caller is responsible for choosing a safe default.
    warning : str or None
        If given, emitted via :func:`warnings.warn` whenever ``default`` is
        returned in place of a real answer, so the auto-decision is visible in
        logs.

    Returns
    -------
    str
        The user's stripped input, or ``default`` in the non-interactive case.
        Always a string, so call sites can compare against ``"1"``/``"2"``/...
    """
    if not stdin_is_interactive():
        if warning:
            warnings.warn(warning, stacklevel=2)
        return default
    try:
        return input(prompt).strip()
    except EOFError:
        #::: stdin was interactive-looking but exhausted (e.g. piped and empty)
        if warning:
            warnings.warn(warning, stacklevel=2)
        return default
