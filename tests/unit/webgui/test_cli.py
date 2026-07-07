"""Tests for the CLI port-freeing behavior."""

from __future__ import annotations

import os
import shutil
import socket

import pytest

from allesfitter.webgui import cli


def _free_tcp_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def test_alive_true_for_self_and_false_for_bogus():
    assert cli._alive(os.getpid()) is True
    # PID 2**31-1 is effectively never a live process
    assert cli._alive(2**31 - 1) is False


def test_free_port_noop_when_nothing_listens():
    port = _free_tcp_port()
    assert cli.free_port("127.0.0.1", port) == []  # nothing to kill, no error


def test_detects_listener_excludes_self():
    if not (shutil.which("lsof") or shutil.which("fuser")):
        pytest.skip("needs lsof or fuser to detect port listeners")
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen()
    port = srv.getsockname()[1]
    try:
        pids = cli._pids_on_port(port)
        assert os.getpid() in pids  # this process holds the socket
        # free_port must never target the current process
        assert cli.free_port("127.0.0.1", port) == []
    finally:
        srv.close()
