"""Tests for the CLI port-freeing behavior."""

from __future__ import annotations

import importlib
import os
import shutil
import socket

import pytest

from allesfitter.webgui import cli

WEBGUI_INSTALLED = all(
    importlib.util.find_spec(module_name) is not None for module_name in cli._WEBGUI_DEPENDENCIES
)


def _free_tcp_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def test_missing_optional_dependency_has_install_instructions(monkeypatch):
    real_import_module = importlib.import_module

    def fail_for_fastapi(name):
        if name == "fastapi":
            raise ModuleNotFoundError("No module named 'fastapi'", name="fastapi")
        return real_import_module(name)

    monkeypatch.setattr(cli.importlib, "import_module", fail_for_fastapi)

    with pytest.raises(SystemExit, match=r"uv sync --extra webgui") as exc_info:
        cli._require_webgui_dependencies()

    assert "allesfitter[webgui]" in str(exc_info.value)
    assert "fastapi" in str(exc_info.value)


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


@pytest.mark.skipif(not WEBGUI_INSTALLED, reason="requires the webgui extra")
def test_cli_passes_root_path_to_application(monkeypatch):
    from allesfitter.webgui import app as web_app

    received = {}
    sentinel = object()

    def fake_create_app(*args, **kwargs):
        received.update(kwargs)
        return sentinel

    monkeypatch.setattr(web_app, "create_app", fake_create_app)
    monkeypatch.setattr("uvicorn.run", lambda app, **kwargs: received.update(app=app, **kwargs))
    cli.main(["--root-path", "/allesfitter/", "--no-kill-existing"])

    assert received["root_path"] == "/allesfitter/"
    assert received["app"] is sentinel


@pytest.mark.skipif(not WEBGUI_INSTALLED, reason="requires the webgui extra")
def test_cli_reload_uses_import_string_and_factory(monkeypatch):
    received = {}
    monkeypatch.setattr("uvicorn.run", lambda app, **kwargs: received.update(app=app, **kwargs))

    cli.main(["--reload", "--runs-root", "runs", "--no-kill-existing"])

    assert received["app"] == "allesfitter.webgui.cli:_reload_app"
    assert received["factory"] is True
    assert received["reload"] is True
