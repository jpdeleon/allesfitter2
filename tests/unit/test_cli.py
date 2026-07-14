import sys
import types

from typer.testing import CliRunner

from allesfitter.cli import app


def test_gui_forwards_only_arguments_to_argparse(monkeypatch):
    received = []
    fake_cli = types.ModuleType("allesfitter.webgui.cli")
    fake_cli.main = lambda argv: received.extend(argv)
    monkeypatch.setitem(sys.modules, "allesfitter.webgui.cli", fake_cli)

    result = CliRunner().invoke(
        app,
        ["gui", "--runs-root", "runs", "--port", "8123", "--no-network"],
    )

    assert result.exit_code == 0
    assert received == [
        "--runs-root=runs",
        "--host=127.0.0.1",
        "--port=8123",
        "--no-network",
    ]


def test_gui_uses_port_5100_by_default(monkeypatch):
    received = []
    fake_cli = types.ModuleType("allesfitter.webgui.cli")
    fake_cli.main = lambda argv: received.extend(argv)
    monkeypatch.setitem(sys.modules, "allesfitter.webgui.cli", fake_cli)

    result = CliRunner().invoke(app, ["gui"])

    assert result.exit_code == 0
    assert "--port=5100" in received


def test_gui_forwards_reload(monkeypatch):
    received = []
    fake_cli = types.ModuleType("allesfitter.webgui.cli")
    fake_cli.main = lambda argv: received.extend(argv)
    monkeypatch.setitem(sys.modules, "allesfitter.webgui.cli", fake_cli)

    result = CliRunner().invoke(app, ["gui", "--reload"])

    assert result.exit_code == 0
    assert "--reload" in received
