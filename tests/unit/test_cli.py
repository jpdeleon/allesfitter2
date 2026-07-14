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


def test_show_params_labels_fit_status_column(tmp_path):
    (tmp_path / "params.csv").write_text(
        "#name,value,fit,bounds,label,unit,coupled_with\n"
        "b_rr,0.0653,1,uniform 0 0.2500,$R_b/R_*$,,\n"
        "b_f_c,0,0,uniform -1 1,$f_c$,,\n"
    )

    result = CliRunner().invoke(app, ["show-params", str(tmp_path)])

    assert result.exit_code == 0, result.output
    header = next(
        line for line in result.output.splitlines() if "name" in line and "bounds" in line
    )
    assert "fit?" in header
    assert "✓" in result.output
    assert "✗" in result.output
