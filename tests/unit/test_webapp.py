from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from allesfitter.webapp import (
    Workbench,
    WorkbenchDB,
    build_job_command,
    build_prepare_command,
    resolve_workspace,
)


@pytest.fixture()
def target(tmp_path: Path) -> dict:
    return {
        "id": 1,
        "name": "TOI-1097",
        "identifier_type": "toi",
        "identifier_value": "1097",
        "data_dir": str(tmp_path / "TOI-1097"),
    }


def test_prepare_command_covers_prepare_script_options(target: dict, tmp_path: Path):
    command = build_prepare_command(
        target,
        {
            "workspace": tmp_path,
            "window_type": "sector",
            "window_value": "12, 13",
            "mission": "tess",
            "pipeline": "qlp",
            "lc_type": "sap",
            "quality": "hard",
            "exptime": "120",
            "sigma": "5",
            "filename": "tess qlp",
            "bandpass": "tess tess",
            "results_dir": "/previous/run",
            "period": "3.14",
            "period_err": "0.01",
            "epoch": "2457000.0",
            "epoch_err": "0.02",
            "duration": "2.5",
            "duration_err": "0.1",
            "depth": "1200",
            "depth_err": "40",
            "interactive": True,
            "update_db": True,
            "overwrite": True,
            "debug": True,
            "lc_only": True,
            "ttv": True,
        },
    )

    assert command[:5] == [sys.executable, "-m", "allesfitter", "prepare", "-toi"]
    assert command.count("--sector") == 2
    assert command.count("--filename") == 2
    assert command.count("--bandpass") == 2
    for flag in (
        "--lc_type",
        "--results_dir",
        "--period",
        "--period-err",
        "--epoch",
        "--epoch-err",
        "--duration",
        "--duration-err",
        "--depth",
        "--depth-err",
        "--interactive",
        "--update_db",
        "--overwrite",
        "--debug",
        "--lc-only",
        "--ttv",
        "-dir",
    ):
        assert flag in command


def test_fit_commands_are_shell_free_and_target_scoped(target: dict, tmp_path: Path):
    command = build_job_command(
        target,
        "optimize",
        {"method": "cmaes", "restarts": 3, "seed": 7, "no_refine": True},
        tmp_path,
    )
    assert command == [
        sys.executable,
        "-m",
        "allesfitter",
        "optimize",
        target["data_dir"],
        "--method",
        "cmaes",
        "--no-refine",
        "--restarts",
        "3",
        "--seed",
        "7",
    ]


def test_database_preserves_targets_and_job_history(tmp_path: Path):
    db_path = tmp_path / "workbench.sqlite3"
    db = WorkbenchDB(db_path)
    created = db.create_target("TOI-1097", "toi", "1097", tmp_path / "TOI-1097")
    job = db.create_job(
        created["id"],
        "prepare",
        ["python", "-m", "allesfitter", "prepare"],
        {"sector": "all"},
        tmp_path,
        tmp_path / "job.log",
    )
    db.update_job(job["id"], status="completed", returncode=0, finished_at="2026-07-17")

    reopened = WorkbenchDB(db_path)
    assert reopened.list_targets()[0]["job_count"] == 1
    persisted = reopened.list_jobs(created["id"])[0]
    assert persisted["status"] == "completed"
    assert persisted["command"][-1] == "prepare"
    assert persisted["options"] == {"sector": "all"}


def test_workbench_rejects_target_path_escape(tmp_path: Path):
    app = Workbench(tmp_path / "workspace")
    created = app.create_target(
        {"name": "HD 209458", "identifier_type": "name", "identifier_value": "HD 209458"}
    )
    with pytest.raises(ValueError, match="leaves the target directory"):
        app.target_path(created["id"], "../workbench.sqlite3")


def test_static_manifest_data_is_json_serializable(tmp_path: Path):
    app = Workbench(tmp_path / "workspace")
    app.create_target({"name": "TIC-1", "identifier_type": "tic", "identifier_value": "1"})
    json.dumps(app.state())


def test_workspace_defaults_to_the_shared_results_root():
    """Without --workspace the GUI collects targets and results in ~/ql/allesfitter."""
    from allesfitter.results import DEFAULT_OUTPUT_BASE

    assert resolve_workspace(None) == DEFAULT_OUTPUT_BASE
    assert resolve_workspace("") == DEFAULT_OUTPUT_BASE


def test_explicit_workspace_overrides_the_default(tmp_path: Path):
    assert resolve_workspace(tmp_path / "workspace") == tmp_path / "workspace"
    assert resolve_workspace("~/somewhere") == Path.home() / "somewhere"


def test_workbench_runner_keeps_results_inside_workspace(tmp_path: Path, monkeypatch):
    """The GUI routes results next to each target's data via ALLESFITTER_RESULTS_DIR."""
    from allesfitter.results import results_directory

    # Even when the ambient default points elsewhere, the workbench overrides it.
    monkeypatch.setenv("ALLESFITTER_RESULTS_DIR", str(tmp_path / "elsewhere"))
    app = Workbench(tmp_path / "workspace")
    created = app.create_target(
        {"name": "TOI-1097", "identifier_type": "name", "identifier_value": "TOI-1097"}
    )
    data_dir = Path(app.db.get_target(created["id"])["data_dir"])

    with monkeypatch.context() as ctx:
        ctx.setenv("ALLESFITTER_RESULTS_DIR", str(app.runner.workspace))
        mcmc = Path(results_directory(data_dir, "mcmc"))

    assert app.runner.workspace == app.workspace
    assert mcmc == data_dir / "mcmc_results"
