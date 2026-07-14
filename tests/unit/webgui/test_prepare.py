"""Tests for the /prepare integration (prepare_allesfit -> web GUI).

The pure-logic tests (argv building, datadir discovery, settings parsing, and the
finalize state machine) run everywhere. The FastAPI route smokes monkeypatch the
subprocess launcher so nothing is actually downloaded or spawned.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from allesfitter.webgui import config_writer, jobs, prepare
from allesfitter.webgui import runstore as rs
from allesfitter.webgui.app import _tail_run_log, create_app
from allesfitter.webgui.validate import ValidationResult


def _after(argv: list[str], flag: str) -> str:
    """Return the token immediately following *flag* in *argv*."""
    return argv[argv.index(flag) + 1]


# --- build_prepare_argv ----------------------------------------------------
def test_argv_name_target_basic():
    argv = prepare.build_prepare_argv(
        {"target": "HD 39091", "id_type": "name", "mission": "tess", "sectors": "1"}
    )
    assert _after(argv, "-name") == "HD 39091"
    assert _after(argv, "-s") == "1"
    assert _after(argv, "-m") == "tess"
    assert _after(argv, "-f") == "tess"  # default filename


def test_argv_toi_selector_numeric():
    argv = prepare.build_prepare_argv({"target": "1234", "id_type": "toi", "sectors": "1"})
    assert _after(argv, "-toi") == "1234"


def test_argv_toi_selector_must_be_integer():
    with pytest.raises(ValueError, match="integer"):
        prepare.build_prepare_argv({"target": "HD 1", "id_type": "toi", "sectors": "1"})


def test_argv_requires_target():
    with pytest.raises(ValueError, match="target is required"):
        prepare.build_prepare_argv({"target": "", "sectors": "1"})


def test_argv_requires_segments():
    with pytest.raises(ValueError, match="sector"):
        prepare.build_prepare_argv({"target": "HD 1", "sectors": ""})


def test_argv_unknown_id_type():
    with pytest.raises(ValueError, match="id_type"):
        prepare.build_prepare_argv({"target": "x", "id_type": "gaia", "sectors": "1"})


def test_argv_multiple_sectors():
    argv = prepare.build_prepare_argv({"target": "HD 1", "sectors": "1 2 3"})
    i = argv.index("-s")
    assert argv[i + 1 : i + 4] == ["1", "2", "3"]


def test_argv_k2_uses_campaign_flag_single_value():
    argv = prepare.build_prepare_argv({"target": "HD 1", "mission": "k2", "sectors": "5 6"})
    assert "-s" not in argv
    assert _after(argv, "-c") == "5"  # campaign takes one value
    assert _after(argv, "-m") == "k2"


def test_argv_kepler_uses_quarter_flag():
    argv = prepare.build_prepare_argv({"target": "HD 1", "mission": "kepler", "sectors": "all"})
    assert _after(argv, "-q") == "all"


def test_argv_chromatic_filenames_and_bandpasses():
    argv = prepare.build_prepare_argv(
        {"target": "HD 1", "sectors": "1", "filename": "kepler tess", "bandpass": "kepler tess"}
    )
    fi = argv.index("-f")
    assert argv[fi + 1 : fi + 3] == ["kepler", "tess"]
    bi = argv.index("-bp")
    assert argv[bi + 1 : bi + 3] == ["kepler", "tess"]


def test_argv_exptime_adds_e_flag():
    argv = prepare.build_prepare_argv({"target": "HD 1", "sectors": "1", "exptime": "120"})
    assert _after(argv, "-e") == "120"


@pytest.mark.parametrize("exptime", [None, "", "  "])
def test_argv_exptime_auto_omits_e_flag(exptime):
    argv = prepare.build_prepare_argv({"target": "HD 1", "sectors": "1", "exptime": exptime})
    #:: "Auto" (blank) means let prepare_allesfit resolve the cadence itself
    assert "-e" not in argv


def test_argv_flags_and_scalars():
    argv = prepare.build_prepare_argv(
        {
            "target": "HD 1",
            "sectors": "1",
            "sigma": "3.0",
            "quality": "hard",
            "pipeline": "spoc",
            "lc_type": "sap",
            "ttv": True,
            "overwrite": True,
        }
    )
    assert _after(argv, "-sig") == "3.0"
    assert _after(argv, "-qb") == "hard"
    assert _after(argv, "-p") == "spoc"
    assert _after(argv, "-lc") == "sap"
    assert "--ttv" in argv
    assert "-o" in argv


def test_argv_work_dir_appends_dir_flag(tmp_path):
    argv = prepare.build_prepare_argv({"target": "HD 1", "sectors": "1"}, work_dir=tmp_path)
    assert _after(argv, "-dir") == str(tmp_path)


# --- discover_datadir / parse_inst_phot ------------------------------------
def _make_datadir(parent, name="HD-1", insts="tess"):
    d = parent / name
    d.mkdir(parents=True)
    (d / "params.csv").write_text("#name,value,fit,bounds,label,unit,coupled_with\n")
    (d / "settings.csv").write_text(f"#name,value\ninst_phot,{insts}\nbandpass,{insts}\n")
    return d


def test_discover_datadir_finds_generated_subdir(tmp_path):
    d = _make_datadir(tmp_path)
    assert prepare.discover_datadir(tmp_path) == d


def test_discover_datadir_none_when_incomplete(tmp_path):
    (tmp_path / "HD-1").mkdir()
    (tmp_path / "HD-1" / "params.csv").write_text("x")  # settings.csv missing
    assert prepare.discover_datadir(tmp_path) is None


def test_parse_inst_phot_reads_labels(tmp_path):
    d = _make_datadir(tmp_path, insts="tess kepler")
    assert prepare.parse_inst_phot(d / "settings.csv") == ["tess", "kepler"]


# --- _finalize state machine -----------------------------------------------
def _prepared_store_and_handle(tmp_path, run_id="r1"):
    store = rs.RunStore(":memory:")
    work_dir = tmp_path / run_id
    work_dir.mkdir()
    store.create_run(run_id=run_id, target="HD 1", run_dir=str(work_dir), state=rs.PREPARING)
    handle = prepare.PrepareHandle(
        run_id, work_dir, process=None, log_path=work_dir / "prepare.log"
    )
    return store, work_dir, handle


def test_finalize_success_registers_prepared(tmp_path, monkeypatch):
    store, work_dir, handle = _prepared_store_and_handle(tmp_path)
    datadir = _make_datadir(work_dir, insts="tess")
    monkeypatch.setattr(
        prepare._validate, "dry_run", lambda d: ValidationResult(True, n_free_params=5)
    )
    monkeypatch.setattr(
        prepare._validate,
        "initial_guess_preview",
        lambda d: Path(d) / "initial_guess.png",
    )

    prepare._finalize(store, handle, 0)

    row = store.get_run("r1")
    assert row["state"] == rs.PREPARED
    assert row["run_dir"] == str(datadir)
    assert row["insts"] == "tess"
    # run.py was re-emitted from the launcher-compatible template.
    assert (datadir / "run.py").read_text() == config_writer.RUN_PY_TEMPLATE


def test_finalize_requires_initial_guess_preview(tmp_path, monkeypatch):
    store, work_dir, handle = _prepared_store_and_handle(tmp_path)
    _make_datadir(work_dir)
    monkeypatch.setattr(prepare._validate, "dry_run", lambda d: ValidationResult(True))
    monkeypatch.setattr(prepare._validate, "initial_guess_preview", lambda d: None)

    prepare._finalize(store, handle, 0)

    row = store.get_run("r1")
    assert row["state"] == rs.FAILED
    assert "show_initial_guess" in row["error"]


def test_finalize_validation_failure_marks_failed_but_keeps_datadir(tmp_path, monkeypatch):
    store, work_dir, handle = _prepared_store_and_handle(tmp_path)
    datadir = _make_datadir(work_dir)
    monkeypatch.setattr(
        prepare._validate, "dry_run", lambda d: ValidationResult(False, error="boom")
    )

    prepare._finalize(store, handle, 0)

    row = store.get_run("r1")
    assert row["state"] == rs.FAILED
    assert "boom" in row["error"]
    assert row["run_dir"] == str(datadir)  # still inspectable


def test_finalize_nonzero_exit_marks_failed(tmp_path):
    store, _work_dir, handle = _prepared_store_and_handle(tmp_path)
    prepare._finalize(store, handle, 3)
    row = store.get_run("r1")
    assert row["state"] == rs.FAILED
    assert "code 3" in row["error"]


def test_finalize_no_datadir_marks_failed(tmp_path):
    store, _work_dir, handle = _prepared_store_and_handle(tmp_path)
    prepare._finalize(store, handle, 0)  # work_dir has no generated subdir
    assert store.get_run("r1")["state"] == rs.FAILED


# --- FastAPI route smokes --------------------------------------------------
def _app(tmp_path, **kw):
    return create_app(tmp_path / "runs", tmp_path / "db.sqlite3", **kw)


def test_prepare_page_renders(tmp_path):
    c = TestClient(_app(tmp_path, allow_network=True))
    resp = c.get("/prepare")
    assert resp.status_code == 200
    assert "prepare" in resp.text.lower()


def test_prepare_sectors_route_returns_exptimes(tmp_path, monkeypatch):
    from allesfitter.webgui import catalog

    monkeypatch.setattr(
        catalog,
        "available_sectors",
        lambda *a, **k: catalog.SectorAvailability(
            target="HD 1",
            mission="tess",
            segments=["1"],
            pipelines=["spoc"],
            exptimes=[20, 120],
        ),
    )
    c = TestClient(_app(tmp_path, allow_network=True))
    resp = c.get("/prepare/sectors", params={"target": "HD 1", "mission": "tess"})
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["ok"] is True
    assert data["exptimes"] == [20, 120]
    assert data["segments"] == ["1"]


def test_prepare_run_launches_and_registers(tmp_path, monkeypatch):
    launched = []
    monkeypatch.setattr(prepare, "launch_prepare", lambda *a, **k: launched.append(a))
    app = _app(tmp_path, allow_network=True)
    c = TestClient(app)
    resp = c.post(
        "/prepare/run",
        json={"target": "HD 39091", "id_type": "name", "sectors": "1", "filename": "tess"},
    )
    assert resp.status_code == 200, resp.text
    run_id = resp.json()["run_id"]
    assert launched  # subprocess launcher was invoked
    runs = c.get("/jobs/status").json()["runs"]
    assert any(r["run_id"] == run_id and r["state"] == rs.PREPARING for r in runs)


def test_prepare_run_makes_relative_runs_root_absolute(tmp_path, monkeypatch):
    """The child cwd must not be prepended to ``-dir`` a second time."""
    monkeypatch.chdir(tmp_path)
    app = create_app("runs", "db.sqlite3", allow_network=True)

    assert app.state.runs_root == (tmp_path / "runs").resolve()


def test_failed_prepare_log_is_available_from_jobs_log(tmp_path):
    work_dir = tmp_path / "runs" / "r1"
    work_dir.mkdir(parents=True)
    (work_dir / "prepare.log").write_text("prepare failure details\n")

    assert _tail_run_log(work_dir, work_dir) == "prepare failure details\n"


def test_failed_prepare_log_is_shown_on_results_page(tmp_path):
    work_dir = tmp_path / "runs" / "r1"
    work_dir.mkdir(parents=True)
    (work_dir / "prepare.log").write_text("prepare failure details\n")
    template = Path(__file__).parents[3] / "allesfitter/webgui/templates/results.html"

    assert "run_log" in template.read_text()


def test_prepare_run_blocked_without_network(tmp_path):
    c = TestClient(_app(tmp_path, allow_network=False))
    resp = c.post("/prepare/run", json={"target": "HD 1", "sectors": "1"})
    assert resp.status_code == 400
    assert "network" in resp.json()["detail"].lower()


def test_prepare_run_rejects_bad_input(tmp_path):
    c = TestClient(_app(tmp_path, allow_network=True))
    resp = c.post("/prepare/run", json={"target": "HD 1", "sectors": ""})
    assert resp.status_code == 400


def test_jobs_fit_launches_prepared_run(tmp_path, monkeypatch):
    launched = []
    monkeypatch.setattr(jobs, "launch", lambda *a, **k: launched.append((a, k)))
    app = _app(tmp_path, allow_network=True)
    c = TestClient(app)
    run_dir = tmp_path / "runs" / "r1"
    run_dir.mkdir(parents=True)
    app.state.store.create_run(run_id="r1", target="HD 1", run_dir=str(run_dir), state=rs.PREPARED)

    resp = c.post("/jobs/fit/r1", params={"sampler": "ns"})
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"run_id": "r1", "sampler": "ns"}
    assert launched and launched[0][1]["sampler"] == "ns"
    assert app.state.store.get_run("r1")["sampler"] == "ns"


def test_jobs_fit_unknown_run_404(tmp_path):
    c = TestClient(_app(tmp_path, allow_network=True))
    assert c.post("/jobs/fit/nope").status_code == 404


def test_prepared_run_can_edit_configs_and_refresh_preview(tmp_path, monkeypatch):
    app = _app(tmp_path, allow_network=True)
    c = TestClient(app)
    run_dir = tmp_path / "runs" / "r1"
    run_dir.mkdir(parents=True)
    (run_dir / "params.csv").write_text("old params\n")
    (run_dir / "settings.csv").write_text("old settings\n")
    (run_dir / "initial_guess.png").write_bytes(b"old")
    app.state.store.create_run(run_id="r1", target="HD 1", run_dir=str(run_dir), state=rs.PREPARED)
    monkeypatch.setattr(
        "allesfitter.webgui.app._validate.dry_run",
        lambda d: ValidationResult(True, n_free_params=7),
    )

    def preview(path):
        output = Path(path) / "initial_guess.png"
        output.write_bytes(b"new")
        return output

    monkeypatch.setattr("allesfitter.webgui.app._validate.initial_guess_preview", preview)

    page = c.get("/results/r1")
    assert "params-csv" in page.text
    assert "Run MCMC" in page.text
    assert "Run Nested Sampling" in page.text
    response = c.post(
        "/results/r1/config",
        json={"params_csv": "new params\n", "settings_csv": "new settings\n"},
    )
    assert response.status_code == 200
    assert response.json() == {"ok": True, "n_free_params": 7}
    assert (run_dir / "params.csv").read_text() == "new params\n"
    assert (run_dir / "settings.csv").read_text() == "new settings\n"
    assert c.get("/results/r1/initial-guess").content == b"new"


def test_unprepared_run_cannot_be_edited_or_fitted(tmp_path):
    app = _app(tmp_path, allow_network=True)
    c = TestClient(app)
    run_dir = tmp_path / "runs" / "r1"
    run_dir.mkdir(parents=True)
    app.state.store.create_run(run_id="r1", target="HD 1", run_dir=str(run_dir), state=rs.PREPARING)

    assert c.post("/results/r1/config", json={}).status_code == 409
    assert c.post("/jobs/fit/r1").status_code == 409
