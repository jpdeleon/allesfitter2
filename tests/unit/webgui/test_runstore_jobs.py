"""Tests for the run registry, the job-store seam, and the subprocess launcher."""

from __future__ import annotations

import sys
import time

from allesfitter.webgui import jobs, runstore
from allesfitter.webgui.job_store import JobStore, RunStoreJobStore


def _store():
    return runstore.RunStore(":memory:")


# --- RunStore --------------------------------------------------------------
def test_create_and_get_run():
    s = _store()
    s.create_run(run_id="r1", target="TOI-1", run_dir="/tmp/r1", insts="tess", bands="tess")
    row = s.get_run("r1")
    assert row["target"] == "TOI-1"
    assert row["state"] == runstore.CREATED
    assert s.get_run("missing") is None


def test_set_state_and_fields():
    s = _store()
    s.create_run(run_id="r1", target="T", run_dir="/tmp/r1")
    s.set_state("r1", runstore.RUNNING, pid=4242)
    assert s.get_run("r1")["pid"] == 4242
    s.set_state("r1", runstore.DONE, logz="-12.3 +- 0.4")
    row = s.get_run("r1")
    assert row["state"] == runstore.DONE
    assert row["logz"] == "-12.3 +- 0.4"


def test_list_and_pending_and_targets():
    s = _store()
    s.create_run(run_id="a", target="T1", run_dir="/tmp/a")
    s.create_run(run_id="b", target="T1", run_dir="/tmp/b")
    s.create_run(run_id="c", target="T2", run_dir="/tmp/c")
    s.set_state("b", runstore.PENDING)
    assert {r["run_id"] for r in s.list_runs("T1")} == {"a", "b"}
    assert [r["run_id"] for r in s.pending()] == ["b"]
    targets = {t["target"]: t for t in s.targets()}
    assert targets["T1"]["n_runs"] == 2
    assert targets["T2"]["n_runs"] == 1


# --- job store seam --------------------------------------------------------
def test_runstore_jobstore_is_protocol_conformant():
    js = RunStoreJobStore(_store())
    assert isinstance(js, JobStore)  # runtime_checkable Protocol


def test_jobstore_transitions():
    s = _store()
    js = RunStoreJobStore(s)
    s.create_run(run_id="r1", target="T", run_dir="/tmp/r1")
    js.enqueue("r1")
    assert js.get("r1")["state"] == runstore.PENDING
    assert [r["run_id"] for r in js.pending()] == ["r1"]
    js.mark_running("r1", 999)
    assert js.get("r1")["state"] == runstore.RUNNING
    js.mark_done("r1", returncode=1, error="boom")
    assert js.get("r1")["state"] == runstore.FAILED
    assert js.get("r1")["error"] == "boom"


# --- launcher --------------------------------------------------------------
def _write_fake_run_py(run_dir, *, exit_code=0, emit_logz=True):
    run_dir.mkdir(parents=True, exist_ok=True)
    logz_line = 'print("log(Z) = -123.4 +- 0.5")' if emit_logz else ""
    (run_dir / "run.py").write_text(f"import sys\n{logz_line}\nsys.exit({exit_code})\n")


def _wait_state(store, run_id, timeout=15.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        state = store.get_run(run_id)["state"]
        if state in (runstore.DONE, runstore.FAILED):
            return state
        time.sleep(0.05)
    return store.get_run(run_id)["state"]


def test_launch_success_records_done_and_logz(tmp_path):
    s = _store()
    js = RunStoreJobStore(s)
    run_dir = tmp_path / "run"
    _write_fake_run_py(run_dir, exit_code=0)
    s.create_run(run_id="r1", target="T", run_dir=str(run_dir))

    jobs.launch(js, "r1", run_dir, sampler="mcmc", python=sys.executable)
    assert _wait_state(s, "r1") == runstore.DONE
    row = s.get_run("r1")
    assert row["logz"] == "-123.4 +- 0.5"
    # log(Z) line is captured in the streamed log
    assert "log(Z)" in jobs.tail_log(run_dir)


def test_launch_failure_records_failed(tmp_path):
    s = _store()
    js = RunStoreJobStore(s)
    run_dir = tmp_path / "run"
    _write_fake_run_py(run_dir, exit_code=3, emit_logz=False)
    s.create_run(run_id="r2", target="T", run_dir=str(run_dir))

    jobs.launch(js, "r2", run_dir, sampler="mcmc", python=sys.executable)
    assert _wait_state(s, "r2") == runstore.FAILED
    assert "code 3" in s.get_run("r2")["error"]
