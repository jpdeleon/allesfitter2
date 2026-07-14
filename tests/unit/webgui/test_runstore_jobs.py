"""Tests for the run registry, the job-store seam, and the subprocess launcher."""

from __future__ import annotations

import sqlite3
import sys
import threading
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


def test_scheduler_schema_migrates_populated_database(tmp_path):
    db = tmp_path / "legacy.sqlite3"
    conn = sqlite3.connect(db)
    conn.executescript(
        """
        CREATE TABLE runs (
          run_id TEXT PRIMARY KEY, target TEXT NOT NULL, sampler TEXT NOT NULL,
          state TEXT NOT NULL, run_dir TEXT NOT NULL, insts TEXT NOT NULL,
          bands TEXT NOT NULL, companions TEXT NOT NULL, logz TEXT NOT NULL,
          error TEXT NOT NULL, pid INTEGER, created_at REAL NOT NULL,
          updated_at REAL NOT NULL
        );
        INSERT INTO runs VALUES
          ('old', 'T', 'mcmc', 'done', '/tmp/old', '', '', '', '', '', NULL, 1, 1);
        """
    )
    conn.commit()
    conn.close()
    row = runstore.RunStore(db).get_run("old")
    assert row["owner"] == ""
    assert row["command"] == ""
    assert row["pgid"] is None


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


def _wait_for(predicate, timeout=10.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


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


def test_scheduler_enforces_capacity_and_fifo(tmp_path):
    s = _store()
    js = RunStoreJobStore(s)
    scheduler = jobs.LocalJobScheduler(js, max_concurrent=1, max_queued_per_user=5)
    try:
        for run_id in ("first", "second"):
            run_dir = tmp_path / run_id
            run_dir.mkdir()
            (run_dir / "run.py").write_text("import time\ntime.sleep(0.3)\n")
            s.create_run(
                run_id=run_id,
                target="T",
                run_dir=run_dir,
                owner="alice",
                state=runstore.PREPARED,
            )
            scheduler.enqueue(run_id, run_dir, owner="alice")
        assert _wait_for(lambda: s.get_run("first")["state"] == runstore.RUNNING)
        assert s.get_run("second")["state"] == runstore.PENDING
        assert _wait_state(s, "first") == runstore.DONE
        assert _wait_state(s, "second") == runstore.DONE
        assert s.get_run("first")["started_at"] <= s.get_run("second")["started_at"]
    finally:
        scheduler.close(stop_jobs=True)


def test_per_user_queue_limit_is_atomic(tmp_path):
    s = _store()
    js = RunStoreJobStore(s)
    for run_id in ("a", "b"):
        run_dir = tmp_path / run_id
        _write_fake_run_py(run_dir)
        s.create_run(
            run_id=run_id,
            target="T",
            run_dir=run_dir,
            owner="alice",
            state=runstore.PREPARED,
        )
    barrier = threading.Barrier(3)
    admitted = []

    def enqueue(run_id):
        barrier.wait()
        admitted.append(js.enqueue(run_id, owner="alice", limit=1))

    threads = [threading.Thread(target=enqueue, args=(run_id,)) for run_id in ("a", "b")]
    for thread in threads:
        thread.start()
    barrier.wait()
    for thread in threads:
        thread.join()
    assert sorted(admitted) == [False, True]
    assert len(s.pending()) == 1


def test_reconcile_uses_marker_and_never_assumes_missing_pid_succeeded(tmp_path):
    s = _store()
    js = RunStoreJobStore(s)
    for run_id, marker_code in (("ok", 0), ("bad", 7), ("stale", None)):
        run_dir = tmp_path / run_id
        (run_dir / "results").mkdir(parents=True)
        if marker_code is not None:
            (run_dir / "results" / "exit.json").write_text(f'{{"returncode": {marker_code}}}')
        s.create_run(run_id=run_id, target="T", run_dir=run_dir)
        s.set_state(run_id, runstore.RUNNING, pid=999999, pid_start="old")
    scheduler = jobs.LocalJobScheduler(js, max_concurrent=1)
    try:
        assert s.get_run("ok")["state"] == runstore.DONE
        assert s.get_run("bad")["state"] == runstore.FAILED
        assert s.get_run("bad")["returncode"] == 7
        assert s.get_run("stale")["state"] == runstore.FAILED
        assert "without a durable exit marker" in s.get_run("stale")["error"]
    finally:
        scheduler.close()


def test_cancellation_is_idempotent_and_terminal(tmp_path):
    s = _store()
    js = RunStoreJobStore(s)
    run_dir = tmp_path / "cancel"
    run_dir.mkdir()
    (run_dir / "run.py").write_text("import time\ntime.sleep(30)\n")
    s.create_run(run_id="cancel", target="T", run_dir=run_dir, state=runstore.PREPARED)
    scheduler = jobs.LocalJobScheduler(js, max_concurrent=1)
    try:
        scheduler.enqueue("cancel", run_dir)
        assert _wait_for(lambda: s.get_run("cancel")["state"] == runstore.RUNNING)
        assert scheduler.stop("cancel") is True
        assert scheduler.stop("cancel") is True
        assert _wait_for(lambda: s.get_run("cancel")["state"] == runstore.STOPPED)
        assert scheduler.stop("cancel") is True
    finally:
        scheduler.close(stop_jobs=True)
