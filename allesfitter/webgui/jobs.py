"""Bounded, durable, single-host fit scheduler."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from allesfitter.webgui import results
from allesfitter.webgui import runstore as rs
from allesfitter.webgui.job_store import JobStore


def _app_version() -> str:
    try:
        return version("allesfitter")
    except PackageNotFoundError:
        return "development"


def _pid_start(pid: int) -> str:
    """Linux process start token, used to avoid signalling a reused PID."""
    try:
        return Path(f"/proc/{pid}/stat").read_text().split()[21]
    except (OSError, IndexError):
        return ""


def _process_matches(row: dict) -> bool:
    pid = row.get("pid")
    if not pid:
        return False
    try:
        os.kill(int(pid), 0)
    except (ProcessLookupError, PermissionError, OSError):
        return False
    recorded = row.get("pid_start") or ""
    return bool(recorded and recorded == _pid_start(int(pid)))


def _marker(run_dir: str | Path) -> Path:
    return Path(run_dir) / "results" / "exit.json"


def _marker_returncode(run_dir: str | Path) -> int | None:
    try:
        value = json.loads(_marker(run_dir).read_text()).get("returncode")
        return int(value)
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None


@dataclass
class JobHandle:
    run_id: str
    run_dir: Path
    sampler: str
    process: subprocess.Popen
    log_path: Path


class QueueLimitError(RuntimeError):
    pass


class LocalJobScheduler:
    """FIFO scheduler whose public methods form the replaceable backend seam."""

    def __init__(
        self,
        store: JobStore,
        *,
        max_concurrent: int = 1,
        max_queued_per_user: int = 2,
        python: str | None = None,
        poll_interval: float = 0.2,
        autostart: bool = True,
    ):
        if max_concurrent < 1 or max_queued_per_user < 1:
            raise ValueError("scheduler limits must be at least 1")
        self.store = store
        self.max_concurrent = max_concurrent
        self.max_queued_per_user = max_queued_per_user
        self.python = python or sys.executable
        self.poll_interval = poll_interval
        self._lock = threading.RLock()
        self._wake = threading.Event()
        self._closed = threading.Event()
        self._active: dict[str, JobHandle] = {}
        self._thread: threading.Thread | None = None
        self.reconcile()
        if autostart:
            self.start()

    def start(self) -> None:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._thread = threading.Thread(
                target=self._run, name="allesfitter-scheduler", daemon=True
            )
            self._thread.start()

    def enqueue(
        self,
        run_id: str,
        run_dir: str | Path,
        sampler: str = "mcmc",
        *,
        owner: str = "",
        extra_env: dict[str, str] | None = None,
    ) -> None:
        self.start()
        if sampler not in ("mcmc", "ns", "optimize"):
            raise ValueError(f"sampler must be 'mcmc', 'ns', or 'optimize', got {sampler!r}")
        if not (Path(run_dir) / "run.py").is_file():
            raise FileNotFoundError(f"no run.py in {run_dir} (was the config written?)")
        # Only non-secret, scheduler-owned fit options are persisted in command.
        command = self._command(run_id, run_dir, extra_env=extra_env)
        self.store.record_command(run_id, json.dumps(command))
        if not self.store.enqueue(run_id, owner=owner, limit=self.max_queued_per_user):
            row = self.store.get(run_id)
            if row and row["state"] != rs.PREPARED:
                raise QueueLimitError("fit is already queued or active")
            raise QueueLimitError(f"user already has {self.max_queued_per_user} pending fits")
        self._wake.set()

    def _command(
        self, run_id: str, run_dir: str | Path, *, extra_env: dict[str, str] | None = None
    ) -> list[str]:
        command = [
            self.python,
            "-m",
            "allesfitter.webgui.job_worker",
            "--run-id",
            run_id,
            "--marker",
            str(_marker(run_dir)),
        ]
        allowed = {
            "ALLESFIT_OPTIMIZE_METHOD",
            "ALLESFIT_OPTIMIZE_REFINE",
            "ALLESFIT_OPTIMIZE_RESTARTS",
        }
        for key, value in sorted((extra_env or {}).items()):
            if key not in allowed:
                raise ValueError(f"unsupported persisted fit option {key!r}")
            command.extend(("--env", f"{key}={value}"))
        command.append("run.py")
        return command

    def _run(self) -> None:
        while not self._closed.is_set():
            self._monitor_recovered()
            self._dispatch()
            self._wake.wait(self.poll_interval)
            self._wake.clear()

    def _dispatch(self) -> None:
        with self._lock:
            occupied = sum(
                1 for row in self.store.all() if row["state"] in (rs.RUNNING, rs.STOPPING)
            )
            for row in self.store.pending():
                if occupied >= self.max_concurrent:
                    break
                self._launch_row(row)
                occupied += 1

    def _launch_row(self, row: dict) -> JobHandle:
        run_dir = Path(row["run_dir"])
        results_dir = run_dir / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        marker = _marker(run_dir)
        marker.unlink(missing_ok=True)
        log_path = results_dir / "run.log"
        try:
            command = json.loads(row.get("command") or "")
            if not isinstance(command, list) or not command:
                raise ValueError
        except (ValueError, TypeError, json.JSONDecodeError):
            command = self._command(row["run_id"], run_dir)
        env = dict(os.environ)
        env["ALLESFIT_SAMPLER"] = row["sampler"]
        env.setdefault("OMP_NUM_THREADS", "1")
        log_file = open(log_path, "w")  # noqa: SIM115 - waiter owns it
        try:
            process = subprocess.Popen(
                command,
                cwd=str(run_dir),
                env=env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        except BaseException:
            log_file.close()
            self.store.set_state(row["run_id"], rs.FAILED, error="fit process failed to start")
            raise
        handle = JobHandle(row["run_id"], run_dir, row["sampler"], process, log_path)
        self.store.mark_running(
            row["run_id"],
            process.pid,
            pgid=os.getpgid(process.pid),
            command=json.dumps(command),
            app_version=_app_version(),
            pid_start=_pid_start(process.pid),
        )
        self._active[row["run_id"]] = handle
        threading.Thread(
            target=self._wait_and_finalize, args=(handle, log_file), daemon=True
        ).start()
        return handle

    def _wait_and_finalize(self, handle: JobHandle, log_file) -> None:
        returncode = handle.process.wait()
        log_file.close()
        self._finalize(handle.run_id, handle.run_dir, returncode)
        with self._lock:
            self._active.pop(handle.run_id, None)
        self._wake.set()

    def _finalize(self, run_id: str, run_dir: Path, returncode: int | None = None) -> None:
        row = self.store.get(run_id)
        if row is None or row["state"] in rs.TERMINAL_STATES:
            return
        if row["state"] == rs.STOPPING:
            self.store.set_state(run_id, rs.STOPPED, returncode=returncode or -signal.SIGTERM)
            return
        marker_code = _marker_returncode(run_dir)
        if marker_code is None:
            self.store.set_state(
                run_id,
                rs.FAILED,
                returncode=returncode if returncode is not None else -1,
                error="fit process disappeared without a durable exit marker",
            )
            return
        logz = results.read_logz(run_dir)
        error = "" if marker_code == 0 else f"run.py exited with code {marker_code}"
        if marker_code == 0 and row["sampler"] == "optimize":
            self.store.set_state(run_id, rs.PREPARED, returncode=marker_code, error="")
        else:
            self.store.set_state(
                run_id,
                rs.DONE if marker_code == 0 else rs.FAILED,
                logz=logz,
                error=error,
                returncode=marker_code,
            )

    def reconcile(self) -> None:
        """Recover running/stopping records and leave pending work queued."""
        for row in self.store.all():
            if row["state"] not in (rs.RUNNING, rs.STOPPING):
                continue
            if _process_matches(row):
                continue
            self._finalize(row["run_id"], Path(row["run_dir"]))

    def _monitor_recovered(self) -> None:
        for row in self.store.all():
            if row["state"] in (rs.RUNNING, rs.STOPPING) and row["run_id"] not in self._active:
                if not _process_matches(row):
                    self._finalize(row["run_id"], Path(row["run_dir"]))

    def stop(self, run_id: str) -> bool:
        with self._lock:
            row = self.store.get(run_id)
            if row is None:
                return False
            if row["state"] == rs.PENDING:
                self.store.set_state(run_id, rs.STOPPED, error="Cancelled while pending")
                self._wake.set()
                return True
            if row["state"] in rs.TERMINAL_STATES:
                return True
            if row["state"] == rs.STOPPING:
                return True
            if row["state"] != rs.RUNNING:
                return False
            self.store.set_state(run_id, rs.STOPPING, error="Cancellation requested")
            if not _process_matches(row):
                self._finalize(run_id, Path(row["run_dir"]))
                return True
            try:
                os.killpg(int(row["pgid"]), signal.SIGTERM)
            except (ProcessLookupError, PermissionError, OSError):
                self._finalize(run_id, Path(row["run_dir"]))
            return True

    def is_running(self, run_id: str) -> bool:
        row = self.store.get(run_id)
        return bool(row and row["state"] in (rs.RUNNING, rs.STOPPING) and _process_matches(row))

    def close(self, *, stop_jobs: bool = False) -> None:
        if stop_jobs:
            for row in self.store.all():
                if row["state"] in (rs.RUNNING, rs.PENDING):
                    self.stop(row["run_id"])
        self._closed.set()
        self._wake.set()
        if self._thread is not None:
            self._thread.join(timeout=2)


_SCHEDULER: LocalJobScheduler | None = None


def set_scheduler(scheduler: LocalJobScheduler | None) -> None:
    global _SCHEDULER
    if _SCHEDULER is not None and _SCHEDULER is not scheduler:
        _SCHEDULER.close()
    _SCHEDULER = scheduler


def get_scheduler() -> LocalJobScheduler:
    if _SCHEDULER is None:
        raise RuntimeError("no scheduler installed")
    return _SCHEDULER


def launch(
    store: JobStore,
    run_id: str,
    run_dir: str | Path,
    sampler: str = "mcmc",
    *,
    python: str | None = None,
    extra_env: dict | None = None,
) -> JobHandle | None:
    """Compatibility helper: enqueue on the app scheduler, or launch one standalone job."""
    if _SCHEDULER is not None and _SCHEDULER.store is store:
        row = store.get(run_id) or {}
        _SCHEDULER.enqueue(
            run_id, run_dir, sampler, owner=row.get("owner", ""), extra_env=extra_env
        )
        return None
    scheduler = LocalJobScheduler(store, max_concurrent=1, python=python)
    set_scheduler(scheduler)
    if (store.get(run_id) or {}).get("state") != rs.PREPARED:
        store.set_state(run_id, rs.PREPARED)
    scheduler.enqueue(run_id, run_dir, sampler, owner=(store.get(run_id) or {}).get("owner", ""))
    deadline = time.time() + 2
    while time.time() < deadline:
        with scheduler._lock:
            handle = scheduler._active.get(run_id)
        if handle:
            return handle
        time.sleep(0.01)
    return None


def is_running(run_id: str) -> bool:
    return bool(_SCHEDULER and _SCHEDULER.is_running(run_id))


def stop(run_id: str) -> bool:
    return bool(_SCHEDULER and _SCHEDULER.stop(run_id))


def tail_log(run_dir: str | Path, n: int = 200) -> str:
    log_path = Path(run_dir) / "results" / "run.log"
    if not log_path.is_file():
        return ""
    try:
        with open(log_path, errors="ignore") as fh:
            return "".join(deque(fh, maxlen=n))
    except OSError:
        return ""
