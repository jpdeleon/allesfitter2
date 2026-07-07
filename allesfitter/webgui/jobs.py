"""Launch and track fit runs as detached subprocesses.

allesfitter's config is module-global, so every fit runs in its **own** process
(muscat-db's ``start_new_session=True`` model). We spawn the run dir's generated
``run.py`` with ``ALLESFIT_SAMPLER`` set, stream its output to
``<run_dir>/results/run.log``, and a background waiter thread records the
terminal state (and scraped ``log(Z)``) in the job store when it exits.

The in-process ``_ACTIVE`` registry mirrors muscat-db's in-RAM live view; the
durable record lives in the job store, so a UI can still show finished/failed
runs after a restart.
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
from collections import deque
from dataclasses import dataclass
from pathlib import Path

from allesfitter.webgui import results
from allesfitter.webgui.job_store import JobStore

# run_id -> live handle for currently-running subprocesses (this process only).
_ACTIVE: dict[str, JobHandle] = {}
_LOCK = threading.Lock()


@dataclass
class JobHandle:
    """A live (or just-finished) subprocess launch."""

    run_id: str
    run_dir: Path
    sampler: str
    process: subprocess.Popen
    log_path: Path


def launch(
    store: JobStore,
    run_id: str,
    run_dir: str | Path,
    sampler: str = "mcmc",
    *,
    python: str | None = None,
    extra_env: dict | None = None,
) -> JobHandle:
    """Start ``run.py`` in *run_dir* as a detached subprocess.

    The job store is moved to ``running`` immediately and to ``done``/``failed``
    by the waiter thread when the process exits.
    """
    run_dir = Path(run_dir)
    if not (run_dir / "run.py").is_file():
        raise FileNotFoundError(f"no run.py in {run_dir} (was the config written?)")
    if sampler not in ("mcmc", "ns"):
        raise ValueError(f"sampler must be 'mcmc' or 'ns', got {sampler!r}")

    results_dir = run_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    log_path = results_dir / "run.log"

    env = dict(os.environ)
    env["ALLESFIT_SAMPLER"] = sampler
    env.setdefault("OMP_NUM_THREADS", "1")
    if extra_env:
        env.update(extra_env)

    log_file = open(log_path, "w")  # noqa: SIM115 -- closed by the waiter thread
    process = subprocess.Popen(
        [python or sys.executable, "run.py"],
        cwd=str(run_dir),
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    store.mark_running(run_id, process.pid)

    handle = JobHandle(run_id, run_dir, sampler, process, log_path)
    with _LOCK:
        _ACTIVE[run_id] = handle
    threading.Thread(target=_wait_and_finalize, args=(store, handle, log_file), daemon=True).start()
    return handle


def _wait_and_finalize(store: JobStore, handle: JobHandle, log_file) -> None:
    try:
        returncode = handle.process.wait()
    finally:
        log_file.close()
    logz = results.read_logz(handle.run_dir)
    error = "" if returncode == 0 else f"run.py exited with code {returncode}"
    store.mark_done(handle.run_id, returncode=returncode, logz=logz, error=error)
    with _LOCK:
        _ACTIVE.pop(handle.run_id, None)


def is_running(run_id: str) -> bool:
    """True while this process is running *run_id*'s subprocess."""
    with _LOCK:
        handle = _ACTIVE.get(run_id)
    return handle is not None and handle.process.poll() is None


def stop(run_id: str) -> bool:
    """Terminate a running job's process group. Returns True if one was signalled."""
    with _LOCK:
        handle = _ACTIVE.get(run_id)
    if handle is None or handle.process.poll() is not None:
        return False
    try:
        os.killpg(os.getpgid(handle.process.pid), 15)  # SIGTERM to the group
    except (ProcessLookupError, PermissionError):
        handle.process.terminate()
    return True


def tail_log(run_dir: str | Path, n: int = 200) -> str:
    """Return the last *n* lines of ``<run_dir>/results/run.log`` (or "")."""
    log_path = Path(run_dir) / "results" / "run.log"
    if not log_path.is_file():
        return ""
    try:
        with open(log_path, errors="ignore") as fh:
            return "".join(deque(fh, maxlen=n))
    except OSError:
        return ""
