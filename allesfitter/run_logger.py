"""Centralized run logger for allesfitter long-running operations.

Append-only JSONL log of every ``ns_fit`` / ``ns_output`` (or any other
named command) invocation. One row per event — a *start* row when the
call begins and an *end* row when it exits (success or failure). Each row
is a single self-contained JSON object so the file is easy to grep, tail,
or parse with ``jq``.

Why
---
When running many nested-sampling fits across datasets (or chromatic
sectors / campaigns / quarters), it is easy to lose track of which fits
are still going and which already produced results. The log file lets
you answer "which fits are currently running, on which host, against
which datadir?" with a single ``tail``.

Usage (from a user-side run.py)
-------------------------------
::

    import allesfitter
    from allesfitter.run_logger import log_run

    with log_run("ns_fit", "."):
        allesfitter.ns_fit(".")

    with log_run("ns_output", "."):
        allesfitter.ns_output(".")

Log location
------------
Default: ``~/.allesfitter/runs.jsonl``
Override with the ``ALLESFITTER_RUN_LOG`` environment variable (absolute
path). The parent directory is created on demand.

Concurrency
-----------
Writes are guarded with ``fcntl.flock`` so concurrent fits on the same
host append cleanly. The locking is per-write (the file is opened, locked,
appended, unlocked, closed) so a process crash never leaves a stale lock.

Schema
------
Each line is a JSON object with at least::

    {"event": "start"|"end",
     "run_id": "<uuid4 hex prefix>",
     "command": "<ns_fit|ns_output|...>",
     "datadir": "<absolute path>",
     "pid": <int>,
     "hostname": "<short host name>",
     "user": "<login name>",
     "timestamp": "<ISO-8601 UTC>"}

Plus, for ``event=end`` only::

    {"status": "completed"|"failed",
     "duration_sec": <float>,
     "error": "<str repr of exception, only when status=failed>"}
"""

from __future__ import annotations

import contextlib
import datetime as _dt
import fcntl
import getpass
import json
import os
import socket
import sys
import time
import traceback
import uuid
from pathlib import Path


__all__ = [
    "DEFAULT_LOG_PATH",
    "get_log_path",
    "log_run",
    "log_event",
    "tail",
]


DEFAULT_LOG_PATH = Path.home() / ".allesfitter" / "runs.jsonl"


def get_log_path() -> Path:
    """Resolve the active run-log path.

    Order of precedence:
      1. ``ALLESFITTER_RUN_LOG`` environment variable (must be an absolute
         path; relative values are resolved against the current working
         directory).
      2. ``~/.allesfitter/runs.jsonl``.
    """
    override = os.environ.get("ALLESFITTER_RUN_LOG")
    if override:
        return Path(override).expanduser().resolve()
    return DEFAULT_LOG_PATH


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def log_event(record: dict, log_path: Path | None = None) -> None:
    """Append a single JSON record to the run log.

    The record is shallow-copied before being augmented with default
    metadata (timestamp, pid, hostname, user) so the caller's dict is not
    mutated. Writes are guarded by an exclusive ``fcntl.flock`` so
    concurrent appends from sibling processes on the same host do not
    interleave.
    """
    path = log_path or get_log_path()
    _ensure_parent(path)
    payload = {
        "timestamp": _dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "pid": os.getpid(),
        "hostname": socket.gethostname().split(".")[0],
        "user": getpass.getuser(),
        **record,
    }
    line = json.dumps(payload, sort_keys=True) + "\n"
    with open(path, "a", encoding="utf-8") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            f.write(line)
            f.flush()
            os.fsync(f.fileno())
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


@contextlib.contextmanager
def log_run(command: str, datadir: str, log_path: Path | None = None,
            extra: dict | None = None, announce: bool = True):
    """Context manager that records start/end of an allesfitter command.

    Parameters
    ----------
    command : str
        Name of the operation being run, e.g. ``"ns_fit"`` or ``"ns_output"``.
        Free-form — the log file accepts any label so wrappers around
        other entrypoints (mcmc_fit, derive, ...) can reuse this.
    datadir : str
        The datadir the command operates on. Resolved to an absolute path
        so log entries remain meaningful when consumed from another cwd.
    log_path : Path, optional
        Override the resolved log file path. Mostly for tests.
    extra : dict, optional
        Additional fields to merge into the start record (e.g. version,
        git sha, host load average). Not merged into the end record.
    announce : bool
        When True (default), print a one-liner to stderr at start so the
        user can see where the log lives.

    Yields
    ------
    str
        The run_id used for both the start and end rows — handy if the
        caller wants to correlate other log lines.
    """
    run_id = uuid.uuid4().hex[:12]
    abs_datadir = str(Path(datadir).resolve())
    start_record = {
        "event": "start",
        "run_id": run_id,
        "command": command,
        "datadir": abs_datadir,
    }
    if extra:
        start_record.update(extra)
    log_event(start_record, log_path=log_path)
    if announce:
        path = log_path or get_log_path()
        print(
            f"[allesfitter.run_logger] {command} run_id={run_id} "
            f"datadir={abs_datadir} log={path}",
            file=sys.stderr,
        )
    t0 = time.perf_counter()
    try:
        yield run_id
    except BaseException as exc:
        log_event(
            {
                "event": "end",
                "run_id": run_id,
                "command": command,
                "datadir": abs_datadir,
                "status": "failed",
                "duration_sec": round(time.perf_counter() - t0, 3),
                "error": "{}: {}".format(type(exc).__name__, exc),
                "traceback": traceback.format_exc(limit=8),
            },
            log_path=log_path,
        )
        raise
    else:
        log_event(
            {
                "event": "end",
                "run_id": run_id,
                "command": command,
                "datadir": abs_datadir,
                "status": "completed",
                "duration_sec": round(time.perf_counter() - t0, 3),
            },
            log_path=log_path,
        )


def tail(n: int = 20, log_path: Path | None = None) -> list[dict]:
    """Return the last ``n`` parsed records from the run log.

    Convenience for interactive inspection; the log file remains the
    authoritative source. Lines that fail to parse as JSON are silently
    skipped so a partially-written tail line never crashes the reader.
    """
    path = log_path or get_log_path()
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    out = []
    for raw in lines[-n:]:
        raw = raw.strip()
        if not raw:
            continue
        try:
            out.append(json.loads(raw))
        except json.JSONDecodeError:
            continue
    return out
