"""Swappable job-store seam (ported from muscat-db's ``job_store.py``).

The web layer and the launcher hold a :class:`JobStore` rather than touching the
database directly, so a future multi-host backend (Celery/Redis) can be dropped
in via :func:`set_job_store` without changing callers. The default
:class:`RunStoreJobStore` delegates to the :class:`~allesfitter.webgui.runstore.RunStore`
``runs`` table — the runs table *is* the durable job record here, so there is no
second table to keep in sync.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from allesfitter.webgui import runstore
from allesfitter.webgui.runstore import RunStore


@runtime_checkable
class JobStore(Protocol):
    """Durable job state keyed by ``run_id``."""

    def enqueue(self, run_id: str) -> None:
        """Mark a created run as pending (queued, not yet launched)."""
        ...

    def mark_running(self, run_id: str, pid: int) -> None:
        """Record that a run's subprocess has started."""
        ...

    def mark_done(self, run_id: str, *, returncode: int, logz: str = "", error: str = "") -> None:
        """Record terminal state (done on rc==0, else failed)."""
        ...

    def get(self, run_id: str) -> dict | None: ...

    def pending(self) -> list[dict]: ...

    def all(self) -> list[dict]: ...


class RunStoreJobStore(JobStore):
    """:class:`JobStore` backed by a :class:`RunStore`."""

    def __init__(self, store: RunStore):
        self.store = store

    def enqueue(self, run_id: str) -> None:
        self.store.set_state(run_id, runstore.PENDING)

    def mark_running(self, run_id: str, pid: int) -> None:
        self.store.set_state(run_id, runstore.RUNNING, pid=pid)

    def mark_done(self, run_id: str, *, returncode: int, logz: str = "", error: str = "") -> None:
        state = runstore.DONE if returncode == 0 else runstore.FAILED
        self.store.set_state(run_id, state, logz=logz, error=error)

    def get(self, run_id: str) -> dict | None:
        return self.store.get_run(run_id)

    def pending(self) -> list[dict]:
        return self.store.pending()

    def all(self) -> list[dict]:
        return self.store.list_runs()


# Process-wide active store. The app installs one at startup via set_job_store();
# tests construct RunStoreJobStore directly.
_STORE: JobStore | None = None


def set_job_store(store: JobStore) -> None:
    """Install the process-wide job store (the swap point for Celery/Redis)."""
    global _STORE
    _STORE = store


def get_job_store() -> JobStore:
    """Return the installed job store, or raise if none was set."""
    if _STORE is None:
        raise RuntimeError("no job store installed; call set_job_store() first")
    return _STORE
