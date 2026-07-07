"""SQLite registry of fit runs (and derived per-target status).

Raw ``sqlite3`` in the muscat-db style: a single ``runs`` table is the durable
record; the per-target status badges shown in the UI are derived on the fly from
it (no second table to keep in sync).

Thread-safety: the launcher's background waiter thread writes terminal state
here, so every operation goes through :meth:`_cursor`, which serializes access
with a lock and closes per-call file connections. A ``:memory:`` store keeps one
shared ``check_same_thread=False`` connection alive for its lifetime.
"""

from __future__ import annotations

import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path

# Run lifecycle states.
CREATED = "created"
PENDING = "pending"
PREPARING = "preparing"  # prepare_allesfit subprocess is downloading/generating
PREPARED = "prepared"  # config generated + validated, ready for the user to fit
RUNNING = "running"
DONE = "done"
FAILED = "failed"
STOPPED = "stopped"

_ACTIVE_STATES = frozenset({CREATED, PENDING, PREPARING, RUNNING})

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id      TEXT PRIMARY KEY,
    target      TEXT NOT NULL,
    sampler     TEXT NOT NULL DEFAULT 'mcmc',
    state       TEXT NOT NULL DEFAULT 'created',
    run_dir     TEXT NOT NULL,
    insts       TEXT NOT NULL DEFAULT '',
    bands       TEXT NOT NULL DEFAULT '',
    companions  TEXT NOT NULL DEFAULT '',
    logz        TEXT NOT NULL DEFAULT '',
    error       TEXT NOT NULL DEFAULT '',
    pid         INTEGER,
    created_at  REAL NOT NULL,
    updated_at  REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_runs_target ON runs(target);
CREATE INDEX IF NOT EXISTS idx_runs_state ON runs(state);
"""


class RunStore:
    """Durable store of runs backed by a SQLite file (or ``:memory:``)."""

    def __init__(self, db_path: str | Path = "webgui.sqlite3"):
        self.db_path = str(db_path)
        self._lock = threading.RLock()
        # A shared in-memory DB must keep one connection alive for its lifetime;
        # it is reached from the waiter thread, hence check_same_thread=False.
        self._mem_conn = (
            sqlite3.connect(self.db_path, check_same_thread=False)
            if self.db_path == ":memory:"
            else None
        )
        with self._cursor() as conn:
            conn.executescript(SCHEMA)

    @contextmanager
    def _cursor(self):
        """Yield a connection under the lock; commit on success, close if per-call."""
        with self._lock:
            conn = self._mem_conn or sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            try:
                yield conn
                conn.commit()
            finally:
                if conn is not self._mem_conn:
                    conn.close()

    # -- writes -------------------------------------------------------------
    def create_run(
        self,
        *,
        run_id: str,
        target: str,
        run_dir: str | Path,
        sampler: str = "mcmc",
        insts: str = "",
        bands: str = "",
        companions: str = "",
        state: str = CREATED,
    ) -> dict:
        now = time.time()
        with self._cursor() as conn:
            conn.execute(
                "INSERT INTO runs (run_id, target, sampler, state, run_dir, insts, bands, "
                "companions, created_at, updated_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
                (run_id, target, sampler, state, str(run_dir), insts, bands, companions, now, now),
            )
        return self.get_run(run_id)  # type: ignore[return-value]

    def set_state(
        self,
        run_id: str,
        state: str,
        *,
        pid: int | None = None,
        logz: str | None = None,
        error: str | None = None,
    ) -> None:
        sets = ["state = ?", "updated_at = ?"]
        vals: list[object] = [state, time.time()]
        if pid is not None:
            sets.append("pid = ?")
            vals.append(pid)
        if logz is not None:
            sets.append("logz = ?")
            vals.append(logz)
        if error is not None:
            sets.append("error = ?")
            vals.append(error)
        vals.append(run_id)
        with self._cursor() as conn:
            conn.execute(f"UPDATE runs SET {', '.join(sets)} WHERE run_id = ?", vals)

    _UPDATABLE = frozenset(
        {"target", "sampler", "state", "run_dir", "insts", "bands", "companions", "logz", "error"}
    )

    def update_run(self, run_id: str, **fields: object) -> None:
        """Update whitelisted columns of a run in one statement (+ ``updated_at``).

        Used by the prepare pipeline to repoint ``run_dir`` at the discovered
        datadir and fill ``insts`` once generation finishes.
        """
        sets: list[str] = []
        vals: list[object] = []
        for key, value in fields.items():
            if key not in self._UPDATABLE:
                raise ValueError(f"cannot update unknown/immutable column {key!r}")
            sets.append(f"{key} = ?")
            vals.append(str(value) if isinstance(value, Path) else value)
        if not sets:
            return
        sets.append("updated_at = ?")
        vals.append(time.time())
        vals.append(run_id)
        with self._cursor() as conn:
            conn.execute(f"UPDATE runs SET {', '.join(sets)} WHERE run_id = ?", vals)

    def delete_run(self, run_id: str) -> None:
        with self._cursor() as conn:
            conn.execute("DELETE FROM runs WHERE run_id = ?", (run_id,))

    # -- reads --------------------------------------------------------------
    def get_run(self, run_id: str) -> dict | None:
        with self._cursor() as conn:
            row = conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        return dict(row) if row else None

    def list_runs(self, target: str | None = None) -> list[dict]:
        with self._cursor() as conn:
            if target is None:
                rows = conn.execute("SELECT * FROM runs ORDER BY created_at DESC").fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM runs WHERE target = ? ORDER BY created_at DESC", (target,)
                ).fetchall()
        return [dict(r) for r in rows]

    def pending(self) -> list[dict]:
        with self._cursor() as conn:
            rows = conn.execute(
                "SELECT * FROM runs WHERE state = ? ORDER BY created_at ASC", (PENDING,)
            ).fetchall()
        return [dict(r) for r in rows]

    def targets(self) -> list[dict]:
        """Per-target aggregate: run count and the most-recent run's state."""
        out: list[dict] = []
        with self._cursor() as conn:
            names = conn.execute(
                "SELECT target, COUNT(*) AS n, MAX(updated_at) AS updated_at "
                "FROM runs GROUP BY target ORDER BY updated_at DESC"
            ).fetchall()
            for row in names:
                latest = conn.execute(
                    "SELECT state FROM runs WHERE target = ? ORDER BY updated_at DESC LIMIT 1",
                    (row["target"],),
                ).fetchone()
                out.append(
                    {
                        "target": row["target"],
                        "n_runs": row["n"],
                        "fit_status": latest["state"] if latest else CREATED,
                        "updated_at": row["updated_at"],
                    }
                )
        return out
