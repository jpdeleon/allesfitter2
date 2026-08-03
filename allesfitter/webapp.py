"""Browser workbench for preparing, running, and reviewing allesfitter jobs.

The server deliberately uses only the Python standard library.  It is intended
to bind to localhost on a remote research machine and be reached through SSH
port forwarding.
"""

from __future__ import annotations

import csv
import json
import mimetypes
import os
import re
import signal
import sqlite3
import subprocess
import sys
import threading
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

from .results import DEFAULT_OUTPUT_BASE, OUTPUT_BASE_ENV

ASSET_DIR = Path(__file__).with_name("web_static")
IDENTIFIER_TYPES = {"toi", "ctoi", "tic", "name"}
JOB_KINDS = {
    "prepare",
    "show-initial-guess",
    "optimize",
    "mcmc-fit",
    "mcmc-output",
    "ns-fit",
    "ns-output",
}
EDITABLE_FILES = {"params.csv", "params_star.csv", "settings.csv"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _slug(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip()).strip("-.")
    return value or "target"


_PREFIXES: dict[str, str] = {"toi": "TOI-", "tic": "TIC-", "ctoi": "CTOI-"}


def _strip_catalog_prefix(identifier_type: str, raw: str) -> str:
    prefix = _PREFIXES.get(identifier_type)
    if prefix and raw.strip().upper().startswith(prefix):
        return raw.strip()[len(prefix) :]
    return raw.strip()


def canonical_target_name(identifier_type: str, identifier_value: str, display_name: str) -> str:
    if identifier_type == "toi":
        return f"TOI-{str(identifier_value).zfill(4)}"
    if identifier_type == "ctoi":
        return f"CTOI-{identifier_value}"
    if identifier_type == "tic":
        return f"TIC-{identifier_value}"
    return display_name.replace(" ", "")


class WorkbenchDB:
    """Small SQLite store; each operation owns its connection for thread safety."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.path, timeout=20)
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA foreign_keys = ON")
        con.execute("PRAGMA journal_mode = WAL")
        return con

    def _init_schema(self) -> None:
        with self.connect() as con:
            con.executescript(
                """
                CREATE TABLE IF NOT EXISTS targets (
                    id INTEGER PRIMARY KEY,
                    name TEXT NOT NULL,
                    identifier_type TEXT NOT NULL,
                    identifier_value TEXT NOT NULL,
                    data_dir TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE(identifier_type, identifier_value)
                );
                CREATE TABLE IF NOT EXISTS jobs (
                    id INTEGER PRIMARY KEY,
                    target_id INTEGER NOT NULL REFERENCES targets(id) ON DELETE CASCADE,
                    kind TEXT NOT NULL,
                    status TEXT NOT NULL,
                    command_json TEXT NOT NULL,
                    options_json TEXT NOT NULL,
                    cwd TEXT NOT NULL,
                    log_path TEXT NOT NULL,
                    pid INTEGER,
                    returncode INTEGER,
                    created_at TEXT NOT NULL,
                    started_at TEXT,
                    finished_at TEXT
                );
                CREATE INDEX IF NOT EXISTS jobs_target_created
                    ON jobs(target_id, created_at DESC);
                """
            )
            con.execute(
                "UPDATE jobs SET status='interrupted', finished_at=? WHERE status IN ('queued', 'running')",
                (_now(),),
            )

    @staticmethod
    def _row(row: sqlite3.Row | None) -> dict[str, Any] | None:
        if row is None:
            return None
        item = dict(row)
        for key in ("command_json", "options_json"):
            if key in item:
                item[key.removesuffix("_json")] = json.loads(item.pop(key))
        return item

    def create_target(
        self, name: str, identifier_type: str, identifier_value: str, data_dir: Path
    ) -> dict[str, Any]:
        if identifier_type not in IDENTIFIER_TYPES:
            raise ValueError("identifier_type must be toi, ctoi, tic, or name")
        if not name.strip() or not identifier_value.strip():
            raise ValueError("Target name and identifier are required")
        timestamp = _now()
        try:
            with self.connect() as con:
                cursor = con.execute(
                    """INSERT INTO targets
                       (name, identifier_type, identifier_value, data_dir, created_at, updated_at)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (
                        name.strip(),
                        identifier_type,
                        identifier_value.strip(),
                        str(data_dir),
                        timestamp,
                        timestamp,
                    ),
                )
                target_id = cursor.lastrowid
        except sqlite3.IntegrityError as exc:
            raise ValueError("This target is already in the workbench") from exc
        return self.get_target(int(target_id))

    def get_target(self, target_id: int) -> dict[str, Any]:
        with self.connect() as con:
            row = con.execute("SELECT * FROM targets WHERE id=?", (target_id,)).fetchone()
        item = self._row(row)
        if item is None:
            raise KeyError("Target not found")
        return item

    def delete_target(self, target_id: int) -> None:
        with self.connect() as con:
            con.execute("PRAGMA foreign_keys = ON")
            con.execute("DELETE FROM targets WHERE id=?", (target_id,))

    def list_targets(self) -> list[dict[str, Any]]:
        with self.connect() as con:
            rows = con.execute(
                """SELECT t.*,
                    (SELECT COUNT(*) FROM jobs j WHERE j.target_id=t.id) AS job_count,
                    (SELECT status FROM jobs j WHERE j.target_id=t.id
                     ORDER BY j.created_at DESC, j.id DESC LIMIT 1) AS latest_status
                    FROM targets t ORDER BY t.updated_at DESC, t.id DESC"""
            ).fetchall()
        return [self._row(row) for row in rows]  # type: ignore[misc]

    def create_job(
        self,
        target_id: int,
        kind: str,
        command: list[str],
        options: dict[str, Any],
        cwd: Path,
        log_path: Path,
    ) -> dict[str, Any]:
        timestamp = _now()
        with self.connect() as con:
            cursor = con.execute(
                """INSERT INTO jobs
                   (target_id, kind, status, command_json, options_json, cwd, log_path, created_at)
                   VALUES (?, ?, 'queued', ?, ?, ?, ?, ?)""",
                (
                    target_id,
                    kind,
                    json.dumps(command),
                    json.dumps(options, sort_keys=True),
                    str(cwd),
                    str(log_path),
                    timestamp,
                ),
            )
            con.execute("UPDATE targets SET updated_at=? WHERE id=?", (timestamp, target_id))
            job_id = cursor.lastrowid
        return self.get_job(int(job_id))

    def get_job(self, job_id: int) -> dict[str, Any]:
        with self.connect() as con:
            row = con.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        item = self._row(row)
        if item is None:
            raise KeyError("Job not found")
        return item

    def list_jobs(self, target_id: int | None = None) -> list[dict[str, Any]]:
        query = "SELECT * FROM jobs"
        params: tuple[Any, ...] = ()
        if target_id is not None:
            query += " WHERE target_id=?"
            params = (target_id,)
        query += " ORDER BY created_at DESC, id DESC"
        with self.connect() as con:
            rows = con.execute(query, params).fetchall()
        return [self._row(row) for row in rows]  # type: ignore[misc]

    def update_job(self, job_id: int, **values: Any) -> None:
        allowed = {"status", "pid", "returncode", "started_at", "finished_at"}
        if not values or not set(values) <= allowed:
            raise ValueError("Invalid job update")
        assignments = ", ".join(f"{key}=?" for key in values)
        with self.connect() as con:
            con.execute(
                f"UPDATE jobs SET {assignments} WHERE id=?",  # noqa: S608 - fixed allow-list
                (*values.values(), job_id),
            )


def build_prepare_command(target: dict[str, Any], options: dict[str, Any]) -> list[str]:
    """Build a shell-free prepare command from the complete public CLI surface."""
    window_type = str(options.get("window_type", "sector"))
    window_value = str(options.get("window_value", "-1")).strip()
    if window_type not in {"sector", "campaign", "quarter"} or not window_value:
        raise ValueError("Choose a sector, campaign, or quarter")

    command = [sys.executable, "-m", "allesfitter", "prepare"]
    command += [f"-{target['identifier_type']}", str(target["identifier_value"])]
    window_flag = {"sector": "--sector", "campaign": "--campaign", "quarter": "--quarter"}[
        window_type
    ]
    values = [value for value in re.split(r"[\s,]+", window_value) if value]
    if window_type == "sector":
        for value in values:
            command += [window_flag, value]
    else:
        command += [window_flag, values[0]]

    scalar_options = {
        "mission": "--mission",
        "pipeline": "--pipeline",
        "lc_type": "--lc_type",
        "quality": "--quality",
        "exptime": "--exptime",
        "sigma": "--sigma",
        "results_dir": "--results_dir",
        "period": "--period",
        "period_err": "--period-err",
        "epoch": "--epoch",
        "epoch_err": "--epoch-err",
        "duration": "--duration",
        "duration_err": "--duration-err",
        "depth": "--depth",
        "depth_err": "--depth-err",
    }
    defaults = {"mission": "tess", "pipeline": "spoc", "lc_type": "pdcsap", "quality": "default"}
    for key, flag in scalar_options.items():
        value = options.get(key)
        if value in (None, "") or (key in defaults and value == defaults[key]):
            continue
        command += [flag, str(value)]

    for key, flag in (("filename", "--filename"), ("bandpass", "--bandpass")):
        raw = options.get(key)
        if not raw:
            continue
        for value in re.split(r"[\s,]+", str(raw).strip()):
            if value:
                command += [flag, value]

    for key, flag in (
        ("interactive", "--interactive"),
        ("update_db", "--update_db"),
        ("overwrite", "--overwrite"),
        ("debug", "--debug"),
        ("lc_only", "--lc-only"),
        ("ttv", "--ttv"),
    ):
        if options.get(key) is True:
            command.append(flag)
    command += ["-dir", str(options["workspace"])]
    return command


def build_job_command(
    target: dict[str, Any], kind: str, options: dict[str, Any], workspace: Path
) -> list[str]:
    if kind not in JOB_KINDS:
        raise ValueError("Unknown job type")
    if kind == "prepare":
        prepared = dict(options)
        prepared["workspace"] = workspace
        return build_prepare_command(target, prepared)

    data_dir = Path(target["data_dir"])
    command = [sys.executable, "-m", "allesfitter", kind, str(data_dir)]
    if kind == "optimize":
        command += ["--method", str(options.get("method", "cmaes"))]
        if options.get("no_refine"):
            command.append("--no-refine")
        command += ["--restarts", str(options.get("restarts", 1))]
        command += ["--seed", str(options.get("seed", 42))]
    elif kind == "mcmc-fit" and options.get("append"):
        command.append("--append")
    elif kind in {"mcmc-output", "ns-fit", "ns-output"} and options.get("overwrite"):
        command.append("--overwrite")
    if kind in {"show-initial-guess", "mcmc-output", "ns-output"}:
        extension = str(options.get("file_extension", ".pdf"))
        if extension:
            command += ["--file-extension", extension]
    return command


class JobRunner:
    def __init__(self, db: WorkbenchDB, workspace: Path):
        self.db = db
        self.workspace = Path(workspace).resolve()
        self._processes: dict[int, subprocess.Popen[bytes]] = {}
        self._lock = threading.Lock()

    def start(self, job_id: int) -> None:
        threading.Thread(target=self._run, args=(job_id,), daemon=True).start()

    def _run(self, job_id: int) -> None:
        job = self.db.get_job(job_id)
        if job["status"] != "queued":
            return
        log_path = Path(job["log_path"])
        log_path.parent.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env["ALLESFITTER_NONINTERACTIVE"] = "1"
        # Keep a workbench's results inside its workspace (~/ql/allesfitter by
        # default): with the workspace as the shared output root,
        # results_directory() resolves to <workspace>/<target>/<sampler>_results,
        # i.e. next to each target's data_dir where the GUI discovers artifacts.
        # Setting it explicitly also shields GUI jobs from an ambient value in
        # the user's shell. Manual CLI/script runs leave it unset and write
        # their results into the data directory itself.
        env[OUTPUT_BASE_ENV] = str(self.workspace)
        try:
            with log_path.open("ab", buffering=0) as output:
                with self._lock:
                    process = subprocess.Popen(
                        job["command"],
                        cwd=job["cwd"],
                        stdout=output,
                        stderr=subprocess.STDOUT,
                        stdin=subprocess.DEVNULL,
                        env=env,
                        start_new_session=True,
                    )
                    self._processes[job_id] = process
                self.db.update_job(job_id, status="running", pid=process.pid, started_at=_now())
                returncode = process.wait()
            current_status = self.db.get_job(job_id)["status"]
            status = (
                "cancelled"
                if current_status == "cancelled"
                else ("completed" if returncode == 0 else "failed")
            )
            self.db.update_job(job_id, status=status, returncode=returncode, finished_at=_now())
        except Exception as exc:  # keep launch failures visible in the job history
            log_path.write_text(f"Job could not start: {exc}\n", encoding="utf-8")
            self.db.update_job(job_id, status="failed", returncode=-1, finished_at=_now())
        finally:
            with self._lock:
                self._processes.pop(job_id, None)

    def cancel(self, job_id: int) -> None:
        job = self.db.get_job(job_id)
        if job["status"] not in {"queued", "running"}:
            raise ValueError("Only queued or running jobs can be cancelled")
        with self._lock:
            process = self._processes.get(job_id)
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
        self.db.update_job(job_id, status="cancelled", finished_at=_now())


class Workbench:
    def __init__(self, workspace: Path):
        self.workspace = Path(workspace).resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.db = WorkbenchDB(self.workspace / "workbench.sqlite3")
        self.runner = JobRunner(self.db, self.workspace)

    def create_target(self, payload: dict[str, Any]) -> dict[str, Any]:
        identifier_type = str(payload.get("identifier_type", "")).strip().lower()
        identifier_value = _strip_catalog_prefix(
            identifier_type, str(payload.get("identifier_value", ""))
        )
        canonical = canonical_target_name(identifier_type, identifier_value, identifier_value)
        name = str(payload.get("name") or canonical).strip()
        return self.db.create_target(
            name, identifier_type, identifier_value, self.workspace / _slug(canonical)
        )

    def create_job(self, payload: dict[str, Any]) -> dict[str, Any]:
        target = self.db.get_target(int(payload["target_id"]))
        kind = str(payload.get("kind", "prepare"))
        options = payload.get("options") or {}
        command = build_job_command(target, kind, options, self.workspace)
        job_dir = self.workspace / ".jobs"
        job_dir.mkdir(exist_ok=True)
        placeholder = self.db.create_job(
            target["id"], kind, command, options, self.workspace, job_dir / "pending.log"
        )
        log_path = job_dir / f"job-{placeholder['id']}.log"
        with self.db.connect() as con:
            con.execute("UPDATE jobs SET log_path=? WHERE id=?", (str(log_path), placeholder["id"]))
        self.runner.start(placeholder["id"])
        return self.db.get_job(placeholder["id"])

    def delete_target(self, target_id: int) -> None:
        target = self.db.get_target(target_id)
        data_dir = Path(target["data_dir"])
        if data_dir.exists():
            import shutil

            shutil.rmtree(data_dir)
        self.db.delete_target(target_id)

    def state(self) -> dict[str, Any]:
        return {"targets": self.db.list_targets(), "jobs": self.db.list_jobs()}

    def target_path(self, target_id: int, relative: str) -> Path:
        base = Path(self.db.get_target(target_id)["data_dir"]).resolve()
        candidate = (base / unquote(relative)).resolve()
        if candidate != base and base not in candidate.parents:
            raise ValueError("Path leaves the target directory")
        return candidate

    def list_artifacts(self, target_id: int) -> list[dict[str, Any]]:
        base = Path(self.db.get_target(target_id)["data_dir"])
        if not base.exists():
            return []
        allowed = {".csv", ".json", ".log", ".txt", ".png", ".jpg", ".jpeg", ".svg", ".pdf", ".h5"}
        items = []
        for path in base.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in allowed:
                continue
            stat = path.stat()
            items.append(
                {
                    "path": str(path.relative_to(base)),
                    "size": stat.st_size,
                    "modified": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(
                        timespec="seconds"
                    ),
                    "preview": path.suffix.lower() in {".png", ".jpg", ".jpeg", ".svg"},
                }
            )
        return sorted(items, key=lambda item: item["modified"], reverse=True)[:300]

    def series(self, target_id: int) -> dict[str, list[float]]:
        base = Path(self.db.get_target(target_id)["data_dir"])
        candidates = [
            path
            for path in base.glob("*.csv")
            if path.name not in EDITABLE_FILES and "table" not in path.name
        ]
        for path in candidates:
            try:
                with path.open(newline="", encoding="utf-8") as handle:
                    lines = []
                    for line in handle:
                        stripped = line.strip()
                        if not stripped:
                            continue
                        if stripped.startswith("#"):
                            header = stripped.lstrip("# ")
                            if header.lower().startswith("time,"):
                                lines.append(header)
                            continue
                        lines.append(line)
                rows = list(csv.DictReader(lines))
                if not rows or not {"time", "flux"} <= set(rows[0]):
                    continue
                step = max(1, len(rows) // 1000)
                sampled = rows[::step]
                return {
                    "time": [float(row["time"]) for row in sampled],
                    "flux": [float(row["flux"]) for row in sampled],
                }
            except (OSError, ValueError, TypeError):
                continue
        return {"time": [], "flux": []}


class WorkbenchHandler(BaseHTTPRequestHandler):
    server: WorkbenchHTTPServer

    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stderr.write(f"[web] {self.address_string()} {fmt % args}\n")

    def _json(self, payload: Any, status: int = 200) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _error(self, exc: Exception, status: int = 400) -> None:
        self._json({"error": str(exc)}, status)

    def _body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length > 2_000_000:
            raise ValueError("Request body is too large")
        return json.loads(self.rfile.read(length) or b"{}")

    def _asset(self, path: Path) -> None:
        if not path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        body = path.read_bytes()
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        parsed = urlparse(self.path)
        route = parsed.path
        try:
            if route == "/":
                self._asset(ASSET_DIR / "index.html")
            elif route.startswith("/assets/"):
                name = Path(route.removeprefix("/assets/")).name
                self._asset(ASSET_DIR / name)
            elif route == "/api/state":
                self._json(self.server.app.state())
            elif route == "/api/artifacts":
                target_id = int(parse_qs(parsed.query)["target_id"][0])
                self._json(self.server.app.list_artifacts(target_id))
            elif route == "/api/series":
                target_id = int(parse_qs(parsed.query)["target_id"][0])
                self._json(self.server.app.series(target_id))
            elif route == "/api/file":
                query = parse_qs(parsed.query)
                target_id = int(query["target_id"][0])
                name = query["name"][0]
                if name not in EDITABLE_FILES:
                    raise ValueError("This file is not editable")
                path = self.server.app.target_path(target_id, name)
                self._json(
                    {
                        "name": name,
                        "exists": path.exists(),
                        "content": path.read_text() if path.exists() else "",
                    }
                )
            elif route == "/api/log":
                job_id = int(parse_qs(parsed.query)["job_id"][0])
                job = self.server.app.db.get_job(job_id)
                path = Path(job["log_path"])
                content = (
                    path.read_bytes()[-200_000:].decode(errors="replace") if path.exists() else ""
                )
                self._json({"content": content, "status": job["status"]})
            elif route == "/api/artifact":
                query = parse_qs(parsed.query)
                path = self.server.app.target_path(int(query["target_id"][0]), query["path"][0])
                self._asset(path)
            else:
                self.send_error(HTTPStatus.NOT_FOUND)
        except (KeyError, ValueError, OSError) as exc:
            self._error(exc, 404 if isinstance(exc, KeyError) else 400)

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        try:
            if self.path == "/api/targets":
                self._json(self.server.app.create_target(self._body()), 201)
            elif self.path == "/api/jobs":
                self._json(self.server.app.create_job(self._body()), 201)
            elif self.path.startswith("/api/jobs/") and self.path.endswith("/cancel"):
                job_id = int(self.path.split("/")[3])
                self.server.app.runner.cancel(job_id)
                self._json(self.server.app.db.get_job(job_id))
            else:
                self.send_error(HTTPStatus.NOT_FOUND)
        except (KeyError, ValueError, OSError) as exc:
            self._error(exc, 404 if isinstance(exc, KeyError) else 400)

    def do_DELETE(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        try:
            parts = self.path.split("/")
            if len(parts) == 4 and parts[1] == "api" and parts[2] == "targets":
                target_id = int(parts[3])
                self.server.app.delete_target(target_id)
                self._json({"deleted": target_id})
            else:
                self.send_error(HTTPStatus.NOT_FOUND)
        except (KeyError, ValueError, OSError) as exc:
            self._error(exc, 404 if isinstance(exc, KeyError) else 400)

    def do_PUT(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        try:
            if self.path != "/api/file":
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            payload = self._body()
            name = str(payload["name"])
            if name not in EDITABLE_FILES:
                raise ValueError("This file is not editable")
            path = self.server.app.target_path(int(payload["target_id"]), name)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(str(payload.get("content", "")), encoding="utf-8")
            self._json({"saved": True, "name": name})
        except (KeyError, ValueError, OSError) as exc:
            self._error(exc, 404 if isinstance(exc, KeyError) else 400)


class WorkbenchHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address: tuple[str, int], app: Workbench):
        self.app = app
        super().__init__(address, WorkbenchHandler)


def resolve_workspace(workspace: str | Path | None) -> Path:
    """Return the workspace directory, defaulting to the shared results root.

    The workspace holds target data, results, job logs, and the SQLite history,
    so leaving it unset collects every GUI run under ``~/ql/allesfitter``.
    """
    if not workspace:
        return DEFAULT_OUTPUT_BASE
    return Path(workspace).expanduser()


def serve(host: str = "127.0.0.1", port: int = 5100, workspace: str | Path | None = None) -> None:
    app = Workbench(resolve_workspace(workspace))
    server = WorkbenchHTTPServer((host, port), app)
    print(f"allesfitter workbench: http://{host}:{port}")
    print(f"workspace: {app.workspace}")
    if host not in {"127.0.0.1", "localhost", "::1"}:
        print("warning: this server has no authentication; prefer localhost with SSH forwarding")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
