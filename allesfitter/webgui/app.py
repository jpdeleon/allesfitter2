"""FastAPI application factory for the allesfitter web GUI.

Wires the engine-free building blocks into an HTTP surface (muscat-db's
server-rendered FastAPI + Jinja2 shape):

    GET  /                          -> targets / recent runs
    GET  /prepare                   -> new-run form (catalog-driven)
    POST /prepare/run               -> download + generate a config, launch prepare
    GET  /prepare/log/{run_id}      -> tail of the prepare log (text)
    GET  /prepare/sectors?target=... -> available sectors/campaigns/quarters (JSON)
    GET  /catalog?target=...        -> ephemeris auto-fill (JSON)
    GET  /jobs                      -> runs dashboard
    GET  /jobs/status               -> runs as JSON (client polls this)
    GET  /jobs/log/{run_id}         -> tail of the live run log (text)
    POST /jobs/fit/{run_id}         -> launch a sampler (mcmc/ns/optimize) on a prepared run
    POST /jobs/stop/{run_id}        -> terminate a running job
    POST /jobs/delete/{run_id}      -> remove a run and its directory
    GET  /results/{run_id}          -> results page (config review, figures, logZ)
    POST /results/{run_id}/config   -> save + dry-run validate an edited config
    GET  /results/{run_id}/initial-guess -> serve the pre-fit initial-guess preview
    GET  /results/{run_id}/file/{name}   -> serve a result figure/document
"""

from __future__ import annotations

import datetime as _dt
import re
import uuid
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from allesfitter.webgui import (
    catalog,
    instruments,
    jobs,
    prepare,
    results,
)
from allesfitter.webgui import runstore as rs
from allesfitter.webgui import validate as _validate
from allesfitter.webgui.job_store import RunStoreJobStore, set_job_store
from allesfitter.webgui.runstore import RunStore

_PKG_DIR = Path(__file__).resolve().parent
_TEMPLATES_DIR = _PKG_DIR / "templates"
_STATIC_DIR = _PKG_DIR / "static"

_SLUG_RE = re.compile(r"[^A-Za-z0-9._-]+")


def _slug(text: str) -> str:
    return _SLUG_RE.sub("-", text.strip()).strip("-") or "target"


def _new_run_id(target: str) -> str:
    stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{_slug(target)}_{stamp}_{uuid.uuid4().hex[:6]}"


def _normalize_root_path(root_path: str) -> str:
    """Return a canonical ASGI root path (empty, or leading slash/no trailing slash)."""
    root_path = root_path.strip()
    if not root_path or root_path == "/":
        return ""
    if not root_path.startswith("/"):
        root_path = f"/{root_path}"
    return root_path.rstrip("/")


def _tail_run_log(run_dir: str | Path, prepare_dir: str | Path, n: int = 300) -> str:
    """Return a fit log when present, otherwise the originating prepare log."""
    fit_log = jobs.tail_log(run_dir, n=n)
    return fit_log or prepare.tail_log(prepare_dir, n=n)


def create_app(
    runs_root: str | Path = "webgui_runs",
    db_path: str | Path | None = None,
    *,
    toi_csv: str | Path | None = None,
    allow_network: bool = True,
    root_path: str = "",
    max_concurrent_fits: int = 1,
    max_queued_fits_per_user: int = 2,
) -> FastAPI:
    """Build the app, optionally mounted externally below ``root_path``."""
    # ``prepare_allesfit`` runs with its per-run directory as cwd.  Keeping the
    # output root relative would prepend that cwd a second time (for example,
    # ``run/webgui_runs/run/target``), beyond the directory discovery boundary.
    runs_root = Path(runs_root).resolve()
    runs_root.mkdir(parents=True, exist_ok=True)
    store = RunStore(str(db_path) if db_path else str(runs_root / "webgui.sqlite3"))
    job_store = RunStoreJobStore(store)
    set_job_store(job_store)
    scheduler = jobs.LocalJobScheduler(
        job_store,
        max_concurrent=max_concurrent_fits,
        max_queued_per_user=max_queued_fits_per_user,
        autostart=False,
    )
    jobs.set_scheduler(scheduler)

    templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))
    root_path = _normalize_root_path(root_path)
    app = FastAPI(title="allesfitter web GUI", root_path=root_path)

    if root_path:

        @app.middleware("http")
        async def keep_redirects_below_root_path(request: Request, call_next):
            response = await call_next(request)
            location = response.headers.get("location")
            if not location:
                return response
            parsed = urlsplit(location)
            same_origin = not parsed.netloc or parsed.netloc == request.url.netloc
            below_prefix = parsed.path == root_path or parsed.path.startswith(root_path + "/")
            if same_origin and parsed.path.startswith("/") and not below_prefix:
                path = root_path + parsed.path
                response.headers["location"] = urlunsplit(
                    (parsed.scheme, parsed.netloc, path, parsed.query, parsed.fragment)
                )
            return response

    if _STATIC_DIR.is_dir():
        app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

    app.state.store = store
    app.state.job_store = job_store
    app.state.runs_root = runs_root
    app.state.toi_csv = str(toi_csv) if toi_csv else None
    app.state.allow_network = allow_network
    app.state.root_path = root_path
    app.state.scheduler = scheduler

    @app.on_event("startup")
    def startup_scheduler() -> None:
        scheduler.start()

    @app.on_event("shutdown")
    def shutdown_scheduler() -> None:
        # Jobs deliberately survive a graceful GUI restart and are reconciled
        # by the next scheduler instance.
        scheduler.close(stop_jobs=False)

    def _run_dir(run_id: str) -> Path:
        row = store.get_run(run_id)
        if row is None:
            raise HTTPException(status_code=404, detail=f"unknown run {run_id!r}")
        return Path(row["run_dir"])

    def _owner(request: Request) -> str:
        """Use the trusted-auth middleware identity when installed (Phase 2/3 seam)."""
        return str(getattr(request.state, "user", ""))

    # -- pages --------------------------------------------------------------
    @app.get("/", response_class=HTMLResponse)
    def index(request: Request):
        all_runs = store.list_runs()
        run_metrics = {"active": 0, "review": 0, "complete": 0, "failed": 0}
        for run in all_runs:
            if run["state"] in (rs.CREATED, rs.PENDING, rs.PREPARING, rs.RUNNING, rs.STOPPING):
                run_metrics["active"] += 1
            elif run["state"] == rs.PREPARED:
                run_metrics["review"] += 1
            elif run["state"] == rs.DONE:
                run_metrics["complete"] += 1
            elif run["state"] in (rs.FAILED, rs.STOPPED):
                run_metrics["failed"] += 1
        return templates.TemplateResponse(
            request,
            "index.html",
            {"targets": store.targets(), "runs": all_runs[:20], "run_metrics": run_metrics},
        )

    @app.get("/prepare", response_class=HTMLResponse)
    def prepare_form(request: Request):
        return templates.TemplateResponse(
            request,
            "prepare.html",
            {
                "missions": instruments.MISSIONS,
                "pipelines": instruments.PIPELINES,
                "lc_types": instruments.LC_TYPES,
                "quality_flags": instruments.QUALITY_FLAGS,
                "allow_network": app.state.allow_network,
            },
        )

    @app.get("/jobs", response_class=HTMLResponse)
    def jobs_page(request: Request):
        return templates.TemplateResponse(request, "jobs.html", {})

    @app.get("/results/{run_id}", response_class=HTMLResponse)
    def results_page(request: Request, run_id: str):
        row = store.get_run(run_id)
        if row is None:
            raise HTTPException(status_code=404, detail="unknown run")
        images = [p.name for p in results.result_images(row["run_dir"])]
        documents = [p.name for p in results.result_documents(row["run_dir"])]
        run_log = _tail_run_log(row["run_dir"], runs_root / run_id)
        run_dir = Path(row["run_dir"])
        editable = row["state"] == rs.PREPARED
        return templates.TemplateResponse(
            request,
            "results.html",
            {
                "run": row,
                "images": images,
                "documents": documents,
                "run_log": run_log,
                "editable": editable,
                "params_csv": (run_dir / "params.csv").read_text() if editable else "",
                "settings_csv": (run_dir / "settings.csv").read_text() if editable else "",
                "has_initial_guess": (run_dir / "initial_guess.png").is_file(),
            },
        )

    # -- JSON / actions -----------------------------------------------------
    @app.get("/catalog")
    def catalog_lookup(target: str):
        eph = catalog.lookup(
            target, toi_csv=app.state.toi_csv, allow_network=app.state.allow_network
        )
        return JSONResponse(eph.__dict__)

    @app.get("/prepare/sectors")
    def prepare_sectors(
        target: str, id_type: str = "name", mission: str = "tess", pipeline: str = ""
    ):
        result = catalog.available_sectors(
            target,
            id_type=id_type,
            mission=mission,
            pipeline=pipeline,
            allow_network=app.state.allow_network,
        )
        return JSONResponse(
            {
                "ok": result.ok,
                "segments": result.segments,
                "pipelines": result.pipelines,
                "exptimes": result.exptimes,
                "error": result.error,
            }
        )

    @app.post("/prepare/run")
    async def prepare_run(request: Request):
        if not app.state.allow_network:
            raise HTTPException(
                status_code=400,
                detail="network is disabled (--no-network); prepare needs to download data",
            )
        payload = await request.json()
        try:
            argv = prepare.build_prepare_argv(payload)
        except (ValueError, KeyError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        target = str(payload.get("target") or "").strip()
        sampler = payload.get("sampler", "mcmc")
        run_id = _new_run_id(target)
        work_dir = runs_root / run_id
        argv += ["-dir", str(work_dir)]

        store.create_run(
            run_id=run_id,
            target=target,
            run_dir=str(work_dir),
            sampler=sampler,
            owner=_owner(request),
            state=rs.PREPARING,
        )
        prepare.launch_prepare(store, run_id, work_dir, argv)
        return JSONResponse({"run_id": run_id})

    @app.get("/prepare/log/{run_id}", response_class=PlainTextResponse)
    def prepare_log(run_id: str, n: int = 300):
        if store.get_run(run_id) is None:
            raise HTTPException(status_code=404, detail=f"unknown run {run_id!r}")
        return prepare.tail_log(runs_root / run_id, n=n)

    @app.post("/jobs/fit/{run_id}")
    def jobs_fit(
        request: Request,
        run_id: str,
        sampler: str = "mcmc",
        method: str = "cmaes",
        refine: bool = True,
        restarts: int = 1,
    ):
        row = store.get_run(run_id)
        if row is None:
            raise HTTPException(status_code=404, detail=f"unknown run {run_id!r}")
        if row["state"] != rs.PREPARED:
            raise HTTPException(status_code=409, detail="run must be prepared before fitting")
        run_dir = Path(row["run_dir"])

        from allesfitter.webgui.config_writer import RUN_PY_TEMPLATE

        try:
            (run_dir / "run.py").write_text(RUN_PY_TEMPLATE)
            try:
                (run_dir / "run.py").chmod(0o755)
            except OSError:
                pass
        except Exception:
            pass

        extra_env = {}
        if sampler == "optimize":
            extra_env = {
                "ALLESFIT_OPTIMIZE_METHOD": method,
                "ALLESFIT_OPTIMIZE_REFINE": "true" if refine else "false",
                "ALLESFIT_OPTIMIZE_RESTARTS": str(restarts),
            }

        store.update_run(run_id, sampler=sampler)
        try:
            jobs.launch(job_store, run_id, run_dir, sampler=sampler, extra_env=extra_env)
        except jobs.QueueLimitError as exc:
            raise HTTPException(status_code=429, detail=str(exc)) from exc
        except (FileNotFoundError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return JSONResponse({"run_id": run_id, "sampler": sampler})

    @app.post("/results/{run_id}/config")
    async def update_run_config(request: Request, run_id: str):
        row = store.get_run(run_id)
        if row is None:
            raise HTTPException(status_code=404, detail="unknown run")
        if row["state"] != rs.PREPARED:
            raise HTTPException(status_code=409, detail="only prepared runs can be edited")
        payload = await request.json()
        params_csv = payload.get("params_csv")
        settings_csv = payload.get("settings_csv")
        if not isinstance(params_csv, str) or not isinstance(settings_csv, str):
            raise HTTPException(status_code=400, detail="both CSV documents are required")
        if max(len(params_csv), len(settings_csv)) > 2_000_000 or "\x00" in (
            params_csv + settings_csv
        ):
            raise HTTPException(status_code=400, detail="invalid or oversized CSV document")

        run_dir = Path(row["run_dir"])
        (run_dir / "params.csv").write_text(params_csv)
        (run_dir / "settings.csv").write_text(settings_csv)
        validation = _validate.dry_run(run_dir)
        if not validation.ok:
            return JSONResponse({"ok": False, "error": validation.error}, status_code=400)
        preview = _validate.initial_guess_preview(run_dir)
        if preview is None:
            return JSONResponse(
                {"ok": False, "error": "show_initial_guess failed to produce a preview"},
                status_code=400,
            )
        store.set_state(run_id, rs.PREPARED, error="")
        return JSONResponse({"ok": True, "n_free_params": validation.n_free_params})

    @app.get("/results/{run_id}/initial-guess")
    def initial_guess_file(run_id: str):
        path = _run_dir(run_id) / "initial_guess.png"
        if not path.is_file():
            raise HTTPException(status_code=404, detail="initial guess not found")
        return FileResponse(path)

    @app.get("/jobs/status")
    def jobs_status(target: str | None = None):
        return JSONResponse({"runs": store.list_runs(target)})

    @app.get("/jobs/log/{run_id}", response_class=PlainTextResponse)
    def jobs_log(run_id: str, n: int = 300):
        return _tail_run_log(_run_dir(run_id), runs_root / run_id, n=n)

    @app.post("/jobs/stop/{run_id}")
    def jobs_stop(run_id: str):
        _run_dir(run_id)  # 404 if unknown
        stopped = jobs.stop(run_id) or prepare.stop(run_id)
        return JSONResponse({"stopped": stopped})

    @app.post("/jobs/delete/{run_id}")
    def jobs_delete(run_id: str):
        run_dir = _run_dir(run_id).resolve()  # 404 if unknown
        try:
            jobs.stop(run_id)
            prepare.stop(run_id)
        except Exception:
            pass

        try:
            run_dir.relative_to(runs_root)
        except ValueError:
            raise HTTPException(
                status_code=400, detail="Forbidden: run directory is outside runs root"
            ) from None

        import shutil

        if run_dir.exists():
            try:
                shutil.rmtree(run_dir)
            except Exception as exc:
                raise HTTPException(
                    status_code=500, detail=f"failed to delete run directory: {exc}"
                ) from exc

        runs_root_dir = runs_root / run_id
        if runs_root_dir.exists() and runs_root_dir != run_dir:
            try:
                shutil.rmtree(runs_root_dir)
            except Exception:
                pass

        store.delete_run(run_id)
        return JSONResponse({"deleted": True})

    @app.get("/results/{run_id}/file/{name}")
    def result_file(run_id: str, name: str):
        path = results.find_result_file(_run_dir(run_id), name)
        if path is None:
            raise HTTPException(status_code=404, detail="result file not found")
        return FileResponse(path)

    return app
