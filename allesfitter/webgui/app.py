"""FastAPI application factory for the allesfitter web GUI.

Wires the engine-free building blocks into an HTTP surface (muscat-db's
server-rendered FastAPI + Jinja2 shape):

    GET  /                     -> targets / recent runs
    GET  /fit                  -> new-fit form
    POST /fit/validate         -> dry-run a config (Basement, no sampler)
    POST /fit/run              -> write + stage + validate + launch a run
    GET  /catalog?target=...   -> ephemeris auto-fill (JSON)
    GET  /jobs                 -> runs dashboard
    GET  /jobs/status          -> runs as JSON (client polls this)
    GET  /jobs/log/{run_id}    -> tail of the live run log (text)
    POST /jobs/stop/{run_id}   -> terminate a running job
    GET  /results/{run_id}     -> results page (figures + logZ)
    GET  /results/{run_id}/image/{name} -> serve a result figure
"""

from __future__ import annotations

import datetime as _dt
import re
import tempfile
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from allesfitter.webgui import (
    catalog,
    config_writer,
    formparse,
    instruments,
    jobs,
    prepare,
    results,
    staging,
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


def create_app(
    runs_root: str | Path = "webgui_runs",
    db_path: str | Path | None = None,
    *,
    toi_csv: str | Path | None = None,
    allow_network: bool = True,
) -> FastAPI:
    """Build the FastAPI app. ``runs_root`` holds per-run datadirs."""
    runs_root = Path(runs_root)
    runs_root.mkdir(parents=True, exist_ok=True)
    store = RunStore(str(db_path) if db_path else str(runs_root / "webgui.sqlite3"))
    job_store = RunStoreJobStore(store)
    set_job_store(job_store)

    templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))
    app = FastAPI(title="allesfitter web GUI")
    if _STATIC_DIR.is_dir():
        app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

    app.state.store = store
    app.state.job_store = job_store
    app.state.runs_root = runs_root
    app.state.toi_csv = str(toi_csv) if toi_csv else None
    app.state.allow_network = allow_network

    def _run_dir(run_id: str) -> Path:
        row = store.get_run(run_id)
        if row is None:
            raise HTTPException(status_code=404, detail=f"unknown run {run_id!r}")
        return Path(row["run_dir"])

    # -- pages --------------------------------------------------------------
    @app.get("/", response_class=HTMLResponse)
    def index(request: Request):
        return templates.TemplateResponse(
            request,
            "index.html",
            {"targets": store.targets(), "runs": store.list_runs()[:20]},
        )

    @app.get("/fit", response_class=HTMLResponse)
    def fit_form(request: Request):
        return templates.TemplateResponse(
            request,
            "fit.html",
            {
                "baseline_models": instruments.BASELINE_MODELS,
                "ld_laws": instruments.LD_LAWS,
                "families": instruments.FAMILIES,
            },
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
        return templates.TemplateResponse(
            request, "results.html", {"run": row, "images": images, "documents": documents}
        )

    # -- JSON / actions -----------------------------------------------------
    @app.get("/catalog")
    def catalog_lookup(target: str):
        eph = catalog.lookup(
            target, toi_csv=app.state.toi_csv, allow_network=app.state.allow_network
        )
        return JSONResponse(eph.__dict__)

    @app.post("/fit/validate")
    async def fit_validate(request: Request):
        payload = await request.json()
        try:
            cfg, staging_items = formparse.build_fit_config(payload)
        except (ValueError, KeyError) as exc:
            return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)
        with tempfile.TemporaryDirectory() as tmp:
            config_writer.write_config(cfg, tmp)
            try:
                staging.stage_all(staging_items, tmp)
            except (FileNotFoundError, ValueError) as exc:
                return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)
            res = _validate.dry_run(tmp)
        return JSONResponse({"ok": res.ok, "error": res.error, "n_free_params": res.n_free_params})

    @app.post("/fit/run")
    async def fit_run(request: Request):
        payload = await request.json()
        sampler = payload.get("sampler", "mcmc")
        try:
            cfg, staging_items = formparse.build_fit_config(payload)
        except (ValueError, KeyError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        run_id = _new_run_id(cfg.target)
        run_dir = runs_root / run_id
        config_writer.write_config(cfg, run_dir)
        try:
            staging.stage_all(staging_items, run_dir)
        except (FileNotFoundError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        res = _validate.dry_run(run_dir)
        if not res.ok:
            raise HTTPException(status_code=400, detail=f"validation failed: {res.error}")

        store.create_run(
            run_id=run_id,
            target=cfg.target,
            run_dir=str(run_dir),
            sampler=sampler,
            insts=" ".join(i.label for i in cfg.instruments),
            bands=" ".join(cfg.unique_bands()),
            companions=" ".join(c.name for c in cfg.companions),
        )
        jobs.launch(job_store, run_id, run_dir, sampler=sampler)
        return JSONResponse({"run_id": run_id})

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
    def jobs_fit(run_id: str, sampler: str = "mcmc"):
        row = store.get_run(run_id)
        if row is None:
            raise HTTPException(status_code=404, detail=f"unknown run {run_id!r}")
        run_dir = Path(row["run_dir"])
        store.update_run(run_id, sampler=sampler)
        try:
            jobs.launch(job_store, run_id, run_dir, sampler=sampler)
        except (FileNotFoundError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return JSONResponse({"run_id": run_id, "sampler": sampler})

    @app.get("/jobs/status")
    def jobs_status(target: str | None = None):
        return JSONResponse({"runs": store.list_runs(target)})

    @app.get("/jobs/log/{run_id}", response_class=PlainTextResponse)
    def jobs_log(run_id: str, n: int = 300):
        return jobs.tail_log(_run_dir(run_id), n=n)

    @app.post("/jobs/stop/{run_id}")
    def jobs_stop(run_id: str):
        _run_dir(run_id)  # 404 if unknown
        stopped = jobs.stop(run_id) or prepare.stop(run_id)
        return JSONResponse({"stopped": stopped})

    @app.get("/results/{run_id}/file/{name}")
    def result_file(run_id: str, name: str):
        path = results.find_result_file(_run_dir(run_id), name)
        if path is None:
            raise HTTPException(status_code=404, detail="result file not found")
        return FileResponse(path)

    return app
