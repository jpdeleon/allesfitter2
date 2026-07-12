"""Smoke tests for the FastAPI routes via TestClient."""

from __future__ import annotations

from html.parser import HTMLParser
from urllib.parse import urlsplit

import numpy as np
from fastapi.testclient import TestClient

from allesfitter.webgui import jobs as _jobs
from allesfitter.webgui import runstore as rs
from allesfitter.webgui.app import create_app


def _client(tmp_path, **kw):
    app = create_app(tmp_path / "runs", tmp_path / "db.sqlite3", allow_network=False, **kw)
    return TestClient(app)


class _Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.urls = []

    def handle_starttag(self, tag, attrs):
        values = dict(attrs)
        for attr in ("href", "src", "action"):
            if attr in values:
                self.urls.append(values[attr])


def _page_urls(html):
    parser = _Links()
    parser.feed(html)
    return parser.urls


def _lc(path, seed=0):
    rng = np.random.default_rng(seed)
    t = np.linspace(2456999.0, 2457007.0, 200)
    rows = ["#time,flux,flux_err"]
    for ti, fi in zip(t, 1.0 + rng.normal(0, 1e-3, t.size)):
        rows.append(f"{ti},{fi},0.001")
    path.write_text("\n".join(rows) + "\n")


def _valid_payload(data_file):
    return {
        "target": "TOI-test",
        "companions": [{"name": "b", "period": 2.0, "epoch": 2457000.0, "rr": 0.05}],
        "instruments": [
            {
                "label": "lco_g",
                "band": "g",
                "data_file": str(data_file),
                "baseline": "sample_linear",
            }
        ],
        "use_host_density_prior": False,
        "fast_fit": False,
        "sampler": "mcmc",
    }


def test_pages_render(tmp_path):
    c = _client(tmp_path)
    for url in ("/", "/jobs"):
        resp = c.get(url)
        assert resp.status_code == 200
        assert "allesfitter" in resp.text.lower()


def test_prefixed_pages_assets_and_navigation_stay_below_prefix(tmp_path):
    prefix = "/allesfitter"
    c = _client(tmp_path, root_path=prefix)

    # TestClient addresses the backend after the reverse proxy has stripped the
    # public prefix, while generated browser URLs retain it.
    pending = ["/"]
    visited = set()
    while pending:
        backend_path = pending.pop()
        if backend_path in visited:
            continue
        visited.add(backend_path)
        response = c.get(backend_path)
        assert response.status_code == 200
        for url in _page_urls(response.text):
            public_path = urlsplit(url).path
            if not public_path:
                # In-page fragment anchors (e.g. "#main-content" skip link) have
                # no path and are inherently prefix-agnostic.
                continue
            assert public_path.startswith(prefix + "/"), url
            if public_path.endswith((".css", ".js")):
                assert c.get(public_path).status_code == 200
            elif public_path.removeprefix(prefix) in ("/", "/prepare", "/jobs"):
                pending.append(public_path.removeprefix(prefix))

    assert {"/", "/prepare", "/jobs"} <= visited


def test_prefixed_client_actions_and_redirects_use_shared_helper(tmp_path):
    c = _client(tmp_path, root_path="/allesfitter/")
    app_js = c.get("/allesfitter/static/app.js").text
    assert "window.AF_ROOT_PATH" in app_js
    for script in ("prepare.js", "jobs.js"):
        text = c.get(f"/allesfitter/static/{script}").text
        assert "window.AF.url(" in text
        assert 'fetch("/' not in text
        assert 'postJSON("/' not in text
        assert 'location.href = "/' not in text

    redirect = c.get("/jobs/", follow_redirects=False)
    assert redirect.status_code in (307, 308)
    assert redirect.headers["location"] == "http://testserver/allesfitter/jobs"


def test_prefixed_result_links_and_files_stay_below_prefix(tmp_path):
    c = _client(tmp_path, root_path="/allesfitter")
    run_id = "test-run"
    run_dir = tmp_path / "runs" / run_id
    result_dir = run_dir / "results"
    result_dir.mkdir(parents=True)
    (result_dir / "figure.png").write_bytes(b"not-a-real-png")
    c.app.state.store.create_run(
        run_id=run_id, target="Target", run_dir=str(run_dir), sampler="mcmc"
    )

    response = c.get(f"/results/{run_id}")
    assert response.status_code == 200
    result_urls = [urlsplit(url).path for url in _page_urls(response.text) if "/file/" in url]
    assert result_urls
    assert all(url.startswith("/allesfitter/results/") for url in result_urls)
    assert c.get(result_urls[0]).status_code == 200


def test_catalog_autofill_from_local_csv(tmp_path):
    toi = tmp_path / "TOIs.csv"
    toi.write_text("TOI,Period (days),Epoch (BJD)\n6715.01,2.862577,2460621.83\n")
    c = _client(tmp_path, toi_csv=str(toi))
    resp = c.get("/catalog", params={"target": "TOI-6715"})
    body = resp.json()
    assert body["source"] == "toi_csv"
    assert body["period"] == 2.862577


def test_prepare_sectors_disabled_without_network(tmp_path):
    c = _client(tmp_path)  # allow_network=False by default
    resp = c.get("/prepare/sectors", params={"target": "TOI-6715"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is False
    assert "network" in body["error"]
    assert body["segments"] == []


def test_prepare_sectors_wires_query_params_into_catalog_lookup(tmp_path, monkeypatch):
    from allesfitter.webgui import app as app_module
    from allesfitter.webgui import catalog

    captured = {}

    def _fake_available_sectors(target, *, id_type, mission, pipeline, allow_network):
        captured.update(
            target=target,
            id_type=id_type,
            mission=mission,
            pipeline=pipeline,
            allow_network=allow_network,
        )
        return catalog.SectorAvailability(
            target=target, mission=mission, segments=["2", "9"], pipelines=["qlp", "spoc"]
        )

    monkeypatch.setattr(app_module.catalog, "available_sectors", _fake_available_sectors)

    app = create_app(tmp_path / "runs", tmp_path / "db.sqlite3", allow_network=True)
    c = TestClient(app)
    resp = c.get(
        "/prepare/sectors",
        params={"target": "TOI-6715", "id_type": "toi", "mission": "tess", "pipeline": "qlp"},
    )
    assert resp.status_code == 200
    assert resp.json() == {
        "ok": True,
        "segments": ["2", "9"],
        "pipelines": ["qlp", "spoc"],
        "error": "",
    }
    assert captured == {
        "target": "TOI-6715",
        "id_type": "toi",
        "mission": "tess",
        "pipeline": "qlp",
        "allow_network": True,
    }


def test_jobs_status_empty(tmp_path):
    c = _client(tmp_path)
    assert c.get("/jobs/status").json() == {"runs": []}


def test_jobs_fit_optimize_wires_method_refine_restarts_into_env(tmp_path, monkeypatch):
    c = _client(tmp_path)
    run_id = "test-optimize-run"
    run_dir = tmp_path / "runs" / run_id
    run_dir.mkdir(parents=True)
    c.app.state.store.create_run(
        run_id=run_id, target="Target", run_dir=str(run_dir), sampler="mcmc", state=rs.PREPARED
    )

    captured = {}

    def _fake_launch(job_store, run_id, run_dir, sampler="mcmc", *, python=None, extra_env=None):
        captured["sampler"] = sampler
        captured["extra_env"] = extra_env

        class _Proc:
            pid = 1234

        return _Proc()

    monkeypatch.setattr(_jobs, "launch", _fake_launch)

    resp = c.post(
        f"/jobs/fit/{run_id}",
        params={
            "sampler": "optimize",
            "method": "dual_annealing",
            "refine": "false",
            "restarts": "3",
        },
    )
    assert resp.status_code == 200
    assert (run_dir / "run.py").is_file()
    assert captured["sampler"] == "optimize"
    assert captured["extra_env"] == {
        "ALLESFIT_OPTIMIZE_METHOD": "dual_annealing",
        "ALLESFIT_OPTIMIZE_REFINE": "false",
        "ALLESFIT_OPTIMIZE_RESTARTS": "3",
    }


def test_jobs_delete(tmp_path):
    c = _client(tmp_path)
    run_id = "test-delete-run"
    run_dir = tmp_path / "runs" / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "dummy.txt").write_text("hello")
    c.app.state.store.create_run(
        run_id=run_id, target="Target", run_dir=str(run_dir), sampler="mcmc"
    )

    resp = c.post(f"/jobs/delete/{run_id}")
    assert resp.status_code == 200
    assert resp.json() == {"deleted": True}

    runs = c.get("/jobs/status").json()["runs"]
    assert not any(r["run_id"] == run_id for r in runs)
    assert not run_dir.exists()
