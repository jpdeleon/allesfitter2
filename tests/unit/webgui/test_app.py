"""Smoke tests for the FastAPI routes via TestClient."""

from __future__ import annotations

import numpy as np
from fastapi.testclient import TestClient

from allesfitter.webgui import jobs
from allesfitter.webgui.app import create_app


def _client(tmp_path, **kw):
    app = create_app(tmp_path / "runs", tmp_path / "db.sqlite3", allow_network=False, **kw)
    return TestClient(app)


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
    for url in ("/", "/fit", "/jobs"):
        resp = c.get(url)
        assert resp.status_code == 200
        assert "allesfitter" in resp.text.lower()


def test_validate_rejects_missing_target(tmp_path):
    c = _client(tmp_path)
    resp = c.post("/fit/validate", json={"instruments": [{"label": "x", "band": "g"}]})
    assert resp.status_code == 400
    assert "target" in resp.json()["error"]


def test_validate_rejects_no_instruments(tmp_path):
    c = _client(tmp_path)
    resp = c.post("/fit/validate", json={"target": "T", "instruments": []})
    assert resp.status_code == 400


def test_catalog_autofill_from_local_csv(tmp_path):
    toi = tmp_path / "TOIs.csv"
    toi.write_text("TOI,Period (days),Epoch (BJD)\n6715.01,2.862577,2460621.83\n")
    c = _client(tmp_path, toi_csv=str(toi))
    resp = c.get("/catalog", params={"target": "TOI-6715"})
    body = resp.json()
    assert body["source"] == "toi_csv"
    assert body["period"] == 2.862577


def test_jobs_status_empty(tmp_path):
    c = _client(tmp_path)
    assert c.get("/jobs/status").json() == {"runs": []}


def test_validate_ok_with_engine(tmp_path, basement):
    lc = tmp_path / "m4g.csv"
    _lc(lc)
    c = _client(tmp_path)
    resp = c.post("/fit/validate", json=_valid_payload(lc))
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True, body["error"]
    assert body["n_free_params"] > 0


def test_run_creates_run_and_launches(tmp_path, basement, monkeypatch):
    launched = []
    monkeypatch.setattr(jobs, "launch", lambda *a, **k: launched.append(a))
    lc = tmp_path / "m4g.csv"
    _lc(lc)
    c = _client(tmp_path)
    resp = c.post("/fit/run", json=_valid_payload(lc))
    assert resp.status_code == 200, resp.text
    run_id = resp.json()["run_id"]

    runs = c.get("/jobs/status").json()["runs"]
    assert any(r["run_id"] == run_id for r in runs)
    assert launched  # launcher was invoked
    # the run dir is a complete datadir
    assert (tmp_path / "runs" / run_id / "settings.csv").is_file()
    assert (tmp_path / "runs" / run_id / "run.py").is_file()
