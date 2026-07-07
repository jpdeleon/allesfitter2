"""Disk-cache tests for simulate_PDF.calculate_skewed_normal_params.

The underlying solver is slow (~10 s) so each test points the cache to a
tmp_path JSON file via the ALLESFITTER_SIMULATE_PDF_CACHE env var. The
solver is invoked once per test fixture and the cache file is asserted
to be populated; subsequent calls with identical inputs return instantly
without hitting the solver again.
"""

from __future__ import annotations

import json
import time

import pytest

try:
    from allesfitter.priors import simulate_PDF as spdf
except Exception:
    pytest.skip("allesfitter not importable", allow_module_level=True)


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def cache_file(tmp_path, monkeypatch):
    """Redirect the cache to a clean per-test JSON path."""
    p = tmp_path / "spdf_cache.json"
    monkeypatch.setenv("ALLESFITTER_SIMULATE_PDF_CACHE", str(p))
    monkeypatch.delenv("ALLESFITTER_SIMULATE_PDF_NO_CACHE", raising=False)
    return p


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------


def test_first_call_solves_and_writes_cache(cache_file):
    assert not cache_file.exists()
    alpha, loc, scale = spdf.calculate_skewed_normal_params(
        median=1.0,
        lower_err=0.05,
        upper_err=0.06,
    )
    assert cache_file.exists() and cache_file.stat().st_size > 0
    payload = json.loads(cache_file.read_text())
    assert len(payload) == 1
    key = next(iter(payload))
    np_alpha, np_loc, np_scale = payload[key]
    # cached values exactly match the return tuple
    assert (alpha, loc, scale) == (np_alpha, np_loc, np_scale)


def test_second_call_uses_cache_and_bypasses_solver(cache_file, monkeypatch):
    # populate via a normal call
    spdf.calculate_skewed_normal_params(median=2.5, lower_err=0.1, upper_err=0.15)
    # now patch the uncached worker so any call to it would crash
    calls = {"n": 0}

    def boom(*a, **k):
        calls["n"] += 1
        raise AssertionError("uncached solver should NOT be called on a hit")

    monkeypatch.setattr(spdf, "_calculate_skewed_normal_params_uncached", boom)
    # second call with identical inputs hits the cache → solver never runs
    a, l, s = spdf.calculate_skewed_normal_params(median=2.5, lower_err=0.1, upper_err=0.15)
    assert calls["n"] == 0
    assert isinstance(a, float) and isinstance(l, float) and isinstance(s, float)


def test_different_inputs_generate_different_keys(cache_file):
    spdf.calculate_skewed_normal_params(1.0, 0.05, 0.06)
    spdf.calculate_skewed_normal_params(1.0, 0.05, 0.07)  # different upper_err
    spdf.calculate_skewed_normal_params(1.0, 0.06, 0.06)  # different lower_err
    spdf.calculate_skewed_normal_params(1.1, 0.05, 0.06)  # different median
    payload = json.loads(cache_file.read_text())
    assert len(payload) == 4


def test_disable_via_env(cache_file, monkeypatch):
    """ALLESFITTER_SIMULATE_PDF_NO_CACHE=1 must bypass both read and write."""
    monkeypatch.setenv("ALLESFITTER_SIMULATE_PDF_NO_CACHE", "1")
    spdf.calculate_skewed_normal_params(3.14, 0.2, 0.3)
    # cache file must not be created
    assert not cache_file.exists()


def test_corrupt_cache_is_tolerated(cache_file):
    cache_file.write_text("{this is not valid json")
    # should silently treat as empty and recompute
    a, l, s = spdf.calculate_skewed_normal_params(1.0, 0.05, 0.06)
    assert all(map(lambda v: isinstance(v, float), (a, l, s)))
    # and overwrite with a valid cache
    payload = json.loads(cache_file.read_text())
    assert len(payload) == 1


def test_cache_speedup_is_real(cache_file):
    """Sanity: second call returns in << 100 ms vs ~seconds for the solve."""
    # warm
    spdf.calculate_skewed_normal_params(5.0, 0.4, 0.5)
    t0 = time.perf_counter()
    spdf.calculate_skewed_normal_params(5.0, 0.4, 0.5)
    dt_hit = time.perf_counter() - t0
    # 100 ms is comfortable; cache hit is typically <5 ms.
    assert dt_hit < 0.1, f"cache hit took {dt_hit*1000:.1f} ms"


def test_repr_key_stability_across_processes(cache_file, monkeypatch):
    """The float-repr key must be byte-stable so two processes generate
    the same key for the same input."""
    k1 = spdf._cache_key(1.234567890123, 0.0123, 0.0456)
    k2 = spdf._cache_key(1.234567890123, 0.0123, 0.0456)
    assert k1 == k2
    # ints and floats with the same value get the same key
    k_int = spdf._cache_key(1.0, 0.0123, 0.0456)
    k_flt = spdf._cache_key(1, 0.0123, 0.0456)
    assert k_int == k_flt
