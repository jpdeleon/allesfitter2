"""Unit tests for :mod:`allesfitter.detection.gpu_tls`.

No CUDA GPU or ``gputls`` install is required (or assumed) to run these:
the device-detection/engine-selection tests mock ``gputls``/CuPy entirely,
and :func:`_assemble_tls_style_result` is exercised against a hand-built
GTLS-shaped ``(model, gtlsResult)`` pair -- since that function's whole job
is turning GTLS's found period/T0/duration/depth into a full
``transitleastsquares``-style dict via the real (CPU, already-installed)
``transitleastsquares.stats`` functions, this validates the real code path
without needing an actual GPU search to have produced the inputs.
"""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest
from transitleastsquares.helpers import transit_mask
from transitleastsquares.results import transitleastsquaresresults

from allesfitter.detection import gpu_tls

#: The exact field set a CPU `transitleastsquares` result carries -- see
#: `transitleastsquares.results.transitleastsquaresresults`.
_TLS_RESULT_KEYS = set(transitleastsquaresresults(*range(41)).keys())


def _fake_gtls_model_and_result(
    period: float = 3.3,
    duration_days: float = 0.12,
    depth_frac: float = 0.01,
    n: int = 3000,
    span_days: float = 30.0,
    seed: int = 0,
):
    """A synthetic light curve with an injected box transit, plus a
    hand-built GTLS ``(model, gtlsResult)`` pair reporting that transit as
    "found" -- standing in for what a real GTLS GPU search would return,
    without needing CUDA/``gputls`` to produce it."""
    rng = np.random.default_rng(seed)
    t = np.sort(rng.uniform(0, span_days, n))
    T0 = period / 2.0

    intransit = transit_mask(t, period, duration_days, T0)
    y = np.ones(n)
    y[intransit] -= depth_frac
    y = y + rng.normal(0, 0.0005, n)
    dy = np.full(n, 0.0005)

    raw_duration = duration_days / period
    raw_durations = np.array([raw_duration * 0.5, raw_duration, raw_duration * 1.5])

    model = SimpleNamespace(
        t=t,
        y=y,
        dy=dy,
        per=1.0,
        rp=0.1,
        a=15.0,
        inc=90.0,
        ecc=0.0,
        w=90.0,
        u=[0.4804, 0.1867],
        limb_dark="quadratic",
    )
    gtls_result = SimpleNamespace(
        period=period,
        T0=T0,
        rawDuration=raw_duration,
        rawDurations=raw_durations,
        depth=depth_frac,
        SDE=12.0,
        periods=np.linspace(1.0, 10.0, 50),
        power=np.linspace(0.0, 12.0, 50),
        raw_power=np.zeros(50),
        chi2=np.zeros(50),
    )
    return model, gtls_result


class TestGetGpuDeviceCount:
    def test_gpu_tls_engine_is_selected_when_cuda_device_is_visible(self, monkeypatch):
        sentinel = object()
        monkeypatch.setattr(gpu_tls, "_gpu_device_count", lambda: 1)
        monkeypatch.setitem(sys.modules, "gputls", types.SimpleNamespace(gtls=sentinel))

        assert gpu_tls.get_gpu_tls_engine() is sentinel

    def test_cpu_fallback_when_no_cuda_device_is_visible(self, monkeypatch):
        monkeypatch.setattr(gpu_tls, "_gpu_device_count", lambda: 0)

        assert gpu_tls.get_gpu_tls_engine() is None

    def test_cpu_fallback_when_gputls_is_not_installed(self, monkeypatch):
        monkeypatch.setattr(gpu_tls, "_gpu_device_count", lambda: 1)
        monkeypatch.setitem(sys.modules, "gputls", None)

        with pytest.warns(UserWarning, match="gputls could not be loaded"):
            assert gpu_tls.get_gpu_tls_engine() is None

    def test_cpu_fallback_when_gpu_detection_raises(self, monkeypatch):
        def fail_detection():
            raise RuntimeError("CUDA driver unavailable")

        monkeypatch.setattr(gpu_tls, "_gpu_device_count", fail_detection)

        assert gpu_tls.get_gpu_tls_engine() is None


class TestAssembleTlsStyleResult:
    def test_produces_the_full_tls_result_key_set(self):
        model, gtls_result = _fake_gtls_model_and_result()

        results = gpu_tls._assemble_tls_style_result(model, gtls_result)

        assert set(results.keys()) == _TLS_RESULT_KEYS

    def test_recovers_the_injected_period_and_depth(self):
        model, gtls_result = _fake_gtls_model_and_result(period=3.3, depth_frac=0.01)

        results = gpu_tls._assemble_tls_style_result(model, gtls_result)

        assert results["period"] == pytest.approx(3.3)
        # TLS convention: `depth` is the in-transit relative flux (near 1),
        # i.e. `1 - dimming`, unlike GTLS's own `depth` (the dimming itself).
        assert results["depth"] == pytest.approx(1.0 - 0.01, abs=1e-3)
        assert 0.0 < results["rp_rs"] < 1.0

    def test_folded_and_lightcurve_model_arrays_are_self_consistent(self):
        model, gtls_result = _fake_gtls_model_and_result()

        results = gpu_tls._assemble_tls_style_result(model, gtls_result)

        assert len(results["folded_phase"]) == len(model.t)
        assert len(results["folded_y"]) == len(model.t)
        assert len(results["model_lightcurve_time"]) == len(results["model_lightcurve_model"])
        assert len(results["model_folded_phase"]) == len(results["model_folded_model"])

    def test_per_transit_stats_cover_every_transit_epoch(self):
        model, gtls_result = _fake_gtls_model_and_result()

        results = gpu_tls._assemble_tls_style_result(model, gtls_result)

        assert results["distinct_transit_count"] >= 1
        assert len(results["transit_depths"]) == results["transit_count"]
        assert len(results["transit_depths_uncertainties"]) == results["transit_count"]
        assert len(results["per_transit_count"]) == results["transit_count"]

    def test_raises_on_non_physical_period(self):
        model, gtls_result = _fake_gtls_model_and_result()
        gtls_result.period = np.nan

        with pytest.raises(ValueError, match="non-physical"):
            gpu_tls._assemble_tls_style_result(model, gtls_result)


class TestRunTls:
    def _patch_cpu_tls(self, monkeypatch, result):
        class FakeCpuTls:
            def __init__(self, *args, **kwargs):
                pass

            def power(self, **kwargs):
                return result

        monkeypatch.setattr("transitleastsquares.transitleastsquares", FakeCpuTls)

    def test_uses_cpu_tls_when_no_gpu_engine_available(self, monkeypatch):
        monkeypatch.setattr(gpu_tls, "get_gpu_tls_engine", lambda: None)
        cpu_result = {"period": 1.0}
        self._patch_cpu_tls(monkeypatch, cpu_result)

        result = gpu_tls.run_tls(
            np.array([1.0]), np.array([1.0]), np.array([0.01]), {}, quiet=False
        )

        assert result == cpu_result

    def test_uses_gpu_result_when_gtls_succeeds(self, monkeypatch):
        class FakeGtls:
            def __init__(self, *args, **kwargs):
                pass

            def power(self, **kwargs):
                return "raw_gtls_result"

        monkeypatch.setattr(gpu_tls, "get_gpu_tls_engine", lambda: FakeGtls)
        sentinel = {"period": 42.0}
        monkeypatch.setattr(
            gpu_tls,
            "_assemble_tls_style_result",
            lambda model, gtls_result: sentinel,
        )

        result = gpu_tls.run_tls(
            np.array([1.0]), np.array([1.0]), np.array([0.01]), {}, quiet=False
        )

        assert result is sentinel

    def test_falls_back_to_cpu_tls_when_gtls_raises(self, monkeypatch):
        class BrokenGtls:
            def __init__(self, *args, **kwargs):
                raise RuntimeError("failed to find CUDA headers")

        monkeypatch.setattr(gpu_tls, "get_gpu_tls_engine", lambda: BrokenGtls)
        cpu_result = {"period": 7.0}
        self._patch_cpu_tls(monkeypatch, cpu_result)

        with pytest.warns(UserWarning, match="GTLS failed"):
            result = gpu_tls.run_tls(
                np.array([1.0]), np.array([1.0]), np.array([0.01]), {}, quiet=False
            )

        assert result == cpu_result

    def test_gtls_fallback_warning_is_emitted_even_under_quiet(self, monkeypatch):
        class BrokenGtls:
            def __init__(self, *args, **kwargs):
                raise RuntimeError("failed to find CUDA headers")

        monkeypatch.setattr(gpu_tls, "get_gpu_tls_engine", lambda: BrokenGtls)
        self._patch_cpu_tls(monkeypatch, {"period": 7.0})

        with pytest.warns(UserWarning, match="GTLS failed"):
            gpu_tls.run_tls(np.array([1.0]), np.array([1.0]), np.array([0.01]), {}, quiet=True)

    def test_quiet_suppresses_stdout_from_the_engine(self, monkeypatch, capsys):
        monkeypatch.setattr(gpu_tls, "get_gpu_tls_engine", lambda: None)

        class NoisyCpuTls:
            def __init__(self, *args, **kwargs):
                pass

            def power(self, **kwargs):
                print("noisy TLS progress output")
                return {"period": 1.0}

        monkeypatch.setattr("transitleastsquares.transitleastsquares", NoisyCpuTls)

        gpu_tls.run_tls(np.array([1.0]), np.array([1.0]), np.array([0.01]), {}, quiet=True)

        assert "noisy TLS progress output" not in capsys.readouterr().out
