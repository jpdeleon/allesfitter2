"""Unit tests for Dynesty backend argument forwarding."""

from __future__ import annotations

import numpy as np

from allesfitter.utils.ns_backends import dynesty_backend


def test_dynamic_single_process_forwards_convergence_settings(monkeypatch, tmp_path):
    calls = {}

    class FakeSampler:
        def __init__(self, *args, **kwargs):
            calls["init"] = kwargs
            self.results = {
                "samples": np.zeros((1, 2)),
                "logwt": np.zeros(1),
                "logz": np.zeros(1),
                "logzerr": np.zeros(1),
            }

        def run_nested(self, **kwargs):
            calls["run_nested"] = kwargs

    monkeypatch.setattr(dynesty_backend.dynesty, "DynamicNestedSampler", FakeSampler)
    settings = {
        "ns_nlive": 123,
        "ns_bound": "multi",
        "ns_sample": "rwalk",
        "ns_tol": 0.25,
        "ns_modus": "dynamic",
        "multiprocess": False,
        "print_progress": False,
    }

    dynesty_backend.run(
        lambda _: 0.0,
        lambda values: values,
        2,
        settings,
        ["x", "y"],
        str(tmp_path / "results"),
        logprint=lambda *_: None,
    )

    assert calls["init"] == {"bound": "multi", "sample": "rwalk"}
    assert calls["run_nested"] == {
        "nlive_init": 123,
        "dlogz_init": 0.25,
        "print_progress": False,
    }
