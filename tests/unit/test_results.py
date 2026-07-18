import gzip
import pickle
from pathlib import Path

import numpy as np
import pytest

from allesfitter.postprocessing.nested_sampling_compare_logZ import compare_logz, main
from allesfitter.results import (
    DEFAULT_OUTPUT_BASE,
    output_base,
    results_directory,
    target_output_directory,
)


def _save_ns_result(run, logz, logzerr, dirname="ns_results"):
    outdir = run / dirname
    outdir.mkdir(parents=True)
    with gzip.open(outdir / "save_ns.pickle.gz", "wb") as stream:
        pickle.dump({"logz": np.array([logz]), "logzerr": np.array([logzerr])}, stream)


def test_output_base_defaults_to_home_ql_allesfitter(monkeypatch):
    monkeypatch.delenv("ALLESFITTER_RESULTS_DIR", raising=False)
    assert output_base() == DEFAULT_OUTPUT_BASE
    assert DEFAULT_OUTPUT_BASE == Path.home() / "ql" / "allesfitter"


def test_output_base_honours_environment_override(monkeypatch, tmp_path):
    monkeypatch.setenv("ALLESFITTER_RESULTS_DIR", str(tmp_path))
    assert output_base() == tmp_path


def test_results_written_under_per_target_output_root(monkeypatch, tmp_path):
    monkeypatch.setenv("ALLESFITTER_RESULTS_DIR", str(tmp_path))
    datadir = tmp_path / "data" / "HIP67522"
    datadir.mkdir(parents=True)

    mcmc = results_directory(datadir, "mcmc", for_write=True)
    ns = results_directory(datadir, "ns", for_write=True)

    assert mcmc == str(tmp_path / "HIP67522" / "mcmc_results")
    assert ns == str(tmp_path / "HIP67522" / "ns_results")
    assert Path(mcmc).is_dir()
    assert Path(ns).is_dir()


def test_distinct_targets_do_not_collide(monkeypatch, tmp_path):
    monkeypatch.setenv("ALLESFITTER_RESULTS_DIR", str(tmp_path))
    first = tmp_path / "a" / "TOI-1"
    second = tmp_path / "b" / "TOI-2"
    first.mkdir(parents=True)
    second.mkdir(parents=True)

    assert target_output_directory(first) == tmp_path / "TOI-1"
    assert target_output_directory(second) == tmp_path / "TOI-2"


def test_results_ignore_legacy_in_place_directory(monkeypatch, tmp_path):
    monkeypatch.setenv("ALLESFITTER_RESULTS_DIR", str(tmp_path))
    datadir = tmp_path / "data" / "TOI-9"
    (datadir / "results").mkdir(parents=True)  # old in-place run

    # New location is used regardless of the legacy directory next to the data.
    assert results_directory(datadir, "mcmc") == str(tmp_path / "TOI-9" / "mcmc_results")


def test_results_directory_rejects_unknown_sampler(tmp_path):
    with pytest.raises(ValueError, match="mcmc, ns"):
        results_directory(tmp_path, "other")


def test_compare_logz_reads_specific_and_legacy_directories(tmp_path):
    reference = tmp_path / "reference"
    alternative = tmp_path / "alternative"
    _save_ns_result(reference, 10.0, 0.3)
    _save_ns_result(alternative, 13.0, 0.4, dirname="results")

    rows = compare_logz([reference, alternative], labels=["reference", "alternative"])

    assert rows[0]["delta_logz"] == 0.0
    assert rows[1] == {
        "label": "alternative",
        "logz": 13.0,
        "logzerr": 0.4,
        "delta_logz": 3.0,
        "delta_logzerr": pytest.approx(0.5),
    }


def test_compare_logz_cli_prints_table(tmp_path, capsys):
    run = tmp_path / "run"
    _save_ns_result(run, 4.5, 0.2)

    main([str(run), "--labels", "model"])

    output = capsys.readouterr().out
    assert "label\tlogz\tlogzerr\tdelta_logz\tdelta_logzerr" in output
    assert "model\t4.5\t0.2\t0\t0.282843" in output
