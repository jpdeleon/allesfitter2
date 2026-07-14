import gzip
import pickle

import numpy as np
import pytest

from allesfitter.postprocessing.nested_sampling_compare_logZ import compare_logz, main
from allesfitter.results import results_directory


def _save_ns_result(run, logz, logzerr, dirname="ns_results"):
    outdir = run / dirname
    outdir.mkdir(parents=True)
    with gzip.open(outdir / "save_ns.pickle.gz", "wb") as stream:
        pickle.dump({"logz": np.array([logz]), "logzerr": np.array([logzerr])}, stream)


def test_sampler_results_directories_are_distinct(tmp_path):
    assert results_directory(tmp_path, "mcmc", for_write=True) == str(tmp_path / "mcmc_results")
    assert results_directory(tmp_path, "ns", for_write=True) == str(tmp_path / "ns_results")


def test_results_directory_falls_back_to_legacy_directory(tmp_path):
    legacy = tmp_path / "results"
    legacy.mkdir()
    assert results_directory(tmp_path, "mcmc") == str(legacy)
    assert results_directory(tmp_path, "ns") == str(legacy)


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
