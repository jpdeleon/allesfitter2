import gzip
import pickle
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from allesfitter.cli import app
from allesfitter.compare import CompareError, compare


def _write_ns_fixture(dir_path: Path, fitkeys, samples, table_rows, logz=10.0, logzerr=0.2):
    """Write a minimal ns_results/{save_ns.pickle.gz,ns_table.csv} fixture pair.

    ``samples`` : (Nsamples, len(fitkeys)) array. Weights are uniform, so the
    resampled posterior is just ``samples`` itself (up to reordering).
    ``table_rows`` : list of (name, median, lower_error, upper_error, label, unit)
    tuples; use ``"(fixed)"`` for lower/upper_error to mark a fixed parameter.
    """
    outdir = dir_path / "ns_results"
    outdir.mkdir(parents=True)
    samples = np.asarray(samples)
    raw = {
        "backend": "dynesty",
        "samples": samples,
        "logwt": np.zeros(samples.shape[0]),
        "logz": np.array([logz]),
        "logzerr": np.array([logzerr]),
        "fitkeys": np.array(fitkeys, dtype=object),
    }
    with gzip.open(outdir / "save_ns.pickle.gz", "wb") as stream:
        pickle.dump(raw, stream)

    lines = ["#name,median,lower_error,upper_error,label,unit", "#Fitted parameters,,,"]
    for name, median, lower, upper, label, unit in table_rows:
        lines.append(f"{name},{median},{lower},{upper},{label},{unit}")
    (outdir / "ns_table.csv").write_text("\n".join(lines) + "\n")


def _make_dirs(tmp_path, with_extra_offset=True):
    rng = np.random.default_rng(0)
    n = 500

    dir_a = tmp_path / "with_offset"
    dir_a.mkdir()
    samples_a = np.column_stack(
        [
            rng.normal(1.0, 0.05, n),
            rng.normal(2.0, 0.05, n),
            rng.normal(3.0, 0.05, n),
        ]
        + ([rng.normal(0.0, 0.01, n)] if with_extra_offset else [])
    )
    fitkeys_a = ["p1", "p2", "p3"] + (["offset"] if with_extra_offset else [])
    table_rows_a = [
        ("p1", 1.0, 0.05, 0.05, "$p_1$", ""),
        ("p2", 2.0, 0.05, 0.05, "$p_2$", ""),
        ("p3", 3.0, 0.05, 0.05, "$p_3$", ""),
        ("fixed_param", 0.0, "(fixed)", "(fixed)", "$f$", ""),
    ]
    if with_extra_offset:
        table_rows_a.append(("offset", 0.0, 0.01, 0.01, "$\\mathrm{offset}$", ""))
    _write_ns_fixture(dir_a, fitkeys_a, samples_a, table_rows_a, logz=10.0, logzerr=0.2)

    dir_b = tmp_path / "without_offset"
    dir_b.mkdir()
    samples_b = np.column_stack(
        [
            rng.normal(1.01, 0.05, n),
            rng.normal(1.99, 0.05, n),
            rng.normal(3.02, 0.05, n),
        ]
    )
    table_rows_b = [
        ("p1", 1.01, 0.05, 0.05, "$p_1$", ""),
        ("p2", 1.99, 0.05, 0.05, "$p_2$", ""),
        ("p3", 3.02, 0.05, 0.05, "$p_3$", ""),
        ("fixed_param", 0.0, "(fixed)", "(fixed)", "$f$", ""),
    ]
    _write_ns_fixture(dir_b, ["p1", "p2", "p3"], samples_b, table_rows_b, logz=13.4, logzerr=0.2)

    return dir_a, dir_b


def test_compare_writes_corner_and_table_for_shared_and_extra_params(tmp_path):
    dir_a, dir_b = _make_dirs(tmp_path)

    result = compare([str(dir_a), str(dir_b)], out_dir=tmp_path / "out")

    assert result.shared_params == ["p1", "p2", "p3"]
    assert result.extra_params["with_offset"] == ["offset"]
    assert result.extra_params["without_offset"] == []
    assert result.corner_path.is_file()
    assert result.table_csv_path.is_file()
    assert "fixed_param" in result.all_names
    assert "offset" in result.all_names

    csv_text = result.table_csv_path.read_text()
    header = csv_text.splitlines()[0]
    assert (
        header
        == "name,label,unit,with_offset_median,with_offset_lower_error,with_offset_upper_error,without_offset_median,without_offset_lower_error,without_offset_upper_error"
    )
    offset_line = next(line for line in csv_text.splitlines() if line.startswith("offset,"))
    assert offset_line.endswith(",,,")  # blank median/lower/upper for the dir that lacks it

    assert len(result.logz_rows) == 2
    assert result.logz_rows[0]["delta_logz"] == 0.0
    assert result.logz_rows[1]["delta_logz"] == pytest.approx(3.4)


def test_compare_labels_default_to_directory_basenames(tmp_path):
    dir_a, dir_b = _make_dirs(tmp_path)

    result = compare([str(dir_a), str(dir_b)], out_dir=tmp_path / "out")

    assert result.labels == ["with_offset", "without_offset"]


def test_compare_accepts_explicit_labels(tmp_path):
    dir_a, dir_b = _make_dirs(tmp_path)

    result = compare([str(dir_a), str(dir_b)], labels=["A", "B"], out_dir=tmp_path / "out")

    assert result.labels == ["A", "B"]
    assert result.extra_params["A"] == ["offset"]


def test_compare_requires_at_least_two_directories(tmp_path):
    dir_a, _ = _make_dirs(tmp_path)

    with pytest.raises(CompareError, match="at least 2"):
        compare([str(dir_a)], out_dir=tmp_path / "out")


def test_compare_rejects_mismatched_label_count(tmp_path):
    dir_a, dir_b = _make_dirs(tmp_path)

    with pytest.raises(CompareError, match="label"):
        compare([str(dir_a), str(dir_b)], labels=["only-one"], out_dir=tmp_path / "out")


def test_compare_rejects_duplicate_labels(tmp_path):
    dir_a, dir_b = _make_dirs(tmp_path)

    with pytest.raises(CompareError, match="unique"):
        compare([str(dir_a), str(dir_b)], labels=["same", "same"], out_dir=tmp_path / "out")


def test_compare_raises_when_no_shared_parameters(tmp_path):
    dir_a = tmp_path / "a"
    dir_a.mkdir()
    _write_ns_fixture(
        dir_a,
        ["only_in_a"],
        np.random.default_rng(0).normal(0, 1, (200, 1)),
        [("only_in_a", 0.0, 0.1, 0.1, "$a$", "")],
    )
    dir_b = tmp_path / "b"
    dir_b.mkdir()
    _write_ns_fixture(
        dir_b,
        ["only_in_b"],
        np.random.default_rng(1).normal(0, 1, (200, 1)),
        [("only_in_b", 0.0, 0.1, 0.1, "$b$", "")],
    )

    with pytest.raises(CompareError, match="No fit parameters are shared"):
        compare([str(dir_a), str(dir_b)], out_dir=tmp_path / "out")


def test_compare_explicit_params_option_selects_subset(tmp_path):
    dir_a, dir_b = _make_dirs(tmp_path)

    result = compare([str(dir_a), str(dir_b)], params=["p1", "p2"], out_dir=tmp_path / "out")

    assert result.shared_params == ["p1", "p2"]


def test_compare_explicit_params_missing_in_a_directory_raises(tmp_path):
    dir_a, dir_b = _make_dirs(tmp_path)

    with pytest.raises(CompareError, match="offset"):
        compare([str(dir_a), str(dir_b)], params=["p1", "offset"], out_dir=tmp_path / "out")


def test_compare_raises_when_pickle_missing(tmp_path):
    dir_a, dir_b = _make_dirs(tmp_path)
    (dir_a / "ns_results" / "save_ns.pickle.gz").unlink()

    with pytest.raises(CompareError, match="save_ns.pickle.gz"):
        compare([str(dir_a), str(dir_b)], out_dir=tmp_path / "out")


def test_compare_raises_when_table_missing(tmp_path):
    dir_a, dir_b = _make_dirs(tmp_path)
    (dir_a / "ns_results" / "ns_table.csv").unlink()

    with pytest.raises(CompareError, match="ns_table.csv"):
        compare([str(dir_a), str(dir_b)], out_dir=tmp_path / "out")


def test_compare_cli_prints_shared_table_and_logz(tmp_path):
    dir_a, dir_b = _make_dirs(tmp_path)

    result = CliRunner().invoke(
        app,
        [
            "compare",
            str(dir_a),
            str(dir_b),
            "--out",
            str(tmp_path / "out"),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Shared Posterior Parameters" in result.output
    assert "p1" in result.output
    assert "with_offset: only in this fit" in result.output
    assert "offset" in result.output
    assert "Log-Evidence" in result.output
    assert "+3.400" in result.output
    assert (tmp_path / "out" / "compare_corner.pdf").is_file()
    assert (tmp_path / "out" / "compare_table.csv").is_file()


def test_compare_cli_requires_two_directories(tmp_path):
    dir_a, _ = _make_dirs(tmp_path)

    result = CliRunner().invoke(app, ["compare", str(dir_a)])

    assert result.exit_code == 1
    assert "at least 2" in result.output


def test_compare_cli_reports_missing_output_with_clear_error(tmp_path):
    empty_a = tmp_path / "empty_a"
    empty_a.mkdir()
    empty_b = tmp_path / "empty_b"
    empty_b.mkdir()

    result = CliRunner().invoke(app, ["compare", str(empty_a), str(empty_b)])

    assert result.exit_code == 1
    assert "save_ns.pickle.gz" in result.output
