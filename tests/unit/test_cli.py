import importlib

from typer.testing import CliRunner

import allesfitter.cli as cli_module
from allesfitter.cli import app


def test_prepare_forwards_tic_transit_arguments_before_running_script(monkeypatch):
    received = {}
    monkeypatch.setattr(
        cli_module,
        "_run_script",
        lambda script_name, argv: received.update(script_name=script_name, argv=argv),
    )

    result = CliRunner().invoke(
        app,
        [
            "prepare",
            "-tic",
            "123",
            "-s",
            "1",
            "--period",
            "3.5",
            "--period-err",
            "0.01",
            "--epoch",
            "2459000.25",
            "--epoch-err",
            "0.02",
            "--duration",
            "2.4",
            "--duration-err",
            "0.1",
            "--depth",
            "1500",
            "--depth-err",
            "100",
        ],
    )

    assert result.exit_code == 0, result.output
    assert received["script_name"] == "prepare_allesfit.py"
    assert received["argv"][-16:] == [
        "--period",
        "3.5",
        "--period-err",
        "0.01",
        "--epoch",
        "2459000.25",
        "--epoch-err",
        "0.02",
        "--duration",
        "2.4",
        "--duration-err",
        "0.1",
        "--depth",
        "1500.0",
        "--depth-err",
        "100.0",
    ]


def test_prepare_forwards_h5_argument_before_running_script(monkeypatch):
    received = {}
    monkeypatch.setattr(
        cli_module,
        "_run_script",
        lambda script_name, argv: received.update(script_name=script_name, argv=argv),
    )

    result = CliRunner().invoke(
        app,
        [
            "prepare",
            "-tic",
            "123",
            "-s",
            "1",
            "--h5",
            "/path/to/TIC123_s1_tls.h5",
        ],
    )

    assert result.exit_code == 0, result.output
    assert received["script_name"] == "prepare_allesfit.py"
    assert received["argv"][-2:] == ["--h5", "/path/to/TIC123_s1_tls.h5"]


def test_optimize_defaults_to_differential_evolution_with_half_cpu_workers(monkeypatch, tmp_path):
    received = {}
    optimize_submodule = importlib.import_module("allesfitter.optimize")
    monkeypatch.setattr(
        optimize_submodule,
        "optimize",
        lambda dir_path, **kwargs: received.update(kwargs),
    )

    result = CliRunner().invoke(app, ["optimize", str(tmp_path)])

    assert result.exit_code == 0, result.output
    assert received["method"] == "differential_evolution"
    assert received["workers"] == cli_module._default_workers()
    assert received["workers"] >= 1
    assert received["maxfevals"] is None


def test_optimize_forwards_workers_and_maxfevals_overrides(monkeypatch, tmp_path):
    received = {}
    optimize_submodule = importlib.import_module("allesfitter.optimize")
    monkeypatch.setattr(
        optimize_submodule,
        "optimize",
        lambda dir_path, **kwargs: received.update(kwargs),
    )

    result = CliRunner().invoke(
        app,
        [
            "optimize",
            str(tmp_path),
            "--workers",
            "4",
            "--maxfevals",
            "200",
        ],
    )

    assert result.exit_code == 0, result.output
    assert received["workers"] == 4
    assert received["maxfevals"] == 200


def test_transit_search_forwards_arguments(monkeypatch, tmp_path):
    received = {}
    blind_search_submodule = importlib.import_module("allesfitter.detection.blind_search")
    monkeypatch.setattr(
        blind_search_submodule,
        "transit_search",
        lambda results_dir, **kwargs: received.update(results_dir=results_dir, **kwargs),
    )

    results_dir = tmp_path / "ns_results"
    results_dir.mkdir()
    result = CliRunner().invoke(
        app,
        [
            "transit-search",
            str(results_dir),
            "--sde-min",
            "6.0",
            "--period-min",
            "0.5",
            "--period-max",
            "15",
            "--mask-width-factor",
            "2.0",
            "--max-candidates",
            "5",
            "--mission",
            "k2",
            "--skip-known-recovery-check",
            "--recovery-period-frac",
            "0.1",
        ],
    )

    assert result.exit_code == 0, result.output
    assert received["results_dir"] == str(results_dir)
    assert received["sde_min"] == 6.0
    assert received["period_min"] == 0.5
    assert received["period_max"] == 15.0
    assert received["mask_width_factor"] == 2.0
    assert received["max_candidates"] == 5
    assert received["mission"] == "k2"
    assert received["check_known_recovery"] is False
    assert received["recovery_period_frac"] == 0.1
    assert received["quiet"] is False


def test_transit_search_defaults(monkeypatch, tmp_path):
    received = {}
    blind_search_submodule = importlib.import_module("allesfitter.detection.blind_search")
    monkeypatch.setattr(
        blind_search_submodule,
        "transit_search",
        lambda results_dir, **kwargs: received.update(results_dir=results_dir, **kwargs),
    )

    results_dir = tmp_path / "mcmc_results"
    results_dir.mkdir()
    result = CliRunner().invoke(app, ["transit-search", str(results_dir)])

    assert result.exit_code == 0, result.output
    assert received["sde_min"] == 8.0
    assert received["period_min"] is None
    assert received["period_max"] is None
    assert received["mask_width_factor"] == 1.5
    assert received["outdir"] is None
    assert received["file_extension"] == ".pdf"
    assert received["max_candidates"] == 20
    assert received["mission"] == "tess"
    assert received["check_known_recovery"] is True
    assert received["recovery_period_frac"] == 0.05


def test_show_params_labels_fit_status_column(tmp_path):
    (tmp_path / "params.csv").write_text(
        "#name,value,fit,bounds,label,unit,coupled_with\n"
        "b_rr,0.0653,1,uniform 0 0.2500,$R_b/R_*$,,\n"
        "b_f_c,0,0,uniform -1 1,$f_c$,,\n"
    )

    result = CliRunner().invoke(app, ["show-params", str(tmp_path)])

    assert result.exit_code == 0, result.output
    header = next(
        line for line in result.output.splitlines() if "name" in line and "bounds" in line
    )
    assert "fit?" in header
    assert "✓" in result.output
    assert "✗" in result.output


def test_show_results_formats_posterior_and_derived_tables(tmp_path):
    from allesfitter.results import target_output_directory

    root = target_output_directory(tmp_path)
    results_dir = root / "mcmc_results"
    results_dir.mkdir(parents=True)
    legacy_dir = root / "results"
    legacy_dir.mkdir(parents=True)
    (legacy_dir / "mcmc_table.csv").write_text(
        "#name,median,lower_error,upper_error,label,unit\nlegacy_parameter,99,1,1,legacy,\n"
    )
    (results_dir / "mcmc_table.csv").write_text(
        "#name,median,lower_error,upper_error,label,unit\n"
        "#Fitted parameters,,,\n"
        "b_rr,0.123456,0.01,0.02,$R_b/R_*$,\n"
        "b_epoch,2459928.960338116,0.000925,0.001116,$T_0$,BJD\n"
        "b_f_c,0.0,(fixed),(fixed),$f_c$,\n"
    )
    (results_dir / "mcmc_derived_table.csv").write_text(
        "#property,value,lower_error,upper_error,source\n"
        "Combined density; $rho_{star,combined}$,4.321,0.12,0.23,derived\n"
    )

    result = CliRunner().invoke(app, ["show-results", str(tmp_path)])

    assert result.exit_code == 0, result.output
    assert "MCMC Posterior Results" in result.output
    assert "MCMC Derived Results" in result.output
    assert "b_rr" in result.output
    assert "0.123" in result.output
    assert "2459928.96034" in result.output
    assert "fixed" in result.output
    assert "Combined density; $rho_{star,combined}$" in result.output
    assert "mcmc_results/mcmc_table.csv" in result.output
    assert "legacy_parameter" not in result.output


def test_show_results_falls_back_to_results_subdir(tmp_path):
    from allesfitter.results import target_output_directory

    results_dir = target_output_directory(tmp_path) / "results"
    results_dir.mkdir(parents=True)
    (results_dir / "ns_table.csv").write_text(
        "#name,median,lower_error,upper_error,label,unit\n"
        "b_period,2.3456789,0.0001,0.0002,$P_b$,d\n"
    )

    result = CliRunner().invoke(app, ["show-results", str(tmp_path)])

    assert result.exit_code == 0, result.output
    assert "Nested Sampling Posterior Results" in result.output
    assert "b_period" in result.output
    assert "results/ns_table.csv" in result.output


def test_show_results_reports_how_to_generate_missing_tables(tmp_path):
    result = CliRunner().invoke(app, ["show-results", str(tmp_path)])

    assert result.exit_code == 1
    assert "No processed result tables found" in result.output
    assert "mcmc-output" in result.output
    assert "ns-output" in result.output
