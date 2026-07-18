from pathlib import Path

from allesfitter.webapp import Workbench


def test_series_reads_allesfitter_commented_csv_header(tmp_path: Path):
    app = Workbench(tmp_path / "workspace")
    target = app.create_target({"name": "TIC-1", "identifier_type": "tic", "identifier_value": "1"})
    data_dir = Path(target["data_dir"])
    data_dir.mkdir()
    (data_dir / "tess.csv").write_text(
        "#time,flux,flux_err\n1.0,0.99,0.01\n2.0,1.01,0.01\n", encoding="utf-8"
    )

    assert app.series(target["id"]) == {"time": [1.0, 2.0], "flux": [0.99, 1.01]}
