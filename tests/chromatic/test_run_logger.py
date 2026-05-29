"""Unit tests for the centralized run logger.

These verify the contract of allesfitter.run_logger.log_run / log_event:
  - start/end JSON rows land on the correct path
  - duration is recorded and non-negative
  - failures are logged with status=failed and the exception type/message
  - ALLESFITTER_RUN_LOG env var overrides the default path
  - concurrent appends from sibling processes do not interleave (smoke test)
"""

from __future__ import annotations

import json
import os
import pathlib
from typing import Any

import pytest

from allesfitter import run_logger


@pytest.fixture
def tmp_log(tmp_path) -> pathlib.Path:
    return tmp_path / "runs.jsonl"


def _load(path: pathlib.Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


class TestLogRunHappyPath:
    def test_writes_start_and_end_rows(self, tmp_log, tmp_path):
        with run_logger.log_run("ns_fit", str(tmp_path), log_path=tmp_log,
                                announce=False) as run_id:
            assert isinstance(run_id, str)
            assert len(run_id) == 12

        rows = _load(tmp_log)
        assert len(rows) == 2
        start, end = rows
        assert start["event"] == "start"
        assert end["event"] == "end"
        assert start["run_id"] == end["run_id"] == run_id
        assert start["command"] == end["command"] == "ns_fit"
        assert start["datadir"] == end["datadir"] == str(tmp_path.resolve())
        assert end["status"] == "completed"
        assert end["duration_sec"] >= 0.0

    def test_metadata_fields_present(self, tmp_log, tmp_path):
        with run_logger.log_run("ns_output", str(tmp_path), log_path=tmp_log,
                                announce=False):
            pass
        for row in _load(tmp_log):
            for k in ("timestamp", "pid", "hostname", "user"):
                assert k in row, f"missing field {k!r} in row {row}"
            assert row["timestamp"].endswith("Z")
            assert isinstance(row["pid"], int) and row["pid"] > 0


class TestLogRunFailure:
    def test_exception_recorded_and_reraised(self, tmp_log, tmp_path):
        with pytest.raises(ValueError, match="boom"):
            with run_logger.log_run("ns_fit", str(tmp_path), log_path=tmp_log,
                                    announce=False):
                raise ValueError("boom")

        rows = _load(tmp_log)
        assert len(rows) == 2
        end = rows[1]
        assert end["event"] == "end"
        assert end["status"] == "failed"
        assert end["error"] == "ValueError: boom"
        assert "traceback" in end and "ValueError: boom" in end["traceback"]
        assert end["duration_sec"] >= 0.0


class TestLogPathResolution:
    def test_env_var_overrides_default(self, tmp_path, monkeypatch):
        override = tmp_path / "custom_runs.jsonl"
        monkeypatch.setenv("ALLESFITTER_RUN_LOG", str(override))
        # Default arg is None → get_log_path() is called inside log_event.
        run_logger.log_event({"event": "start", "command": "x", "datadir": "."})
        assert override.exists()
        assert run_logger.get_log_path() == override

    def test_default_path_when_env_unset(self, monkeypatch):
        monkeypatch.delenv("ALLESFITTER_RUN_LOG", raising=False)
        assert run_logger.get_log_path() == run_logger.DEFAULT_LOG_PATH


class TestExtraFields:
    def test_start_row_merges_extra(self, tmp_log, tmp_path):
        with run_logger.log_run("ns_fit", str(tmp_path), log_path=tmp_log,
                                extra={"git_sha": "deadbeef", "n_inst": 4},
                                announce=False):
            pass
        rows = _load(tmp_log)
        assert rows[0]["git_sha"] == "deadbeef"
        assert rows[0]["n_inst"] == 4
        # end row must NOT carry the extras — that would conflate static
        # config with timing/status fields.
        assert "git_sha" not in rows[1]


class TestTail:
    def test_tail_returns_most_recent(self, tmp_log, tmp_path):
        for i in range(5):
            with run_logger.log_run(f"cmd_{i}", str(tmp_path), log_path=tmp_log,
                                    announce=False):
                pass
        # 5 runs → 10 lines; tail(4) returns the last 4 lines.
        out = run_logger.tail(n=4, log_path=tmp_log)
        assert len(out) == 4
        # Last line is the end of cmd_4.
        assert out[-1]["event"] == "end"
        assert out[-1]["command"] == "cmd_4"

    def test_tail_empty_file_returns_empty_list(self, tmp_log):
        # File doesn't exist yet.
        assert run_logger.tail(n=10, log_path=tmp_log) == []


class TestConcurrentAppendSmoke:
    def test_lock_serializes_writes(self, tmp_log, tmp_path):
        # Smoke test: 50 rapid append calls from the same process should
        # produce 50 well-formed JSON lines. Cross-process locking is
        # exercised by fcntl in production; here we verify the wrapper
        # never corrupts the file.
        for i in range(50):
            run_logger.log_event(
                {"event": "start", "command": "ping", "datadir": str(tmp_path),
                 "i": i},
                log_path=tmp_log,
            )
        rows = _load(tmp_log)
        assert len(rows) == 50
        assert [r["i"] for r in rows] == list(range(50))
