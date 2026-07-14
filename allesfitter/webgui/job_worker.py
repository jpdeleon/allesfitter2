"""Small fit wrapper that leaves a durable exit marker for restart recovery."""

from __future__ import annotations

import argparse
import json
import os
import runpy
import time
import traceback
from pathlib import Path


def _write_marker(path: Path, returncode: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps({"returncode": returncode, "finished_at": time.time()}))
    os.replace(temporary, path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--marker", required=True)
    parser.add_argument("--env", action="append", default=[])
    parser.add_argument("script", nargs="?", default="run.py")
    args = parser.parse_args(argv)
    for assignment in args.env:
        key, separator, value = assignment.partition("=")
        if not separator or key not in {
            "ALLESFIT_OPTIMIZE_METHOD",
            "ALLESFIT_OPTIMIZE_REFINE",
            "ALLESFIT_OPTIMIZE_RESTARTS",
        }:
            parser.error("unsupported worker environment option")
        os.environ[key] = value
    returncode = 0
    try:
        runpy.run_path(args.script, run_name="__main__")
    except SystemExit as exc:
        returncode = int(exc.code or 0) if isinstance(exc.code, (int, type(None))) else 1
    except BaseException:  # the traceback belongs in the run log
        traceback.print_exc()
        returncode = 1
    finally:
        _write_marker(Path(args.marker), returncode)
    return returncode


if __name__ == "__main__":
    raise SystemExit(main())
