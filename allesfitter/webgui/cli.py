"""``allesfitter-gui`` console entry point — serve the web GUI with uvicorn.

On startup the target ``--port`` is freed first: any process already listening
on it is terminated, so re-running the GUI never fails with "address already in
use" or leaves an orphaned server behind. Disable with ``--no-kill-existing``.
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import time


def _pids_on_port(port: int) -> list[int]:
    """PIDs listening on *port* (best-effort via ``lsof``, then ``fuser``)."""
    pids: set[int] = set()
    try:
        out = subprocess.run(
            ["lsof", "-ti", f"tcp:{port}", "-sTCP:LISTEN"],
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout
        pids.update(int(tok) for tok in out.split() if tok.strip().isdigit())
    except (FileNotFoundError, subprocess.SubprocessError):
        pass
    if not pids:
        try:
            res = subprocess.run(
                ["fuser", f"{port}/tcp"], capture_output=True, text=True, timeout=5
            )
            # fuser prints the PIDs to stderr, e.g. "9000/tcp:  12345 67890".
            for tok in (res.stdout + " " + res.stderr).replace(":", " ").split():
                if tok.isdigit():
                    pids.add(int(tok))
        except (FileNotFoundError, subprocess.SubprocessError):
            pass
    return sorted(pids)


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def free_port(host: str, port: int) -> list[int]:
    """Terminate any process listening on *port* (excluding this one).

    Returns the list of PIDs that were signalled. SIGTERM first, then SIGKILL
    for anything still alive after a short grace period.
    """
    me = os.getpid()
    pids = [p for p in _pids_on_port(port) if p != me]
    if not pids:
        return []
    print(f"[allesfitter-gui] freeing port {port}: stopping existing server (pid {pids})")
    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.time() + 3.0
    while time.time() < deadline and any(_alive(p) for p in pids):
        time.sleep(0.1)
    for pid in pids:
        if _alive(pid):
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    return pids


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="allesfitter-gui",
        description="Serve the allesfitter web GUI (multi-band / multi-epoch transit fitting).",
    )
    parser.add_argument("--runs-root", default="webgui_runs", help="directory for per-run datadirs")
    parser.add_argument(
        "--db", default=None, help="SQLite path (default: <runs-root>/webgui.sqlite3)"
    )
    parser.add_argument("--toi-csv", default=None, help="local ExoFOP TOI table for auto-fill")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-network", action="store_true", help="disable NASA Archive lookups")
    parser.add_argument(
        "--no-kill-existing",
        action="store_true",
        help="do not stop a process already listening on --port before starting",
    )
    args = parser.parse_args(argv)

    if not args.no_kill_existing:
        free_port(args.host, args.port)

    import uvicorn

    from allesfitter.webgui.app import create_app

    app = create_app(
        args.runs_root,
        args.db,
        toi_csv=args.toi_csv,
        allow_network=not args.no_network,
    )
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
