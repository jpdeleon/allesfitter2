"""``allesfitter-gui`` console entry point — serve the web GUI with uvicorn.

The server fails safely when its port is occupied. Development users may opt in
to terminating an existing listener with ``--kill-existing``.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import time

_RELOAD_CONFIG_ENV = "ALLESFITTER_GUI_RELOAD_CONFIG"


def _reload_app():
    """Uvicorn factory used by the reload worker process."""
    from allesfitter.webgui.app import create_app

    config = json.loads(os.environ[_RELOAD_CONFIG_ENV])
    return create_app(**config)


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
    parser.add_argument("--port", type=int, default=5100)
    parser.add_argument(
        "--root-path",
        default="",
        help="external URL prefix when served by a reverse proxy (for example /allesfitter)",
    )
    parser.add_argument("--no-network", action="store_true", help="disable NASA Archive lookups")
    parser.add_argument(
        "--reload",
        action="store_true",
        help="reload the development server when Python source files change",
    )
    port_group = parser.add_mutually_exclusive_group()
    port_group.add_argument(
        "--kill-existing",
        action="store_true",
        help="development only: terminate a process already listening on --port",
    )
    port_group.add_argument(
        "--no-kill-existing",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--max-concurrent-fits", type=int, default=1)
    parser.add_argument("--max-queued-fits-per-user", type=int, default=2)
    args = parser.parse_args(argv)

    if args.kill_existing:
        free_port(args.host, args.port)

    import uvicorn

    app_config = {
        "runs_root": args.runs_root,
        "db_path": args.db,
        "toi_csv": args.toi_csv,
        "allow_network": not args.no_network,
        "root_path": args.root_path,
        "max_concurrent_fits": args.max_concurrent_fits,
        "max_queued_fits_per_user": args.max_queued_fits_per_user,
    }
    if args.reload:
        # Uvicorn's reload supervisor requires an import string so each worker
        # can import a fresh application after a source change.
        os.environ[_RELOAD_CONFIG_ENV] = json.dumps(app_config)
        uvicorn.run(
            "allesfitter.webgui.cli:_reload_app",
            factory=True,
            host=args.host,
            port=args.port,
            reload=True,
        )
    else:
        from allesfitter.webgui.app import create_app

        uvicorn.run(create_app(**app_config), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
