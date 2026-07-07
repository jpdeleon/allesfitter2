"""``allesfitter-gui`` console entry point — serve the web GUI with uvicorn."""

from __future__ import annotations

import argparse


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
    args = parser.parse_args(argv)

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
