#!/usr/bin/env python3
"""Drive an allesfitter model grid: run each datadir's run.py as a separate
process and record status + log(Z) back into grid.csv.

Each fit runs in its OWN process (allesfitter config is module-global, and the
fit already parallelises internally via multiprocess_cores) — so this driver is
sequential by default. Re-runnable: only `pending`/`failed` rows are launched
unless you pass --all.

The grid directory holds `grid.csv` plus one subdirectory per `run_id`, each
containing a `run.py` launcher (as produced by `generate_grid.py build`). It is
a required positional argument; running with no arguments prints this help.

Examples:
    python run_allesfitter_grid.py examples/TOI-6715/datasets
    python run_allesfitter_grid.py DATADIR --filter chroma=chr   # only chromatic runs
    python run_allesfitter_grid.py DATADIR --filter dataset=tess_spoc120 svar=1
    python run_allesfitter_grid.py DATADIR --sampler mcmc --all  # re-run with MCMC
    python run_allesfitter_grid.py DATADIR --dry-run             # show what would run
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys
from pathlib import Path

LOGZ_RE = re.compile(r"log\(Z\)\s*=\s*([-\d.eE+]+)\s*\+-\s*([-\d.eE+]+)")


def load_rows(grid_csv: Path) -> list[dict]:
    if not grid_csv.exists():
        sys.exit(f"{grid_csv} not found — run `generate_grid.py build` first.")
    with grid_csv.open(newline="") as fh:
        return list(csv.DictReader(fh))


def save_rows(grid_csv: Path, rows: list[dict]) -> None:
    with grid_csv.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def matches(row: dict, filters: dict[str, str]) -> bool:
    return all(str(row.get(k, "")) == v for k, v in filters.items())


def extract_logz(run_dir: Path) -> str:
    """Best-effort: scan results/ logs for `log(Z) = <val> +- <err>`."""
    results = run_dir / "results"
    if not results.exists():
        return ""
    best = ""
    for f in sorted(results.rglob("*")):
        if f.suffix.lower() not in (".log", ".txt"):
            continue
        m = None
        for _m in LOGZ_RE.finditer(f.read_text(errors="ignore")):
            m = _m  # keep last match in file
        if m:
            best = f"{float(m.group(1)):.2f} +- {float(m.group(2)):.2f}"
    return best


def run_one(grid_dir: Path, row: dict, sampler: str) -> tuple[str, str]:
    run_dir = grid_dir / row["run_id"]
    launcher = run_dir / "run.py"
    if not launcher.exists():
        return "missing", ""
    env = {**os.environ, "ALLESFIT_SAMPLER": sampler}
    log_path = run_dir / "results" / "driver.log"
    log_path.parent.mkdir(exist_ok=True)
    with log_path.open("w") as log:
        proc = subprocess.run(
            [sys.executable, "run.py"], cwd=run_dir, env=env, stdout=log, stderr=subprocess.STDOUT
        )
    if proc.returncode != 0:
        return "failed", ""
    return "done", extract_logz(run_dir)


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "grid_dir",
        type=Path,
        metavar="GRID_DIR",
        help="directory holding grid.csv and the run_id subdirs",
    )
    p.add_argument(
        "--filter",
        nargs="*",
        default=[],
        metavar="KEY=VAL",
        help="only run rows matching every KEY=VAL (e.g. chroma=chr)",
    )
    p.add_argument("--sampler", choices=["ns", "mcmc"], default="mcmc")
    p.add_argument("--all", action="store_true", help="re-run rows even if already done")
    p.add_argument("--dry-run", action="store_true")

    #::: no arguments at all -> print full help instead of a terse usage error
    if len(sys.argv) == 1:
        p.print_help()
        return

    args = p.parse_args()

    grid_dir = args.grid_dir.resolve()
    grid_csv = grid_dir / "grid.csv"

    filters = dict(kv.split("=", 1) for kv in args.filter)
    rows = load_rows(grid_csv)
    todo = [
        r
        for r in rows
        if matches(r, filters) and (args.all or r["status"] in ("pending", "failed"))
    ]

    if not todo:
        print("nothing to run (check --filter / --all).")
        return
    print(f"{len(todo)} run(s) to launch with sampler={args.sampler}:")
    for r in todo:
        print(f"  {r['run_id']}")
    if args.dry_run:
        return

    print("")
    for i, row in enumerate(todo, 1):
        print(f"[{i}/{len(todo)}] {grid_dir / row['run_id']}")
        print("  running … ", end="", flush=True)
        status, logz = run_one(grid_dir, row, args.sampler)
        row["status"], row["logZ"] = status, logz
        print(f"{status}" + (f"  log(Z)={logz}" if logz else ""))
        save_rows(grid_csv, rows)  # persist after each fit so progress survives interrupts

    done = sum(r["status"] == "done" for r in rows)
    print(f"done: {done}/{len(rows)} rows complete; results in {grid_csv}")


if __name__ == "__main__":
    main()
