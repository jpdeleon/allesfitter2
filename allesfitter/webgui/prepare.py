"""Run ``scripts/prepare_allesfit.py`` from the GUI to build a ready-to-fit run.

The manual ``/fit`` flow requires the user to already have light-curve files and
every prior typed by hand. This module bridges the *other* allesfitter tool — the
``prepare_allesfit`` CLI — into the web GUI: given a target + sector it
**auto-downloads** the light curves and **auto-generates** the whole config
(``params.csv`` / ``settings.csv`` / light-curve CSVs / GP·dilution·TTV priors).

Because ``prepare_allesfit`` is a 2311-line CLI whose orchestration lives entirely
in ``main()`` (and allesfitter's config is module-global), we run it as a detached
**subprocess** — the same shape as :mod:`allesfitter.webgui.jobs` — instead of
importing it. The child is spawned importably against the GUI's own interpreter::

    python -c "from allesfitter.scripts import prepare_allesfit as p; p()" <argv>

which is exactly what the ``prepare_allesfit`` console entry point does.

Two details make the produced dir launch-compatible with :func:`jobs.launch`:

* ``prepare_allesfit`` writes the datadir to ``<-dir>/<target_name>`` where
  ``target_name`` is derived internally, so we point ``-dir`` at a fresh per-run
  work dir and :func:`discover_datadir` finds the produced subdir afterward.
* its generated ``run.py`` has the sampler calls commented out (manual uncomment),
  so once generation succeeds we re-emit ``run.py`` from
  :data:`allesfitter.webgui.config_writer.RUN_PY_TEMPLATE`, whose
  ``ALLESFIT_SAMPLER`` dispatch is what the launcher relies on.
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
from collections import deque
from dataclasses import dataclass
from pathlib import Path

from allesfitter.webgui import config_writer
from allesfitter.webgui import runstore as rs
from allesfitter.webgui import validate as _validate

# Child spawned against the GUI's own interpreter so it resolves the same
# allesfitter install (no reliance on PATH or an editable console script).
_WRAPPER = "from allesfitter.scripts import prepare_allesfit as _p; _p()"

# Maps the GUI's mission choice to prepare_allesfit's mutually-exclusive segment
# flag. TESS sectors accept several values (-s nargs=+); K2/Kepler take one.
_SEGMENT_FLAG = {"tess": "-s", "k2": "-c", "kepler": "-q"}
_TARGET_FLAG = {"name": "-name", "toi": "-toi", "tic": "-tic", "ctoi": "-ctoi"}

# run_id -> live handle for currently-running prepare subprocesses (this process).
_ACTIVE: dict[str, PrepareHandle] = {}
_LOCK = threading.Lock()


@dataclass
class PrepareHandle:
    """A live (or just-finished) prepare subprocess launch."""

    run_id: str
    work_dir: Path
    process: subprocess.Popen
    log_path: Path


# --- argv construction (pure, unit-testable) -------------------------------
def _tokens(value: object) -> list[str]:
    """Normalize a space-separated string or a list into a token list."""
    if value in (None, ""):
        return []
    if isinstance(value, (list, tuple)):
        return [str(v).strip() for v in value if str(v).strip()]
    return [tok for tok in str(value).split() if tok]


def _is_int(value: str) -> bool:
    try:
        int(value)
    except (TypeError, ValueError):
        return False
    return True


def _truthy(value: object) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return bool(value)


def build_prepare_argv(spec: dict, *, work_dir: str | Path | None = None) -> list[str]:
    """Turn a ``/prepare`` form payload into ``prepare_allesfit`` CLI args.

    Raises :class:`ValueError` (routes turn this into HTTP 400) when a target id
    or a sector/campaign/quarter is missing, or an id that must be numeric isn't.
    """
    id_type = (spec.get("id_type") or "name").strip().lower()
    flag = _TARGET_FLAG.get(id_type)
    if flag is None:
        raise ValueError(f"unknown id_type {id_type!r} (use name/toi/tic/ctoi)")
    target = str(spec.get("target") or "").strip()
    if not target:
        raise ValueError("target is required")
    if id_type != "name" and not _is_int(target):
        raise ValueError(f"{id_type} id must be an integer, got {target!r}")

    argv: list[str] = [flag, target]

    mission = (spec.get("mission") or "tess").strip().lower()
    seg_tokens = _tokens(spec.get("sectors"))
    if not seg_tokens:
        raise ValueError("at least one sector/campaign/quarter is required")
    seg_flag = _SEGMENT_FLAG.get(mission, "-s")
    if seg_flag == "-s":
        argv += ["-s", *seg_tokens]
    else:  # campaign/quarter take a single value
        argv += [seg_flag, seg_tokens[0]]
    argv += ["-m", mission]

    pipeline = (spec.get("pipeline") or "").strip()
    if pipeline:
        argv += ["-p", pipeline]
    lc_type = (spec.get("lc_type") or "").strip()
    if lc_type:
        argv += ["-lc", lc_type]

    exptime = str(spec.get("exptime") or "").strip()
    if exptime:
        argv += ["-e", exptime]

    argv += ["-f", *(_tokens(spec.get("filename")) or ["tess"])]

    bandpass = _tokens(spec.get("bandpass"))
    if bandpass:
        argv += ["-bp", *bandpass]

    sigma = spec.get("sigma")
    if sigma not in (None, ""):
        argv += ["-sig", str(sigma)]
    quality = (spec.get("quality") or "").strip()
    if quality:
        argv += ["-qb", quality]

    if _truthy(spec.get("ttv")):
        argv.append("--ttv")
    if _truthy(spec.get("overwrite")):
        argv.append("-o")

    wd = work_dir if work_dir is not None else spec.get("work_dir")
    if wd:
        argv += ["-dir", str(wd)]
    return argv


def _prepare_cmd(argv: list[str], python: str | None = None) -> list[str]:
    return [python or sys.executable, "-c", _WRAPPER, *argv]


# --- discovery / parsing ---------------------------------------------------
def discover_datadir(work_dir: str | Path) -> Path | None:
    """Return the newest child dir of *work_dir* holding params.csv+settings.csv."""
    work_dir = Path(work_dir)
    if not work_dir.is_dir():
        return None
    candidates = [
        child
        for child in work_dir.iterdir()
        if child.is_dir()
        and (child / "params.csv").is_file()
        and (child / "settings.csv").is_file()
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def parse_inst_phot(settings_path: str | Path) -> list[str]:
    """Read the ``inst_phot`` labels from a generated settings.csv (best-effort)."""
    try:
        for line in Path(settings_path).read_text().splitlines():
            if line.startswith("#"):
                continue
            key, _, value = line.partition(",")
            if key.strip() == "inst_phot":
                return value.split()
    except OSError:
        pass
    return []


# --- launch / finalize -----------------------------------------------------
def launch_prepare(
    store: rs.RunStore,
    run_id: str,
    work_dir: str | Path,
    argv: list[str],
    *,
    python: str | None = None,
    extra_env: dict | None = None,
) -> PrepareHandle:
    """Spawn ``prepare_allesfit`` in *work_dir* and finalize the run on exit.

    The run is expected to already exist in *store* (state ``preparing``); the
    waiter thread moves it to ``prepared`` (validated) or ``failed``.
    """
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    log_path = work_dir / "prepare.log"

    env = dict(os.environ)
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MPLBACKEND", "Agg")  # prepare may render contamination figures
    if extra_env:
        env.update(extra_env)

    log_file = open(log_path, "w")  # noqa: SIM115 -- closed by the waiter thread
    process = subprocess.Popen(
        _prepare_cmd(argv, python),
        cwd=str(work_dir),
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    store.set_state(run_id, rs.PREPARING, pid=process.pid)

    handle = PrepareHandle(run_id, work_dir, process, log_path)
    with _LOCK:
        _ACTIVE[run_id] = handle
    threading.Thread(target=_wait_and_finalize, args=(store, handle, log_file), daemon=True).start()
    return handle


def _wait_and_finalize(store: rs.RunStore, handle: PrepareHandle, log_file) -> None:
    try:
        returncode = handle.process.wait()
    finally:
        log_file.close()
    try:
        _finalize(store, handle, returncode)
    finally:
        with _LOCK:
            _ACTIVE.pop(handle.run_id, None)


def _finalize(store: rs.RunStore, handle: PrepareHandle, returncode: int) -> None:
    """Validate + register the produced datadir, or mark the run failed."""
    run = store.get_run(handle.run_id)
    if run and run.get("state") == rs.STOPPED:
        return

    if returncode != 0:
        store.set_state(
            handle.run_id, rs.FAILED, error=f"prepare_allesfit exited with code {returncode}"
        )
        return

    datadir = discover_datadir(handle.work_dir)
    if datadir is None:
        store.set_state(
            handle.run_id, rs.FAILED, error="prepare produced no datadir (see prepare.log)"
        )
        return

    # Re-emit run.py so the GUI launcher's ALLESFIT_SAMPLER dispatch works.
    (datadir / "run.py").write_text(config_writer.RUN_PY_TEMPLATE)
    insts = parse_inst_phot(datadir / "settings.csv")
    # Point the run at the real datadir + fill the instruments badge either way,
    # so a validation failure still lets the user inspect what was generated.
    store.update_run(handle.run_id, run_dir=str(datadir), insts=" ".join(insts))

    result = _validate.dry_run(datadir)
    if not result.ok:
        store.set_state(handle.run_id, rs.FAILED, error=f"validation failed: {result.error}")
        return

    preview = _validate.initial_guess_preview(datadir)
    if preview is None:
        store.set_state(
            handle.run_id,
            rs.FAILED,
            error="show_initial_guess failed to produce a preview",
        )
        return
    store.set_state(handle.run_id, rs.PREPARED, error="")


# --- live controls (mirror jobs.py) ----------------------------------------
def is_running(run_id: str) -> bool:
    """True while this process is running *run_id*'s prepare subprocess."""
    with _LOCK:
        handle = _ACTIVE.get(run_id)
    return handle is not None and handle.process.poll() is None


def stop(run_id: str) -> bool:
    """Terminate a running prepare process group. True if one was signalled."""
    from allesfitter.webgui import runstore as rs
    from allesfitter.webgui.job_store import get_job_store

    with _LOCK:
        handle = _ACTIVE.get(run_id)

    pid = None
    if handle is not None:
        pid = handle.process.pid
    else:
        try:
            store = get_job_store().store
            run = store.get_run(run_id)
            if run and run.get("pid"):
                pid = run["pid"]
        except Exception:
            pass

    signalled = False
    if pid is not None:
        try:
            os.killpg(os.getpgid(pid), 15)  # SIGTERM to the group
            signalled = True
        except (ProcessLookupError, PermissionError, OSError):
            try:
                os.kill(pid, 15)
                signalled = True
            except Exception:
                pass
        except Exception:
            pass

    try:
        store = get_job_store().store
        store.set_state(run_id, rs.STOPPED, error="Stopped by user")
    except Exception:
        pass

    return signalled or (pid is not None)


def tail_log(work_dir: str | Path, n: int = 300) -> str:
    """Return the last *n* lines of ``<work_dir>/prepare.log`` (or "")."""
    log_path = Path(work_dir) / "prepare.log"
    if not log_path.is_file():
        return ""
    try:
        with open(log_path, errors="ignore") as fh:
            return "".join(deque(fh, maxlen=n))
    except OSError:
        return ""
