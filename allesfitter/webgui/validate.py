"""Pre-launch dry-run validation of a staged run directory.

Loads the generated ``settings.csv`` / ``params.csv`` + staged ``<inst>.csv``
files through the engine's own :class:`allesfitter.basement.Basement`
(``quiet=True``, no sampler). That reuses the exact validator the real fit uses —
unknown-key rejection, share-group checks, ``_cols`` covariate resolution,
bandpass/companion consistency — turning what would be a run-time crash into an
actionable form-level message.

The engine import is lazy so importing this module never requires the compiled
fitting dependencies; :func:`dry_run` degrades to a clear message when they are
unavailable.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class ValidationResult:
    """Outcome of a dry-run validation."""

    ok: bool
    error: str = ""
    n_instruments: int = 0
    n_free_params: int = 0

    def __bool__(self) -> bool:
        return self.ok


def dry_run(run_dir: str | Path) -> ValidationResult:
    """Validate a run directory by loading it into ``Basement``.

    Returns a :class:`ValidationResult`; never raises for expected
    configuration errors (they are captured in ``error``).
    """
    run_dir = Path(run_dir)
    if not (run_dir / "settings.csv").is_file():
        return ValidationResult(False, error=f"no settings.csv in {run_dir}")
    if not (run_dir / "params.csv").is_file():
        return ValidationResult(False, error=f"no params.csv in {run_dir}")

    try:
        from allesfitter.basement import Basement
    except Exception as exc:  # engine (ellc/celerite) not importable
        return ValidationResult(False, error=f"fitting engine unavailable: {exc}")

    try:
        b = Basement(str(run_dir), quiet=True)
    except Exception as exc:
        return ValidationResult(False, error=str(exc))

    n_inst = len(b.settings.get("inst_phot", []))
    n_free = _count_free_params(b)
    return ValidationResult(True, n_instruments=n_inst, n_free_params=n_free)


def _count_free_params(basement) -> int:
    """Number of fitted parameters, robust to the engine's attribute naming."""
    fitkeys = getattr(basement, "fitkeys", None)
    if fitkeys is not None:
        return len(fitkeys)
    params = getattr(basement, "params", {})
    fit = params.get("fit") if isinstance(params, dict) else None
    if fit is not None:
        try:
            return int(sum(1 for f in fit if int(f) == 1))
        except (TypeError, ValueError):
            return 0
    return 0


def initial_guess_preview(run_dir: str | Path, out_path: str | Path | None = None):
    """Render allesfitter's initial-guess figure for *run_dir*, if possible.

    Returns the saved image path, or ``None`` when the engine or plotting stack
    is unavailable. Best-effort — never raises.
    """
    run_dir = Path(run_dir)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import allesfitter
    except Exception:
        return None

    try:
        fig = allesfitter.show_initial_guess(str(run_dir), file_extension=".png", return_figs=True)
    except Exception:
        return None
    if fig is None:
        return None

    figs = fig if isinstance(fig, (list, tuple)) else [fig]
    out_path = Path(out_path) if out_path else run_dir / "initial_guess.png"
    try:
        figs[0].savefig(out_path, dpi=100, bbox_inches="tight")
    except Exception:
        return None
    return out_path
