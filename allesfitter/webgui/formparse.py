"""Build a :class:`~allesfitter.webgui.models.FitConfig` from a JSON payload.

The web form (``fit.html``) serializes its dynamic companion/instrument rows into
a JSON body; this module turns that neutral dict into a typed ``FitConfig`` plus
the ``(label, source_path)`` list that :mod:`staging` needs. Kept separate from
the route so it is unit-testable without FastAPI.
"""

from __future__ import annotations

from allesfitter.webgui import instruments as _inst
from allesfitter.webgui import models as m
from allesfitter.webgui.staging import validate_label


def _f(value, default=None) -> float | None:
    if value in (None, ""):
        return default
    return float(value)


def _build_companion(spec: dict) -> m.CompanionSpec:
    name = spec.get("name", "b")
    comp = m.default_companion(
        name,
        period=_f(spec.get("period"), 1.0),
        epoch=_f(spec.get("epoch"), 0.0),
        rr=_f(spec.get("rr"), 0.1),
    )
    period_err = _f(spec.get("period_err"))
    if period_err is not None:
        comp.period = m.normal(
            comp.period.value, comp.period.value, period_err, label=comp.period.label, unit="d"
        )
    epoch_err = _f(spec.get("epoch_err"))
    if epoch_err is not None:
        comp.epoch = m.normal(
            comp.epoch.value, comp.epoch.value, epoch_err, label=comp.epoch.label, unit="BJD"
        )
    return comp


def _build_instrument(spec: dict) -> m.InstrumentSpec:
    label = spec.get("label", "").strip()
    if not label:
        raise ValueError("every instrument needs a non-empty label")
    validate_label(label)
    band = (spec.get("band") or _inst.suggest_band(label)).strip()
    if not band:
        raise ValueError(f"instrument '{label}' has no bandpass (set one explicitly)")
    inst = m.InstrumentSpec(
        label=label,
        band=band,
        data_file=spec.get("data_file", ""),
        baseline=spec.get("baseline", "sample_linear"),
        baseline_cols=tuple(spec.get("baseline_cols", []) or []),
        ld_law=spec.get("ld_law", "quad"),
        t_exp=_f(spec.get("t_exp")),
        grid=spec.get("grid", "very_sparse"),
        error=spec.get("error", "sample"),
    )
    if spec.get("dilution"):
        inst.dilution = m.default_dilution()
    return inst


def build_fit_config(payload: dict) -> tuple[m.FitConfig, list[tuple[str, str]]]:
    """Return ``(fit_config, staging_items)`` from a form JSON *payload*.

    ``staging_items`` is a list of ``(label, source_path)`` for :func:`staging.stage_all`.
    Raises ``ValueError`` on malformed input (routes turn this into HTTP 400).
    """
    target = (payload.get("target") or "").strip()
    if not target:
        raise ValueError("target is required")

    comp_specs = payload.get("companions") or [{"name": "b"}]
    companions = [_build_companion(c) for c in comp_specs]

    inst_specs = payload.get("instruments") or []
    if not inst_specs:
        raise ValueError("at least one instrument is required")
    instruments = [_build_instrument(i) for i in inst_specs]

    staging_items = [(i.label, i.data_file) for i in instruments if i.data_file]

    share_groups = tuple(
        tuple(g) for g in (payload.get("share_groups") or []) if isinstance(g, list) and len(g) > 1
    )

    ldc: dict[str, m.BandLDC] = {}
    for band, coeffs in (payload.get("ldc") or {}).items():
        cb = _inst.canonical_band(band)
        ldc[cb] = m.BandLDC(
            q1=m.uniform(_f(coeffs.get("q1"), 0.5), 0, 1, label=f"$q_{{1; {cb}}}$"),
            q2=m.uniform(_f(coeffs.get("q2"), 0.5), 0, 1, label=f"$q_{{2; {cb}}}$"),
        )

    chromatic = payload.get("chromatic", None)
    if isinstance(chromatic, str):
        chromatic = {"true": True, "false": False, "auto": None, "": None}.get(
            chromatic.lower(), None
        )

    cfg = m.FitConfig(
        target=target,
        companions=companions,
        instruments=instruments,
        ldc=ldc,
        share_groups=share_groups,
        chromatic=chromatic,
        use_host_density_prior=bool(payload.get("use_host_density_prior", True)),
        fast_fit=bool(payload.get("fast_fit", True)),
        fit_ttvs=bool(payload.get("fit_ttvs", False)),
        multiprocess_cores=int(payload.get("multiprocess_cores", 4) or 4),
    )
    return cfg, staging_items
