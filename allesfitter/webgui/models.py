"""Typed configuration model for a single allesfitter web-GUI fit.

These dataclasses are the neutral in-memory contract between the web form (which
parses HTTP form data into a :class:`FitConfig`) and :mod:`config_writer` (which
serializes a :class:`FitConfig` into ``settings.csv`` / ``params.csv``). Tests
build a :class:`FitConfig` directly, bypassing the web layer.

Nothing here imports the engine or FastAPI.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from allesfitter.webgui import instruments as _inst

# --- sampler defaults ------------------------------------------------------
DEFAULT_MCMC: dict[str, object] = {
    "mcmc_nwalkers": 100,
    "mcmc_total_steps": 2000,
    "mcmc_burn_steps": 1000,
    "mcmc_thin_by": 2,
}
DEFAULT_NS: dict[str, object] = {
    "ns_modus": "dynamic",
    "ns_nlive": 500,
    "ns_bound": "single",
    "ns_sample": "auto",
    "ns_tol": 100,
}


# --- priors ----------------------------------------------------------------
@dataclass
class Prior:
    """One ``params.csv`` row's fit-relevant fields.

    ``bounds`` uses the engine grammar: ``uniform lo hi`` / ``normal mu sig`` /
    ``trunc_normal lo hi mu sig``. It must be non-empty for a fitted parameter.
    """

    value: float
    fit: bool = True
    bounds: str = ""
    label: str = ""
    unit: str = ""
    coupled_with: str = ""


def uniform(value: float, lo: float, hi: float, **kw: object) -> Prior:
    return Prior(value, bounds=f"uniform {lo} {hi}", **kw)  # type: ignore[arg-type]


def normal(value: float, mu: float, sig: float, **kw: object) -> Prior:
    return Prior(value, bounds=f"normal {mu} {sig}", **kw)  # type: ignore[arg-type]


def trunc_normal(value: float, lo: float, hi: float, mu: float, sig: float, **kw: object) -> Prior:
    return Prior(value, bounds=f"trunc_normal {lo} {hi} {mu} {sig}", **kw)  # type: ignore[arg-type]


def fixed(value: float, bounds: str = "uniform 0 1", **kw: object) -> Prior:
    """A non-fitted parameter. ``bounds`` is still written (engine parses it)."""
    return Prior(value, fit=False, bounds=bounds, **kw)  # type: ignore[arg-type]


# --- limb darkening --------------------------------------------------------
@dataclass
class BandLDC:
    """Quadratic limb-darkening coefficients (q-space) for one bandpass."""

    q1: Prior
    q2: Prior


def default_ldc(band: str) -> BandLDC:
    return BandLDC(
        q1=uniform(0.5, 0, 1, label=f"$q_{{1; {band}}}$"),
        q2=uniform(0.5, 0, 1, label=f"$q_{{2; {band}}}$"),
    )


# --- instruments -----------------------------------------------------------
@dataclass
class InstrumentSpec:
    """One photometric instrument in the fit.

    Priors left ``None`` are filled with sensible defaults by the config writer,
    so a minimal ``InstrumentSpec(label=..., band=...)`` is valid.
    """

    label: str
    band: str
    data_file: str = ""  # source path staged into <label>.csv (staging fills it)
    baseline: str = "sample_linear"
    baseline_cols: tuple[str, ...] = ()
    ld_law: str = "quad"
    t_exp: float | None = None
    grid: str = "very_sparse"
    error: str = "sample"  # error_flux_<inst>
    # optional priors (defaulted by the writer when None)
    ln_err: Prior | None = None
    dilution: Prior | None = None  # None -> no dil_<inst> row
    baseline_offset: Prior | None = None
    baseline_slope: Prior | None = None
    gp_lnsigma: Prior | None = None
    gp_lnrho: Prior | None = None


def default_ln_err() -> Prior:
    return uniform(-6, -10, -3, label=r"$\log{\sigma}$", unit="rel. flux")


def default_offset() -> Prior:
    return uniform(0.0, -0.01, 0.01)


def default_slope() -> Prior:
    return uniform(0.0, -0.1, 0.1)


def default_gp_lnsigma() -> Prior:
    return uniform(-5, -15, 0)


def default_gp_lnrho() -> Prior:
    return uniform(0.0, -5, 10)


def default_dilution() -> Prior:
    return uniform(0.0, 0, 1)


# --- companions ------------------------------------------------------------
def _default_rr() -> Prior:
    return uniform(0.1, 0, 0.3, label=r"$R_b / R_\star$")


def _default_rsuma() -> Prior:
    return uniform(0.2, 0, 1, label=r"$(R_\star + R_b) / a_b$")


def _default_cosi() -> Prior:
    return uniform(0.0, 0, 1, label=r"$\cos{i_b}$")


def _default_epoch() -> Prior:
    return normal(0.0, 0.0, 0.01, label=r"$T_{0;b}$", unit="BJD")


def _default_period() -> Prior:
    return normal(1.0, 1.0, 1e-4, label=r"$P_b$", unit="d")


def _default_f_c() -> Prior:
    return fixed(0.0, "uniform -1 1", label=r"$\sqrt{e_b} \cos{\omega_b}$")


def _default_f_s() -> Prior:
    return fixed(0.0, "uniform -1 1", label=r"$\sqrt{e_b} \sin{\omega_b}$")


@dataclass
class CompanionSpec:
    """One transiting companion's astrophysical parameters.

    ``rr`` is the single radius-ratio prior; in chromatic mode the writer
    replicates it into one ``<name>_rr_<band>`` row per bandpass.
    """

    name: str = "b"
    rr: Prior = field(default_factory=_default_rr)
    rsuma: Prior = field(default_factory=_default_rsuma)
    cosi: Prior = field(default_factory=_default_cosi)
    epoch: Prior = field(default_factory=_default_epoch)
    period: Prior = field(default_factory=_default_period)
    f_c: Prior = field(default_factory=_default_f_c)
    f_s: Prior = field(default_factory=_default_f_s)


def default_companion(
    name: str = "b", *, period: float = 1.0, epoch: float = 0.0, rr: float = 0.1
) -> CompanionSpec:
    """Convenience factory with name-aware labels and seeded ephemeris."""
    return CompanionSpec(
        name=name,
        rr=uniform(rr, 0, 0.3, label=rf"$R_{name} / R_\star$"),
        rsuma=uniform(0.2, 0, 1, label=rf"$(R_\star + R_{name}) / a_{name}$"),
        cosi=uniform(0.0, 0, 1, label=rf"$\cos{{i_{name}}}$"),
        epoch=normal(epoch, epoch, 0.01, label=rf"$T_{{0;{name}}}$", unit="BJD"),
        period=normal(period, period, 1e-4, label=rf"$P_{name}$", unit="d"),
        f_c=fixed(0.0, "uniform -1 1", label=rf"$\sqrt{{e_{name}}} \cos{{\omega_{name}}}$"),
        f_s=fixed(0.0, "uniform -1 1", label=rf"$\sqrt{{e_{name}}} \sin{{\omega_{name}}}$"),
    )


# --- top-level fit config --------------------------------------------------
@dataclass
class FitConfig:
    """A complete single-fit configuration."""

    target: str
    companions: list[CompanionSpec]
    instruments: list[InstrumentSpec]
    ldc: dict[str, BandLDC] = field(default_factory=dict)  # keyed by canonical band
    share_groups: tuple[tuple[str, ...], ...] = ()
    chromatic: bool | None = None  # None -> engine auto-detects; True/False -> explicit
    use_host_density_prior: bool = True
    fast_fit: bool = True
    fast_fit_width: float = 1.0 / 3.0
    shift_epoch: bool = True
    fit_ttvs: bool = False
    multiprocess: bool = True
    multiprocess_cores: int = 4
    time_format: str = "BJD_TDB"
    mcmc: dict[str, object] = field(default_factory=lambda: dict(DEFAULT_MCMC))
    ns: dict[str, object] = field(default_factory=lambda: dict(DEFAULT_NS))

    def unique_bands(self) -> list[str]:
        """Canonical bands present across all instruments, blue -> red."""
        seen: list[str] = []
        for inst in self.instruments:
            b = _inst.canonical_band(inst.band)
            if b and b not in seen:
                seen.append(b)
        return sorted(seen, key=_inst.band_sort_key)

    def follower_labels(self) -> set[str]:
        """Instrument labels that are share-group followers (leader excluded)."""
        return {f for group in self.share_groups for f in group[1:]}
