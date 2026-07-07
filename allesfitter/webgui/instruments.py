"""Instrument / band taxonomy for the allesfitter web GUI.

Ported in spirit from muscat-db's frozen-dataclass ``InstrumentConfig`` registry
and its ``band_utils`` helpers. This module supplies sensible *defaults* — the
bandpass, exposure time, baseline model, limb-darkening law, and covariate
columns — that the fit form pre-populates for a newly added instrument.

The actual instrument *label* in an allesfitter fit (e.g. ``m4g``, ``cpt_z``,
``qlp600``) is user-chosen and encodes site + band; this module only infers
defaults from that label's family and band. Nothing here talks to the engine.
"""

from __future__ import annotations

from dataclasses import dataclass

# --- bands -----------------------------------------------------------------
# Canonical photometric band ordering, blue -> red, then the space missions.
CANONICAL_BANDS: tuple[str, ...] = ("u", "g", "r", "i", "z", "y", "tess", "kepler", "cheops")

# Single-letter ground-based band tokens used when inferring a band from a label.
_BAND_LETTERS: tuple[str, ...] = ("u", "g", "r", "i", "z", "y")

# Map common survey/filter aliases onto a canonical band label.
BAND_ALIASES: dict[str, str] = {
    "up": "u",
    "u'": "u",
    "gp": "g",
    "g'": "g",
    "sg": "g",
    "rp": "r",
    "r'": "r",
    "sr": "r",
    "ip": "i",
    "i'": "i",
    "si": "i",
    "zs": "z",
    "z'": "z",
    "zp": "z",
    "sz": "z",
    "yp": "y",
    "y'": "y",
    "t": "tess",
    "tess": "tess",
    "kep": "kepler",
    "kepler": "kepler",
}


def canonical_band(label: str) -> str:
    """Return the canonical band label for a raw filter/band string."""
    key = label.strip().lower()
    return BAND_ALIASES.get(key, key)


def band_sort_key(band: str) -> tuple[int, str]:
    """Sort key that orders bands blue -> red, unknown bands last (alphabetical)."""
    b = canonical_band(band)
    try:
        return (CANONICAL_BANDS.index(b), b)
    except ValueError:
        return (len(CANONICAL_BANDS), b)


# --- limb darkening + baseline models --------------------------------------
# allesfitter limb-darkening laws (quad is the safe default).
LD_LAWS: tuple[str, ...] = ("quad", "lin", "sqrt", "log", "exp", "power2", "nonlinear")

# Baseline models the config writer can currently emit parameters for. The
# engine supports more (sample_GP_SHO/real/complex, sample_linear_multi); those
# are offered later once their sampled-parameter naming is wired in.
BASELINE_MODELS: tuple[str, ...] = (
    "none",
    "sample_offset",
    "sample_linear",
    "hybrid_offset",
    "hybrid_poly_1",
    "hybrid_poly_2",
    "hybrid_poly_3",
    "hybrid_spline",
    "hybrid_linear_multi",
    "sample_GP_Matern32",
)

# Baseline models that place a celerite GP on the flux (eligible to lead/join a
# baseline_share_flux group).
GP_BASELINES: frozenset[str] = frozenset(
    {"sample_GP_Matern32", "sample_GP_SHO", "sample_GP_real", "sample_GP_complex"}
)

# Baseline models that require a ``baseline_flux_<inst>_cols`` covariate list.
MULTI_BASELINES: frozenset[str] = frozenset({"sample_linear_multi", "hybrid_linear_multi"})


# --- instrument families ---------------------------------------------------
@dataclass(frozen=True)
class InstrumentFamily:
    """Per-family defaults used to pre-populate a newly added instrument."""

    key: str
    display: str
    fixed_band: str | None  # None -> band is inferred from the label (ground multiband)
    default_t_exp: float | None  # exposure time in days, or None to omit
    default_baseline: str
    default_ld_law: str = "quad"
    default_grid: str = "very_sparse"
    covariate_cols: tuple[str, ...] = ()


# MuSCAT-style covariate columns (as found in ``*_cov.csv`` headers).
_MUSCAT_COVS: tuple[str, ...] = ("Airmass", "DX(pix)", "DY(pix)", "FWHM(pix)", "Peak(ADU)")

_SEC = 1.0 / 86400.0  # one second in days

FAMILIES: dict[str, InstrumentFamily] = {
    "tess_spoc": InstrumentFamily(
        "tess_spoc", "TESS SPOC (120s)", "tess", 120 * _SEC, "sample_GP_Matern32"
    ),
    "tess_spoc1800": InstrumentFamily(
        "tess_spoc1800", "TESS SPOC (1800s)", "tess", 1800 * _SEC, "sample_GP_Matern32"
    ),
    "tess_qlp": InstrumentFamily("tess_qlp", "TESS QLP", "tess", 600 * _SEC, "sample_GP_Matern32"),
    "muscat": InstrumentFamily(
        "muscat",
        "MuSCAT / MuSCAT2/3/4",
        None,
        60 * _SEC,
        "sample_GP_Matern32",
        covariate_cols=_MUSCAT_COVS,
    ),
    "lco": InstrumentFamily(
        "lco",
        "LCO / Sinistro",
        None,
        60 * _SEC,
        "sample_linear",
        covariate_cols=("Airmass",),
    ),
    "generic": InstrumentFamily("generic", "Generic", None, None, "sample_linear"),
}

# Label-prefix hints -> family key, longest/most-specific first.
_FAMILY_HINTS: tuple[tuple[str, str], ...] = (
    ("spoc1800", "tess_spoc1800"),
    ("spoc120", "tess_spoc"),
    ("spoc", "tess_spoc"),
    ("qlp", "tess_qlp"),
    ("tess", "tess_spoc"),
    ("muscat", "muscat"),
    ("m2", "muscat"),
    ("m3", "muscat"),
    ("m4", "muscat"),
    ("sinistro", "lco"),
    ("cpt", "lco"),
    ("lsc", "lco"),
    ("coj", "lco"),
    ("elp", "lco"),
    ("ogg", "lco"),
    ("tfn", "lco"),
)


def detect_family(inst_label: str) -> InstrumentFamily:
    """Infer the :class:`InstrumentFamily` from an instrument label."""
    key = inst_label.strip().lower()
    for hint, fam in _FAMILY_HINTS:
        if key.startswith(hint) or hint in key:
            return FAMILIES[fam]
    return FAMILIES["generic"]


def suggest_band(inst_label: str) -> str:
    """Best-effort canonical band for an instrument label.

    TESS families resolve to the fixed ``tess`` band; ground-based multiband
    labels (``m4g``, ``cpt_z``, ``lsc_i2``) resolve to the trailing band letter
    (ignoring a repeat-night digit suffix). Returns ``""`` when no band is found.
    """
    fam = detect_family(inst_label)
    if fam.fixed_band:
        return fam.fixed_band
    stem = inst_label.strip().lower().rstrip("0123456789")
    for ch in reversed(stem):
        if ch in _BAND_LETTERS:
            return ch
    return ""


def suggest_defaults(inst_label: str) -> dict:
    """Suggested per-instrument defaults for the fit form (band, t_exp, ...)."""
    fam = detect_family(inst_label)
    return {
        "family": fam.key,
        "band": suggest_band(inst_label),
        "t_exp": fam.default_t_exp,
        "baseline": fam.default_baseline,
        "ld_law": fam.default_ld_law,
        "grid": fam.default_grid,
        "covariate_cols": list(fam.covariate_cols),
    }
