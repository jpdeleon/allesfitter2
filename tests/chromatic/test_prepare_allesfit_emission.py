"""Regression: settings.csv/params.csv shapes emitted by prepare_allesfit.py
must satisfy the new validators (orphan-suffix + default-quad LD law).

We don't invoke the script (it pulls lightkurve / does network I/O). Instead
we synthesize the exact text shape it emits in three scenarios:

  1. No --bp (achromatic): per-inst LDC suffixes, no `bandpass` row.
  2. --bp tess kepler (chromatic, distinct bandpasses == inst names).
  3. --bp tess kepler with distinct inst names (`tess_pdcsap`, `k2sff`).
  4. --bp tess tess (shared bandpass, two instruments).

Each scenario must:
  - parse via config.init without raising
  - have chromatic flag set correctly
  - have ldc_suffix resolution match expectations
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

from allesfitter import config

from tests.chromatic._helpers import (
    NOISE_SIGMA, N_POINTS_PER_INST, RNG_SEED, TRUE_EPOCH, TRUE_PERIOD,
    TRUE_RR_TESS, phase_sampled_time, simulate_lightcurve, write_data_csv,
)


def _emit_settings(datadir, inst_phot, bandpass=None):
    """Mirror the per-instrument settings rows that prepare_allesfit.py
    writes when called with -f <inst_phot> [-bp <bandpass>]."""
    lines = [
        "#name,value",
        f"companions_phot,b",
        f"companions_rv,",
        f"inst_phot,{' '.join(inst_phot)}",
        f"inst_rv,",
        "time_format,BJD_TDB",
        "multiprocess,False",
        "fast_fit,False",
        "secondary_eclipse,False",
        "phase_curve,False",
        "shift_epoch,False",
        "inst_for_b_epoch,all",
        "mcmc_nwalkers,20",
        "mcmc_total_steps,200",
        "mcmc_burn_steps,100",
        "mcmc_thin_by,1",
        "ns_modus,static",
        "ns_nlive,50",
        "ns_bound,single",
        "ns_sample,unif",
        "ns_tol,1.0",
        "fit_ttvs,False",
        "use_host_density_prior,False",
        "use_tidal_eccentricity_prior,False",
    ]
    if bandpass is not None:
        lines.append(f"bandpass,{bandpass}")
    # The crucial per-instrument keys prepare_allesfit.py emits:
    for inst in inst_phot:
        lines.append(f"host_ld_law_{inst},quad")
        lines.append(f"host_grid_{inst},very_sparse")
        lines.append(f"t_exp_{inst},0.0014")          # 120s in days
        lines.append(f"baseline_flux_{inst},sample_GP_Matern32")
        lines.append(f"error_flux_{inst},sample")
    (datadir / "settings.csv").write_text("\n".join(lines) + "\n")


def _emit_params(datadir, ldc_suffixes, rr_suffixes, inst_phot):
    """Mirror the params.csv shape prepare_allesfit.py writes — keys map to
    ``ldc_suffixes`` (bandpass when chromatic, inst otherwise) for LDC and
    ``rr_suffixes`` for the per-band rr."""
    header = "#name,value,fit,bounds,label,unit,coupled_with"
    rows = [header]
    # rr (per-bandpass when chromatic; single achromatic key otherwise)
    if not rr_suffixes:
        rows.append("b_rr,0.1,1,uniform 0 0.3,rr,,")
    else:
        for bp in rr_suffixes:
            rows.append(f"b_rr_{bp},0.1,1,uniform 0 0.3,rr_{bp},,")
    rows.extend([
        "b_rsuma,0.12,1,uniform 0.01 0.5,rsuma,,",
        "b_cosi,0.01,1,uniform 0 1,cosi,,",
        f"b_epoch,{TRUE_EPOCH},1,uniform {TRUE_EPOCH - 0.05} {TRUE_EPOCH + 0.05},epoch,BJD,",
        f"b_period,{TRUE_PERIOD},1,uniform {TRUE_PERIOD - 0.01} {TRUE_PERIOD + 0.01},period,d,",
        "b_f_c,0,0,uniform -1 1,f_c,,",
        "b_f_s,0,0,uniform -1 1,f_s,,",
    ])
    for inst in inst_phot:
        rows.append(f"dil_{inst},0,0,uniform -1 1,dil_{inst},,")
    # LDCs keyed by ldc_suffixes (bandpass in chromatic, inst otherwise)
    for s in ldc_suffixes:
        rows.append(f"host_ldc_q1_{s},0.5,1,uniform 0 1,q1_{s},,")
        rows.append(f"host_ldc_q2_{s},0.5,1,uniform 0 1,q2_{s},,")
    for inst in inst_phot:
        rows.append(f"ln_err_flux_{inst},-7,0,uniform -15 0,ln_err_{inst},,")
        rows.append(f"baseline_offset_flux_{inst},0,0,uniform -0.05 0.05,offset_{inst},,")
    (datadir / "params.csv").write_text("\n".join(rows) + "\n")


def _write_data(datadir, inst_phot, rng_seed):
    rng = np.random.default_rng(rng_seed)
    for inst in inst_phot:
        t = phase_sampled_time(N_POINTS_PER_INST, TRUE_PERIOD, TRUE_EPOCH, rng=rng)
        f, e = simulate_lightcurve(t, TRUE_RR_TESS, NOISE_SIGMA, rng)
        write_data_csv(datadir / f"{inst}.csv", t, f, e)


def _build(tmp_path, name, inst_phot, bandpass, ldc_suffixes, rr_suffixes,
           rng_seed):
    d = tmp_path / name
    d.mkdir()
    _write_data(d, inst_phot, rng_seed)
    _emit_settings(d, inst_phot, bandpass=bandpass)
    _emit_params(d, ldc_suffixes=ldc_suffixes, rr_suffixes=rr_suffixes,
                 inst_phot=inst_phot)
    return d


class TestPrepareAllesfitEmission:
    def test_achromatic_no_bp_flag(self, tmp_path):
        d = _build(tmp_path, "achr", inst_phot=["tess"], bandpass=None,
                   ldc_suffixes=["tess"], rr_suffixes=[],
                   rng_seed=RNG_SEED + 110)
        config.init(str(d), quiet=True)
        b = config.BASEMENT
        assert b.settings["chromatic"] is False
        assert b.settings["bandpass"] == {}
        assert b.settings["host_ld_law_tess"] == "quad"
        # LDC must be keyed by inst (no bandpass dict).
        assert "host_ldc_q1_tess" in b.params

    def test_chromatic_distinct_bandpasses_inst_names_match(self, tmp_path):
        d = _build(tmp_path, "chrom_match",
                   inst_phot=["tess", "kepler"], bandpass="tess kepler",
                   ldc_suffixes=["tess", "kepler"],
                   rr_suffixes=["tess", "kepler"],
                   rng_seed=RNG_SEED + 111)
        config.init(str(d), quiet=True)
        b = config.BASEMENT
        assert b.settings["chromatic"] is True
        assert b.settings["bandpass"] == {"tess": "tess", "kepler": "kepler"}
        assert b.settings["host_ld_law_tess"] == "quad"
        assert b.settings["host_ld_law_kepler"] == "quad"
        assert "b_rr_tess" in b.params
        assert "b_rr_kepler" in b.params
        assert "host_ldc_q1_tess" in b.params
        assert "host_ldc_q1_kepler" in b.params

    def test_chromatic_distinct_inst_names_vs_bandpasses(self, tmp_path):
        d = _build(tmp_path, "chrom_distinct",
                   inst_phot=["tess_pdcsap", "k2sff"],
                   bandpass="tess k2",
                   ldc_suffixes=["tess", "k2"],
                   rr_suffixes=["tess", "k2"],
                   rng_seed=RNG_SEED + 112)
        config.init(str(d), quiet=True)
        b = config.BASEMENT
        assert b.settings["chromatic"] is True
        assert b.settings["bandpass"] == {
            "tess_pdcsap": "tess", "k2sff": "k2"}
        # Per-instrument settings keyed by actual inst names — this is the
        # case that, if mis-emitted, would trigger the orphan-suffix
        # validator.
        assert b.settings["host_ld_law_tess_pdcsap"] == "quad"
        assert b.settings["host_ld_law_k2sff"] == "quad"
        # LDC scalars keyed by bandpass labels.
        assert "host_ldc_q1_tess" in b.params
        assert "host_ldc_q1_k2" in b.params

    def test_shared_bandpass_two_insts(self, tmp_path):
        # The user-reported scenario shape: bandpass='tess tess' with two
        # distinct instruments sharing one bandpass. chromatic=False because
        # only one unique label, but LDC keys are bandpass-suffixed.
        d = _build(tmp_path, "shared_bp",
                   inst_phot=["tglc120_s90", "tglc120_s63s64"],
                   bandpass="tess tess",
                   ldc_suffixes=["tess"],
                   rr_suffixes=["tess"],
                   rng_seed=RNG_SEED + 113)
        config.init(str(d), quiet=True)
        b = config.BASEMENT
        assert b.settings["chromatic"] is False
        assert set(b.settings["bandpass"].values()) == {"tess"}
        for inst in ("tglc120_s90", "tglc120_s63s64"):
            assert b.settings[f"host_ld_law_{inst}"] == "quad"
        # Single bandpass-suffixed LDC, shared by both instruments.
        assert "host_ldc_q1_tess" in b.params
        # Both instruments resolve to the same ldc_suffix='_tess' via
        # get_bandpass(); the parser validator already pins this.
        assert b.get_ldc_key("host", 1, "tglc120_s90") == "host_ldc_u1_tess"
        assert b.get_ldc_key("host", 1, "tglc120_s63s64") == "host_ldc_u1_tess"
