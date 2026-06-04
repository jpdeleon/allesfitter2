# allesfitter2 — usage notes & decision guide

Companion to the [README](README.md). The README covers installation, the
`prepare_allesfit` CLI, and feature walk-throughs. This file collects the
advisory material: how to use allesfitter2 effectively, how to choose
`settings.csv` / `params.csv` values for your use case, plus troubleshooting,
parameter-source, and contamination references migrated out of the README so
they stay easy to scan.

**Contents**

- [Effective use of allesfitter2](#effective-use-of-allesfitter2)
- [Choosing settings & priors by use case](#choosing-settings--priors-by-use-case)
- [Prior cookbook](#prior-cookbook)
- [Best practices](#best-practices)
- [Parameter sources & databases](#parameter-sources--databases)
- [Troubleshooting](#troubleshooting)
- [TIC contratio vs SPOC CROWDSAP](#tic-contratio-vs-spoc-crowdsap)

---

## Effective use of allesfitter2

A short mental model that prevents most wasted fits.

**The two files are a contract.** `settings.csv` declares *what* is being
modelled (which companions, which instruments, which baseline/error model per
instrument, achromatic vs chromatic). `params.csv` declares *every free or
fixed number* in that model and its prior. They must agree: every instrument in
`inst_phot`/`inst_rv` needs its per-instrument rows, and every fitted parameter
(`fit=1`) needs finite bounds. The structural validators
(`allesfitter.validation.config_checks`) and the heuristic prior checks
(`allesfitter.validation.prior_sanity`) catch the common mismatches at
`config.init` — read their messages literally; they name the exact key to add
or remove.

**Always look before you fit.** Run `show_initial_guess` first. If the model
curve at the initial guess does not visually track the data, no sampler will
rescue it — fix the ephemeris, depth, or baseline first. This is the single
highest-leverage habit.

**Warm-start, then sample.** For anything non-trivial, run
`allesfitter.optimize()` (L-BFGS-B or CMA-ES) to find a good starting point
*before* launching MCMC/NS. It turns "sampler never converges" into "sampler
converges quickly," and CMA-ES state resumes across calls so you can extend a
search incrementally (see the README "Warm-starting" use case).

**Iterate cheap → expensive.** Start on one clean sector with `fast_fit,True`
and loose convergence to shake out config errors in seconds. Only then move to
multi-sector data, GP baselines, and strict tolerances for the publication run.

**Let the data set GP/noise priors.** Hand-picked GP bounds routinely let the
GP absorb the transit. `prepare_allesfit` derives dataset-aware bounds
(`_dataset_aware_gp_bounds`) — the key invariant is that the GP correlation
length `ρ` is floored *above* the transit duration so the GP cannot eat
ingress/egress. If you edit GP priors by hand, keep that invariant.

**Reproducibility.** Fits are logged centrally to
`~/.allesfitter/runs.jsonl` (`pid`, `hostname`, `datadir`, `run_id`,
`status`, `duration_sec`). Tail it to see what is running where.

---

## Choosing settings & priors by use case

Pick the row that matches your data, then apply the settings and priors in that
column. Keys are written as they appear in `settings.csv` / `params.csv`.

### Quick map

| Use case | Key `settings.csv` | Baseline | Error | Sampler |
|---|---|---|---|---|
| Single clean space-based sector, known ephemeris | `fast_fit,True` | `baseline_flux_<inst>,sample_offset` | `error_flux_<inst>,sample` | NS (evidence + clean posteriors) |
| Space-based with correlated systematics | `fast_fit,True` | `sample_GP_Matern32` | `sample` | NS |
| Ground-based single band, airmass/FWHM trends | — | `sample_linear_multi` (or `hybrid_linear_multi`) | `sample` | NS or MCMC |
| Multi-band ground-based (color info) | `bandpass,<...>` (chromatic) | `sample_GP_Matern32` per inst, or `baseline_share_flux` | `sample` | NS |
| Joint GP across simultaneous instruments | `baseline_share_flux,<insts>` | shared `sample_GP_Matern32` | `sample` | NS |
| TTV / individual transit times | reuse previous-run results | per-transit epochs | `sample` | MCMC |
| Refining an existing solution | `fast_fit,True` | as fitted | `sample` | MCMC after `optimize()` |

### Decision notes

**`fast_fit`** — restricts the model evaluation to windows around transits.
Turn it **on** whenever you only care about the transit (it is much faster and
harmless for clean data). Turn it **off** when the out-of-transit baseline
itself carries signal you are modelling (e.g. phase curves, long GP trends you
want constrained globally).

**Baseline model (`baseline_flux_<inst>`)** — escalate only as far as the data
demand:
- `none` / `sample_offset` — flat or single-offset light curves.
- `sample_linear` — a single linear slope in time.
- `sample_linear_multi` / `hybrid_linear_multi` — detrend against ancillary
  covariates (airmass, FWHM, sky, centroid…). Use the **hybrid** variant when
  the weights are nuisance parameters you want marginalised analytically
  (fewer fit dimensions, faster); use the **sample** variant when you want the
  per-covariate weight posteriors.
- `sample_GP_Matern32` — stochastic correlated noise (stellar granulation,
  ground-based red noise). Most flexible, most prone to swallowing signal — pin
  the priors (below).

**Error model (`error_flux_<inst>`)** — keep `sample` (a free
`ln_err_flux_<inst>` jitter term) as the default; reported error bars are
almost always optimistic. Only fix it when you have a trusted noise model.

**Chromatic vs achromatic** — declare `bandpass` in `settings.csv` only when
bands genuinely differ (ground multi-band, or you want per-band `Rp/R★` to test
for an atmosphere / blends). Then `params.csv` carries one `b_rr_<band>` per
unique bandpass instead of a single `b_rr`. Don't go chromatic for same-band
data — it just adds correlated free parameters.

**Sampler choice** — Nested Sampling (`ns_fit`) is the default: it returns the
Bayesian evidence (model comparison) and handles multimodal posteriors
robustly. Use MCMC (`mcmc_fit`) for fast refinement of an already-good solution
(after `optimize()`), or when you specifically want chain diagnostics.

---

## Prior cookbook

allesfitter prior strings in `params.csv` (`bounds` column): `uniform <lo> <hi>`,
`normal <mu> <sigma>`, `trunc_normal <lo> <hi> <mu> <sigma>`.

| Parameter | Default / discovery prior | Informative prior (when literature constrains) |
|---|---|---|
| `b_rr` (Rp/R★) | `uniform 0 <rr_upper>` (cap ≈ literature ×1.05, ≤0.5) | `trunc_normal 0 0.5 <rr> <rr_err>` |
| `b_rsuma` ((R★+Rp)/a) | `uniform <lo> <hi>` from density | `normal <rsuma> <rsuma_err>` |
| `b_cosi` | `uniform 0 <cosi_max>` (geometry-capped) | `trunc_normal 0 1 <cosi> <err>` |
| `b_epoch` (T0) | `uniform` ±½ window | `normal <T0> <T0_err>` (tight from ephemeris) |
| `b_period` | `normal <P> <P_err>` (usually well known) | fix (`fit=0`) for single-sector |
| `b_f_c`, `b_f_s` | fix at 0 (circular) | `uniform -1 1` only if eccentricity is the goal |
| `dil_<inst>` | fix at 0, or `uniform 0 1` | `normal <1−CROWDSAP> <sigma>` from SPOC (see below) |
| `host_ldc_q1/q2_<inst>` | `uniform 0 1` (Kipping 2013 triangular) | `normal <q> 0.05` from theoretical LDC |
| `ln_err_flux_<inst>` | `uniform` centred on observed RMS, ≤10% flux | — |
| GP `lnsigma` | data-driven, capped below transit depth | — |
| GP `lnrho` | data-driven, floored **above** transit duration | — |

**Rules of thumb**

- **Wide where you're discovering, tight where you know.** Discovery /
  vetting → uniform with physics-capped edges. Confirmed planet refinement →
  Gaussian priors from the catalog values so the sampler spends time where the
  evidence is.
- **Break the `Rp/R★`–`a/R★`–`i` degeneracy with a stellar density prior.**
  Set `use_host_density_prior=True` and provide `params_star.csv` (R★, M★ with
  asymmetric errors). This is the most effective single constraint for transit
  geometry. (First solve per star is ~22 s; cached afterwards — see README
  Performance Notes.)
- **Eccentricity:** default to circular (`b_f_c=b_f_s=0`, `fit=0`) unless you
  have RVs or strong secondary-eclipse timing; freeing `f_c/f_s` on a single
  transit just inflates the depth/duration posterior.
- **Limb darkening:** the `quad` law with `q1/q2 ~ uniform 0 1` is the safe
  default. Fix or Gaussian-constrain to theoretical (`ldtk`/`limbdark`) values
  only for low-S/N transits where LDC is unconstrained by the data. Keep LDC
  keyed correctly (band vs instrument) — see the README "Limb-darkening keying"
  use case.
- **GP priors:** never let the GP correlation length reach the transit
  duration, and cap the GP amplitude below the transit depth. Prefer the
  `prepare_allesfit` data-driven bounds over hand-tuning.

---

## Best practices

### 1. Parameter validation
- Always use `--debug` for first-time targets
- Verify stellar parameters match literature values
- Check transit duration consistency between methods
- Review generated plots before fitting

### 2. Data quality
- Use `pdcsap` over `sap` for SPOC pipeline
- Apply sigma clipping for noisy data: `-sig 3`
- Choose appropriate quality bitmask level
- Inspect lightcurve plots for systematics

### 3. Analysis strategy
- Start with single sector for parameter estimation
- Use multi-sector data for refined parameters
- Enable `fast_fit` for initial exploration
- Use strict convergence (`ns_tol,0.01`) for final results

### 4. Pipeline selection
- **SPOC:** Better systematics correction, slower cadence
- **QLP:** Faster processing, higher cadence available
- Compare both pipelines for robust results

---

## Parameter sources & databases

| Database | Use Case | Parameter Source | Reliability |
|----------|----------|------------------|-------------|
| **NExSci** (`-name`) | Confirmed exoplanets | NASA Exoplanet Archive | Highest |
| **TOI** (`-toi`) | TESS candidates | TFOP database | High |
| **CTOI** (`-ctoi`) | Community candidates | Community observations | Medium |
| **TIC** (`-tic`) | Custom analysis | TIC catalog + manual input | Variable |

---

## Troubleshooting

### Common issues and solutions

**"No light curves found"**
- Verify target name spelling and database availability
- Try alternative identifiers (TIC vs TOI vs star name)
- Use `--debug` to see search results
- Check if target was observed by TESS

**"Multiple exposure times available"**
- Specify exposure time: `-e 120` or `-e 600`
- Use `--debug` to see available options

**"Sector not available"**
- Check available sectors with `-s all`
- Verify target was observed in requested sector

**"Missing stellar parameters"**
- Enable interactive mode: `-i`
- Check target in TIC catalog
- Verify coordinates are correct

**"Parameter derivation failed"**
- Use `--debug` for detailed error messages
- Check database connectivity with `-u`
- Try interactive mode for manual input

**"Chromatic configuration mismatch between settings.csv and params.csv"**
- Your `settings.csv` declares `bandpass,<...>` (chromatic) but `params.csv` still has the achromatic `b_rr` row, or has only some of the expected `b_rr_<bandpass>` rows.
- Replace `b_rr` with one row per unique bandpass: `b_rr_tess`, `b_rr_k2`, etc.
- The error message lists exactly which keys to add and/or remove.

**"params.csv references unknown bandpass(es)"**
- A `b_rr_<suffix>` row uses a suffix that isn't in your `settings.csv` `bandpass` list — usually a typo (`b_rr_tes` vs `tess`).
- Fix the suffix to match one of the labels in `bandpass`, or add a new label to `bandpass`.

**"settings.csv 'bandpass' has N entries but inst_phot has M entries"**
- `bandpass` and `inst_phot` must have the same number of space-separated entries; repeat a label to share a band across instruments.

**`KeyError: 'b_rr'` in chromatic mode**
- Fixed in current main; pull the latest and re-run `show_initial_guess`/`ns_fit`.

**`settings.csv contains per-instrument keys whose suffix is not a known instrument name`**
- A per-instrument settings row (e.g. `host_ld_law_`, `host_grid_`, `baseline_flux_`, `t_exp_`, …) carries a suffix that is **not** in `inst_phot ∪ inst_rv ∪ inst_rv2`. The most common cause is confusing a bandpass label with an instrument name. If the orphan suffix matches a bandpass, the error explicitly hints at the affected instruments — repeat the row once per instrument that uses that bandpass.
- Example: with `inst_phot,tglc120_s90 tglc120_s63s64` and `bandpass,tess tess`, write `host_ld_law_tglc120_s90,quad` and `host_ld_law_tglc120_s63s64,quad`, **not** `host_ld_law_tess,quad`.

**Transit shape looks unchanged when editing `host_ldc_q1/q2`**
- Until v2 the `host_ld_law_<inst>` default was `None`, which silently disabled limb darkening — `ldc_1=None` reached `ellc` and the q1/q2 values had no effect. The current default is `quad`, so this is fixed for new datadirs; for hand-edited configs, ensure `host_ld_law_<inst>,quad` is present per actual instrument. Explicit `host_ld_law_<inst>,none` still opts out cleanly.
- Note that small q1 deltas (~0.03) only move ellc's `u1, u2` by ~1–2%, which is sub-mmag and often invisible at the default plot scale. Use a larger delta (e.g. 0.1 → 0.9) when sanity-checking the LDC pipeline visually.

**Where are my fits running?**
- Tail the centralized run log: `tail -n 10 ~/.allesfitter/runs.jsonl` (or whatever path `$ALLESFITTER_RUN_LOG` resolves to). Each `start` row carries `pid`, `hostname`, absolute `datadir`, and `run_id`; the matching `end` row carries `status` and `duration_sec`. See the "Tracking long-running fits" use case in the README.

### Debug mode

Enable comprehensive diagnostics:
```bash
prepare_allesfit -name "HD 39091" -s 1 --debug
```

Shows:
- Database query results
- Parameter derivation steps
- Intermediate calculations
- Generated file contents
- Error traces

---

## TIC contratio vs SPOC CROWDSAP

Both describe photometric contamination, but they are computed differently and
are not interchangeable when setting `dil_<inst>`.

**TICv8 contratio**

- Source: Stassun et al. 2019 (TIC catalog construction).
- Definition: ratio of contaminant flux to target flux, integrated within an assumed TESS aperture (typically modeled as a 2-pixel radius, ~42″ around the target).
- Computation: catalog-based. Uses Gaia DR2 positions and magnitudes (with empirical T-mag transformations) for all neighbors within a fixed radius, weighted by a model TESS PSF.
- Static: one value per TIC ID, computed once at catalog ingest. Does not depend on which sector/camera/CCD the target lands on.

**SPOC CROWDSAP**

- Source: SPOC pipeline per-sector data products.
- Definition: target flux fraction (target / total) inside the actually-used optimal aperture for that sector. So 1 − CROWDSAP is the dilution fraction within that real aperture.
- Computation: pipeline-derived. Uses the actual PRF model at the target's location on that specific camera/CCD, the actual optimal aperture chosen for that sector (varies sector to sector, especially for FFI vs. 2-min targets), and the TIC catalog as the neighbor list.
- Dynamic: changes sector-to-sector with aperture choice, camera, focus, and pointing.

**Why they differ in practice**

1. Aperture shape and size: TICv8 assumes a fixed circular aperture; SPOC uses the actual irregular optimal aperture that varies per sector. CROWDSAP aperture is usually smaller and aperture-optimised, so dilution tends to be lower than what TICv8 implies.
2. PSF/PRF model: TICv8 uses an idealized analytic PSF; SPOC uses the empirical TESS PRF including focal-plane distortions.
3. Saturation and bleed: SPOC accounts for bleed columns and saturated-pixel masking; TICv8 does not.
4. Neighbor catalog vintage: both use Gaia (DR2 for original TICv8; DR3 in newer revisions), but TICv8 freezes at catalog build time while SPOC re-evaluates per sector.

**Practical guidance.** For `dil_<inst>`, prefer the per-sector SPOC value
(`1 − CROWDSAP`); `prepare_allesfit` extracts it and writes a commented
`normal`-prior `dil_<inst>` row you can activate (see the README "Fitting
dilution" use case). Fall back to TICv8 contratio only when no SPOC product
exists for that sector.
