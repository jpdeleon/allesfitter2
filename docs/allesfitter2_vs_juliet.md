# Transit Fitting: allesfitter2 vs. juliet

A capability comparison of the two transit-fitting packages as installed in
`ext_tools/`, focused on **how their common features differ**. Both packages
solve the same core problem — Bayesian inference of transit (and joint
RV/eclipse/phase-curve) parameters from photometric time series — but they make
different choices in transit model, parametrization, noise handling, and
sampling.

| | **allesfitter2** | **juliet** |
|---|---|---|
| Location | `ext_tools/allesfitter2/` | `ext_tools/juliet/` |
| Upstream | fork `jpdeleon/allesfitter2` | `nespinozi/juliet` (official) |
| Transit engine | **ellc** | **batman** (+ optional catwoman) |
| Config style | CSV files (`params.csv`, `settings.csv`) on disk | Python `priors` dict or `.dat` priors file |
| Primary samplers | **emcee** (MCMC) + **dynesty** (nested) | **dynesty** / **ultranest** / **MultiNest** (nested), emcee/zeus (MCMC) |
| Philosophy | "project directory" workflow + CLI prep tools | lightweight library, scripted in Python |

---

## 1. How the same fit is set up

This is the single biggest day-to-day difference.

**allesfitter2** is **file/directory driven**. A fit lives in a folder
containing per-instrument light-curve CSVs plus two control files:

- `params.csv` — one row per parameter: `name, value, fit, bounds, label, unit, coupled_with`
- `settings.csv` — model + sampler configuration (companions, baselines, sampler steps)

You then call, from `run.py`:

```python
import allesfitter
allesfitter.show_initial_guess('.')
allesfitter.ns_fit('.'); allesfitter.ns_output('.')      # nested sampling
# or
allesfitter.mcmc_fit('.'); allesfitter.mcmc_output('.')  # emcee
```

A CLI helper `prepare_allesfit` can auto-generate the whole directory from
TESS/K2/Kepler data.

**juliet** is **API/dict driven**. There is no project directory; you pass data
arrays and a priors dictionary directly:

```python
import juliet
dataset = juliet.load(priors=priors, t_lc=t, y_lc=f, yerr_lc=ferr,
                      out_folder='myfit')
results = dataset.fit(sampler='dynesty')
```

Priors can equivalently come from a `priors.dat` text file
(`name  Distribution  hyperparameters`).

> **Net difference:** allesfitter2 trades flexibility for a reproducible,
> inspectable on-disk project (good for archiving a fit, CLI prep, multi-band
> bookkeeping). juliet trades structure for programmatic control (good for
> scripting many targets, injection/recovery loops, notebooks).

---

## 2. Transit model and the parameters you actually fit

Both fit the same physics but **name and parametrize the geometry
differently** — the most important thing to know when porting a fit between them.

| Quantity | allesfitter2 (`params.csv`) | juliet (priors) |
|---|---|---|
| Radius ratio Rp/Rs | `b_rr` | `p_p1` (or via `r1_p1`,`r2_p1`) |
| Orbit scale | `b_rsuma` = (R★+Rp)/a | `a_p1` = a/R★, **or** global `rho` |
| Inclination | `b_cosi` = cos i | `b_p1` (impact param), or via `r1`/`r2` |
| Epoch | `b_epoch` (T0, BJD) | `t0_p1` |
| Period | `b_period` | `P_p1` |
| Eccentricity | `b_f_c`=√e·cos ω, `b_f_s`=√e·sin ω | `ecc_p1`+`omega_p1`, or `sesinomega`/`secosomega` |
| RV semi-amplitude | `b_K` | `K_p1` |

Key structural differences:

- **Companion prefix vs. planet suffix.** allesfitter2 keys every parameter to
  a companion letter (`b_`, `c_`, …); juliet uses `_p1`, `_p2`, … suffixes.
- **Impact parameter.** juliet fits `b` (or the Espinoza 2018 `r1`/`r2`
  reparametrization that samples uniformly over the physically allowed (b, p)
  plane). allesfitter2 instead fits `cos i` together with `(R★+Rp)/a`.
- **Stellar density.** juliet can fit a single `rho` and derive `a/R★` per
  planet via Kepler's third law (efficient for multi-planet systems sharing a
  star). allesfitter2 parametrizes the scaled sum of radii `(R★+Rp)/a` per
  companion; a stellar-density prior can be applied but is not the native knob.
- **Eccentricity.** Both default to the numerically safe √e form
  (allesfitter2 `b_f_c`/`b_f_s`; juliet `secosomega`/`sesinomega`), and both
  allow a fixed circular orbit.

---

## 3. Limb darkening — same Kipping parametrization, different defaults

Both adopt the **Kipping (2013) `q1`/`q2`** triangular sampling for the
quadratic law (uniform 0–1, guarantees physical coefficients).

- **allesfitter2:** per-instrument `host_ldc_q1_<inst>`, `host_ldc_q2_<inst>`;
  law set via `host_ld_law_<inst>` (default `quad`; also `none`, etc.).
  Theoretical Claret coefficients via the `limbdark2` helper.
- **juliet:** per-instrument `q1_<inst>`, `q2_<inst>`; law chosen in
  `juliet.load(..., ld_laws='TESS-quadratic,K2-logarithmic,...')`. Supports
  **linear** (single `q1`), **quadratic** (default), and **logarithmic**.
  LD parameters can be **shared** across instruments with a combined key,
  e.g. `q1_TESS_K2`.

> **Net difference:** juliet exposes log/linear/quadratic laws and easy
> cross-instrument LD sharing in one call; allesfitter2 sets the law per
> instrument in `settings.csv` and is tied to ellc's supported laws.

---

## 4. Detrending / noise — both GP-capable, different toolkits

Both model correlated noise + a white jitter term, but with **different GP
backends and configuration surfaces**.

**allesfitter2** (baseline configured in `settings.csv`, `baseline_<key>_<inst>`):

- White jitter per instrument.
- Deterministic baselines: offset, linear, polynomial (`hybrid_poly_N`),
  spline (`hybrid_spline`).
- GP baselines: `sample_GP_Matern32`, `sample_GP_SHO`, `sample_GP_real`,
  `sample_GP_complex`.
- **Ancillary covariate regression** — per-instrument CSV columns (airmass,
  FWHM, sky…) detrended via `baseline_<key>_<inst>_against,<name>`, plus
  multi-covariate linear (`sample_linear_multi`).
- **Shared GP across instruments**
  (`baseline_share_flux,muscat_g:muscat_r:...`) for simultaneous multiband
  photometry.

**juliet** (GP configured via priors + `load()`):

- White jitter `sigma_w_<inst>`, mean flux `mflux_<inst>`, dilution
  `mdilution_<inst>`.
- GP via **george** and **celerite**: exp-squared, Matérn-3/2 & 5/2, SHO,
  stellar-rotation (quasi-periodic) kernels.
- GP regressors supplied as arrays/files; hyperparameters scoped as
  `<pname>_<inst>` (per instrument), `<pname>_lc` / `<pname>_rv` (global).

> **Net difference:** juliet's strength is its kernel library (notably the
> celerite quasi-periodic rotation kernel for stellar activity).
> allesfitter2's strength is **instrument-aware detrending** — named ancillary
> covariates and a GP shared across bandpasses, designed for ground-based
> multiband campaigns.

---

## 5. Multiband / chromatic transits

- **allesfitter2** natively fits a **chromatic** radius ratio: `b_rr_<bandpass>`
  per band while orbital parameters stay globally shared — purpose-built for
  MuSCAT-style simultaneous multiband data, with shared-GP support across bands.
  (See `docs/chromatic_validation.md` in this same folder.)
- **juliet** handles multiple instruments and can share/separate parameters via
  key naming, but band-dependent Rp/Rs is expressed by defining separate `p`
  parameters per instrument group rather than a dedicated chromatic mode.

---

## 6. Sampling and inference

| | allesfitter2 | juliet |
|---|---|---|
| Nested sampling | **dynesty** (`ns_modus` static/dynamic, `ns_nlive`, `ns_bound`, `ns_sample`, `ns_tol`) | **dynesty** (default), **ultranest**, **PyMultiNest** (fallback) |
| MCMC | **emcee** (`mcmc_nwalkers`, `mcmc_total_steps`, `mcmc_burn_steps`, `mcmc_thin_by`) | emcee, zeus |
| Optimization | CMA-ES warm start, dual annealing, differential evolution, L-BFGS-B, Powell | — (sampler-only) |
| Selection | choose `mcmc_fit` vs `ns_fit` | `dataset.fit(sampler=...)` |

> **Net difference:** allesfitter2 ships first-class **MCMC + global
> optimizers** alongside nested sampling and exposes both via dedicated
> functions. juliet is **nested-sampling-first** (evidence `Z` for model
> comparison out of the box) with several interchangeable backends, MCMC
> available but secondary.

---

## 7. Beyond the transit (shared extras)

| Feature | allesfitter2 | juliet |
|---|---|---|
| Secondary eclipse | `secondary_eclipse, True` | `fp_p1`, batman `transittype='secondary'` |
| Phase curves | `sine_series`, `sine_physical`, `ellc_physical`, `GP` | Cowan & Agol (2013) |
| TTVs | `prepare_ttv_fit()` | per-planet transit-timing support |
| Joint RV | companion `b_K` + RV instruments | `K_p1`, `mu_<inst>`, `rv_slope/quad` (radvel) |
| Dilution | `dil_<inst>` | `mdilution_<inst>` |

Both do joint photometry + RV and multi-planet/companion systems; the
mechanics again differ only in config surface (CSV rows vs. priors keys).

---

## 8. When to prefer which (practical guidance)

- **Ground-based multiband (MuSCAT, etc.):** allesfitter2 — chromatic Rp/Rs,
  ancillary-covariate detrending, shared GP across bands.
- **Stellar activity / quasi-periodic noise:** juliet — celerite rotation
  kernel and a richer GP kernel set.
- **Reproducible, archivable project + CLI prep from TESS/K2:** allesfitter2 —
  `prepare_allesfit`, on-disk `params.csv`/`settings.csv`, run log.
- **Scripted batch fitting / injection-recovery / notebooks:** juliet —
  pure-Python `load()`/`fit()`, `generate_priors()`.
- **Model comparison by evidence:** juliet — nested-sampling-first with clean
  `Z` output (allesfitter2 also gives `Z` via dynesty).
- **Want both MCMC and global optimizers in one tool:** allesfitter2.

---

## 9. Caveats

- This compares **declared/observed capabilities** from the code and docs in
  `ext_tools`; it is **not** a numerical benchmark. The two packages are
  installed and used **independently** here — there is no shared wrapper or
  head-to-head script in the repo.
- A true equivalence test (fit the *same* light curve with both and compare
  posteriors) would need a common dataset and careful prior matching across the
  differing parametrizations (§2). That is recommended as a follow-up if
  numerical agreement matters.

---

### Source files referenced

- allesfitter2: `ext_tools/allesfitter2/README.md`, `allesfitter/` core modules,
  `examples/*/run.py`, `examples/*/params.csv`, `examples/*/settings.csv`,
  `docs/chromatic_validation.md`.
- juliet: `ext_tools/juliet/juliet/fit.py`, `docs/user/priorsnparameters.rst`,
  `docs/tutorials/{transitfits,jointfits,gps}.rst`, `tests/test_transit.py`,
  `tutorial1/priors/K2-140_priors_fullcircular.dat`.
