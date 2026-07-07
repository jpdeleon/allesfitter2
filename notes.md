# allesfitter2 — modeling reference, usage notes & decision guide

Companion to the [README](README.md). The README is the high-level entry point:
installation, the `prepare_allesfit` CLI, and a scannable **Modeling capabilities**
overview. This file holds the depth — the per-feature modeling walkthroughs plus the
advisory material (how to choose `settings.csv` / `params.csv` values, prior cookbook,
best practices, parameter sources, troubleshooting) — so the README stays short and
this stays the place you grep when you actually configure a fit.

**Contents**

Modeling capabilities — detailed walkthroughs

- [Chromatic transit modeling (per-band Rp/Rs)](#chromatic-transit-modeling-per-band-rprs)
- [Sharing a baseline GP across instruments](#sharing-a-baseline-gp-across-instruments-joint-celerite-realization)
- [Detrending baselines against ancillary covariates](#detrending-baselines-against-ancillary-covariates-airmass-fwhm-sky-)
- [N-D linear baseline detrending](#n-d-linear-baseline-detrending-sample_linear_multi--hybrid_linear_multi)
- [Warm-starting MCMC / NS with `optimize()`](#warm-starting-mcmc--ns-with-allesfitteroptimize)
- [Fitting dilution (`dil_<inst>`)](#fitting-dilution-dil_inst-in-a-chromatic-model)
- [Limb-darkening keying (band vs instrument)](#limb-darkening-keying-band-vs-instrument)
- [OOM-safe diagnostic plots for high-dim fits](#oom-safe-diagnostic-plots-for-high-dim-fits)
- [Tracking long-running fits (run log)](#tracking-long-running-fits-with-the-centralized-run-log)
- [Raw-flux outlier clipping](#raw-flux-outlier-clipping)
- [Lightcurves from different pipelines](#lightcurves-from-different-pipelines)
- [Lightcurves from different exposure times](#lightcurves-from-different-exposure-times)
- [Using a previous run for a TTV fit](#using-a-previous-run-for-a-ttv-fit)
- [Performance & caching (`simulate_PDF` disk cache)](#performance--caching-simulate_pdf-disk-cache)

Usage & advisory

- [Effective use of allesfitter2](#effective-use-of-allesfitter2)
- [Choosing settings & priors by use case](#choosing-settings--priors-by-use-case)
- [Prior cookbook](#prior-cookbook)
- [Best practices](#best-practices)
- [Parameter sources & databases](#parameter-sources--databases)
- [Troubleshooting](#troubleshooting)
- [TIC contratio vs SPOC CROWDSAP](#tic-contratio-vs-spoc-crowdsap)

---

# Modeling capabilities — detailed walkthroughs

These sections were migrated out of the README so it stays a scannable overview. Each
corresponds to a row in the README's [Modeling capabilities](README.md#modeling-capabilities)
table.

## Chromatic transit modeling (per-band Rp/Rs)

Fit a separate radius ratio per bandpass while keeping `period`, `epoch`, `cosi`, `rsuma`, and eccentricity globally shared. Useful when combining TESS+Kepler, multiple ground-based filters, or cross-mission archival data.

```bash
prepare_allesfit -name "K2-237" -m k2 -c all -p everest -f tess k2 -bp tess k2
```

This emits a `bandpass,tess k2` row in `settings.csv` and per-bandpass rows in `params.csv`:

```csv
# settings.csv
inst_phot,tess k2
bandpass,tess k2

# params.csv
b_rr_tess,0.0823,1,uniform 0 0.3,$R_b / R_\star (\mathrm{tess})$,,
b_rr_k2,  0.0801,1,uniform 0 0.3,$R_b / R_\star (\mathrm{k2})$,,
host_ldc_q1_tess, ...
host_ldc_q1_k2,   ...
```

Chromatic mode activates when there are **≥ 2 unique** bandpass labels. The parser then requires:

| Parameter family | Scope |
|---|---|
| `b_period`, `b_epoch`, `b_cosi`, `b_rsuma`, `b_f_c`, `b_f_s`, `b_K` | **Global** — single-keyed |
| `b_rr_<bandpass>` | **Per bandpass** — one row per unique band |
| `host_ldc_q1/q2_<bandpass>`, `host_ldc_u*_<bandpass>` | **Per bandpass** |
| `dil_<inst>`, `baseline_*_flux_<inst>`, `ln_err_flux_<inst>`, `b_sbratio_<inst>` | **Per instrument** |

The validator raises clearly on common mistakes:
- `bandpass` entry count ≠ `inst_phot` entry count
- Duplicate parameter rows in `params.csv`
- Unknown bandpass suffix (e.g. typo `b_rr_tes` vs `tess`)
- Chromatic `settings.csv` paired with achromatic `params.csv` (only `b_rr` row)
- Mixed shape (`b_rr` + partial `b_rr_<bp>` rows)

To **share** a single bandpass across multiple instruments (e.g. two TESS pipelines), repeat the label:
```bash
prepare_allesfit -name "HD 39091" -s 1 -f tess_pdcsap tess_qlp -bp tess tess
```
This stays `chromatic=False` (single unique band) but ties both instruments to one `b_rr_tess` and one set of bandpass-suffixed LDCs.

If `-bp` is **omitted** with multiple distinct `-f` instruments, the script warns and falls back to an achromatic `params.csv` for backward compatibility.

To force achromatic mode even when `bandpass` lists multiple labels (e.g. to fit a low-S/N target with one shared `b_rr` across MuSCAT g/r/i/z), set `chromatic,False` explicitly in `settings.csv`. The `bandpass` row stays useful for plot labels and per-band LDC priors; the validator then requires `b_rr` in `params.csv` and rejects per-band `b_rr_<bp>` rows with an actionable message. The auto-detection rule (`len(unique_bandpasses) > 1`) only fires when `chromatic` is absent from `settings.csv`, so omitting it preserves the legacy behaviour bit-for-bit.

After `ns_fit` + `ns_output`, the per-bandpass `Rp/Rs` posteriors are overlaid in `ns_chromatic_rr_<companion>.pdf` — useful for spotting wavelength-dependent depth differences (atmospheric features, spot contamination, dilution mismatch) at a glance. When `params_star.csv` is present, a second panel below the histograms shows the implied planet radius posterior with twin x-axes (`R⊕` bottom, `R_Jup` top) — R★ uncertainty is propagated from the asymmetric normal in `params_star.csv`.

## Sharing a baseline GP across instruments (joint celerite realization)

For simultaneous multi-band photometry (MuSCAT g/r/i/z, MuSCAT2/3/4, LCO MuSCAT clones, …) the dominant noise — airmass, seeing, atmospheric transparency — is *common-mode* across bands. The natural model is **one** celerite GP whose realization modulates every band at the shared timestamps, not N independent GPs.

`coupled_with` in `params.csv` can only alias scalar parameters; it cannot tie together a stochastic GP realization. The `baseline_share_<key>` setting solves that at the instrument-group level:

```csv
# settings.csv
inst_phot,muscat_g muscat_r muscat_i muscat_z
baseline_flux_muscat_g,sample_GP_Matern32
baseline_share_flux,muscat_g:muscat_r:muscat_i:muscat_z
```

- Space-separated **groups**, colon-separated **members**. The first member is the **leader** and owns the sampled GP hyperparameters.
- Followers inherit the leader's `baseline_<key>_<inst>` type and `_against` setting automatically — leave their entries blank.
- Only the leader needs the GP hyperparameter rows in `params.csv`:

  ```csv
  baseline_gp_matern32_lnsigma_flux_muscat_g,-5,1,uniform -10 -3,...,,
  baseline_gp_matern32_lnrho_flux_muscat_g, 0,1,uniform  -1  3,...,,
  ```

  Follower rows for `muscat_r/i/z` are not required (and if present with `fit=1` and no `coupled_with`, the loader refuses to start).
- At likelihood time, residuals from all group members are concatenated, sorted, and fed to a **single** celerite GP under the leader's name (`calculate_lnlike_total` Case 2b in `computer.py`). Predictive plotting (`baseline_sample_GP`) returns per-band slices of the same joint draw.
- Deterministic baseline components (`sample_offset`, `sample_linear`, `hybrid_*`) remain per-band — only the GP is shared.

Multiple groups in one file are supported:

```csv
baseline_share_flux,m1_g:m1_r:m1_i:m1_z m2_g:m2_r:m2_i:m2_z
```

Symmetric keys `baseline_share_rv` / `baseline_share_rv2` exist for RV parity. Backward compatibility: omitting `baseline_share_<key>` produces identical per-instrument-GP behaviour to earlier releases (regression-tested in `tests/test_share_baseline.py::test_single_member_group_matches_legacy_lnlike`).

The loader enforces the following consistency checks at startup so cross-file mistakes surface immediately, not deep inside the first likelihood evaluation:

| Trigger | Outcome |
|---|---|
| Leader or follower not listed in `inst_<key2>` | `ValueError` |
| Inst appears in more than one share group, or as both leader and follower | `ValueError` |
| Duplicate members within a single group (e.g. `a:a:b`) | `ValueError` |
| Leader's `baseline_<key>_<leader>` is not one of `sample_GP_Matern32 / _SHO / _real / _complex` | `ValueError` |
| Follower's `baseline_<key>_<follower>` is set and differs from the leader's kernel | `ValueError` |
| Leader's `baseline_<key>_<leader>_against` is not `time` | `ValueError` |
| Follower explicitly sets `baseline_<key>_<follower>_against` to something other than `time` | `ValueError` |
| `params.csv` is missing a required hyperparameter row for the leader's declared kernel (e.g. `baseline_gp_matern32_lnsigma_flux_<leader>`) | `ValueError` |
| Follower has its own `fit=1` GP hyper row without a `coupled_with` | `ValueError` |
| Follower has `coupled_with=X` where `X` is not the corresponding leader row | `ValueError` |
| Singleton share group (single member, nothing shared) | `UserWarning` |

## Detrending baselines against ancillary covariates (airmass, FWHM, sky, …)

Per-instrument CSV files can carry **named ancillary columns** beyond the required `time, flux, flux_err`. Add a `#`-prefixed header on the first non-blank line listing every column:

```csv
#time,flux,flux_err,airmass,fwhm,sky
2459123.45123,1.00012,0.00118,1.23,1.45,250.0
2459123.45289,0.99984,0.00121,1.22,1.46,251.1
...
```

Then point a baseline at a named covariate in `settings.csv`:

```
baseline_flux_lco,hybrid_poly_2
baseline_flux_lco_against,airmass
```

This works with every existing baseline kernel — `sample_offset`, `sample_linear`, `hybrid_poly_<N>`, `hybrid_spline`, `sample_GP_Matern32`, `sample_GP_SHO`, `sample_GP_real`, `sample_GP_complex` — because the dispatch goes through a single regression-axis lookup. The GP path automatically sorts the abscissa and breaks tied values (celerite needs strictly-sorted x); the predictive plotting path reuses the same sorted GP draw.

This is the same covariate-decorrelation approach the transit literature applies to **TESS** data (centroid `mom_centr1/2`, `pos_corr1/2`, `sap_bkg`, PSF FWHM, CBVs) as well as ground-based photometry — the machinery is instrument-agnostic; only the covariate columns differ. See the investigation in `~/.claude/plans/` for the per-pipeline covariate availability table.

Backward compatibility (no migration required for existing fits):

| CSV layout | Behaviour |
|---|---|
| `#time,flux,flux_err,<name1>,<name2>,...` (headered) | `data[inst]['covariates']` keyed by name; first ancillary column is also aliased to the legacy `custom_series` slot |
| 4 positional columns, no header | Legacy: column 4 → `custom_series`; `covariates` dict empty |
| 3 positional columns, no header | Legacy: `custom_series` = zeros; `covariates` dict empty |

The loader validates after-the-fact: every `baseline_<key>_<inst>_against=<name>` setting that names something other than `time` / `custom_series` must resolve to a real column in the corresponding CSV, or `Basement(...)` refuses to start with the actionable error `<inst>.csv has no column named '<name>'. Known options: ...`.

Limitations to be aware of:

- **GPs are 1-D** (celerite). `sample_GP_*` with `_against=<covariate>` fits a 1-D GP regressing residuals against the covariate value; multi-dimensional GPs would require switching to `george` or `tinygp` — not in scope.
- **Share-baseline groups must use `_against=time`**. The joint celerite GP across `baseline_share_flux,m4g:m4r:m4i:m4z` is a shared *realization* on the shared time grid; covariate-based regression is only meaningful per inst. The loader enforces this constraint.
- For true multi-covariate detrending against `airmass + fwhm + sky` simultaneously, use the `*_linear_multi` baselines below rather than a single `_against` axis.

## N-D linear baseline detrending (`sample_linear_multi` / `hybrid_linear_multi`)

Declare a joint linear model in any number of ancillary covariates:

```
baseline_flux_<inst>,sample_linear_multi
baseline_flux_<inst>_cols,Airmass FWHM(pix) bias
```

- `_cols` is a space-separated list of covariate column names; the special token `bias` injects a column of ones (intercept). Covariates are auto-standardized to zero mean / unit variance before the fit (`basement.py` `_build_linear_design_matrix`).
- The `sample_*` variant samples each weight as a free fit parameter (you get the per-covariate weight posteriors).
- The `hybrid_*` variant **analytically marginalises** the Gaussian weights in closed form at every likelihood call — zero added fit dimensions, while still recovering the optimal per-evaluation MAP weights for predictive plotting. Prefer it when the weights are nuisances you don't want to sample.

This is the multi-covariate generalisation of the single-axis `_against` detrending above, and the recommended way to decorrelate TESS/ground-based flux against several instrumental covariates at once.

## Warm-starting MCMC / NS with `allesfitter.optimize()`

The user's `params.csv` initial values + a stochastic emcee pre-run only get you so far. For 20+ dimensional fits with GP baselines (often multi-modal in `lnsigma`/`lnrho`) it is usually worth running a proper global optimizer **before** `mcmc_fit` / `ns_fit` to land the walker ball in a high-probability basin.

```python
import allesfitter

allesfitter.show_initial_guess('.')
res = allesfitter.optimize('.', method='cmaes', polish=True, n_restarts=4)
if res.accepted:
    print(f"optimize OK: lnprob {res.lnprob_initial:+.1f} -> {res.lnprob_opt:+.1f} "
          f"(Δ={res.delta_lnprob:+.1f}) in {res.nfev} evals, {res.wallclock_s:.1f}s")
else:
    print(f"optimize rejected ({res.reject_reason}); using original theta_0")
allesfitter.mcmc_fit('.')      # warm-started iff res.accepted
allesfitter.mcmc_output('.')
```

The optimum is pushed into `config.BASEMENT.theta_0` (so subsequent samplers pick it up automatically) only when **all** acceptance gates pass — otherwise `theta_0` is left untouched and the next sampler call sees the original initial values. This makes `optimize()` safe to call unconditionally in `run.py`.

### Methods

| `method=` | When to prefer | Notes |
|---|---|---|
| `'cmaes'` *(default)* | Almost everything in 10–50 dims | Requires `pip install cma`. Derivative-free, self-tuning step size, handles multimodal GP marginal likelihoods. Best general choice. |
| `'dual_annealing'` | scipy-only environments | Single chain; slower than CMA-ES wall-clock but no extra dep. |
| `'differential_evolution'` | Highly multimodal or `ndim > 40` | Population-based, parallel via `workers=N`. Needs more tuning. |
| `'L-BFGS-B'` | Already near the MAP / want a fast local polish | Finite-diff gradient; gets stuck on local maxima. |
| `'Powell'` | Derivative-free local | Drop-in for L-BFGS-B when finite-diff is too noisy. |

`polish=True` (default) appends a short L-BFGS-B refine to any global method.

### Acceptance gates

The result is pushed into `BASEMENT.theta_0` only when **all** of these pass:

1. **Improvement**: `lnprob_opt − lnprob_initial ≥ improvement_threshold` (default `0.5·ndim`, a loose AIC-style margin).
2. **Bounds**: no component of `theta_opt` sits within 0.01 % of a prior edge. Override with `skip_bounds_check=True` when a parameter is genuinely meant to live near its physical limit.
3. **Multistart consistency** (only when `n_restarts > 1`): spread of restart lnprobs is `< consistency_threshold` (default `1.0`). Larger spreads indicate genuine multimodality the user should investigate.

On reject, `OptimizeResult.reject_reason` records which gate fired; `BASEMENT.theta_0` is **not** mutated; the next `mcmc_fit` runs unchanged. The full result (theta, lnprobs, restart spread, nfev, wallclock, reject reason) is also persisted to `<datadir>/results/optimize_save.json`.

`optimize()` reports progress through `logprint`, so the run is visible both on the console and in the same `<datadir>/results/logfile_<now>.log` as the sampler runs:

- a **start** line — `optimize[cmaes] starting: ndim=… n_restarts=… maxfevals=… lnprob_initial=…`,
- one **per-restart** line — `optimize[cmaes] restart k/N: lnprob=… nfev=… ok=…`,
- the final **summary** — `optimize[cmaes] lnprob: A -> B (Δ=…) … [accepted/rejected]`.

This means the warm-start is captured in the run log even on unattended jobs, and you can see it is alive during a long global search. `quiet=True` suppresses only the console echo; the logfile is still written. For the full per-generation CMA-ES convergence table, pass `verbose=True` (CMA-ES only) — it streams the `cma` library's own report to the console on top of the `logprint` lines.

### CMA-ES warm-resume across calls

Every CMA-ES call pickles the final strategy (adapted covariance `C`, step size σ, generation counter, internal state) to `<datadir>/results/optimize_cma_state.pkl`. A subsequent call with `resume=True` reloads that state and **continues** evolution from where it left off — preserving the adapted covariance instead of restarting with the full prior-scale isotropic σ₀.

```python
# Day 1: 5-min budget, see how it converges
allesfitter.optimize('.', method='cmaes', maxfevals=500)
# Day 2: extend the same search for another 5 min
allesfitter.optimize('.', method='cmaes', maxfevals=500, resume=True)
```

`resume=True` requires `n_restarts=1` (a single trajectory cannot be split into multiple restarts) and `method='cmaes'`. If the pickled state is missing the call falls back to a fresh start with a warning; if it's incompatible (different `ndim`/`bounds` after a `params.csv` edit) the call raises `ValueError` so you delete the stale pickle deliberately. `OptimizeResult.resumed_from_pickle` reports whether the resume actually happened.

## Fitting dilution (`dil_<inst>`) in a chromatic model

`dil_<inst>` is **per-instrument**, never per-bandpass — two instruments mapped to the same bandpass each get their own dilution variable, independent of `chromatic=True`/`False`. The parser, validator, and likelihood-assembly tests all enforce this scope.

**How it enters the model.** In `flux_subfct_ellc`:

```
model_flux = 1 + ( (host_flux + companion_flux - 1) · (1 - dil_<inst>) )
```

So `dil` scales the **depth** of the transit for that instrument by `(1 − dil)`. Orbital geometry (`rsuma`, `cosi`, `period`, `epoch`) and limb-darkening shape are unaffected — only the in-transit floor moves up or down. Negative `dil` is allowed by the prior (`uniform -1 1` by default) and corresponds to *over*-correction (the true depth is deeper than the observed one).

**The chromatic-specific concern: degeneracy with per-band `rr`.** Observed transit depth in band `bp` for instrument `i` is approximately

```
δ_observed  ≈  rr_bp²  ·  (1 − dil_i)
```

Fit both `rr_bp` (per-band) and `dil_i` (per-inst) with broad priors and the photometry only pins down the **product**. Posteriors become highly anti-correlated (a deeper planet trades for a larger dilution and vice versa). Chromatic mode makes this worse because:

- the per-band rr keys multiply the number of `(rr_bp, dil_i)` ridges in the joint posterior;
- a real wavelength-dependent depth signal (atmosphere, spots, true dilution mismatch) can be absorbed into per-instrument dilution offsets, making the `ns_chromatic_rr_<companion>.pdf` overlays look artificially flat across bands.

**When to fit `dil`, when to fix it:**

| Case | Recommendation |
|---|---|
| Single instrument per bandpass, no known contamination | `fit=0, value=0` (default). Leave it alone. |
| Crowded TESS field, known contaminant in aperture | `fit=1` on the affected instrument with a **tight** prior, e.g. `normal <TICv8_contam> <σ≈0.05>`. Avoid the broad `uniform -1 1`. |
| Two instruments sharing one bandpass with different aperture sizes | Fit `dil` on the larger-aperture inst, fix it at 0 on the smaller-aperture inst. This anchors the contrast. |
| Bandpass A is known-clean (small-aperture ground-based) and bandpass B is known-dirty (TESS) | Fix `dil` on inst-A, fit it on inst-B with a strong informative prior from a contamination catalog. |
| You want to disentangle atmosphere/spots vs. dilution as the cause of band-to-band depth differences | Run two fits: one with `dil` fixed, one with `dil` fit under informative priors. Compare `log Z` (Bayes factor) — the data choose. |

**Practical diagnostics.** After a chromatic fit with free `dil`, check:

1. **Posterior correlation** between `b_rr_<bp>` and `dil_<inst>` for each instrument mapped to that bandpass (from `ns_corner.pdf` or the posterior samples). `|r| > 0.7` means the degeneracy is dominating.
2. **Width of the `dil_<inst>` posterior**: if it spans the full prior interval, the data isn't constraining it — you've added a free parameter that absorbs noise and inflates the `rr` uncertainty.
3. **`ns_chromatic_rr_<companion>.pdf`**: if all bandpasses' `rr` posteriors collapse to similar widths despite very different photometric S/N, dilution is likely soaking up the band-to-band variation.

**Bottom line.** The chromatic depth signal you're hunting for is *exactly the same kind of signal* a free dilution parameter can mimic. Fit `dil` only with strong external priors from contamination catalogs, and never fit it on every instrument simultaneously without anchoring at least one. The conservative `fit=0, value=0` default that `prepare_allesfit.py` emits is the right starting point; promote to `fit=1` only when you have catalog evidence of a contaminant and a defensible prior. See also [TIC contratio vs SPOC CROWDSAP](#tic-contratio-vs-spoc-crowdsap).

## Limb-darkening keying (band vs. instrument)

Limb darkening is split across **two keys with different scopes**, and it helps to keep them straight:

| Key | Lives in | Scope | Resolution |
|---|---|---|---|
| `host_ldc_q*` (the coefficients, fit parameters) | `params.csv` | **Per bandpass** | suffix = bandpass when a `bandpass` row exists, else falls back to `<inst>` (`get_ldc_bandpass` / `get_ldc_key`) |
| `host_ld_law` (the law, a setting) | `settings.csv` | **Per instrument** (canonical), but **authorable per bandpass** | explicit `host_ld_law_<inst>` wins; else `host_ld_law_<band>` fans out to every instrument on that band; else defaults to `quad` |

**Author it by bandpass.** Limb darkening is a function of wavelength, not of the detector, so the ergonomic and physically-correct way to write both keys is per bandpass:

```
# settings.csv
bandpass,tess k2
host_ld_law_tess,quad
host_ld_law_k2,quad
# params.csv
host_ldc_q1_tess, ...
host_ldc_q2_tess, ...
host_ldc_q1_k2,   ...
host_ldc_q2_k2,   ...
```

Two instruments mapped to the **same** bandpass therefore **share one set of `q*` parameters** — they are tied by construction, not duplicated. The `_<inst>` forms remain accepted for backward compatibility and as a deliberate per-instrument override.

**Why the law stays instrument-keyed internally.** Every downstream consumer (`computer.py`, `deriver.py`) reads `host_ld_law_<inst>`, because the transit model is *evaluated* per instrument. The per-bandpass form is canonicalized to per-instrument during config parsing. This keeps the rare-but-valid override available (e.g. disable LD on one noisy detector with `host_ld_law_<inst>,none`) without forcing a band-only world on the model layer.

> Note the **reversed precedence** between the two keys: the law resolves *override-first* (instrument beats bandpass), while the coefficients resolve *band-first* (bandpass beats instrument). Both follow the same principle — bandpass is the convenient default, instrument is the unit of evaluation and override.

**Does the number of LD parameters depend on the bandpass or the instrument?** The **count is set by the law** (`none`→0, `lin`→1 `q1`, `quad`→2 `q1,q2`, `sing`→3 `q1,q2,q3` via `LDC3`; see `deriver.py`). That law is canonically **per instrument**, but it **defaults from the bandpass**, and the coefficient parameters it counts are themselves **bandpass-keyed**. So in normal use — where each bandpass maps to one law — **the number of limb-darkening parameters is effectively per bandpass**: instruments sharing a band share the same law *and* the same `q*` parameters. It is per-instrument only if you deliberately override `host_ld_law_<inst>` to a different law than its band-mates, which is discouraged (the shared `q*` parameters would no longer match the per-instrument law's arity).

## OOM-safe diagnostic plots for high-dim fits

Chromatic multi-band fits with per-instrument baseline GPs routinely produce 25–60 free parameters. The default `ns_output` / `mcmc_output` diagnostic plots scale **poorly** at that dimensionality: a 60×60 corner is ~3 600 subplots, the matplotlib canvas at the implicit `figsize=(2·ndim, 2·ndim)` runs to hundreds of megapixels at 100 dpi, and the OOM killer terminates the post-processing run before any tables are written.

Both `ns_output` and `mcmc_output` apply three coordinated protections:

1. **Hard figure-size caps** — the chains-vs-step figure and corner figure are clamped to `_MAX_CHAINS_INCHES` and `_MAX_CORNER_INCHES`, with chain-step subsampling above `_MAX_CHAIN_PLOT_STEPS` and posterior subsampling above `_MAX_CORNER_SAMPLES`.
2. **Nuisance filter for the corner plot** — when `ndim > 25`, the corner drops every row whose fitkey starts with `baseline_`, `ln_err_flux_`, `ln_jitter_rv_`, or `stellar_var_gp_`. These are GP hypers, per-band white-noise floors, and other nuisance parameters that almost never repay the visual cost of being in a giant pairs plot. The dropped names are logged so the user knows what's hidden:
   ```
   ! corner: hiding 12 nuisance params for readability (ndim 28 > 25).
     Example: baseline_gp_matern32_lnsigma_flux_m4g, baseline_gp_matern32_lnrho_flux_m4g, ln_err_flux_m4g, ...
   ```
   The full posterior, including every nuisance parameter, still goes into `mcmc_table.csv` / `ns_table.csv`, the LaTeX summary table, and `deriver.derive`.
3. **`MemoryError`-tolerant save sites** — if any single plot still OOMs (extreme cases: hundreds of dims, very long chains, tiny memory budget), the failure is caught, logged as `! plot_* failed (...)`, and **the rest of the pipeline continues** — tables, derived parameters, residual statistics, and per-companion fit PDFs all still get written. The user keeps the numerical outputs even when one diagnostic figure can't be rendered.

Thresholds live as module-level constants in `allesfitter/nested_sampling_output.py` and `allesfitter/mcmc_output.py` (`_HARD_NDIM_CAP`, `_CORNER_HIDE_NUISANCE_NDIM_THRESHOLD`, `_MAX_CORNER_SAMPLES`, etc.) — edit those if your hardware or use case wants different cut-offs. The hard skip kicks in above `ndim > 60`, where corner.corner becomes effectively unusable regardless of memory.

## Tracking long-running fits with the centralized run log

Every `ns_fit` / `ns_output` / `mcmc_fit` / `mcmc_output` call wrapped with `allesfitter.log_run(...)` appends start and end rows to `~/.allesfitter/runs.jsonl` (one JSON object per line). Use it to answer "which fits are still running, where, against which datadir?" with a single `tail` or `jq`.

`prepare_allesfit` already emits a `run.py` template with the wrapping pre-written; uncomment and go:

```python
import allesfitter

with allesfitter.log_run("ns_fit", "."):
    allesfitter.ns_fit(".")

with allesfitter.log_run("ns_output", "."):
    allesfitter.ns_output(".")
```

Inspect the log:

```bash
# raw
tail -n 20 ~/.allesfitter/runs.jsonl

# pretty-print the last 5 entries
jq -c '.' ~/.allesfitter/runs.jsonl | tail -n 5

# only currently-running fits
jq -c 'select(.event=="start")' ~/.allesfitter/runs.jsonl | \
  comm -23 - <(jq -c 'select(.event=="end") | {run_id, command}' \
                ~/.allesfitter/runs.jsonl)
```

Or from Python:

```python
import allesfitter
for row in allesfitter.run_log_tail(50):
    print(row)
```

Override the log path with `export ALLESFITTER_RUN_LOG=/abs/path/runs.jsonl` (useful for shared/networked logs across machines). Each row carries `pid`, `hostname`, `user`, absolute `datadir`, `run_id`, `duration_sec` and (on failure) `error` + truncated `traceback`. Writes are serialized with `fcntl.flock`, so concurrent fits on the same host append cleanly.

## Raw-flux outlier clipping

Add to `settings.csv` to drop rows outside the given flux window:

```
flux_min_raw,0.9
flux_max_raw,1.1
```

Either bound is optional (one-sided clipping). Clipped points are removed from the likelihood **and** retained for display — they appear as red ✕ markers on the photometric `full` panels in `initial_guess.pdf` and `ns_fit_<companion>.pdf`.

## Lightcurves from different pipelines

You should be specific which lightcurve produced by which pipeline to use. SPOC produces two lightcurves `pdcsap` and `sap` and `pdcsap` is usually better so it is used by default. Inspect the plots in the `.png` file which shows both `pdcsap` and `sap` lightcurves. In the event you want to use `sap` instead, then specify `-lc sap` in the script.

In case you want to study the impact on derived transit parameters by the choice of lightcurves produced by two different TESS pipelines, try
```bash
$ prepare_allesfit -name "HD 39091" -s 1 -p spoc -f spoc -dir spoc
$ prepare_allesfit -name "HD 39091" -s 1 -p qlp -f qlp -dir qlp
```
Note that two directories are produced. Before fitting, you should manually combine each parameter file into one `params.csv` and `settings.csv`. In `settings.csv`, set `inst_phot,spoc qlp` to specify that two instruments are used for fitting. You should also indicate two names in limb darkening, error, and baseline model parameters e.g.
```
host_ld_law_spoc,quad
host_ld_law_qlp,quad
...
error_flux_spoc,sample
error_flux_qlp,sample
...
baseline_flux_spoc,sample_GP_Matern32
baseline_flux_qlp,sample_GP_Matern32
```
The pipeline will expect two lightcurves named `spoc.csv` and `qlp.csv`.

## Lightcurves from different exposure times

You should also be specific which exposure time to use. By default, 120s is good enough to produce reliable results.

In case you want the effect of the choice of lightcurves with different exposure times, try
```bash
$ prepare_allesfit -toi HD39091 -s -1 -p qlp -e 200 -f qlp200 -dir qlp200
$ prepare_allesfit -toi HD39091 -s -1 -p qlp -e 600 -f qlp600 -dir qlp600
```
As before, you should merge the outputs into one `params.csv` and `settings.csv`. In case only long cadence e.g. exp=1800s data is available, you should also set `t_exp_qlp1800,0.02` which is the exposure time in unit of days for the `qlp1800` lightcurve.

## Using a previous run for a TTV fit

Run the command below to read the posterior samples in `HD39091/results` and create a `params2.csv` file.
```bash
$ prepare_allesfit -name "HD 39091" -r HD39091 -s 1
```
where `-r` specifies the path to the directory of your previous run. The `-s` flag is not important so any number is fine.

You can then copy the values or re-name the file entirely to `params.csv`. You may fix all the transit, lightcurve, and baseline parameters (default is 1) and then add all the necessary ttv parameters e.g. `b_ttv_transit_1` in `params.csv`. Set `fit_ttvs,True` in `settings.csv` and run `python run.py` with a different directory name e.g. `allesfitter.ns_fit('ttv')`. This procedure is useful not only for TTVs but also for other fits with iterative refinement. For more info, see files in the [TOI-216 example](https://github.com/MNGuenther/allesfitter/tree/master/paper/TOI-216).

## Performance & caching (`simulate_PDF` disk cache)

General performance behaviour:

- **Runtime** depends on data volume and convergence criteria.
- **Memory usage** scales with sector count and cadence.
- **Convergence** varies by parameter complexity and data quality.
- **Parallel processing** significantly reduces fit time.

### Cold-start latency: `simulate_PDF` disk cache

When `use_host_density_prior=True` (the default for transit fits) and `params_star.csv` carries asymmetric error bars on `R_star` / `M_star`, `Basement.load_params` invokes `simulate_PDF.calculate_skewed_normal_params` twice. Each call inverts the skewed-normal CDF via `scipy.stats.skewnorm.ppf` inside a `scipy.optimize.minimize` loop — about 11 s per call (~6 000 numerical CDF inversions). Without caching this dominates `config.init` (≈22 s) and pads every `mcmc_fit` / `ns_fit` / `mcmc_output` invocation that re-bootstraps the Basement.

allesfitter persists the `(median, lower_err, upper_err) → (alpha, loc, scale)` mapping to **`~/.allesfitter/simulate_PDF_cache.json`** after the first solve. Subsequent process invocations with identical inputs read the cache (~1 ms) and skip the entire scipy loop.

| Setup | Inputs | `config.init` wall-clock |
|---|---|---|
| `use_host_density_prior=False` | n/a (skipped) | <1 s |
| `True`, **cache miss** | first time you see this star | ~22 s |
| `True`, **cache hit** | same star ever seen by this user | <100 ms |

| Environment variable | Effect |
|---|---|
| `ALLESFITTER_SIMULATE_PDF_CACHE` | Override cache file path (default `~/.allesfitter/simulate_PDF_cache.json`) |
| `ALLESFITTER_SIMULATE_PDF_NO_CACHE=1` | Disable cache (always recompute, never write) |

The cache is content-addressed by a stable `repr(float)` triplet, so two different projects that fit the same target reuse the same entry. Writes are atomic (`os.replace` on a temp file), so concurrent processes can't corrupt the JSON.

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
search incrementally (see the [Warm-starting](#warm-starting-mcmc--ns-with-allesfitteroptimize) walkthrough).

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

**`binning`** — optional; **None** by default. A positive float bins every
photometric light curve to that width **in days** at load time (e.g.
`binning,0.0208333` ≈ 30 min); covariate columns are mean-binned on the same
grid. Use it to speed up fits or suppress short-timescale correlated noise.
Caveats the validator flags: keep it well below the transit duration (coarse
bins smear the transit), above the native cadence (otherwise it is a no-op),
and set a matching `t_exp_<inst>` so the model is supersampled over the bin
width. Applies to photometry only; RV is untouched. Invalid values
(non-numeric, ≤ 0, or ≥ the observation baseline) are hard errors.

When an instrument is binned and you have **not** set `t_exp_<inst>`, allesfitter
auto-sets `t_exp_<inst>` to the bin width and seeds `t_exp_n_int_<inst>` (default
10), logging both — binned points are time-averages over the bin width, so the
transit model must be integrated over that window. An explicit `t_exp_<inst>` is
never overwritten; a value that differs from the bin width is surfaced as a
warning instead.

**`binning_<inst>`** — optional per-instrument override of `binning` (same
units, days). Set it to bin only specific light curves: e.g. leave the global
`binning` empty and add `binning_TESS,0.0208333` to bin TESS alone, or set a
different width per instrument. A `binning_<inst>` key that is present but empty
/ `None` turns binning **off** for that instrument even when the global
`binning` bins everything else. Instruments without an override fall back to the
global `binning`. Validated exactly like `binning` (per-instrument).

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
  geometry. (First solve per star is ~22 s; cached afterwards — see
  [Performance & caching](#performance--caching-simulate_pdf-disk-cache).)
- **Eccentricity:** default to circular (`b_f_c=b_f_s=0`, `fit=0`) unless you
  have RVs or strong secondary-eclipse timing; freeing `f_c/f_s` on a single
  transit just inflates the depth/duration posterior.
- **Limb darkening:** the `quad` law with `q1/q2 ~ uniform 0 1` is the safe
  default. Fix or Gaussian-constrain to theoretical (`ldtk`/`limbdark`) values
  only for low-S/N transits where LDC is unconstrained by the data. Keep LDC
  keyed correctly (band vs instrument) — see [Limb-darkening keying](#limb-darkening-keying-band-vs-instrument).
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
- Tail the centralized run log: `tail -n 10 ~/.allesfitter/runs.jsonl` (or whatever path `$ALLESFITTER_RUN_LOG` resolves to). Each `start` row carries `pid`, `hostname`, absolute `datadir`, and `run_id`; the matching `end` row carries `status` and `duration_sec`. See [Tracking long-running fits](#tracking-long-running-fits-with-the-centralized-run-log).

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
`normal`-prior `dil_<inst>` row you can activate (see the
[Fitting dilution](#fitting-dilution-dil_inst-in-a-chromatic-model) walkthrough).
Fall back to TICv8 contratio only when no SPOC product exists for that sector.
