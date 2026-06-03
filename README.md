# allesfitter2
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/jpdeleon/allesfitter2)

allesfitter2 is an extension of the allesfitter package, providing a comprehensive framework for global inference of photometry and radial velocity (RV) data. It specializes in the characterization of exoplanetary systems and eclipsing binaries by integrating transit and RV modeling with robust Bayesian inference engines.
Several utilities and scripts are added to streamline the process of downloading TESS, K2, and Kepler lightcurves and automatically generating all necessary files to run allesfitter.

## New Features

- **Automated lightcurve download** from TESS, K2, and Kepler mission data
- **Multi-pipeline support** (SPOC, QLP, EVEREST, K2SFF) with configurable parameters
- **Parameter derivation** from multiple astronomical databases (NExSci, TOI, CTOI, TIC)
- **Flexible time-window selection** — TESS sectors, K2 campaigns (including split campaigns 11a/11b), and Kepler quarters (single, multiple, or all)
- **Chromatic transit modeling** — fit a separate `Rp/Rs` per bandpass while keeping orbital parameters globally shared
- **Shared baseline GP across bands** — declare `baseline_share_flux,muscat_g:muscat_r:muscat_i:muscat_z` in `settings.csv` to fit a *single* celerite GP realization jointly across all members of a share group (ideal for simultaneous multi-band photometry like MuSCAT, where airmass/seeing systematics are common-mode). Backward compatible: omit the key for legacy per-instrument GPs.
- **Warm-start sampler with `allesfitter.optimize()`** — global optimization (CMA-ES default, plus `dual_annealing` / `differential_evolution` / `L-BFGS-B` / `Powell`) finds a MAP point that is then pushed into `BASEMENT.theta_0`, so the next `mcmc_fit` / `ns_fit` starts from a well-converged ball. Safe acceptance gates (improvement, prior-bound, multistart consistency) prevent the optimizer from poisoning the sampler start when it doesn't actually improve. CMA-ES supports warm-resume across calls via a pickled strategy state.
- **OOM-safe post-processing** — `ns_output` and `mcmc_output` cap figure sizes, subsample posterior draws for the corner plot, and wrap every save site in a `MemoryError`-tolerant try/except. When the fit has more than 25 free parameters, the corner plot automatically hides nuisance rows (`baseline_*`, `ln_err_*`, `ln_jitter_*`, `stellar_var_gp_*`) so the science-relevant parameters stay readable; above 60 dims the corner is skipped entirely with a placeholder. The full posterior still goes into `*_table.csv`, the LaTeX table, and `derive`.
- **Named ancillary covariates in input CSVs** — per-instrument CSVs may now carry extra columns beyond `time, flux, flux_err` (e.g. `airmass`, `fwhm`, `sky`, `x_centroid`). A `#`-prefixed header line on the first row names them; baselines select a covariate as their regression axis via `baseline_<key>_<inst>_against,<name>`. Works with every existing baseline (`sample_linear`, `hybrid_poly_N`, `hybrid_spline`, `sample_GP_*`). Legacy 3-column and 4-column positional CSVs continue to load unchanged.
- **N-D linear baseline detrending (`sample_linear_multi` / `hybrid_linear_multi`)** — declare `baseline_flux_<inst>,sample_linear_multi` + `baseline_flux_<inst>_cols,Airmass FWHM(pix) bias` to fit a joint linear model in any number of ancillary covariates. The `sample_*` variant samples each weight as a free parameter; the `hybrid_*` variant **analytically marginalises** the Gaussian weights in closed form at every likelihood call, adding zero fit dimensions while still recovering the optimal per-evaluation MAP weights for predictive plotting.
- **Fast Basement init via cached `simulate_PDF`** — the skewed-normal fit to `R_star` / `M_star` from `params_star.csv` (used by `use_host_density_prior=True`) used to add ~22 s to *every* `config.init` call. Results are now persisted to `~/.allesfitter/simulate_PDF_cache.json` keyed on `(median, lower_err, upper_err)`, so the second and every subsequent invocation skips the scipy solve entirely. Bypass via `ALLESFITTER_SIMULATE_PDF_NO_CACHE=1`; redirect via `ALLESFITTER_SIMULATE_PDF_CACHE=/path/to/cache.json`.
- **Per-bandpass Rp/Rs posterior plot** — `ns_output()` automatically emits `ns_chromatic_rr_<companion>.pdf` overlaying per-bandpass posteriors with a canonical color map (`tess=k`, `g=C0`, `r=C2`, `i=C8`, `z=C3`; viridis fallback for unknown labels)
- **Strict configuration validation** — clear errors for bandpass/instrument count mismatch, duplicate params, unknown bandpass suffixes, chromatic/achromatic shape inconsistencies, or per-instrument settings keys with orphan suffixes (catches `host_ld_law_<bandpass>` typos)
- **Band-dependent limb darkening** - `host_ldc_q*_<band>` in params.csv and `host_ld_law_<band>` in settings.csv are now accepted. Although `host_ldc_q*_<inst>` and `host_ld_law_<inst>` are still accepted for backward-compatibility. The number of limb-darkening parameters should depend on the bandpass instead of per instrument.
- **Sensible LD default** — `host_ld_law_<inst>` now defaults to `quad` (was `None`, which silently disabled limb darkening); explicit `host_ld_law_<inst>,none` still opts out
- **Raw-flux outlier clipping** via `flux_min_raw` / `flux_max_raw` — clipped points are removed from the fit but overlaid in red on `initial_guess.pdf`
- **Centralized run log** — every `ns_fit` / `ns_output` / `mcmc_fit` call records a JSONL row at `~/.allesfitter/runs.jsonl` (override via `ALLESFITTER_RUN_LOG`) with absolute datadir, pid, hostname, duration, and status. Inspect with `jq` or `allesfitter.run_log_tail()`.
- **Theoretical limb darkening** coefficients from Claret tables using [limbdark](https://github.com/jpdeleon/limbdark2)
- **Test suite** under `tests/chromatic/` pins the chromatic + logger contracts with 64+ unit, parsing, likelihood-assembly, end-to-end fit, and run-logger tests

## Installation

```bash
git clone https://github.com/jpdeleon/allesfitter2.git
cd allesfitter2
pip install .
```

If you encounter a problem with ellc, you can use a custom installation script
```bash
cd allesfitter2
bash install.sh
```
which includes cloning ellc package and building the fortran extension modules.

## Quick Start

```bash
prepare_allesfit -name "HD 39091" -s 1 -f tess
```

This creates a complete analysis directory:

```
HD39091/
├── params.csv                      
├── settings.csv                    
├── params_star.csv                 
├── run.py                              
├── tess.csv   
└── tess.png    
```

### Running the Analysis

1. **Initial setup check:**
   ```bash
   cd HD39091/
   python run.py  # Shows initial parameter plots
   ```

2. **Full analysis:**
   Edit `run.py` to uncomment the fitting lines:
   ```python
   #!/usr/bin/env python 
   import allesfitter

   fig = allesfitter.show_initial_guess('.')
   allesfitter.prepare_ttv_fit('.', style='tessplot')

   allesfitter.ns_fit('.')      # Uncomment for nested sampling
   allesfitter.ns_output('.')   # Uncomment for parameter derivation
   ```

3. **Execute:** `python run.py`

Results are saved in `HD39091/results/`

## Use cases

### Single TESS lightcurve
```bash
$ prepare_allesfit -name "HD 39091" -s 1
```
The command above will create a directory with six files inside:
```
HD39091/
├── params.csv         # the parameters file which contains the parameters to fit or fixed and their priors
├── settings.csv       # the settings file which contains the setup of the transit and baseline model, sampler, etc
├── params_star.csv    # input parameters of the star used for deriving planet properties after the fit; used as prior if `use_host_density_prior,True` in `settings.csv`
├── run.py             # file for running allesfitter
├── tess_spoc_pdcsap_s1_exp120s.csv  # TESS lightcurve produced by SPOC pipeline (default) with exposure time of 120s (default)
└── tess_spoc_pdcsap_s1_exp120s.png  # Light curve plot just for checking
```
In `params.csv`, inspect the values of all planets in the system, their corresponding priors, and decide whether they should be free=1 or fixed=0.
Since `-name "HD 39091"` was used, the planet and star parameters came from the published values cataloged in [NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=PSCompPars).
Instead if `-toi 144` or `-tic 261136679` were used (which refers to the same star), the values would come from the (unpublished) TOI catalog which can be viewed in [exofop](https://exofop.ipac.caltech.edu/tess/target.php?id=HD39091).

`params.csv` file includes instrument-dependent parameters such as limb-darkening coefficients (e.g. host_ldc_q1_tess). 
Here, `tess` is the default instrument so it expects a lightcurve file named `tess.csv`.
Thus, you should rename `tess_spoc_pdcsap_s1_exp120s.csv` to `tess.csv`.
You can set the instrument name in `params.csv` and the lightcurve filename by adding the `-f <name>` flag.
By default, a GP with Matern-3/2 kernel is used for the baseline model. Check the full setup in `settings.csv`.

Now you can run alesfitter with
```python
$ cd HD39091/
$ python run.py
```
which will save the outputs in `HD39091/results`.
By default, the first run only plots the transit and baseline models using the initial guess values.
If you're satistied with the plots, then you can run `python run.py` again after removing the `#` in `ns_fit` to run nested sampling,
and in `ns_output` to derive the parameters after the fit.
```python
#!/usr/bin/env python 
import allesfitter

fig = allesfitter.show_initial_guess('.')
allesfitter.prepare_ttv_fit('.', style='tessplot')

#allesfitter.ns_fit('.')    #<- uncomment
#allesfitter.ns_output('.') #<- uncomment
```
The run time will depend on the number of datapoints and the convergence criterion defined in `settings.py`.
By default, `fast_fit,True` and `fast_fit_width.0.33` are set which means only 33% of baseline data of each transit is used.
You can increase the the baseline by setting larger fast_fit_width. You can set `fast_fit,False` to fit the entire phase.
For preliminary analysis, you can set the delta log evidence `ns_tol,1` from 1 (default) to 10 or 100 to make your run time shorter.
Use only `ns_tol,0.01` for the final run.

### Using the results of previous run for TTV fit
Run the command below to read the posterior samples in `HD39091/results` and create `params2.csv` file.
```bash
$ prepare_allesfit -name "HD 39091" -r HD39091 -s 1
```
where `-r` specifies the path to the directory of your previous run.
The `-s` flag is not important so any number is fine.

You can then copy the values or re-name the file entirely to `params.csv`.
You may fix all the transit, lightcurve, and baseline parameters (default is 1) and then add all the necessary ttv parameters e.g. `b_ttv_transit_1` in `params.csv`.
Set `fit_ttvs,True` in `settings.csv` and run `python run.py` with a different a different directory name e.g. `allesfitter.ns_fit('ttv')`.
This procedure is useful not only for TTVs but also for other fits with iterative refinement.
For more info, see files in the [TOI-216 example](https://github.com/MNGuenther/allesfitter/tree/master/paper/TOI-216).

### Lightcurves from different pipelines
You should be specific which lightcurve produced by which pipeline to use. SPOC produces two lightcurves `pdcsap` and `sap` and `pdcsap` is usually better so it is used by default. Inspect the plots in `.png` file which shows both `pdcsap` and `sap` lightcurves. In the event you want to use `sap` instead, then specify `-lc sap` in the script.

In case you want to study the impact on derived transit parameters by the choice of lightcurves produced by two different TESS pipelines, try
```bash
$ prepare_allesfit -name "HD 39091" -s 1 -p spoc -f spoc -dir spoc
$ prepare_allesfit -name "HD 39091" -s 1 -p qlp -f qlp -dir qlp
```
Note that two directories are produced. 
Before fitting, you should manually combine the each parameter file into one `params.csv` and `settings.csv`.
In `settings.csv`, set `inst_phot,spoc qlp` to specify that two instruments are used for fitting.
You should also indicate two names in limb darkening, error, and baseline model parameters e.g. 
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

### Chromatic transit modeling (per-band Rp/Rs)

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

### Sharing a baseline GP across instruments (joint celerite realization)

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

### Warm-starting MCMC / NS with `allesfitter.optimize()`

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

#### Methods

| `method=` | When to prefer | Notes |
|---|---|---|
| `'cmaes'` *(default)* | Almost everything in 10–50 dims | Requires `pip install cma`. Derivative-free, self-tuning step size, handles multimodal GP marginal likelihoods. Best general choice. |
| `'dual_annealing'` | scipy-only environments | Single chain; slower than CMA-ES wall-clock but no extra dep. |
| `'differential_evolution'` | Highly multimodal or `ndim > 40` | Population-based, parallel via `workers=N`. Needs more tuning. |
| `'L-BFGS-B'` | Already near the MAP / want a fast local polish | Finite-diff gradient; gets stuck on local maxima. |
| `'Powell'` | Derivative-free local | Drop-in for L-BFGS-B when finite-diff is too noisy. |

`polish=True` (default) appends a short L-BFGS-B refine to any global method.

#### Acceptance gates

The result is pushed into `BASEMENT.theta_0` only when **all** of these pass:

1. **Improvement**: `lnprob_opt − lnprob_initial ≥ improvement_threshold` (default `0.5·ndim`, a loose AIC-style margin).
2. **Bounds**: no component of `theta_opt` sits within 0.01 % of a prior edge. Override with `skip_bounds_check=True` when a parameter is genuinely meant to live near its physical limit.
3. **Multistart consistency** (only when `n_restarts > 1`): spread of restart lnprobs is `< consistency_threshold` (default `1.0`). Larger spreads indicate genuine multimodality the user should investigate.

On reject, `OptimizeResult.reject_reason` records which gate fired; `BASEMENT.theta_0` is **not** mutated; the next `mcmc_fit` runs unchanged. The full result (theta, lnprobs, restart spread, nfev, wallclock, reject reason) is also persisted to `<datadir>/results/optimize_save.json`.

#### CMA-ES warm-resume across calls

Every CMA-ES call pickles the final strategy (adapted covariance `C`, step size σ, generation counter, internal state) to `<datadir>/results/optimize_cma_state.pkl`. A subsequent call with `resume=True` reloads that state and **continues** evolution from where it left off — preserving the adapted covariance instead of restarting with the full prior-scale isotropic σ₀.

```python
# Day 1: 5-min budget, see how it converges
allesfitter.optimize('.', method='cmaes', maxfevals=500)
# Day 2: extend the same search for another 5 min
allesfitter.optimize('.', method='cmaes', maxfevals=500, resume=True)
```

`resume=True` requires `n_restarts=1` (a single trajectory cannot be split into multiple restarts) and `method='cmaes'`. If the pickled state is missing the call falls back to a fresh start with a warning; if it's incompatible (different `ndim`/`bounds` after a `params.csv` edit) the call raises `ValueError` so you delete the stale pickle deliberately. `OptimizeResult.resumed_from_pickle` reports whether the resume actually happened.

### Detrending baselines against ancillary covariates (airmass, FWHM, sky, …)

Per-instrument CSV files can now carry **named ancillary columns** beyond the required `time, flux, flux_err`. Add a `#`-prefixed header on the first non-blank line listing every column:

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

Backward compatibility (no migration required for existing fits):

| CSV layout | Behaviour |
|---|---|
| `#time,flux,flux_err,<name1>,<name2>,...` (headered) | `data[inst]['covariates']` keyed by name; first ancillary column is also aliased to the legacy `custom_series` slot |
| 4 positional columns, no header | Legacy: column 4 → `custom_series`; `covariates` dict empty |
| 3 positional columns, no header | Legacy: `custom_series` = zeros; `covariates` dict empty |

The loader validates after-the-fact: every `baseline_<key>_<inst>_against=<name>` setting that names something other than `time` / `custom_series` must resolve to a real column in the corresponding CSV, or `Basement(...)` refuses to start with the actionable error `<inst>.csv has no column named '<name>'. Known options: ...`.

Limitations to be aware of:

- **One covariate per baseline (v1)**. True multi-covariate detrending against `airmass + fwhm + sky` simultaneously needs a separate `hybrid_linear_multi` baseline kind that isn't shipped yet. The common workflow today is a `hybrid_poly_N` against the dominant covariate plus a `sample_GP_*` for the residual structure.
- **GPs are 1-D** (celerite). `sample_GP_*` with `_against=<covariate>` fits a 1-D GP regressing residuals against the covariate value; multi-dimensional GPs would require switching to `george` or `tinygp` — not in scope.
- **Share-baseline groups must use `_against=time`**. The joint celerite GP across `baseline_share_flux,m4g:m4r:m4i:m4z` is a shared *realization* on the shared time grid; covariate-based regression is only meaningful per inst. The loader enforces this constraint.

### OOM-safe diagnostic plots for high-dim fits

Chromatic multi-band fits with per-instrument baseline GPs routinely produce 25–60 free parameters. The default `ns_output` / `mcmc_output` diagnostic plots scale **poorly** at that dimensionality: a 60×60 corner is ~3 600 subplots, the matplotlib canvas at the implicit `figsize=(2·ndim, 2·ndim)` runs to hundreds of megapixels at 100 dpi, and the OOM killer terminates the post-processing run before any tables are written.

Both `ns_output` and `mcmc_output` now apply three coordinated protections:

1. **Hard figure-size caps** — the chains-vs-step figure and corner figure are clamped to `_MAX_CHAINS_INCHES` and `_MAX_CORNER_INCHES`, with chain-step subsampling above `_MAX_CHAIN_PLOT_STEPS` and posterior subsampling above `_MAX_CORNER_SAMPLES`.
2. **Nuisance filter for the corner plot** — when `ndim > 25`, the corner drops every row whose fitkey starts with `baseline_`, `ln_err_flux_`, `ln_jitter_rv_`, or `stellar_var_gp_`. These are GP hypers, per-band white-noise floors, and other nuisance parameters that almost never repay the visual cost of being in a giant pairs plot. The dropped names are logged so the user knows what's hidden:
   ```
   ! corner: hiding 12 nuisance params for readability (ndim 28 > 25).
     Example: baseline_gp_matern32_lnsigma_flux_m4g, baseline_gp_matern32_lnrho_flux_m4g, ln_err_flux_m4g, ...
   ```
   The full posterior, including every nuisance parameter, still goes into `mcmc_table.csv` / `ns_table.csv`, the LaTeX summary table, and `deriver.derive`.
3. **`MemoryError`-tolerant save sites** — if any single plot still OOMs (extreme cases: hundreds of dims, very long chains, tiny memory budget), the failure is caught, logged as `! plot_* failed (...)`, and **the rest of the pipeline continues** — tables, derived parameters, residual statistics, and per-companion fit PDFs all still get written. The user keeps the numerical outputs even when one diagnostic figure can't be rendered.

Thresholds live as module-level constants in `allesfitter/nested_sampling_output.py` and `allesfitter/mcmc_output.py` (`_HARD_NDIM_CAP`, `_CORNER_HIDE_NUISANCE_NDIM_THRESHOLD`, `_MAX_CORNER_SAMPLES`, etc.) — edit those if your hardware or use case wants different cut-offs. The hard skip kicks in above `ndim > 60`, where corner.corner becomes effectively unusable regardless of memory.

### Fitting dilution (`dil_<inst>`) in a chromatic model

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

**Bottom line.** The chromatic depth signal you're hunting for is *exactly the same kind of signal* a free dilution parameter can mimic. Fit `dil` only with strong external priors from contamination catalogs, and never fit it on every instrument simultaneously without anchoring at least one. The conservative `fit=0, value=0` default that `prepare_allesfit.py` emits is the right starting point; promote to `fit=1` only when you have catalog evidence of a contaminant and a defensible prior.

### Limb-darkening keying (band vs. instrument)

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

### Tracking long-running fits with the centralized run log

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

### Raw-flux outlier clipping

Add to `settings.csv` to drop rows outside the given flux window:

```
flux_min_raw,0.9
flux_max_raw,1.1
```

Either bound is optional (one-sided clipping). Clipped points are removed from the likelihood **and** retained for display — they appear as red ✕ markers on the photometric `full` panels in `initial_guess.pdf` and `ns_fit_<companion>.pdf`.

### Lightcurves from different exposure times
You should also be specific which exposure time to use. By default, 120s is good enough to produce reliable results. 

In case you want to effect of the choice of lightcurves with different exposure times, try
```bash
$ prepare_allesfit -toi HD39091 -s -1 -p qlp -e 200 -f qlp200 -dir qlp200
$ prepare_allesfit -toi HD39091 -s -1 -p qlp -e 600 -f qlp600 -dir qlp600
```
As before, you should merge the outputs into one `params.csv` and `settings.csv`.
In case only long cadence e.g. exp=1800s data is available, you should also set `t_exp_qlp1800,0.02` which is the exposure time in unit of days for the `qlp1800` lightcurve.

## Command Line Options

### Target Selection (mutually exclusive, required)

| Option | Description | Example |
|--------|-------------|---------|
| `-toi <ID>` | TOI (TESS Object of Interest) ID | `-toi 144` |
| `-tic <ID>` | TIC (TESS Input Catalog) ID | `-tic 261136679` |
| `-ctoi <ID>` | CTOI (Community TOI) ID | `-ctoi 12345` |
| `-name <NAME>` | Target name (NExSci database) | `-name "HD 39091"` |

### Data Selection (mutually exclusive, required)

Exactly one of `-s`, `-c`, or `-q` must be given, matched to the mission supplied via `-m`.

| Option | Mission | Description | Example |
|--------|---------|-------------|---------|
| `-s, --sector <S>` | TESS | Sector number(s) | `-s 1 3 5` |
| `-c, --campaign <C>` | K2 | Campaign number(s) (supports `11a`, `11b`) | `-c all` |
| `-q, --quarter <Q>` | Kepler | Quarter number(s) | `-q 4` |

Each accepts:
- explicit numbers/labels: `-s 1 3 5`, `-c 11a 11b`
- `-1` for the most recent (default), `0` for the first
- `all` for every available time window

### Data Processing Options

| Option | Description | Default |
|--------|-------------|---------|
| `-m, --mission <NAME>` | `tess`, `k2`, or `kepler` | `tess` |
| `-e, --exptime <SEC>` | Exposure time in seconds (`120`, `200`, `600`, `1800`); written to `settings.csv` as `t_exp_<inst>` in days (`-e 120` → `0.001389`) | None (uses lightkurve metadata) |
| `-p, --pipeline <NAME>` | Data pipeline | `spoc` |
| `-lc, --lc_type <TYPE>` | Light curve type | `pdcsap` |
| `-sig, --sigma <N>` | Sigma clipping threshold | None |
| `-qb, --quality <LEVEL>` | Quality bitmask | `default` |

**Pipeline Options:**
- TESS: `spoc` (recommended), `qlp`
- K2: `k2`, `everest`, `k2sff`
- Kepler: `kepler`

**Light Curve Types:**
- `pdcsap`: Pre-search Data Conditioning SAP (recommended)
- `sap`: Simple Aperture Photometry

**Quality Levels:** `none`, `default`, `hard`, `hardest`

### Chromatic Modeling

| Option | Description | Default |
|--------|-------------|---------|
| `-bp, --bandpass <LABELS>` | Space-separated bandpass labels, one per `--filename`. Activates chromatic mode in the generated `settings.csv` and emits per-band `{pl}_rr_<bp>` / `host_ldc_q*_<bp>` rows in `params.csv`. Repeat a label to share a band across instruments. | None (achromatic) |

When `-f` has ≥2 distinct instruments and `-bp` is omitted, the script logs a warning that the run will stay achromatic and shows the exact command to enable chromatic mode.

### File Management

| Option | Description | Default |
|--------|-------------|---------|
| `-f, --filename <NAME>` | Output filename prefix; accepts multiple instruments | `tess` |
| `-dir <PATH>` | Base directory | current |
| `-o, --overwrite` | Overwrite existing files | False |
| `-r, --results_dir <PATH>` | Update from previous results | None |
| `--lc-only` | Only download the lightcurve, skip generating config files | False |
| `--ttv` | Emit per-transit TTV rows in `params.csv` (count derived from observed transits) | False |

### Interactive Options

| Option | Description |
|--------|-------------|
| `-i, --interactive` | Manual parameter input |
| `-u, --update_db` | Force database updates |
| `--debug` | Detailed diagnostic output |

## Parameter Sources & Databases

### Database Priority and Usage

| Database | Use Case | Parameter Source | Reliability |
|----------|----------|------------------|-------------|
| **NExSci** (`-name`) | Confirmed exoplanets | NASA Exoplanet Archive | Highest |
| **TOI** (`-toi`) | TESS candidates | TFOP database | High |
| **CTOI** (`-ctoi`) | Community candidates | Community observations | Medium |
| **TIC** (`-tic`) | Custom analysis | TIC catalog + manual input | Variable |

## Troubleshooting

### Common Issues and Solutions

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
- Tail the centralized run log: `tail -n 10 ~/.allesfitter/runs.jsonl` (or whatever path `$ALLESFITTER_RUN_LOG` resolves to). Each `start` row carries `pid`, `hostname`, absolute `datadir`, and `run_id`; the matching `end` row carries `status` and `duration_sec`. See the "Tracking long-running fits" use case above.

### Debug Mode

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

## Best Practices

### 1. Parameter Validation
- Always use `--debug` for first-time targets
- Verify stellar parameters match literature values
- Check transit duration consistency between methods
- Review generated plots before fitting

### 2. Data Quality
- Use `pdcsap` over `sap` for SPOC pipeline
- Apply sigma clipping for noisy data: `-sig 3`
- Choose appropriate quality bitmask level
- Inspect lightcurve plots for systematics

### 3. Analysis Strategy
- Start with single sector for parameter estimation
- Use multi-sector data for refined parameters
- Enable `fast_fit` for initial exploration
- Use strict convergence (`ns_tol,0.01`) for final results

### 4. Pipeline Selection
- **SPOC:** Better systematics correction, slower cadence
- **QLP:** Faster processing, higher cadence available
- Compare both pipelines for robust results

## Performance Notes

- **Runtime** depends on data volume and convergence criteria
- **Memory usage** scales with sector count and cadence
- **Convergence** varies by parameter complexity and data quality
- **Parallel processing** significantly reduces fit time

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

## Testing

A pytest-based regression suite under `tests/` pins the chromatic contract and core utilities. Install the optional `test` extras and run:

```bash
pip install -e ".[test]"

# Fast suite (default — excludes @pytest.mark.slow):
pytest tests/

# Chromatic-only:
pytest tests/chromatic/

# Include the end-to-end NS fits (~30 s extra):
pytest tests/chromatic/ -m ''
```

`tests/chromatic/` covers:

- **Scope mapping** — global vs. per-bandpass vs. per-instrument keys, `get_rr_key`/`get_ldc_key` semantics, three-state edge cases (achromatic / single-bandpass / chromatic).
- **Parser errors** — bandpass count mismatch, duplicate rows, unknown bandpass suffix, chromatic-vs-achromatic shape inconsistencies, and orphan per-instrument suffixes (e.g. `host_ld_law_<bandpass>` when no instrument has that name).
- **LD law defaults** — `host_ld_law_<inst>` defaults to `quad` when absent/blank; explicit `none` still disables limb darkening.
- **Likelihood assembly** — `ellc.fluxes` is monkeypatched to assert per-band `rr`, per-inst LDC, and bit-equal shared orbital params across instruments. Includes the shared-bandpass q-space propagation regression: editing `host_ldc_q1_<bp>` in `params.csv` must actually change the LDC vector ellc receives (atol 1e-12).
- **`prepare_allesfit` emission shapes** — pins the four `settings.csv` / `params.csv` shapes the script writes (achromatic, chromatic with inst==bandpass, chromatic with distinct inst/bandpass, shared bandpass) and asserts they all pass `config.init`.
- **Raw-flux clipping** — clipped rows excluded from the fit and retained under `data[inst]['raw_clipped_*']` for the red overlay.
- **End-to-end NS fit** — recovers injected `b_rr_tess` and `b_rr_k2` from synthetic two-band data; achromatic backcompat baseline.
- **Run logger** (`tests/chromatic/test_run_logger.py`) — start/end rows, failure traceback capture, `ALLESFITTER_RUN_LOG` env override, `extra` field merging into start row only, `tail()` semantics, and concurrent-append smoke.

See `docs/chromatic_validation.md` for the full requirement → code → test mapping.

## Citation

If you use this code, please cite:

**Original allesfitter:**
[Günther & Daylan 2021, ApJS, 254, 13](https://ui.adsabs.harvard.edu/abs/2021ApJS..254...13G)

**Original code repository:**
https://github.com/MNGuenther/allesfitter

## Contributing

Issues and pull requests are welcome. Please ensure your contributions maintain compatibility with the original allesfitter framework.

## License

This project extends the original allesfitter package. Please refer to the original license terms.
