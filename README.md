# allesfitter2
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/jpdeleon/allesfitter2)

allesfitter2 is an extension of the [allesfitter](https://github.com/MNGuenther/allesfitter) package, providing a comprehensive framework for global Bayesian inference of photometry and radial velocity (RV) data. It specializes in characterizing exoplanetary systems and eclipsing binaries by combining transit and RV modeling with nested-sampling and MCMC engines. Added utilities streamline downloading TESS, K2, and Kepler light curves and auto-generating every file needed to run a fit.

- **What it models** → [Modeling capabilities](#modeling-capabilities) (below)
- **How to configure each capability** → [notes.md](notes.md) (detailed walkthroughs + decision guide)
- **Open work / package audit** → [TODO.md](TODO.md)

## Modeling capabilities

allesfitter2 builds a single joint likelihood over all your photometric and RV
instruments. The physical transit/eclipse model is evaluated with
[`ellc`](https://github.com/pmaxted/ellc); each capability below is switched on
through `settings.csv` / `params.csv` keys. Follow the link in each row for the
full walkthrough in [notes.md](notes.md).

| Capability | What you get | Key `settings.csv` / `params.csv` | Details |
|---|---|---|---|
| **Transit & eclipse photometry** | Multi-companion transits/occultations via `ellc` | `inst_phot`, `b_rr`, `b_rsuma`, `b_cosi`, `b_epoch`, `b_period` | — |
| **Radial velocity** | Keplerian RV, joint with photometry; second-order RV (`rv2`) supported | `inst_rv`, `b_K`, `b_f_c`, `b_f_s` | — |
| **Multiple companions / EBs** | Any number of companions; eclipsing-binary surface-brightness ratios | `companions_phot`, `companions_rv`, `b_sbratio_<inst>` | — |
| **Chromatic depth** | Separate `Rp/R★` per bandpass, orbit shared globally | `bandpass,<...>`, `b_rr_<band>` | [link](notes.md#chromatic-transit-modeling-per-band-rprs) |
| **Limb darkening** | `none`/`lin`/`quad`/`sing` (LDC3); per-bandpass keying; theoretical coeffs from Claret tables via [limbdark](https://github.com/jpdeleon/limbdark2) | `host_ld_law_<inst>`, `host_ldc_q*_<band>` | [link](notes.md#limb-darkening-keying-band-vs-instrument) |
| **Baseline / detrending** | `offset`, `linear`, `poly_N`, `spline`, GP (Matern32/SHO/real/complex); `hybrid` (analytic) vs `sample` variants | `baseline_flux_<inst>,<model>` | [link](notes.md#detrending-baselines-against-ancillary-covariates-airmass-fwhm-sky-) |
| **Covariate decorrelation** | Detrend flux against named ancillary columns (airmass, FWHM, sky, centroid, background, CBVs…) — same approach for TESS and ground-based data | `baseline_*_<inst>_against,<col>` | [link](notes.md#detrending-baselines-against-ancillary-covariates-airmass-fwhm-sky-) |
| **N-D linear detrending** | Joint linear model over many covariates; `hybrid` marginalises weights analytically (zero fit dims) | `baseline_flux_<inst>,sample_linear_multi` + `_cols` | [link](notes.md#n-d-linear-baseline-detrending-sample_linear_multi--hybrid_linear_multi) |
| **Shared baseline GP** | One celerite GP realization across simultaneous instruments (MuSCAT g/r/i/z common-mode noise) | `baseline_share_flux,<a:b:c>` | [link](notes.md#sharing-a-baseline-gp-across-instruments-joint-celerite-realization) |
| **Noise / error model** | Per-instrument white-noise jitter; correlated noise via GP | `error_flux_<inst>,sample`, `ln_err_flux_<inst>` | — |
| **Dilution / contamination** | Per-instrument depth dilution (`(1−dil)` scaling) with degeneracy guidance | `dil_<inst>` | [link](notes.md#fitting-dilution-dil_inst-in-a-chromatic-model) |
| **Stellar variability / spots / flares** | GP stellar-variability term; spot and flare models | `stellar_var_*`, `spots.py`, `flares/` | — |
| **TARS light curves** | Reader for TARS HLSP files, plus a patch that teaches lightkurve itself to read them (`prepare_allesfit -p tars`) | `allesfitter.tars.read_tars`, `patch_lightkurve` | [link](notes.md#tars-light-curves-not-supported-by-lightkurve) |
| **TTVs** | Per-transit timing offsets via iterative refinement from a prior run (`N` is a global index across all `inst_phot`, see [numbering](notes.md#how-companion_ttv_transit_n-is-numbered)) | `fit_ttvs,True`, `b_ttv_transit_N` | [link](notes.md#using-a-previous-run-for-a-ttv-fit) |
| **Stellar-density prior** | Break the `Rp/R★`–`a/R★`–`i` degeneracy from R★, M★ | `use_host_density_prior,True` + `params_star.csv` | — |
| **Required transit geometry** | Condition an individual photometric companion on producing a primary transit, including eccentric and grazing geometries | `require_<companion>_transit,True` | — |
| **Inference engines** | Nested sampling (evidence + multimodal) and MCMC | `ns_fit` / `mcmc_fit` | — |
| **Warm-start optimizer** | Global optimizer (`differential_evolution` default, parallelizable across CPU cores via `--workers`) lands the sampler in a good basin, with safe acceptance gates and CMA warm-resume | `allesfitter.optimize(...)` | [link](notes.md#warm-starting-mcmc--ns-with-allesfitteroptimize) |
| **Fast / binned evaluation** | Transit-window-only evaluation (`fast_fit`); load-time binning with auto supersampling | `fast_fit,True`, `binning`/`binning_<inst>` | [link](notes.md#choosing-settings--priors-by-use-case) |
| **Raw-flux clipping** | Drop out-of-window points from the fit, keep them flagged on plots | `flux_min_raw` / `flux_max_raw` | [link](notes.md#raw-flux-outlier-clipping) |
| **Data preparation** | Auto-download + multi-pipeline light curves (TESS/K2/Kepler) and full config generation | `prepare_allesfit` CLI | [CLI](#command-line-options) |
| **Robust post-processing** | OOM-safe diagnostic plots for high-dim fits; centralized JSONL run log | `ns_output` / `mcmc_output`, `log_run` | [plots](notes.md#oom-safe-diagnostic-plots-for-high-dim-fits) · [run log](notes.md#tracking-long-running-fits-with-the-centralized-run-log) |
| **Blind transit search** | Detrend with the best fit, mask known planets, iteratively search the residual with TLS until SDE drops below threshold; per-candidate figure + quicklook-style `.h5` (readable by `--h5`) | `transit-search` CLI | [CLI](#blind-transit-search-transit-search) |

**Inputs.** Two CSVs are a contract: `settings.csv` declares *what* is modelled
(companions, instruments, baseline/error model per instrument, achromatic vs
chromatic); `params.csv` declares *every* free/fixed number and its prior
(`uniform`, `normal`, `trunc_normal`). `params_star.csv` carries R★/M★ for the
density prior and post-fit derivation. Strict validators catch mismatches at
`config.init` and name the exact key to fix. See
[Effective use of allesfitter2](notes.md#effective-use-of-allesfitter2).

## Installation

```bash
git clone https://github.com/jpdeleon/allesfitter2.git
cd allesfitter2
```

<details open>
<summary><strong>Install with uv (recommended)</strong></summary>

```bash
uv sync --no-dev
```


</details>

<details>
<summary><strong>Install with pip</strong></summary>

```bash
python -m pip install .
```


</details>

On Debian/Ubuntu Linux, the GNU Fortran runtime may be required to run `ellc`:

```bash
sudo apt-get install -y libgfortran5
```

If you encounter another problem with `ellc`, use the custom installation
script, which clones `ellc` and builds the Fortran extension modules:

```bash
cd allesfitter2
bash install.sh
```

## Quick Start

```bash
prepare_allesfit -name "HD 39091" -s 1 -f tess
```

This creates a complete analysis directory:

```
HD39091/
├── params.csv          # parameters to fit/fix and their priors
├── settings.csv        # transit/baseline model, sampler, instruments
├── params_star.csv     # stellar params for derivation / density prior
├── run.py              # entry point for running allesfitter
├── tess.csv            # downloaded light curve
└── tess.png            # quick-look plot
```

### Running the analysis

1. **Initial setup check** — plots the model at the initial guess (always look before you fit):
   ```bash
   cd HD39091/
   python run.py
   ```
2. **Full analysis** — edit `run.py` to uncomment the fitting lines:
   ```python
   import allesfitter
   allesfitter.show_initial_guess('.')
   allesfitter.optimize('.')
   allesfitter.ns_fit('.')      # nested sampling
   allesfitter.ns_output('.')   # parameter derivation
   ```
   With more than one `inst_phot`, `show_initial_guess` writes the per-transit
   plots as one multi-page PDF per planet
   (`results/initial_guess_per_transit_<companion>.pdf`), ordered
   chronologically with one transit per page and the instrument in each panel
   title — see [details](notes.md#per-transit-plot-files-from-show_initial_guess).
   `ns_output` / `mcmc_output` write the same one-PDF-per-planet layout for
   their posterior per-transit plots
   (`{ns,mcmc}_results/{ns,mcmc}_fit_per_transit_<companion>.pdf`).
3. **Execute:** `python run.py` — MCMC and nested-sampling outputs are saved separately
   inside the data directory, so the above target writes to `HD39091/mcmc_results/` and
   `HD39091/ns_results/`. The same holds for the CLI: `uv run allesfitter ns-fit HD39091`
   writes to `HD39091/ns_results/`.

   Set the `ALLESFITTER_RESULTS_DIR` environment variable to collect results from many
   targets under one shared root instead. Each target then gets its own subdirectory,
   named after its data directory's basename — with
   `ALLESFITTER_RESULTS_DIR=~/ql/allesfitter` the target above writes to
   `~/ql/allesfitter/HD39091/mcmc_results/`. The browser workbench
   (`allesfitter gui`) sets this to its workspace automatically, which defaults to
   `~/ql/allesfitter`.

By default `fast_fit,True` restricts evaluation to transit windows. For a quick
preliminary run, loosen `ns_tol` (e.g. `10` or `100`); use `ns_tol,0.01` only for
the final run. The recommended workflow — warm-start with `optimize()`, iterate
cheap→expensive, let the data set GP priors — is in
[Effective use of allesfitter2](notes.md#effective-use-of-allesfitter2).
Accepted optimizer results are written back in the original `params.csv`
epoch frame; with `shift_epoch,True`, subsequent commands recenter both the
optimized epoch and its prior consistently inside the data window.

Compare nested-sampling evidences from Python with
`allesfitter.compare_logz(["model_a", "model_b"])`, or from the command line with
`python -m allesfitter.postprocessing.nested_sampling_compare_logZ model_a model_b`.

## Command Line Options

### CLI command reference

Run commands from the project environment as `uv run allesfitter <command>`.
Use `uv run allesfitter <command> --help` for the complete options of any
command.

| Command | Description |
|---------|-------------|
| `prepare` | Download TESS/Kepler/K2 data and prepare configuration files |
| `grid <grid-dir>` | Run models from a `grid.csv` manifest |
| `show-initial-guess <dir>` | Plot data with the current `params.csv` values |
| `optimize <dir>` | Globally optimize parameters to warm-start inference |
| `mcmc-fit <dir>` | Run or resume emcee MCMC sampling |
| `mcmc-output <dir>` | Generate MCMC posteriors, plots, and summaries |
| `ns-fit <dir>` | Run nested sampling |
| `ns-output <dir>` | Generate nested-sampling posteriors, evidence, and plots |
| `show-settings <dir>` | Display `settings.csv` as grouped Rich tables |
| `show-params <dir>` | Display fitted and fixed parameters from `params.csv` |
| `show-results <dir>` | Display available MCMC/NS posterior and derived-result tables |
| `transit-search <results-dir>` | Blind TLS search for un-modeled transits, after detrending with the best fit and masking known planets |
| `gui` | Run the target, job, configuration, and results workbench |

### Blind transit search (`transit-search`)

After a fit has completed, `transit-search` looks for additional
un-modeled transiting signals in the residual light curve:

1. Loads the posterior-median parameters from the given `mcmc_results` or
   `ns_results` directory — the best-fit baseline (GP or otherwise) and
   every known companion's ephemeris.
2. Reconstructs each instrument's full raw light curve (before any
   `fast_fit` windowing) and subtracts the best-fit baseline model from it.
3. Masks out every known companion's transits (period/epoch from the
   posterior, duration from the transit-chord equation).
4. Runs `transitleastsquares` on what's left, masking each new detection
   and repeating, until the found signal's SDE drops below
   `--sde-threshold` (default `5.0`).

```bash
uv run allesfitter transit-search HD39091/ns_results --sde-threshold 6
```

For each candidate above threshold, it writes to `--outdir` (default
`<target>/transit_search_results/`):

- `candidate_<N>.pdf` — raw light curve, flattened (detrended) light curve
  with the candidate's TLS model overlaid, the TLS periodogram (harmonics
  and known-planet periods marked), and phase-folded data + model.
- `candidate_<N>_tls.h5` — a quicklook-format TLS results file, readable by
  `prepare -tic/-toi ... --h5 candidate_<N>_tls.h5` to seed a new
  `params.csv` companion row for a confirmed candidate (see
  [`--h5` ephemeris seeding](#raw-tic-transit-parameters--tic-and---h5-ephemeris-seeding)).
- `candidates_summary.csv` — period/epoch/duration/depth/SDE/SNR for every
  candidate found.

| Option | Description | Default |
|--------|-------------|---------|
| `--sde-threshold` | Keep searching while the found signal's SDE stays at or above this value | `5.0` |
| `--period-min`, `--period-max` | Period search range (days) | TLS's own default |
| `--mask-width-factor` | Mask known companions out to this many times their transit duration | `1.5` |
| `-o, --outdir <DIR>` | Output directory | `<target>/transit_search_results` |
| `-e, --file-extension` | Figure format (`pdf`, `png`, `jpg`, `svg`, `webp`) | `.pdf` |
| `--max-candidates` | Safety cap on the number of candidates kept | `20` |
| `-m, --mission <NAME>` | `tess`, `k2`, or `kepler`; sets the h5 files' BJD offset | `tess` |

Candidates should always be vetted by eye before trusting them — the
best-fit baseline is only well constrained near the known transits it was
conditioned on (especially with `fast_fit` enabled), so a long-period
candidate close to the search's upper period bound can be leftover,
un-flattened stellar variability rather than a real transit; the raw and
flattened light-curve panels in each candidate's figure make that easy to
check directly.

### Browser workbench over SSH

Run the server on the remote machine, bound to localhost:

```bash
uv run allesfitter gui --workspace /path/to/allesfitter-workspace
```

Then forward the port from your local machine:

```bash
ssh -L 5100:127.0.0.1:5100 user@remote-machine
```

Open `http://127.0.0.1:5100`. The workspace stores target directories, their
results, job logs, and `workbench.sqlite3`; commands run non-interactively
without a shell. Omit `--workspace` to use `~/ql/allesfitter`, which keeps
every workbench target and its fits in one place.

`show-results <dir>` resolves results the same way as the fit commands (inside
`<dir>`, or under `$ALLESFITTER_RESULTS_DIR/<dir-basename>` when that variable
is set), reads `mcmc_results/` and `ns_results/` there (falling back to a
`results/` subdirectory), and shows both samplers when both outputs are available.

The following option tables document the `prepare` command.

### Target selection (mutually exclusive, required)

| Option | Description | Example |
|--------|-------------|---------|
| `-toi <ID>` | TOI (TESS Object of Interest) ID | `-toi 144` |
| `-tic <ID>` | TIC (TESS Input Catalog) ID | `-tic 261136679` |
| `-ctoi <ID>` | CTOI (Community TOI) ID | `-ctoi 12345` |
| `-name <NAME>` | Target name (NExSci database) | `-name "HD 39091"` |

### Data selection (mutually exclusive, required)

Exactly one of `-s`, `-c`, or `-q` must be given, matched to the mission supplied via `-m`.

| Option | Mission | Description | Example |
|--------|---------|-------------|---------|
| `-s, --sector <S>` | TESS | Sector number(s) | `-s 1 3 5` |
| `-c, --campaign <C>` | K2 | Campaign number(s) (supports `11a`, `11b`) | `-c all` |
| `-q, --quarter <Q>` | Kepler | Quarter number(s) | `-q 4` |

Each accepts explicit numbers/labels (`-s 1 3 5`), `-1` for the most recent (default), `0` for the first, or `all` for every available window.

### Raw TIC transit parameters (`-tic`) and `--h5` ephemeris seeding

A `-tic <ID>` target has no catalog entry, so its ephemeris comes from the
command line (or an interactive prompt) instead of TOI/CTOI/NExSci lookups.

| Option | Description | Units |
|--------|-------------|-------|
| `--period`, `--period-err` | Orbital period (+ uncertainty) | days |
| `--epoch`, `--epoch-err` | Transit epoch (+ uncertainty) | BJD |
| `--duration`, `--duration-err` | Transit duration (+ uncertainty) | hours |
| `--depth`, `--depth-err` | Transit depth (+ uncertainty) | ppm |
| `--h5 <PATH>` | A [quicklook](https://github.com/jpdeleon/quicklook) TLS results `.h5` file — the primary `..._tls.h5` or an iterative-search companion `..._tls_p2.h5` — used to seed period/epoch/duration/depth (and period/depth uncertainties, where TLS reports them) | — |

Any field the command line doesn't supply directly, and `--h5` doesn't
supply either, falls back to an interactive prompt — so an explicit flag
always overrides the matching `--h5` value, and `--h5` fills in the rest.
TLS carries no epoch or duration uncertainty, so `--epoch-err` and
`--duration-err` still need an explicit value or a prompt answer even when
`--h5` is given. Example, for a candidate not yet in the TOI catalog:

```bash
uv run allesfitter prepare -tic 436478932 -s 83 \
  --h5 ~/ql/TIC436478932_s83_pdcsap_sc_tls_p2.h5 \
  --epoch-err 0.01 --duration-err 0.2
```

`--h5` also works with `-toi <ID>`, for a companion quicklook's iterative
TLS search found that isn't in the TOI catalog yet. The target's known
catalog planet(s) are prepared as usual (one `params.csv` companion row
each, `b`, `c`, ... in catalog order), and the `--h5` candidate is appended
as one more companion after them — the catalog rows are never modified.
`--period`/`--epoch`/etc. still take priority over `--h5` for the appended
row, same as raw `-tic`; anything neither supplies falls back to a prompt.

```bash
uv run allesfitter prepare -toi 1234 -s 83 \
  --h5 ~/ql/TIC.../TIC..._s83_pdcsap_sc_tls_p2.h5
```

#### Light curve reuse

`--h5` also lets `prepare` reuse the light curve already inside the h5 file
instead of re-downloading it — but only for the exact `--sector`/`--pipeline`
that h5 was built from (compared against the `sector`/`pipeline` quicklook
recorded alongside the light curve). Any other sector/pipeline this run
touches is downloaded normally, so mixing a reused segment with freshly
downloaded ones (e.g. `-s 37 90` where only 37 is in the h5) works fine.
This only applies to `-m tess` and only when the h5 file actually carries a
light curve — an iterative-search companion (`..._tls_p2.h5`,
`..._tls_p3.h5`, ...) only stores the TLS results, not the light curve, so
`--h5 ..._tls_p2.h5` always downloads normally; point `--h5` at the primary
`..._tls.h5` instead to reuse its light curve. SPOC's PDCSAP/SAP dilution
comparison is skipped for a reused light curve (no FITS header to read
`CROWDSAP` from).

### Data processing options

| Option | Description | Default |
|--------|-------------|---------|
| `-m, --mission <NAME>` | `tess`, `k2`, or `kepler` | `tess` |
| `-e, --exptime <SEC>...` | Exposure time(s) in seconds (`120`, `200`, `600`, `1800`); written to `settings.csv` as `t_exp_<inst>` in days. Accepts multiple, one per `--pipeline`, or a single value shared by all | None (lightkurve metadata) |
| `-p, --pipeline <NAME>...` | Data pipeline(s). A single value (default) downloads only the first `--filename`, as before. Give one `--pipeline` per `--filename` to download every instrument — each from its own pipeline/exptime — in one run, e.g. `-f spoc120 qlp600 -p spoc qlp -e 120 600` | `spoc` |
| `-lc, --lc_type <TYPE>` | Light curve type | `pdcsap` |
| `-sig, --sigma <N>` | Sigma clipping threshold | None |
| `-qb, --quality <LEVEL>` | Quality bitmask (`none`/`default`/`hard`/`hardest`) | `default` |

**Pipelines:** TESS → `spoc` (recommended), `qlp`; K2 → `k2`, `everest`, `k2sff`; Kepler → `kepler`.
**Light curve types:** `pdcsap` (recommended), `sap`.

### Chromatic modeling

| Option | Description | Default |
|--------|-------------|---------|
| `-bp, --bandpass <LABELS>` | Space-separated bandpass labels, one per `--filename`. Activates chromatic mode and emits per-band `{pl}_rr_<bp>` / `host_ldc_q*_<bp>` rows. Repeat a label to share a band across instruments. | None (achromatic) |

When `-f` has ≥2 distinct instruments and `-bp` is omitted, the script warns that the run stays achromatic and prints the command to enable chromatic mode. See [Chromatic transit modeling](notes.md#chromatic-transit-modeling-per-band-rprs).

### File management

| Option | Description | Default |
|--------|-------------|---------|
| `-f, --filename <NAME>` | Output filename prefix; accepts multiple instruments | `tess` |
| `-dir <PATH>` | Base directory | current |
| `-o, --overwrite` | Overwrite existing files | False |
| `-r, --results_dir <PATH>` | Update from previous results (e.g. for TTV fits) | None |
| `--lc-only` | Only download the light curve, skip config generation | False |
| `--ttv` | Emit per-transit TTV rows in `params.csv` | False |

### Interactive options

| Option | Description |
|--------|-------------|
| `-i, --interactive` | Manual parameter input |
| `-u, --update_db` | Force database updates |
| `--debug` | Detailed diagnostic output |

### Non-interactive / headless runs

The fitter runs a few "luser proof" sanity checks that normally ask a
yes/no question on the terminal (e.g. *"the initial guess for `b_epoch`
lies more than 3 sigma from its prior — continue?"*, or *"output file
already exists — overwrite?"*). Under a batch scheduler, subprocess, or
notebook kernel with no attached TTY these are auto-answered with the
safe default (continue / overwrite) plus a warning, so a run never hangs
waiting for input. Set `ALLESFITTER_NONINTERACTIVE=1` to force that
non-interactive behaviour even when a terminal is attached.

`ALLESFITTER_NONINTERACTIVE=1` additionally makes `prepare_allesfit.py`
auto-select the **shortest** available cadence when a search returns
multiple exposure times (e.g. TESS Sector 64 in both 20 s and 120 s),
instead of exiting to let you re-run with `-e/--exptime`. It logs which
cadence it picked; pass `-e` to override.

## Usage notes, modeling walkthroughs & troubleshooting

Detailed per-feature walkthroughs and all advisory material now live in
**[notes.md](notes.md)**:

- **Modeling walkthroughs** — [chromatic](notes.md#chromatic-transit-modeling-per-band-rprs), [shared baseline GP](notes.md#sharing-a-baseline-gp-across-instruments-joint-celerite-realization), [covariate detrending](notes.md#detrending-baselines-against-ancillary-covariates-airmass-fwhm-sky-), [N-D linear detrending](notes.md#n-d-linear-baseline-detrending-sample_linear_multi--hybrid_linear_multi), [warm-start `optimize()`](notes.md#warm-starting-mcmc--ns-with-allesfitteroptimize), [dilution](notes.md#fitting-dilution-dil_inst-in-a-chromatic-model), [limb-darkening keying](notes.md#limb-darkening-keying-band-vs-instrument), [OOM-safe plots](notes.md#oom-safe-diagnostic-plots-for-high-dim-fits), [run log](notes.md#tracking-long-running-fits-with-the-centralized-run-log), [pipelines](notes.md#lightcurves-from-different-pipelines) / [exposure times](notes.md#lightcurves-from-different-exposure-times), [TTV fits](notes.md#using-a-previous-run-for-a-ttv-fit), [performance & caching](notes.md#performance--caching-simulate_pdf-disk-cache)
- **Decision guide** — [effective use](notes.md#effective-use-of-allesfitter2), [settings & priors by use case](notes.md#choosing-settings--priors-by-use-case), [prior cookbook](notes.md#prior-cookbook), [best practices](notes.md#best-practices)
- **Reference** — [parameter sources](notes.md#parameter-sources--databases), [troubleshooting](notes.md#troubleshooting), [TIC contratio vs SPOC CROWDSAP](notes.md#tic-contratio-vs-spoc-crowdsap)

## Testing

A pytest regression suite under `tests/` pins the chromatic contract and core utilities.

```bash
pip install -e ".[test]"

pytest tests/                 # fast suite (excludes @pytest.mark.slow)
pytest tests/chromatic/       # chromatic-only
pytest tests/chromatic/ -m '' # include end-to-end NS fits (~30 s extra)
```

`tests/chromatic/` covers scope mapping (global vs per-bandpass vs per-instrument keys), parser-error messages, LD-law defaults, likelihood assembly (monkeypatched `ellc.fluxes`), `prepare_allesfit` emission shapes, raw-flux clipping, an end-to-end two-band NS fit, and the run logger. See `docs/chromatic_validation.md` for the requirement → code → test mapping.


### Transit geometry parameterization

For a companion known to transit, set
`require_<companion>_transit,True`. allesfitter then rejects sampled primary
impact parameters outside the eccentricity-aware transit boundary, while
still permitting grazing transits. This prevents the fit from escaping to a
non-transiting geometry in which the radius ratio becomes unconstrained and a
baseline model can absorb the transit.

The current orbital parameterization samples `cosi` directly. This is
physically natural for an isotropic orientation prior, but inefficient when
the analysis is explicitly conditioned on a known transit: the valid region
is a correlated wedge in `cosi`–`rsuma`–eccentricity–argument-of-periastron
space, so broad independent bounds can produce many rejected proposals.
Moreover, applying a hard transit cut to an independent `cosi` prior weights
the marginal orbital prior by the geometric transit probability. Nested
sampling evidence then includes that probability rather than being strictly
conditional on the known transit.

A preferable future sampling parameter is the normalized primary impact
parameter

```text
beta = b_primary / (1 + rr),    0 <= beta <= 1
cosi = beta * rsuma * (1 + e sin(omega)) / (1 - e^2)
```

A uniform `beta` prior gives a rectangular sampling domain and the usual
uniform impact-parameter prior conditioned on a transit. Normalizing by
`1 + rr` is preferable to sampling raw `b_primary`, whose upper boundary
moves with the radius ratio. For a fully physical chromatic model, sampling
`a/Rstar` alongside `beta` would also be cleaner than combining a shared
`rsuma` with band-dependent radius ratios. Until that parameterization is
implemented, use `require_<companion>_transit,True` and reasonably tight,
physically informed `cosi` and `rsuma` bounds.

See the executable
[transit-geometry parameterization notebook](notebooks/transit_geometry_beta_experiment.ipynb)
for the derivation, induced-prior comparison, an `emcee` efficiency experiment,
and a comparison with the published Espinoza (2018) $(r_1,r_2)$ mapping.
The companion
[duration–orbital-scale–density notebook](notebooks/transit_duration_arstar_density_experiment.ipynb)
derives the transformations among $T_{14}$, $a/R_\star$, and $\rho_\star$ and
separates sampling-coordinate effects from external-prior constraints.

## Citation

If you use this code, please cite the original allesfitter:

**[Günther & Daylan 2021, ApJS, 254, 13](https://ui.adsabs.harvard.edu/abs/2021ApJS..254...13G)** — code: https://github.com/MNGuenther/allesfitter

## Contributing

Issues and pull requests are welcome. Please keep contributions compatible with the original allesfitter framework, and see [TODO.md](TODO.md) for the current backlog and package-audit items.

## License

This project extends the original allesfitter package. Please refer to the original license terms.
