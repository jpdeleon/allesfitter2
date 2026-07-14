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
| **TTVs** | Per-transit timing offsets via iterative refinement from a prior run | `fit_ttvs,True`, `b_ttv_transit_N` | [link](notes.md#using-a-previous-run-for-a-ttv-fit) |
| **Stellar-density prior** | Break the `Rp/R★`–`a/R★`–`i` degeneracy from R★, M★ | `use_host_density_prior,True` + `params_star.csv` | — |
| **Inference engines** | Nested sampling (evidence + multimodal) and MCMC | `ns_fit` / `mcmc_fit` | — |
| **Warm-start optimizer** | Global optimizer (CMA-ES default) lands the sampler in a good basin, with safe acceptance gates and CMA warm-resume | `allesfitter.optimize(...)` | [link](notes.md#warm-starting-mcmc--ns-with-allesfitteroptimize) |
| **Fast / binned evaluation** | Transit-window-only evaluation (`fast_fit`); load-time binning with auto supersampling | `fast_fit,True`, `binning`/`binning_<inst>` | [link](notes.md#choosing-settings--priors-by-use-case) |
| **Raw-flux clipping** | Drop out-of-window points from the fit, keep them flagged on plots | `flux_min_raw` / `flux_max_raw` | [link](notes.md#raw-flux-outlier-clipping) |
| **Data preparation** | Auto-download + multi-pipeline light curves (TESS/K2/Kepler) and full config generation | `prepare_allesfit` CLI | [CLI](#command-line-options) |
| **Robust post-processing** | OOM-safe diagnostic plots for high-dim fits; centralized JSONL run log | `ns_output` / `mcmc_output`, `log_run` | [plots](notes.md#oom-safe-diagnostic-plots-for-high-dim-fits) · [run log](notes.md#tracking-long-running-fits-with-the-centralized-run-log) |

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

To include the optional Web GUI:

```bash
uv sync --extra webgui --no-dev
```

</details>

<details>
<summary><strong>Install with pip</strong></summary>

```bash
python -m pip install .
```

To include the optional Web GUI:

```bash
python -m pip install ".[webgui]"
```

</details>

If you encounter a problem with ellc, use the custom installation script, which clones ellc and builds the Fortran extension modules:
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
   allesfitter.ns_fit('.')      # nested sampling
   allesfitter.ns_output('.')   # parameter derivation
   ```
3. **Execute:** `python run.py` — MCMC and nested-sampling outputs are saved separately
   in `HD39091/mcmc_results/` and `HD39091/ns_results/`, respectively. Existing runs
   using the legacy `HD39091/results/` directory remain readable.

By default `fast_fit,True` restricts evaluation to transit windows. For a quick
preliminary run, loosen `ns_tol` (e.g. `10` or `100`); use `ns_tol,0.01` only for
the final run. The recommended workflow — warm-start with `optimize()`, iterate
cheap→expensive, let the data set GP priors — is in
[Effective use of allesfitter2](notes.md#effective-use-of-allesfitter2).

Compare nested-sampling evidences from Python with
`allesfitter.compare_logz(["model_a", "model_b"])`, or from the command line with
`python -m allesfitter.postprocessing.nested_sampling_compare_logZ model_a model_b`.

## Command Line Options

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

### Data processing options

| Option | Description | Default |
|--------|-------------|---------|
| `-m, --mission <NAME>` | `tess`, `k2`, or `kepler` | `tess` |
| `-e, --exptime <SEC>` | Exposure time in seconds (`120`, `200`, `600`, `1800`); written to `settings.csv` as `t_exp_<inst>` in days | None (lightkurve metadata) |
| `-p, --pipeline <NAME>` | Data pipeline | `spoc` |
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

## Web GUI (`allesfitter-gui`)

A browser-based GUI for configuring, launching, and reviewing **multi-band /
multi-epoch transit fits** without hand-editing `settings.csv` / `params.csv`.
The GUI is optional: a normal `pip install allesfitter` installs only the core
package. Install the `webgui` extra when you want the browser interface; both
`allesfitter-gui` and `allesfitter gui` print the required install command if
that extra is absent.
It is a thin FastAPI + Jinja2 shell over the existing engine
(`allesfitter/webgui/`): a form generates a valid allesfitter datadir, stages the
light curves, validates the config through `Basement` before launching, then runs
each fit as its own detached subprocess (allesfitter's config is module-global) with
live-log streaming and rendered result figures.

It reproduces the full TOI-6715 single-fit vocabulary: many instruments in one
joint fit, chromatic radius-ratio/limb-darkening keyed by **bandpass**, per-instrument
heterogeneous baselines including joint/coupled-GP share groups
(`baseline_share_flux`) and `hybrid_linear_multi` covariate detrending.

**Prepare from catalog.** The **Prepare** page wires the `prepare_allesfit` pipeline
into the GUI: give it a target (name/TOI/TIC/CTOI) and a sector/campaign/quarter and
it runs `prepare_allesfit` as a subprocess to **auto-download** the TESS/K2/Kepler
light curves and **auto-generate** the whole datadir (light-curve CSVs, `params.csv`,
`settings.csv`, GP/dilution/TTV priors) — no manual upload or hand-typed ephemerides.
A live archive lookup fills the sector chips, and an **exposure-time dropdown**
lets you pin the cadence (passed through as `-e`): TESS offers its standard set
(20/120/600/1800 s) and other missions list what the archive reports. Leave it on
*Auto* for a single cadence, or pick one when several are available (e.g. TESS
20 s vs 120 s).
The run appears on **Jobs** as `preparing`. Preparation validates through
`Basement`, renders and saves `show_initial_guess`, then stops in the `prepared`
state. Open **Review** to inspect that preview and edit `params.csv` or
`settings.csv` directly. **Run MCMC** and **Run Nested Sampling** each save the
editors, validate the revised configuration, refresh the initial-guess preview,
and only then launch the selected sampler through the same subprocess launcher as
a manual fit. Requires network access (not `--no-network`). Choose the instructions
for your package manager:

<details open>
<summary><strong>Run with uv (recommended)</strong></summary>

```bash
uv sync --extra webgui
uv run allesfitter-gui             # serve at http://127.0.0.1:5100
uv run allesfitter-gui --runs-root ./runs --toi-csv data/TOIs.csv --port 8080
uv run allesfitter-gui --no-network
uv run allesfitter gui --reload    # development server with automatic reload
```

</details>

<details>
<summary><strong>Run with pip</strong></summary>

```bash
python -m pip install -e ".[webgui]"
allesfitter-gui                    # serve at http://127.0.0.1:5100
allesfitter-gui --runs-root ./runs --toi-csv data/TOIs.csv --port 8080
allesfitter-gui --no-network       # disable NASA Exoplanet Archive auto-fill
allesfitter gui --reload           # development server with automatic reload
```

</details>

For a standalone loopback deployment with an explicit runs directory:

```bash
uv run allesfitter-gui --host 127.0.0.1 --port 5100 --runs-root ./webgui_runs
```

When a reverse proxy exposes the GUI below a URL prefix, pass that same public
prefix explicitly. The proxy should strip the prefix before forwarding requests;
generated links, static assets, browser actions, redirects, and result files will
retain it:

```bash
uv run allesfitter-gui \
  --host 127.0.0.1 \
  --port 5001 \
  --root-path /allesfitter \
  --runs-root /path/to/persistent/allesfitter-runs \
  --max-concurrent-fits 2 \
  --max-queued-fits-per-user 3 \
  --no-kill-existing
```

An empty `--root-path` (the default) keeps the standalone URLs rooted at `/`.
The safe default is to fail if the port is occupied; `--kill-existing` is an
explicit development-only opt-in.

Fit jobs use a durable FIFO queue. `--max-concurrent-fits` bounds total running
fits and `--max-queued-fits-per-user` bounds each owner's pending fits. Running
jobs are placed in their own process groups, so Stop cancels child workers as
well as the launcher. A graceful GUI shutdown leaves fits running. On restart,
the scheduler verifies the recorded PID identity and reconciles each job using
the worker's durable `results/exit.json` marker. A vanished process without that
marker is failed, never assumed successful. For operational restarts, stop the
GUI, restart it with the same database and runs root, and confirm recovered
states on **Jobs**; use the Stop action before host shutdown when fits must not
survive.

Pages: **Targets** (status badges), **Prepare** (catalog target + sector →
auto-downloaded, validated, ready-to-fit run), **New fit** (form + Validate/Run +
ephemeris auto-fill), **Jobs** (live status + streaming log + Fit/Stop actions),
**Results** (figures + `log(Z)`).
Experiment-grid / `log(Z)` model comparison (reusing `run_allesfitter_grid`) and RV
fitting are planned follow-on phases.

## Testing

A pytest regression suite under `tests/` pins the chromatic contract and core utilities (546+ collected tests).

```bash
pip install -e ".[test]"

pytest tests/                 # fast suite (excludes @pytest.mark.slow)
pytest tests/chromatic/       # chromatic-only
pytest tests/chromatic/ -m '' # include end-to-end NS fits (~30 s extra)
```

`tests/chromatic/` covers scope mapping (global vs per-bandpass vs per-instrument keys), parser-error messages, LD-law defaults, likelihood assembly (monkeypatched `ellc.fluxes`), `prepare_allesfit` emission shapes, raw-flux clipping, an end-to-end two-band NS fit, and the run logger. See `docs/chromatic_validation.md` for the requirement → code → test mapping.

`tests/unit/webgui/` covers the web GUI: config generation round-tripped through `Basement`, chromatic-by-bandpass keys, share-group / covariate baselines, staging, the run registry + job seam, the subprocess launcher, result discovery, the `prepare_allesfit` integration (argv building, datadir discovery, the prepare finalize state machine, and the `/prepare` + `/jobs/fit` routes), and FastAPI route smoke tests. Engine-dependent tests skip cleanly where the compiled deps are unavailable.

## Citation

If you use this code, please cite the original allesfitter:

**[Günther & Daylan 2021, ApJS, 254, 13](https://ui.adsabs.harvard.edu/abs/2021ApJS..254...13G)** — code: https://github.com/MNGuenther/allesfitter

## Contributing

Issues and pull requests are welcome. Please keep contributions compatible with the original allesfitter framework, and see [TODO.md](TODO.md) for the current backlog and package-audit items.

## License

This project extends the original allesfitter package. Please refer to the original license terms.
