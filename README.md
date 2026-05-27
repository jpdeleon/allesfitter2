# allesfitter2

An extension of the original allesfitter package that streamlines the process of downloading TESS, K2, and Kepler lightcurves and automatically generating all necessary files to run allesfitter.

## Features

- **Automated lightcurve download** from TESS, K2, and Kepler mission data
- **Multi-pipeline support** (SPOC, QLP, EVEREST, K2SFF) with configurable parameters
- **Parameter derivation** from multiple astronomical databases (NExSci, TOI, CTOI, TIC)
- **Flexible time-window selection** — TESS sectors, K2 campaigns (including split campaigns 11a/11b), and Kepler quarters (single, multiple, or all)
- **Chromatic transit modeling** — fit a separate `Rp/Rs` per bandpass while keeping orbital parameters globally shared
- **Strict configuration validation** — clear errors for bandpass/instrument count mismatch, duplicate params, unknown bandpass suffixes, or chromatic/achromatic shape inconsistencies (no more silent fallback)
- **Raw-flux outlier clipping** via `flux_min_raw` / `flux_max_raw` — clipped points are removed from the fit but overlaid in red on `initial_guess.pdf`
- **Built-in quality control** with outlier removal and quality masking
- **Theoretical limb darkening** coefficients from Claret tables using [limbdark](https://github.com/jpdeleon/limbdark2)
- **Test suite** under `tests/chromatic/` pins the chromatic contract with 46 unit, parsing, likelihood-assembly, and end-to-end fit tests

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
- **Parser errors** — bandpass count mismatch, duplicate rows, unknown bandpass suffix, chromatic-vs-achromatic shape inconsistencies.
- **Likelihood assembly** — `ellc.fluxes` is monkeypatched to assert per-band `rr`, per-inst LDC, and bit-equal shared orbital params across instruments.
- **Raw-flux clipping** — clipped rows excluded from the fit and retained under `data[inst]['raw_clipped_*']` for the red overlay.
- **End-to-end NS fit** — recovers injected `b_rr_tess` and `b_rr_k2` from synthetic two-band data; achromatic backcompat baseline.

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
