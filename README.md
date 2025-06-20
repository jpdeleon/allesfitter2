# allesfitter2

An extension of the original allesfitter package that streamlines the process of downloading TESS lightcurves and automatically generating all necessary files to run allesfitter.

## Features

- **Automated lightcurve download** from TESS mission data
- **Multi-pipeline support** (SPOC, QLP) with configurable parameters
- **Parameter derivation** from multiple astronomical databases
- **Flexible sector selection** (single, multiple, or all available sectors)
- **Built-in quality control** with outlier removal and quality masking
- **Theoretical limb darkening** coefficients from Claret tables using [limbdark](https://github.com/jpdeleon/limbdark2)

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

### Data Selection (required)

| Option | Description | Example |
|--------|-------------|---------|
| `-s, --sector <SECTORS>` | TESS sectors | `-s 1 3 5` |
| | Most recent sector | `-s -1` |
| | First sector | `-s 0` |
| | All available sectors | `-s all` |

### Data Processing Options

| Option | Description | Default |
|--------|-------------|---------|
| `-e, --exptime <SEC>` | Exposure time filter | None |
| `-p, --pipeline <NAME>` | Data pipeline | `spoc` |
| `-lc, --lc_type <TYPE>` | Light curve type | `pdcsap` |
| `-sig, --sigma <N>` | Sigma clipping threshold | None |
| `-qb, --quality <LEVEL>` | Quality bitmask | `default` |

**Pipeline Options:**
- `spoc`: TESS Science Processing Operations Center (recommended)
- `qlp`: Quick Look Pipeline

**Light Curve Types:**
- `pdcsap`: Pre-search Data Conditioning SAP (recommended)
- `sap`: Simple Aperture Photometry

**Quality Levels:** `none`, `default`, `hard`, `hardest`

### File Management

| Option | Description | Default |
|--------|-------------|---------|
| `-f, --filename <NAME>` | Output filename prefix | `tess` |
| `-dir <PATH>` | Base directory | current |
| `-o, --overwrite` | Overwrite existing files | False |
| `-r, --results_dir <PATH>` | Update from previous results | None |

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
