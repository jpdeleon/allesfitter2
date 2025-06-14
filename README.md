This code is an extension of the original allesfitter package.

## Example usage

```bash
$ prepare_allesfit -toi 3933 -s all -p spoc -e 120 
```

The command above will create four files needed to run allesfitter:
* all available TESS light curve produced by SPOC pipeline, with exposure time set to 120s
* `params.csv`, 
* `settings.csv`,
* `params_star.csv`,
* `run.py`

The TESS light curve (.csv) file should be renamed to `tess.csv`. Inspect the values and priors in `params.csv` and `settings.csv`.

Then run the fit using
```bash
$ python run.py
```

If you use this code, please cite the original [author](https://github.com/MNGuenther/allesfitter)