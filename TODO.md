# TODO

## Support log-uniform (Jeffreys) priors for transit/orbital parameters

allesfitter only supports `uniform`, `normal`, and `trunc_normal` priors. There is
no log-uniform prior, and no transit/orbital parameter (`b_period`, `b_rr`,
`b_rsuma`, `b_cosi`, `b_epoch`, `b_K`, `b_q`, `dil_*`, …) has a built-in
log-space variant. Only noise/GP parameters do (`ln_err_flux_<inst>`,
`ln_jitter_rv_<inst>`, `log_S0`/`log_Q`/`log_omega0`/`log_sigma`/`log_rho`).

**Approach (preferred):** sample the transit parameter in log-space with a plain
`uniform` prior — a uniform prior on `log(x)` *is* a log-uniform prior on `x`.
This mirrors the existing `np.exp` pattern for noise terms and avoids adding a
new prior type (which would touch ~7 files, including a correct nested-sampling
inverse-CDF transform).

- [ ] Add a log-space derivation in `computer.py` `update_params`, mirroring
      `params['err_flux_'+inst] = np.exp(params['ln_err_flux_'+inst])`
      (line ~320). E.g. `params[companion+'_period'] = np.exp(params[companion+'_log_period'])`.
- [ ] Declare the log row in `params.csv` (e.g. `b_log_period`) with a `uniform` prior.
- [ ] **Caveat:** `b_period` (and other physical keys) are read at *load time*
      too — fast-fit windowing, plotting, a/Rs derivation — outside the per-`theta`
      `update_params` path. Ensure the physical key stays populated everywhere it
      is read, or those paths will `KeyError`. Identify each load-time read site
      for the chosen parameter before wiring in the conversion.

**Alternative (not recommended — higher effort/risk):** add a `loguniform` prior
type. Requires edits across `validation/parsing.py` (arity table) + validator,
`basement.py` (load check ~2185, initial-guess checks ~2197, eccentricity prior
combination block ~2544-2587), `mcmc.py` (~88-98, ~192), `nested_sampling.py`
(~84-91, needs a correct log-uniform inverse-CDF prior transform),
`computer.py` (~1186-1191), `optimize.py` (~78-82), `_output_shared.py` (~141-147).
