# TODO

1. **Improve evaluation speed** — faster function evaluation might be achieved by
   vectorizing certain operations, or using numba JIT similar to pytransit.
   No `numba`/`jit`/`vectorize` usage found yet in `computer.py`.

2. **Iterative outlier clipping** — implement iterative outlier clipping during the
   optimization step and before MCMC sampling, similar to the example in exoplanet:
   https://gallery.exoplanet.codes/tutorials/lc-multi/#the-probabilistic-model
   (`sigma_clip`/`slide_clip` exist in `time_series.py` and are used for raw-flux
   pre-processing, but there's no iterative clip-refit-reclip loop wired into
   `optimize.py`/`mcmc.py` yet.)

3. **Show prior samples** — add an argument in `show_initial_guess()` to plot n=50
   prior samples (similar to prior-predictive checks in pymc). Only reasonable when
   priors in `params.csv` are tight; wide priors for NS sampling would show
   non-sensical prior samples. Not present in `general_output.py`'s
   `show_initial_guess()` signature yet.

4. **Support log-uniform (Jeffreys) priors for transit/orbital parameters** —
   allesfitter only supports `uniform`, `normal`, and `trunc_normal`. No
   transit/orbital parameter (`b_period`, `b_rr`, `b_rsuma`, `b_cosi`, `b_epoch`,
   `b_K`, `b_q`, `dil_*`, …) has a log-space variant; only noise/GP parameters do.

   **Approach (preferred):** sample the transit parameter in log-space with a plain
   `uniform` prior — a uniform prior on `log(x)` *is* a log-uniform prior on `x`.
   Mirrors the existing `np.exp` pattern for noise terms and avoids adding a new
   prior type (which would touch ~7 files, including a correct nested-sampling
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
   `basement.py` (load check, initial-guess checks, eccentricity prior combination
   block), `mcmc.py`, `nested_sampling.py` (needs a correct log-uniform inverse-CDF
   prior transform), `computer.py`, `optimize.py`, `_output_shared.py`.

5. **No coverage gate** — no `pytest-cov` dependency and no `--cov` config in
   `pyproject.toml`/`pytest.ini`. Add `pytest-cov` and report coverage in CI
   (user standard is 80%).

6. **`print()` calls in library code** — 423 occurrences across `allesfitter/`.
   Route diagnostics through the existing logging setup (`run_logger.py`) so
   callers/notebooks can control verbosity instead of stdout spam. Keep `print`
   only in `scripts/`.

7. **Scattered `# TODO/FIXME/HACK/XXX`** — 28 remaining in `allesfitter/*.py`.
   Sweep into this file so the backlog lives in one place.

8. **Oversized modules** far exceed the 800-line target and concentrate risk
   (current line counts): `basement.py` (3687), `computer.py` (2953),
   `general_output.py` (2260), `nested_sampling_output.py` (1031),
   `deriver.py` (1186), `observables.py` (826). These have grown since the
   2026-06-11 audit — extract cohesive units (e.g. baseline models out of
   `computer.py`; config loading/validation slices out of `basement.py`) to make
   them testable in isolation.

9. **`spots.py` is documented but not wired in** — README.md's capability table
   cites `spots.py` as delivering "Stellar variability / spots / flares," but
   nothing in the package imports it (only reachable via manual
   `allesfitter.spots.plot_spots_from_posteriors(...)`), it's not called from
   `general_output.py`/`mcmc_output.py`/`nested_sampling_output.py`'s plotting
   pipelines, and it has zero test coverage. (Note: the actual spot *fitting* is
   unrelated — `computer.py` passes `spots_1`/`spots_2` straight to `ellc`;
   `spots.py` is a separate posterior-visualization tool.) Either wire it into
   the standard post-fit plotting output or add tests + a usage example so the
   README claim reflects a real, exercised path.

10. **`detection/injection_recovery.py` fails to import** —
    `ModuleNotFoundError: No module named 'transitleastsquares'` at import time
    (also depends on `pathos`, and optionally `exoworlds.tess.tessio`, none of
    which are declared project dependencies). `detection/injection_recovery_output.py`
    is only consumed by this file, so it's blocked on the same fix. Either
    declare `transitleastsquares`/`pathos` as an optional extra and guard the
    import, or drop both files if injection-recovery isn't a supported workflow.
