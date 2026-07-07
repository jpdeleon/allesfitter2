# TODO

##

## Differentiate sampler results output directory
Adopt a convention e.g. mcmc_results vs ns_results
applicable if both mcmc and nested sampling were run on the same model.

## Add function to compare logz from dynesty samples
callable either in a standalone script or a function in run.py

## Improve evaluation speed
Faster function evaluation might be achieved by vectorizing certain operations,
or using numba JIT similar to pytransit.

## Iterative outlier clipping
Implement an iterative outlier clipping during optimization step and before mcmc samplgin,
similar to the example in exoplanet:
https://gallery.exoplanet.codes/tutorials/lc-multi/#the-probabilistic-model

## Show prior samples
Add an argument in show_initial_guess() to plot n=50 priors samples.
This is similar to prior_predictive check in pymc.
This might be reasonable only when priors in params.csv are very tight.
Wide priors for ns sampling will show non-sensical prior samples.

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

---

# Package audit (2026-06-11)

Snapshot audit of the `allesfitter/` package. Package imports OK (`v1.2.10`) and
the suite collects (546/559 tests, 13 deselected), so these are health/maintainability
items, not a broken build. Grouped by priority; each item names the evidence.

## P0 — correctness / likely bugs  ✅ RESOLVED 2026-06-11

Verified after fix: `ruff check . --select F821` → *All checks passed*; 0 invalid-syntax;
`import allesfitter` OK (1.2.10); all six edited files `py_compile` clean.

- [x] **Broken module `priors/estimate_noise_wrap.py`** — 20× `F821 Undefined name
      'INPUT'`. Confirmed nothing imports it. **Fix:** removed the two orphaned
      ipywidgets-GUI inner helpers (`fwrite_params`, `get_median_and_error_strings`)
      that were never called; kept the working `estimate_noise_wrap` entry point.
- [x] **1 invalid-syntax error** (`ruff`: "Trailing comma not allowed"). Located in
      `allesfitter/GUI_tess_transit_search.ipynb` (2nd code cell): a dangling trailing
      comma in `from allesfitter.detection.transit_search import get_tls_kwargs_by_tic,
      tls_search,`. **Fix:** removed the trailing comma (via NotebookEdit).
- [x] **29 `F821` undefined-name total** — triaged and fixed each:
      - `lightcurve_tools.py`: `basestring`→`str` (Py2 relic, ×2); `timegap=TODO*3600`
        → `timegap=3600` (placeholder matching `binning1D_per_night`'s default).
      - `time_series.py`: `flux = flux.value` → `y = y.value` (real bug — param is `y`).
      - `v2/generator.py`: `translate(...)` → `translator.translate(...)` (module already
        imported; function exists in `v2/translator.py`).
      - `v2/detection/injection_recovery.py`: added `from .. import defaults`; replaced
        the broken `defaults.get_default_params()/fill_params()` pair (non-existent
        method) with `defaults.fill_params(injection_params)` (real API); restored the
        commented-out `R_host=1.` parameter used in the depth calc.
      - `tests/integration/test_integration_fit.py`: added the local `import allesfitter`
        the method was missing (every sibling test has it).

      Note (follow-up, not P0): `v2/detection/injection_recovery.py` imports the optional
      `transitleastsquares` package and stays unimportable until it is installed; the
      module is not wired into any live path. Track under the P3 "v2 generations" item.

## P1 — robustness (aligns with "never swallow errors")  ✅ RESOLVED 2026-06-11

Verified after fix: `ruff check . --select E722,B006,B008,B023,B904,B007` →
B006/B008/B023/B904/B007 all **0**; E722 **0 in `.py`** (10 remaining are GUI-notebook
cells, outside the original 87-`.py` scope). `import allesfitter` OK; every touched file
`py_compile`-clean.

- [x] **87 bare `except:`** → narrowed every `.py` occurrence to `except Exception:`
      (across 28 files incl. `computer.py`, `basement.py`, `general_output.py`,
      `nested_sampling*.py`, `v2/translator.py` ×37, etc.). This lets `KeyboardInterrupt`/
      `SystemExit` propagate while preserving the catch-all for ordinary errors — the
      minimal, behavior-preserving narrowing. (Per-site narrowing to the exact exception
      class + logging remains a future refinement.) The 10 remaining bare excepts live in
      GUI notebooks (`GUI.ipynb`, `GUI_tess_transit_search.ipynb`), not the `.py` package.
- [x] **3 `B904` raise-without-`from`** — added `from e` at all three sites
      (`utils/scripting.py:41,168`, `scripts/prepare_allesfit.py:68`); each already bound
      `as e`, so the original cause is now preserved in the traceback.
- [x] **12 `B006` mutable defaults + 1 `B008` call-in-default** — read-only list defaults
      (`ldc=[...]`, `xlim=[...]`, `plotparticles=[]`) converted to immutable tuples;
      `labels={}` (overwritten in-body) → `None`; the `colors=sns.color_palette('deep')`
      call-default → `None` + in-body sentinel guard. All verified read-only first.
- [x] **12 `B023` loop-variable closures** (all in `flares/aflare.py`) — bound the loop
      var in the two `np.piecewise` lambdas via `lambda x, i=i:`.
- [x] **24 `B007` unused loop vars** — 10 auto-fixed; remainder renamed to `_` after
      checking each for after-loop use. One genuine false positive
      (`run_allesfitter_grid.py:62`, last-match idiom where `m` is used after the loop)
      got a rule-coded `# noqa: B007` instead of a breaking rename.

## P2 — engineering hygiene / tooling

- [ ] **No CI** — `.github/workflows/` is absent although a real pytest suite and a
      `.pre-commit-config.yaml` exist. Add a GitHub Actions workflow running
      `uv run ruff check`, `uv run ruff format --check`, and `uv run pytest` on PRs so
      regressions are caught. (Project mandates uv + Ruff per `CLAUDE.md`.)
- [ ] **1604 Ruff violations** (560 auto-fixable, 261 more with `--unsafe-fixes`)
      despite Ruff being the declared standard. Land the safe autofixes first
      (`uv run ruff check --fix .`), review the rest, then make lint blocking in CI so
      the count can't creep back up. Top codes: `UP031` printf-format (33),
      `F821` (29), `B007` (24), `F841` unused-var (16), `E401` (15).
- [ ] **No coverage gate** — no `--cov` config in `pyproject.toml`/`pytest.ini`. Add
      `pytest-cov` and report coverage in CI (user standard is 80%).
- [ ] **431 `print()` calls in library code** — route diagnostics through the existing
      logging setup (the package already has `run_logger.py`) so callers/notebooks can
      control verbosity instead of stdout spam. Keep `print` only in `scripts/`.
- [ ] **Import-time warning** — `import allesfitter` emits
      `nthreads cannot be larger than environment variable "NUMEXPR_MAX_THREADS"`.
      Set/clamp `NUMEXPR_MAX_THREADS` defensively at import or stop forcing nthreads.
- [ ] **33 scattered `# TODO/FIXME/HACK/XXX`** in `allesfitter/` — sweep into this file
      so the backlog lives in one place.

## P3 — structure (maintainability; per 800-line guideline)

- [ ] **Oversized modules** far exceed the 800-line target and concentrate risk:
      `basement.py` (3223), `computer.py` (2584), `general_output.py` (1664),
      `nested_sampling_output.py` (916), `deriver.py` (834), `observables.py` (830).
      Extract cohesive units (e.g. baseline models out of `computer.py`; config
      loading/validation slices out of `basement.py`) to make them testable in isolation.
- [ ] **Three coexisting generations** — `allesfitter/` (current), `allesfitter/v0/`,
      `allesfitter/v2/`. Document the intended status of each (legacy/active/experimental)
      in `notes.md`, and prune or clearly mark dead trees so contributors aren't misled.
