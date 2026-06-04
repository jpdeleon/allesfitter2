# Tests

```
tests/
├── conftest.py          # shared fixtures (sample_time, sample_flux, ...)
├── unit/                # fast, isolated unit tests (default suite)
├── integration/         # end-to-end / sampler-backed tests (marked @pytest.mark.slow)
├── chromatic/           # chromatic-fit feature tests (+ local helpers/conftest)
├── script/              # output tests for scripts/ entry points (offline helpers)
└── _artifacts/          # generated fixtures/benchmarks (not test modules)
```

## Running

```bash
pytest                       # fast suite (slow tests deselected via pytest.ini)
pytest -m slow               # only the slow / integration tests
pytest tests/unit            # just the unit tests
pytest tests/integration     # just the integration tests
pytest tests/chromatic       # just the chromatic-feature tests
pytest tests/script          # just the scripts/ output tests
```

The default `addopts` in `pytest.ini` apply `-m "not slow"`, so a bare
`pytest` skips the heavy sampler runs.

## Markers

| Marker        | Meaning                                                      |
|---------------|-------------------------------------------------------------|
| `unit`        | fast, isolated; no real sampler                             |
| `integration` | end-to-end / sampler-backed (lives in `tests/integration`)  |
| `slow`        | deselected by default; real MCMC / nested-sampling runs     |

## Layout notes

- `tests/conftest.py` fixtures are visible to every subdirectory.
- New pure-logic tests go in `tests/unit/`. New end-to-end tests go in
  `tests/integration/` and should carry `@pytest.mark.slow` if they run a
  real sampler.
- `tests/script/` covers the output of `scripts/` entry points. The scripts'
  `main()` functions are network-driven (catalog queries, `lightkurve`
  downloads), so these tests exercise the deterministic output helpers
  instead (prior-bound math, `params.csv` rewrites, SPOC dilution reports)
  and skip the module if optional script deps are unavailable.
- Validation tests:
  - `tests/unit/test_config_checks.py` — structural `params.csv`/`settings.csv`
    checks (`allesfitter.validation.config_checks`).
  - `tests/unit/test_prior_checks.py` — heuristic GP/noise prior checks
    (`allesfitter.validation.prior_checks`).
  - `tests/unit/test_parsing.py` — shared CSV readers + bounds parsing
    (`allesfitter.validation.parsing`).
