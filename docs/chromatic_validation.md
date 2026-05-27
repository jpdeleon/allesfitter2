# Chromatic Transit Modeling — Validation Report

This document records the audit of the chromatic transit modeling pipeline on
the `chromatic` branch, the hardening changes made to keep it correct, and the
regression test suite that pins the contract going forward.

## Scope contract

| Parameter family | Scope | Canonical key (chromatic) | Canonical key (achromatic) | Pinned by |
|---|---|---|---|---|
| Orbital period | global | `{c}_period` | `{c}_period` | `TestGlobalOrbitalScope` |
| Transit mid-time (epoch) | global | `{c}_epoch` | `{c}_epoch` | `TestGlobalOrbitalScope` |
| Inclination (cosi) | global | `{c}_cosi` | `{c}_cosi` | `TestGlobalOrbitalScope` |
| Sum of radii / semi-major axis (rsuma) | global | `{c}_rsuma` | `{c}_rsuma` | `TestGlobalOrbitalScope`, `test_rsuma_implied_consistency_per_band` |
| Eccentricity parametrization | global | `{c}_f_c`, `{c}_f_s` | same | `TestGlobalOrbitalScope` |
| RV semi-amplitude | global | `{c}_K` | `{c}_K` | (RV out of scope) |
| Argument of periastron (host) | global | `host_lambda`, `host_vsini` | same | (verified at `basement.py:1164-1165`) |
| Radius ratio (rr) | per-bandpass | `{c}_rr_{bandpass}` | `{c}_rr` | `TestRRScope`, `test_ns_recovers_per_band_rr` |
| Limb darkening scalars | per-bandpass (chromatic) / per-instrument (achromatic) | `{role}_ldc_{q/u}{n}_{bandpass}` | `{role}_ldc_{q/u}{n}_{inst}` | `TestLDCScope`, `TestPerInstLDC` |
| LDC list (passed to `ellc`) | per-instrument | `{role}_ldc_{inst}` | same | `TestPerInstLDC` |
| Surface brightness ratio | per-instrument | `{c}_sbratio_{inst}` | same | `TestInstrumentScope` |
| Dilution | per-instrument | `dil_{inst}` | same | `TestInstrumentScope` |
| Baseline (offset, GP, etc.) | per-instrument | `baseline_*_flux_{inst}` | same | (parser preserves per-inst) |
| Photometric error scaling | per-instrument | `ln_err_flux_{inst}` | same | (parser preserves per-inst) |

`{c}` denotes companion identifier (`b`, `c`, ...). `{role}` is `host` or a
companion. `{bandpass}` is the user-supplied bandpass label (e.g. `tess`,
`kepler`); `{inst}` is the user-supplied instrument label (e.g. `tess_pdcsap`).

## Audit findings

### Confirmed correct
- `Basement.load_settings` parses `bandpass` from `settings.csv` and sets
  `settings['chromatic']` to True iff there are ≥2 unique bandpass labels
  (`basement.py:362-377`).
- `Basement.get_rr_key(companion, inst)` returns `{companion}_rr_{bandpass}`
  when the instrument has a bandpass assigned, else `{companion}_rr`
  (`basement.py:239-258`).
- `computer.flux_subfct_ellc` resolves `rr` per band via `get_rr_key` with an
  achromatic fallback at `computer.py:525-530`, then recomputes
  `radius_1 = rsuma / (1 + rr)` and `radius_2 = radius_1 * rr` so
  `radius_1 + radius_2 == rsuma` holds for each bandpass
  (`computer.py:538-542`).
- The LDC assembly at `computer.py:148-165` builds the per-instrument LDC list
  consumed by `ellc.fluxes` from bandpass-scoped scalars when chromatic and
  from instrument-scoped scalars otherwise. The same logic mirrors in
  `v2/translator.py:75-105`.
- Shared orbital parameters (period, epoch, cosi, rsuma, f_c, f_s, K) are
  validated as single keys with no per-inst or per-bandpass suffix
  (`basement.py:1117-1124`).

### Issues found and fixed inline
1. **Bandpass count mismatch silently broadcast.** Prior code zipped
   `bp_list` and `inst_phot` and silently broadcast `bp_list[0]` when shorter.
   - **Fix:** raise `ValueError` if `len(bp_list) != len(inst_phot)` so users
     must explicitly repeat a bandpass label when intent is shared.
     (`allesfitter/basement.py` — `load_settings`).
   - **Pinned by:** `TestBandpassCountMismatch::test_too_few_bandpasses_raises`
     and `test_too_many_bandpasses_raises`.
2. **Duplicate `params.csv` rows silently last-wins.** `np.genfromtxt` retains
   duplicate name rows but the downstream dict-assembly loop overwrites the
   first with the second, hiding a hand-edit typo (e.g. two `b_rr_tess` rows
   with conflicting priors).
   - **Fix:** detect duplicates after `genfromtxt` and raise
     `ValueError(...duplicate rows...)`. (`load_params`).
   - **Pinned by:** `TestDuplicateRows`.
3. **Unknown bandpass suffix in `params.csv` silently ignored.** A typo
   `b_rr_tes` (vs. `tess`) would slip through validation (the permissive
   `is_valid_key` prefix-matches `b_rr`) and the validate() loop would default
   `b_rr_tess` to `None`, leaving the fit broken.
   - **Fix:** in chromatic mode, scan `<companion>_rr_*` rows and raise if the
     suffix is not in `set(bandpass.values())`. (`load_params`).
   - **Pinned by:** `TestUnknownBandpass`.
4. **No structured helper for LDC scope.** Five sites in `computer.py` and
   `v2/translator.py` repeated `if bandpass: suffix='_'+bandpass else
   suffix='_'+inst`, easy to drift apart.
   - **Fix:** add `Basement.get_ldc_key(role, n, inst, space='u'|'q')` mirroring
     `get_rr_key`. The new helper is the single source of truth for LDC-scalar
     keys; existing call sites remain valid because they already encode the same
     decision via `ldc_suffix`. New callers should use the helper.
   - **Pinned by:** `TestKeyHelpers::test_get_ldc_key_*`.

### Subtle behaviors documented (not bugs)
- **Single-unique-bandpass three-state.** If `settings.csv` sets
  `bandpass = tess tess` over two instruments, `chromatic` is False (no
  wavelength variation), but parameter keys still use the **bandpass** suffix
  (`b_rr_tess`, `host_ldc_q1_tess`). Both instruments share one scalar each.
  This is the canonical way to group instruments under one wavelength scope
  and is exercised by `one_band_two_inst_datadir`.
- **Achromatic fallback for missing chromatic key.** `flux_subfct_ellc`
  (`computer.py:529-530`) falls back to `params[f'{c}_rr']` when the chromatic
  key is absent. This preserves backward compatibility with achromatic
  configs that happen to set a `chromatic` flag indirectly.
  - Pinned by `TestAchromaticFallback::test_missing_chromatic_rr_falls_back_to_unsuffixed`.

## Test suite

Located under `tests/chromatic/`. Five files, 36 tests; the fast subset (34
tests) runs in <30 s, the two `@pytest.mark.slow` E2E tests add ~40 s.

```
tests/chromatic/
├── _helpers.py                  # shared CSV-row builders and ellc synth
├── conftest.py                  # tmp_path fixture factories + truth dict
├── test_config_scope.py         # 21 tests: scope mapping & key helpers
├── test_config_errors.py        # 7 tests: parser hardening raises
├── test_likelihood_assembly.py  # 6 tests: ellc.fluxes monkeypatch
└── test_e2e_chromatic_fit.py    # 2 slow tests: NS fit recovers truth
```

### Running

```bash
# fast (default, excludes slow):
pytest tests/chromatic

# include slow E2E fits:
pytest tests/chromatic -m ''
```

### Requirement → test mapping

| User requirement | Pinned by |
|---|---|
| Globally shared orbital params consistently parsed | `test_global_orbital_params_single_keyed`, `test_orbital_kwargs_bit_equal_across_insts` |
| Globally shared orbital params linked across datasets | `test_shared_orbital_params_identical_across_insts` (via monkeypatched ellc kwargs) |
| rr varies per band | `test_chromatic_one_rr_per_unique_bandpass`, `test_radius_2_over_radius_1_matches_per_band_rr` |
| LDCs independent per band/inst | `TestLDCScope`, `TestPerInstLDC` |
| Per-inst systematics/noise stays isolated | `TestInstrumentScope` |
| Limb darkening assigned correctly | `test_get_ldc_key_*`, `test_ldc_lists_match_per_band_scalars` |
| Mixed configurations parse correctly | `TestChromaticFlag` (3 fixtures: 2-band, 1-band, achromatic) |
| Edge cases robust (single-band multi-inst) | `test_shared_bandpass_groups_ldc_scalars`, `test_get_rr_key_shared_bandpass_groups_instruments` |
| Informative errors for inconsistent configs | `TestBandpassCountMismatch`, `TestDuplicateRows`, `TestUnknownBandpass` |
| Achromatic backward compatibility | `TestAchromaticBackcompat`, `test_ns_achromatic_backcompat_runs_end_to_end` |
| Chromatic NS fit recovers per-band rr | `test_ns_recovers_per_band_rr` |

## Files modified

- `allesfitter/basement.py` — `get_ldc_key` helper, bandpass-count raise,
  duplicate-row raise, unknown-bandpass raise.
- `scripts/prepare_allesfit.py` — new `-bp/--bandpass` CLI arg. When given,
  emits a real `bandpass,<...>` row in `settings.csv`, swaps the single
  `{pl}_rr` row for one `{pl}_rr_<bandpass>` row per unique bandpass, and
  keys LDC scalars by bandpass instead of by instrument. Omitting the flag
  preserves the prior achromatic behaviour.
- `tests/chromatic/` — new package with fixtures, scope tests, error tests,
  likelihood-assembly tests, and slow E2E fits.
- `docs/chromatic_validation.md` — this report.

### `prepare_allesfit.py --bandpass` usage

```bash
# Achromatic (unchanged):
python scripts/prepare_allesfit.py -toi 1224 -s all -f tess

# Two instruments, two bands (chromatic):
python scripts/prepare_allesfit.py -toi 1224 -s all -f tess kepler -bp tess kepler

# Two instruments sharing one bandpass (single-band, shared rr/LDC):
python scripts/prepare_allesfit.py -toi 1224 -s all -f tess_pdcsap tess_qlp -bp tess tess
```

The number of `--bandpass` entries must match `--filename`; the script raises
a clear error otherwise. Repeated labels are how a single bandpass is
explicitly shared across multiple instruments.

## Out of scope (deliberately deferred)

- Master-parity golden test (pickled reference posterior from `master`) —
  would need a long preparatory run on `master` and a committed binary.
  Backward compatibility is instead pinned by `TestAchromaticBackcompat`
  + `test_ns_achromatic_backcompat_runs_end_to_end`, which together verify
  the achromatic pipeline initializes, fits, and recovers truth.
- Chromatic TTV: `prepare_ttv_fit.py` already uses `get_rr_key`; full
  multi-band TTV regression is a separate effort.
- Chromatic phase curves: phase-curve amplitudes are already per-inst and
  not affected by the chromatic flag.
- RV: wavelength-independent in this codebase.
