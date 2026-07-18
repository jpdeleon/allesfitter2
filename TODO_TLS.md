# TODO: TLS-assisted parameter initialization

Adding `--tls` is viable, but it changes `prepare_allesfit.py` from a
catalog-driven configuration generator into a detection-plus-initialization
pipeline.

The main implication is that TLS values should initialize the fit, not
automatically become tight priors.

## Parameter mapping

TLS provides:

- `period` -> `{planet}_period`
- `T0` -> `{planet}_epoch`
- `correct_duration` -> no direct allesfitter parameter

Duration must be converted into `{planet}_rsuma` using the existing
`get_rsuma()` machinery and assumptions about radius ratio, inclination, and
impact parameter. It must also update the duration-dependent GP bounds.

A critical time-system detail is that TLS must receive full BJD timestamps, or
its `T0` must have the mission offset added. The downloaded light curve
initially uses BTJD/BKJD, while `params.csv` expects BJD.

## Required workflow change

Currently, `params.csv` is constructed before the light curve is downloaded.
TLS can only run after downloading, stitching, cleaning, and converting the
timestamps.

Therefore, either:

1. Refactor into `download -> TLS -> construct params.csv` (preferred); or
2. Rewrite `params.csv` afterward, while also recomputing period-dependent
   `rsuma`, duration, GP bounds, TTV rows, and other derived values.

Simply patching the period and epoch rows would leave an internally
inconsistent configuration.

## Scientific risks

- TLS can select a half-period/double-period alias, eclipsing binary, stellar
  variability, or instrumental systematic.
- On a multi-planet target, the strongest candidate does not necessarily
  correspond to companion `b`.
- The current TLS defaults effectively accept candidates based primarily on
  `SNR >= 5`; SDE and FAP are unrestricted. This is too permissive for
  automatically modifying priors.
- Deriving tight TLS priors from the same photometry subsequently fitted by
  allesfitter is an empirical reuse of the data. It can produce overconfident
  posteriors and invalid Bayesian evidence comparisons.

Use TLS estimates as starting values with conservative priors, while recording
TLS uncertainties and detection statistics.

## Implementation requirements

- Add `transitleastsquares` as a dependency or optional extra; it is currently
  absent from `pyproject.toml`.
- Propagate the option through the Typer CLI and, if desired, the web GUI.
- Use a separately detrended light curve for detection while retaining the
  original light curve for fitting.
- Save a reviewable `tls_results.json` and diagnostic plot.
- Define the behavior when TLS finds no acceptable signal.
- Define which TLS candidate updates which allesfitter companion.
- For a blind TIC search, also use TLS depth or `rp_rs`; current parameter
  generation requires a radius ratio in addition to period, epoch, and
  duration.
- Add tests covering time-offset conversion, multi-sector input, aliases,
  failed detections, multi-candidate selection, parameter mapping, and CLI
  forwarding.

## Recommended interface

Prefer the name `--tls-init`, explicitly meaning:

> Run TLS on the combined selected sectors, take the strongest validated
> candidate, and use it to initialize an explicitly selected companion.

For a single-candidate target, the default companion could be `b`. For known
multi-planet systems, candidate-to-companion matching should be explicit rather
than automatic.

Recommended safeguards:

- Require configurable SNR, SDE, FAP, and minimum distinct-transit thresholds.
- Save all candidates and selection metrics, even if only one initializes the
  fit.
- Use TLS uncertainties with conservative floors; do not create zero-width or
  excessively narrow priors.
- Preserve catalog values and their provenance in the output or TLS report.
- Fail clearly, or explicitly fall back to catalog parameters, according to a
  documented policy.
