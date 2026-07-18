# Transit parameterization for allesfitter2

## Scope

The most common allesfitter2 use cases are:

- one or more sectors of TESS photometry;
- multiple planets orbiting the same star;
- simultaneous or single-epoch multiband ground-based photometry;
- partial ground-based transits;
- shared orbital geometry with band-dependent radius ratios, limb darkening,
  dilution, baselines, and noise models.

No single sampling coordinate is optimal for every likelihood. The robust
design is therefore to separate the **canonical physical model** from the
coordinates exposed to the sampler.

## Recommended canonical physical model

Use the following physical quantities internally:

- per star: stellar density, $\rho_\star$;
- per planet: $P$, $T_0$, impact parameter $b$, $e$, and $\omega$;
- per bandpass: radius ratio $k_j=R_{p,j}/R_\star$.

For each planet, derive the scaled semimajor axis from Kepler's third law:

$$
A \equiv \frac{a}{R_\star}
=
\left[
\frac{G P^2(1+q)\rho_\star}{3\pi}
\right]^{1/3},
$$

where $q=M_p/M_\star$. For ordinary planets, $q\simeq0$ is normally
adequate. Stellar and brown-dwarf companions require an explicit mass-ratio
treatment.

At primary conjunction,

$$
b=A\cos i\frac{1-e^2}{1+e\sin\omega},
$$

so

$$
\cos i=
\frac{b}{A}
\frac{1+e\sin\omega}{1-e^2}.
$$

For each bandpass $j$, derive the fractional radii used by the light-curve
model:

$$
\frac{R_\star}{a}=\frac{1}{A},
\qquad
\frac{R_{p,j}}{a}=\frac{k_j}{A},
\qquad
r_{\mathrm{suma},j}=\frac{1+k_j}{A}.
$$

Transit duration is then a derived quantity. Under the usual conjunction
approximation,

$$
T_{14,j}=\frac{P}{\pi}
\arcsin\left[
\frac{\sqrt{(1+k_j)^2-b^2}}{A\sin i}
\frac{\sqrt{1-e^2}}{1+e\sin\omega}
\right].
$$

This physical layer supports one or many planets and one or many bandpasses
without changing the meaning of the orbital parameters.

## Why $\rho_\star+b+k_j$ is the most flexible default

### Achromatic orbital geometry

The impact parameter $b$ describes the orbit in stellar-radius units and is
shared by all instruments and bandpasses. Radius ratios $k_j$ may remain
band-dependent.

This is more general than

$$
\beta=\frac{b}{1+k},
$$

because a chromatic model has several $k_j$ values and therefore no unique
band-independent $\beta$. One can define $\beta$ relative to a reference
radius, but that introduces an arbitrary reference band or an additional
achromatic radius parameter.

### Physically consistent chromatic radii

A shared `rsuma` combined with band-dependent radius ratios implies

$$
\frac{R_\star}{a}=\frac{r_\mathrm{suma}}{1+k_j},
$$

which changes with bandpass. Sampling or deriving a shared $a/R_\star$
instead keeps the stellar orbit fixed and lets only the planetary radius vary
with wavelength.

### Natural multiplanet coupling

One sampled $\rho_\star$ gives each planet its own $a/R_\star$ through its
period. The model therefore enforces a common host star without fitting
unrelated orbital-scale parameters for every planet.

This coupling must not silently assume circular orbits. If $e$ and $\omega$
are uncertain, they should remain explicit parameters, for example through

$$
\sqrt e\cos\omega,
\qquad
\sqrt e\sin\omega.
$$

Otherwise, a forced circular density can bias $b$, $k$, or $\rho_\star$.

### Partial-transit robustness

A partial light curve generally cannot determine its own duration reliably.
Duration can trade against:

- transit time;
- baseline slope and curvature;
- normalization;
- limb darkening;
- impact parameter;
- radius ratio.

A partial ground-based transit should therefore share $\rho_\star$ (or
$a/R_\star$), $b$, $P$, $e$, and $\omega$ with TESS or with the other data for
the system. It may retain instrument- or epoch-specific radius ratio, limb
darkening, dilution, baseline, noise model, and TTV offset.

For a ground-only partial event without TESS or an external density
measurement, the geometry will necessarily be prior-dominated. The output
should make that dependence visible.

## Sampling-coordinate layer

The canonical model does not require every fit to sample the same coordinate.
allesfitter2 can expose several transformations:

1. `rho_b`: sample stellar density and per-planet impact parameters;
2. `arstar_b`: sample $a/R_\star$ and $b$ per planet;
3. `duration_b`: sample $T_{14}$ and $b$, then transform to $a/R_\star$;
4. `legacy_cosi_rsuma`: retain the existing `cosi` and `rsuma` interface.

A possible setting is:

```csv
transit_geometry_parameterization,rho_b
```

New prepared transit datasets can use `rho_b`, while existing datasets remain
in legacy mode until explicitly migrated.

### Density coordinate

A positive log-density coordinate is computationally convenient:

```csv
host_lnrho,-0.45,1,uniform -3 2,$\ln\rho_\star$,,
```

but a uniform prior on `host_lnrho` is a log-uniform prior on density. If the
intended prior is uniform or Gaussian in $\rho_\star$, sample `host_rho`
directly or apply the appropriate change-of-variable Jacobian.

### Impact-parameter coordinate

Example rows are:

```csv
b_impact,0.5,1,uniform 0 1.2,$b_b$,,
c_impact,0.5,1,uniform 0 1.2,$b_c$,,
```

For a companion required to transit, check every modeled bandpass:

$$
0\leq b<1+k_j.
$$

Do not enforce $b<1-k_j$ by default, because that would remove grazing
transits.

Direct $b$ sampling is preferable to $\beta$ for the general chromatic case.
For a single achromatic transit, the Espinoza (2018) $(r_1,r_2)$ mapping is a
published, rejection-free alternative that jointly samples the valid $(b,k)$
region.

### Duration coordinate

Sampling $T_{14}$ can align well with a complete, high-S/N,
transit-dominated likelihood. It is less robust for partial transits and does
not naturally enforce a shared stellar density across several planets.

For fixed $P$, $b$, $k$, $e$, and $\omega$, the duration can be inverted to
$A=a/R_\star$. Define

$$
C_b=\frac{1-e^2}{1+e\sin\omega},
\qquad
C_T=\frac{\sqrt{1-e^2}}{1+e\sin\omega},
\qquad
x=\frac{\pi T_{14}}{P}.
$$

Then

$$
A(T_{14})=
\sqrt{
\left(\frac{b}{C_b}\right)^2
+
\frac{[(1+k)^2-b^2]C_T^2}{\sin^2x}
}.
$$

The transformation remains coupled to $b$, $k$, $e$, and $\omega$, which is
why a duration constraint is not interchangeable with a density constraint.

## Coordinate transformations and priors

Changing a sampling coordinate should not change the physical posterior when
the probability measure is transformed correctly.

For density,

$$
\rho_\star\propto A^3,
\qquad
\left|\frac{dA}{d\rho_\star}\right|
=\frac{A}{3\rho_\star}.
$$

For duration, with

$$
D=[(1+k)^2-b^2]C_T^2,
$$

the relevant Jacobian is

$$
\left|\frac{dA}{dT_{14}}\right|
=
\frac{D\pi|\cos x|}{A P\sin^3x}.
$$

Therefore:

- uniform $A$;
- uniform $\rho_\star$;
- uniform $\ln\rho_\star$;
- uniform $T_{14}$

are different physical priors. A sampler transformation must declare which
measure is intended and include the corresponding Jacobian when preserving a
prior defined in another coordinate.

## External measurements are not coordinate transformations

An independent stellar-density measurement should enter as an external
likelihood factor evaluated on the physical density:

$$
\ln L_\rho
=
-\frac{1}{2}
\left[
\frac{\rho_\star-\hat\rho_\star}{\sigma_\rho}
\right]^2.
$$

Likewise, an independent duration measurement can be evaluated as

$$
\ln L_T
=
-\frac{1}{2}
\left[
\frac{T_{14}(A,b,k,e,\omega)-\hat T_{14}}{\sigma_T}
\right]^2.
$$

No coordinate Jacobian is attached to an external-data likelihood. A
Jacobian is required when changing the integration variable of a probability
density, not when evaluating a measurement model.

This distinction is important because “sample density” and “apply a density
prior” are not automatically equivalent statements.

### Gaussian $\rho_\star$ versus Gaussian $a/R_\star$

Because $\rho_\star\propto A^3$, a Gaussian density constraint becomes a
skewed function of $A$. Linear uncertainty propagation gives

$$
\frac{\sigma_A}{A}
\simeq
\frac{1}{3}
\frac{\sigma_\rho}{\rho},
$$

but this Gaussian approximation is reliable only for narrow density
uncertainties. The exact transformed density constraint should be used when
the uncertainty is broad or asymmetric.

## Multiband and multi-epoch layout

The recommended hierarchy is:

### Per star

- $\rho_\star$ or an explicitly chosen transformed density coordinate;
- stellar external-likelihood information.

### Per planet

- $P$;
- reference epoch $T_0$;
- $b$;
- $\sqrt e\cos\omega$ and $\sqrt e\sin\omega$;
- optional individual-transit TTV offsets.

### Per bandpass

- $k_j=R_{p,j}/R_\star$;
- limb-darkening coefficients;
- optional chromatic dilution parameters.

### Per instrument, sector, or night

- baseline parameters;
- white-noise and correlated-noise parameters;
- exposure integration;
- optional instrument-specific flux offsets.

A useful chromatic alternative is

$$
k_j=k_\mathrm{ref}+\Delta k_j,
$$

which shares a well-constrained reference radius and fits small chromatic
offsets. This can be more stable than fitting unrelated absolute radius ratios
in low-S/N ground-based bands.

## Implications for the current density-prior implementation

Sampling one physical host density would remove the need to infer a separate
host density from each planet's transit parameters. It would also remove the
radius-ratio-dependent transit-only branch in which the density calculation is
skipped when `rr**3 >= 0.01`.

The external stellar constraint from `params_star.csv` would instead apply
directly to the shared sampled $\rho_\star$. A large or poorly constrained
radius ratio could no longer disable the host-density constraint.

## Empirical notebook results

Two executable notebooks accompany this analysis:

- [`notebooks/transit_geometry_beta_experiment.ipynb`](notebooks/transit_geometry_beta_experiment.ipynb)
  derives $\beta$, compares `cosi`, $b$, and $\beta$, and implements the
  Espinoza (2018) mapping;
- [`notebooks/transit_duration_arstar_density_experiment.ipynb`](notebooks/transit_duration_arstar_density_experiment.ipynb)
  derives $T_{14}$, $a/R_\star$, and $\rho_\star$, checks the analytic
  Jacobians, and compares sampling coordinates with external constraints.

The controlled MCMC experiments demonstrate that no coordinate wins
universally:

- in the transit-only duration experiment, direct $T_{14}$ sampling produced
  the highest worst-coordinate effective sample size;
- after adding a precise density measurement, direct $\rho_\star$ sampling
  performed best;
- in the separate impact-parameter experiment, `cosi` mixed faster than
  $\beta$ for the chosen smooth posterior even though $\beta$ had completely
  valid rectangular support.

These results support a canonical physical model with optional sampling
transformations rather than one mandatory coordinate for every dataset.

## Backward-compatible implementation plan

1. Add tested transformations among $\rho_\star$, $a/R_\star$, `rsuma`,
   `cosi`, $b$, and $T_{14}$.
2. Introduce a `transit_geometry_parameterization` setting. Default legacy
   datasets to `legacy_cosi_rsuma`.
3. Add physical `host_rho` or transformed `host_lnrho` parameters and
   per-companion `<companion>_impact` parameters.
4. In the parameter-update layer, derive $a/R_\star$, inclination, and
   band-specific fractional radii.
5. Pass band-specific $R_\star/a$ and $R_p/a$ consistently to the light-curve
   model.
6. Apply `require_<companion>_transit` to every modeled bandpass while
   retaining grazing geometries.
7. Refactor the stellar-density likelihood so `params_star.csv` constrains the
   shared physical density directly.
8. Make `prepare_allesfit` emit the new physical parameterization for newly
   prepared transit datasets.
9. Preserve legacy input and output names through derived aliases during a
   deprecation period.
10. Validate with injected complete, partial, grazing, eccentric, chromatic,
    multiplanet, and long-cadence light curves using both MCMC and nested
    sampling.

## Recommendation

For the general allesfitter2 production model, use shared physical
$\rho_\star$, per-planet $b$, and per-bandpass $k_j$ as the canonical layer.
Allow $T_{14}$, $a/R_\star$, $\rho_\star$, and the legacy `cosi`/`rsuma` pair
as selectable sampling coordinates with explicit prior measures and correct
Jacobians.

For TESS plus partial multiband ground-based data, share the physical orbital
geometry across all observations and let only genuinely chromatic or
instrument-specific quantities vary. Use external duration and density
information as measurement likelihoods rather than allowing them to silently
change the sampling prior.
