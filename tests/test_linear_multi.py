"""Tests for the ``sample_linear_multi`` (Tier 1) and
``hybrid_linear_multi`` (Tier 2) baselines that mimic timex's
N-D linear systematics model.

Tier 1 (this file, lines 1-)
  - design matrix shape + standardization
  - weight synthesis into the fit vector
  - baseline subroutine returns X @ weights
  - end-to-end recovery: synthetic data with known weights, run the
    likelihood at the true weights and confirm finite + better than zero

Tier 2 tests are appended once Tier 2 lands.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

try:
    import allesfitter  # noqa: F401
    from allesfitter import config
    from allesfitter.basement import Basement, _build_linear_design_matrix
    from allesfitter.computer import (
        baseline_sample_linear_multi, update_params, calculate_baseline,
    )
    from allesfitter.mcmc import mcmc_lnlike
except Exception:
    pytest.skip("allesfitter not importable", allow_module_level=True)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _params_csv() -> str:
    return (
        "#name,value,fit,bounds,label,unit,coupled_with\n"
        "b_rr,0.10,1,uniform 0.05 0.30,$R_b/R_*$,,\n"
        "b_rsuma,0.15,1,uniform 0.05 0.30,$R/a$,,\n"
        "b_cosi,0.10,1,uniform 0 1,$\\cos i$,,\n"
        "b_epoch,0.5,1,uniform 0.0 1.0,$T_0$,d,\n"
        "b_period,3.5,1,normal 3.5 0.1,$P$,d,\n"
        "b_f_c,0,0,uniform -1 1,,,\n"
        "b_f_s,0,0,uniform -1 1,,,\n"
        "dil_lco,0,0,uniform -1 1,,,\n"
        "host_ldc_q1_lco,0.5,1,uniform 0 1,,,\n"
        "host_ldc_q2_lco,0.5,1,uniform 0 1,,,\n"
        "ln_err_flux_lco,-6,1,uniform -10 -1,,,\n"
    )


def _settings_csv(baseline_type='sample_linear_multi',
                  cols='airmass fwhm bias') -> str:
    return (
        "#name,value\n"
        "companions_phot,b\n"
        "companions_rv,\n"
        "inst_phot,lco\n"
        "inst_rv,\n"
        "time_format,BJD_TDB\n"
        "multiprocess,False\n"
        "print_progress,False\n"
        "fast_fit,False\n"
        "shift_epoch,False\n"
        "host_ld_law_lco,quad\n"
        "error_flux_lco,sample\n"
        f"baseline_flux_lco,{baseline_type}\n"
        f"baseline_flux_lco_cols,{cols}\n"
    )


def _write_lc(path: Path, weights=(0.0, 0.0, 0.0), noise=1e-3, n=200, seed=0):
    """Synthesize a flat lightcurve with a linear-in-(airmass, fwhm)
    baseline. weights = (w_airmass_standardized, w_fwhm_standardized, w_bias)."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 1.0, n)
    airmass = 1.0 + 0.5 * np.cos(np.pi * t)
    fwhm = 1.2 + 0.3 * np.sin(2 * np.pi * t)
    # Standardize to match what _build_linear_design_matrix does internally
    a_std = (airmass - airmass.mean()) / airmass.std()
    f_std = (fwhm - fwhm.mean()) / fwhm.std()
    bias = np.ones_like(t)
    baseline = weights[0] * a_std + weights[1] * f_std + weights[2] * bias
    flux = 1.0 + baseline + rng.normal(0, noise, n)
    err = np.full(n, noise)
    lines = ["#time,flux,flux_err,airmass,fwhm"]
    for i in range(n):
        lines.append(
            f"{t[i]:.10f},{flux[i]:.10f},{err[i]:.10f},"
            f"{airmass[i]:.6f},{fwhm[i]:.6f}"
        )
    path.write_text("\n".join(lines) + "\n")
    return t, flux, err, airmass, fwhm


def _build_datadir(tmp_path, baseline_type='sample_linear_multi',
                   cols='airmass fwhm bias', **lc_kwargs):
    d = tmp_path / "data"
    d.mkdir()
    (d / "results").mkdir()
    (d / "params.csv").write_text(_params_csv())
    (d / "settings.csv").write_text(_settings_csv(baseline_type, cols))
    _write_lc(d / "lco.csv", **lc_kwargs)
    return d


# ---------------------------------------------------------------------------
# 1) design matrix helper
# ---------------------------------------------------------------------------

def test_build_design_matrix_shape_and_standardization():
    t = np.linspace(0, 1, 100)
    covs = {'airmass': 1.0 + 0.5 * np.cos(np.pi * t),
            'fwhm': 1.2 + 0.3 * np.sin(2 * np.pi * t)}
    data_inst = {'covariates': covs, 'time': t}
    X, cols = _build_linear_design_matrix(data_inst,
                                          ['airmass', 'fwhm', 'bias'], t)
    assert X.shape == (100, 3)
    assert cols == ['airmass', 'fwhm', 'bias']
    # named covariate columns standardized to zero-mean, unit-variance
    np.testing.assert_allclose(X[:, 0].mean(), 0.0, atol=1e-12)
    np.testing.assert_allclose(X[:, 0].std(), 1.0, atol=1e-12)
    np.testing.assert_allclose(X[:, 1].mean(), 0.0, atol=1e-12)
    np.testing.assert_allclose(X[:, 1].std(), 1.0, atol=1e-12)
    # bias column is NOT standardized — it's a constant 1
    np.testing.assert_array_equal(X[:, 2], np.ones(100))


def test_build_design_matrix_unknown_token_raises():
    t = np.linspace(0, 1, 50)
    data_inst = {'covariates': {'airmass': np.ones_like(t)}, 'time': t}
    with pytest.raises(ValueError, match="unknown token 'fwhm'"):
        _build_linear_design_matrix(data_inst, ['fwhm'], t)


def test_build_design_matrix_constant_covariate_zero_column():
    t = np.linspace(0, 1, 50)
    data_inst = {'covariates': {'airmass': np.full_like(t, 1.5)}, 'time': t}
    X, _ = _build_linear_design_matrix(data_inst, ['airmass'], t)
    # std==0 -> column is (v - mean), i.e., all zeros (no division by zero)
    np.testing.assert_array_equal(X[:, 0], np.zeros(50))


# ---------------------------------------------------------------------------
# 2) weight synthesis into the fit vector
# ---------------------------------------------------------------------------

def test_synthesize_appends_weight_params(tmp_path):
    d = _build_datadir(tmp_path, cols='airmass fwhm bias',
                       weights=(0.0, 0.0, 0.0))
    b = Basement(str(d), quiet=True)
    # design matrix stored on data dict
    assert 'design_matrix' in b.data['lco']
    assert b.data['lco']['design_matrix'].shape == (200, 3)
    assert b.data['lco']['design_matrix_cols'] == ['airmass', 'fwhm', 'bias']
    # three new fitkeys appended (one per column)
    expected = {
        'baseline_linmulti_airmass_flux_lco',
        'baseline_linmulti_fwhm_flux_lco',
        'baseline_linmulti_bias_flux_lco',
    }
    assert expected.issubset(set(map(str, b.fitkeys)))
    # default prior is normal(0, 1e3) for each synthesized weight
    for k in expected:
        i = list(b.fitkeys).index(k)
        assert list(b.bounds[i][:2]) == ['normal', 0.0]


def test_user_declared_weight_row_overrides_synthesis(tmp_path):
    d = _build_datadir(tmp_path, cols='airmass bias',
                       weights=(0.0, 0.0, 0.0))
    # Add the user's explicit weight row with a tighter prior.
    p = (d / 'params.csv').read_text()
    p += ("baseline_linmulti_airmass_flux_lco,0.0,1,"
          "uniform -0.5 0.5,$w_{airmass}$,,\n")
    (d / 'params.csv').write_text(p)
    b = Basement(str(d), quiet=True)
    # The user's row is honoured — its bound is uniform, not normal
    i = list(b.fitkeys).index('baseline_linmulti_airmass_flux_lco')
    assert b.bounds[i][0] == 'uniform'
    # The bias weight is still auto-synthesized
    assert 'baseline_linmulti_bias_flux_lco' in list(b.fitkeys)


# ---------------------------------------------------------------------------
# 3) baseline subroutine returns X @ weights
# ---------------------------------------------------------------------------

def test_baseline_returns_X_at_weights(tmp_path):
    # weights chosen so the baseline is non-trivial: 0.5*airmass + (-0.3)*fwhm + 0.01
    true_w = (0.5, -0.3, 0.01)
    d = _build_datadir(tmp_path, cols='airmass fwhm bias',
                       weights=true_w, noise=1e-6)  # tiny but nonzero
    b = Basement(str(d), quiet=True)
    config.BASEMENT = b
    theta = np.array(b.theta_0, dtype=float)
    # Inject the true weights via theta (synthesized weights live at the
    # end of the fit vector after our synthesis pass).
    for col, w in zip(['airmass', 'fwhm', 'bias'], true_w):
        idx = list(b.fitkeys).index('baseline_linmulti_'+col+'_flux_lco')
        theta[idx] = w
    params = update_params(theta)
    bl = calculate_baseline(params, 'lco', 'flux',
                            model=np.ones_like(b.data['lco']['flux']),
                            yerr_w=b.data['lco']['err_scales_flux'])
    expected = b.data['lco']['design_matrix'] @ np.array(true_w)
    np.testing.assert_allclose(bl, expected, rtol=1e-10, atol=1e-12)


# ---------------------------------------------------------------------------
# 4) lnlike at true weights > lnlike at zero weights (end-to-end recovery)
# ---------------------------------------------------------------------------

def test_lnlike_prefers_true_weights_over_zero(tmp_path):
    """Compose the full pipeline: a linear-multi baseline is in the
    likelihood path; injecting the true weights should give a meaningfully
    higher lnprob than the zero-weight starting point."""
    true_w = (0.5, -0.3, 0.0)
    d = _build_datadir(tmp_path, cols='airmass fwhm bias',
                       weights=true_w, noise=1e-3, n=400)
    b = Basement(str(d), quiet=True)
    config.BASEMENT = b
    theta = np.array(b.theta_0, dtype=float)
    ll_zero = float(mcmc_lnlike(theta))
    for col, w in zip(['airmass', 'fwhm', 'bias'], true_w):
        idx = list(b.fitkeys).index('baseline_linmulti_'+col+'_flux_lco')
        theta[idx] = w
    ll_true = float(mcmc_lnlike(theta))
    assert np.isfinite(ll_zero) and np.isfinite(ll_true)
    # At the true weights the baseline cancels the systematic; residual is
    # pure white noise. lnlike should improve by ~ N/2 * (Δχ²/N) ≈ many σ.
    assert ll_true - ll_zero > 50.0, (ll_zero, ll_true)


# ---------------------------------------------------------------------------
# 5) empty / missing _cols setting
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Tier 2: hybrid_linear_multi (analytic marginalization)
# ---------------------------------------------------------------------------


def test_hybrid_linear_multi_adds_no_fit_dims(tmp_path):
    """Tier 2 must NOT inject weight rows into the fit vector — the
    weights are marginalised analytically."""
    d = _build_datadir(tmp_path, baseline_type='hybrid_linear_multi',
                       cols='airmass fwhm bias', weights=(0., 0., 0.))
    b = Basement(str(d), quiet=True)
    # design matrix still built for the predictive solve
    assert 'design_matrix' in b.data['lco']
    assert b.data['lco']['design_matrix'].shape[1] == 3
    # no `baseline_linmulti_*` rows in the fit vector
    assert not any(str(k).startswith('baseline_linmulti_')
                   for k in b.fitkeys), list(b.fitkeys)


def test_hybrid_solve_recovers_known_weights(tmp_path):
    """With near-zero noise, the analytic MAP weights should reproduce
    the synthetic signal."""
    from allesfitter.computer import _hybrid_linear_multi_solve
    true_w = (0.50, -0.30, 0.005)
    d = _build_datadir(tmp_path, baseline_type='hybrid_linear_multi',
                       cols='airmass fwhm bias', weights=true_w,
                       noise=1e-6, n=400)
    b = Basement(str(d), quiet=True)
    config.BASEMENT = b
    # Residuals at zero transit-model = flux - 1 (synthetic was f = 1 + X@w + noise)
    y_resid = b.data['lco']['flux']
    yerr = b.data['lco']['err_scales_flux']
    # Need un-normalised yerr; rebuild from raw flux_err in CSV:
    yerr_raw = np.full_like(y_resid, 1e-6)
    w_hat, corr = _hybrid_linear_multi_solve('lco', 'flux',
                                              y_resid - 1.0, yerr_raw)
    np.testing.assert_allclose(w_hat, np.array(true_w), atol=1e-4)
    assert np.isfinite(corr)


def test_hybrid_baseline_returns_X_at_w_hat(tmp_path):
    """baseline_hybrid_linear_multi should return X @ ŵ where ŵ is the
    analytic solve on the supplied residual vector."""
    from allesfitter.computer import (
        baseline_hybrid_linear_multi, _hybrid_linear_multi_solve,
    )
    true_w = (0.30, 0.10, 0.0)
    d = _build_datadir(tmp_path, baseline_type='hybrid_linear_multi',
                       cols='airmass fwhm bias', weights=true_w,
                       noise=1e-6, n=300)
    b = Basement(str(d), quiet=True)
    config.BASEMENT = b
    x = b.data['lco']['time']
    yerr = np.full_like(x, 1e-6)
    y = b.data['lco']['flux'] - 1.0   # residual against a flat transit model
    out = baseline_hybrid_linear_multi(x, y, yerr, x, {}, 'lco', 'flux')
    # Independent re-solve to compare
    w_hat, _ = _hybrid_linear_multi_solve('lco', 'flux', y, yerr)
    expected = b.data['lco']['design_matrix'] @ w_hat
    np.testing.assert_allclose(out, expected, rtol=1e-10)
    # And the recovered baseline should look like the injected one
    expected_signal = b.data['lco']['design_matrix'] @ np.array(true_w)
    np.testing.assert_allclose(out, expected_signal, atol=1e-4)


def test_hybrid_lnlike_better_than_tier1_at_zero(tmp_path):
    """At theta_0 (with all Tier-1 weights at zero), Tier-2 already
    fits the optimal linear baseline analytically and should give a
    *higher* lnlike than Tier-1 with zero weights."""
    true_w = (0.50, -0.30, 0.0)
    # Tier-1 datadir
    d1 = tmp_path / "tier1"; d1.mkdir()
    (d1 / "data").mkdir(); (d1 / "data" / "results").mkdir()
    (d1 / "data" / "params.csv").write_text(_params_csv())
    (d1 / "data" / "settings.csv").write_text(
        _settings_csv('sample_linear_multi', 'airmass fwhm bias'))
    _write_lc(d1 / "data" / "lco.csv", weights=true_w, noise=1e-3, n=400)

    # Tier-2 datadir (identical data)
    d2 = tmp_path / "tier2"; d2.mkdir()
    (d2 / "data").mkdir(); (d2 / "data" / "results").mkdir()
    (d2 / "data" / "params.csv").write_text(_params_csv())
    (d2 / "data" / "settings.csv").write_text(
        _settings_csv('hybrid_linear_multi', 'airmass fwhm bias'))
    _write_lc(d2 / "data" / "lco.csv", weights=true_w, noise=1e-3, n=400)

    b1 = Basement(str(d1 / "data"), quiet=True)
    config.BASEMENT = b1
    ll_tier1_zero = float(mcmc_lnlike(np.array(b1.theta_0, dtype=float)))

    b2 = Basement(str(d2 / "data"), quiet=True)
    config.BASEMENT = b2
    ll_tier2 = float(mcmc_lnlike(np.array(b2.theta_0, dtype=float)))

    assert np.isfinite(ll_tier1_zero) and np.isfinite(ll_tier2)
    # Tier 2's analytic ŵ already detrends the airmass+fwhm systematics
    # at theta_0; Tier 1 sees the same data but with linear weights still
    # at zero — much worse residuals.
    assert ll_tier2 > ll_tier1_zero + 50.0, (ll_tier1_zero, ll_tier2)


def test_hybrid_lnlike_matches_tier1_at_true_weights(tmp_path):
    """At Tier-1's MAP weights, the standard chi² lnlike equals Tier-2's
    marginal lnlike up to the determinant + prior corrections. Concretely:
    Tier-2 = Tier-1_at_MAP + (-½ŵᵀΛŵ - ½logdet(A) + ½logdet(Λ)).
    With the wide σ_p=1e3 prior, the correction term is bounded and
    Tier-2 should be within a few units of Tier-1 at the MAP."""
    from allesfitter.computer import _hybrid_linear_multi_solve

    true_w = (0.30, -0.10, 0.0)
    # Tier-1 datadir
    d1 = tmp_path / "tier1"; d1.mkdir()
    (d1 / "data").mkdir(); (d1 / "data" / "results").mkdir()
    (d1 / "data" / "params.csv").write_text(_params_csv())
    (d1 / "data" / "settings.csv").write_text(
        _settings_csv('sample_linear_multi', 'airmass fwhm bias'))
    _write_lc(d1 / "data" / "lco.csv", weights=true_w, noise=1e-3, n=400)
    b1 = Basement(str(d1 / "data"), quiet=True)
    config.BASEMENT = b1
    theta1 = np.array(b1.theta_0, dtype=float)

    # Set Tier-1 weights to the analytic MAP solution (with stellar_var=0
    # the MAP weights equal the analytic solve on data - transit_model).
    # Use calculate_yerr_w so the test's solve sees the same noise model
    # mcmc_lnlike does (yerr_w depends on ln_err_flux_lco via the params).
    from allesfitter.computer import (
        calculate_model, calculate_yerr_w, update_params,
    )
    params1 = update_params(theta1)
    transit_model = calculate_model(params1, 'lco', 'flux')
    yerr_w = calculate_yerr_w(params1, 'lco', 'flux')
    w_hat, corr = _hybrid_linear_multi_solve(
        'lco', 'flux', b1.data['lco']['flux'] - transit_model, yerr_w)
    for col, w in zip(['airmass', 'fwhm', 'bias'], w_hat):
        idx = list(b1.fitkeys).index('baseline_linmulti_'+col+'_flux_lco')
        theta1[idx] = w
    ll_tier1_at_map = float(mcmc_lnlike(theta1))

    # Tier-2 lnlike (same data, no weight params)
    d2 = tmp_path / "tier2"; d2.mkdir()
    (d2 / "data").mkdir(); (d2 / "data" / "results").mkdir()
    (d2 / "data" / "params.csv").write_text(_params_csv())
    (d2 / "data" / "settings.csv").write_text(
        _settings_csv('hybrid_linear_multi', 'airmass fwhm bias'))
    _write_lc(d2 / "data" / "lco.csv", weights=true_w, noise=1e-3, n=400)
    b2 = Basement(str(d2 / "data"), quiet=True)
    config.BASEMENT = b2
    ll_tier2 = float(mcmc_lnlike(np.array(b2.theta_0, dtype=float)))

    # Tier-2 - Tier-1_at_MAP must equal the correction term we computed.
    delta_pred = corr
    delta_actual = ll_tier2 - ll_tier1_at_map
    np.testing.assert_allclose(delta_actual, delta_pred, atol=0.1, rtol=0.0)


def test_baseline_shape_matches_xx_for_short_predictive_grid(tmp_path):
    """Regression for the 'shape (48,) vs (450,)' silent skip in
    afplot: when the plotting caller passes a shorter `xx` than the
    training grid, the baseline must be returned at xx's length, NOT
    the training length."""
    true_w = (0.30, -0.10, 0.0)
    d = _build_datadir(tmp_path, baseline_type='sample_linear_multi',
                       cols='airmass fwhm bias', weights=true_w,
                       noise=1e-3, n=200)
    b = Basement(str(d), quiet=True)
    config.BASEMENT = b
    theta = np.array(b.theta_0, dtype=float)
    for col, w in zip(['airmass', 'fwhm', 'bias'], true_w):
        idx = list(b.fitkeys).index('baseline_linmulti_'+col+'_flux_lco')
        theta[idx] = w
    params = update_params(theta)
    # Short xx grid — what afplot passes in plotting mode
    xx_short = np.linspace(0.1, 0.9, 48)
    bl_short = calculate_baseline(
        params, 'lco', 'flux', xx=xx_short,
        model=np.ones_like(b.data['lco']['flux']),
        yerr_w=b.data['lco']['err_scales_flux'])
    assert bl_short.shape == xx_short.shape, (bl_short.shape, xx_short.shape)
    # And longer xx (oversampled) also works
    xx_long = np.linspace(0.0, 1.0, 1000)
    bl_long = calculate_baseline(
        params, 'lco', 'flux', xx=xx_long,
        model=np.ones_like(b.data['lco']['flux']),
        yerr_w=b.data['lco']['err_scales_flux'])
    assert bl_long.shape == xx_long.shape

    # Same regression for Tier 2 (hybrid_linear_multi).
    (tmp_path / 'hybrid').mkdir()
    d2 = _build_datadir(tmp_path / 'hybrid', baseline_type='hybrid_linear_multi',
                        cols='airmass fwhm bias', weights=true_w,
                        noise=1e-3, n=200)
    b2 = Basement(str(d2), quiet=True)
    config.BASEMENT = b2
    p2 = update_params(np.array(b2.theta_0, dtype=float))
    bl_h = calculate_baseline(
        p2, 'lco', 'flux', xx=xx_short,
        model=np.ones_like(b2.data['lco']['flux']),
        yerr_w=b2.data['lco']['err_scales_flux'])
    assert bl_h.shape == xx_short.shape


def test_plot_linear_baseline_components_sample(tmp_path):
    """Tier-1 (sample_linear_multi) writes <prefix>_linear_baseline_<inst>.pdf
    with the design-matrix columns + the X@w combination panel."""
    from allesfitter.nested_sampling_output import plot_linear_baseline_components
    true_w = (0.2, -0.1, 0.0)
    d = _build_datadir(tmp_path, baseline_type='sample_linear_multi',
                       cols='airmass fwhm bias', weights=true_w,
                       noise=1e-3, n=200)
    b = Basement(str(d), quiet=True)
    config.BASEMENT = b
    # Synthetic posterior centred on theta_0; median == theta_0
    rng = np.random.default_rng(0)
    samples = np.array(b.theta_0)[None, :] + rng.normal(0, 1e-6, (200, b.ndim))
    plot_linear_baseline_components(samples, prefix='ns')
    out = Path(b.outdir) / 'ns_linear_baseline_lco.pdf'
    assert out.exists() and out.stat().st_size > 0


def test_plot_linear_baseline_components_hybrid(tmp_path):
    """Tier-2 (hybrid_linear_multi): no weight rows in the fit vector,
    weights solved analytically inside the plotting helper."""
    from allesfitter.nested_sampling_output import plot_linear_baseline_components
    true_w = (0.2, -0.1, 0.0)
    d = _build_datadir(tmp_path, baseline_type='hybrid_linear_multi',
                       cols='airmass fwhm bias', weights=true_w,
                       noise=1e-3, n=200)
    b = Basement(str(d), quiet=True)
    config.BASEMENT = b
    samples = np.array(b.theta_0)[None, :] + np.random.default_rng(1).normal(
        0, 1e-6, (200, b.ndim))
    plot_linear_baseline_components(samples, prefix='mcmc')
    out = Path(b.outdir) / 'mcmc_linear_baseline_lco.pdf'
    assert out.exists() and out.stat().st_size > 0


def test_plot_linear_baseline_components_noop_when_no_linmulti(tmp_path):
    """When no inst uses a linear-multi baseline the helper must be a
    safe no-op — no PDFs written, no exception raised."""
    from allesfitter.nested_sampling_output import plot_linear_baseline_components
    # Reuse the sample_linear_multi fixture but flip the baseline to
    # sample_linear (single-axis). The helper should NOT emit a file.
    d = tmp_path / 'data'; d.mkdir()
    (d / 'results').mkdir()
    (d / 'params.csv').write_text(_params_csv())
    s = _settings_csv('sample_linear_multi', 'airmass fwhm bias').replace(
        'sample_linear_multi', 'sample_linear')
    # sample_linear needs baseline_offset_flux_lco + baseline_slope_flux_lco
    # in params.csv — add them.
    p = (d / 'params.csv').read_text()
    p += (
        'baseline_offset_flux_lco,0,1,uniform -0.01 0.01,$o$,,\n'
        'baseline_slope_flux_lco,0,1,uniform -0.1 0.1,$s$,,\n'
    )
    (d / 'params.csv').write_text(p)
    # Drop the _cols line (irrelevant for sample_linear).
    s = '\n'.join(l for l in s.splitlines() if 'baseline_flux_lco_cols' not in l) + '\n'
    (d / 'settings.csv').write_text(s)
    _write_lc(d / 'lco.csv', weights=(0., 0., 0.))
    b = Basement(str(d), quiet=True)
    config.BASEMENT = b
    plot_linear_baseline_components(prefix='ns')
    assert not any(p.name.startswith('ns_linear_baseline_')
                   for p in Path(b.outdir).iterdir())


def test_show_initial_guess_draws_event_axvlines(tmp_path):
    """plot_midtransit / plot_ingress / plot_egress flags must add black
    axvline markers on the 'full'-style time-series panel."""
    import allesfitter
    d = _build_datadir(tmp_path, baseline_type='sample_linear_multi',
                       cols='airmass fwhm bias', weights=(0., 0., 0.))
    # The synthetic LC spans t in [0, 1] with epoch=0.5, period=3.5 → only
    # one mid-transit fits inside; we still want the helper to draw it.
    figs = allesfitter.show_initial_guess(
        str(d), quiet=True, do_logprint=False, return_figs=True,
        plot_midtransit=True, plot_ingress=True, plot_egress=True,
    )
    assert figs is not None and len(figs) >= 1
    fig = figs[0]
    # The 'full' panel is column 0 of row 0. Count black, semi-transparent
    # axvlines on it — the helper inserts at least one per event flag.
    ax_full = fig.axes[0]
    n_vlines = 0
    for line in ax_full.lines:
        x = line.get_xdata()
        if len(x) == 2 and x[0] == x[1]:
            # axvline draws a Line2D with constant x; matplotlib colour
            # comparison uses tuples so check via to_rgba.
            from matplotlib.colors import to_rgba
            if to_rgba(line.get_color()) == to_rgba('k'):
                n_vlines += 1
    # 1 midtransit + 1 ingress + 1 egress per visible transit
    assert n_vlines >= 1, n_vlines


def test_show_initial_guess_default_midtransit_only(tmp_path):
    """Default kwargs draw mid-transit but no ingress/egress."""
    import allesfitter
    d = _build_datadir(tmp_path, baseline_type='sample_linear_multi',
                       cols='airmass fwhm bias', weights=(0., 0., 0.))
    figs = allesfitter.show_initial_guess(
        str(d), quiet=True, do_logprint=False, return_figs=True,
    )
    assert figs is not None and len(figs) >= 1
    # At least one dashed mid-transit line, no dotted ingress/egress lines
    ax = figs[0].axes[0]
    dashed = sum(1 for ln in ax.lines if ln.get_linestyle() == '--')
    dotted = sum(1 for ln in ax.lines if ln.get_linestyle() == ':')
    assert dashed >= 1, dashed
    assert dotted == 0, dotted


def test_show_initial_guess_typo_kwarg_actionable(tmp_path):
    """A typo'd kwarg to show_initial_guess must raise TypeError with a
    Did-you-mean hint pointing at the closest valid name."""
    import allesfitter
    with pytest.raises(TypeError) as ei:
        allesfitter.show_initial_guess('.', plot_midtranst=True)
    msg = str(ei.value)
    assert "plot_midtranst" in msg
    assert "did you mean" in msg.lower()
    assert "plot_midtransit" in msg


def test_unknown_baseline_kind_actionable_error(tmp_path):
    """`baseline_flux_<inst>` set to a typo'd value must raise a
    KeyError that (a) names the offending setting key + value,
    (b) suggests the closest valid kinds (Did you mean: ...),
    (c) lists every valid kind."""
    d = tmp_path / 'data'; d.mkdir(); (d / 'results').mkdir()
    (d / 'params.csv').write_text(_params_csv())
    s = _settings_csv('sample_linear_multi', 'airmass fwhm bias')
    # Strip the _cols line and inject a typo'd baseline kind.
    s = '\n'.join(l for l in s.splitlines()
                  if not l.startswith('baseline_flux_lco_cols'))
    s = s.replace('sample_linear_multi', 'sample_Matern32')   # typo
    (d / 'settings.csv').write_text(s + '\n')
    _write_lc(d / 'lco.csv', weights=(0., 0., 0.))
    from allesfitter.basement import Basement
    from allesfitter.computer import calculate_baseline, update_params
    b = Basement(str(d), quiet=True)
    config.BASEMENT = b
    params = update_params(np.array(b.theta_0, dtype=float))
    with pytest.raises(KeyError) as ei:
        calculate_baseline(
            params, 'lco', 'flux',
            model=np.ones_like(b.data['lco']['flux']),
            yerr_w=b.data['lco']['err_scales_flux'])
    msg = str(ei.value)
    assert "baseline_flux_lco='sample_Matern32'" in msg
    assert "sample_GP_Matern32" in msg               # closest-match hint
    assert "Did you mean" in msg


def test_missing_cols_raises(tmp_path):
    d = tmp_path / "data"
    d.mkdir()
    (d / "results").mkdir()
    (d / "params.csv").write_text(_params_csv())
    s = _settings_csv()
    # strip the _cols line
    s = "\n".join(l for l in s.splitlines() if not l.startswith('baseline_flux_lco_cols'))
    (d / "settings.csv").write_text(s + "\n")
    _write_lc(d / "lco.csv")
    with pytest.raises(ValueError, match="requires baseline_flux_lco_cols"):
        Basement(str(d), quiet=True)
