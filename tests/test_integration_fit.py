#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Integration tests for allesfitter MCMC and Nested Sampling fitting.

These tests run actual short MCMC and NS fits on synthetic transit data
generated using ellc to verify the fitting pipeline works end-to-end.
"""

import numpy as np
import pytest
import os
import tempfile
import shutil
from pathlib import Path


def generate_transit_lightcurve_ellc(time, period, epoch, rr, rsuma, cosi,
                                     baseline=1.0, noise_level=0.001, seed=42,
                                     ldc=None):
    """
    Generate synthetic transit light curve using ellc.
    
    Parameters
    ----------
    time : array
        Time values (BJD)
    period : float
        Orbital period (days)
    epoch : float
        Transit epoch (BJD)
    rr : float
        Planet-to-star radius ratio
    rsuma : float
        Sum of radii over semi-major axis
    cosi : float
        Cosine of orbital inclination
    baseline : float
        Baseline flux level
    noise_level : float
        Standard deviation of Gaussian noise
    seed : int
        Random seed for reproducibility
    ldc : list
        Limb darkening coefficients [u1, u2] for quadratic law
        
    Returns
    -------
    flux : array
        Synthetic flux values
    flux_err : array
        Synthetic flux uncertainties
    """
    try:
        import ellc
    except ImportError:
        pytest.skip("ellc not installed")
    
    np.random.seed(seed)
    
    if ldc is None:
        ldc = [0.5, 0.5]
    
    radius_1 = rsuma / (1.0 + rr)
    radius_2 = radius_1 * rr
    incl = np.arccos(cosi) / np.pi * 180.0
    
    model_flux = ellc.lc(
        t_obs=time,
        radius_1=radius_1,
        radius_2=radius_2,
        sbratio=0.0,
        incl=incl,
        t_zero=epoch,
        period=period,
        ldc_1=ldc,
        ld_1='quad',
        verbose=False
    )
    
    flux = model_flux * baseline + (1.0 - baseline)
    flux += np.random.normal(0, noise_level, len(flux))
    flux_err = np.ones_like(flux) * noise_level
    
    return flux, flux_err


def create_minimal_params_csv(datadir, instrument='tess'):
    """Create minimal params.csv for testing."""
    params_content = """#name,value,fit,bounds,label,unit,coupled_with
b_rr,0.1,1,uniform 0.05 0.3,$R_b / R_\\star$,,
b_rsuma,0.15,1,uniform 0.1 0.3,$(R_\\star + R_b) / a_b$,,
b_cosi,0.1,1,uniform 0 1,$\\cos{i_b}$,,
b_epoch,2457000.0,1,normal 2457000.0 0.1,$T_{0;b}$,BJD,
b_period,3.5,1,normal 3.5 0.01,$P_b$,d,
b_f_c,0,0,uniform -1 1,$\\sqrt{e_b} \\cos{\\omega_b}$,,
b_f_s,0,0,uniform -1 1,$\\sqrt{e_b} \\sin{\\omega_b}$,,
""" + f"""dil_{instrument},0,0,uniform -1 1,$D_\\mathrm{{0; {instrument}}},,
host_ldc_q1_{instrument},0.5,1,uniform 0 1,$q_{{1; \\mathrm{{{instrument}}}}}$,,
host_ldc_q2_{instrument},0.5,1,uniform 0 1,$q_{{2; \\mathrm{{{instrument}}}}}$,,
ln_err_flux_{instrument},-6,1,uniform -10 -1,$\\log{{\\sigma ({instrument})}}$,rel. flux,
"""
    params_path = os.path.join(datadir, 'params.csv')
    with open(params_path, 'w') as f:
        f.write(params_content)
    return params_path


def create_minimal_settings_csv(datadir, mcmc_steps=100, mcmc_walkers=20, 
                                 ns_nlive=50, instrument='tess',
                                 ns_modus='static'):
    """Create minimal settings.csv for testing."""
    settings_content = f"""#name,value
companions_phot,b
companions_rv,
inst_phot,{instrument}
inst_rv,
time_format,BJD_TDB
multiprocess,False
print_progress,False
fast_fit,True
fast_fit_width,0.3333333333333333
shift_epoch,True
host_ld_law_{instrument},quad
baseline_flux_{instrument},none
error_flux_{instrument},sample
mcmc_nwalkers,{mcmc_walkers}
mcmc_total_steps,{mcmc_steps}
mcmc_burn_steps,50
mcmc_thin_by,1
mcmc_pre_run_loops,0
mcmc_pre_run_steps,0
ns_modus,{ns_modus}
ns_nlive,{ns_nlive}
ns_bound,single
ns_sample,auto
ns_tol,0.5
"""
    settings_path = os.path.join(datadir, 'settings.csv')
    with open(settings_path, 'w') as f:
        f.write(settings_content)
    return settings_path


def create_lightcurve_csv(datadir, instrument, time, flux, flux_err):
    """Create lightcurve CSV file."""
    lc_path = os.path.join(datadir, f'{instrument}.csv')
    with open(lc_path, 'w') as f:
        f.write(f'#time,flux,flux_err\n')
        for t, fl, fl_err in zip(time, flux, flux_err):
            f.write(f'{t:.10f},{fl:.10f},{fl_err:.10f}\n')
    return lc_path


def check_allesfitter_importable():
    """Check if allesfitter and all dependencies are importable."""
    try:
        import allesfitter
        return True
    except (ImportError, ModuleNotFoundError) as e:
        return False


def check_ellc_importable():
    """Check if ellc is importable."""
    try:
        import ellc
        return True
    except ImportError:
        return False


class TestSyntheticDataGeneration:
    """Tests for synthetic light curve generation using ellc."""
    
    def test_generate_transit_returns_correct_shape(self):
        """Test that generated light curve has correct shape."""
        time = np.linspace(0, 10, 100)
        flux, flux_err = generate_transit_lightcurve_ellc(
            time, period=3.5, epoch=5.0, rr=0.1, rsuma=0.15, cosi=0.1
        )
        assert len(flux) == len(time)
        assert len(flux_err) == len(time)
    
    def test_generate_transit_with_different_seeds(self):
        """Test that different seeds produce different results."""
        time = np.linspace(0, 10, 100)
        flux1, _ = generate_transit_lightcurve_ellc(
            time, period=3.5, epoch=5.0, rr=0.1, rsuma=0.15, cosi=0.1, seed=42
        )
        flux2, _ = generate_transit_lightcurve_ellc(
            time, period=3.5, epoch=5.0, rr=0.1, rsuma=0.15, cosi=0.1, seed=123
        )
        assert not np.allclose(flux1, flux2)
    
    def test_generate_transit_baseline(self):
        """Test that out-of-transit flux is near baseline."""
        time = np.linspace(0, 10, 100)
        flux, _ = generate_transit_lightcurve_ellc(
            time, period=3.5, epoch=50.0, rr=0.1, rsuma=0.15, cosi=0.1,
            noise_level=0.0, baseline=1.0
        )
        mean_out_of_transit = np.mean(flux)
        assert abs(mean_out_of_transit - 1.0) < 0.01
    
    def test_generate_transit_depth(self):
        """Test that transit produces measurable depth."""
        time = np.linspace(0, 10, 1000)
        flux, _ = generate_transit_lightcurve_ellc(
            time, period=3.5, epoch=5.0, rr=0.1, rsuma=0.15, cosi=0.1,
            noise_level=0.0
        )
        min_flux = np.min(flux)
        expected_depth = 0.1 ** 2
        assert (1.0 - min_flux) > 0.005
    
    def test_generate_transit_with_custom_ldc(self):
        """Test that custom limb darkening coefficients work."""
        time = np.linspace(0, 10, 100)
        ldc = [0.3, 0.2]
        flux, flux_err = generate_transit_lightcurve_ellc(
            time, period=3.5, epoch=5.0, rr=0.1, rsuma=0.15, cosi=0.1,
            ldc=ldc
        )
        assert len(flux) == len(time)
        assert not np.all(flux == 1.0)


@pytest.fixture
def temp_datadir():
    """Create a temporary directory for test datadir."""
    temp_dir = tempfile.mkdtemp(prefix='allesfitter_test_')
    results_dir = os.path.join(temp_dir, 'results')
    os.makedirs(results_dir)
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def synthetic_lc_data():
    """Generate synthetic transit light curve for testing using ellc."""
    time = np.linspace(2456990.0, 2457010.0, 500)
    flux, flux_err = generate_transit_lightcurve_ellc(
        time, 
        period=3.5, 
        epoch=2457000.0, 
        rr=0.1, 
        rsuma=0.15, 
        cosi=0.1,
        baseline=1.0,
        noise_level=0.001,
        seed=42,
        ldc=[0.5, 0.5]
    )
    return {'time': time, 'flux': flux, 'flux_err': flux_err}


@pytest.fixture
def setup_test_datadir(temp_datadir, synthetic_lc_data):
    """Set up a complete test datadir with synthetic data."""
    create_minimal_params_csv(temp_datadir, instrument='tess')
    create_minimal_settings_csv(temp_datadir, 
                                 mcmc_steps=100, 
                                 mcmc_walkers=20,
                                 ns_nlive=50,
                                 instrument='tess',
                                 ns_modus='static')
    create_lightcurve_csv(
        temp_datadir, 'tess',
        synthetic_lc_data['time'],
        synthetic_lc_data['flux'],
        synthetic_lc_data['flux_err']
    )
    return temp_datadir


@pytest.mark.skipif(not check_allesfitter_importable(), 
                    reason="allesfitter or dependencies not installed")
class TestMCMCIntegration:
    """Integration tests for MCMC fitting."""
    
    def test_mcmc_fit_initializes_config(self, setup_test_datadir, monkeypatch):
        """Test that MCMC fit initializes the config correctly."""
        import allesfitter
        from allesfitter import config
        
        monkeypatch.chdir(setup_test_datadir)
        config.init(setup_test_datadir)
        
        assert config.BASEMENT is not None
        assert hasattr(config.BASEMENT, 'settings')
        assert hasattr(config.BASEMENT, 'params')
    
    def test_mcmc_lnprior_bounds(self, setup_test_datadir, monkeypatch):
        """Test that MCMC prior respects bounds."""
        import allesfitter
        from allesfitter import config
        from allesfitter.mcmc import mcmc_lnprior
        
        monkeypatch.chdir(setup_test_datadir)
        config.init(setup_test_datadir)
        
        theta = config.BASEMENT.theta_0
        lp = mcmc_lnprior(theta)
        assert np.isfinite(lp)
    
    def test_mcmc_lnprior_out_of_bounds(self, setup_test_datadir, monkeypatch):
        """Test that MCMC prior returns -inf for out-of-bounds values."""
        import allesfitter
        from allesfitter import config
        from allesfitter.mcmc import mcmc_lnprior
        
        monkeypatch.chdir(setup_test_datadir)
        config.init(setup_test_datadir)
        
        theta = np.array([0.5, 0.5, 0.5, 2457000.0, 3.5, 0.0, 0.0, 0.5, 0.5, -6.0])
        lp = mcmc_lnprior(theta)
        assert lp == -np.inf
    
    def test_mcmc_lnlike_computes(self, setup_test_datadir, monkeypatch):
        """Test that MCMC log-likelihood computes without error."""
        import allesfitter
        from allesfitter import config
        from allesfitter.mcmc import mcmc_lnlike
        
        monkeypatch.chdir(setup_test_datadir)
        config.init(setup_test_datadir)
        
        theta = config.BASEMENT.theta_0
        lnlike = mcmc_lnlike(theta)
        assert np.isfinite(lnlike)
    
    def test_mcmc_lnprob_computes(self, setup_test_datadir, monkeypatch):
        """Test that MCMC log-probability computes."""
        import allesfitter
        from allesfitter import config
        from allesfitter.mcmc import mcmc_lnprob
        
        monkeypatch.chdir(setup_test_datadir)
        config.init(setup_test_datadir)
        
        theta = config.BASEMENT.theta_0
        lnprob = mcmc_lnprob(theta)
        assert np.isfinite(lnprob)
    
    @pytest.mark.slow
    def test_mcmc_fit_completes(self, setup_test_datadir, monkeypatch):
        """Test that MCMC fit completes successfully (slow test)."""
        import allesfitter
        from allesfitter import config
        
        monkeypatch.chdir(setup_test_datadir)
        config.init(setup_test_datadir)
        
        allesfitter.mcmc_fit(setup_test_datadir)
        
        save_file = os.path.join(setup_test_datadir, 'results', 'mcmc_save.h5')
        assert os.path.exists(save_file), "MCMC save file not created"
    
    @pytest.mark.slow
    def test_mcmc_backend_saved(self, setup_test_datadir, monkeypatch):
        """Test that MCMC backend is properly saved."""
        import allesfitter
        from allesfitter import config
        
        monkeypatch.chdir(setup_test_datadir)
        
        allesfitter.mcmc_fit(setup_test_datadir)
        
        import emcee
        reader = emcee.backends.HDFBackend(
            os.path.join(setup_test_datadir, 'results', 'mcmc_save.h5')
        )
        
        assert reader.iteration > 0
        assert reader.shape[0] > 0


@pytest.mark.skipif(not check_allesfitter_importable(), 
                    reason="allesfitter or dependencies not installed")
class TestNestedSamplingIntegration:
    """Integration tests for Nested Sampling fitting."""
    
    def test_ns_fit_initializes_config(self, setup_test_datadir, monkeypatch):
        """Test that NS fit initializes the config correctly."""
        import allesfitter
        from allesfitter import config
        
        monkeypatch.chdir(setup_test_datadir)
        config.init(setup_test_datadir)
        
        assert config.BASEMENT is not None
        assert hasattr(config.BASEMENT, 'settings')
    
    def test_ns_lnlike_computes(self, setup_test_datadir, monkeypatch):
        """Test that NS log-likelihood computes."""
        import allesfitter
        from allesfitter import config
        from allesfitter.nested_sampling import ns_lnlike
        
        monkeypatch.chdir(setup_test_datadir)
        config.init(setup_test_datadir)
        
        theta = config.BASEMENT.theta_0
        lnlike = ns_lnlike(theta)
        assert np.isfinite(lnlike)
    
    def test_ns_prior_transform(self, setup_test_datadir, monkeypatch):
        """Test that NS prior transform works."""
        import allesfitter
        from allesfitter import config
        from allesfitter.nested_sampling import ns_prior_transform
        
        monkeypatch.chdir(setup_test_datadir)
        config.init(setup_test_datadir)
        
        utheta = np.random.uniform(0, 1, config.BASEMENT.ndim)
        theta = ns_prior_transform(utheta)
        
        assert len(theta) == config.BASEMENT.ndim
        assert all(np.isfinite(theta))
    
    @pytest.mark.slow
    def test_ns_fit_completes(self, setup_test_datadir, monkeypatch):
        """Test that NS fit completes successfully (slow test)."""
        import allesfitter
        from allesfitter import config
        
        monkeypatch.chdir(setup_test_datadir)
        
        allesfitter.ns_fit(setup_test_datadir)
        
        save_file = os.path.join(setup_test_datadir, 'results', 'save_ns.pickle.gz')
        assert os.path.exists(save_file), "NS save file not created"
    
    @pytest.mark.slow
    def test_ns_results_loadable(self, setup_test_datadir, monkeypatch):
        """Test that NS results can be loaded."""
        import gzip
        import pickle
        
        monkeypatch.chdir(setup_test_datadir)
        
        allesfitter.ns_fit(setup_test_datadir)
        
        save_file = os.path.join(setup_test_datadir, 'results', 'save_ns.pickle.gz')
        with gzip.open(save_file, 'rb') as f:
            results = pickle.load(f)
        assert results is not None
        assert hasattr(results, 'logl')


@pytest.mark.skipif(not check_allesfitter_importable(), 
                    reason="allesfitter or dependencies not installed")
class TestAllesclassLoading:
    """Tests for allesclass loading from fit results."""
    
    @pytest.mark.slow
    def test_allesclass_from_mcmc(self, setup_test_datadir, monkeypatch):
        """Test that allesclass can load MCMC results."""
        import allesfitter
        
        monkeypatch.chdir(setup_test_datadir)
        
        allesfitter.mcmc_fit(setup_test_datadir)
        
        alles = allesfitter.allesclass(setup_test_datadir)
        assert hasattr(alles, 'posterior_samples')
        assert hasattr(alles, 'posterior_params')
    
    @pytest.mark.slow
    def test_allesclass_from_ns(self, setup_test_datadir, monkeypatch):
        """Test that allesclass can load NS results."""
        import allesfitter
        
        monkeypatch.chdir(setup_test_datadir)
        
        allesfitter.ns_fit(setup_test_datadir)
        
        alles = allesfitter.allesclass(setup_test_datadir)
        assert hasattr(alles, 'posterior_samples')
        assert hasattr(alles, 'posterior_params')


@pytest.mark.skipif(not check_allesfitter_importable(), 
                    reason="allesfitter or dependencies not installed")
class TestOutputFunctions:
    """Tests for output functions."""
    
    def test_show_initial_guess(self, setup_test_datadir, monkeypatch):
        """Test that show_initial_guess runs without error."""
        import allesfitter
        from allesfitter import config
        
        monkeypatch.chdir(setup_test_datadir)
        config.init(setup_test_datadir)
        
        allesfitter.show_initial_guess(setup_test_datadir)
    
    def test_get_labels(self, setup_test_datadir, monkeypatch):
        """Test that get_labels returns proper structure."""
        from allesfitter.general_output import get_labels
        
        monkeypatch.chdir(setup_test_datadir)
        
        labels = get_labels(setup_test_datadir, as_type='dic')
        assert isinstance(labels, dict)
    
    def test_get_data(self, setup_test_datadir, monkeypatch):
        """Test that get_data returns proper structure."""
        from allesfitter.general_output import get_data
        
        monkeypatch.chdir(setup_test_datadir)
        
        data = get_data(setup_test_datadir)
        assert isinstance(data, dict)
    
    def test_get_settings(self, setup_test_datadir, monkeypatch):
        """Test that get_settings returns proper structure."""
        from allesfitter.general_output import get_settings
        
        monkeypatch.chdir(setup_test_datadir)
        
        settings = get_settings(setup_test_datadir)
        assert isinstance(settings, dict)


class TestConfigFileCreation:
    """Tests for config file creation helpers."""
    
    def test_create_params_csv(self, temp_datadir):
        """Test that params.csv is created correctly."""
        path = create_minimal_params_csv(temp_datadir)
        assert os.path.exists(path)
        
        with open(path) as f:
            content = f.read()
        assert 'b_rr' in content
        assert 'b_period' in content
        assert 'uniform' in content or 'normal' in content
    
    def test_create_settings_csv(self, temp_datadir):
        """Test that settings.csv is created correctly."""
        path = create_minimal_settings_csv(temp_datadir)
        assert os.path.exists(path)
        
        with open(path) as f:
            content = f.read()
        assert 'companions_phot' in content
        assert 'inst_phot' in content
        assert 'mcmc_nwalkers' in content
    
    def test_create_lightcurve_csv(self, temp_datadir):
        """Test that lightcurve CSV is created correctly."""
        time = np.linspace(0, 10, 100)
        flux = np.ones_like(time)
        flux_err = np.ones_like(time) * 0.001
        
        path = create_lightcurve_csv(temp_datadir, 'tess', time, flux, flux_err)
        assert os.path.exists(path)
        
        with open(path) as f:
            lines = f.readlines()
        assert len(lines) == 101  # header + 100 data points
