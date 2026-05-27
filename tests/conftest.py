import numpy as np
import pytest


@pytest.fixture
def sample_time():
    return np.linspace(0, 10, 100)


@pytest.fixture
def sample_flux():
    np.random.seed(42)
    return np.random.normal(1.0, 0.01, 100)


@pytest.fixture
def sample_flux_with_outliers():
    np.random.seed(42)
    flux = np.random.normal(1.0, 0.01, 100)
    flux[10] = 1.5
    flux[50] = 0.5
    return flux


@pytest.fixture
def sample_time_sorted():
    return np.arange(0, 10, 0.1)


@pytest.fixture
def symmetric_errors():
    return (1.0, 0.1, 0.1)


@pytest.fixture
def asymmetric_errors():
    return (1.0, 0.1, 0.2)


@pytest.fixture
def inclination_values():
    return (82.5, 0.2, 0.3)


@pytest.fixture
def limb_darkening_quad():
    return ([0.3, 0.1], [0.2, 0.15, 0.1])


@pytest.fixture
def physical_params():
    return {
        'rr': 0.1,
        'rsuma': 0.11,
        'cosi': 0.1,
        'f_s': 0.0,
        'f_c': 0.0,
        'period': 3.5,
        'epoch': 2457000.0,
    }
