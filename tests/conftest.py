import numpy as np
import pytest


@pytest.fixture(autouse=True)
def isolate_results_dir(monkeypatch):
    """Clear the shared results root so every test uses the shipped default.

    Unset, results land inside each test's own (temporary) data directory, so
    nothing leaks into ``~/ql/allesfitter`` or a value exported in the shell.
    Tests covering the shared-root mode set the variable themselves.
    """
    monkeypatch.delenv("ALLESFITTER_RESULTS_DIR", raising=False)


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
        "rr": 0.1,
        "rsuma": 0.11,
        "cosi": 0.1,
        "f_s": 0.0,
        "f_c": 0.0,
        "period": 3.5,
        "epoch": 2457000.0,
    }
