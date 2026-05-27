import numpy as np
import pytest


def sample_time():
    return np.linspace(0, 10, 100)


def sample_flux():
    np.random.seed(42)
    return np.random.normal(1.0, 0.01, 100)


def sample_flux_with_outliers():
    np.random.seed(42)
    flux = np.random.normal(1.0, 0.01, 100)
    flux[10] = 1.5
    flux[50] = 0.5
    return flux


def symmetric_errors():
    return (1.0, 0.1, 0.1)


def asymmetric_errors():
    return (1.0, 0.1, 0.2)
