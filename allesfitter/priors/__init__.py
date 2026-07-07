"""
Priors module for allesfitter.

This module provides functions for prior transformations, parameter conversions,
and noise estimation. It is used during Bayesian inference to transform
unit cube samples to physical parameter space.

Functions:
    transform_priors: Transform unit cube samples to physical parameters.
    get_cosi_from_i: Convert inclination to cos(inclination).
    get_Rsuma_from_a_over_Rstar: Convert a/R* to sum of radii.
    estimate_noise: Estimate photometric noise from data.
    estimate_noise_out_of_transit: Estimate noise from out-of-transit data.

Submodules:
    transform_priors: Parameter transformation functions.
    estimate_noise: Noise estimation utilities.
    simulate_PDF: PDF simulation for error propagation.
"""
