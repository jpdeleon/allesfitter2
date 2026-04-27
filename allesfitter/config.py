#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Configuration and initialization module for allesfitter.

This module provides the main entry point for initializing the allesfitter
framework. It loads data directories, settings, and prepares the Basement
class which contains all model parameters and observational data.

Functions:
    init: Initialize the allesfitter configuration for a given data directory.
"""

from __future__ import print_function, division, absolute_import

from .basement import Basement


def init(datadir: str, quiet: bool = False) -> None:
    """Initialize allesfitter for a given data directory.
    
    Loads all configuration files (settings.csv, params.csv,观测数据) and
    creates a global BASEMENT object containing all model parameters,
    observational data, and fitting settings.
    
    Parameters
    ----------
    datadir : str
        Path to the working directory containing all required input files:
        - settings.csv: Fitting configuration and instrumental settings
        - params.csv: Initial parameter guesses and priors
        - data files: Light curves and/or radial velocity measurements
    quiet : bool, optional
        If True, suppress verbose output during initialization (default: False)
    
    Returns
    -------
    None
        Sets the global BASEMENT object in the config module
    
    Raises
    ------
    FileNotFoundError
        If required input files are missing from datadir
    ValueError
        If settings.csv contains invalid or conflicting settings
    
    Examples
    --------
    >>> import allesfitter
    >>> allesfitter.config.init('/path/to/datadir')
    """
    global BASEMENT
    BASEMENT = Basement(datadir, quiet=quiet)
    # Apply detrended-flux outlier clip if flux_min_flat / flux_max_flat are
    # set in settings.csv. This must run AFTER the global BASEMENT is bound
    # because calculate_baseline / calculate_stellar_var read config.BASEMENT.
    BASEMENT.apply_flat_clip()