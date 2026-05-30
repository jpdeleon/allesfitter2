#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Core data and settings container for everything.

The Basement class serves as the central data structure for allesfitter,
containing all observational data, model parameters, fitting configuration,
and derived quantities. It handles loading from CSV files, validation,
and initialization of all components needed for Bayesian inference.

Classes:
    Basement: Main container class for all fitting data and settings.

Module-Level Constants:
    DEFAULT_LD_CODES: Mapping of limb darkening law integer codes to strings.
"""

from __future__ import print_function, division, absolute_import

from typing import Any
import numpy as np
import os
import sys
import fnmatch
import collections
from datetime import datetime
from multiprocessing import cpu_count
import warnings
warnings.formatwarning = lambda msg, *args, **kwargs: f'\n! WARNING:\n {msg}\ntype: {args[0]}, file: {args[1]}, line: {args[2]}\n'
warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning) 
warnings.filterwarnings('ignore', category=np.RankWarning) 
from scipy.stats import truncnorm

#::: allesfitter modules
from .exoworlds_rdx.lightcurves.index_transits import index_transits, index_eclipses, get_first_epoch, get_tmid_observed_transits
from .priors.simulate_PDF import simulate_PDF
from .utils.mcmc_move_translator import translate_str_to_move

#::: plotting settings
import seaborn as sns
sns.set(context='paper', style='ticks', palette='deep', font='sans-serif', font_scale=1.5, color_codes=True)
sns.set_style({"xtick.direction": "in","ytick.direction": "in"})
sns.set_context(rc={'lines.markeredgewidth': 1})
                     
    
    
    
# Per-hyperparameter prefixes for all celerite baseline kernels supported by
# baseline_get_gp() in computer.py. Used to enumerate which params.csv rows are
# GP hyperparameters when aliasing share-group followers to their leader.
BASELINE_GP_HYPER_PREFIXES = (
    'baseline_gp_matern32_lnsigma_',
    'baseline_gp_matern32_lnrho_',
    'baseline_gp_sho_lnS0_',
    'baseline_gp_sho_lnQ_',
    'baseline_gp_sho_lnomega0_',
    'baseline_gp_real_lna_',
    'baseline_gp_real_lnc_',
    'baseline_gp_complex_lna_',
    'baseline_gp_complex_lnb_',
    'baseline_gp_complex_lnc_',
    'baseline_gp_complex_lnd_',
    'baseline_gp_offset_',
)

# REQUIRED hyperparameter row prefixes per supported baseline GP kernel.
# Used by load_params() to verify the share-group leader actually has the
# rows celerite will need before sampling starts. (baseline_gp_offset_ is
# optional and intentionally not listed here.)
BASELINE_GP_REQUIRED_HYPERS = {
    'sample_GP_Matern32': (
        'baseline_gp_matern32_lnsigma_',
        'baseline_gp_matern32_lnrho_',
    ),
    'sample_GP_SHO': (
        'baseline_gp_sho_lnS0_',
        'baseline_gp_sho_lnQ_',
        'baseline_gp_sho_lnomega0_',
    ),
    'sample_GP_real': (
        'baseline_gp_real_lna_',
        'baseline_gp_real_lnc_',
    ),
    'sample_GP_complex': (
        'baseline_gp_complex_lna_',
        'baseline_gp_complex_lnb_',
        'baseline_gp_complex_lnc_',
        'baseline_gp_complex_lnd_',
    ),
}


###############################################################################
#::: 'Basement' class, which contains all the data, settings, etc.
###############################################################################
class Basement:
    """The 'Basement' class contains all the data, settings, etc.
    
    This is the core data container for everything, holding:
        - All observational data (photometry, radial velocity)
        - Model parameters and their priors
        - Fitting configuration and settings
        - Derived stellar parameters
        - External priors (e.g., stellar density)
    
    Attributes
    ----------
    datadir : str
        Path to the data directory.
    outdir : str
        Path to the output directory where results are saved.
    settings : dict
        Configuration settings loaded from settings.csv.
    params : OrderedDict
        Model parameters loaded from params.csv.
    data : dict
        Nested dict of observational data by instrument and type.
    fulldata : dict
        Complete data including all metadata.
    labels : dict
        Parameter labels for plotting and output.
    external_priors : dict
        External priors such as stellar density constraints.
    
    Examples
    --------
    >>> from allesfitter import config
    >>> config.init('/path/to/datadir')
    >>> base = config.BASEMENT
    >>> print(base.settings['inst_phot'])
    """
    
    ###############################################################################
    #::: init
    ###############################################################################
    def __init__(self, datadir: str, quiet: bool = False) -> None:
        """Initialize the Basement with data from a directory.
        
        Parameters
        ----------
        datadir : str
            The working directory for allesfitter.
            Must contain all the data files:
            - settings.csv: Fitting configuration
            - params.csv: Initial parameter guesses
            - Data files: Light curves and/or RV measurements
            Output directories and files will also be created inside datadir.
        quiet : bool, optional
            If True, suppress verbose output during initialization (default: False).
        
        Returns
        -------
        None
        
        Raises
        ------
        FileNotFoundError
            If required input files are missing.
        ValueError
            If settings contain invalid values or conflicts.
        
        Notes
        -----
        This method:
            1. Creates output directory structure
            2. Loads and validates settings from settings.csv
            3. Loads and validates parameters from params.csv
            4. Loads observational data from CSV files
            5. Applies epoch shifting if configured
            6. Sets up TTV fitting if enabled
            7. Loads stellar priors if available
        """
        
        print('Filling the Basement')
        
        self.quiet = quiet
        self.now = "{:%Y-%m-%d_%H-%M-%S}".format(datetime.now())
        self.datadir = datadir
        self.outdir = os.path.join(datadir,'results') 
        if not os.path.exists( self.outdir ): os.makedirs( self.outdir )
        
        print('')
        self.logprint('\nallesfitter version')
        self.logprint('---------------------')
        self.logprint('v1.2.10')
        
        self.load_settings()
        self.load_params()
        self.load_data()
        
        if self.settings['shift_epoch']:
            try:
                self.change_epoch()
            except:
                warnings.warn('\nCould not shift epoch (you can peacefully ignore this warning if no period was given)\n')
        
        if self.settings['fit_ttvs']:  
            self.setup_ttv_fit()
        
        #::: external priors (e.g. stellar density)
        self.external_priors = {}
        self.load_stellar_priors()
        
        #::: if baseline model == sample_GP, set up a GP object for photometric data
#        self.setup_GPs()
        
        #::: translate limb darkening codes from params.csv (int) into str for ellc
        self.ldcode_to_ldstr = ["none",#   :  0,
                                "lin",#    :  1,
                                "quad",#   :  2,
                                "sing",#   :  3,
                                "claret",# :  4,
                                "log",#  :  5,
                                "sqrt",#  :  6,
                                "exp",#    :  7,
                                "power-2",#:  8,
                                "mugrid"]# : -1

        #::: check if the input is consistent
        for inst in self.settings['inst_phot']:
            key='flux'
            if (self.settings['baseline_'+key+'_'+inst] in ['sample_GP_Matern32', 'sample_GP_SHO']) &\
               (self.settings['error_'+key+'_'+inst] != 'sample'):
                   raise ValueError('If you want to use '+self.settings['baseline_'+key+'_'+inst]+', you will want to sample the jitters, too!')
            
                 
                    
    ###############################################################################
    #::: print function that prints into console and logfile at the same time
    ############################################################################### 
    def logprint(self, *text: Any) -> None:
        """Print to both console and logfile.
        
        Outputs text to stdout and appends to a timestamped log file
        in the output directory.
        
        Parameters
        ----------
        *text : Any
            Any objects to be printed (like the built-in print function).
        
        Returns
        -------
        None
        
        Notes
        -----
        If quiet=True was set during initialization, this method does nothing.
        """
        if not self.quiet:
            print(*text)
            original = sys.stdout
            with open(os.path.join(self.outdir, 'logfile_' + self.now + '.log'), 'a') as f:
                sys.stdout = f
                print(*text)
            sys.stdout = original
        else:
            pass
    
    
    ###############################################################################
    #::: helper to get bandpass for an instrument
    ###############################################################################
    def get_bandpass(self, inst):
        """
        Return bandpass for an instrument, or None if achromatic.
        
        Parameters
        ----------
        inst : str
            Instrument name (e.g., 'tess', 'kepler')
        
        Returns
        -------
        str or None
            Bandpass name if chromatic, None if achromatic
        """
        return self.settings.get('bandpass', {}).get(inst)
    
    
    ###############################################################################
    #::: get_rr_key: helper to get the correct rr key for a companion/instrument
    ###############################################################################
    def get_rr_key(self, companion, inst):
        """
        Return the parameter key for radius ratio (rr) for a given companion and instrument.

        Parameters
        ----------
        companion : str
            Companion name (e.g., 'b', 'c')
        inst : str
            Instrument name (e.g., 'tess', 'kepler')

        Returns
        -------
        str
            Parameter key: 'b_rr' for achromatic, 'b_rr_tess' for chromatic
        """
        bandpass = self.get_bandpass(inst)
        if bandpass:
            return f'{companion}_rr_{bandpass}'
        return f'{companion}_rr'


    ###############################################################################
    #::: get_ldc_key: helper to get the correct LDC scalar key for a role/n/inst
    ###############################################################################
    def get_ldc_key(self, role, n, inst, space='u'):
        """
        Return the per-coefficient LDC key for a role, coefficient index, and instrument.

        The suffix is the instrument's bandpass when chromatic (so multiple
        instruments sharing a bandpass share a single LDC scalar) and the
        instrument name otherwise — matching the suffix used by the validator
        in ``load_params`` and the assembler in ``computer.py``.

        Parameters
        ----------
        role : str
            'host' or a companion identifier ('b', 'c', ...).
        n : int
            Coefficient index (1..4).
        inst : str
            Instrument name (e.g., 'tess', 'kepler', 'tess_pdcsap').
        space : str, optional
            'u' (default) or 'q'.

        Returns
        -------
        str
            For example ``host_ldc_u1_tess`` (chromatic with bandpass='tess')
            or ``host_ldc_u1_tess_pdcsap`` (achromatic, inst='tess_pdcsap').
        """
        if space not in ('u', 'q'):
            raise ValueError(f"space must be 'u' or 'q', got {space!r}")
        bandpass = self.get_bandpass(inst)
        suffix = bandpass if bandpass else inst
        return f'{role}_ldc_{space}{n}_{suffix}'


    ###############################################################################
    #::: load settings
    ###############################################################################
    def load_settings(self):
        '''
        For the full list of options see www.allesfitter.com
        '''
        
        
        def set_bool(text):
            if text.lower() in ['true', '1']:
                return True
            else:
                return False
            
        
        def is_empty_or_none(key):
            return (key not in self.settings) or (str(self.settings[key]).lower() == 'none') or (len(self.settings[key])==0)
            
        
        def unique(array):
            uniq, index = np.unique(array, return_index=True)
            return uniq[index.argsort()]
            
        
        rows = np.genfromtxt( os.path.join(self.datadir,'settings.csv'),dtype=None,encoding='utf-8',delimiter=',' )

        #::: make backwards compatible
        for i, row in enumerate(rows):
#            print(row)
            name = row[0]
            if name[:7]=='planets':
                rows[i][0] = 'companions'+name[7:]
                warnings.warn('You are using outdated keywords. Automatically renaming '+name+' ---> '+rows[i][0]+'. Please fix this before the Duolingo owl comes to get you.') #, category=DeprecationWarning)
            if name[:6]=='ld_law':
                rows[i][0] = 'host_ld_law'+name[6:]
                warnings.warn('You are using outdated keywords. Automatically renaming '+name+' ---> '+rows[i][0]+'. Please fix this before the Duolingo owl comes to get you.') #, category=DeprecationWarning)
                
#        self.settings = {r[0]:r[1] for r in rows}
        self.settings = collections.OrderedDict( [('user-given:','')]+[ (r[0],r[1] ) for r in rows ]+[('automatically set:','')] )

        # Snapshot the set of keys that came from the user's settings.csv,
        # BEFORE any defaults are filled in below. Used by the nested-sampling
        # dispatcher to tell the user when a backend-relevant knob was left
        # implicit (defaulted) vs. explicitly set in the CSV.
        self._settings_raw_keys = {r[0] for r in rows}

        #::: check for unrecognized settings keys
        valid_settings_keys = {
            'companions_phot', 'companions_rv', 'companions_all', 'inst_phot', 'inst_rv', 'inst_rv2', 'inst_all',
            'time_format', 'multiprocess', 'multiprocess_cores', 'fast_fit', 'fast_fit_width',
            'secondary_eclipse', 'phase_curve', 'phase_curve_style', 'shift_epoch',
            'inst_for_b_epoch', 'inst_for_c_epoch', 'inst_for_d_epoch', 'inst_for_e_epoch', 'inst_for_f_epoch', 'inst_for_g_epoch',
            'mcmc_nwalkers', 'mcmc_total_steps', 'mcmc_burn_steps', 'mcmc_thin_by', 'mcmc_pre_run_loops', 'mcmc_pre_run_steps', 'mcmc_moves',
            'ns_modus', 'ns_nlive', 'ns_bound', 'ns_sample', 'ns_tol', 'ns_backend',
            'un_min_ess', 'un_max_iters',
            'bandpass', 'chromatic',
            'fit_ttvs', 'exact_grav', 'use_host_density_prior', 'use_tidal_eccentricity_prior',
            'N_flares', 'N_spots',
            't_exp_tess', 't_exp_kepler', 't_exp_n_int_tess', 't_exp_n_int_kepler',
            'print_progress', 'quiet',
            'flux_min_raw', 'flux_max_raw',
            'flux_min_flat', 'flux_max_flat',
            'baseline_share_flux', 'baseline_share_rv', 'baseline_share_rv2',
        }
        for key in self.settings:
            if key in ['user-given:', 'automatically set:']:
                continue
            if key not in valid_settings_keys and not any(key.startswith(prefix) for prefix in [
                'host_ld_law_', 'host_ld_space_', 'host_grid_', 'host_shape_', 'host_flux_weighted_',
                'host_rotfac_', 'host_hf_', 'host_bfac_', 'host_heat_', 'host_lambda_', 'host_N_spots_',
                'b_ld_law_', 'b_ld_space_', 'b_grid_', 'b_shape_', 'b_flux_weighted_', 'b_N_spots_',
                'c_ld_law_', 'c_ld_space_', 'c_grid_', 'c_shape_', 'c_flux_weighted_', 'c_N_spots_',
                'd_ld_law_', 'd_ld_space_', 'd_grid_', 'd_shape_', 'd_flux_weighted_', 'd_N_spots_',
                'e_ld_law_', 'e_ld_space_', 'e_grid_', 'e_shape_', 'e_flux_weighted_', 'e_N_spots_',
                'f_ld_law_', 'f_ld_space_', 'f_grid_', 'f_shape_', 'f_flux_weighted_', 'f_N_spots_',
                'g_ld_law_', 'g_ld_space_', 'g_grid_', 'g_shape_', 'g_flux_weighted_', 'g_N_spots_',
                'baseline_flux_', 'baseline_rv_', 'baseline_rv2_',
                'error_flux_', 'error_rv_', 'error_rv2_',
                't_exp_', 'stellar_var_flux', 'stellar_var_rv',
            ]):
                warnings.warn('Unrecognized setting key "'+key+'" in settings.csv. This may be a typo or deprecated keyword.')

        
        #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        #::: Main settings
        #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        if 'time_format' not in self.settings:
            self.settings['time_format'] = 'BJD_TDB'
            
            
        for key in ['companions_phot', 'companions_rv', 'inst_phot', 'inst_rv', 'inst_rv2']:
            if key not in self.settings:
                self.settings[key] = []
            elif len(self.settings[key]): 
                self.settings[key] = str(self.settings[key]).split(' ')
            else:                       
                self.settings[key] = []
        
        self.settings['companions_all']  = list(np.unique(self.settings['companions_phot']+self.settings['companions_rv'])) #sorted by b, c, d...
        self.settings['inst_all'] = list(unique( self.settings['inst_phot']+self.settings['inst_rv']+self.settings['inst_rv2'] )) #sorted like user input
    
        if len(self.settings['inst_phot'])==0 and len(self.settings['companions_phot'])>0:
            raise ValueError('No photometric instrument is selected, but photometric companions are given.')
        if len(self.settings['inst_rv'])==0 and len(self.settings['companions_rv'])>0:
           raise ValueError('No RV instrument is selected, but RV companions are given.')
           
           
        #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        #::: Bandpass settings (for chromatic transit modeling)
        #::: If not specified → achromatic (all instruments share same rr)
        #::: If specified with multiple unique values → chromatic
        #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        if 'bandpass' not in self.settings or is_empty_or_none('bandpass'):
            self.settings['bandpass'] = {}  # empty = achromatic
        else:
            bp_list = str(self.settings['bandpass']).split()
            n_inst = len(self.settings['inst_phot'])
            if len(bp_list) != n_inst:
                raise ValueError(
                    "settings.csv 'bandpass' has {n_bp} entries but inst_phot has "
                    "{n_inst} entries; each photometric instrument needs an explicit "
                    "bandpass label (repeat the same label to keep instruments achromatic). "
                    "Got bandpass={bp_list!r}, inst_phot={inst_phot!r}.".format(
                        n_bp=len(bp_list), n_inst=n_inst,
                        bp_list=bp_list, inst_phot=self.settings['inst_phot']
                    )
                )
            self.settings['bandpass'] = {
                inst: bp_list[i] for i, inst in enumerate(self.settings['inst_phot'])
            }

        # Determine if chromatic (multiple unique bandpasses) or achromatic
        unique_bandpasses = set(self.settings['bandpass'].values())
        self.settings['chromatic'] = len(unique_bandpasses) > 1


        #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        #::: Validate per-instrument settings suffixes
        #::: Catches typos like ``host_ld_law_tess,quad`` when no instrument is
        #::: named "tess" (e.g. inst_phot=['tglc120_s90', ...] + bandpass='tess').
        #::: Without this guard, the default at ~685 silently sets the LD law to
        #::: None for every real instrument, ldc_1=None reaches ellc, and the
        #::: q1/q2 values in params.csv have zero effect on the transit shape.
        #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        _per_inst_prefixes = (
            'host_ld_law_', 'host_ld_space_', 'host_grid_', 'host_shape_',
            'host_flux_weighted_', 'host_rotfac_', 'host_hf_', 'host_bfac_',
            'host_heat_', 'host_N_spots_',
        )
        for _comp in ('b', 'c', 'd', 'e', 'f', 'g'):
            _per_inst_prefixes = _per_inst_prefixes + (
                _comp + '_ld_law_', _comp + '_ld_space_', _comp + '_grid_',
                _comp + '_shape_', _comp + '_flux_weighted_',
                _comp + '_N_spots_',
            )
        _per_inst_prefixes = _per_inst_prefixes + (
            'baseline_flux_', 'baseline_rv_', 'baseline_rv2_',
            'error_flux_', 'error_rv_', 'error_rv2_',
            't_exp_', 't_exp_n_int_',
            'stellar_var_flux_', 'stellar_var_rv_',
        )
        _known_insts = set(self.settings['inst_all'])
        _known_bands = unique_bandpasses
        _orphans = []
        for _key in list(self.settings.keys()):
            if _key in ('user-given:', 'automatically set:'):
                continue
            for _pref in _per_inst_prefixes:
                if _key.startswith(_pref):
                    _suffix = _key[len(_pref):]
                    if _suffix and _suffix not in _known_insts:
                        _orphans.append((_key, _pref, _suffix))
                    break
        if _orphans:
            _hint_lines = []
            for _k, _p, _s in _orphans:
                _msg = "  '{}': suffix '{}' is not in inst_phot+inst_rv+inst_rv2 ({})".format(
                    _k, _s, sorted(_known_insts)
                )
                if _s in _known_bands:
                    _affected = sorted(
                        i for i, b in self.settings['bandpass'].items() if b == _s
                    )
                    _msg += (
                        "  [hint: '{}' is a BANDPASS label, not an instrument. "
                        "Repeat this row once per instrument using that bandpass: {}]"
                    ).format(_s, _affected)
                _hint_lines.append(_msg)
            raise ValueError(
                "settings.csv contains per-instrument keys whose suffix is "
                "not a known instrument name. The suffix must match an entry "
                "of inst_phot/inst_rv/inst_rv2 (NOT a bandpass label).\n"
                "Offending rows:\n" + "\n".join(_hint_lines)
            )


            
        #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        #::: General settings
        #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        if 'print_progress' in self.settings:
            self.settings['print_progress'] = set_bool(self.settings['print_progress'] )
        else:
            self.settings['print_progress'] = True
        
        
        #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        #::: Epoch settings
        #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        if 'shift_epoch' in self.settings:
            self.settings['shift_epoch'] = set_bool(self.settings['shift_epoch'] )
        else:
            self.settings['shift_epoch'] = True 
            
            
        for companion in self.settings['companions_all']:
            if 'inst_for_'+companion+'_epoch' not in self.settings:
                self.settings['inst_for_'+companion+'_epoch'] = 'all'
        
            if self.settings['inst_for_'+companion+'_epoch'] in ['all','none']:
                self.settings['inst_for_'+companion+'_epoch'] = self.settings['inst_all']
            else:
                if len(self.settings['inst_for_'+companion+'_epoch']): 
                    self.settings['inst_for_'+companion+'_epoch'] = str(self.settings['inst_for_'+companion+'_epoch']).split(' ')
                else:                       
                    self.settings['inst_for_'+companion+'_epoch'] = []
        
        
        #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        #::: Multiprocess settings
        #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        self.settings['multiprocess'] = set_bool(self.settings['multiprocess'])
        
        from pprint import pprint
        pprint(self.settings)
        
        if 'multiprocess_cores' not in self.settings.keys():
            self.settings['multiprocess_cores'] = cpu_count()-1
        elif self.settings['multiprocess_cores'] == 'all':
            self.settings['multiprocess_cores'] = cpu_count()-1
        else:
            self.settings['multiprocess_cores'] = int(self.settings['multiprocess_cores'])
            if self.settings['multiprocess_cores'] == cpu_count():
                string = 'You are pushing your luck: you want to run on '+str(self.settings['multiprocess_cores'])+' cores, but your computer has only '+str(cpu_count())+'. I will let you go through with it this time...'
                warnings.warn(string)
            if self.settings['multiprocess_cores'] > cpu_count():
                string = 'Oops, you want to run on '+str(self.settings['multiprocess_cores'])+' cores, but your computer has only '+str(cpu_count())+'. Maybe try running on '+str(cpu_count()-1)+'?'


        #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        #::: Phase variations
        #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        if ('phase_variations' in self.settings.keys()) and len(self.settings['phase_variations']):
            warnings.warn('You are using outdated keywords. Automatically renaming "phase_variations" ---> "phase_curve".'+'. Please fix this before the Duolingo owl comes to get you.')
            self.settings['phase_curve'] = self.settings['phase_variations']
            
        if ('phase_curve' in self.settings.keys()) and len(self.settings['phase_curve']):
            self.settings['phase_curve'] = set_bool(self.settings['phase_curve'])
            if self.settings['phase_curve']==True:                
                # self.logprint('The user set phase_curve==True. Automatically set fast_fit=False and secondary_eclispe=True, and overwrite other settings.')
                self.settings['fast_fit'] = 'False'
                self.settings['secondary_eclipse'] = 'True'
        else:
            self.settings['phase_curve'] = False
            
            
        #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        #::: Fast fit
        #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        if ('fast_fit' in self.settings.keys()) and len(self.settings['fast_fit']):
            self.settings['fast_fit'] = set_bool(self.settings['fast_fit'])
        else:
            self.settings['fast_fit'] = False
        
        if ('fast_fit_width' in self.settings.keys()) and len(self.settings['fast_fit_width']):
            self.settings['fast_fit_width'] = float(self.settings['fast_fit_width'])
        else:
            self.settings['fast_fit_width'] = 8./24.

        #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        #::: Raw-flux outlier clip bounds (applied at load time in load_data())
        #::: If `flux_min_raw` / `flux_max_raw` are present, rows with flux
        #::: outside [flux_min_raw, flux_max_raw] are dropped from each
        #::: photometric instrument's data before any further reduction.
        #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        if ('flux_min_raw' in self.settings.keys()) and not is_empty_or_none('flux_min_raw'):
            self.settings['flux_min_raw'] = float(self.settings['flux_min_raw'])
        else:
            self.settings['flux_min_raw'] = None

        if ('flux_max_raw' in self.settings.keys()) and not is_empty_or_none('flux_max_raw'):
            self.settings['flux_max_raw'] = float(self.settings['flux_max_raw'])
        else:
            self.settings['flux_max_raw'] = None

        if (self.settings['flux_min_raw'] is not None
                and self.settings['flux_max_raw'] is not None
                and self.settings['flux_min_raw'] >= self.settings['flux_max_raw']):
            raise ValueError(
                'flux_min_raw (%s) must be < flux_max_raw (%s) in settings.csv.'
                % (self.settings['flux_min_raw'], self.settings['flux_max_raw'])
            )

        #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        #::: Flattened-flux outlier clip bounds (applied by config.init() AFTER
        #::: BASEMENT is constructed, since computing the trend requires
        #::: calculate_baseline which reads config.BASEMENT.{settings,data}).
        #::: Bounds are interpreted on the detrended flux:
        #:::     flat = flux - baseline(initial-guess) - stellar_var(initial-guess)
        #::: Rows with `flat` outside [flux_min_flat, flux_max_flat] are dropped.
        #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        if ('flux_min_flat' in self.settings.keys()) and not is_empty_or_none('flux_min_flat'):
            self.settings['flux_min_flat'] = float(self.settings['flux_min_flat'])
        else:
            self.settings['flux_min_flat'] = None

        if ('flux_max_flat' in self.settings.keys()) and not is_empty_or_none('flux_max_flat'):
            self.settings['flux_max_flat'] = float(self.settings['flux_max_flat'])
        else:
            self.settings['flux_max_flat'] = None

        if (self.settings['flux_min_flat'] is not None
                and self.settings['flux_max_flat'] is not None
                and self.settings['flux_min_flat'] >= self.settings['flux_max_flat']):
            raise ValueError(
                'flux_min_flat (%s) must be < flux_max_flat (%s) in settings.csv.'
                % (self.settings['flux_min_flat'], self.settings['flux_max_flat'])
            )
                
            
        #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        #::: Host stellar density prior
        #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        if 'use_host_density_prior' in self.settings:
            self.settings['use_host_density_prior'] = set_bool(self.settings['use_host_density_prior'] )
        else:
            self.settings['use_host_density_prior'] = True
        
        
        #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        #::: Host stellar density prior
        #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        if 'use_tidal_eccentricity_prior' in self.settings:
            self.settings['use_tidal_eccentricity_prior'] = set_bool(self.settings['use_tidal_eccentricity_prior'] )
        else:
            self.settings['use_tidal_eccentricity_prior'] = False
            
        
        #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        #::: TTVs
        #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        if ('fit_ttvs' in self.settings.keys()) and len(self.settings['fit_ttvs']):
            self.settings['fit_ttvs'] = set_bool(self.settings['fit_ttvs'])
            if (self.settings['fit_ttvs']==True) and (self.settings['fast_fit']==False):
                raise ValueError('fit_ttvs==True, but fast_fit==False.'+\
                                 'Currently, you can only fit for TTVs if fast_fit==True.'+\
                                 'Please choose different settings.')
        else:
            self.settings['fit_ttvs'] = False
        
        
        #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        #::: Secondary eclipse
        #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        if ('secondary_eclipse' in self.settings.keys()) and len(self.settings['secondary_eclipse']):
            self.settings['secondary_eclipse'] = set_bool(self.settings['secondary_eclipse'])
        else:
            self.settings['secondary_eclipse'] = False
                        
            
        #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        #::: MCMC settings
        #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        if 'mcmc_pre_run_loops' not in self.settings: 
            self.settings['mcmc_pre_run_loops'] = 0
        if 'mcmc_pre_run_steps' not in self.settings: 
            self.settings['mcmc_pre_run_steps'] = 0
        if 'mcmc_nwalkers' not in self.settings: 
            self.settings['mcmc_nwalkers'] = 100
        if 'mcmc_total_steps' not in self.settings: 
            self.settings['mcmc_total_steps'] = 2000
        if 'mcmc_burn_steps' not in self.settings: 
            self.settings['mcmc_burn_steps'] = 1000
        if 'mcmc_thin_by' not in self.settings: 
            self.settings['mcmc_thin_by'] = 1
        if 'mcmc_moves' not in self.settings: 
            self.settings['mcmc_moves'] = 'DEMove'
            
        #::: make sure these are integers
        for key in ['mcmc_nwalkers','mcmc_pre_run_loops','mcmc_pre_run_steps',
                    'mcmc_total_steps','mcmc_burn_steps','mcmc_thin_by']:
            self.settings[key] = int(self.settings[key])
            
        #::: luser proof
        if self.settings['mcmc_total_steps'] <= self.settings['mcmc_burn_steps']:
            raise ValueError('Your setting for mcmc_total_steps must be larger than mcmc_burn_steps (check your settings.csv).')
                
            
        #::: translate the mcmc_move string into a list of emcee commands
        self.settings['mcmc_moves'] = translate_str_to_move(self.settings['mcmc_moves'])
            
        # N_evaluation_samples = int( 1. * self.settings['mcmc_nwalkers'] * (self.settings['mcmc_total_steps']-self.settings['mcmc_burn_steps']) / self.settings['mcmc_thin_by'] )
        # self.logprint('\nAnticipating ' + str(N_evaluation_samples) + 'MCMC evaluation samples.\n')
        # if N_evaluation_samples>200000:
        #     answer = input('It seems like you are asking for ' + str(N_evaluation_samples) + 'MCMC evaluation samples (calculated as mcmc_nwalkers * (mcmc_total_steps-mcmc_burn_steps) / mcmc_thin_by).'+\
        #                    'That is an aweful lot of samples.'+\
        #                    'What do you want to do?\n'+\
        #                    '1 : continue at any sacrifice\n'+\
        #                    '2 : abort and increase the mcmc_thin_by parameter in settings.csv (do not do this if you continued an old run!)\n')
        #     if answer==1: 
        #         pass
        #     else:
        #         raise ValueError('User aborted the run.')

        
        #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        #::: Nested Sampling settings
        #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        if 'ns_modus' not in self.settings: 
            self.settings['ns_modus'] = 'static'
        if 'ns_nlive' not in self.settings: 
            self.settings['ns_nlive'] = 500
        if 'ns_bound' not in self.settings: 
            self.settings['ns_bound'] = 'single'
        if 'ns_sample' not in self.settings: 
            self.settings['ns_sample'] = 'rwalk'
        if 'ns_tol' not in self.settings:
            self.settings['ns_tol'] = 0.01
        if 'ns_backend' not in self.settings:
            self.settings['ns_backend'] = 'dynesty'
        # UltraNest-specific knobs (ignored when ns_backend != 'ultranest')
        if 'un_min_ess' not in self.settings:
            self.settings['un_min_ess'] = 400
        if 'un_max_iters' not in self.settings:
            self.settings['un_max_iters'] = None

        self.settings['ns_nlive'] = int(self.settings['ns_nlive'])
        self.settings['ns_tol'] = float(self.settings['ns_tol'])
        self.settings['ns_backend'] = str(self.settings['ns_backend']).strip().lower()
        self.settings['un_min_ess'] = int(self.settings['un_min_ess'])
        if self.settings['un_max_iters'] in (None, '', 'None', 'none'):
            self.settings['un_max_iters'] = None
        else:
            self.settings['un_max_iters'] = int(self.settings['un_max_iters'])
        
#        if self.settings['ns_sample'] == 'auto':
#            if self.ndim < 10:
#                self.settings['ns_sample'] = 'unif'
#                print('Using ns_sample=="unif".')
#            elif 10 <= self.ndim <= 20:
#                self.settings['ns_sample'] = 'rwalk'
#                print('Using ns_sample=="rwalk".')
#            else:
#                self.settings['ns_sample'] = 'slice'
#                print('Using ns_sample=="slice".')
        
        
        #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        #::: host & companion grids, limb darkening laws, shapes, etc.
        #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        for companion in self.settings['companions_all']:
            for inst in self.settings['inst_all']:
                
                if 'host_grid_'+inst not in self.settings: 
                    self.settings['host_grid_'+inst] = 'default'
                    
                if companion+'_grid_'+inst not in self.settings: 
                    self.settings[companion+'_grid_'+inst] = 'default'
                    
                # host_ld_law default: when the key is absent or blank, fall
                # back to 'quad'. The prior None default silently disabled
                # limb darkening, which made host_ldc_q1/q2 in params.csv
                # appear to have no effect on the transit shape. Users who
                # genuinely want no LD must write host_ld_law_<inst>,none
                # explicitly (handled by the elif → None below).
                _h_key = 'host_ld_law_'+inst
                if (_h_key not in self.settings) or (len(str(self.settings[_h_key]))==0):
                    self.settings[_h_key] = 'quad'
                elif str(self.settings[_h_key]).lower() == 'none':
                    self.settings[_h_key] = None

                if is_empty_or_none(companion+'_ld_law_'+inst):
                    self.settings[companion+'_ld_law_'+inst] = None
 
                if is_empty_or_none('host_ld_space_'+inst): 
                    self.settings['host_ld_space_'+inst] = 'q'
                    
                if is_empty_or_none(companion+'_ld_space_'+inst):
                    self.settings[companion+'_ld_space_'+inst] = 'q'        
                    
                if 'host_shape_'+inst not in self.settings: 
                    self.settings['host_shape_'+inst] = 'sphere'
                    
                if companion+'_shape_'+inst not in self.settings: 
                    self.settings[companion+'_shape_'+inst] = 'sphere'
                    
                    
        for companion in self.settings['companions_rv']:
            for inst in list(self.settings['inst_rv']) + list(self.settings['inst_rv2']):
                if companion+'_flux_weighted_'+inst in self.settings: 
                    self.settings[companion+'_flux_weighted_'+inst] = set_bool(self.settings[companion+'_flux_weighted_'+inst])
                else:
                    self.settings[companion+'_flux_weighted_'+inst] = False
        
    
        if 'exact_grav' in self.settings: 
            self.settings['exact_grav'] = set_bool(self.settings['exact_grav'])
        else:
            self.settings['exact_grav'] = False
        
        
        #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        #::: Phase curve styles
        #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        if is_empty_or_none('phase_curve_style'):
            self.settings['phase_curve_style'] = None
        if self.settings['phase_curve_style'] not in [None, 'sine_series', 'sine_physical', 'ellc_physical', 'GP']:
            raise ValueError("The setting 'phase_curve_style' must be one of [None, 'sine_series', 'sine_physical', 'ellc_physical', 'GP'], but was '"+str(self.settings['phase_curve_style'])+"'.")
        if (self.settings['phase_curve'] is True) and (self.settings['phase_curve_style'] is None):
            raise ValueError("You chose 'phase_curve=True' but did not select a 'phase_curve_style'; please select one of ['sine_series', 'sine_physical', 'ellc_physical', 'GP'].")
        if (self.settings['phase_curve'] is False) and (self.settings['phase_curve_style'] in ['sine_series', 'sine_physical', 'ellc_physical', 'GP']):
           raise ValueError("You chose 'phase_curve=False' but also selected a 'phase_curve_style'; please double check and set 'phase_curve_style=None' (or remove it).")
                               
            
        #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        #::: Stellar variability
        #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        for key in ['flux', 'rv', 'rv2']:
            if ('stellar_var_'+key not in self.settings) or (self.settings['stellar_var_'+key] is None) or (self.settings['stellar_var_'+key].lower()=='none'): 
                self.settings['stellar_var_'+key] = 'none'
                     
                     
        #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        #::: Baselines
        #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        for inst in self.settings['inst_all']:
            if inst in self.settings['inst_phot']: key='flux'
            elif inst in self.settings['inst_rv']: key='rv'
            elif inst in self.settings['inst_rv2']: key='rv2'
            
            #::: default
            #::: if the user gives no baseline, the default is 'none'
            if 'baseline_'+key+'_'+inst not in self.settings: 
                self.settings['baseline_'+key+'_'+inst] = 'none'

            #::: hybrid_spline
            #::: the user can define the s value directly, e.g. as "hybrid_spline 0.001"
            #::: this block serves to split up this input and assign it to the right functions
            if ('hybrid_spline' in self.settings['baseline_'+key+'_'+inst])\
                and (len(self.settings['baseline_'+key+'_'+inst].split(' '))>1): 
                s = self.settings['baseline_'+key+'_'+inst].split(' ')[1]
                self.settings['baseline_'+key+'_'+inst] = 'hybrid_spline_s'
                self.settings['baseline_'+key+'_'+inst+'_args'] = s #any arguments coming with this baseline (for future expandability; for now it is simply the s-value)
                
            #::: sample_GP
            #::: make sure the keywords are updated correctly
            elif self.settings['baseline_'+key+'_'+inst] == 'sample_GP': 
                 warnings.warn('You are using outdated keywords. Automatically renaming sample_GP ---> sample_GP_Matern32.'+'. Please update your files before the Duolingo owl comes to get you.') #, category=DeprecationWarning)
                 self.settings['baseline_'+key+'_'+inst] = 'sample_GP_Matern32'
                 
            #::: baseline against custom series
            #::: allows the user to fit a baseline not vs. time but vs. a chosen custom series
            if 'baseline_'+key+'_'+inst+'_against' not in self.settings:
                self.settings['baseline_'+key+'_'+inst+'_against'] = 'time'
            if self.settings['baseline_'+key+'_'+inst+'_against'] not in ['time','custom_series']:
                raise ValueError("The setting 'baseline_'+key+'_'+inst+'_against' must be one of ['time', custom_series'], but was '" + self.settings['baseline_'+key+'_'+inst+'_against'] + "'.")


        #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        #::: Baseline share groups (joint celerite GP across instruments)
        #::: settings.csv format:
        #:::   baseline_share_flux,muscat_g:muscat_r:muscat_i:muscat_z
        #::: or multiple groups separated by spaces:
        #:::   baseline_share_flux,g1lead:g1f1 g2lead:g2f1:g2f2
        #::: All members of a group must be in inst_<key2>, must use the same
        #::: sample_GP_* baseline type as the leader (first member), and may
        #::: appear in at most one group. Followers inherit the leader's GP
        #::: hyperparameters via the existing coupled_with mechanism (see
        #::: load_params), so plotting/output paths see consistent draws.
        #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        _supported_share_gps = {
            'sample_GP_Matern32',
            'sample_GP_SHO',
            'sample_GP_real',
            'sample_GP_complex',
        }
        for key, key2 in zip(['flux', 'rv', 'rv2'],
                             ['inst_phot', 'inst_rv', 'inst_rv2']):
            skey = 'baseline_share_' + key
            raw = self.settings.get(skey, None)
            groups = []
            if raw is not None and str(raw).strip() not in ('', 'none', 'None'):
                for group_str in str(raw).split():
                    members = [m for m in group_str.split(':') if m]
                    if len(members) >= 1:
                        groups.append(members)
            leader_of = {}
            followers_of = {}
            for members in groups:
                # Duplicate members within a single group are a typo, not a
                # legal aliasing — refuse before they shadow real instruments.
                if len(set(members)) != len(members):
                    raise ValueError(
                        "baseline_share_{k}: group '{g}' contains duplicate "
                        "members.".format(k=key, g=':'.join(members))
                    )
                # A singleton group shares nothing (joint-GP path collapses to
                # the legacy per-inst path). Almost always a user typo where
                # the colon-separated follower list got dropped — warn so they
                # notice instead of silently getting independent-GP behaviour.
                if len(members) == 1:
                    warnings.warn(
                        "baseline_share_{k}: group '{g}' has only one member "
                        "and shares nothing. Did you forget the colon-"
                        "separated follower list?".format(
                            k=key, g=members[0])
                    )

                leader = members[0]
                followers = members[1:]

                # The leader of group A cannot also appear (as leader OR
                # follower) in group B — that would make the alias graph
                # ambiguous. The follower-check below covers leader-as-
                # follower; this check covers leader-as-duplicate-leader.
                if leader in leader_of:
                    raise ValueError(
                        "baseline_share_{k}: '{l}' appears as a leader in "
                        "more than one group (or as a follower in an earlier "
                        "group).".format(k=key, l=leader)
                    )

                if leader not in self.settings[key2]:
                    raise ValueError(
                        "baseline_share_{k}: leader '{l}' is not in {k2}. "
                        "Every member of a share group must be listed in {k2}.".format(
                            k=key, l=leader, k2=key2,
                        )
                    )
                leader_base = self.settings.get('baseline_'+key+'_'+leader, 'none')
                if leader_base not in _supported_share_gps:
                    raise ValueError(
                        "baseline_share_{k}: leader '{l}' has baseline "
                        "'{b}' which is not a supported GP kernel for sharing "
                        "(must be one of {gps}).".format(
                            k=key, l=leader, b=leader_base,
                            gps=sorted(_supported_share_gps),
                        )
                    )
                if self.settings.get(
                        'baseline_'+key+'_'+leader+'_against', 'time') != 'time':
                    raise ValueError(
                        "baseline_share_{k}: leader '{l}' must use "
                        "baseline_{k}_{l}_against=time for a joint GP to be "
                        "well-defined across instruments.".format(k=key, l=leader)
                    )
                for f in followers:
                    if f not in self.settings[key2]:
                        raise ValueError(
                            "baseline_share_{k}: follower '{f}' is not in "
                            "{k2}.".format(k=key, f=f, k2=key2)
                        )
                    if f in leader_of:
                        raise ValueError(
                            "baseline_share_{k}: '{f}' appears in more than "
                            "one share group.".format(k=key, f=f)
                        )
                    f_base = self.settings.get('baseline_'+key+'_'+f, 'none')
                    if f_base not in ('none', leader_base):
                        raise ValueError(
                            "baseline_share_{k}: follower '{f}' has "
                            "baseline_{k}_{f}={fb} but leader '{l}' has "
                            "{lb}. Followers must inherit the leader's GP "
                            "type (leave blank or matching).".format(
                                k=key, f=f, fb=f_base, l=leader, lb=leader_base,
                            )
                        )
                    # If the follower's `_against` is explicitly set in the
                    # user's settings.csv (not just the default), it must
                    # match the leader's. Silently overriding the user would
                    # hide a real configuration mistake.
                    f_against_key = 'baseline_'+key+'_'+f+'_against'
                    if f_against_key in self._settings_raw_keys:
                        f_against = self.settings.get(f_against_key, 'time')
                        if f_against != 'time':
                            raise ValueError(
                                "baseline_share_{k}: follower '{f}' has "
                                "{ak}={av} but the share group requires "
                                "'time'. Remove the explicit setting.".format(
                                    k=key, f=f, ak=f_against_key, av=f_against,
                                )
                            )
                    # propagate leader's baseline settings to follower
                    self.settings['baseline_'+key+'_'+f] = leader_base
                    self.settings['baseline_'+key+'_'+f+'_against'] = 'time'
                    args_key = 'baseline_'+key+'_'+leader+'_args'
                    if args_key in self.settings:
                        self.settings['baseline_'+key+'_'+f+'_args'] = self.settings[args_key]
                    leader_of[f] = leader
                    followers_of.setdefault(leader, []).append(f)
                leader_of.setdefault(leader, leader)
            self.settings[skey + '_groups'] = groups
            self.settings[skey + '_leader_of'] = leader_of
            self.settings[skey + '_followers_of'] = followers_of


        #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        #::: Errors
        #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        for inst in self.settings['inst_all']:
            if inst in self.settings['inst_phot']: key='flux'
            elif inst in self.settings['inst_rv']: key='rv'
            elif inst in self.settings['inst_rv2']: key='rv2'
            if 'error_'+key+'_'+inst not in self.settings: 
                self.settings['error_'+key+'_'+inst] = 'sample'
            
        # for inst in self.settings['inst_phot']:
        #     for key in ['flux']:
        #         if 'error_'+key+'_'+inst not in self.settings: 
        #             self.settings['error_'+key+'_'+inst] = 'sample'

        # for inst in self.settings['inst_rv']:
        #     for key in ['rv']:
        #         if 'error_'+key+'_'+inst not in self.settings: 
        #             self.settings['error_'+key+'_'+inst] = 'sample'
                    
                
        #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        #::: Color plot
        #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        if 'color_plot' not in self.settings.keys():
            self.settings['color_plot'] = False
            
            
        #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        #::: Companion colors
        #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        for i, companion in enumerate( self.settings['companions_all'] ):
            self.settings[companion+'_color'] = sns.color_palette()[i]
        
        
        #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        #::: Plot zoom window
        #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        if 'zoom_window' not in self.settings:
            self.settings['zoom_window'] = 8./24. #8h window around transit/eclipse midpoint by Default
        else:
            self.settings['zoom_window'] = float(self.settings['zoom_window'])
            
            
        #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        #::: Exposure time interpolation
        #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        for inst in self.settings['inst_all']:
            #::: if t_exp is given
            if 't_exp_'+inst in self.settings.keys() and len(self.settings['t_exp_'+inst]):
                t_exp = self.settings['t_exp_'+inst].split(' ')
                #if float
                if len(t_exp)==1:
                    self.settings['t_exp_'+inst] = float(t_exp[0])
                #if array
                else:
                    self.settings['t_exp_'+inst] = np.array([ float(t) for t in t_exp ])
            #::: if not given / given as an empty field
            else:
                self.settings['t_exp_'+inst] = None
                
            #::: if t_exp_n_int is given
            if 't_exp_'+inst in self.settings \
                and 't_exp_n_int_'+inst in self.settings \
                and len(self.settings['t_exp_n_int_'+inst]):
                    
                self.settings['t_exp_n_int_'+inst] = int(self.settings['t_exp_n_int_'+inst])
                if self.settings['t_exp_n_int_'+inst] < 1:
                    raise ValueError('"t_exp_n_int_'+inst+'" must be >= 1, but is given as '+str(self.settings['t_exp_n_int_'+inst])+' in params.csv')
            else:
                self.settings['t_exp_n_int_'+inst] = None
  
    
        #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        #::: Number of spots
        #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        for inst in self.settings['inst_all']:
            if 'host_N_spots_'+inst in self.settings and len(self.settings['host_N_spots_'+inst]):
                self.settings['host_N_spots_'+inst] = int(self.settings['host_N_spots_'+inst])
            else:
                self.settings['host_N_spots_'+inst] = 0
        
            for companion in self.settings['companions_all']:
                if companion+'_N_spots_'+inst in self.settings:
                    self.settings[companion+'_N_spots_'+inst] = int(self.settings[companion+'_N_spots_'+inst])
                else:
                    self.settings[companion+'_N_spots_'+inst] = 0
                    
        
        #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        #::: Number of flares
        #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        if 'N_flares' in self.settings and len(self.settings['N_flares'])>0:
            self.settings['N_flares'] = int(self.settings['N_flares'])
        else:
            self.settings['N_flares'] = 0
        
        
        
        
                
    ###############################################################################
    #::: load params
    ###############################################################################
    def load_params(self):
        '''
        For the full list of options see www.allesfitter.com
        '''
    
        #==========================================================================
        #::: load params.csv
        #==========================================================================
        buf = np.genfromtxt(os.path.join(self.datadir,'params.csv'), delimiter=',',comments='#',dtype=None,encoding='utf-8',names=True)

        #==========================================================================
        #::: luser-proof: reject duplicate parameter rows
        #::: numpy.genfromtxt is happy with duplicate names but downstream
        #::: dict-assembly silently last-wins, which corrupts chromatic configs
        #::: edited by hand (e.g. two rows for b_rr_tess with different priors).
        #==========================================================================
        _names = list(np.atleast_1d(buf['name']))
        _stripped = [n.strip() for n in _names if str(n).strip() not in ('user-given:', 'automatically set:')]
        _seen = {}
        for _n in _stripped:
            _seen[_n] = _seen.get(_n, 0) + 1
        _dups = sorted(k for k, v in _seen.items() if v > 1)
        if _dups:
            raise ValueError(
                "params.csv contains duplicate rows for: " + ", ".join(_dups) +
                ". Each parameter must be defined exactly once."
            )

        #==========================================================================
        #::: luser-proof: chromatic suffix must match a known bandpass
        #::: Catches typos (e.g. b_rr_tes vs tess) that would otherwise be
        #::: silently ignored, leaving the fit to use a default 0 for the rr.
        #==========================================================================
        _bandpass_map = self.settings.get('bandpass', {}) or {}
        _known_bands = set(_bandpass_map.values())
        _companions = self.settings.get('companions_all', []) or []
        _is_chromatic = bool(self.settings.get('chromatic', False))
        if _known_bands and _companions:
            _bad_chromatic = []
            for _c in _companions:
                _prefix = _c + '_rr_'
                for _n in _stripped:
                    if _n.startswith(_prefix):
                        _suffix = _n[len(_prefix):]
                        if _suffix not in _known_bands:
                            _bad_chromatic.append((_n, _suffix))
            if _bad_chromatic:
                _msg_pairs = "; ".join(f"{k} (suffix '{s}')" for k, s in _bad_chromatic)
                raise ValueError(
                    "params.csv references unknown bandpass(es): " + _msg_pairs +
                    ". Known bandpasses (from settings.csv 'bandpass'): " +
                    sorted(_known_bands).__repr__() + "."
                )

        #==========================================================================
        #::: luser-proof: chromatic mode requires one rr row per (companion,
        #::: bandpass). Without that, the validator silently defaults the
        #::: missing keys to None and the likelihood falls back to the
        #::: unsuffixed b_rr — a fit that looks chromatic in settings.csv but
        #::: is achromatic in practice. Catch both half-states up front:
        #:::   - chromatic settings + plain `<c>_rr` row present (ambiguous),
        #:::   - chromatic settings + at least one expected `<c>_rr_<bp>` row missing.
        #==========================================================================
        if _is_chromatic and _companions:
            _problems = []
            _stripped_set = set(_stripped)
            for _c in _companions:
                _achromatic_key = _c + '_rr'
                _expected = {f'{_c}_rr_{bp}' for bp in _known_bands}
                _present = _expected & _stripped_set
                _missing = sorted(_expected - _present)
                _has_achromatic = _achromatic_key in _stripped_set
                if _has_achromatic and not _present:
                    _problems.append(
                        f"companion '{_c}': params.csv has '{_achromatic_key}' but "
                        f"settings.csv 'bandpass' is chromatic. Replace it with "
                        f"one row per bandpass: " + ", ".join(sorted(_expected)) + "."
                    )
                elif _missing and not _has_achromatic:
                    _problems.append(
                        f"companion '{_c}': chromatic mode requires a row per "
                        f"bandpass; missing " + ", ".join(_missing) + "."
                    )
                elif _missing and _has_achromatic:
                    _problems.append(
                        f"companion '{_c}': params.csv mixes the achromatic key "
                        f"'{_achromatic_key}' with chromatic rows; missing "
                        + ", ".join(_missing) + ". Pick one shape and remove "
                        f"'{_achromatic_key}'."
                    )
            if _problems:
                raise ValueError(
                    "Chromatic configuration mismatch between settings.csv and "
                    "params.csv:\n  - " + "\n  - ".join(_problems)
                )


        #==========================================================================
        #::: function to assure backwards compability
        #==========================================================================
        def backwards_compability(key_new, key_deprecated):
            if key_deprecated in np.atleast_1d(buf['name']):
                warnings.warn('You are using outdated keywords. Automatically renaming '+key_deprecated+' ---> '+key_new+'. Please fix this before the Duolingo owl comes to get you.') #, category=DeprecationWarning)
                ind = np.where(buf['name'] == key_deprecated)[0]
                np.atleast_1d(buf['name'])[ind] = key_new
                
                
        #==========================================================================
        #::: luser-proof: backwards compability 
        # (has to happend first thing and right inside buf['name'])
        #==========================================================================
        for inst in self.settings['inst_all']:
            backwards_compability(key_new='host_ldc_q1_'+inst, key_deprecated='ldc_q1_'+inst)
            backwards_compability(key_new='host_ldc_q2_'+inst, key_deprecated='ldc_q2_'+inst)
            backwards_compability(key_new='host_ldc_q3_'+inst, key_deprecated='ldc_q3_'+inst)
            backwards_compability(key_new='host_ldc_q4_'+inst, key_deprecated='ldc_q4_'+inst)
            backwards_compability(key_new='ln_err_flux_'+inst, key_deprecated='log_err_flux_'+inst)
            backwards_compability(key_new='ln_jitter_rv_'+inst, key_deprecated='log_jitter_rv_'+inst)
            backwards_compability(key_new='baseline_gp_matern32_lnsigma_flux_'+inst, key_deprecated='baseline_gp1_flux_'+inst)
            backwards_compability(key_new='baseline_gp_matern32_lnrho_flux_'+inst, key_deprecated='baseline_gp2_flux_'+inst)
            backwards_compability(key_new='baseline_gp_matern32_lnsigma_rv_'+inst, key_deprecated='baseline_gp1_rv_'+inst)
            backwards_compability(key_new='baseline_gp_matern32_lnrho_rv_'+inst, key_deprecated='baseline_gp2_rv_'+inst)
                   
                    
        #==========================================================================
        #::: luser-proof: check for allowed keys to catch typos etc.
        #==========================================================================  
        def get_valid_param_patterns():
            valid_patterns = set()
            companions = self.settings.get('companions_all', [])
            inst_all = self.settings.get('inst_all', [])
            inst_phot = self.settings.get('inst_phot', [])
            inst_rv = self.settings.get('inst_rv', [])
            
            for companion in companions:
                valid_patterns.add(companion+'_rr')
                valid_patterns.add(companion+'_rsuma')
                valid_patterns.add(companion+'_cosi')
                valid_patterns.add(companion+'_epoch')
                valid_patterns.add(companion+'_period')
                valid_patterns.add(companion+'_f_c')
                valid_patterns.add(companion+'_f_s')
                valid_patterns.add(companion+'_sbratio')
                valid_patterns.add(companion+'_a')
                valid_patterns.add(companion+'_q')
                valid_patterns.add(companion+'_K')
                valid_patterns.add(companion+'_dil')
                valid_patterns.add(companion+'_ld_law')
                valid_patterns.add(companion+'_ld_space')
                valid_patterns.add(companion+'_shape')
                valid_patterns.add(companion+'_grid')
                valid_patterns.add(companion+'_flux_weighted')
                valid_patterns.add(companion+'_rotfac')
                valid_patterns.add(companion+'_hf')
                valid_patterns.add(companion+'_bfac')
                valid_patterns.add(companion+'_heat')
                valid_patterns.add(companion+'_lambda')
                valid_patterns.add(companion+'_vsini')
                valid_patterns.add(companion+'_N_spots')
                valid_patterns.add(companion+'_phase_curve_beaming')
                valid_patterns.add(companion+'_phase_curve_atmospheric')
                valid_patterns.add(companion+'_phase_curve_ellipsoidal')
                for i in range(1, 50):
                    valid_patterns.add(companion+'_ttv_transit_'+str(i))
                
                for inst in inst_all:
                    valid_patterns.add(companion+'_ldc_'+inst)
                    for j in range(1, 10):
                        valid_patterns.add(companion+'_ldc_q'+str(j)+'_'+inst)
                        valid_patterns.add(companion+'_ldc_u'+str(j)+'_'+inst)
                    valid_patterns.add(companion+'_gdc_'+inst)
                    valid_patterns.add(companion+'_spots_'+inst)
                    for k in range(1, 10):
                        valid_patterns.add(companion+'_spot_'+str(k)+'_long_'+inst)
                        valid_patterns.add(companion+'_spot_'+str(k)+'_lat_'+inst)
                        valid_patterns.add(companion+'_spot_'+str(k)+'_size_'+inst)
                        valid_patterns.add(companion+'_spot_'+str(k)+'_brightness_'+inst)
            
            for inst in inst_all:
                valid_patterns.add('host_ld_law_'+inst)
                valid_patterns.add('host_ld_space_'+inst)
                valid_patterns.add('host_grid_'+inst)
                valid_patterns.add('host_shape_'+inst)
                valid_patterns.add('host_flux_weighted_'+inst)
                valid_patterns.add('host_rotfac_'+inst)
                valid_patterns.add('host_hf_'+inst)
                valid_patterns.add('host_bfac_'+inst)
                valid_patterns.add('host_heat_'+inst)
                valid_patterns.add('host_lambda_'+inst)
                valid_patterns.add('host_N_spots_'+inst)
                valid_patterns.add('host_spots_'+inst)
                for j in range(1, 10):
                    valid_patterns.add('host_ldc_q'+str(j)+'_'+inst)
                    valid_patterns.add('host_ldc_u'+str(j)+'_'+inst)
                    valid_patterns.add('host_spot_'+str(j)+'_long_'+inst)
                    valid_patterns.add('host_spot_'+str(j)+'_lat_'+inst)
                    valid_patterns.add('host_spot_'+str(j)+'_size_'+inst)
                    valid_patterns.add('host_spot_'+str(j)+'_brightness_'+inst)
            
            for inst in inst_all:
                valid_patterns.add('dil_'+inst)
                valid_patterns.add('host_gdc_'+inst)
            
            for inst in inst_phot:
                valid_patterns.add('ln_err_flux_'+inst)
                valid_patterns.add('baseline_offset_flux_'+inst)
                valid_patterns.add('baseline_slope_flux_'+inst)
                valid_patterns.add('baseline_gp_matern32_lnsigma_flux_'+inst)
                valid_patterns.add('baseline_gp_matern32_lnrho_flux_'+inst)
                valid_patterns.add('baseline_gp_sho_omega_flux_'+inst)
                valid_patterns.add('baseline_gp_sho_A_flux_'+inst)
                valid_patterns.add('baseline_gp_real_omega_flux_'+inst)
                valid_patterns.add('baseline_gp_real_A_flux_'+inst)
                valid_patterns.add('baseline_gp_complex_omega_flux_'+inst)
                valid_patterns.add('baseline_gp_complex_A_flux_'+inst)
                valid_patterns.add('baseline_gp_complex_Q_flux_'+inst)
            
            for inst in inst_rv:
                valid_patterns.add('ln_jitter_rv_'+inst)
                valid_patterns.add('baseline_offset_rv_'+inst)
                valid_patterns.add('baseline_slope_rv_'+inst)
                valid_patterns.add('baseline_gp_matern32_lnsigma_rv_'+inst)
                valid_patterns.add('baseline_gp_matern32_lnrho_rv_'+inst)
            
            valid_patterns.add('R_host')
            valid_patterns.add('M_host')
            valid_patterns.add('Teff_host')
            valid_patterns.add('host_vsini')
            valid_patterns.add('host_rotfac')
            valid_patterns.add('R_host_err')
            valid_patterns.add('M_host_err')
            valid_patterns.add('Teff_host_err')
            
            for i in range(1, 10):
                valid_patterns.add('flare_'+str(i)+'_epoch')
                valid_patterns.add('flare_'+str(i)+'_duration')
                valid_patterns.add('flare_'+str(i)+'_amplitude')
                valid_patterns.add('flare_'+str(i)+'_beta')
            
            return valid_patterns
        
        def is_valid_key(key, valid_patterns):
            if key in valid_patterns:
                return True
            for pattern in valid_patterns:
                if key.startswith(pattern):
                    return True
            if key.startswith('host_ldc_q') and '_' in key:
                return True
            if key.startswith('host_ldc_u') and '_' in key:
                return True
            if '_ldc_q' in key and '_' in key:
                return True
            if '_ldc_u' in key and '_' in key:
                return True
            if key.startswith('dil_') and '_' in key:
                return True
            if key.startswith('ln_err_flux_') and '_' in key:
                return True
            if key.startswith('ln_jitter_rv_') and '_' in key:
                return True
            if key.startswith('baseline_') and '_' in key:
                return True
            return False
        
        valid_patterns = get_valid_param_patterns()
        allkeys_list = list(buf['name'])
        
        unrecognized = []
        for key in allkeys_list:
            key_clean = key.strip()
            if key_clean in ['user-given:', 'automatically set:']:
                continue
            if not is_valid_key(key_clean, valid_patterns):
                unrecognized.append(key_clean)
        
        if unrecognized:
            self.logprint('\nWARNING: The following parameters in params.csv are not recognized and will be ignored:')
            for key in unrecognized:
                self.logprint('  - '+key)
            self.logprint('')
                
                
        #==========================================================================
        #::: set up stuff   
        #==========================================================================          
        self.allkeys = np.atleast_1d(buf['name']) #len(all rows in params.csv)
        self.labels = np.atleast_1d(buf['label']) #len(all rows in params.csv)
        self.units = np.atleast_1d(buf['unit']) #len(all rows in params.csv)
        if 'truth' in buf.dtype.names:
            self.truths = np.atleast_1d(buf['truth']) #len(all rows in params.csv)
        else:
            self.truths = np.nan * np.ones(len(self.allkeys))
            
        self.params = collections.OrderedDict() #len(all rows in params.csv)
        self.params['user-given:'] = '' #just for pretty printing
        for i,key in enumerate(self.allkeys):
            #::: if it's not a "coupled parameter", then use the given value
            if np.atleast_1d(buf['value'])[i] not in list(self.allkeys):
                self.params[key] = float(np.atleast_1d(buf['value'])[i])
            #::: if it's a "coupled parameter", then write the string of the key it is coupled to
            else:
                self.params[key] = np.atleast_1d(buf['value'])[i]
                
                
        #==========================================================================
        #::: function to automatically set default params if they were not given
        #==========================================================================
        def validate(key, default, default_min, default_max):
            if (key in self.params) and (self.params[key] is not None):
                if (self.params[key] < default_min) or (self.params[key] > default_max):
                    raise ValueError("User input for "+key+" is "+str(self.params[key])+" but must lie within ["+str(default_min)+","+str(default_max)+"].")
            if (key not in self.params):
                self.params[key] = default
        
        
        #==========================================================================
        #::: luser-proof: make sure the limb darkening values are uniquely 
        #::: from either the u- or q-space
        #==========================================================================  
        def check_ld(obj, inst):
           if self.settings[obj+'_ld_space_'+inst] == 'q': 
                matches = fnmatch.filter(self.allkeys, obj+'_ldc_u*_'+inst)
                if len(matches) > 0:
                    raise ValueError("The following user input is inconsistent:\n"+\
                                     "Setting: '"+key+"' = 'q'\n"+\
                                     "Parameters: {}".format(matches))   
                        
           elif self.settings[obj+'_ld_space_'+inst] == 'u': 
                matches = fnmatch.filter(self.allkeys, obj+'_ldc_q*_'+inst)
                if len(matches) > 0:
                    raise ValueError("The following user input is inconsistent:\n"+\
                                     "Setting: '"+key+"' = 'u'\n"+\
                                     "Parameters: {}".format(matches))  
                        
        for inst in self.settings['inst_all']:
            for obj in ['host'] + self.settings['companions_all']:   
                check_ld(obj, inst)
            
            
        #==========================================================================
        #::: validate that initial guess params have reasonable values
        #==========================================================================
        self.params['automatically set:'] = '' #just for pretty printing
        for companion in self.settings['companions_all']:
            for inst in self.settings['inst_all']:
                
                # Get bandpass for this instrument (None if achromatic)
                bandpass = self.get_bandpass(inst)
                
                # Determine suffix for parameter keys
                # For chromatic mode: use bandpass name (e.g., 'tess')
                # For achromatic mode: use instrument name (e.g., 'tess')
                # This way, existing parameter naming is preserved for achromatic
                if bandpass:
                    bp_suffix = '_' + bandpass
                else:
                    bp_suffix = ''  # Will use inst as suffix in LDC below
                
                #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
                #::: ellc defaults
                #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
                
                #::: frequently used parameters
                # rr is per-bandpass in chromatic mode, per-companion in achromatic
                if bandpass:
                    rr_key = companion + '_rr' + bp_suffix
                    validate(rr_key, None, 0., np.inf)
                else:
                    validate(companion+'_rr', None, 0., np.inf)
                validate(companion+'_rsuma', None, 0., np.inf)
                validate(companion+'_cosi', 0., 0., 1.)
                validate(companion+'_epoch', 0., -np.inf, np.inf)
                validate(companion+'_period', 0., 0., np.inf)
                validate(companion+'_sbratio_'+inst, 0., 0., np.inf)
                validate(companion+'_K', 0., 0., np.inf)
                validate(companion+'_f_s', 0., -1, 1)
                validate(companion+'_f_c', 0., -1, 1)
                validate('dil_'+inst, 0., -np.inf, np.inf)
                
                #::: limb darkenings, u-space (per-bandpass in chromatic, per-inst in achromatic)
                ldc_suffix = bp_suffix if bandpass else '_' + inst
                validate('host_ldc_u1'+ldc_suffix, None, 0, 1)
                validate('host_ldc_u2'+ldc_suffix, None, 0, 1)
                validate('host_ldc_u3'+ldc_suffix, None, 0, 1)
                validate('host_ldc_u4'+ldc_suffix, None, 0, 1)
                validate(companion+'_ldc_u1'+ldc_suffix, None, 0, 1)
                validate(companion+'_ldc_u2'+ldc_suffix, None, 0, 1)
                validate(companion+'_ldc_u3'+ldc_suffix, None, 0, 1)
                validate(companion+'_ldc_u4'+ldc_suffix, None, 0, 1)

                #::: limb darkenings, q-space
                validate('host_ldc_q1'+ldc_suffix, None, 0, 1)
                validate('host_ldc_q2'+ldc_suffix, None, 0, 1)
                validate('host_ldc_q3'+ldc_suffix, None, 0, 1)
                validate('host_ldc_q4'+ldc_suffix, None, 0, 1)
                validate(companion+'_ldc_q1'+ldc_suffix, None, 0, 1)
                validate(companion+'_ldc_q2'+ldc_suffix, None, 0, 1)
                validate(companion+'_ldc_q3'+ldc_suffix, None, 0, 1)
                validate(companion+'_ldc_q4'+ldc_suffix, None, 0, 1)
                
                #::: catch exceptions
                if self.params[companion+'_period'] is None:
                    self.settings['do_not_phase_fold'] = True
                
                #::: advanced parameters
                validate(companion+'_a', None, 0., np.inf)
                validate(companion+'_q', 1., 0., np.inf)
                
                validate('didt_'+inst, None, -np.inf, np.inf)
                validate('domdt_'+inst, None, -np.inf, np.inf)
                
                validate('host_gdc_'+inst, None, 0., 1.)
                validate('host_rotfac_'+inst, 1., 0., np.inf)
                validate('host_hf_'+inst, 1.5, -np.inf, np.inf)
                validate('host_bfac_'+inst, None, -np.inf, np.inf)
                validate('host_heat_'+inst, None, -np.inf, np.inf)
                validate('host_lambda', None, -np.inf, np.inf)
                validate('host_vsini', None, -np.inf, np.inf)
                
                validate(companion+'_gdc_'+inst, None, 0., 1.)
                validate(companion+'_rotfac_'+inst, 1., 0., np.inf)
                validate(companion+'_hf_'+inst, 1.5, -np.inf, np.inf)
                validate(companion+'_bfac_'+inst, None, -np.inf, np.inf)
                validate(companion+'_heat_'+inst, None, -np.inf, np.inf)
                validate(companion+'_lambda', None, -np.inf, np.inf)
                validate(companion+'_vsini', None, -np.inf, np.inf)
        
                #::: special parameters (list type)
                if 'host_spots_'+inst not in self.params:
                    self.params['host_spots_'+inst] = None
                if companion+'_spots_'+inst not in self.params:
                    self.params[companion+'_spots_'+inst] = None
                    
                
                #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
                #::: errors and jitters
                #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
                #TODO: add validations for all errors / jitters
                    
                
                #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
                #::: baselines (and backwards compability)
                #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
                #TODO: add validations for all baseline params
                
                    
                #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
                #::: phase curve style: sine_series
                # all in ppt
                # A1 (beaming)
                # B1 (atmospheric), can be split in thermal and reflected
                # B2 (ellipsoidal)
                # B3 (ellipsoidal 2nd order)
                #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
                # if (self.settings['phase_curve_style'] == 'sine_series') and (inst in self.settings['inst_phot']):
                if (inst in self.settings['inst_phot']):
                    validate(companion+'_phase_curve_A1_'+inst, None, 0., np.inf)
                    validate(companion+'_phase_curve_B1_'+inst, None, -np.inf, 0.)
                    validate(companion+'_phase_curve_B1_shift_'+inst, 0., -np.inf, np.inf)
                    validate(companion+'_phase_curve_B1t_'+inst, None, -np.inf, 0.)
                    validate(companion+'_phase_curve_B1t_shift_'+inst, 0., -np.inf, np.inf)
                    validate(companion+'_phase_curve_B1r_'+inst, None, -np.inf, 0.)
                    validate(companion+'_phase_curve_B1r_shift_'+inst, 0., -np.inf, np.inf)
                    validate(companion+'_phase_curve_B2_'+inst, None, -np.inf, 0.)
                    validate(companion+'_phase_curve_B3_'+inst, None, -np.inf, 0.)

                                       
                #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
                #::: phase curve style: sine_physical  
                # A1 (beaming)
                # B1 (atmospheric), can be split in thermal and reflected
                # B2 (ellipsoidal)
                # B3 (ellipsoidal 2nd order)  
                #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
                # if (self.settings['phase_curve_style'] == 'sine_physical') and (inst in self.settings['inst_phot']):
                if (inst in self.settings['inst_phot']):
                    validate(companion+'_phase_curve_beaming_'+inst, None, 0., np.inf)
                    validate(companion+'_phase_curve_atmospheric_'+inst, None, 0., np.inf)
                    validate(companion+'_phase_curve_atmospheric_shift_'+inst, 0., -np.inf, np.inf)
                    validate(companion+'_phase_curve_atmospheric_thermal_'+inst, None, 0., np.inf)
                    validate(companion+'_phase_curve_atmospheric_thermal_shift_'+inst, 0., -np.inf, np.inf)
                    validate(companion+'_phase_curve_atmospheric_reflected_'+inst, None, 0., np.inf)
                    validate(companion+'_phase_curve_atmospheric_reflected_shift_'+inst, 0., -np.inf, np.inf)
                    validate(companion+'_phase_curve_ellipsoidal_'+inst, None, 0., np.inf)
                    validate(companion+'_phase_curve_ellipsoidal_2nd_'+inst, None, 0., np.inf)
                
                
                #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
                #::: to avoid a bug/feature in ellc, if either property is >0, set the other to 1-15 (not 0):
                #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
                if self.params[companion+'_heat_'+inst] is not None:
                    if (self.params[companion+'_sbratio_'+inst] == 0) and (self.params[companion+'_heat_'+inst] > 0):
                        self.params[companion+'_sbratio_'+inst] = 1e-15        #this is to avoid a bug/feature in ellc
                    if (self.params[companion+'_sbratio_'+inst] > 0) and (self.params[companion+'_heat_'+inst] == 0):
                        self.params[companion+'_heat_'+inst] = 1e-15           #this is to avoid a bug/feature in ellc
              

                #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
                #::: luser proof: avoid conflicting/degenerate phase curve commands
                #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
                if (inst in self.settings['inst_phot']) and (self.settings['phase_curve'] == True):
                    phase_curve_model_1 = (self.params[companion+'_phase_curve_B1_'+inst] is not None)
                    phase_curve_model_2 = ((self.params[companion+'_phase_curve_B1t_'+inst] is not None) or (self.params[companion+'_phase_curve_B1r_'+inst] is not None))
                    phase_curve_model_3 = (self.params[companion+'_phase_curve_atmospheric_'+inst] is not None)
                    phase_curve_model_4 = ((self.params[companion+'_phase_curve_atmospheric_thermal_'+inst] is not None) or (self.params[companion+'_phase_curve_atmospheric_reflected_'+inst] is not None))
                    phase_curve_model_5 = ((self.params['host_bfac_'+inst] is not None) or (self.params['host_heat_'+inst] is not None) or \
                                           (self.params['host_gdc_'+inst] is not None) or (self.settings['host_shape_'+inst]!='sphere') or \
                                           (self.params[companion+'_bfac_'+inst] is not None) or (self.params[companion+'_heat_'+inst] is not None) or \
                                           (self.params[companion+'_gdc_'+inst] is not None) or (self.settings[companion+'_shape_'+inst]!='sphere'))
                    if (phase_curve_model_1 + phase_curve_model_2 + phase_curve_model_3 + phase_curve_model_4 + phase_curve_model_5) > 1:
                        raise ValueError('You can use either\n'\
                                         +'1) the sine_series phase curve model with "*_phase_curve_B1_*",\n'\
                                         +'2) the sine_series phase curve model with "*_phase_curve_B1t_*" and "*_phase_curve_B1r_*", or\n'\
                                         +'3) the sine_physical phase curve model with "*_phase_curve_atmospheric_*",\n'\
                                         +'4) the sine_physical phase curve model with "*_phase_curve_atmospheric_thermal_*" and "*_phase_curve_atmospheric_reflected_*", or\n'\
                                         +'5) the ellc_physical phase curve model with "*_bfac_*", "*_heat_*", "*_gdc_*" etc.\n'\
                                         +'but you shall not pass with a mix&match.')
                    
                        
        #==========================================================================
        #::: coupled params
        #==========================================================================
        if 'coupled_with' in buf.dtype.names:
            self.coupled_with = buf['coupled_with']
        else:
            self.coupled_with = [None]*len(self.allkeys)
            
        for i, key in enumerate(self.allkeys):
            if isinstance(self.coupled_with[i], str) and (len(self.coupled_with[i])>0):
                self.params[key] = self.params[self.coupled_with[i]]           #luser proof: automatically set the values of the params coupled to another param
                buf['fit'][i] = 0                                              #luser proof: automatically set fit=0 for the params coupled to another param


        #==========================================================================
        #::: baseline share groups: alias follower GP hyperparameters to leader
        #::: Each follower GP hyperparameter (baseline_gp_*_{key}_{follower})
        #::: gets the leader's value here so self.params is self-consistent at
        #::: load time. Per-iteration re-aliasing happens in computer.update_params
        #::: so that follower entries track the leader as theta changes. Together
        #::: this means the NS fit vector contains only the leader's hypers,
        #::: while computer.py's joint-GP code path assembles all group members'
        #::: residuals into a single celerite GP under the leader's name.
        #==========================================================================
        _allkeys_set = set(self.allkeys)
        for k_share in ('flux', 'rv', 'rv2'):
            followers_of = self.settings.get(
                'baseline_share_'+k_share+'_followers_of', {})
            if not followers_of:
                continue
            for leader, followers in followers_of.items():
                # Cross-file sanity: settings.csv says the leader uses a
                # specific GP kernel — every required hyperparameter row for
                # that kernel must exist in params.csv. Without this check,
                # the user would only see a KeyError deep inside
                # baseline_get_gp at the first likelihood call.
                leader_base = self.settings.get(
                    'baseline_'+k_share+'_'+leader, 'none')
                required_prefixes = BASELINE_GP_REQUIRED_HYPERS.get(
                    leader_base, ())
                missing = []
                for rp in required_prefixes:
                    if rp + k_share + '_' + leader not in self.params:
                        missing.append(rp + k_share + '_' + leader)
                if missing:
                    raise ValueError(
                        "baseline_share_{k}: leader '{l}' declares "
                        "baseline_{k}_{l}={b} but params.csv is missing "
                        "the required row(s) {m}. Every share-group leader "
                        "must own all GP hyperparameter rows for its "
                        "declared kernel.".format(
                            k=k_share, l=leader, b=leader_base, m=missing,
                        )
                    )

                for prefix in BASELINE_GP_HYPER_PREFIXES:
                    leader_key = prefix + k_share + '_' + leader
                    if leader_key not in self.params:
                        continue
                    for f in followers:
                        follower_key = prefix + k_share + '_' + f
                        if follower_key in _allkeys_set:
                            idx = list(self.allkeys).index(follower_key)
                            coupled = self.coupled_with[idx]
                            is_coupled = (
                                isinstance(coupled, str) and len(coupled) > 0
                            )
                            if is_coupled and coupled != leader_key:
                                # Coupled to something other than the leader's
                                # corresponding row — fundamentally
                                # inconsistent with the share-group alias.
                                raise ValueError(
                                    "baseline_share_{k}: '{fk}' is "
                                    "coupled_with='{cw}' but instrument "
                                    "'{f}' is in a share group led by '{l}', "
                                    "which expects coupled_with='{lk}'. "
                                    "Remove the explicit coupling or point "
                                    "it at the leader's key.".format(
                                        k=k_share, fk=follower_key, cw=coupled,
                                        f=f, l=leader, lk=leader_key,
                                    )
                                )
                            if int(np.atleast_1d(buf['fit'])[idx]) == 1 and not is_coupled:
                                raise ValueError(
                                    "baseline_share_{k}: '{fk}' has fit=1 in "
                                    "params.csv but instrument '{f}' is in a "
                                    "share group led by '{l}'. Either remove "
                                    "the follower row, set fit=0, or use "
                                    "coupled_with={lk}.".format(
                                        k=k_share, fk=follower_key, f=f,
                                        l=leader, lk=leader_key,
                                    )
                                )
                            buf['fit'][idx] = 0
                        self.params[follower_key] = self.params[leader_key]


        #==========================================================================
        #::: mark to be fitted params
        #==========================================================================
        self.ind_fit = (buf['fit']==1)                  #len(all rows in params.csv)
        
        self.fitkeys = buf['name'][ self.ind_fit ]      #len(ndim)
        self.fitlabels = self.labels[ self.ind_fit ]    #len(ndim)
        self.fitunits = self.units[ self.ind_fit ]      #len(ndim)
        self.fittruths = self.truths[ self.ind_fit ]    #len(ndim)
        self.theta_0 = buf['value'][ self.ind_fit ]     #len(ndim)
        
        if 'init_err' in buf.dtype.names:
            self.init_err = buf['init_err'][ self.ind_fit ] #len(ndim)
        else:
            self.init_err = 1e-8
        
        self.bounds = [ str(item).split(' ') for item in buf['bounds'][ self.ind_fit ] ] #len(ndim)
        for i, item in enumerate(self.bounds):
            if item[0] in ['uniform', 'normal']:
                self.bounds[i] = [ item[0], float(item[1]), float(item[2]) ]
            elif item[0] in ['trunc_normal']:
                self.bounds[i] = [ item[0], float(item[1]), float(item[2]), float(item[3]), float(item[4]) ]
            else:
                raise ValueError('Bounds have to be "uniform", "normal" or "trunc_normal". Input from "params.csv" was "'+self.bounds[i][0]+'".')
    
        self.ndim = len(self.theta_0)                   #len(ndim)

    
        #==========================================================================
        #::: luser proof: check if all initial guesses lie within their bounds
        #==========================================================================
        #TODO: make this part of the validate() function
        for th, b, key in zip(self.theta_0, self.bounds, self.fitkeys):
                  
            #:::: test bounds
            if (b[0] == 'uniform') and not (b[1] <= th <= b[2]): 
                raise ValueError('The initial guess for '+key+' lies outside of its bounds.')
                
            elif (b[0] == 'normal') and ( np.abs(th - b[1]) > 3*b[2] ):
                answer = input('The initial guess for '+key+' lies more than 3 sigma from its prior\n'+\
                      'What do you want to do?\n'+\
                      '1 : continue at any sacrifice \n'+\
                      '2 : stop and let me fix the params.csv file \n')
                if answer==1: 
                    pass
                else:
                    raise ValueError('User aborted the run.')
                    
            elif (b[0] == 'trunc_normal') and not (b[1] <= th <= b[2]): 
                raise ValueError('The initial guess for '+key+' lies outside of its bounds.')
                
            elif (b[0] == 'trunc_normal') and ( np.abs(th - b[3]) > 3*b[4] ): 
                answer = input('The initial guess for '+key+' lies more than 3 sigma from its prior\n'+\
                      'What do you want to do?\n'+\
                      '1 : continue at any sacrifice \n'+\
                      '2 : stop and let me fix the params.csv file \n')
                if answer==1: 
                    pass
                else:
                    raise ValueError('User aborted the run.')
            
            

    ###############################################################################
    #::: load data
    ###############################################################################
    def load_data(self):
        '''
        Example: 
        -------
            A lightcurve is stored as
                data['TESS']['time'], data['TESS']['flux'], etc.
            A RV curve is stored as
                data['HARPS']['time'], data['HARPS']['flux'], etc.
        '''
        self.fulldata = {}
        self.data = {}
        
        #======================================================================
        #::: photometry
        #======================================================================
        for inst in self.settings['inst_phot']:
            try:
                time, flux, flux_err, custom_series = np.genfromtxt(os.path.join(self.datadir,inst+'.csv'), delimiter=',', dtype=float, unpack=True)[0:4]     
            except:
                time, flux, flux_err = np.genfromtxt(os.path.join(self.datadir,inst+'.csv'), delimiter=',', dtype=float, unpack=True)[0:3]     
                custom_series = np.zeros_like(time)
            if any(np.isnan(time*flux*flux_err*custom_series)):
                raise ValueError('There are NaN values in "'+inst+'.csv". Please make sure everything is fine with your data, then exclude these rows from the file and restart.')
            if any(flux_err==0):
                raise ValueError('There are uncertainties with values of 0 in "'+inst+'.csv". Please make sure everything is fine with your data, then exclude these rows from the file and restart.')
            if any(flux_err<0):
                raise ValueError('There are uncertainties with negative values in "'+inst+'.csv". Please make sure everything is fine with your data, then exclude these rows from the file and restart.')
            if not all(np.diff(time)>=0):
                raise ValueError('The time array in "'+inst+'.csv" is not sorted. Please make sure the file is not corrupted, then sort it by time and restart.')
            elif not all(np.diff(time)>0):
                warnings.warn('There are repeated time stamps in the time array in "'+inst+'.csv". Please make sure the file is not corrupted (e.g. insuffiecient precision in your time stamps).')
#                overwrite = str(input('There are repeated time stamps in the time array in "'+inst+'.csv". Please make sure the file is not corrupted (e.g. insuffiecient precision in your time stamps).'+\
#                                      'What do you want to do?\n'+\
#                                      '1 : continue and hope for the best; no risk, no fun; #yolo\n'+\
#                                      '2 : abort\n'))
#                if (overwrite == '1'):
#                    pass
#                else:
#                    raise ValueError('User aborted operation.')

            #::: Raw-flux outlier removal (applied before fulldata is captured
            #::: so that all downstream consumers see the clipped rows).
            #::: Drops rows outside [flux_min_raw, flux_max_raw]; either bound
            #::: may be None to make it one-sided. Clipped points are kept
            #::: aside under ``raw_clipped_*`` so initial_guess plots can
            #::: surface them in red without affecting the fit.
            _fmin = self.settings.get('flux_min_raw')
            _fmax = self.settings.get('flux_max_raw')
            _clipped_time = np.empty(0, dtype=float)
            _clipped_flux = np.empty(0, dtype=float)
            _clipped_flux_err = np.empty(0, dtype=float)
            if _fmin is not None or _fmax is not None:
                _mask = np.ones_like(flux, dtype=bool)
                if _fmin is not None:
                    _mask &= (flux >= _fmin)
                if _fmax is not None:
                    _mask &= (flux <= _fmax)
                _n_drop = int(np.sum(~_mask))
                if _n_drop > 0:
                    warnings.warn(
                        '%d/%d rows in "%s.csv" dropped by flux_min_raw=%s, flux_max_raw=%s.'
                        % (_n_drop, len(flux), inst, _fmin, _fmax)
                    )
                if not np.any(_mask):
                    raise ValueError(
                        'All rows in "'+inst+'.csv" were removed by flux_min_raw/flux_max_raw. '
                        'Check that the bounds bracket your normalized flux level.'
                    )
                _clipped_time = time[~_mask]
                _clipped_flux = flux[~_mask]
                _clipped_flux_err = flux_err[~_mask]
                time = time[_mask]
                flux = flux[_mask]
                flux_err = flux_err[_mask]
                custom_series = custom_series[_mask]

            self.fulldata[inst] = {
                          'time':time,
                          'flux':flux,
                          'err_scales_flux':flux_err/np.nanmean(flux_err),
                          'custom_series':custom_series,
                          'raw_clipped_time':_clipped_time,
                          'raw_clipped_flux':_clipped_flux,
                          'raw_clipped_flux_err':_clipped_flux_err,
                         }
            if (self.settings['fast_fit']) and (len(self.settings['inst_phot'])>0):
                time, flux, flux_err, custom_series = self.reduce_phot_data(time, flux, flux_err, custom_series=custom_series, inst=inst)
            self.data[inst] = {
                          'time':time,
                          'flux':flux,
                          'err_scales_flux':flux_err/np.nanmean(flux_err),
                          'custom_series':custom_series,
                          'raw_clipped_time':_clipped_time,
                          'raw_clipped_flux':_clipped_flux,
                          'raw_clipped_flux_err':_clipped_flux_err,
                         }

        #======================================================================
        #::: detect duplicate inst_phot input files (e.g. user accidentally
        #::: copied qlp1800.csv to qlp600.csv). Pairwise compare the
        #::: pre-fast-fit `fulldata` time+flux arrays — fulldata is the raw
        #::: read-in data, so identical content here is unambiguous.
        #======================================================================
        _inst_phot_list = list(self.settings['inst_phot'])
        for _i in range(len(_inst_phot_list)):
            for _j in range(_i + 1, len(_inst_phot_list)):
                _a_inst, _b_inst = _inst_phot_list[_i], _inst_phot_list[_j]
                _a, _b = self.fulldata[_a_inst], self.fulldata[_b_inst]
                if (len(_a['time']) == len(_b['time'])
                        and np.array_equal(_a['time'], _b['time'])
                        and np.array_equal(_a['flux'], _b['flux'])):
                    raise ValueError(
                        '"%s.csv" and "%s.csv" contain identical data '
                        '(N=%d, time=[%.4f, %.4f]). Likely a duplicated file. '
                        'Please verify each inst_phot points to a distinct '
                        'lightcurve, or remove the duplicate from inst_phot.'
                        % (_a_inst, _b_inst, len(_a['time']),
                           float(_a['time'][0]), float(_a['time'][-1]))
                    )

        #======================================================================
        #::: RV
        #======================================================================
        for inst in self.settings['inst_rv']:
            try:
                time, rv, rv_err, custom_series = np.genfromtxt( os.path.join(self.datadir,inst+'.csv'), delimiter=',', dtype=float, unpack=True)[0:4]       
            except:
                time, rv, rv_err = np.genfromtxt( os.path.join(self.datadir,inst+'.csv'), delimiter=',', dtype=float, unpack=True)[0:3]              
                custom_series = np.zeros_like(time)
            if any(np.isnan(time*rv*rv_err*custom_series)):
                raise ValueError('There are NaN values in "'+inst+'.csv". Please make sure everything is fine with your data, then exclude these rows from the file and restart.')
            #aCkTuaLLLyy rv_err=0 is ok, since we add a jitter term here anyway (instead of scaling)
            # if any(rv_err==0):
            #     raise ValueError('There are uncertainties with values of 0 in "'+inst+'.csv". Please make sure everything is fine with your data, then exclude these rows from the file and restart.')
            if any(rv_err<0):
                raise ValueError('There are uncertainties with negative values in "'+inst+'.csv". Please make sure everything is fine with your data, then exclude these rows from the file and restart.')
            if not all(np.diff(time)>0):
                raise ValueError('Your time array in "'+inst+'.csv" is not sorted. You will want to check that...')
            self.data[inst] = {
                          'time':time,
                          'rv':rv,
                          'white_noise_rv':rv_err,
                          'custom_series':custom_series
                         }
            
        #======================================================================
        #::: RV2 (for detached binaries)
        #======================================================================
        for inst in self.settings['inst_rv2']:
            try:
                time, rv, rv_err, custom_series = np.genfromtxt( os.path.join(self.datadir,inst+'.csv'), delimiter=',', dtype=float, unpack=True)[0:4]       
            except:
                time, rv, rv_err = np.genfromtxt( os.path.join(self.datadir,inst+'.csv'), delimiter=',', dtype=float, unpack=True)[0:3]              
                custom_series = np.zeros_like(time)
            if not all(np.diff(time)>0):
                raise ValueError('Your time array in "'+inst+'.csv" is not sorted. You will want to check that...')
            self.data[inst] = {
                          'time':time,
                          'rv2':rv,
                          'white_noise_rv2':rv_err,
                          'custom_series':custom_series
                         }
        
        #======================================================================
        #::: also save the combined time series
        #::: for cases where all instruments are treated together
        #::: e.g. for stellar variability GPs
        #======================================================================
        self.data['inst_phot'] = {'time':[],'flux':[],'flux_err':[],'inst':[]}
        for inst in self.settings['inst_phot']:
            self.data['inst_phot']['time'] += list(self.data[inst]['time'])
            self.data['inst_phot']['flux'] += list(self.data[inst]['flux'])
            self.data['inst_phot']['flux_err'] += [inst]*len(self.data[inst]['time']) #errors will be sampled/derived later
            self.data['inst_phot']['inst'] += [inst]*len(self.data[inst]['time'])
        ind_sort = np.argsort(self.data['inst_phot']['time'])
        self.data['inst_phot']['ind_sort'] = ind_sort
        self.data['inst_phot']['time'] = np.array(self.data['inst_phot']['time'])[ind_sort]
        self.data['inst_phot']['flux'] = np.array(self.data['inst_phot']['flux'])[ind_sort]
        self.data['inst_phot']['flux_err'] = np.array(self.data['inst_phot']['flux_err'])[ind_sort]      
        self.data['inst_phot']['inst'] = np.array(self.data['inst_phot']['inst'])[ind_sort]
    
        self.data['inst_rv'] = {'time':[],'rv':[],'rv_err':[],'inst':[]}
        for inst in self.settings['inst_rv']:
            self.data['inst_rv']['time'] += list(self.data[inst]['time'])
            self.data['inst_rv']['rv'] += list(self.data[inst]['rv'])
            self.data['inst_rv']['rv_err'] += list(np.nan*self.data[inst]['rv']) #errors will be sampled/derived later
            self.data['inst_rv']['inst'] += [inst]*len(self.data[inst]['time'])
        ind_sort = np.argsort(self.data['inst_rv']['time'])
        self.data['inst_rv']['ind_sort'] = ind_sort
        self.data['inst_rv']['time'] = np.array(self.data['inst_rv']['time'])[ind_sort]
        self.data['inst_rv']['rv'] = np.array(self.data['inst_rv']['rv'])[ind_sort]
        self.data['inst_rv']['rv_err'] = np.array(self.data['inst_rv']['rv_err'])[ind_sort]   
        self.data['inst_rv']['inst'] = np.array(self.data['inst_rv']['inst'])[ind_sort]
    
        self.data['inst_rv2'] = {'time':[],'rv2':[],'rv2_err':[],'inst':[]}
        for inst in self.settings['inst_rv2']:
            self.data['inst_rv2']['time'] += list(self.data[inst]['time'])
            self.data['inst_rv2']['rv2'] += list(self.data[inst]['rv2'])
            self.data['inst_rv2']['rv2_err'] += list(np.nan*self.data[inst]['rv2']) #errors will be sampled/derived later
            self.data['inst_rv2']['inst'] += [inst]*len(self.data[inst]['time'])
        ind_sort = np.argsort(self.data['inst_rv2']['time'])
        self.data['inst_rv2']['ind_sort'] = ind_sort
        self.data['inst_rv2']['time'] = np.array(self.data['inst_rv2']['time'])[ind_sort]
        self.data['inst_rv2']['rv2'] = np.array(self.data['inst_rv2']['rv2'])[ind_sort]
        self.data['inst_rv2']['rv2_err'] = np.array(self.data['inst_rv2']['rv2_err'])[ind_sort]   
        self.data['inst_rv2']['inst'] = np.array(self.data['inst_rv2']['inst'])[ind_sort]

        
            
    ###############################################################################
    #::: change epoch
    ###############################################################################

    def my_truncnorm_isf(q,a,b,mean,std):
        a_scipy = 1.*(a - mean) / std
        b_scipy = 1.*(b - mean) / std
        return truncnorm.isf(q,a_scipy,b_scipy,loc=mean,scale=std)


    def change_epoch(self):
        '''
        change epoch entry from params.csv to set epoch into the middle of the range
        '''
        
        self.logprint('\nShifting epochs into the data center')
        self.logprint('------------------------------------')
        # Echo the datadir so logs from multiple concurrent fits are
        # easy to attribute when grepped or tailed.
        self.logprint('datadir: ' + os.path.abspath(self.datadir))
        
        #::: for all companions
        for companion in self.settings['companions_all']:
            
            self.logprint('Companion',companion)
            self.logprint('\tinput epoch:',self.params[companion+'_epoch'])
            
            #::: get data time range
            alldata = []
            for inst in self.settings['inst_for_'+companion+'_epoch']:
                alldata += list(self.data[inst]['time'])
            start = np.nanmin( alldata )
            end = np.nanmax( alldata )
            
            #::: get the given values
            user_epoch  = 1.*self.params[companion+'_epoch']
            period      = 1.*self.params[companion+'_period']
#            buf = self.bounds[ind_e].copy()
                
            #::: calculate the true first_epoch
            if 'fast_fit_width' in self.settings and self.settings['fast_fit_width'] is not None:
                width = self.settings['fast_fit_width']
            else:
                width = 0
            first_epoch = get_first_epoch(alldata, self.params[companion+'_epoch'], self.params[companion+'_period'], width=width)
            
            #::: calculate the mid_epoch (in the middle of the data set)
            N = int(np.round((end-start)/2./period))
            self.settings['mid_epoch'] = first_epoch + N * period
            
            #::: calculate how much the user_epoch has to be shifted to get the mid_epoch
            N_shift = int(np.round((self.settings['mid_epoch']-user_epoch)/period))
            
            #::: set the new initial guess (and truth)
            self.params[companion+'_epoch'] = 1.*self.settings['mid_epoch']
           
            #::: also shift the truth (implies that the turth epoch is set where the initial guess is)
            try:
                ind_e = np.where(self.fitkeys==companion+'_epoch')[0][0]
                ind_p = np.where(self.fitkeys==companion+'_period')[0][0]
                N_truth_shift = int(np.round((self.settings['mid_epoch']-self.fittruths[ind_e])/self.fittruths[ind_p]))
                self.fittruths[ind_e] += N_truth_shift * self.fittruths[ind_p]
            except:
                pass
            
            #::: if a fit param, also update the bounds accordingly
            if (N_shift != 0) and (companion+'_epoch' in self.fitkeys):
                ind_e = np.where(self.fitkeys==companion+'_epoch')[0][0]
                ind_p = np.where(self.fitkeys==companion+'_period')[0][0]
                
#                print('\n')
#                print('############################################################################')
#                print('user_epoch', user_epoch, self.bounds[ind_e])
#                print('user_period', period, self.bounds[ind_p])
#                print('----------------------------------------------------------------------------')
                  
                #::: set the new initial guess
                self.theta_0[ind_e] = 1.*self.settings['mid_epoch']
                
                #::: get the bounds / errors
                #::: if the epoch and period priors are both uniform
                if (self.bounds[ind_e][0] == 'uniform') & (self.bounds[ind_p][0] == 'uniform'):
                    if N_shift > 0:
                        self.bounds[ind_e][1] = self.bounds[ind_e][1] + N_shift * self.bounds[ind_p][1] #lower bound
                        self.bounds[ind_e][2] = self.bounds[ind_e][2] + N_shift * self.bounds[ind_p][2] #upper bound
                    elif N_shift < 0:
                        self.bounds[ind_e][1] = self.bounds[ind_e][1] + N_shift * self.bounds[ind_p][2] #lower bound; period bounds switched if N_shift is negative
                        self.bounds[ind_e][2] = self.bounds[ind_e][2] + N_shift * self.bounds[ind_p][1] #upper bound; period bounds switched if N_shift is negative
                
                #::: if the epoch and period priors are both normal
                elif (self.bounds[ind_e][0] == 'normal') & (self.bounds[ind_p][0] == 'normal'):
                    self.bounds[ind_e][1] = self.bounds[ind_e][1] + N_shift * self.bounds[ind_p][1] #mean (in case the prior-mean is not the initial-guess-mean)
                    self.bounds[ind_e][2] = np.sqrt( self.bounds[ind_e][2]**2 + N_shift**2 * self.bounds[ind_p][2]**2 ) #std (in case the prior-mean is not the initial-guess-mean)
                                        
                #::: if the epoch and period priors are both trunc_normal
                elif (self.bounds[ind_e][0] == 'trunc_normal') & (self.bounds[ind_p][0] == 'trunc_normal'):
                    if N_shift > 0:
                        self.bounds[ind_e][1] = self.bounds[ind_e][1] + N_shift * self.bounds[ind_p][1] #lower bound
                        self.bounds[ind_e][2] = self.bounds[ind_e][2] + N_shift * self.bounds[ind_p][2] #upper bound
                    elif N_shift < 0:
                        self.bounds[ind_e][1] = self.bounds[ind_e][1] + N_shift * self.bounds[ind_p][2] #lower bound; period bounds switched if N_shift is negative
                        self.bounds[ind_e][2] = self.bounds[ind_e][2] + N_shift * self.bounds[ind_p][1] #upper bound; period bounds switched if N_shift is negative
                    self.bounds[ind_e][3] = self.bounds[ind_e][3] + N_shift * self.bounds[ind_p][3] #mean (in case the prior-mean is not the initial-guess-mean)
                    self.bounds[ind_e][4] = np.sqrt( self.bounds[ind_e][4]**2 + N_shift**2 * self.bounds[ind_p][4]**2 ) #std (in case the prior-mean is not the initial-guess-mean)
            
                #::: if the epoch prior is uniform and period prior is normal
                elif (self.bounds[ind_e][0] == 'uniform') & (self.bounds[ind_p][0] == 'normal'):
                    self.bounds[ind_e][1] = self.bounds[ind_e][1] + N_shift * (period + self.bounds[ind_p][2]) #lower bound epoch + Nshift * period + Nshift * std_period
                    self.bounds[ind_e][2] = self.bounds[ind_e][2] + N_shift * (period + self.bounds[ind_p][2]) #upper bound + Nshift * period + Nshift * std_period

                #::: if the epoch prior is uniform and period prior is trunc_normal
                elif (self.bounds[ind_e][0] == 'uniform') & (self.bounds[ind_p][0] == 'trunc_normal'):
                    self.bounds[ind_e][1] = self.bounds[ind_e][1] + N_shift * (period + self.bounds[ind_p][4]) #lower bound epoch + Nshift * period + Nshift * std_period
                    self.bounds[ind_e][2] = self.bounds[ind_e][2] + N_shift * (period + self.bounds[ind_p][4]) #upper bound + Nshift * period + Nshift * std_period

                elif (self.bounds[ind_e][0] == 'normal') & (self.bounds[ind_p][0] == 'uniform'):
                    raise ValueError('shift_epoch with different priors for epoch and period is not yet implemented.')
                    
                elif (self.bounds[ind_e][0] == 'normal') & (self.bounds[ind_p][0] == 'trunc_normal'):
                    raise ValueError('shift_epoch with different priors for epoch and period is not yet implemented.')
                    
                elif (self.bounds[ind_e][0] == 'trunc_normal') & (self.bounds[ind_p][0] == 'uniform'):
                    raise ValueError('shift_epoch with different priors for epoch and period is not yet implemented.')
                    
                elif (self.bounds[ind_e][0] == 'trunc_normal') & (self.bounds[ind_p][0] == 'normal'):
                    raise ValueError('shift_epoch with different priors for epoch and period is not yet implemented.')
                    
                else:
                    raise ValueError('Parameters "bounds" have to be "uniform", "normal" or "trunc_normal".')
                    
        
                self.logprint('\tshifted epoch:',self.params[companion+'_epoch'])
                self.logprint('\tshifted by',N_shift,'periods')
                


    ###############################################################################
    #::: reduce_phot_data
    ###############################################################################
    def reduce_phot_data(self, time, flux, flux_err, custom_series=None, inst=None):
        ind_in = []
              
        for companion in self.settings['companions_phot']:
            epoch  = self.params[companion+'_epoch']
            period = self.params[companion+'_period']
            width  = self.settings['fast_fit_width']
            if self.settings['secondary_eclipse']:
                ind_ecl1x, ind_ecl2x, ind_outx = index_eclipses(time,epoch,period,width,width) #TODO: currently this assumes width_occ == width_tra
                ind_in += list(ind_ecl1x)
                ind_in += list(ind_ecl2x)
                self.fulldata[inst][companion+'_ind_ecl1'] = ind_ecl1x
                self.fulldata[inst][companion+'_ind_ecl2'] = ind_ecl2x
                self.fulldata[inst][companion+'_ind_out'] = ind_outx
            else:
                ind_inx, ind_outx = index_transits(time,epoch,period,width)
                ind_in += list(ind_inx)
                self.fulldata[inst][companion+'_ind_in'] = ind_inx
                self.fulldata[inst][companion+'_ind_out'] = ind_outx
                
        ind_in = np.sort(np.unique(ind_in))
        self.fulldata[inst]['all_ind_in'] = ind_in
        self.fulldata[inst]['all_ind_out'] = np.delete( np.arange(len(self.fulldata[inst]['time'])), ind_in )
        
        if len(ind_in)==0:
            raise ValueError(inst+'.csv does not contain any in-transit data. Check that your epoch and period guess are correct.')
        
        time = time[ind_in]
        flux = flux[ind_in]
        flux_err = flux_err[ind_in]
        if custom_series is None: 
            return time, flux, flux_err
        else:
            custom_series = custom_series[ind_in]
            return time, flux, flux_err, custom_series
    
    
    
    ###############################################################################
    #::: setup TTV fit (if chosen)
    ###############################################################################
    def setup_ttv_fit(self):
        '''
        this must be run *after* reduce_phot_data()
        '''
        
        #::: the window we choose to look for transits is determined by fast_fit_width
        window = self.settings['fast_fit_width']
        
        #::: for each companion, stitch together all the time stamps observed by all photometric instruments
        #::: and check which of these times overlap with a potential transit window (determined by fast_fit_width)
        for companion in self.settings['companions_phot']:
            times_combined = []
            for inst in self.settings['inst_phot']:
                times_combined += list(self.data[inst]['time'])
            times_combined = np.sort(times_combined)
            
            self.data[companion+'_tmid_observed_transits'] = get_tmid_observed_transits(times_combined,
                                                                                        self.params[companion+'_epoch'],
                                                                                        self.params[companion+'_period'],
                                                                                        window)
            
            for inst in self.settings['inst_phot']:
                time = self.data[inst]['time']
                for i, t in enumerate(self.data[companion+'_tmid_observed_transits']):
                    ind = np.where((time >= (t - window/2.)) & (time <= (t + window/2.)))[0]
                    self.data[inst][companion+'_ind_time_transit_'+str(i+1)] = ind
                    self.data[inst][companion+'_time_transit_'+str(i+1)] = time[ind]
                    

            #::: THE FOLLOWING PART MOVED INTO THE SEPARATE SCRIPT "PREPARE_TTV_FIT.PY"
            #::: plots
            # if self.settings['fit_ttvs']:  
            #     flux_min = np.nanmin(all_flux)
            #     flux_max = np.nanmax(all_flux)
            #     N_days = int( np.max(all_times) - np.min(all_times) )
            #     figsizex = np.min( [1, int(N_days/20.)] )*5
            #     fig, ax = plt.subplots(figsize=(figsizex, 4)) #figsize * 5 for every 20 days
            #     for inst in self.settings['inst_phot']:
            #         ax.plot(self.data[inst]['time'], self.data[inst]['flux'],ls='none',marker='.',label=inst)
            #     ax.plot( self.data[companion+'_tmid_observed_transits'], np.ones_like(self.data[companion+'_tmid_observed_transits'])*0.995*flux_min, 'k^' )
            #     for i, tmid in enumerate(self.data[companion+'_tmid_observed_transits']):
            #         ax.text( tmid, 0.9925*flux_min, str(i+1), ha='center' )  
            #     ax.set(ylim=[0.99*flux_min, flux_max], xlabel='Time (BJD)', ylabel='Relative Flux') 
            #     if not os.path.exists( os.path.join(self.datadir,'results') ):
            #         os.makedirs(os.path.join(self.datadir,'results'))
            #     ax.legend()
            #     fname = os.path.join(self.datadir,'results','preparation_for_TTV_fit_'+companion+'.pdf')
            #     if os.path.exists(fname):
            #         overwrite = str(input('Figure "preparation_for_TTV_fit_'+companion+'.pdf" already exists.\n'+\
            #                               'What do you want to do?\n'+\
            #                               '1 : overwrite it\n'+\
            #                               '2 : skip it and move on\n'))
            #         if (overwrite == '1'):
            #             fig.savefig(fname, bbox_inches='tight' )    
            #         else:
            #             pass        
            #     plt.close(fig)
            
                
                
    ###############################################################################
    #::: apply flattened-flux outlier clip
    ###############################################################################
    def apply_flat_clip(self):
        '''
        Drops rows from each photometric instrument whose detrended flux falls
        outside [flux_min_flat, flux_max_flat]. The trend is computed from the
        initial-guess parameters using the same primitives that
        `show_initial_guess` plots:
            flat = flux - calculate_baseline(...) - calculate_stellar_var(...)
        Must be called AFTER config.BASEMENT has been assigned, because
        calculate_baseline/calculate_stellar_var read config.BASEMENT.{settings,data}.
        Re-runs setup_ttv_fit() if fit_ttvs is enabled, since per-transit
        index arrays depend on the data length.
        '''
        fmin = self.settings.get('flux_min_flat')
        fmax = self.settings.get('flux_max_flat')
        if fmin is None and fmax is None:
            return

        # Local imports to avoid a circular import at module load time
        # (computer.py imports config which imports basement).
        from . import config as _config
        from .computer import (
            calculate_model, calculate_baseline, calculate_stellar_var,
        )

        if _config.BASEMENT is not self:
            # Defensive: this method needs the global BASEMENT to point at us
            # so the calculate_* helpers see the right data/settings.
            warnings.warn(
                'apply_flat_clip(): config.BASEMENT is not this Basement; '
                'skipping flat clip.'
            )
            return

        for inst in self.settings['inst_phot']:
            try:
                model = calculate_model(self.params, inst, 'flux')
                baseline = calculate_baseline(self.params, inst, 'flux', model=model)
                stellar_var = calculate_stellar_var(
                    self.params, inst, 'flux', model=model, baseline=baseline,
                )
            except Exception as e:
                warnings.warn(
                    'apply_flat_clip(): could not compute trend for "%s" '
                    '(%s); skipping flat clip for this instrument.' % (inst, e)
                )
                continue

            flux = self.data[inst]['flux']
            flat = flux - baseline - stellar_var

            mask = np.ones_like(flat, dtype=bool)
            if fmin is not None:
                mask &= (flat >= fmin)
            if fmax is not None:
                mask &= (flat <= fmax)

            n_drop = int(np.sum(~mask))
            if n_drop == 0:
                continue
            if not np.any(mask):
                raise ValueError(
                    'All rows in "'+inst+'" were removed by flux_min_flat/flux_max_flat. '
                    'Check the bounds against your detrended flux level.'
                )
            warnings.warn(
                '%d/%d rows in "%s" dropped by flux_min_flat=%s, flux_max_flat=%s '
                '(applied to flux - baseline - stellar_var from initial guess).'
                % (n_drop, len(flat), inst, fmin, fmax)
            )

            for k in ('time', 'flux', 'err_scales_flux', 'custom_series'):
                if k in self.data[inst]:
                    self.data[inst][k] = self.data[inst][k][mask]

        if self.settings.get('fit_ttvs'):
            try:
                self.setup_ttv_fit()
            except Exception as e:
                warnings.warn(
                    'apply_flat_clip(): setup_ttv_fit() failed after clip (%s).' % e
                )



    ###############################################################################
    #::: stellar priors
    ###############################################################################
    def load_stellar_priors(self, N_samples=10000):
        if os.path.exists(os.path.join(self.datadir,'params_star.csv')) and (self.settings['use_host_density_prior'] is True):
            buf = np.genfromtxt( os.path.join(self.datadir,'params_star.csv'), delimiter=',', names=True, dtype=None, encoding='utf-8', comments='#' )
            radius = simulate_PDF(buf['R_star'], buf['R_star_lerr'], buf['R_star_uerr'], size=N_samples, plot=False) * 6.957e10 #in cgs
            mass = simulate_PDF(buf['M_star'], buf['M_star_lerr'], buf['M_star_uerr'], size=N_samples, plot=False) * 1.9884754153381438e+33 #in cgs
            volume = (4./3.)*np.pi*radius**3 #in cgs
            density = mass / volume #in cgs
            self.params_star = {'R_star_median':buf['R_star'],
                                'R_star_lerr':buf['R_star_lerr'],
                                'R_star_uerr':buf['R_star_uerr'],
                                'M_star_median':buf['M_star'],
                                'M_star_lerr':buf['M_star_lerr'],
                                'M_star_uerr':buf['M_star_uerr']
                                }
            self.external_priors['host_density'] = ['normal', np.median(density), np.max( [np.median(density)-np.percentile(density,16), np.percentile(density,84)-np.median(density)] ) ] #in cgs
            
            