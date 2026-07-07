#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model computation engine for allesfitter.

This module provides the core functions for computing astrophysical models
including transit light curves, radial velocity curves, baseline models,
and stellar variability. It interfaces with ellc (exoplanet light curve)
and celerite for Gaussian Process modeling.

Functions:
    calculate_model: Compute the full astrophysical model (transit + RV + baseline).
    calculate_baseline: Compute the baseline model only.
    calculate_stellar_var: Compute stellar variability (spots, flares, etc.).
    calculate_yerr_w: Calculate weighted error bars including jitter.
    update_params: Update parameter dictionary with phased values.

Module-Level Constants:
    GPs: List of supported Gaussian Process kernel types.
    FCTs: List of supported baseline function types.
"""

from __future__ import print_function, division, absolute_import

#::: plotting settings
import seaborn as sns
sns.set(context='paper', style='ticks', palette='deep', font='sans-serif', font_scale=1.5, color_codes=True)
sns.set_style({"xtick.direction": "in","ytick.direction": "in"})
sns.set_context(rc={'lines.markeredgewidth': 1})

#::: modules
import numpy as np
import ellc
from astropy import units as u
from scipy.optimize import minimize
from scipy.interpolate import UnivariateSpline
import numpy.polynomial.polynomial as poly
import warnings
warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning) 
warnings.filterwarnings('ignore', category=np.RankWarning) 
warnings.filterwarnings('ignore', category=RuntimeWarning) 

#::: for now, only use the original celerite
try:
    import celerite
    from celerite import terms
    celerite_version = 1
except ImportError:
    warnings.warn("Cannot import package 'celerite', thus GP baseline models will not be supported.")

#TODO: handling two celerite versions
# celerite_version = 0 #global. nice.
# try:
#     import celerite2
#     from celerite2 import terms
#     celerite_version = 2
# except ImportError:
#     warnings.warn("Cannot import package 'celerite2', importing package 'celerite' instead.")
#     try:
#         import celerite
#         from celerite import terms
#         celerite_version = 1
#     except ImportError:
#         warnings.warn("Cannot import package 'celerite2' nor 'celerite', thus GP baseline models will not be supported.")

#allesfitter modules
from . import config
import difflib as _difflib
from .basement import BASELINE_GP_HYPER_PREFIXES
# from .limb_darkening import LDC3
from .flares.aflare import aflare1
from .bumps.bump import bump_model
# from .exoworlds_rdx.lightcurves.lightcurve_tools import calc_phase
from .lightcurves import translate_limb_darkening_from_q_to_u as q_to_u
# from .lightcurves import translate_limb_darkening_from_u_to_q as u_to_q
from .observables import calc_M_comp_from_RV, calc_rho, calc_rho_host


def _calc_light_3(dil):
    '''Safely calculate light_3 (dilution factor) avoiding division by zero.'''
    if dil is None or dil >= 1.0:
        return 0.0
    return dil / (1. - dil)



'''
README

for dilution:
    our definition of dilution is D_0 = 1 - F_source / (F_source + F_blend)
    ellc's original definition of third light is light_3 = F_blend / F_source
    therefore we can relate them as light_3 = D_0 / (1 - D_0)
    
    on another note, the TESS SPOC lightcurve parameter CROWDSAP = F_source / (F_source + F_blend) d
    hence D_0 = 1 - CROWDSAP
'''




###############################################################################
#::: possible functions
###############################################################################  
GPs = ['sample_GP_real', 'sample_GP_complex', 'sample_GP_Matern32', 'sample_GP_SHO']
FCTs = ['none', 'offset', 'linear', 'hybrid_offset', 'hybrid_poly_0', 'hybrid_poly_1', 'hybrid_poly_2', 'hybrid_poly_3', 'hybrid_poly_4', 'hybrid_poly_5', 'hybrid_poly_6', 'hybrid_spline', 'hybrid_spline_s', 'sample_offset', 'sample_linear', 'sample_linear_multi', 'hybrid_linear_multi']



###############################################################################
#::: divider that catches 'None' types
###############################################################################  
def divide(a,b):
    if a is not None:
        return 1.*a/b
    else:
        return None
    
    

###############################################################################
#::: convert input params into ellc params
###############################################################################  
def update_params(theta):
    
    params = config.BASEMENT.params.copy()
    
    
    #=========================================================================
    #::: first, sync over from theta
    #=========================================================================
    for i, key in enumerate(config.BASEMENT.fitkeys):
        params[key] = theta[i]   
    
    
    #=========================================================================
    #::: second, deal with coupled params before updates
    #=========================================================================
    for i, key in enumerate(config.BASEMENT.allkeys):
        if isinstance(config.BASEMENT.coupled_with[i], str) and (len(config.BASEMENT.coupled_with[i])>0):
            params[key] = params[config.BASEMENT.coupled_with[i]]


    #=========================================================================
    #::: baseline share groups: alias follower GP hypers to leader
    #::: Run AFTER coupled_with so any user-declared follower row that was
    #::: explicitly coupled to the leader is overwritten with the same value
    #::: (no-op), and any synthesized follower key (not in allkeys) gets the
    #::: leader's *current* value for this theta draw.
    #=========================================================================
    for _k_share in ('flux', 'rv', 'rv2'):
        _followers_of = config.BASEMENT.settings.get(
            'baseline_share_'+_k_share+'_followers_of', {})
        if not _followers_of:
            continue
        for _leader, _followers in _followers_of.items():
            for _prefix in BASELINE_GP_HYPER_PREFIXES:
                _lk = _prefix + _k_share + '_' + _leader
                if _lk not in params:
                    continue
                for _f in _followers:
                    params[_prefix + _k_share + '_' + _f] = params[_lk]


    #=========================================================================
    #::: inclination, per companion
    #=========================================================================
    for companion in config.BASEMENT.settings['companions_all']:
        try:
            params[companion+'_incl'] = np.arccos( params[companion+'_cosi'] )/np.pi*180.
        except (TypeError, KeyError, ValueError):
            params[companion+'_incl'] = None
        
       
    #=========================================================================
    #::: radii, per companion (radius_1 from rsuma, radius_2 from rr)
    #::: In achromatic mode radius_1 is a single shared value.
    #::: In chromatic mode rsuma is achromatic but rr varies per bandpass,
    #::: so rsuma = radius_1 + radius_2 cannot hold simultaneously across
    #::: bands with one shared radius_1. We store a reference value here
    #::: (from the achromatic base rr if present, else any bandpass rr) for
    #::: callers that need a scalar (eclipse-geometry checks, phase-curve
    #::: hacks). flux_subfct_ellc recomputes radius_1/radius_2 per-band so
    #::: each bandpass's transit model is self-consistent.
    #=========================================================================
    for companion in config.BASEMENT.settings['companions_all']:
        rr = params.get(companion+'_rr')
        if rr is None:
            # Fall back to the bandpass-suffixed key. Covers both:
            #   * true chromatic (>=2 unique bandpasses): rr lives under
            #     `{c}_rr_<bandpass>`; the achromatic `{c}_rr` is never
            #     written by the parser.
            #   * single-band chromatic infrastructure (bandpass set with
            #     one label → chromatic=False): rr still lives under
            #     `{c}_rr_<bandpass>`, NOT `{c}_rr`, because the parser
            #     uses the bandpass suffix as soon as the bandpass row is
            #     present. Without this fallback, b_radius_1 and the b_rr
            #     legacy alias would silently be None in that mode.
            # For true achromatic (no bandpass row), get_rr_key returns
            # `{c}_rr` and this lookup is a no-op identical to the first.
            inst_phot = config.BASEMENT.settings.get('inst_phot', [])
            if inst_phot:
                rr_key = config.BASEMENT.get_rr_key(companion, inst_phot[0])
                rr = params.get(rr_key)
        # Backfill the achromatic key so legacy code paths that read
        # `params[c+'_rr']` directly (host-density prior, host-density-only
        # branch, phase-curve hack at ~805) still work in chromatic mode.
        # flux_subfct_ellc keeps using get_rr_key first, so the per-band
        # value still wins where it matters; this is only a fallback scalar.
        if (companion+'_rr') not in params or params[companion+'_rr'] is None:
            params[companion+'_rr'] = rr
        try:
            params[companion+'_radius_1'] = params[companion+'_rsuma'] / (1. + rr)
        except (TypeError, KeyError, ZeroDivisionError):
            params[companion+'_radius_1'] = None
                
                
    #=========================================================================
    #::: limb darkening, per instrument (per-bandpass in chromatic mode)
    #=========================================================================
    for inst in config.BASEMENT.settings['inst_all']:
        for obj in ['host']+config.BASEMENT.settings['companions_all']:
        
            # LDC keys are bandpass-only when settings.csv carries a
            # bandpass row, regardless of the chromatic flag — limb
            # darkening depends on wavelength, not on the rr-naming
            # convention. Use get_ldc_bandpass (not get_bandpass) so the
            # chromatic,False override does not collapse LDC keys back
            # to the inst name.
            bandpass = config.BASEMENT.get_ldc_bandpass(inst)
            if bandpass:
                ldc_suffix = '_' + bandpass
            else:
                ldc_suffix = '_' + inst
            
            #::: if we sampled in q-space, convert the params to u-space for ellc
            if config.BASEMENT.settings[obj+'_ld_space_'+inst] == 'q': 
                
                if config.BASEMENT.settings[obj+'_ld_law_'+inst] is None:
                    params[obj+'_ldc_'+inst] = None
                    
                elif config.BASEMENT.settings[obj+'_ld_law_'+inst] == 'lin':
                    params[obj+'_ldc_'+inst] = params[obj+'_ldc_q1'+ldc_suffix]
                    
                elif config.BASEMENT.settings[obj+'_ld_law_'+inst] == 'quad':
                    _q1k = obj+'_ldc_q1'+ldc_suffix
                    _q2k = obj+'_ldc_q2'+ldc_suffix
                    if params.get(_q1k) is None or params.get(_q2k) is None:
                        raise ValueError(
                            "Limb-darkening coefficients '%s' and/or '%s' are "
                            "missing from params.csv (got None) but settings.csv "
                            "declares %s_ld_law_%s='quad'. Add the q1/q2 rows "
                            "to params.csv, or set %s_ld_law_%s=None to disable "
                            "limb darkening for this instrument."
                            % (_q1k, _q2k, obj, inst, obj, inst)
                        )
                    params[obj+'_ldc_'+inst] = q_to_u([params[_q1k], params[_q2k]],
                                                      law='quad')

                elif config.BASEMENT.settings[obj+'_ld_law_'+inst] == 'sing':
                    _q1k = obj+'_ldc_q1'+ldc_suffix
                    _q2k = obj+'_ldc_q2'+ldc_suffix
                    _q3k = obj+'_ldc_q3'+ldc_suffix
                    if (params.get(_q1k) is None or params.get(_q2k) is None
                            or params.get(_q3k) is None):
                        raise ValueError(
                            "Limb-darkening coefficients '%s', '%s', '%s' are "
                            "missing from params.csv (got None) but settings.csv "
                            "declares %s_ld_law_%s='sing'."
                            % (_q1k, _q2k, _q3k, obj, inst)
                        )
                    params[obj+'_ldc_'+inst] = q_to_u([params[_q1k],
                                                       params[_q2k],
                                                       params[_q3k]],
                                                      law='sing')
        
                else:
                    raise ValueError("You are sampling the limb darkening in q-space,"+\
                                     "where only the options 'none', 'lin', 'quad' and 'sing'"+\
                                     "are supported. However, your input was:"+\
                                     config.BASEMENT.settings[obj+'_ld_law_'+inst]+".")
    
    
            #::: if we sampled in u-space, just stack them into a list for ellc
            elif config.BASEMENT.settings[obj+'_ld_space_'+inst] == 'u':
                
                if config.BASEMENT.settings[obj+'_ld_law_'+inst] is None:
                    params[obj+'_ldc_'+inst] = None
                    
                elif config.BASEMENT.settings[obj+'_ld_law_'+inst] == 'lin':
                    params[obj+'_ldc_'+inst] = params[obj+'_ldc_u1'+ldc_suffix]
                    
                elif config.BASEMENT.settings[obj+'_ld_law_'+inst] in ('quad','sqrt','exp','log'):
                    params[obj+'_ldc_'+inst] = [ params[obj+'_ldc_u1'+ldc_suffix], 
                                                 params[obj+'_ldc_u2'+ldc_suffix] ]
                    
                elif config.BASEMENT.settings[obj+'_ld_law_'+inst] == 'sing':
                    params[obj+'_ldc_'+inst] = [ params[obj+'_ldc_u1'+ldc_suffix], 
                                                 params[obj+'_ldc_u2'+ldc_suffix], 
                                                 params[obj+'_ldc_u3'+ldc_suffix] ]
                    
                elif config.BASEMENT.settings[obj+'_ld_law_'+inst] == 'claret':
                    params[obj+'_ldc_'+inst] = [ params[obj+'_ldc_u1'+ldc_suffix], 
                                                 params[obj+'_ldc_u2'+ldc_suffix], 
                                                 params[obj+'_ldc_u3'+ldc_suffix], 
                                                 params[obj+'_ldc_u4'+ldc_suffix] ]
        
                else:
                    raise ValueError("Only 'none', 'lin', 'quad', 'sqrt', 'exp',"+\
                                     "'log', 'sing', and 'claret' limb darkening "+\
                                     "laws are supported. However, your input was:"+\
                                     config.BASEMENT.settings[obj+'_ld_law_'+inst]+".")
    
                
    
    #=========================================================================
    #::: photometric errors, per instrument
    #=========================================================================
    for inst in config.BASEMENT.settings['inst_phot']:
        key='flux'
        params['err_'+key+'_'+inst] = np.exp( params['ln_err_'+key+'_'+inst] )
        
        
    #=========================================================================
    #::: RV jitter, per instrument
    #=========================================================================
    for inst in config.BASEMENT.settings['inst_rv']:
        key='rv'
        params['jitter_'+key+'_'+inst] = np.exp( params['ln_jitter_'+key+'_'+inst] )
        
    for inst in config.BASEMENT.settings['inst_rv2']:
        key='rv2'
        params['jitter_'+key+'_'+inst] = np.exp( params['ln_jitter_'+key+'_'+inst] )
        
        
    #=========================================================================
    #::: semi-major axes, per companion
    #=========================================================================
    for companion in config.BASEMENT.settings['companions_all']:
        params[companion+'_ecc'] = params[companion+'_f_s']**2 + params[companion+'_f_c']**2
        try:
            a_1 = 0.019771142 * params[companion+'_K'] * params[companion+'_period'] * np.sqrt(1. - params[companion+'_ecc']**2)/np.sin(params[companion+'_incl']*np.pi/180.)
            params[companion+'_a'] = (1.+1./params[companion+'_q'])*a_1
        except (TypeError, KeyError, ZeroDivisionError, ValueError):
            params[companion+'_a'] = None
        if params[companion+'_a'] == 0.:
            params[companion+'_a'] = None
               
                
    #=========================================================================
    #::: stellar spots, per companion and instrument
    #=========================================================================
    for companion in config.BASEMENT.settings['companions_all']:
        for inst in config.BASEMENT.settings['inst_all']:
            
            #---------------------------------------------------------------------
            #::: host spots
            #---------------------------------------------------------------------
            if config.BASEMENT.settings['host_N_spots_'+inst] > 0:
                params['host_spots_'+inst] = [
                                     [params['host_spot_'+str(i)+'_long_'+inst] for i in range(1,config.BASEMENT.settings['host_N_spots_'+inst]+1) ],
                                     [params['host_spot_'+str(i)+'_lat_'+inst] for i in range(1,config.BASEMENT.settings['host_N_spots_'+inst]+1) ],
                                     [params['host_spot_'+str(i)+'_size_'+inst] for i in range(1,config.BASEMENT.settings['host_N_spots_'+inst]+1) ],
                                     [params['host_spot_'+str(i)+'_brightness_'+inst] for i in range(1,config.BASEMENT.settings['host_N_spots_'+inst]+1) ]
                                    ]
        
            #---------------------------------------------------------------------
            #::: companion spots
            #---------------------------------------------------------------------
            if config.BASEMENT.settings[companion+'_N_spots_'+inst] > 0:
                params[companion+'_spots_'+inst] = [
                                     [params[companion+'_spot_'+str(i)+'_long_'+inst] for i in range(1,config.BASEMENT.settings[companion+'_N_spots_'+inst]+1) ],
                                     [params[companion+'_spot_'+str(i)+'_lat_'+inst] for i in range(1,config.BASEMENT.settings[companion+'_N_spots_'+inst]+1) ],
                                     [params[companion+'_spot_'+str(i)+'_size_'+inst] for i in range(1,config.BASEMENT.settings[companion+'_N_spots_'+inst]+1) ],
                                     [params[companion+'_spot_'+str(i)+'_brightness_'+inst] for i in range(1,config.BASEMENT.settings[companion+'_N_spots_'+inst]+1) ]
                                    ]
        
        
    #=========================================================================
    #::: host stellar density, per companion
    #::: (in cgs units)
    #::: this can directly be computed from Kepler's third law, by dividing by R_host**3
    #::: see also Seager & Mallen-Ornelas 2003 and Winn 2010, Eq. 30
    #::: only do this if it's actually requested by the user
    #=========================================================================
    if (config.BASEMENT.settings['use_host_density_prior'] is True) \
        and ('host_density' in config.BASEMENT.external_priors):
    
            for companion in config.BASEMENT.settings['companions_phot']:
        
                # """
                # If we have transit and RV data, we can constrain each companion's mass 
                # and density directly during sampling
                # """
                if (params[companion+'_rr'] is not None) and (params[companion+'_rr'] > 0) \
                    and (params[companion+'_K'] is not None) and (params[companion+'_K'] > 0):
                        
                    M_comp = calc_M_comp_from_RV(K = params[companion+'_K'],
                                                 P = params[companion+'_period'], 
                                                 incl = params[companion+'_incl'], 
                                                 ecc = params[companion+'_ecc'], 
                                                 M_host = config.BASEMENT.params_star['M_star_median'], 
                                                 return_unit = u.Msun) #in Msun
                    
                    R_comp = params[companion+'_rr'] * config.BASEMENT.params_star['R_star_median'] #in Rsun
                    
                    rho_comp = calc_rho(R = R_comp, 
                                        M = M_comp, 
                                        R_unit = u.Rsun,
                                        M_unit = u.Msun,
                                        return_unit='cgs') #in cgs units
                    
                    params[companion+'_host_density'] = calc_rho_host(P = params[companion+'_period'], 
                                                                      radius_1 = params[companion+'_radius_1'], 
                                                                      rr = params[companion+'_rr'], 
                                                                      rho_comp = rho_comp, 
                                                                      return_unit = 'cgs')
                        
                # """
                # Elif we only have transit data, we need to assume rr^3 * rho_comp := (R_companion/R_star)^3 * rho_comp --> 0. 
                # Hence, we here demand rr**3 < 0.01, allowing a 1% erroneous contribution by the planet density.
                # """
                elif (params[companion+'_rr'] is not None) and (params[companion+'_rr'] > 0) \
                    and (params[companion+'_rr']**3 < 0.01):
                        
                    params[companion+'_host_density'] = calc_rho_host(P = params[companion+'_period'], 
                                                                      radius_1 = params[companion+'_radius_1'], 
                                                                      rr = params[companion+'_rr'], 
                                                                      rho_comp = 0., #just to set the whole term to 0 
                                                                      return_unit = 'cgs')
                        
                # """
                # Else we can't do anything. 
                # """
                else:
                    params[companion+'_host_density'] = None
            
        
    #=========================================================================
    #::: lastly, deal with coupled params again after updates
    #=========================================================================
    for i, key in enumerate(config.BASEMENT.allkeys):
        if isinstance(config.BASEMENT.coupled_with[i], str) and (len(config.BASEMENT.coupled_with[i])>0):
            params[key] = params[config.BASEMENT.coupled_with[i]]
            
            
    return params




###############################################################################
#::: flux fct
###############################################################################
    
#==============================================================================
#::: flux fct: main
#==============================================================================
def flux_fct(params, inst, companion, xx=None, settings=None):
    '''
    ! params must be updated via update_params() before calling this function !
    '''
    
    if settings is None: 
        settings = config.BASEMENT.settings
    
    if settings['fit_ttvs']==False:
        return flux_fct_full(params, inst, companion, xx=xx, settings=settings)
    else:
        return flux_fct_piecewise(params, inst, companion, xx=xx, settings=settings)



#==============================================================================
#::: flux fct: full curve (no TTVs)
#==============================================================================
def flux_fct_full(params, inst, companion, xx=None, settings=None):
    '''
    ! params must be updated via update_params() before calling this function !
    '''
    
    #-------------------------------------------------------------------------- 
    #::: defaults
    #-------------------------------------------------------------------------- 
    if settings is None: 
        settings = config.BASEMENT.settings
        
    if xx is None: 
        xx = config.BASEMENT.data[inst]['time']
        t_exp = settings['t_exp_'+inst]
        n_int = settings['t_exp_n_int_'+inst]
    else:
        t_exp = None
        n_int = None
    
    
    #-------------------------------------------------------------------------- 
    #::: flux sub-fct: ellc lightcurve 
    # for eclipses, transits, occultations, 
    # spots, rotation,
    # and physical phase curve models; but no phase shift and no thermal vs reflected
    #-------------------------------------------------------------------------- 
    model_flux, model_flux1, model_flux2 = flux_subfct_ellc(params, inst, companion, xx=xx, settings=settings, t_exp=t_exp, n_int=n_int, return_fluxes=True)
        
        
    #-------------------------------------------------------------------------- 
    #::: flux sub-fct: ellc hacked phase curves
    # allowing phase shifts and thermal vs reflected
    #-------------------------------------------------------------------------- 
    '''
    if (companion+'_thermal_emission_amplitude_'+inst in params) and (params[companion+'_thermal_emission_amplitude_'+inst]>0):
        model_flux += calc_thermal_curve(params, inst, companion, xx, t_exp, n_int)
    '''
    
    
    #-------------------------------------------------------------------------- 
    #::: flux sub-fct: sinusoidal phase curves
    # allowing phase shifts and thermal vs reflected
    #-------------------------------------------------------------------------- 
    if settings['phase_curve_style'] in ['sine_series', 'sine_physical']:
        model_flux += flux_subfct_sinusoidal_phase_curves(params, inst, companion, model_flux2, xx=xx, settings=settings) - 1.
    
        
    #-------------------------------------------------------------------------- 
    #::: flux sub-fct: flare models
    #-------------------------------------------------------------------------- 
    if settings['N_flares'] > 0:
        model_flux += flux_subfct_flares(params, inst, companion, xx=xx, settings=settings) - 1.


    #--------------------------------------------------------------------------
    #::: flux sub-fct: bump models (starspot-crossing events)
    #--------------------------------------------------------------------------
    if settings.get('N_bumps_'+_bump_bandpass_suffix(inst, settings), 0) > 0:
        model_flux += flux_subfct_bumps(params, inst, companion, xx=xx, settings=settings) - 1.


    #--------------------------------------------------------------------------
    #::: return
    #--------------------------------------------------------------------------
    return model_flux



#==============================================================================
#::: flux sub-fct: ellc lightcurve
# for transits, occultations, eclipses, 
# and physical phase curve models (no phase shift; no thermal vs reflected)
#==============================================================================
def flux_subfct_ellc(params, inst, companion, xx=None, settings=None, t_exp=None, n_int=None, return_fluxes=False):
    '''
    ! params must be updated via update_params() before calling this function !
    '''
    
    #-------------------------------------------------------------------------- 
    #::: defaults
    #-------------------------------------------------------------------------- 
    if settings is None: 
        settings = config.BASEMENT.settings
        
    if xx is None: 
        xx = config.BASEMENT.data[inst]['time']
        t_exp = settings['t_exp_'+inst]
        n_int = settings['t_exp_n_int_'+inst]
    # else:
    #     t_exp = None
    #     n_int = None
    

    #-------------------------------------------------------------------------- 
    #::: index transits and occultations
    #-------------------------------------------------------------------------- 
    # ind_ecl1, ind_ecl2, ind_out = index_eclipses_smart(xx, params[companion+'_epoch'], params[companion+'_period'], params[companion+'_rr'], params[companion+'_rsuma'], params[companion+'_cosi'], params[companion+'_f_s'], params[companion+'_f_c'], extra_factor=1.5)
    #TODO: possible future speed improvement
        
        
    #-------------------------------------------------------------------------- 
    #::: if: planet and EB lightcurve model
    #-------------------------------------------------------------------------- 
    # Get bandpass-specific rr key for chromatic mode
    rr_key = config.BASEMENT.get_rr_key(companion, inst)
    rr = params.get(rr_key)
    
    # Fall back to achromatic key if chromatic key doesn't exist in params
    if rr is None:
        rr = params.get(f'{companion}_rr')
    
    if (rr is not None) and (rr > 0):
        # Recompute radius_1/radius_2 per-band so rsuma = radius_1 + radius_2
        # holds for this bandpass (required for chromatic mode; reduces to the
        # achromatic value when rr == params[companion+'_rr']).
        rsuma = params[companion+'_rsuma']
        if rsuma is not None:
            radius_1 = rsuma / (1. + rr)
            radius_2 = radius_1 * rr
        else:
            radius_1 = params[companion+'_radius_1']
            radius_2 = radius_1 * rr

        model_flux1, model_flux2 = ellc.fluxes(
                                    t_obs =       xx,
                                    radius_1 =    radius_1,
                                    radius_2 =    radius_2,
                                    sbratio =     params[companion+'_sbratio_'+inst], 
                                    incl =        params[companion+'_incl'], 
                                    # light_3 =     _calc_light_3(params.get('dil_'+inst)), #fluxes does not take light_3
                                    t_zero =      params[companion+'_epoch'],
                                    period =      params[companion+'_period'],
                                    a =           params[companion+'_a'],
                                    q =           params[companion+'_q'],
                                    f_c =         params[companion+'_f_c'],
                                    f_s =         params[companion+'_f_s'],
                                    ldc_1 =       params['host_ldc_'+inst],
                                    ldc_2 =       params[companion+'_ldc_'+inst],
                                    gdc_1 =       params['host_gdc_'+inst],
                                    gdc_2 =       params[companion+'_gdc_'+inst],
                                    didt =        params['didt_'+inst], 
                                    domdt =       params['domdt_'+inst], 
                                    rotfac_1 =    params['host_rotfac_'+inst], 
                                    rotfac_2 =    params[companion+'_rotfac_'+inst], 
                                    hf_1 =        params['host_hf_'+inst], #1.5, 
                                    hf_2 =        params[companion+'_hf_'+inst], #1.5,
                                    bfac_1 =      params['host_bfac_'+inst],
                                    bfac_2 =      params[companion+'_bfac_'+inst], 
                                    heat_1 =      divide(params['host_heat_'+inst],2.),
                                    heat_2 =      divide(params[companion+'_heat_'+inst],2.),
                                    lambda_1 =    params['host_lambda'], 
                                    lambda_2 =    params[companion+'_lambda'], 
                                    vsini_1 =     params['host_vsini'],
                                    vsini_2 =     params[companion+'_vsini'], 
                                    t_exp =       t_exp,
                                    n_int =       n_int,
                                    grid_1 =      settings['host_grid_'+inst],
                                    grid_2 =      settings[companion+'_grid_'+inst],
                                    ld_1 =        settings['host_ld_law_'+inst],
                                    ld_2 =        settings[companion+'_ld_law_'+inst],
                                    shape_1 =     settings['host_shape_'+inst],
                                    shape_2 =     settings[companion+'_shape_'+inst],
                                    spots_1 =     params['host_spots_'+inst], 
                                    spots_2 =     params[companion+'_spots_'+inst], 
                                    exact_grav =  settings['exact_grav'],
                                    verbose =     False
                                    )
  
        #::: combine the host and companion fluxes, and account for dilution
        model_flux = 1. + ( (model_flux1+model_flux2-1.) * (1.-params['dil_'+inst]) )  


    #-------------------------------------------------------------------------- 
    #::: else: constant 1
    #-------------------------------------------------------------------------- 
    else:
        model_flux1 = np.ones_like(xx)
        model_flux2 = np.zeros_like(xx)
        model_flux = np.ones_like(xx)

        
    #-------------------------------------------------------------------------- 
    #::: return
    #-------------------------------------------------------------------------- 
    if not return_fluxes:
        return model_flux
    else:
        return model_flux, model_flux1, model_flux2
    
    

#==============================================================================
#::: flux sub-fct: sinusoidal phase curves, allowing phase shifts
#==============================================================================
def flux_subfct_sinusoidal_phase_curves(params, inst, companion, model_flux2, xx=None, settings=None):    
    '''
    ! params must be updated via update_params() before calling this function !
    '''
    
    #-------------------------------------------------------------------------- 
    #::: defaults
    #-------------------------------------------------------------------------- 
    if settings is None:
        settings = config.BASEMENT.settings
        
    if xx is None:
        xx = config.BASEMENT.data[inst]['time']
        
    model_flux = np.ones_like(xx)
        
        
    #-------------------------------------------------------------------------- 
    #::: derive normalized curve for fractional loss of "atmospheric light" 
    # during occultation / secondary eclipse 
    #-------------------------------------------------------------------------- 
    if all(model_flux2==0.): flux2_norm = 1.
    else: flux2_norm = model_flux2/np.nanmax(model_flux2)
    
    
    #-------------------------------------------------------------------------- 
    #::: sine/cosine phase curve model
    # A1 (beaming)
    # B1 (atmospheric), can be split in thermal and reflected
    # B2 (ellipsoidal)
    # B3 (ellipsoidal 2nd order)
    #-------------------------------------------------------------------------- 
    
    #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    #::: the standard sine/cosine definition, e.g. used by Shporer and Wong
    #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    if settings['phase_curve_style'] == 'sine_series':
        
        if (params[companion+'_phase_curve_A1_'+inst] is not None): #A1 (beaming)
            model_flux += (1.-params['dil_'+inst]) * 1e-3*params[companion+'_phase_curve_A1_'+inst] * np.sin(2.*np.pi/params[companion+'_period'] * (xx - params[companion+'_epoch']))
            
        if (params[companion+'_phase_curve_B1_'+inst] is not None): #B1 (atmospheric)
            model_flux += (1.-params['dil_'+inst]) * 1e-3*params[companion+'_phase_curve_B1_'+inst] * (flux2_norm * (np.cos(2.*np.pi/params[companion+'_period'] * (xx - params[companion+'_epoch'] + params[companion+'_phase_curve_B1_shift_'+inst])) - 1) + 1 )

        if (params[companion+'_phase_curve_B1t_'+inst] is not None): #B1 (atmospheric) thermal part
            model_flux += (1.-params['dil_'+inst]) * 1e-3*params[companion+'_phase_curve_B1t_'+inst] * (flux2_norm * (np.cos(2.*np.pi/params[companion+'_period'] * (xx - params[companion+'_epoch'] + params[companion+'_phase_curve_B1t_shift_'+inst])) - 1) + 1 )

        if (params[companion+'_phase_curve_B1r_'+inst] is not None): #B1 (atmospheric) reflected part
            model_flux += (1.-params['dil_'+inst]) * 1e-3*params[companion+'_phase_curve_B1r_'+inst] * (flux2_norm * (np.cos(2.*np.pi/params[companion+'_period'] * (xx - params[companion+'_epoch'] + params[companion+'_phase_curve_B1r_shift_'+inst])) - 1) + 1 )

        if (params[companion+'_phase_curve_B2_'+inst] is not None): #B2 (ellipsoidal)
            model_flux += (1.-params['dil_'+inst]) * 1e-3*params[companion+'_phase_curve_B2_'+inst] * np.cos(2. * 2.*np.pi/params[companion+'_period'] * (xx - params[companion+'_epoch']))

        if (params[companion+'_phase_curve_B3_'+inst] is not None): #B3 (ellipsoidal)
            model_flux += (1.-params['dil_'+inst]) * 1e-3*params[companion+'_phase_curve_B3_'+inst] * np.cos(3. * 2.*np.pi/params[companion+'_period'] * (xx - params[companion+'_epoch']))

        #::: for debugging:
            
        # if (params[companion+'_phase_curve_A1_'+inst] is not None): #A1 (beaming)
        #     flux_A1 = (1.-params['dil_'+inst]) * 1e-3*params[companion+'_phase_curve_A1_'+inst] * np.sin(2.*np.pi/params[companion+'_period'] * (xx - params[companion+'_epoch']))
            
        # if (params[companion+'_phase_curve_B1_'+inst] is not None): #B1 (atmospheric)
        #     flux_B1 = (1.-params['dil_'+inst]) * 1e-3*params[companion+'_phase_curve_B1_'+inst] * (flux2_norm * (np.cos(2.*np.pi/params[companion+'_period'] * (xx - params[companion+'_epoch'] + params[companion+'_phase_curve_B1_shift_'+inst])) - 1) + 1 )

        # if (params[companion+'_phase_curve_B2_'+inst] is not None): #B2 (ellipsoidal)
        #     flux_B2 = (1.-params['dil_'+inst]) * 1e-3*params[companion+'_phase_curve_B2_'+inst] * np.cos(2. * 2.*np.pi/params[companion+'_period'] * (xx - params[companion+'_epoch']))

        # import matplotlib.pyplot as plt
        
        # plt.figure()
        # plt.plot(xx, flux2_norm, label='flux2_norm')
        # plt.legend()
        
        # plt.figure()
        # plt.plot(xx, flux_A1, label='A1')
        # plt.plot(xx, flux_B1, label='B1')
        # plt.plot(xx, flux_B2, label='B2')
        # plt.ylim([0.999-1,1.001-1])
        # plt.legend()
        

    #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    #::: the additive sine/cosine definition, e.g. used by Daylan
    #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    elif settings['phase_curve_style'] == 'sine_physical':
        
        if (params[companion+'_phase_curve_beaming_'+inst] is not None): #A1, rescaled
            model_flux += (1.-params['dil_'+inst]) * 1e-3*params[companion+'_phase_curve_beaming_'+inst] * np.sin(2.*np.pi/params[companion+'_period'] * (xx - params[companion+'_epoch']))
            
        if (params[companion+'_phase_curve_atmospheric_'+inst] is not None): #B1, rescaled
            model_flux += (1.-params['dil_'+inst]) * 1e-3*params[companion+'_phase_curve_atmospheric_'+inst] * flux2_norm * 0.5 * (1. - np.cos(2.*np.pi/params[companion+'_period'] * (xx - params[companion+'_epoch'] + params[companion+'_phase_curve_atmospheric_shift_'+inst])))
    
        if (params[companion+'_phase_curve_atmospheric_thermal_'+inst] is not None): #B1 thermal part, rescaled
            model_flux += (1.-params['dil_'+inst]) * 1e-3*params[companion+'_phase_curve_atmospheric_thermal_'+inst] * flux2_norm * 0.5 * (1. - np.cos(2.*np.pi/params[companion+'_period'] * (xx - params[companion+'_epoch'] + params[companion+'_phase_curve_atmospheric_thermal_shift_'+inst])))
    
        if (params[companion+'_phase_curve_atmospheric_reflected_'+inst] is not None): #B1 reflected part, rescaled
            model_flux += (1.-params['dil_'+inst]) * 1e-3*params[companion+'_phase_curve_atmospheric_reflected_'+inst] * flux2_norm * 0.5 * (1. - np.cos(2.*np.pi/params[companion+'_period'] * (xx - params[companion+'_epoch'] + params[companion+'_phase_curve_atmospheric_reflected_shift_'+inst])))
    
        if (params[companion+'_phase_curve_ellipsoidal_'+inst] is not None): #B2, rescaled
            model_flux += (1.-params['dil_'+inst]) * 1e-3*params[companion+'_phase_curve_ellipsoidal_'+inst] * 0.5 * (1. - np.cos(2. * 2.*np.pi/params[companion+'_period'] * (xx - params[companion+'_epoch'])))
   
        if (params[companion+'_phase_curve_ellipsoidal_2nd_'+inst] is not None): #B3, rescaled
            model_flux += (1.-params['dil_'+inst]) * 1e-3*params[companion+'_phase_curve_ellipsoidal_2nd_'+inst] * 0.5 * (1. - np.cos(3. * 2.*np.pi/params[companion+'_period'] * (xx - params[companion+'_epoch'])))
   
        
    # err
    #-------------------------------------------------------------------------- 
    #::: return
    #-------------------------------------------------------------------------- 
    return model_flux
    


#==============================================================================
#::: flux sub-fct: flare models
#==============================================================================
def flux_subfct_flares(params, inst, companion, xx=None, settings=None, return_fluxes=False):
    
    #-------------------------------------------------------------------------- 
    #::: defaults
    #-------------------------------------------------------------------------- 
    if settings is None:
        settings = config.BASEMENT.settings
        
    if xx is None:
        xx = config.BASEMENT.data[inst]['time']
        
    model_flux = np.ones_like(xx)
    
    
    #-------------------------------------------------------------------------- 
    #::: flares
    #-------------------------------------------------------------------------- 
    if settings['N_flares'] > 0:
        for i in range(1,settings['N_flares']+1):
            model_flux += (1.-params['dil_'+inst]) * aflare1(xx, params['flare_tpeak_'+str(i)], params['flare_fwhm_'+str(i)], params['flare_ampl_'+str(i)], upsample=True, uptime=10)


    #--------------------------------------------------------------------------
    #::: return
    #--------------------------------------------------------------------------
    return model_flux



#==============================================================================
#::: flux sub-fct: bump models (starspot-crossing events during transit)
#==============================================================================
def _bump_bandpass_suffix(inst, settings):
    '''
    The spot-crossing depth is wavelength-dependent (like limb darkening), so
    bump keys are suffixed by bandpass when a bandpass row is present, and fall
    back to the instrument name otherwise. Mirrors Basement.get_ldc_bandpass.
    '''
    bandpass = (settings.get('bandpass') or {}).get(inst)
    return bandpass if bandpass else inst


def flux_subfct_bumps(params, inst, companion, xx=None, settings=None, return_fluxes=False):

    #--------------------------------------------------------------------------
    #::: defaults
    #--------------------------------------------------------------------------
    if settings is None:
        settings = config.BASEMENT.settings

    if xx is None:
        xx = config.BASEMENT.data[inst]['time']

    model_flux = np.ones_like(xx)


    #--------------------------------------------------------------------------
    #::: bumps (count + amplitude keyed by bandpass; geometry shared across bands)
    #::: if bumps_persistent: a long-lived spot is crossed every transit, so the
    #::: bump recurs at tpeak + n*period (companion's orbital period) across xx
    #--------------------------------------------------------------------------
    suffix = _bump_bandpass_suffix(inst, settings)
    n_bumps = settings.get('N_bumps_'+suffix, 0)
    if n_bumps > 0:
        persistent = settings.get('bumps_persistent', False)
        for i in range(1, n_bumps+1):
            tpeak = params['bump_tpeak_'+str(i)]
            width = params['bump_width_'+str(i)]
            ampl = params['bump_ampl_'+suffix+'_'+str(i)]
            if persistent:
                period = params[companion+'_period']
                #::: only the epochs whose bump center lands within the data span
                n_lo = int(np.floor((np.min(xx) - tpeak) / period))
                n_hi = int(np.ceil((np.max(xx) - tpeak) / period))
                for n in range(n_lo, n_hi+1):
                    model_flux += (1.-params['dil_'+inst]) * bump_model(xx, tpeak + n*period, width, ampl)
            else:
                model_flux += (1.-params['dil_'+inst]) * bump_model(xx, tpeak, width, ampl)


    #--------------------------------------------------------------------------
    #::: return
    #--------------------------------------------------------------------------
    return model_flux



#==============================================================================
#::: flux sub-fct: ellc lightcurves piecewise (for TTVs; no phase curve)
#==============================================================================
def flux_fct_piecewise(params, inst, companion, xx=None, settings=None):
    '''
    ! params must be updated via update_params() before calling this function !
    '''
    
    #-------------------------------------------------------------------------- 
    #::: defaults
    #-------------------------------------------------------------------------- 
    if settings is None:
        settings = config.BASEMENT.settings
        
    if xx is None:
        t_exp = settings['t_exp_'+inst]
        n_int = settings['t_exp_n_int_'+inst]
        model_flux = np.ones_like(config.BASEMENT.data[inst]['time']) #* np.nan               
    else:
        t_exp = None
        n_int = None
        model_flux = np.ones_like(xx) #* np.nan     
    
    
    #-------------------------------------------------------------------------- 
    #::: go through the time series transit by transit to fit for TTVs
    #-------------------------------------------------------------------------- 
    for n_transit in range(len(config.BASEMENT.data[companion+'_tmid_observed_transits'])):
        
        if xx is None:
            ind = config.BASEMENT.data[inst][companion+'_ind_time_transit_'+str(n_transit+1)]
            xx_piecewise = config.BASEMENT.data[inst][companion+'_time_transit_'+str(n_transit+1)]
        else:
            tmid = config.BASEMENT.data[companion+'_tmid_observed_transits'][n_transit]
            width = settings['fast_fit_width']
            ind = np.where( (xx>=(tmid-width/2.)) \
                          & (xx<=(tmid+width/2.)) )[0]
            xx_piecewise = xx[ind]
        
        if len(xx_piecewise)>0:
            
            #::: FutureMax warning: one does not simply replace this with flux_subfct_ellc, because it needs the additive term after epoch
            # model_flux_piecewise = flux_subfct_ellc(params, inst, companion, xx=xx_piecewise, settings=settings, t_exp=t_exp, n_int=n_int)

            #::: planet and EB transit lightcurve model
            if (params[companion+'_rr'] is not None) and (params[companion+'_rr'] > 0):
                model_flux_piecewise = ellc.lc(
                                  t_obs =       xx_piecewise, 
                                  radius_1 =    params[companion+'_radius_1'], 
                                  radius_2 =    params[companion+'_radius_2'], 
                                  sbratio =     params[companion+'_sbratio_'+inst], 
                                  incl =        params[companion+'_incl'], 
                                  light_3 =     _calc_light_3(params.get('dil_'+inst)),
                                  t_zero =      params[companion+'_epoch'] + params[companion+'_ttv_transit_'+str(n_transit+1)],
                                  period =      params[companion+'_period'],
                                  a =           params[companion+'_a'],
                                  q =           params[companion+'_q'],
                                  f_c =         params[companion+'_f_c'],
                                  f_s =         params[companion+'_f_s'],
                                  ldc_1 =       params['host_ldc_'+inst],
                                  ldc_2 =       params[companion+'_ldc_'+inst],
                                  gdc_1 =       params['host_gdc_'+inst],
                                  gdc_2 =       params[companion+'_gdc_'+inst],
                                  didt =        params['didt_'+inst], 
                                  domdt =       params['domdt_'+inst], 
                                  rotfac_1 =    params['host_rotfac_'+inst], 
                                  rotfac_2 =    params[companion+'_rotfac_'+inst], 
                                  hf_1 =        params['host_hf_'+inst], #1.5, 
                                  hf_2 =        params[companion+'_hf_'+inst], #1.5,
                                  bfac_1 =      params['host_bfac_'+inst],
                                  bfac_2 =      params[companion+'_bfac_'+inst], 
                                  heat_1 =      divide(params['host_heat_'+inst],2.),
                                  heat_2 =      divide(params[companion+'_heat_'+inst],2.),
                                  lambda_1 =    params['host_lambda'], 
                                  lambda_2 =    params[companion+'_lambda'], 
                                  vsini_1 =     params['host_vsini'],
                                  vsini_2 =     params[companion+'_vsini'], 
                                  t_exp =       t_exp,
                                  n_int =       n_int,
                                  grid_1 =      settings['host_grid_'+inst],
                                  grid_2 =      settings[companion+'_grid_'+inst],
                                  ld_1 =        settings['host_ld_law_'+inst],
                                  ld_2 =        settings[companion+'_ld_law_'+inst],
                                  shape_1 =     settings['host_shape_'+inst],
                                  shape_2 =     settings[companion+'_shape_'+inst],
                                  spots_1 =     params['host_spots_'+inst], 
                                  spots_2 =     params[companion+'_spots_'+inst], 
                                  exact_grav =  settings['exact_grav'],
                                  verbose =     False
                                  )
                
                #::: and here comes an ugly hack around ellc, for those who want to fit reflected light and thermal emission separately
                '''
                if (companion+'_thermal_emission_amplitude_'+inst in params) and (params[companion+'_thermal_emission_amplitude_'+inst]>0):
                    model_flux += calc_thermal_curve(params, inst, companion, xx, t_exp, n_int)
                '''
                
            else:
                model_flux_piecewise = np.ones_like(xx)
                    
            model_flux[ind] = model_flux_piecewise


    #--------------------------------------------------------------------------
    #::: flux sub-fct: bump models (starspot-crossing events)
    #--------------------------------------------------------------------------
    if settings.get('N_bumps_'+_bump_bandpass_suffix(inst, settings), 0) > 0:
        model_flux += flux_subfct_bumps(params, inst, companion, xx=xx, settings=settings) - 1.


    #--------------------------------------------------------------------------
    #::: return
    #--------------------------------------------------------------------------
    return model_flux




'''
#==============================================================================
#::: flux sub-fct: ellc phase curve hack 
#==============================================================================
#::: and here comes an ugly hack around ellc, for those who want to fit reflected light (i.e. geometric albedo) and thermal emission separately
def flux_subfct_ellc_phase_curve_hack(params, inst, companion, xx, t_exp, n_int):

    #::: a shift in the phase curve
    if (companion+'_thermal_emission_timeshift_'+inst in params):
        xx_shifted = xx - params[companion+'_thermal_emission_timeshift_'+inst]
    
    
    #::: the thermal curve evaluated at the requested time values (arbitrary scaling)
    occultation = ellc.lc( 
                      t_obs =       xx, 
                      radius_1 =    params[companion+'_radius_1'], 
                      radius_2 =    params[companion+'_radius_2'], 
                      sbratio =     1e12, #a hack to get the flux drop to 0 during occulation
                      incl =        params[companion+'_incl'], 
                      light_3 =     _calc_light_3(params.get('dil_'+inst)),
                      t_zero =      params[companion+'_epoch'],
                      period =      params[companion+'_period'],
                      a =           params[companion+'_a'],
                      q =           params[companion+'_q'],
                      f_c =         params[companion+'_f_c'],
                      f_s =         params[companion+'_f_s'],
                      ldc_1 =       params['host_ldc_'+inst],
                      ldc_2 =       params[companion+'_ldc_'+inst],
                      gdc_1 =       0,
                      gdc_2 =       0,
                      didt =        params['didt_'+inst], 
                      domdt =       params['domdt_'+inst], 
                      rotfac_1 =    params['host_rotfac_'+inst], 
                      rotfac_2 =    params[companion+'_rotfac_'+inst], 
                      hf_1 =        params['host_hf_'+inst], #1.5, 
                      hf_2 =        params[companion+'_hf_'+inst], #1.5,
                      bfac_1 =      0,
                      bfac_2 =      0, 
                      heat_1 =      0,
                      heat_2 =      0,
                      lambda_1 =    params['host_lambda'], 
                      lambda_2 =    params[companion+'_lambda'], 
                      vsini_1 =     params['host_vsini'],
                      vsini_2 =     params[companion+'_vsini'], 
                      t_exp =       t_exp,
                      n_int =       n_int,
                      grid_1 =      config.BASEMENT.settings['host_grid_'+inst],
                      grid_2 =      config.BASEMENT.settings[companion+'_grid_'+inst],
                      ld_1 =        config.BASEMENT.settings['host_ld_law_'+inst],
                      ld_2 =        config.BASEMENT.settings[companion+'_ld_law_'+inst],
                      shape_1 =     'sphere',
                      shape_2 =     'sphere',
                      spots_1 =     None, 
                      spots_2 =     None, 
                      exact_grav =  config.BASEMENT.settings['exact_grav'],
                      verbose =     False
                      )
    
    #::: the thermal curve evaluated at the requested time values (arbitrary scaling)
    thermal_curve = ellc.lc( 
                      t_obs =       xx_shifted, 
                      radius_1 =    params[companion+'_radius_1'], 
                      radius_2 =    params[companion+'_radius_2'], 
                      sbratio =     0, 
                      incl =        params[companion+'_incl'], 
                      light_3 =     _calc_light_3(params.get('dil_'+inst)),
                      t_zero =      params[companion+'_epoch'],
                      period =      params[companion+'_period'],
                      a =           params[companion+'_a'],
                      q =           params[companion+'_q'],
                      f_c =         params[companion+'_f_c'],
                      f_s =         params[companion+'_f_s'],
                      ldc_1 =       params['host_ldc_'+inst],
                      ldc_2 =       params[companion+'_ldc_'+inst],
                      gdc_1 =       0,
                      gdc_2 =       0,
                      didt =        params['didt_'+inst], 
                      domdt =       params['domdt_'+inst], 
                      rotfac_1 =    params['host_rotfac_'+inst], 
                      rotfac_2 =    params[companion+'_rotfac_'+inst], 
                      hf_1 =        params['host_hf_'+inst], #1.5, 
                      hf_2 =        params[companion+'_hf_'+inst], #1.5,
                      bfac_1 =      0,
                      bfac_2 =      0, 
                      heat_1 =      0,
                      heat_2 =      0.1,
                      lambda_1 =    params['host_lambda'], 
                      lambda_2 =    params[companion+'_lambda'], 
                      vsini_1 =     params['host_vsini'],
                      vsini_2 =     params[companion+'_vsini'], 
                      t_exp =       t_exp,
                      n_int =       n_int,
                      grid_1 =      config.BASEMENT.settings['host_grid_'+inst],
                      grid_2 =      config.BASEMENT.settings[companion+'_grid_'+inst],
                      ld_1 =        config.BASEMENT.settings['host_ld_law_'+inst],
                      ld_2 =        config.BASEMENT.settings[companion+'_ld_law_'+inst],
                      shape_1 =     'sphere',
                      shape_2 =     'sphere',
                      spots_1 =     None, 
                      spots_2 =     None, 
                      exact_grav =  config.BASEMENT.settings['exact_grav'],
                      verbose =     False
                      )
    
    #::: a finely sampled thermal curve (arbitray scaling; fine sampling to get the maximum)
    thermal_curve_fine = ellc.lc(
                      t_obs =       np.linspace(params[companion+'_epoch'], params[companion+'_epoch']+params[companion+'_period'], 1000), 
                      radius_1 =    params[companion+'_radius_1'], 
                      radius_2 =    params[companion+'_radius_2'], 
                      sbratio =     0,
                      incl =        params[companion+'_incl'], 
                      light_3 =     _calc_light_3(params.get('dil_'+inst)),
                      t_zero =      params[companion+'_epoch'],
                      period =      params[companion+'_period'],
                      a =           params[companion+'_a'],
                      q =           params[companion+'_q'],
                      f_c =         params[companion+'_f_c'],
                      f_s =         params[companion+'_f_s'],
                      ldc_1 =       params['host_ldc_'+inst],
                      ldc_2 =       params[companion+'_ldc_'+inst],
                      gdc_1 =       0,
                      gdc_2 =       0,
                      didt =        params['didt_'+inst], 
                      domdt =       params['domdt_'+inst], 
                      rotfac_1 =    params['host_rotfac_'+inst], 
                      rotfac_2 =    params[companion+'_rotfac_'+inst], 
                      hf_1 =        params['host_hf_'+inst], #1.5, 
                      hf_2 =        params[companion+'_hf_'+inst], #1.5,
                      bfac_1 =      0,
                      bfac_2 =      0, 
                      heat_1 =      0,
                      heat_2 =      0.1,
                      lambda_1 =    params['host_lambda'], 
                      lambda_2 =    params[companion+'_lambda'], 
                      vsini_1 =     params['host_vsini'],
                      vsini_2 =     params[companion+'_vsini'], 
                      t_exp =       t_exp,
                      n_int =       n_int,
                      grid_1 =      config.BASEMENT.settings['host_grid_'+inst],
                      grid_2 =      config.BASEMENT.settings[companion+'_grid_'+inst],
                      ld_1 =        config.BASEMENT.settings['host_ld_law_'+inst],
                      ld_2 =        config.BASEMENT.settings[companion+'_ld_law_'+inst],
                      shape_1 =     'sphere',
                      shape_2 =     'sphere',
                      spots_1 =     None, 
                      spots_2 =     None, 
                      exact_grav =  config.BASEMENT.settings['exact_grav'],
                      verbose =     False
                      )
    
#    import matplotlib.pyplot as plt
#    
#    plt.figure()
#    plt.plot(xx_shifted, occultation)
#    
#    plt.figure()
#    plt.plot(xx_shifted, thermal_curve)
    
    #::: now scale the thermal curve
    thermal_curve[ thermal_curve<1. ] = 1.
    thermal_curve -= 1.
    thermal_curve *= occultation #a hack, he occultation time series is constant 0 in the occultation and constant 1 everywhere else
    thermal_curve /= np.max(thermal_curve_fine-1) #scaled from 0 to 1
    thermal_curve *= params[companion+'_thermal_emission_amplitude_'+inst]
    
    #::: cosine approximation
#            phi = calc_phase(xx_shifted, params[companion+'_period'], params[companion+'_epoch'])
#            thermal_curve += params[companion+'_thermal_emission_'+inst] * (0.5-0.5*np.cos(phi*2*np.pi))
    
    return thermal_curve
'''

    

###############################################################################
#::: rv fct
###############################################################################
def rv_fct(params, inst, companion, xx=None, settings=None):
    '''
    ! params must be updated via update_params() before calling this function !
    '''
    if xx is None:
        xx    = config.BASEMENT.data[inst]['time']
        t_exp = config.BASEMENT.settings['t_exp_'+inst]
        n_int = config.BASEMENT.settings['t_exp_n_int_'+inst]
    else:
        t_exp = None
        n_int = None
    
    if (params[companion+'_K'] is not None) and (params[companion+'_K'] > 0):
        model_rv1, model_rv2 = ellc.rv(
                          t_obs =       xx, 
                          radius_1 =    params[companion+'_radius_1'], 
                          radius_2 =    params[companion+'_radius_2'], 
                          sbratio =     params[companion+'_sbratio_'+inst], 
                          incl =        params[companion+'_incl'], 
                          t_zero =      params[companion+'_epoch'],
                          period =      params[companion+'_period'],
                          a =           params[companion+'_a'],
                          q =           params[companion+'_q'],
                          f_c =         params[companion+'_f_c'],
                          f_s =         params[companion+'_f_s'],
                          ldc_1 =       params['host_ldc_'+inst],
                          ldc_2 =       params[companion+'_ldc_'+inst],
                          gdc_1 =       params['host_gdc_'+inst],
                          gdc_2 =       params[companion+'_gdc_'+inst],
                          didt =        params['didt_'+inst], 
                          domdt =       params['domdt_'+inst], 
                          rotfac_1 =    params['host_rotfac_'+inst], 
                          rotfac_2 =    params[companion+'_rotfac_'+inst], 
                          hf_1 =        params['host_hf_'+inst], #1.5, 
                          hf_2 =        params[companion+'_hf_'+inst], #1.5,
                          bfac_1 =      params['host_bfac_'+inst],
                          bfac_2 =      params[companion+'_bfac_'+inst], 
                          heat_1 =      divide(params['host_heat_'+inst],2.),
                          heat_2 =      divide(params[companion+'_heat_'+inst],2.),
                          lambda_1 =    params['host_lambda'],
                          lambda_2 =    params[companion+'_lambda'], 
                          vsini_1 =     params['host_vsini'],
                          vsini_2 =     params[companion+'_vsini'], 
                          t_exp =       t_exp,
                          n_int =       n_int,
                          grid_1 =      config.BASEMENT.settings['host_grid_'+inst],
                          grid_2 =      config.BASEMENT.settings[companion+'_grid_'+inst],
                          ld_1 =        config.BASEMENT.settings['host_ld_law_'+inst],
                          ld_2 =        config.BASEMENT.settings[companion+'_ld_law_'+inst],
                          shape_1 =     config.BASEMENT.settings['host_shape_'+inst],
                          shape_2 =     config.BASEMENT.settings[companion+'_shape_'+inst],
                          spots_1 =     params['host_spots_'+inst], 
                          spots_2 =     params[companion+'_spots_'+inst], 
                          flux_weighted = config.BASEMENT.settings[companion+'_flux_weighted_'+inst],
                          verbose =     False
                          )
        
    else:
        model_rv1 = np.zeros_like(xx)
        model_rv2 = np.zeros_like(xx)
    
    return model_rv1, model_rv2




###############################################################################
#::: calculate external priors (e.g. stellar density and eccentricity cutoff)
###############################################################################  
def calculate_external_priors(params):
    '''
    ! params must be updated via update_params() before calling this function !
    
    bounds has to be list of len(theta), containing tuples of form
    ('none'), ('uniform', lower bound, upper bound), or ('normal', mean, std)
    '''
    lnp = 0.        
    
    #::: stellar density prior
    if (config.BASEMENT.settings['use_host_density_prior'] is True) \
        and ('host_density' in config.BASEMENT.external_priors):
            
        for companion in config.BASEMENT.settings['companions_phot']:
            '''
            The stellar density computed from R_host and M_host
            can be directly compared with the stellar density computed from the orbital motions (see e.g. Winn 2010)
            '''
            if params[companion+'_host_density'] is not None:
                    
                b = config.BASEMENT.external_priors['host_density']
                if b[0] == 'uniform':
                    if not (b[1] <= params[companion+'_host_density'] <= b[2]): return -np.inf
                elif b[0] == 'normal':
                    lnp += np.log( 1./(np.sqrt(2*np.pi) * b[2]) * np.exp( - (params[companion+'_host_density'] - b[1])**2 / (2.*b[2]**2) ) )
                else:
                    raise ValueError('Bounds have to be "uniform" or "normal". Input was "'+b[0]+'".')
    
    
    #::: constrain eccentricities to avoid ellc crashes
    for companion in config.BASEMENT.settings['companions_all']:
        '''
        The EXOFASTv2 paper (https://arxiv.org/abs/1907.09480; page 37) introduces these constrains from collisions:
        ecc < 1 - (a+R2)/R1
        and from tidal limits (optional):
        ecc < 1 - 3a/R1
        '''
        #::: avoid runaway eccentricities
        if not (params[companion+'_ecc'] < 1.):
            lnp = -np.inf
        
        #::: avoid collisions
        if (params[companion+'_rsuma'] is not None) \
            and not ((params[companion+'_ecc'] < (1. - params[companion+'_rsuma']))): 
            lnp = -np.inf
            
        # ::: avoid tidal circularizaion
        if (params[companion+'_radius_1'] is not None) \
            and (config.BASEMENT.settings['use_tidal_eccentricity_prior'] is True) \
            and not (params[companion+'_ecc'] < (1. - 3*params[companion+'_radius_1'])): 
            lnp = -np.inf
            
            
    #::: constrain dilution to avoid ellc crashes
    for inst in config.BASEMENT.settings['inst_all']:
        if (params['dil_'+inst] > 0.999):
            lnp = -np.inf
            
    return lnp




###############################################################################
#::: calculate lnlike
###############################################################################  

#==============================================================================
#::: Subfunction A: calculate lnlike for a detached binary
#==============================================================================
# def calculate_lnlike_detached_binary(params, inst):
#     """
#     This only works right now for the following conditions:
#         - it's a detached binary, consisting of 'host' and 'B'
#         - there are no other companions
#         - can't use hybrid baseline, hybrid errors, nor GPs for RVs
#     """  
    
#     lnlike_total = 0
    
#     #--------------------------------------------------------------------------  
#     #::: first, check and add external priors
#     #--------------------------------------------------------------------------  
#     lnprior_external = calculate_external_priors(params)   
#     lnlike_total += lnprior_external       
    
            
#     #--------------------------------------------------------------------------  
#     #::: directly catch any issues
#     #--------------------------------------------------------------------------  
#     if np.isnan(lnlike_total) or np.isinf(lnlike_total):
#         return -np.inf
    
    
#     #--------------------------------------------------------------------------  
#     #::: for all photometry instruments
#     #--------------------------------------------------------------------------   
#     key, key2 = 'flux', 'inst_phot'
    
#     for inst in config.BASEMENT.settings[key2]:
#         #::: calculate the model; if there are any NaN, return -np.inf
#         model = calculate_model(params, inst, key)
#         if any(np.isnan(model)) or any(np.isinf(model)): 
#             return -np.inf
        
#         #::: calculate errors, baseline and stellar variability
#         yerr_w = calculate_yerr_w(params, inst, key)
#         baseline = calculate_baseline(params, inst, key, model=model, yerr_w=yerr_w)
#         stellar_var = calculate_stellar_var(params, inst, key, model=model, baseline=baseline, yerr_w=yerr_w)
        
#         #::: calculate residuals and inv_simga2
#         residuals = config.BASEMENT.data[inst][key] - model - baseline - stellar_var
#         if any(np.isnan(residuals)): 
#             return -np.inf
#         inv_sigma2_w = 1./yerr_w**2
        
#         #::: calculate lnlike
#         lnlike_total += -0.5*(np.sum((residuals)**2 * inv_sigma2_w - np.log(inv_sigma2_w/2./np.pi))) #use np.sum to catch any nan and then set lnlike to nan
    
    
#     #--------------------------------------------------------------------------  
#     #::: for all RV instruments
#     #--------------------------------------------------------------------------   
#     for inst in config.BASEMENT.settings['inst_rv']:
#         key = 'rv12'
        
#         #::: calculate the models; if there are any NaN, return -np.inf
#         model_rv1, model_rv2 = calculate_model(params, inst, 'rv12')
#         if any(np.isnan(model_rv1*model_rv2)) or any(np.isinf(model_rv1*model_rv2)): 
#             return -np.inf
        
#         #::: calculate errors, baseline and stellar variability
#         yerr_w = calculate_yerr_w(params, inst, key)
#         baseline = calculate_baseline(params, inst, key, model=model, yerr_w=yerr_w)
#         stellar_var = calculate_stellar_var(params, inst, key, model=model, baseline=baseline, yerr_w=yerr_w)
        
#         #::: calculate residuals and inv_simga2
#         residuals = config.BASEMENT.data[inst][key] - model - baseline - stellar_var
#         if any(np.isnan(residuals)): 
#             return -np.inf
#         inv_sigma2_w = 1./yerr_w**2
        
#         #::: calculate lnlike
#         lnlike_total += -0.5*(np.sum((residuals)**2 * inv_sigma2_w - np.log(inv_sigma2_w/2./np.pi))) #use np.sum to catch any nan and then set lnlike to nan
    


#==============================================================================
#::: calculate all instruments linked (for stellar variability)
#==============================================================================
def calculate_lnlike_total(params):
    
    lnlike_total = 0
    
    #--------------------------------------------------------------------------  
    #::: first, check and add external priors
    #--------------------------------------------------------------------------  
    lnprior_external = calculate_external_priors(params)   
    lnlike_total += lnprior_external       
    
            
    #--------------------------------------------------------------------------  
    #::: directly catch any issues
    #--------------------------------------------------------------------------  
    if np.isnan(lnlike_total) or np.isinf(lnlike_total):
        return -np.inf
    
    
    #--------------------------------------------------------------------------  
    #::: for all instruments
    #--------------------------------------------------------------------------   
    for key, key2 in zip(['flux', 'rv', 'rv2'], ['inst_phot', 'inst_rv', 'inst_rv2']):      
        """
        Fitting detached binaries (with rv2 and inst_rv2) only works under the following conditions:
            - it's a detached binary, consisting of only 'host' and 'B'
            - the host's RV signal is given in the inst_rv file
            - the companion's RV signal is given in the inst_rv2 file
            - there are no other companions
            - can't use hybrid baseline, hybrid errors, nor GPs for RVs
            - the sample baseline and sample errors for RVs should be coupled
        """  
        
        #--------------------------------------------------------------------------       
        #::: CASE 1)
        #::: flux/rv stellar variability in FCTs --> can be calculated per inst (only GP needs to know about all other instruments) 
        #::: all flux/rv baselines in FCTs
        # TODO Qué?! An offset or a hybrid spline or polynomial also needs to know all other instruments!
        #-------------------------------------------------------------------------- 
        if ( config.BASEMENT.settings['stellar_var_'+key] in FCTs ) and all( [config.BASEMENT.settings['baseline_'+key+'_'+inst] in FCTs for inst in config.BASEMENT.settings[key2]] ):
#            print('CASE 1')
            for inst in config.BASEMENT.settings[key2]:
                
                #::: calculate the model; if there are any NaN, return -np.inf
                model = calculate_model(params, inst, key)
                if any(np.isnan(model)) or any(np.isinf(model)): 
                    return -np.inf
                
                #::: calculate errors, baseline and stellar variability
                yerr_w = calculate_yerr_w(params, inst, key)
                baseline = calculate_baseline(params, inst, key, model=model, yerr_w=yerr_w)
                stellar_var = calculate_stellar_var(params, inst, key, model=model, baseline=baseline, yerr_w=yerr_w)
                
                #::: calculate residuals and inv_simga2
                residuals = config.BASEMENT.data[inst][key] - model - baseline - stellar_var
                if any(np.isnan(residuals)): 
                    return -np.inf
                inv_sigma2_w = 1./yerr_w**2
                
                #::: calculate lnlike
                lnlike_total += -0.5*(np.sum((residuals)**2 * inv_sigma2_w - np.log(inv_sigma2_w/2./np.pi))) #use np.sum to catch any nan and then set lnlike to nan
                # Tier-2 marginal-lnlike correction: when the baseline is
                # hybrid_linear_multi the standard chi² above evaluates at
                # the MAP weights ŵ. Adding -½ŵᵀΛŵ - ½logdet(A) + ½logdet(Λ)
                # promotes it to the true marginal log-likelihood
                # (Gaussian-Gaussian closed form).
                if config.BASEMENT.settings['baseline_'+key+'_'+inst] == 'hybrid_linear_multi':
                    _y_resid = config.BASEMENT.data[inst][key] - model
                    _w_hat, _corr = _hybrid_linear_multi_solve(
                        inst, key, _y_resid, yerr_w)
                    lnlike_total += _corr



        #--------------------------------------------------------------------------
        #::: CASES 2a) and 2b)
        #::: stellar variability in FCTs --> can be calculated per inst (only GP needs to know about all other instruments) 
        #::: baseline in FCTs or in GPs
        #::: then DO NOT calculate the residuals via flux - gp.predict
        #::: but use gp.log_likelihood instead
        #--------------------------------------------------------------------------  
        elif ( config.BASEMENT.settings['stellar_var_'+key] in FCTs ) and any( [config.BASEMENT.settings['baseline_'+key+'_'+inst] in GPs for inst in config.BASEMENT.settings[key2]] ):
#            print('CASE 2')
            # Accumulate per-share-group (leader -> list of (x, y, yerr_w))
            # so members of a baseline_share_<key> group contribute one
            # joint GP log-likelihood instead of N independent ones.
            # Instruments not in any group form singleton "groups" which
            # collapse to today's per-inst path inside _stack_and_sort.
            _gp_groups = {}  # dict preserves insertion order on Python 3.7+
            for inst in config.BASEMENT.settings[key2]:

                #::: calculate the model; if there are any NaN, return -np.inf
                model = calculate_model(params, inst, key)
                if any(np.isnan(model)) or any(np.isinf(model)): return -np.inf


                #::: if that baseline is in FCTs
                if ( config.BASEMENT.settings['baseline_'+key+'_'+inst] in FCTs ):
#                    print('CASE 2a')

                    #::: calculate errors, baseline and stellar variability
                    yerr_w = calculate_yerr_w(params, inst, key)
                    baseline = calculate_baseline(params, inst, key, model=model, yerr_w=yerr_w)
                    stellar_var = calculate_stellar_var(params, inst, key, model=model, baseline=baseline, yerr_w=yerr_w)

                    #::: calculate residuals and inv_simga2
                    residuals = config.BASEMENT.data[inst][key] - model - baseline - stellar_var
                    if any(np.isnan(residuals)): raise ValueError('There are NaN in the residuals. Something horrible happened.')
                    inv_sigma2_w = 1./yerr_w**2

                    #::: calculate lnlike
                    lnlike_total += -0.5*(np.sum((residuals)**2 * inv_sigma2_w - np.log(inv_sigma2_w/2./np.pi))) #use np.sum to catch any nan and then set lnlike to nan
                    # Tier-2 marginal-lnlike correction (same logic as
                    # Case 1) — promotes the chi² at ŵ into the true
                    # Gaussian-marginalised log-likelihood.
                    if config.BASEMENT.settings['baseline_'+key+'_'+inst] == 'hybrid_linear_multi':
                        _y_resid = config.BASEMENT.data[inst][key] - model
                        _w_hat, _corr = _hybrid_linear_multi_solve(
                            inst, key, _y_resid, yerr_w)
                        lnlike_total += _corr


                #::: if that baseline is in GPs
                elif ( config.BASEMENT.settings['baseline_'+key+'_'+inst] in GPs ):
#                    print('CASE 2b')

                    #::: calculate the errors and stellar variability (assuming baseline=0.)
                    yerr_w = calculate_yerr_w(params, inst, key)
                    stellar_var = calculate_stellar_var(params, inst, key, model=model, baseline=0., yerr_w=yerr_w)

                    #::: collect this inst's residuals under its share leader
                    #::: (leader == inst for non-shared instruments). For
                    #::: non-shared insts the regression axis honours
                    #::: `baseline_<key>_<inst>_against` so covariate-vs-
                    #::: residual GP detrending is consistent with the
                    #::: predictive path in baseline_sample_GP.
                    x = _baseline_x_for(inst, key)
                    y = config.BASEMENT.data[inst][key] - model - stellar_var
                    _leader = _share_leader_of(inst, key)
                    _gp_groups.setdefault(_leader, []).append((x, y, yerr_w))


                else:
                    raise ValueError('Kaput.')

            #::: one joint GP per share group (or per inst when no sharing)
            for _leader, _parts in _gp_groups.items():
                x_j, y_j, yerr_j = _stack_and_sort(_parts)
                # cosort_for_gp is a no-op when x_j is already strictly
                # increasing (the multi-member path inside _stack_and_sort
                # already handled that). It kicks in for singleton buckets
                # whose axis is a non-monotone covariate.
                x_j, y_j, yerr_j = _cosort_for_gp(x_j, y_j, yerr_j)
                gp = baseline_get_gp(params, _leader, key)
                try:
                    gp.compute(x_j, yerr=yerr_j)
                    lnlike_total += gp.log_likelihood(y_j)
                except Exception:
                    return -np.inf
                    
                    
                
        #--------------------------------------------------------------------------       
        #::: CASE 3) 
        #::: stellar variability in GPs
        #::: baseline in FCTs
        #::: do stuff
        #-------------------------------------------------------------------------- 
        elif ( config.BASEMENT.settings['stellar_var_'+key] in GPs ):
#            print('CASE 3')
            y, yerr_w = [], []
            for inst in config.BASEMENT.settings[key2]:
                
                #::: calculate the model. if there are any NaN, return -np.inf
                model_i = calculate_model(params, inst, key)
                if any(np.isnan(model_i)) or any(np.isinf(model_i)): return -np.inf
                               
                #::: calculate the errors and baseline
                yerr_w_i = calculate_yerr_w(params, inst, key)
                baseline_i = calculate_baseline(params, inst, key, model=model_i, yerr_w=yerr_w_i)
                residuals_i = config.BASEMENT.data[inst][key] - model_i - baseline_i

                y += list(residuals_i)
                yerr_w += list(yerr_w_i)
                
            #::: sort in time
            ind_sort = config.BASEMENT.data[key2]['ind_sort']
            x = 1.*config.BASEMENT.data[key2]['time']
            y = np.array(y)[ind_sort]
            yerr = np.array(yerr_w)[ind_sort]  
            
            #::: calculate the stellar variability's gp.log_likelihood (instead of evaluating the gp)
            gp = stellar_var_get_gp(params, key)
            try:
                gp.compute(x, yerr=yerr)
                lnlike_total += gp.log_likelihood(y)
            except Exception:
                return -np.inf
            
            
                
        #--------------------------------------------------------------------------       
        #::: CASE 4) 
        #::: stellar variability in GPs
        #::: baseline in GPs
        #::: raise an error
        #--------------------------------------------------------------------------  
        elif ( config.BASEMENT.settings['stellar_var_'+key] in GPs )\
           and any( [config.BASEMENT.settings['baseline_'+key+'_'+inst] in GPs for inst in config.BASEMENT.settings[key2]] ):
            raise KeyError('Currently you cannot use a GP for stellar variability and a GP for the baseline at the same time.')

                    
            
    #--------------------------------------------------------------------------  
    #::: again, catch any issues
    #--------------------------------------------------------------------------  
    if np.isnan(lnlike_total) or np.isinf(lnlike_total):
        return -np.inf
        
        
    return lnlike_total


    
#==============================================================================
#::: calculate all instruments separately 
#::: (only possible if no stellar variability GP is involved)
#==============================================================================
#def calculate_lnlike(params, inst, key):
#        
#    #::: calculate the model. if there are any NaN, return -np.inf
#    model = calculate_model(params, inst, key)
#    if any(np.isnan(model)) or any(np.isinf(model)):
#        return -np.inf
#            
#    #--------------------------------------------------------------------------       
#    #::: CASE 3) 
#    #::: if no stellar variability GP and
#    #::: no baseline GP,
#    #::: then calculate lnlike manually
#    #--------------------------------------------------------------------------  
#    elif ( config.BASEMENT.settings['stellar_var_'+key] not in GPs ) \
#        and (config.BASEMENT.settings['baseline_'+key+'_'+inst] not in GPs ):
#        
#        yerr_w = calculate_yerr_w(params, inst, key)
#        baseline = calculate_baseline(params, inst, key, model=model, yerr_w=yerr_w)
#        
#        residuals = config.BASEMENT.data[inst][key] - model - baseline
#        inv_sigma2_w = 1./yerr_w**2
#        
#        lnlike = -0.5*(np.sum((residuals)**2 * inv_sigma2_w - np.log(inv_sigma2_w/2./np.pi))) #use np.sum to catch any nan and then set lnlike to nan
#    
#        if any(np.isnan(residuals)):
#            raise ValueError('There are NaN in the residuals. Something horrible happened.')
#    
#    
#    #--------------------------------------------------------------------------  
#    #::: CASE 4)
#    #::: if no stellar variability GP and
#    #::: baseline GP, 
#    #::: then DO NOT calculate the residuals via flux - gp.predict
#    #::: but use gp.log_likelihood instead
#    #--------------------------------------------------------------------------  
#    else:
#        x = config.BASEMENT.data[inst]['time']
#        y = config.BASEMENT.data[inst][key] - model
#        yerr_w = calculate_yerr_w(params, inst, key)
#        gp = baseline_get_gp(params, inst, key)
#        try:
#            gp.compute(x, yerr=yerr_w)
#            lnlike = gp.log_likelihood(y)
#        except Exception:
#            lnlike = -np.inf
#        
#        
#    return lnlike
    
    
    

###############################################################################
#::: calculate yerr
############################################################################### 
def calculate_yerr_w(params, inst, key):
    '''
    Returns:
    --------
    yerr_w : array of float
        the weighted yerr
    '''
    if inst in config.BASEMENT.settings['inst_phot']:
        yerr_w = config.BASEMENT.data[inst]['err_scales_'+key] * params['err_'+key+'_'+inst]
    elif (inst in config.BASEMENT.settings['inst_rv']) or (inst in config.BASEMENT.settings['inst_rv2']):
        yerr_w = np.sqrt( config.BASEMENT.data[inst]['white_noise_'+key]**2 + params['jitter_'+key+'_'+inst]**2 )
    return yerr_w


        

###############################################################################
#::: calculate residuals
# TODO DEPRECATED
###############################################################################  
# def calculate_residuals(params, inst, key):
#     '''
#     Note:
#     -----
#     No 'xx' here, because residuals can only be calculated on given data
#     (not on an arbitrary xx grid)
#     '''       
#     model = calculate_model(params, inst, key)
#     baseline = calculate_baseline(params, inst, key, model=model)
#     residuals = config.BASEMENT.data[inst][key] - model - baseline
#     return residuals


    

###############################################################################
#::: calculate model
###############################################################################      
def calculate_model(params, inst, key, xx=None, settings=None):
            
    if settings is None:
        settings = config.BASEMENT.settings
    
    if key=='flux':
        depth = 0.
        for companion in settings['companions_phot']:
            depth += ( 1. - flux_fct(params, inst, companion, xx=xx, settings=settings) )
        model_flux = 1. - depth
        return model_flux
    
    elif key=='rv':
        model_rv = 0.
        for companion in settings['companions_rv']:
            model_rv += rv_fct(params, inst, companion, xx=xx, settings=settings)[0] 
            # RV signal measured from the host/primary (caused by a sepcific companion's gravity)
        return model_rv
    
    elif key=='rv2':
        model_rv2 = 0.
        for companion in settings['companions_rv']:
            model_rv2 += rv_fct(params, inst, companion, xx=xx, settings=settings)[1] 
            # RV_2 signal measured from the companion/secondary (caused by the host's gravity)
        return model_rv2
    
    elif key=='rv12':
        model_rv = 0.
        model_rv2 = 0.
        for companion in settings['companions_rv']:
            model_rv_temp, model_rv2_temp = rv_fct(params, inst, companion, xx=xx, settings=settings) 
            model_rv += model_rv_temp
            model_rv2 += model_rv2_temp
            # RV signal measured from the host/primary (caused by a sepcific companion's gravity)
            # RV_2 signal measured from the companion/secondary (caused by the host's gravity)
        return model_rv, model_rv2
    
    elif (key=='centdx') | (key=='centdy'):
        raise ValueError("Fitting for 'centdx' and 'centdy' not yet implemented.")
        #TODO
        
    else:
        raise ValueError("Variable 'key' has to be 'flux', 'rv', 'centdx', or 'centdy'.")




###############################################################################
#::: calculate baseline
############################################################################### 
        
#==============================================================================
#::: calculate baseline: main
#==============================================================================
def calculate_baseline(params, inst, key, model=None, yerr_w=None, xx=None):

    '''
    Inputs:
    -------
    params : dict
        ...
    inst : str
        ...
    key : str
        ...
    model = array of float (optional; default=None)
        ...
    xx : array of float (optional; default=None)
        if given, evaluate the baseline fit on the xx values 
        (e.g. a finer time grid for plotting)
        else, it's the same as data[inst]['time'] or data[inst]['custom_series']
        
    Returns: 
    --------
    baseline : array of float
        the baseline evaluate on the grid x (or xx, if xx!=None)
    '''
    
    if model is None: 
        model = calculate_model(params, inst, key, xx=None) #the model has to be evaluated on the time grid

    if yerr_w is None: 
        yerr_w = calculate_yerr_w(params, inst, key)
        
    _against = config.BASEMENT.settings['baseline_'+key+'_'+inst+'_against']
    if _against == 'time':
        x = config.BASEMENT.data[inst]['time']
    elif _against == 'custom_series':
        x = config.BASEMENT.data[inst]['custom_series']
    else:
        # Named covariate column from the CSV header.
        _covs = config.BASEMENT.data[inst].get('covariates', {})
        if _against in _covs:
            x = _covs[_against]
        else:
            raise KeyError(
                "baseline_{k}_{i}_against={a!r} but {i}.csv has no column "
                "of that name. Known options: time, custom_series, {names}".format(
                    k=key, i=inst, a=_against,
                    names=sorted(_covs.keys()),
                )
            )
        
    y = config.BASEMENT.data[inst][key] - model
    
    if xx is None:  
        xx = 1.*x
    '''
    x : array of float
        time stamps of the data
    y : array of float
        y = data_y - model_y
        i.e., the values that you want to constrain the baseline on
    yerr_w : array of float
        the weighted yerr
    yerr_weights : array of float
        normalized error weights on y
    '''
    
    baseline_method = config.BASEMENT.settings['baseline_'+key+'_'+inst]

    if baseline_method not in baseline_switch:
        _setting = 'baseline_'+key+'_'+inst
        _valid = sorted(baseline_switch.keys())
        _close = _difflib.get_close_matches(
            str(baseline_method), _valid, n=3, cutoff=0.4)
        _hint = ('Did you mean: '+', '.join(repr(c) for c in _close)+'?  '
                 ) if _close else ''
        raise KeyError(
            "settings.csv has {s}={m!r}, which is not a known baseline kind. "
            "{hint}Valid options are: {valid}.".format(
                s=_setting, m=baseline_method, hint=_hint, valid=_valid))
    return baseline_switch[baseline_method](x, y, yerr_w, xx, params, inst, key)



#==============================================================================
#::: calculate baseline: hybrid_offset (like Gillon+2012, but only remove mean offset)
#==============================================================================
def baseline_hybrid_offset(*args):
    x, y, yerr_w, xx, params, inst, key = args
    yerr_weights = yerr_w/np.nanmean(yerr_w)
    weights = 1./yerr_weights
    ind = np.isfinite(y) #np.average can't handle NaN
    return np.average(y[ind], weights=weights[ind]) * np.ones_like(xx)
 

    
#==============================================================================
#::: calculate baseline: hybrid_poly (like Gillon+2012)
#==============================================================================   
def baseline_hybrid_poly(*args):
    x, y, yerr_w, xx, params, inst, key = args
    polyorder = int(config.BASEMENT.settings['baseline_'+key+'_'+inst][-1])
    xx = (xx - x[0])/x[-1] #polyfit needs the xx-axis scaled to [0,1], otherwise it goes nuts
    x = (x - x[0])/x[-1] #polyfit needs the x-axis scaled to [0,1], otherwise it goes nuts
    if polyorder>=0:
        yerr_weights = yerr_w/np.nanmean(yerr_w)
        weights = 1./yerr_weights
        ind = np.isfinite(y) #polyfit can't handle NaN
        params_poly = poly.polyfit(x[ind],y[ind],polyorder,w=weights[ind]) #WARNING: returns params in reverse order than np.polyfit!!!
        baseline = poly.polyval(xx, params_poly) #evaluate on xx (!)
    else:
        raise ValueError("'polyorder' has to be > 0.")
        
    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots()
    # ax.plot(x,y,'b.')
    # ax.plot(xx,baseline,'r-', lw=2)
    # ax.set(xlim=[np.min(xx), 2.458332e6])
    # plt.show()
    # input('press enter to continue')
    return baseline    



#==============================================================================
#::: calculate baseline: hybrid_spline (like Gillon+2012, but with a cubic spline)
#==============================================================================
def baseline_hybrid_spline(*args):
    x, y, yerr_w, xx, params, inst, key = args
    yerr_weights = yerr_w/np.nanmean(yerr_w)
    weights = 1./yerr_weights
    ind = np.isfinite(y) #mask NaN
    spl = UnivariateSpline(x[ind],y[ind],w=weights[ind],s=np.sum(weights[ind]))
    baseline = spl(xx) #evaluate on xx (!)
    
#    if any(np.isnan(baseline)):
    # import matplotlib.pyplot as plt
    # print('x:\n', x[0:5], '\n in range:', np.min(x), np.median(x), np.max(x), '\n Nall=', len(x), ' Nvalid=', len(np.isfinite(x)))
    # print('xx:\n', xx[0:5], '\n in range:', np.min(xx), np.median(xx), np.max(xx), '\n Nall=', len(xx), ' Nvalid=', len(np.isfinite(xx)))
    # print('y:\n', y[0:5], '\n in range:', np.min(y), np.median(y), np.max(y), '\n Nall=', len(y), ' Nvalid=', len(np.isfinite(y)))
    # print('weights:\n', weights[0:5], '\n in range:', np.min(weights), np.median(weights), np.max(weights), '\n Nall=', len(weights), ' Nvalid=', len(np.isfinite(weights)))
    # print('baseline:\n', baseline[0:5], '\n in range:', np.min(baseline), np.median(baseline), np.max(baseline), '\n Nall=', len(baseline), ' Nvalid=', len(np.isfinite(baseline)))
    # fig, ax = plt.subplots()
    # ax.plot(x,y,'b.')
    # ax.plot(xx,baseline,'r-', lw=2)
    # ax.set(xlim=[np.min(xx), 2.458332e6])
    # plt.show()
    # input('press enter to continue')
    
    return baseline   



#==============================================================================
#::: calculate baseline: hybrid_spline 
#::: (like Gillon+2012, but with a cubic spline, here with a manually given s value)
#==============================================================================
def baseline_hybrid_spline_s(*args):
    x, y, yerr_w, xx, params, inst, key = args
    yerr_weights = yerr_w/np.nanmean(yerr_w)
    weights = 1./yerr_weights
    ind = np.isfinite(y) #mask NaN
    spl = UnivariateSpline(x[ind],y[ind],w=weights[ind],
                           s=float(config.BASEMENT.settings['baseline_'+key+'_'+inst+'_args']))
    baseline = spl(xx) #evaluate on xx (!)
    
    return baseline   
     
    
    
#==============================================================================
#::: calculate baseline: hybrid_GP (like Gillon+2012, but with a GP)
#==============================================================================           
def baseline_hybrid_GP(*args):
    x, y, yerr_w, xx, params, inst, key = args
    
    if celerite_version == 2:
        kernel = terms.Matern32Term(log_sigma=1., log_rho=1.)
        gp = celerite.GP(kernel, mean=np.nanmean(y)) 
    elif celerite_version == 1:
        kernel = terms.Matern32Term(log_sigma=1., log_rho=1.)
        gp = celerite.GP(kernel, mean=np.nanmean(y)) 
    else:
        raise ImportError('You have come too far; you need celerite or celerite2 to do what you want to do.')
    gp.compute(x, yerr=yerr_w) #constrain on x/y/yerr
     
    def neg_log_like(gp_params, y, gp): #TODO: this is not yet suited for celerite2
        gp.set_parameter_vector(gp_params)
        return -gp.log_likelihood(y)
    
    def grad_neg_log_like(gp_params, y, gp): #TODO: this is not yet suited for celerite2
        gp.set_parameter_vector(gp_params)
        return -gp.grad_log_likelihood(y)[1]
    
    initial_params = gp.get_parameter_vector()
    bounds = gp.get_parameter_bounds()
    soln = minimize(neg_log_like, initial_params, jac=grad_neg_log_like,
                    method="L-BFGS-B", bounds=bounds, args=(y, gp))
    gp.set_parameter_vector(soln.x)
    
    baseline = gp_predict_in_chunks(gp, y, xx)[0]
#    baseline = gp.predict(y, xx)[0] #constrain on x/y/yerr, evaluate on xx (!)
    return baseline 



#==============================================================================
#::: calculate baseline: sample_offset
########################################################################### 
def baseline_sample_offset(*args):
    x, y, yerr_w, xx, params, inst, key = args
    return params['baseline_offset_'+key+'_'+inst] * np.ones_like(xx)
        


#==============================================================================
#::: calculate baseline: sample_linear
#============================================================================== 
def baseline_sample_linear(*args):
    x, y, yerr_w, xx, params, inst, key = args
    xx_norm = (xx-x[0]) / (x[-1]-x[0])
    return params['baseline_slope_'+key+'_'+inst] * xx_norm + params['baseline_offset_'+key+'_'+inst]



#==============================================================================
#::: calculate baseline: sample_linear_multi (N-D linear regression on a
#::: pre-built design matrix; one weight per column sampled directly).
#::: Mirrors timex's `pt.dot(X, weights)` systematics model.
#==============================================================================
def _linmulti_interp_to_xx(bl_train, inst, xx):
    """Project a linear-multi baseline from the training grid onto ``xx``.

    The Tier-1 / Tier-2 linear-multi baselines compute ``X @ w`` on the
    instrument's training time grid (length ``N_inst``). Callers in the
    plotting pipeline often pass a denser ``xx`` (e.g. a high-resolution
    grid for smooth display), so the raw ``X @ w`` shape doesn't match
    what ``afplot`` expects. We do a 1-D linear interpolation in time;
    when ``xx`` is identical to (or a sub/superset of) the training grid
    this is exact, and outside the training span the behaviour is
    constant-extrapolation (numpy's ``interp`` default).
    """
    x_train = config.BASEMENT.data[inst]['time']
    xx_arr = np.asarray(xx, dtype=float)
    bl_train = np.asarray(bl_train, dtype=float)
    # Fast path: same length and same values → return unchanged.
    if xx_arr.shape == x_train.shape and np.array_equal(xx_arr, x_train):
        return bl_train
    return np.interp(xx_arr, x_train, bl_train)


def baseline_sample_linear_multi(*args):
    """Return the N-D linear systematics model ``X @ weights``.

    The design matrix ``X`` and column ordering are built once in
    :meth:`Basement.load_data` and stored on
    ``config.BASEMENT.data[inst]['design_matrix']`` /
    ``['design_matrix_cols']``. Weights are pulled from ``params`` by
    name in the same order; each weight is a free fit parameter named
    ``baseline_linmulti_<col>_<key>_<inst>`` (or
    ``baseline_linmulti_w<i>_<key>_<inst>`` for synthetic columns
    like `bias`).
    """
    x, y, yerr_w, xx, params, inst, key = args
    X = config.BASEMENT.data[inst].get('design_matrix')
    cols = config.BASEMENT.data[inst].get('design_matrix_cols', [])
    if X is None or len(cols) == 0:
        raise KeyError(
            "baseline_{k}_{i}=sample_linear_multi but no design matrix is "
            "stored for {i}. Did you set baseline_{k}_{i}_cols=<names>?".format(
                k=key, i=inst))
    w = np.array([
        params['baseline_linmulti_'+name+'_'+key+'_'+inst]
        for name in cols
    ], dtype=float)
    return _linmulti_interp_to_xx(X @ w, inst, xx)


#==============================================================================
#::: calculate baseline: hybrid_linear_multi (Tier 2 — analytically
#::: marginalised Gaussian linear regression). The weights are NOT
#::: sampled; they are solved in closed form at every likelihood call
#::: from the post-transit-model residuals. ndim stays the same as it
#::: would without any linear-multi baseline.
#==============================================================================
_HYBRID_LINEAR_MULTI_PRIOR_SIGMA = 1e3   # matches timex's pm.Normal(0, 1e3)


def _hybrid_linear_multi_solve(inst, key, y_resid, yerr_w):
    """Solve for the MAP weights ŵ and return the per-instrument
    correction term for the marginal log-likelihood.

    Math (Gaussian likelihood, Gaussian prior on weights):

        A   = Xᵀ Σ⁻¹ X + Λ                # Σ = diag(yerr²),  Λ = (1/σ_p²) I
        ŵ   = A⁻¹ Xᵀ Σ⁻¹ y_resid
        log p(y) - log p_no_baseline(y_after_subtracting_X@ŵ)
            = -½ ŵᵀ Λ ŵ - ½ log det(A) + ½ log det(Λ)

    The caller computes the *standard* gaussian chi² with the MAP
    baseline (``y_resid - X @ ŵ``) and then adds this correction; the
    sum equals the true marginal log-likelihood.

    Returns
    -------
    w_hat : ndarray, shape (k,)
        MAP weights given the current residuals.
    correction : float
        Additive contribution to the marginal log-likelihood relative
        to the chi² at ŵ.
    """
    X = config.BASEMENT.data[inst]['design_matrix']
    k = X.shape[1]
    inv_sigma2 = 1.0 / np.asarray(yerr_w, dtype=float)**2
    sigma_p2 = _HYBRID_LINEAR_MULTI_PRIOR_SIGMA**2
    A = X.T @ (X * inv_sigma2[:, None]) + (1.0 / sigma_p2) * np.eye(k)
    b = X.T @ (np.asarray(y_resid, dtype=float) * inv_sigma2)
    try:
        L = np.linalg.cholesky(A)
    except np.linalg.LinAlgError:
        # Ridge nudge for near-singular A (e.g., collinear covariates).
        A = A + 1e-12 * np.eye(k)
        L = np.linalg.cholesky(A)
    w_hat = np.linalg.solve(L.T, np.linalg.solve(L, b))
    logdet_A = 2.0 * float(np.sum(np.log(np.diag(L))))
    # log det Λ = log det((1/σ_p²) I) = -k log σ_p²
    logdet_Lambda = -k * np.log(sigma_p2)
    prior_quad = float(w_hat @ ((1.0 / sigma_p2) * w_hat))
    correction = -0.5 * prior_quad - 0.5 * logdet_A + 0.5 * logdet_Lambda
    return w_hat, correction


def baseline_hybrid_linear_multi(*args):
    """Return ``X @ ŵ`` where ŵ is the analytic MAP weights given the
    current residuals. No fit-vector parameters are consumed."""
    x, y, yerr_w, xx, params, inst, key = args
    if 'design_matrix' not in config.BASEMENT.data[inst]:
        raise KeyError(
            "baseline_{k}_{i}=hybrid_linear_multi but no design matrix is "
            "stored for {i}. Did you set baseline_{k}_{i}_cols=<names>?".format(
                k=key, i=inst))
    X = config.BASEMENT.data[inst]['design_matrix']
    w_hat, _ = _hybrid_linear_multi_solve(inst, key, y, yerr_w)
    return _linmulti_interp_to_xx(X @ w_hat, inst, xx)


#==============================================================================
#::: baseline share group helpers
#::: A share group is a list of instruments (members) that fit a *single*
#::: celerite GP jointly across all their residuals (e.g. MuSCAT g/r/i/z).
#::: The first member of each group is the leader and owns the sampled GP
#::: hyperparameters; followers' GP-hyper rows are aliased to the leader's
#::: in update_params() above. These helpers return (leader, [members])
#::: with sensible defaults so non-shared instruments collapse to single-
#::: element groups, leaving legacy behaviour bit-for-bit unchanged.
#==============================================================================
def _share_leader_of(inst, key):
    return config.BASEMENT.settings.get(
        'baseline_share_'+key+'_leader_of', {}).get(inst, inst)


def _share_members_of(leader, key):
    followers = config.BASEMENT.settings.get(
        'baseline_share_'+key+'_followers_of', {}).get(leader, [])
    return [leader] + list(followers)


def _stack_and_sort(parts):
    '''Concatenate per-instrument (x, y, yerr) triples and return a single
    strictly-sorted (x_cat, y_cat, yerr_cat) suitable for celerite.

    For a single-element input the original arrays are returned untouched,
    preserving bit-for-bit legacy behaviour for non-shared instruments.
    For multi-element inputs, exact-timestamp ties (common in simultaneous
    multi-band photometry) are broken by nudging duplicates one ULP forward
    via np.nextafter so the sampler still sees strictly increasing x.
    '''
    if len(parts) == 1:
        return parts[0]
    x_cat = np.concatenate([np.asarray(p[0], dtype=float) for p in parts])
    y_cat = np.concatenate([np.asarray(p[1], dtype=float) for p in parts])
    yerr_cat = np.concatenate([np.asarray(p[2], dtype=float) for p in parts])
    ind = np.argsort(x_cat, kind='stable')
    x_sorted = x_cat[ind].copy()
    y_sorted = y_cat[ind]
    yerr_sorted = yerr_cat[ind]
    for i in range(1, len(x_sorted)):
        if x_sorted[i] <= x_sorted[i-1]:
            x_sorted[i] = np.nextafter(x_sorted[i-1], np.inf)
    return x_sorted, y_sorted, yerr_sorted


def _baseline_x_for(inst, key):
    """Return the regression axis for the GP baseline of ``inst``.

    Respects ``baseline_<key>_<inst>_against``: ``time`` (default),
    ``custom_series`` (legacy alias), or a named covariate column from
    ``<inst>.csv``. Share-group followers are validated to use ``time``
    in :meth:`Basement.load_settings`, so the joint-GP path always sees
    a time axis when ``len(members) > 1``.
    """
    against = config.BASEMENT.settings.get(
        'baseline_'+key+'_'+inst+'_against', 'time')
    if against == 'time':
        return config.BASEMENT.data[inst]['time']
    if against == 'custom_series':
        return config.BASEMENT.data[inst]['custom_series']
    covs = config.BASEMENT.data[inst].get('covariates', {})
    if against in covs:
        return covs[against]
    raise KeyError(
        "baseline_{k}_{i}_against={a!r} but no such column in "
        "{i}.csv covariates ({names}).".format(
            k=key, i=inst, a=against, names=sorted(covs.keys())))


def _cosort_for_gp(x, y, yerr):
    """Sort ``(x, y, yerr)`` by ``x`` and nudge tied ``x`` values one
    ULP apart so celerite's strict-sort precondition is satisfied.
    No-op when ``x`` is already strictly increasing.
    """
    x_arr = np.asarray(x, dtype=float)
    if x_arr.size <= 1 or np.all(np.diff(x_arr) > 0):
        return x_arr, np.asarray(y, dtype=float), np.asarray(yerr, dtype=float)
    perm = np.argsort(x_arr, kind='stable')
    xs = x_arr[perm].copy()
    ys = np.asarray(y, dtype=float)[perm]
    es = np.asarray(yerr, dtype=float)[perm]
    for i in range(1, len(xs)):
        if xs[i] <= xs[i-1]:
            xs[i] = np.nextafter(xs[i-1], np.inf)
    return xs, ys, es


def _collect_joint_residuals(params, members, key):
    '''Build per-group (x, y, yerr) triples for the GP baseline path.

    For each member, y = data - model - stellar_var (the residual the
    baseline GP is fit to), matching exactly what calculate_lnlike_total
    Case 2b builds per inst today. The x axis is selected per-inst via
    `_baseline_x_for`, which defers to `baseline_<key>_<inst>_against`
    (singleton "groups" can therefore regress against a covariate; true
    share groups always get time because the loader enforces it).
    '''
    parts = []
    for m in members:
        x_m = _baseline_x_for(m, key)
        model_m = calculate_model(params, m, key)
        yerr_m = calculate_yerr_w(params, m, key)
        stellar_var_m = calculate_stellar_var(
            params, m, key, model=model_m, baseline=0., yerr_w=yerr_m)
        y_m = config.BASEMENT.data[m][key] - model_m - stellar_var_m
        parts.append((x_m, y_m, yerr_m))
    return _stack_and_sort(parts)


#==============================================================================
#::: calculate baseline: sample_GP
#==============================================================================
def baseline_sample_GP(*args):
    x, y, yerr_w, xx, params, inst, key = args
    leader = _share_leader_of(inst, key)
    members = _share_members_of(leader, key)
    if len(members) <= 1:
        # Legacy path: single-instrument GP. Time is already sorted by the
        # loader; covariate axes may not be — cosort to satisfy celerite.
        x_s, y_s, e_s = _cosort_for_gp(x, y, yerr_w)
        gp = baseline_get_gp(params, leader, key)
        gp.compute(x_s, yerr=e_s)
        baseline = gp_predict_in_chunks(gp, y_s, xx)[0]
        return baseline
    # Shared-realization path: build the joint GP under the leader's
    # hyperparameters and predict at this inst's grid (xx).
    x_j, y_j, yerr_j = _collect_joint_residuals(params, members, key)
    gp = baseline_get_gp(params, leader, key)
    gp.compute(x_j, yerr=yerr_j)
    baseline = gp_predict_in_chunks(gp, y_j, xx)[0]
#    baseline = gp.predict(y, xx)[0]
    
#    baseline2 = gp.predict(y, x)[0]
#    plt.figure()
#    plt.plot(x,y,'k.', color='grey')
#    plt.plot(xx,baseline,'r-', lw=2)
#    plt.plot(x,baseline2,'ro', lw=2)
#    plt.title(inst+' '+key+' '+str(gp.get_parameter_vector()))
#    plt.show()
#    raw_input('press enter to continue')
    
    return baseline



#==============================================================================
#::: calculate baseline: get GP kernel
#============================================================================== 
def baseline_get_gp(params, inst, key):
    
    #::: kernel
    if config.BASEMENT.settings['baseline_'+key+'_'+inst] == 'sample_GP_real':
        kernel = terms.RealTerm(log_a=params['baseline_gp_real_lna_'+key+'_'+inst], 
                                log_c=params['baseline_gp_real_lnc_'+key+'_'+inst])
        
    elif config.BASEMENT.settings['baseline_'+key+'_'+inst] == 'sample_GP_complex':
        kernel = terms.ComplexTerm(log_a=params['baseline_gp_complex_lna_'+key+'_'+inst], 
                                log_b=params['baseline_gp_complex_lnb_'+key+'_'+inst], 
                                log_c=params['baseline_gp_complex_lnc_'+key+'_'+inst], 
                                log_d=params['baseline_gp_complex_lnd_'+key+'_'+inst])
                  
    elif config.BASEMENT.settings['baseline_'+key+'_'+inst] == 'sample_GP_Matern32':
        kernel = terms.Matern32Term(log_sigma=params['baseline_gp_matern32_lnsigma_'+key+'_'+inst], 
                                    log_rho=params['baseline_gp_matern32_lnrho_'+key+'_'+inst])
        
    elif config.BASEMENT.settings['baseline_'+key+'_'+inst] == 'sample_GP_SHO':
        kernel = terms.SHOTerm(log_S0=params['baseline_gp_sho_lnS0_'+key+'_'+inst], 
                               log_Q=params['baseline_gp_sho_lnQ_'+key+'_'+inst],
                               log_omega0=params['baseline_gp_sho_lnomega0_'+key+'_'+inst])                               
        
    else: 
        raise KeyError('GP settings and params do not match.')
    
    #::: GP and mean (simple offset)  
    if 'baseline_gp_offset_'+key+'_'+inst in params:
        gp = celerite.GP(kernel, mean=params['baseline_gp_offset_'+key+'_'+inst], fit_mean=True)
    else:
        gp = celerite.GP(kernel, mean=0.) #mean=0. because it is the mean of the residuals, as the GP is applied to the residuals
        
    return gp



#==============================================================================
#::: calculate baseline: none
#============================================================================== 
def baseline_none(*args):
    x, y, yerr_w, xx, params, inst, key = args
    return np.zeros_like(xx)



#==============================================================================
#::: calculate baseline: raise error
#==============================================================================
def baseline_raise_error(*args):
    x, y, yerr_w, xx, params, inst, key = args
    raise ValueError('Setting '+'baseline_'+key+'_'+inst+' has to be sample_offset / sample_linear / sample_GP / hybrid_offset / hybrid_poly_1 / hybrid_poly_2 / hybrid_poly_3 / hybrid_pol_4 / hybrid_spline / hybrid_GP, '+\
                     "\nbut is:"+config.BASEMENT.settings['baseline_'+key+'_'+inst])



#==============================================================================
#::: calculate baseline: baseline_switch
#==============================================================================    
baseline_switch = \
    {
    'hybrid_offset' : baseline_hybrid_offset,
    'hybrid_poly_0' : baseline_hybrid_poly,
    'hybrid_poly_1' : baseline_hybrid_poly,
    'hybrid_poly_2' : baseline_hybrid_poly,
    'hybrid_poly_3' : baseline_hybrid_poly,
    'hybrid_poly_4' : baseline_hybrid_poly,
    'hybrid_poly_5' : baseline_hybrid_poly,
    'hybrid_poly_6' : baseline_hybrid_poly,
    'hybrid_spline' : baseline_hybrid_spline,
    'hybrid_spline_s' : baseline_hybrid_spline_s, #hybrid spline but with a manually given s-value (to avoid any slow if/else in the function)
    'hybrid_GP'     : baseline_hybrid_GP,
    'sample_offset' : baseline_sample_offset,
    'sample_linear' : baseline_sample_linear,
    'sample_linear_multi' : baseline_sample_linear_multi,
    'hybrid_linear_multi' : baseline_hybrid_linear_multi,
    'sample_GP_real'         : baseline_sample_GP, #only for plotting
    'sample_GP_complex'      : baseline_sample_GP, #only for plotting   
    'sample_GP_Matern32'     : baseline_sample_GP, #only for plotting
    'sample_GP_SHO'          : baseline_sample_GP, #only for plotting         
    'none'          : baseline_none #only for plotting    
    }
    

    
    
###########################################################################
#::: GP predict in chunks (to avoid memory crashes)
########################################################################### 
def gp_predict_in_chunks(gp, y, x, chunk_size=5000):
    mu = []
    var = []
    for i in range( int(1.*len(x)/chunk_size)+1 ):
        m, v = gp.predict(y, x[i*chunk_size:(i+1)*chunk_size], return_var=True)
        mu += list(m)
        var += list(v)
    return np.array(mu), np.array(var)




###############################################################################
#::: Stellar Variability
############################################################################### 
            
#==============================================================================
#::: Stellar Variability: main
#==============================================================================
def calculate_stellar_var(params, inst, key, model=None, baseline=None, yerr_w=None, xx=None):

    #--------------------------------------------------------------------------
    #::: over all instruments (needed for GP)
    #--------------------------------------------------------------------------
    stellar_var_method = config.BASEMENT.settings['stellar_var_'+key]   
    
    if stellar_var_method not in ['none']:
        if key=='flux': key2 = 'inst_phot'
        elif key=='rv': key2 = 'inst_rv'
        else: raise KeyError('Kaput.')
        
        if inst=='all': insts = config.BASEMENT.settings[key2]
        else: insts = [inst]
        
        #::: loop over a separate variable so the `inst` argument is NOT
        #::: shadowed — we still need to know below whether it was 'all'
        #::: (shared GP over all instruments) or a single instrument.
        y_list,yerr_w_list = [],[]
        for inst_i in insts:
            if model is None:
                model_i = calculate_model(params, inst_i, key)
            else:
                model_i = model
            if baseline is None:
                baseline_i = calculate_baseline(params, inst_i, key, model=model)
            else:
                baseline_i = baseline
            residuals = config.BASEMENT.data[inst_i][key] - model_i - baseline_i
            y_list += list(residuals)

            if yerr_w is None:
                yerr_w_list += list(calculate_yerr_w(params, inst_i, key))
            else:
                yerr_w_list += list(yerr_w)

        #::: 'all' uses the merged, time-sorted phot series (ind_sort maps the
        #::: instrument-concatenated y/yerr into that time order); a single
        #::: instrument uses its own time grid as-is.
        if inst=='all':
            ind_sort = config.BASEMENT.data[key2]['ind_sort']
            x = 1.*config.BASEMENT.data[key2]['time']
        else:
            ind_sort = slice(None)
            x = 1.*config.BASEMENT.data[inst]['time']

        y = np.array(y_list)[ind_sort]
        yerr_w = np.array(yerr_w_list)[ind_sort]
        if xx is None: xx = 1.*x
    
        if stellar_var_method not in stellar_var_switch:
            _setting = 'stellar_var_'+key
            _valid = sorted(stellar_var_switch.keys())
            _close = _difflib.get_close_matches(
                str(stellar_var_method), _valid, n=3, cutoff=0.4)
            _hint = ('Did you mean: '+', '.join(repr(c) for c in _close)+'?  '
                     ) if _close else ''
            raise KeyError(
                "settings.csv has {s}={m!r}, which is not a known stellar-"
                "variability kind. {hint}Valid options are: {valid}.".format(
                    s=_setting, m=stellar_var_method, hint=_hint, valid=_valid))
        return stellar_var_switch[stellar_var_method](x, y, yerr_w, xx, params, key)
    
    else:
        return 0.
    


#==============================================================================
#::: Stellar Variability: get GP kernel
#============================================================================== 
def stellar_var_get_gp(params, key):
    
    #::: kernel
    if config.BASEMENT.settings['stellar_var_'+key] == 'sample_GP_real':
        kernel = terms.RealTerm(log_a=params['stellar_var_gp_real_lna_'+key], 
                                log_c=params['stellar_var_gp_real_lnc_'+key])
        
    elif config.BASEMENT.settings['stellar_var_'+key] == 'sample_GP_complex':
        kernel = terms.ComplexTerm(log_a=params['stellar_var_gp_complex_lna_'+key], 
                                log_b=params['stellar_var_gp_complex_lnb_'+key], 
                                log_c=params['stellar_var_gp_complex_lnc_'+key], 
                                log_d=params['stellar_var_gp_complex_lnd_'+key])
                  
    elif config.BASEMENT.settings['stellar_var_'+key] == 'sample_GP_Matern32':
        kernel = terms.Matern32Term(log_sigma=params['stellar_var_gp_matern32_lnsigma_'+key], 
                                    log_rho=params['stellar_var_gp_matern32_lnrho_'+key])
        
    elif config.BASEMENT.settings['stellar_var_'+key] == 'sample_GP_SHO':
        kernel = terms.SHOTerm(log_S0=params['stellar_var_gp_sho_lnS0_'+key], 
                               log_Q=params['stellar_var_gp_sho_lnQ_'+key],
                               log_omega0=params['stellar_var_gp_sho_lnomega0_'+key])                               
        
    else: 
        raise KeyError('GP settings and params do not match.')
    
    #::: GP and mean (simple offset)  
    if 'stellar_var_gp_offset_'+key in params:
        gp = celerite.GP(kernel, mean=params['stellar_var_gp_offset_'+key], fit_mean=True)
    else:
        gp = celerite.GP(kernel, mean=0.) #mean=0. because it is the mean of the residuals, as the GP is applied to the residuals
        
        
    return gp



#==============================================================================
#::: Stellar Variability: sample_GP
#============================================================================== 
def stellar_var_sample_GP(*args):
    x, y, yerr_w, xx, params, key = args
    gp = stellar_var_get_gp(params, key)
    gp.compute(x, yerr=yerr_w)
    stellar_var = gp_predict_in_chunks(gp, y, xx)[0]
    return stellar_var



#==============================================================================
#::: Stellar Variability: sample_linear
#============================================================================== 
def stellar_var_sample_linear(*args):
    x, y, yerr_w, xx, params, key = args
    if key=='flux': key2 = 'inst_phot'
    elif key=='rv': key2 = 'inst_rv'
    x_all = 1.*config.BASEMENT.data[key2]['time']
    xx_norm = (xx-x_all[0]) / (x_all[-1]-x_all[0])
#    xx_norm = (xx - config.BASEMENT.settings['mid_epoch'])
#    xx_norm = (xx - 2457000.) / 1000.
    return params['stellar_var_slope_'+key] * xx_norm + params['stellar_var_offset_'+key]
        
    

#==============================================================================
#::: Stellar Variability: none
#============================================================================== 
def stellar_var_none(*args):
    x, y, yerr_w, xx, params, key = args
    return np.zeros_like(xx)



#==============================================================================
#::: Stellar Variability: stellar_var_switch
#==============================================================================    
stellar_var_switch = \
    {
    'sample_linear'          : stellar_var_sample_linear,
    'sample_GP_real'         : stellar_var_sample_GP, #only for plotting
    'sample_GP_complex'      : stellar_var_sample_GP, #only for plotting   
    'sample_GP_Matern32'     : stellar_var_sample_GP, #only for plotting
    'sample_GP_SHO'          : stellar_var_sample_GP, #only for plotting           
    'none'                   : stellar_var_none #only for plotting    
    }
    


    
################################################################################
##::: def calculate inv_sigma2
################################################################################  
#def calculate_inv_sigma2_w(params, inst, key, residuals=None):
#    '''
#    _w means "weighted", a.k.a. multiplied by data[inst]['err_scales_'+key]**(-2)
#    '''
#    
#    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::    
#    #::: traditional (sampling in MCMC)
#    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::  
#    if config.BASEMENT.settings['error_'+key+'_'+inst].lower() == 'sample':
#        yerr_w = calculate_yerr_w(params, inst, key)
#        inv_sigma2_w = 1./yerr_w**2
#        return inv_sigma2_w
#    
#     
#    #:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::: 
#    #::: 'hybrid_inv_sigma2'
#    #:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::: 
#    elif config.BASEMENT.settings['error_'+key+'_'+inst].lower() == 'hybrid': 
#        raise ValueError('Currently no longer implemented.')
##        if residuals is None:
##            residuals = calculate_residuals(params, inst, key)
##        
##        #::: neg log like
##        def neg_log_like(inv_sigma2, inst, key, residuals):
###            inv_sigma2_w = config.BASEMENT.data[inst]['err_scales_'+key]**(-2) * inv_sigma2
##            inv_sigma2_w = calculate_inv_sigma2_w_1(inv_sigma2, inst, key)                
##            return + 0.5*(np.nansum((residuals)**2 * inv_sigma2_w - np.log(inv_sigma2_w)))
##            
##        
##        #::: grad neg log like
##        def grad_neg_log_like(inv_sigma2, inst, key, residuals):
###            inv_sigma2_w = config.BASEMENT.data[inst]['err_scales_'+key]**(-2) * inv_sigma2
##            inv_sigma2_w = calculate_inv_sigma2_w_1(inv_sigma2, inst, key)                
##            return np.array( + 0.5*(np.nansum((residuals)**2 - 1./inv_sigma2_w)) )
##        
##        
###        guess = params['inv_sigma2_'+key+'_'+inst]
##        guess = 1./np.std(residuals)**2 #Warning: this relies on a good initial guess for the model, otherwise std(residuals) will be nuts
##
##        #::: MLE (gradient based)
##        soln_MLE = minimize(neg_log_like, guess,
##                        method = 'L-BFGS-B', jac=grad_neg_log_like,
##                        bounds=[(1e-16,1e+16)], args=(inst, key, residuals))
##        
##        #::: Diff. Evol.
###        bounds = [(0.001*guess,1000.*guess)]
###        soln_DE = differential_evolution(neg_log_like, bounds, args=(inst, key, residuals))
##
##        inv_sigma2 = soln_MLE.x[0]      
##        inv_sigma2_w = calculate_inv_sigma2_w_1(inv_sigma2, inst, key)                
##
###        print inst, key
###        print '\tguess:', int(guess), 'lnlike:', neg_log_like(guess, inst, key, residuals)
###        print '\tMLE:', int(soln_MLE.x[0]), 'lnlike:', neg_log_like(soln_MLE.x[0], inst, key, residuals) 
###        print '\tDE:', int(soln_DE.x[0]), 'lnlike:', neg_log_like(soln_DE.x[0], inst, key, residuals)
##        
##        return inv_sigma2_w
#    
#
#    else:
#        raise ValueError('Setting '+'error_'+key+'_'+inst+' has to be sample / hybrid, '+\
#                         "\nbut is:"+params['error_'+key+'_'+inst])
