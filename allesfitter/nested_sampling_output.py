#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 14:28:55 2018

@author:
Dr. Maximilian N. Günther
European Space Agency (ESA)
European Space Research and Technology Centre (ESTEC)
Keplerlaan 1, 2201 AZ Noordwijk, The Netherlands
Email: maximilian.guenther@esa.int
GitHub: mnguenther
Twitter: m_n_guenther
Web: www.mnguenther.com
"""

from __future__ import print_function, division, absolute_import

#::: plotting settings
import seaborn as sns
sns.set(context='paper', style='ticks', palette='deep', font='sans-serif', font_scale=1.5, color_codes=True)
sns.set_style({"xtick.direction": "in","ytick.direction": "in"})
sns.set_context(rc={'lines.markeredgewidth': 1})

#::: modules
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, MaxNLocator
import os
import gzip
try:
   import cPickle as pickle
except:
   import pickle
from copy import deepcopy
from dynesty import utils as dyutils
from dynesty import plotting as dyplot
import warnings
try:
    import corner as _corner  # used as backend-agnostic cornerplot fallback
except ImportError:           # corner is a transitive dynesty dep, but guard anyway
    _corner = None

#::: allesfitter modules
from . import config
from . import deriver
from .computer import calculate_model, calculate_baseline, calculate_stellar_var
from .general_output import afplot, afplot_per_transit, save_table, save_latex_table, logprint, get_params_from_samples, plot_ttv_results
from .plot_top_down_view import plot_top_down_view
from .utils.colormaputil import truncate_colormap
from .utils.latex_printer import round_tex
from .statistics import residual_stats
                     

    

###############################################################################
#::: draw samples from the ns results (internally in the code)
###############################################################################
def draw_ns_posterior_samples(results, Nsamples=None, as_type='2d_array'):
    '''
    ! posterior samples are drawn as resampled weighted samples !
    ! do not confuse posterior_samples (weighted, resampled) with results['samples'] (unweighted) !
    '''
    weights = np.exp(results['logwt'] - results['logz'][-1])
    np.random.seed(42)
    posterior_samples = dyutils.resample_equal(results['samples'], weights)    
    if Nsamples:
        posterior_samples = posterior_samples[np.random.randint(len(posterior_samples), size=Nsamples)]

    if as_type=='2d_array':
        return posterior_samples
    
    elif as_type=='dic':
        posterior_samples_dic = {}
        for key in config.BASEMENT.fitkeys:
            ind = np.where(config.BASEMENT.fitkeys==key)[0]
            posterior_samples_dic[key] = posterior_samples[:,ind].flatten()
        return posterior_samples_dic



###############################################################################
#::: backend-agnostic plotting helpers (used when backend != 'dynesty')
###############################################################################
def _simple_traceplot(results, labels, truths):
    '''Lightweight traceplot when dynesty's runplot/traceplot is unavailable.

    Mirrors the (ndim, 2) layout dyplot.traceplot returns so the downstream
    title-setting code (``taxes[i,1].set_title(...)`` etc.) keeps working.
    Left column: index vs. parameter value (raw samples).
    Right column: weighted-posterior histogram.
    '''
    samples = np.asarray(results['samples'])
    logwt = np.asarray(results['logwt'])
    logz_final = float(np.asarray(results['logz'])[-1])
    weights = np.exp(logwt - logz_final)
    weights = weights / weights.sum()
    ndim = samples.shape[1]
    fig, axes = plt.subplots(ndim, 2, figsize=(12, 2.5 * ndim))
    if ndim == 1:
        axes = np.array([axes])
    for i in range(ndim):
        axes[i, 0].plot(samples[:, i], lw=0.5, color='grey', rasterized=True)
        axes[i, 0].set_ylabel(labels[i] if i < len(labels) else '')
        axes[i, 1].hist(samples[:, i], bins=60, weights=weights,
                        color='grey', histtype='stepfilled', alpha=0.6)
        if truths is not None and i < len(truths) and truths[i] is not None:
            axes[i, 0].axhline(truths[i], color='C3', lw=0.8)
            axes[i, 1].axvline(truths[i], color='C3', lw=0.8)
    axes[-1, 0].set_xlabel('sample index')
    axes[-1, 1].set_xlabel('value')
    return fig, axes


def _corner_from_results(results, labels, truths, fontsize, ndim):
    '''Backend-agnostic cornerplot via ``corner.corner``.

    Resamples to equal weight using the same ``logwt``/``logz`` contract
    as :func:`draw_ns_posterior_samples`, then hands off to ``corner``.
    Returns ``(fig, axes_2d)`` where ``axes_2d`` matches dyplot.cornerplot's
    shape so existing title-setting code is unchanged.
    '''
    samples = np.asarray(results['samples'])
    logwt = np.asarray(results['logwt'])
    logz_final = float(np.asarray(results['logz'])[-1])
    weights = np.exp(logwt - logz_final)
    weights = weights / weights.sum()
    eq = dyutils.resample_equal(samples, weights)
    if _corner is None:
        # graceful empty grid if corner.py is somehow missing
        cfig, caxes = plt.subplots(ndim, ndim, figsize=(2 * ndim, 2 * ndim))
        return cfig, caxes
    cfig = _corner.corner(
        eq,
        labels=labels,
        truths=truths,
        quantiles=[0.16, 0.5, 0.84],
        show_titles=False,
        label_kwargs={"fontsize": fontsize, "rotation": 45,
                      "horizontalalignment": "right"},
        hist_kwargs={"alpha": 0.25, "linewidth": 0, "histtype": "stepfilled"},
    )
    caxes = np.array(cfig.axes).reshape((ndim, ndim))
    return cfig, caxes


###############################################################################
#::: chromatic Rp/Rs posterior histogram
###############################################################################
def _load_R_star_samples(n_samples, seed=42):
    '''Load R_star posterior from params_star.csv and draw n_samples
    from an asymmetric normal. Returns (R_star_samples_in_Rsun, R_star_med)
    or (None, None) if params_star.csv is absent or malformed.

    Honors both columns ``R_star_lerr`` (lower 1-sigma) and ``R_star_uerr``
    (upper 1-sigma); a sample's draw uses the side appropriate to its sign.
    '''
    path = os.path.join(config.BASEMENT.datadir, 'params_star.csv')
    if not os.path.exists(path):
        return None, None
    try:
        buf = np.genfromtxt(
            path, delimiter=',', names=True, dtype=None,
            encoding='utf-8', comments='#',
        )
        R_star = float(np.atleast_1d(buf['R_star'])[0])
        R_lerr = float(np.atleast_1d(buf['R_star_lerr'])[0])
        R_uerr = float(np.atleast_1d(buf['R_star_uerr'])[0])
    except (KeyError, ValueError, TypeError, IndexError):
        return None, None
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n_samples)
    samples = np.where(z < 0, R_star + z * R_lerr, R_star + z * R_uerr)
    samples = np.clip(samples, 1e-4, None)  # guard against negatives
    return samples, R_star


def plot_chromatic_rr_histogram(posterior_samples):
    '''
    Overlay posterior histograms of per-bandpass Rp/Rs for chromatic fits.

    Only runs when settings.csv defines ``bandpass`` with >=2 unique labels
    (``config.BASEMENT.settings['chromatic'] is True``). One PDF is written
    per photometric companion at
    ``<outdir>/ns_chromatic_rr_<companion>.pdf``.

    When ``params_star.csv`` is present in the datadir, a second panel is
    added below showing the implied planet radius posterior with twin
    x-axes: the bottom axis in Earth radii and the top axis in Jupiter
    radii (sharing the same data). R_star uncertainty is propagated by
    sampling from the asymmetric normal described in params_star.csv.

    Inputs
    ------
    posterior_samples : 2d ndarray
        Weighted posterior samples with shape (Nsamples, ndim) matching
        ``config.BASEMENT.fitkeys`` ordering.
    '''
    if not config.BASEMENT.settings.get('chromatic', False):
        return

    bandpass_map = config.BASEMENT.settings.get('bandpass', {}) or {}
    unique_bandpasses = sorted(set(bandpass_map.values()))
    if len(unique_bandpasses) < 2:
        return

    fitkeys = list(config.BASEMENT.fitkeys)

    # Solar / planetary radius constants (CODATA-consistent with astropy).
    # Local constants keep this function importable without astropy at top
    # level; matches the conversions used in deriver.py:329-330.
    R_SUN_KM = 6.957e5
    R_EARTH_KM = 6.3781e3
    R_JUP_KM = 7.1492e4
    SUN_TO_EARTH = R_SUN_KM / R_EARTH_KM   # ~109.08
    EARTH_TO_JUP = R_EARTH_KM / R_JUP_KM   # ~0.0892

    for companion in config.BASEMENT.settings['companions_phot']:
        per_band = []
        for bp in unique_bandpasses:
            key = '{}_rr_{}'.format(companion, bp)
            if key in fitkeys:
                ind = fitkeys.index(key)
                per_band.append((bp, posterior_samples[:, ind]))

        if len(per_band) < 2:
            continue

        # Decide layout: 2 panels if params_star.csv is available, else 1.
        n_post = len(per_band[0][1])
        R_star_samples, R_star_med = _load_R_star_samples(n_post)
        two_panel = R_star_samples is not None

        # Canonical per-bandpass color map; unknown bandpasses fall back to
        # the viridis ramp so the plot still works for arbitrary labels.
        color_map = {'tess': 'k', 'g': 'C0', 'r': 'C2', 'i': 'C8', 'z': 'C3'}
        fallback = plt.cm.viridis(np.linspace(0.15, 0.85, len(per_band)))
        colors = [
            color_map.get(bp, fallback[i]) for i, (bp, _) in enumerate(per_band)
        ]

        if two_panel:
            fig, (ax, ax_r) = plt.subplots(2, 1, figsize=(7, 9))
        else:
            fig, ax = plt.subplots(figsize=(7, 5))
            ax_r = None

        # Top panel: Rp/Rs histograms (unchanged).
        for (bp, samples), color in zip(per_band, colors):
            med = np.median(samples)
            lo, hi = np.percentile(samples, [16, 84])
            label = '{}: ${:.4f}^{{+{:.4f}}}_{{-{:.4f}}}$'.format(
                bp, med, hi - med, med - lo)
            ax.hist(samples, bins=40, density=True, alpha=0.5,
                    color=color, label=label, histtype='stepfilled',
                    edgecolor=color, linewidth=1.2)
            ax.axvline(med, color=color, linestyle='--',
                       linewidth=1.0, alpha=0.9)

        ax.set_xlabel(r'$R_p / R_\star$ (companion ' + companion + ')')
        ax.set_ylabel('Posterior density')
        ax.set_title('Chromatic transit depth posterior (companion '
                     + companion + ')')
        ax.legend(loc='best', fontsize=10)

        # Bottom panel: implied R_p in Earth radii (bottom x-axis) with a
        # twin top x-axis in Jupiter radii. Only when R_star is available.
        if two_panel:
            for (bp, rr_samples), color in zip(per_band, colors):
                # Propagate R_star uncertainty by drawing fresh R_star
                # samples per band so the two posteriors are independent
                # (they share the same R_star prior, so any correlated
                # offset would be a stellar-systematic, not band).
                R_p_earth = R_star_samples * rr_samples * SUN_TO_EARTH
                med = np.median(R_p_earth)
                lo, hi = np.percentile(R_p_earth, [16, 84])
                label = '{}: ${:.2f}^{{+{:.2f}}}_{{-{:.2f}}}\\,R_\\oplus$'.format(
                    bp, med, hi - med, med - lo)
                ax_r.hist(R_p_earth, bins=40, density=True, alpha=0.5,
                          color=color, label=label, histtype='stepfilled',
                          edgecolor=color, linewidth=1.2)
                ax_r.axvline(med, color=color, linestyle='--',
                             linewidth=1.0, alpha=0.9)

            ax_r.set_xlabel(
                r'$R_p$ ($R_\oplus$)  '
                + r'(assuming $R_\star = {:.2f}\,R_\odot$)'.format(
                    R_star_med)
            )
            ax_r.set_ylabel('Posterior density')
            ax_r.legend(loc='best', fontsize=10)

            # Twin top x-axis in Jupiter radii.
            ax_r_jup = ax_r.secondary_xaxis(
                'top',
                functions=(lambda r: r * EARTH_TO_JUP,
                           lambda r: r / EARTH_TO_JUP),
            )
            ax_r_jup.set_xlabel(r'$R_p$ ($R_\mathrm{Jup}$)')

        fig.tight_layout()
        fig.savefig(os.path.join(config.BASEMENT.outdir,
                                 'ns_chromatic_rr_' + companion + '.pdf'),
                    bbox_inches='tight')
        plt.close(fig)


###############################################################################
#::: analyse the output from save_ns.pickle file
###############################################################################
def ns_output(datadir, backend=None):
    '''
    Inputs:
    -------
    datadir : str
        the working directory for allesfitter
        must contain all the data files
        output directories and files will also be created inside datadir
    backend : {'dynesty', 'ultranest', None}, optional
        Which sampler produced the saved results. If ``None`` (default),
        auto-detect from the ``backend`` key on the unified-schema results
        dict; legacy dynesty pickles (no ``backend`` key) are treated as
        ``'dynesty'``. Affects only the trace plot (which uses
        ``dynesty.plotting.traceplot`` when available); the corner plot
        and downstream analysis are backend-agnostic.

    Outputs:
    --------
    This will output information into the console, and create a output files
    into datadir/results/ (or datadir/QL/ if QL==True)
    '''
    config.init(datadir)
    
    
    #::: security check
    if os.path.exists(os.path.join(config.BASEMENT.outdir,'ns_table.csv')):
        try:
            overwrite = str(input('Nested Sampling output files already exists in '+config.BASEMENT.outdir+'\n'+\
                                  '-----------------------------------------------'+''.join(['-']*len(config.BASEMENT.outdir))+'\n'\
                                  'What do you want to do?\n'+\
                                  '1 : overwrite the output files\n'+\
                                  '2 : abort\n'))
            if (overwrite == '1'):
                print('\n')
                pass
            else:
                raise ValueError('User aborted operation.')
        except EOFError:
            warnings.warn("Nested Sampling output files already existed from a previous run, and were automatically overwritten.")
            pass
    
    
    #::: load the save_ns.pickle
    f = gzip.GzipFile(os.path.join(config.BASEMENT.outdir,'save_ns.pickle.gz'), 'rb')
    results = pickle.load(f)
    f.close()

    #::: resolve which sampler produced this file
    #    - new unified-schema files carry results['backend']
    #    - legacy dynesty pickles do not; default to 'dynesty'
    try:
        detected = results.get('backend') if isinstance(results, dict) else None
    except Exception:
        detected = None
    resolved_backend = (detected or backend or 'dynesty').lower()
           
    
    #::: plot the fit
    posterior_samples_for_plot = draw_ns_posterior_samples(results, Nsamples=20) #only 20 samples for plotting
    
    for companion in config.BASEMENT.settings['companions_all']:
        fig, axes = afplot(posterior_samples_for_plot, companion)
        if fig is not None:
            fig.savefig( os.path.join(config.BASEMENT.outdir,'ns_fit_'+companion+'.pdf'), bbox_inches='tight' )       
            plt.close(fig)

    for companion in config.BASEMENT.settings['companions_phot']:
        for inst in config.BASEMENT.settings['inst_phot']:
            first_transit = 0
            while (first_transit >= 0):
                try:
                    fig, axes, last_transit, total_transits = afplot_per_transit(posterior_samples_for_plot, inst, companion, kwargs_dict={'first_transit':first_transit})
                    fig.savefig( os.path.join(config.BASEMENT.outdir,'ns_fit_per_transit_'+inst+'_'+companion+'_' + str(last_transit) + 'th.pdf'), bbox_inches='tight' )
                    plt.close(fig)
                    if total_transits > 0 and last_transit < total_transits - 1:
                        first_transit = last_transit
                    else:
                        first_transit = -1
                except:
                    first_transit = -1
                    pass
            
    
    
    #::: retrieve the results
    posterior_samples = draw_ns_posterior_samples(results)                               # all weighted posterior_samples
    params_median, params_ll, params_ul = get_params_from_samples(posterior_samples)     # params drawn form these posterior_samples


    #::: chromatic Rp/Rs posterior histograms (no-op when achromatic)
    try:
        plot_chromatic_rr_histogram(posterior_samples)
    except Exception as _e:
        logprint('\n! WARNING: chromatic Rp/Rs histogram could not be produced: ' + str(_e))
    
    
    #::: output the results
    logprint('\nResults:')
    logprint('----------')
#    print(results.summary())
    logZ = float(np.asarray(results['logz'])[-1])                                        # value of logZ
    logZerr = float(np.asarray(results['logzerr'])[-1])                                  # uncertainty on logZ
    logprint('Backend: {}'.format(resolved_backend))
    logprint('log(Z) = {} +- {}'.format(logZ, logZerr))
    logprint('Nr. of posterior samples: {}'.format(len(posterior_samples)))
    
    
    #::: make pretty titles for the plots  
    labels, units = [], []
    for i,l in enumerate(config.BASEMENT.fitlabels):
        labels.append( str(config.BASEMENT.fitlabels[i]) )
        units.append( str(config.BASEMENT.fitunits[i]) )
        
    results2 = deepcopy(results) # results.copy() does not work anymore since dynesty 1.2                 
    params_median2, params_ll2, params_ul2 = params_median.copy(), params_ll.copy(), params_ul.copy()     # params drawn form these posterior_samples; only needed for plots (subtract epoch offset)  
    fittruths2 = config.BASEMENT.fittruths.copy()
    for companion in config.BASEMENT.settings['companions_all']:
        
        if companion+'_epoch' in config.BASEMENT.fitkeys:
            ind = np.where(config.BASEMENT.fitkeys==companion+'_epoch')[0][0]
            results2['samples'][:,ind] -= int(params_median[companion+'_epoch'])                #np.round(params_median[companion+'_epoch'],decimals=0)
            units[ind] = str(units[ind]+'-'+str(int(params_median[companion+'_epoch']))+'d')    #np.format_float_positional(params_median[companion+'_epoch'],0)+'d')
            fittruths2[ind] -= int(params_median[companion+'_epoch'])
            params_median2[companion+'_epoch'] -= int(params_median[companion+'_epoch'])
                

    for i,l in enumerate(labels):
        if len( units[i].strip(' ') ) > 0:
            labels[i] = str(labels[i]+' ('+units[i]+')')
        
        
    #::: traceplot
    # dyplot.traceplot needs the dynesty-native fields ('logvol', 'niter', ...).
    # New unified-schema saves carry them through; legacy saves and non-dynesty
    # backends do not — fall back to the minimal in-house traceplot.
    cmap = truncate_colormap( 'Greys', minval=0.2, maxval=0.8, n=256 )
    _dyplot_keys = ('logvol', 'niter', 'logl', 'nlive')
    _can_dyplot = (
        resolved_backend == 'dynesty'
        and all(k in results2 for k in _dyplot_keys)
    )
    if _can_dyplot:
        try:
            tfig, taxes = dyplot.traceplot(results2, labels=labels, quantiles=[0.16, 0.5, 0.84], truths=fittruths2, post_color='grey', trace_cmap=[cmap]*config.BASEMENT.ndim, trace_kwargs={'rasterized':True})
        except (KeyError, ValueError) as _exc:
            logprint('! dyplot.traceplot failed ({}); using fallback.'.format(_exc))
            tfig, taxes = _simple_traceplot(results2, labels=labels, truths=fittruths2)
    else:
        if resolved_backend == 'dynesty':
            logprint('! Legacy save_ns.pickle.gz missing dynesty fields {} — '
                     'using fallback traceplot. Re-run ns_fit to restore '
                     'native dynesty trace plots.'.format(list(_dyplot_keys)))
        tfig, taxes = _simple_traceplot(results2, labels=labels, truths=fittruths2)
    plt.tight_layout()


    #::: cornerplot
    # ndim = results2['samples'].shape[1]
    fontsize = np.min(( 24. + 0.5*config.BASEMENT.ndim, 40 ))
    if _can_dyplot:
        try:
            cfig, caxes = dyplot.cornerplot(results2, labels=labels, span=[0.997 for i in range(config.BASEMENT.ndim)], quantiles=[0.16, 0.5, 0.84], truths=fittruths2, hist_kwargs={'alpha':0.25,'linewidth':0,'histtype':'stepfilled'},
                                            label_kwargs={"fontsize":fontsize, "rotation":45, "horizontalalignment":'right'})
        except Exception as _exc:
            logprint('! dyplot.cornerplot failed ({}); using corner.corner fallback.'.format(_exc))
            cfig, caxes = _corner_from_results(results2, labels=labels, truths=fittruths2, fontsize=fontsize, ndim=config.BASEMENT.ndim)
    else:
        cfig, caxes = _corner_from_results(results2, labels=labels, truths=fittruths2, fontsize=fontsize, ndim=config.BASEMENT.ndim)
        
        
    #::: runplot
#    rfig, raxes = dyplot.runplot(results)
#    rfig.savefig( os.path.join(config.BASEMENT.outdir,'ns_run.jpg'), dpi=100, bbox_inches='tight' )
#    plt.close(rfig)
    
    
    #::: set allesfitter titles and labels
    for i, key in enumerate(config.BASEMENT.fitkeys): 
        
        value = round_tex(params_median2[key], params_ll2[key], params_ul2[key])
        ctitle = r'' + labels[i] + '\n' + r'$=' + value + '$'
        ttitle = r'' + labels[i] + r'$=' + value + '$'
        if len(config.BASEMENT.fitkeys)>1:
            # caxes[i,i].set_title(ctitle)
            caxes[i,i].set_title(ctitle, fontsize=fontsize, rotation=45, horizontalalignment='left')
            taxes[i,1].set_title(ttitle)
            for i in range(caxes.shape[0]):
                for j in range(caxes.shape[1]):
                    caxes[i,j].xaxis.set_label_coords(0.5, -0.5)
                    caxes[i,j].yaxis.set_label_coords(-0.5, 0.5)
        
                    if i==(caxes.shape[0]-1): 
                        fmt = ScalarFormatter(useOffset=False)
                        caxes[i,j].xaxis.set_major_locator(MaxNLocator(nbins=3))
                        caxes[i,j].xaxis.set_major_formatter(fmt)
                    if (i>0) and (j==0):
                        fmt = ScalarFormatter(useOffset=False)
                        caxes[i,j].yaxis.set_major_locator(MaxNLocator(nbins=3))
                        caxes[i,j].yaxis.set_major_formatter(fmt)
                        
                    #for tick in caxes[i,j].xaxis.get_major_ticks(): tick.label.set_fontsize(24) 
                    #for tick in caxes[i,j].yaxis.get_major_ticks(): tick.label.set_fontsize(24)    
        else:
            caxes[i,i].set_title(ctitle)
            taxes[1].set_title(ttitle)
            caxes[i,i].xaxis.set_label_coords(0.5, -0.5)
            caxes[i,i].yaxis.set_label_coords(-0.5, 0.5)
               
            
    #::: save and close the trace- and cornerplot
    tfig.savefig( os.path.join(config.BASEMENT.outdir,'ns_trace.pdf'), bbox_inches='tight' )
    plt.close(tfig)
    cfig.savefig( os.path.join(config.BASEMENT.outdir,'ns_corner.pdf'), bbox_inches='tight' )
    plt.close(cfig)


    #::: save the tables
    save_table(posterior_samples, 'ns')
    save_latex_table(posterior_samples, 'ns')
    

    #::: derive values (using stellar parameters from params_star.csv)
    deriver.derive(posterior_samples, 'ns')
    
    
    #::: check the residuals
    for inst in config.BASEMENT.settings['inst_all']:
        if inst in config.BASEMENT.settings['inst_phot']: key='flux'
        elif inst in config.BASEMENT.settings['inst_rv']: key='rv'
        elif inst in config.BASEMENT.settings['inst_rv2']: key='rv2'
        model = calculate_model(params_median, inst, key)
        baseline = calculate_baseline(params_median, inst, key)
        stellar_var = calculate_stellar_var(params_median, inst, key)
        residuals = config.BASEMENT.data[inst][key] - model - baseline - stellar_var
        residual_stats(residuals)
    
    
    #::: make top-down orbit plot (using stellar parameters from params_star.csv)
    try:
        params_star = np.genfromtxt( os.path.join(config.BASEMENT.datadir,'params_star.csv'), delimiter=',', names=True, dtype=None, encoding='utf-8', comments='#' )
        fig, ax = plot_top_down_view(params_median, params_star)
        fig.savefig( os.path.join(config.BASEMENT.outdir,'top_down_view.pdf'), bbox_inches='tight' )
        plt.close(fig)        
    except:
        logprint('\nOrbital plots could not be produced.')
    
    
    #::: plot TTV results (if wished for)
    if config.BASEMENT.settings['fit_ttvs'] == True:
        plot_ttv_results(params_median, params_ll, params_ul)
    
    
    #::: clean up
    logprint('\nDone. For all outputs, see', config.BASEMENT.outdir)
    
    
    #::: return a nerdy quote
    try:
        with open(os.path.join(os.path.dirname(__file__), 'utils', 'quotes.txt')) as dataset:
            return(np.random.choice([l for l in dataset]))
    except:
        return('42')
    
    

def ns_derive(datadir): #emergency function if matplotlib and Mac OSX crash
    posterior_samples = get_ns_posterior_samples(datadir, as_type='2d_array')
    deriver.derive(posterior_samples, 'ns')
    
    
    
###############################################################################
#::: get NS samples (for top-level user)
###############################################################################
def get_ns_posterior_samples(datadir, Nsamples=None, as_type='dic'):
    config.init(datadir)
    
    try:
        f = gzip.GzipFile(os.path.join(datadir,'results','save_ns.pickle.gz'), 'rb')
        results = pickle.load(f)
        f.close()
        
    except:
        with open(os.path.join(datadir,'results','save_ns.pickle'),'rb') as f:
            results = pickle.load(f)    

    return draw_ns_posterior_samples(results, Nsamples=Nsamples, as_type=as_type)
    
    
 
###############################################################################
#::: get NS params (for top-level user)
###############################################################################   
def get_ns_params(datadir):
    posterior_samples = get_ns_posterior_samples(datadir, Nsamples=None, as_type='2d_array')  # all weighted posterior_samples
    params_median, params_ll, params_ul = get_params_from_samples(posterior_samples)     # params drawn form these posterior_samples
    return params_median, params_ll, params_ul
