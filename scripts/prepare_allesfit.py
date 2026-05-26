#!/usr/bin/env python
"""
Usage
$python prepare_allesfit.py -toi 1097 -s all -e 120
$python prepare_allesfit.py -tic 273586149 -s -1 -p qlp
$python prepare_allesfit.py -name 'HIP 67522' -o -i --debug

Uses parameter from TOI/CTIO/NExSci databse and
creates a directory with the files needed to run allesfitter:
1. params.csv
2. settings.csv
3. run.py
4. params_star.csv
5. mission.csv e.g. tess.csv, k2.csv, kepler.csv
======
* for precise transit transit timing, some parameters can be fixed
* limb darkening can be fixed to theoretical values derived using ~ldtk~ limbdark;
  assumes feh=(0,0.1) dex if feh is not available
* uses tess-point to check if target was observed by TESS
(useful to know even if `lightkurve.search_lightcurve` returned None)
* uses aliases (K2 name --> EPIC)
https://exoplanetarchive.ipac.caltech.edu/docs/sysaliases.html
======
"""
import sys
from typing import Tuple
from argparse import ArgumentParser
from pathlib import Path
from math import ceil
import numpy as np
import lightkurve as lk
import astropy.units as u
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import anderson
from astropy.coordinates import SkyCoord
from allesfitter import allesclass  # , config, nested_sampling_output, general_output
from loguru import logger
# from ldtk import LDPSetCreator, BoxcarFilter
from tess_stars2px import tess_stars2px_function_entry
from allesfitter.utils.scripting import (
    catalog_info_TIC,
    get_tfop_info,
    get_tois,
    get_ctois,
    rho_from_mr,
    as_from_rhop,
    a_from_rhoprs,
    get_nexsci,
    get_name_aliases,
    get_tdur,
    get_rsuma,
)
from allesfitter.exoworlds_rdx.lightcurves.index_transits import (
    get_tmid_observed_transits,
)
logger.remove() 
log_format = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{message}</level>"
logger.add(sys.stderr, format=log_format)
try:
    import limbdark as ld
except Exception as e:
    logger.error(f"Error: {e}")
    command = (
        "pip install git+https://github.com/john-livingston/limbdark.git#egg=limbdark"
    )
    raise ModuleNotFoundError(command)

assert lk.__version__[0] == "2"

filter_widths = {
    "gp": (400, 550),
    "V": (480, 600),
    "rp": (560, 700),
    "ip": (700, 820),
    "zs": (825, 920),
    "I+z": (720, 1030),
    "tess": (585, 1050),
}

#home = Path.home()
#sys.path.insert(0, f"{home}/github/research/project/young_ttvs/code")

cols = ["time", "flux", "flux_err"]

# Preset t_exp values (days) keyed by the instrument label used in settings.csv.
# When a --filename entry matches one of these keys, the t_exp_{fn} row is
# written with the preset value instead of being recomputed from --exp.
# Values are seconds / 86400.
T_EXP_DAYS = {
    120:   0.001389,   # 120 s
    200:   0.002312,   # 200 s
    600:   0.006944,   # 600 s
    1800:  0.020833,   # 1800 s    
}

Nsamples = 10_000
planets = "b c d e f g h i j k".split()
quartiles_1sig = [16.0, 50.0, 84.0]  # 1-sigma
quartiles_2sig = [2.275, 50.0, 97.725]  # 2-sigma
quartiles_3sig = [0.135, 50.0, 99.865]  # 3-sigma

def catalog_info_name(df) -> Tuple:
    Teff, Teff_err = df["st_teff"].astype(float), np.sqrt(
        df["st_tefferr1"] ** 2 + df["st_tefferr2"] ** 2
    )
    logg, logg_err = df["st_logg"].astype(float), np.sqrt(
        df["st_loggerr1"] ** 2 + df["st_loggerr2"] ** 2
    )
    feh, feh_err = 0, 0.1
    radius, radius_err = df["st_rad"].astype(float), np.sqrt(
        df["st_raderr1"] ** 2 + df["st_raderr2"] ** 2
    )
    mass, mass_err = df["st_mass"].astype(float), np.sqrt(
        df["st_masserr1"] ** 2 + df["st_masserr2"] ** 2
    )
    return (
        Teff,
        Teff_err,
        logg,
        logg_err,
        feh,
        feh_err,
        radius,
        radius_err,
        mass,
        mass_err,
    )


def parse_target_name(
    toiid=None, ticid=None, ctoiid=None, name=None, update_db=False
) -> Tuple:
    if toiid:
        df = get_tois(clobber=update_db)
        logger.info("Using parameters from TOI database (use --update_db to update).")
        logger.info(f"To use published parameters in NExSci, use -name=TOI-{toiid}")
        key = "TOI"
        id = str(toiid)
        idx = df[key].apply(lambda x: str(x).split(".")[0] == id)
        source = "tfop"
        target_name = f"TOI-{id.zfill(4)}"
    if ticid:
        logger.info("Using parameters from TIC catalog (use --update_db to update).")
        key = "TIC"
        id = str(ticid)
        target_name = f"TIC-{ticid}"
        source = "custom"
        Porb = float(input("Porb (d): "))
        Porberr = float(input("Porb err (d): "))
        epoch = float(input("Epoch (BJD): "))
        epocherr = float(input("Epoch err (BJD): "))
        tdur = float(input("Tdur (h): "))
        tdurerr = float(input("Tdur err (h): "))
        depth = float(input("Depth (ppm): "))
        deptherr = float(input("Depth err (ppm): "))
        cols = [
            "Period (days)",
            "Period (days) err",
            "Epoch (BJD)",
            "Epoch (BJD) err",
            "Duration (hours)",
            "Duration (hours) err",
            "Depth (ppm)",
            "Depth (ppm) err",
        ]
        df = pd.DataFrame(
            [[Porb, Porberr, epoch, epocherr, tdur, tdurerr, depth, deptherr]],
            columns=cols,
        )
        idx = [True]
    if ctoiid:
        logger.info("Using parameters from CTOI database (use --update_db to update).")
        df = get_ctois(clobber=update_db)
        key = "CTOI"
        id = ctoiid
        idx = df["TIC ID"] == int(ctoiid)
        target_name = f"CTOI-{ctoiid}"
        source = "ctoi"
    if name:
        logger.info("Using parameters from NExSci database (use --update_db to update).")
        df = get_nexsci("pscomppars", clobber=update_db)
        # df = df[df['default_flag']==1]
        df["Period (days)"] = df["pl_orbper"].astype(float)
        df["Period (days) err"] = np.sqrt(
            df["pl_orbpererr1"] ** 2 + df["pl_orbpererr2"] ** 2
        )
        df["Epoch (BJD)"] = df["pl_tranmid"].astype(float)
        df["Epoch (BJD) err"] = 0.1
        df["Depth (ppm)"] = df["pl_trandep"].astype(float) / 100 * 1e6  # % to ppm
        df["Depth (ppm) err"] = 1_000
        df["Duration (hours)"] = df["pl_trandur"].astype(float)
        df["Duration (hours) err"] = np.sqrt(
            df["pl_trandurerr1"].astype(float) ** 2
            + df["pl_trandurerr2"].astype(float) ** 2
        )
        key = "hostname"
        id = name.lower()
        target_name = name.strip().replace(" ", "")

        def _norm(s):
            return str(s).lower().replace(" ", "").replace("-", "").replace("_", "")

        hostnames_norm = df[key].astype(str).map(_norm)
        idx = hostnames_norm == _norm(name)
        if sum(idx) == 0:
            # Fall back to NExSci alias lookup: the hostname in pscomppars is
            # often a TIC/2MASS/HD identifier rather than the TOI/K2 alias the
            # user supplied. Resolve all known aliases and try each one.
            try:
                logger.info(
                    f"Hostname '{name}' not found directly; resolving NExSci aliases..."
                )
                aliases = get_name_aliases(name)
                for alias in aliases:
                    match = hostnames_norm == _norm(alias)
                    if match.any():
                        logger.info(f"Matched NExSci hostname via alias: {alias}")
                        idx = match
                        break
            except Exception as e:
                logger.error(f"Alias lookup failed: {e}")
        source = "nexsci"
    msg = f"Coulnd't find {key} {id} in {key} database."
    assert sum(idx) > 0, logger.error(msg)
    return target_name, df[idx].reset_index(drop=True), source


def get_tess_sectors(
    target_name: str, df: pd.DataFrame, toiid=None, ctoiid=None, name=None
) -> Tuple:
    if toiid or ctoiid:
        # if target is available in TOI,CTOI,or NexSci
        coord = SkyCoord(*df[["RA", "Dec"]].values[0], unit=("hourangle", "deg"))
        ra, dec = coord.ra.deg, coord.dec.deg
        ticid = df["TIC ID"].unique()[0]
    elif name:
        # for other targets
        data_json = get_tfop_info(target_name)
        ra = float(data_json["coordinates"]["ra"])
        dec = float(data_json["coordinates"]["dec"])
        ticid = int(data_json["basic_info"]["tic_id"])
    else:
        msg = "Set toiid, ctoiid, or name."
        logger.error(msg); sys.exit()
    try:
        (
            outID,
            outEclipLong,
            outEclipLat,
            outSec,
            outCam,
            outCcd,
            outColPix,
            outRowPix,
            scinfo,
        ) = tess_stars2px_function_entry(ticid, ra, dec)
    except Exception as e:
        logger.error(f"Error: {e}")
    return ticid, outSec


def check_if_sector_is_available(
    target_name: str, given_sector, all_sectors: list
) -> str:
    """
    All cases for given_sector=(None, 0, 1, 'all', [1,2], -1)
    Check only if given_sector is non-negative int or list
    """
    if given_sector is None:
        return "default"
    else:
        assert isinstance(given_sector, list)
        assert isinstance(all_sectors, np.ndarray)
        if len(given_sector) == 1:
            if given_sector == ["all"]:
                return "all_sector"
            elif given_sector == ["-1"]:
                return "last"
            elif given_sector == ["0"]:
                return "first"
        # check if given_sector exists if not 'all','0',or '-1'
        idx = np.array([True if int(s) in all_sectors else False for s in given_sector])
        msg = (
            f"{target_name} was not observed in sector={np.array(given_sector)[~idx]}\n"
        )
        msg += f"Try sector={all_sectors}."
        assert np.all(idx), logger.error(msg)
        return "multi_sector"


def main():
    ap = ArgumentParser()
    group1 = ap.add_mutually_exclusive_group(required=True)
    group1.add_argument("-toi", help="TOI ID", type=int)
    group1.add_argument("-ctoi", help="CTOI ID", type=int)
    group1.add_argument("-tic", help="TIC ID", type=int)
    group1.add_argument("-name", help="Name", type=str)
    group2 = ap.add_mutually_exclusive_group(required=True)
    group2.add_argument(
        "-s",
        "--sector",
        nargs="+",
        help="sector=-1 uses most recent TESS sector (default); try -sector=all to use all",
        default=None,
    )
    group2.add_argument(
        "-c",
        "--campaign",
        help="campaign=-1 uses most recent K2 campaign (default); try -campaign=all to use all",
        default=None,
    )
    group2.add_argument(
        "-q",
        "--quarter",
        help="quarter=-1 uses most recent Kepler quarter (default); try -quarter=all to use all",
        default=None,
    )
    # ap.add_argument("-s", "--sector", "--sector", nargs='+',
    #                 help="-sector=-1 uses most recent TESS sector (default); try -sector=all to use all",
    #                 default=None)
    ap.add_argument(
        "-e", "--exptime", help="exposure time (default=None)", type=float, default=None
    )
    ap.add_argument(
        "-p", "--pipeline", help="TESS/Kepler data pipeline (default='spoc')", type=str, default="spoc"
    )
    ap.add_argument(
        "-f", "--filename",
        help="filename(s) of lightcurve used as inst_phot (default='tess'). "
             "Accepts multiple, e.g. -f kepler tess",
        type=str, nargs="+", default=["tess"],
    )
    ap.add_argument(
        "-m", "--mission", help="satellite mission (default='tess')", type=str, default="tess", choices=["tess", "k2", "kepler"]
    )
    ap.add_argument(
        "-lc",
        "--lc_type",
        help="type of light curve (default=pdcsap)",
        choices=["pdcsap", "sap"],
        type=str,
        default="pdcsap",
    )
    ap.add_argument(
        "-sig",
        "--sigma",
        help="sigma for removing outliers in (combined) TESS lc",
        type=float,
        default=None,
    )
    ap.add_argument(
        "-qb",
        "--quality",
        choices=["none", "default", "hard", "hardest"],
        type=str,
        default="default",
    )
    ap.add_argument("-dir", help="base directory", type=str, default=".")
    ap.add_argument(
        "-i",
        "--interactive",
        help="manually input missing values (default=False)",
        action="store_true",
        default=False,
    )
    ap.add_argument(
        "-u",
        "--update_db",
        help="update TOI or NExSci database (default=False)",
        action="store_true",
        default=False,
    )
    ap.add_argument(
        "-r",
        "--results_dir",
        help="path to the results dir of a previous run to be used in params.csv",
        default=None,
    )
    ap.add_argument(
        "-o",
        "--overwrite",
        help="overwrite files (default=False)",
        action="store_true",
        default=False,
    )
    ap.add_argument("--debug", action="store_true", default=False)
    ap.add_argument(
        "--lc-only",
        help="only download and save lightcurve, skip generating config files",
        action="store_true",
        default=False,
    )
    ap.add_argument(
        "--ttv",
        help=(
            "emit per-transit TTV parameters in params.csv (or params2.csv "
            "when --results_dir is given). The number of rows per planet "
            "equals the count of transits with actual data coverage, "
            "computed via get_tmid_observed_transits. The index N in "
            "{pl}_ttv_transit_N is the N-th chronologically observed transit."
        ),
        action="store_true",
        default=False,
    )

    args = ap.parse_args(None if sys.argv[1:] else ["-h"])

    basedir = args.dir
    outdir = Path(basedir)
    results_dir = args.results_dir
    debug = args.debug

    if args.lc_only:
        # Minimal path: just download lightcurve
        assert args.sector is not None, "Sector required for --lc-only mode"
        assert any([args.tic, args.toi, args.ctoi, args.name]), (
            "One of -tic/-toi/-ctoi/-name required for --lc-only mode"
        )

        mission = args.mission.lower()
        ticid = args.tic
        pipeline = args.pipeline
        sector = args.sector
        exptime = args.exptime
        lc_type = "sap_flux" if pipeline == "qlp" else args.lc_type + "_flux"
        quality_bitmask = args.quality

        if args.tic:
            query_name = f"TIC {ticid}"
            label = f"TIC-{ticid}"
        elif args.toi:
            query_name = f"TOI {args.toi}"
            label = f"TOI-{str(args.toi).zfill(4)}"
        elif args.ctoi:
            query_name = f"CTOI {args.ctoi}"
            label = f"CTOI-{args.ctoi}"
        else:
            query_name = args.name
            label = args.name.strip().replace(" ", "")
        
        result = lk.search_lightcurve(
            query_name, author=pipeline, exptime=exptime, mission=mission
        )
        
        if not result:
            logger.error("No lightcurve found. Check inputs.")
            sys.exit()
        
        sectors = list(map(int, [s.split()[-1] for s in result.mission]))
        unique_sectors = sorted(set(sectors))
        
        # Handle sector flag (args.sector is a list like ["all"] or [10])
        if sector == ["all"]:
            sector_to_use = unique_sectors
        else:
            sector_to_use = sector if isinstance(sector, list) else [sector]
        
        idx = [str(s) in [str(x) for x in sector_to_use] for s in sectors]
        
        if sum(idx) == 0:
            msg = f"Sector {sector} not available. Available: {unique_sectors}"
            logger.error(msg)
            sys.exit()
        
        filtered_result = result[idx]
        
        unique_exptimes = filtered_result.table.to_pandas().exptime.unique()
        exptime = unique_exptimes[0] if exptime is None else exptime
        
        if len(sector_to_use) != len(filtered_result):
            msg = f"Multiple exposure times available. Use -e {unique_exptimes}"
            logger.error(msg)
            sys.exit()
        
        lc = filtered_result.download_all(
            flux_column=lc_type, quality_bitmask=quality_bitmask
        ).stitch()
        
        df = lc.to_pandas()
        if len(df) == 0:
            logger.error("Lightcurve data is empty.")
            sys.exit()
        
        # Handle time (lightkurve uses time as index; add mission BJD offset)
        _bjd_offset = {"tess": 2457000, "k2": 2454833, "kepler": 2454833}.get(mission, 2457000)
        df["time"] = df.index + _bjd_offset
        df = df.reset_index(drop=True).sort_values(by="time")

        cols = ["time", "flux", "flux_err"]
        msg = f"Somehow, `flux_err` is all NaN.\n{df[cols]}\n"
        if len(df['flux_err'].dropna(axis='index'))==0:
            df['flux_err'] = 1
            msg += "Setting flux error column = 1 (See allesfitter documentation)."
            logger.error(msg)
        df2 = df[cols].dropna(axis='index')
                
        # Build output filename
        sector_str = "_".join(map(str, sector_to_use)) if isinstance(sector_to_use, list) else str(sector_to_use)
        fn = f"{label}_{pipeline}_s{sector_str}_exp{int(exptime)}s.csv"
        fp = Path(basedir, fn)
        fp.parent.mkdir(parents=True, exist_ok=True)
        
        df2[cols].to_csv(fp, sep=",", header=False, index=False)
        logger.info(f"Saved: {fp}")
        logger.info(f"Ndata: {len(df2):,}")
        logger.info("Lightcurve saved. Exiting (--lc-only mode).")
        return

    if results_dir:
        alles = allesclass(outdir.joinpath(results_dir))
        logger.info("Updating params.csv")

        # =====Update params.csv=====
        # Iterate BASEMENT.allkeys (every row in the prior params.csv), not
        # just BASEMENT.fitkeys. Previously, fixed parameters (e.g. an LDC
        # pair pinned at a fitted value, dilution, fixed eccentricity terms)
        # were silently dropped from params2.csv — leading to None values at
        # ns_fit time and a cryptic numpy ufunc TypeError when q_to_u tried
        # to sqrt(None). Now fitted keys get the posterior-derived prior row
        # and fixed keys get a faithful `name,value,0,,label,unit,` copy.
        text = """#name,value,fit,bounds,label,unit,coupled_with\n"""
        try:
            ll = alles.posterior_params_ll.copy()      # 16th percentile
            median = alles.posterior_params_median.copy()  # 50th percentile
            ul = alles.posterior_params_ul.copy()      # 84th percentile
            _fitkeys_set = set(alles.BASEMENT.fitkeys)
            for i, name in enumerate(alles.BASEMENT.allkeys):
                label = alles.BASEMENT.labels[i]
                unit = alles.BASEMENT.units[i]
                if name in _fitkeys_set:
                    # Fitted parameter: derive prior from posterior shape.
                    norm_test = anderson(alles.posterior_params[name], dist='norm')
                    # crit vals at 15/10/5/2.5/1%; statistic < crit[0] (15%)
                    # → fail to reject normality.
                    dist = ('normal'
                            if norm_test.statistic < norm_test.critical_values[0]
                            else 'uniform')
                    if dist == 'normal':
                        l_err, mid, u_err = ll[name], median[name], ul[name]
                        sig = np.sqrt(l_err**2 + u_err**2)
                        text += (
                            f"{name},{mid:6f},1,normal {mid:6f} {sig:6f},"
                            f"{label},{unit},\n"
                        )
                        if debug:
                            logger.info(
                                f"{name}: {mid:.6f} +{u_err:.6f} -{l_err:.6f} (normal)"
                            )
                    elif dist == 'uniform':
                        l_limit, mid, u_limit = np.nanpercentile(
                            alles.posterior_params[name], q=[1, 50, 99]
                        )
                        text += (
                            f"{name},{mid:6f},1,uniform {l_limit:6f} {u_limit:6f},"
                            f"{label},{unit},\n"
                        )
                        if debug:
                            logger.info(
                                f"{name}: {l_limit:.6f} < {mid:.6f} < {u_limit:.6f} (uniform)"
                            )
                    else:
                        msg = "distribution is not uniform or normal!"
                        logger.error(msg); sys.exit()
                else:
                    # Fixed parameter: preserve from BASEMENT.params so the
                    # generated params2.csv stays a complete drop-in for
                    # params.csv. Skip None values (intentionally absent
                    # entries — e.g. LD law set to None for an instrument).
                    val = alles.BASEMENT.params.get(name)
                    if val is None:
                        if debug:
                            logger.info(f"{name}: skipping (value is None)")
                        continue
                    text += f"{name},{val},0,,{label},{unit},\n"
                    if debug:
                        logger.info(f"{name}: {val} (fixed)")
        except Exception as e:
            logger.error(f"Error: {e}")

        # Append per-transit TTV rows using the observed-transit midpoints.
        # BASEMENT only caches `{c}_tmid_observed_transits` when
        # settings['fit_ttvs']==True (basement.setup_ttv_fit). On a cache
        # miss, recompute via get_tmid_observed_transits using the per-
        # instrument time arrays and posterior-median epoch/period — same
        # approach as afplot_per_transit in general_output.py.
        if args.ttv:
            _settings = alles.BASEMENT.settings
            _data = alles.BASEMENT.data
            _fast_fit_width = float(
                _settings.get("fast_fit_width", 0.3333333333333333)
            )
            _inst_list = list(_settings.get("inst_phot", []))
            # Per-instrument span / count diagnostic — surfaces cases where
            # one instrument's data is missing, empty, or falls outside the
            # epoch+period grid the cache was built against.
            for _inst in _inst_list:
                _t_arr = _data.get(_inst, {}).get("time")
                if _t_arr is None or len(_t_arr) == 0:
                    logger.warning(
                        f"TTV: inst '{_inst}' has no time data in BASEMENT.data"
                    )
                else:
                    _t_arr = np.asarray(_t_arr, dtype=float)
                    logger.info(
                        f"TTV: inst '{_inst}' N={len(_t_arr)} "
                        f"time=[{_t_arr.min():.4f}, {_t_arr.max():.4f}] "
                        f"span={_t_arr.max() - _t_arr.min():.3f} d"
                    )
            for _c in _settings.get("companions_phot", []):
                _epoch = float(median.get(
                    f"{_c}_epoch", alles.BASEMENT.params[f"{_c}_epoch"]
                ))
                _period = float(median.get(
                    f"{_c}_period", alles.BASEMENT.params[f"{_c}_period"]
                ))
                _tmids = _data.get(f"{_c}_tmid_observed_transits")
                _used_cache = _tmids is not None and len(_tmids) > 0
                if not _used_cache:
                    _times = []
                    for _inst in _inst_list:
                        _times += list(_data[_inst]["time"])
                    _times = np.sort(np.asarray(_times, dtype=float))
                    _tmids = get_tmid_observed_transits(
                        _times, _epoch, _period, _fast_fit_width,
                    )
                    logger.info(
                        f"TTV: {_c} cache miss; recomputed via "
                        f"get_tmid_observed_transits "
                        f"(width={_fast_fit_width:.4f} d)"
                    )
                else:
                    logger.info(
                        f"TTV: {_c} using BASEMENT cache "
                        f"(populated by setup_ttv_fit; fit_ttvs={_settings.get('fit_ttvs')})"
                    )
                # Cross-check: independently recompute the transit count from
                # the union of all inst_phot times. If this disagrees with
                # the cache (which can happen if the cache was built from
                # fast-fit-reduced data of only one instrument or with a
                # stale epoch/period), prefer the fresh union count.
                _times_union = []
                for _inst in _inst_list:
                    _times_union += list(_data[_inst]["time"])
                _times_union = np.sort(np.asarray(_times_union, dtype=float))
                _tmids_union = get_tmid_observed_transits(
                    _times_union, _epoch, _period, _fast_fit_width,
                )
                if len(_tmids_union) != len(_tmids):
                    logger.warning(
                        f"TTV: {_c} cache_count={len(_tmids)} "
                        f"vs fresh union_count={len(_tmids_union)} "
                        f"(epoch={_epoch:.6f}, period={_period:.6f}, "
                        f"width={_fast_fit_width:.4f}). "
                        f"Using union count."
                    )
                    _tmids = _tmids_union
                _n_obs = len(_tmids)
                logger.info(f"TTV: {_c} has {_n_obs} transits with data")
                if _n_obs == 0:
                    continue
                text += f"#TTV companion {_c},,,,,\n"
                for _j in range(_n_obs):
                    text += (
                        f"{_c}_ttv_transit_{_j+1},0,1,uniform -0.1 0.1,"
                        f"TTV$_\\mathrm{{ttv;{_j+1}}}$,d,\n"
                    )
        if debug:
            logger.info(text)
        fp = Path(results_dir, f"params2.csv")
        np.savetxt(fp, [text], fmt="%s")
        logger.info(f"Saved: {fp}")
        sys.exit()

    else:
        toiid = args.toi
        ticid = args.tic
        ctoiid = args.ctoi
        name = args.name
        exptime = args.exptime
        mission = args.mission.lower()
        quality_bitmask = args.quality
        sigma = args.sigma
        interactive = args.interactive
        ttv = args.ttv
        fns = args.filename if isinstance(args.filename, list) else [args.filename]
        fn = fns[0]  # first instrument (used where a single filename is needed)

        # Unify the "segment" concept across missions: `sector` is always a
        # list-of-str id — TESS sector / K2 campaign / Kepler quarter.
        # --campaign and --quarter act as scalar aliases when -s is not given.
        def _as_segment_list(x):
            if x is None:
                return None
            if isinstance(x, list):
                return [str(v) for v in x]
            return [str(x)]

        if mission == "tess":
            sector = args.sector
        elif mission == "k2":
            sector = args.sector if args.sector is not None else _as_segment_list(
                args.campaign if args.campaign is not None else -1
            )
        elif mission == "kepler":
            sector = args.sector if args.sector is not None else _as_segment_list(
                args.quarter if args.quarter is not None else -1
            )
        else:
            sector = args.sector

        # Mission-appropriate BJD offset used when reconstructing full BJD.
        bjd_offset = {"tess": 2457000, "k2": 2454833, "kepler": 2454833}[mission]

        pipeline = args.pipeline
        lc_type = "sap_flux" if pipeline == "qlp" else args.lc_type + "_flux"

        overwrite = args.overwrite
        update_db = args.update_db

        target_name, target_df, source = parse_target_name(
            toiid, ticid, ctoiid, name, update_db
        )
        if ticid:
            name = target_name.replace("-", "")
        if mission == "tess":
            tic_id, outSec = get_tess_sectors(target_name, target_df, toiid, ctoiid, name)
            sector_flag = check_if_sector_is_available(
                target_name, given_sector=sector, all_sectors=outSec
            )
        else:
            # K2/Kepler: skip TESS-point lookup. Mirror the check_if_sector_is_available
            # flag logic without validating against a known segment list (lightkurve
            # will surface missing campaigns/quarters when the search returns empty).
            tic_id = None
            outSec = np.array([])
            if sector is None:
                sector_flag = "default"
            elif sector == ["all"]:
                sector_flag = "all_sector"
            elif sector == ["-1"]:
                sector_flag = "last"
            elif sector == ["0"]:
                sector_flag = "first"
            else:
                sector_flag = "multi_sector"

        fp = Path(basedir, target_name).joinpath(f"{target_name}.log")
        logger.add(fp, format=log_format)
        if debug:
            logger.info(target_df)

        try:
            if toiid or ctoiid or ticid:
                (
                    Teff,
                    Teff_err,
                    logg,
                    logg_err,
                    feh,
                    feh_err,
                    radius,
                    radius_err,
                    mass,
                    mass_err,
                ) = catalog_info_TIC(int(tic_id))
            elif name:
                (
                    Teff,
                    Teff_err,
                    logg,
                    logg_err,
                    feh,
                    feh_err,
                    radius,
                    radius_err,
                    mass,
                    mass_err,
                ) = catalog_info_name(target_df.iloc[0])
            if debug:
                logger.info(
                    Teff,
                    Teff_err,
                    logg,
                    logg_err,
                    feh,
                    feh_err,
                    radius,
                    radius_err,
                    mass,
                    mass_err,
                )
        except Exception as e:
            logger.error(f"Error: {e}")

        ticid = tic_id if ticid is None else ticid
        rhostar_prior = True
        if str(radius) == "nan":
            if interactive:
                radius = float(input("Rstar [Rsun]:"))
                rhostar_prior = False
            else:
                msg = "use --interactive to input missing value"
                logger.error(msg); sys.exit()
        if str(mass) == "nan":
            if interactive:
                mass = float(input("Mstar [Msun]:"))
                rhostar_prior = False
            else:
                msg = "use --interactive to input missing value"
                raise ValueError(msg)
        if str(radius_err) == "nan":
            radius_err = 0.1
            logger.info("Rstar err is nan; setting to 0.1")
        if str(mass_err) == "nan":
            mass_err = 0.1
            logger.info("Mstar err is nan; setting to 0.1")
        if debug:
            # logger.info(f"Teff={Teff:.0f}+/-{Teff_err:.0f},
            #     logg={logg:.2f}+/-{logg_err:.2f},
            #     feh={feh:.2f}+/-{feh_err:.2f}")
            logger.info(f"Rs={radius:.2f}+/-{radius_err:.2f}, Ms={mass:.2f}+/-{mass_err:.2f}")

        # band = mission.lower()
        if np.isnan(feh):
            feh = float(input("[Fe/H]: ")) if interactive else 0  # solar metallicity
        if np.isnan(feh_err):
            feh_err = float(input("[Fe/H] err: ")) if interactive else 0.1
        logger.info(f"Using [Fe/H]=({feh},{feh_err}) dex")
        if np.isnan(logg):
            if interactive:
                logg = float(input("logg: "))
            else:
                msg = "use --interactive to input missing value"
                logger.error(msg); sys.exit() # no assumption
        if np.isnan(logg_err):
            logg_err = float(input("logg err: ")) if interactive else 0.1
        logger.info(f"Using logg=({logg:.2f},{logg_err:.2f}) cm/s^2")
        if np.isnan(Teff):
            if interactive:
                Teff = float(input("Teff: "))
            else:
                msg = "use --interactive to input missing value"
                logger.error(msg); sys.exit() # no assumption
        if np.isnan(Teff_err):
            Teff_err = float(input("Teff err: ")) if interactive else 500
        logger.info(f"Using Teff=({Teff:.0f},{Teff_err:.0f}) K")

        q1, q1_err, q2, q2_err = ld.claret(
            band="T",
            teff=Teff,
            uteff=Teff_err,
            logg=logg,
            ulogg=logg_err,
            feh=feh,
            ufeh=feh_err,
            law="quadratic",
        )
        # ===== Write files =====#
        outdir = Path(basedir, target_name)
        try:
            outdir.mkdir(parents=True, exist_ok=overwrite)
        except FileExistsError:
            raise FileExistsError(
                f"{outdir} already exists. Use --overwrite to overwrite files."
            )

        # ===== Create params.csv =====#
        text = """#name,value,fit,bounds,label,unit,coupled_with\n"""
        for i, row in target_df.iterrows():
            pl = planets[i]
            if debug:
                logger.info(f"=====Planet {pl}=====")

            # tic = row['TIC ID']
            Porb = row["Period (days)"]
            Porberr = row["Period (days) err"]
            epoch = row["Epoch (BJD)"]
            epocherr = row["Epoch (BJD) err"]
            tdur = row["Duration (hours)"]
            tdurerr = row["Duration (hours) err"]

            if interactive and not np.all([Porb > 0, epoch > 0, tdur > 0]):
                Porb = float(input("Porb: "))
                Porberr = float(input("Porb err: "))
                epoch = float(input("Epoch: "))
                epocherr = float(input("Epoch err: "))
            # tdur = float(input("Tdur: "))
            # tdurerr = float(input("Tdur err: "))
            else:
                assert np.all([Porb > 0, epoch > 0, tdur > 0])
            Porb_samples = np.random.normal(Porb, Porberr, size=Nsamples)

            if debug:
                logger.info(f"P={Porb:.4f}+/-{Porberr:.4f}, T0={epoch:.4f}+/-{epocherr:.4f}")

            rprs = np.sqrt(row["Depth (ppm)"] / 1e6)
            rprserr = np.sqrt(row["Depth (ppm) err"] / 1e6)

            if str(rprs) == "nan" or str(rprserr) == "nan":
                if interactive:
                    logger.info(
                        f"rprs={row['Depth (ppm)']/1e3:.3f}, rprserr={row['Depth (ppm) err']/1e3:.3f} ppt"
                    )
                    try:
                        rprs = input(f"Planet {pl} Rp/Rs (ppt): ")
                        rprs = float(rprs) / 1e3
                        rprserr = input(f"Planet {pl} Rp/Rs err (ppt): ")
                        rprserr = float(rprserr) / 1e3
                    except Exception as e:
                        msg = f"Error in inputs.\n{e}"
                        logger.error(msg)
                elif hasattr(row, "pl_rade"):
                    try:
                        rprs = row["pl_rade"] * u.Rearth.to(u.Rsun) / row["st_rad"]
                        Rperr = np.sqrt(
                            row["pl_radeerr1"] ** 2 + row["pl_radeerr2"] ** 2
                        )
                        rprserr = Rperr * u.Rearth.to(u.Rsun) / radius_err
                    except Exception as e:
                        msg = f"Error in parsing rp/rs\n{e}"
                        logger.error(msg)
                else:
                    msg = "Rp/Rs is nan. Try --interactive for manual input"
                    logger.error(msg); sys.exit()
            assert rprs > 0

            rprs_samples = np.random.normal(rprs, rprserr, size=Nsamples)
            rprs_min, rprs, rprs_max = np.percentile(rprs_samples, q=quartiles_3sig)

            radius_samples = np.random.normal(radius, radius_err, size=Nsamples)
            mass_samples = np.random.normal(mass, mass_err, size=Nsamples)

            # compute a/Rs from stellar density
            rho_samples = rho_from_mr(mass_samples, radius_samples)
            a_over_Rs_samples = as_from_rhop(rho_samples, Porb_samples)
            if debug:
                rhomin, rho, rhomax = np.percentile(rho_samples, q=quartiles_3sig)
                as_min, a, as_max = np.percentile(a_over_Rs_samples, q=quartiles_3sig)
                a_au_s = a_from_rhoprs(rho_samples, Porb_samples, radius_samples)
                a_au_min, a_au, a_au_max = np.percentile(a_au_s, q=quartiles_3sig)

            # FIXME: a_over_Rs_samples produces some NaNs e.g. for Kepler-51
            rsuma_samples = 1/a_over_Rs_samples * (1+rprs_samples)
            idx = (rsuma_samples > 0) & (rsuma_samples < 1)
            rsuma_min, rsuma, rsuma_max = np.percentile(rsuma_samples[idx], q=quartiles_3sig)
            if True:
                # uniformly distributed from inc_min to 90 deg
                cosi = np.random.uniform(0, 1/min(a_over_Rs_samples), size=Nsamples)
                inc_samples = np.arccos(cosi)
            else:
                # normally distributed
                inc_samples = np.arcos(1/a_over_Rs_samples)
            inc_min, inc, inc_max = np.percentile(inc_samples, q=quartiles_3sig)
            # FIXME: compute b taking into account ecc 
            b_samples = np.random.uniform(0, 1, size=Nsamples)
            # FIXME: compute including grazing orbits
            # b_samples = np.random.uniform(0, 1+max(rprs_samples), size=Nsamples)
            b_min, b, b_max = np.percentile(b_samples, q=quartiles_3sig)

            if str(tdur) != "nan":
                # check if tdur derived from rhostar is consistent with tdur from tfop
                tdur_rhostar = get_tdur(Porb, rsuma, inc, rprs, b) * 24
                logger.info(
                    f"tdur={tdur:.1f}h ({source}) {tdur_rhostar:.1f}h (derived from rhostar)"
                )
                # check if tdur derived from orbit is consistent with tdur from tfop
                try:
                    tdurerr = 1 if str(tdurerr) == "nan" else tdurerr
                    tdur_samples = np.random.normal(tdur, tdurerr, size=Nsamples) / 24
                    rsuma_samples = get_rsuma(tdur_samples, Porb, inc_samples, rprs_samples, b_samples)
                    idx = (rsuma_samples > 0) & (rsuma_samples < 1)
                    tdur_orbit = get_tdur(Porb, np.median(rsuma_samples[idx]), inc, rprs, b) * 24
                    logger.info(
                        f"tdur={tdur:.1f}h ({source}) {tdur_orbit:.1f}h (derived from orbit)"
                    )
                    if (
                        np.nanargmin(
                            np.abs(np.array([tdur_rhostar, tdur_orbit]) - tdur)
                        )
                        == 1
                    ):
                        logger.info("Using Rstar/a derived from orbit.")
                        rsuma_min, rsuma, rsuma_max = np.percentile(
                            rsuma_samples[idx], q=quartiles_3sig
                        )
                    else:
                        logger.info("Using Rstar/a derived from rhostar.")
                except Exception as e:
                    logger.error(f"Error: {e}")

            if debug:
                logger.info(f"rsuma={rsuma:.4f}")
                logger.info(f"rprs={rprs:.4f}")
                logger.info(f"rho={rho:.4f}")
                logger.info(f"a_s={a:.4f}")
                logger.info(f"a_au={a_au:.4f}")
                logger.info(f"inc={np.rad2deg(inc):.2f}")
                logger.info(f"b={b:.2f}")
            text += f"#companion {pl} astrophysical params,,,,,,\n"
            text += f"{pl}_rr,{rprs:.4f},1,uniform 0 {ceil(rprs_max*10)/10:.4f},$R_{pl} / R_\star$,,\n"
            text += f"{pl}_rsuma,{rsuma:.4f},1,uniform {rsuma_min:.4f} {ceil(rsuma_max*10)/10:.4f},$(R_\star + R_{pl}) / a_{pl}$,,\n"
            #text += f"{pl}_rsuma,{rsuma:.4f},1,uniform 0 1,$(R_\star + R_{pl}) / a_{pl}$,,\n"
            text += f"{pl}_cosi,0,1,uniform 0 1,$\cos" + "{i_" + pl + "}$,,\n"
            text += (
                f"{pl}_epoch,{epoch:.6f},1,normal {epoch:.6f} {epocherr:.6f},$T_"
                + "{0;"
                + pl
                + "}$,BJD,\n"
            )
            text += (
                f"{pl}_period,{Porb:.6f},1,normal {Porb:.6f} {Porberr:.6f},$P_{pl}$,d,\n"
            )
            text += f"{pl}_f_c,0,0,uniform -1 1,$\sqrt{{e_{pl}}} \cos{{\omega_{pl}}}$,,\n"
            text += f"{pl}_f_s,0,0,uniform -1 1,$\sqrt{{e_{pl}}} \sin{{\omega_{pl}}}$,,\n"
        text += "#dilution per instrument,,,,,,\n"
        for inst in fns:
            text += f"dil_{inst},0,0,uniform -1 1,$D_\mathrm{{0; {inst}}}$,,\n"
        text += "#limb darkening coefficients per instrument,,,,,,\n"
        for inst in fns:
            text += (
                f"#host_ldc_q1_{inst},{q1:.2f},1,normal {q1:.2f} {q1_err:.2f},"
                + f"$q_{{1; \\mathrm{{{inst}}}}}$"
                + ",,\n"
            )
            text += (
                f"#host_ldc_q2_{inst},{q2:.2f},1,normal {q2:.2f} {q2_err:.2f},"
                + f"$q_{{2; \\mathrm{{{inst}}}}}$"
                + ",,\n"
            )
            text += f"host_ldc_q1_{inst},0.5,1,uniform 0 1,$q_{{1; \\mathrm{{{inst}}}}}$,,\n"
            text += f"host_ldc_q2_{inst},0.5,1,uniform 0 1,$q_{{2; \\mathrm{{{inst}}}}}$,,\n"

        text += "#errors per instrument,,,,,,\n"
        for inst in fns:
            text += (
                f"ln_err_flux_{inst},-6,1,uniform -10 -1,$\log{{\sigma ({inst})}}$,rel. flux,\n"
            )
        text += "#baseline per instrument,,,,,,\n"
        for inst in fns:
            text += f"#baseline_gp_offset_flux_{inst},0,1,uniform -0.1 0.1,$\mathrm{{offset ({inst})}}$,,\n"
            text += f"baseline_gp_matern32_lnsigma_flux_{inst},-5,1,uniform -15 0,$\mathrm{{gp ln \sigma ({inst})}}$,,\n"
            text += f"baseline_gp_matern32_lnrho_flux_{inst},0,1,uniform -5 10,$\mathrm{{gp ln \\rho ({inst})}}$,,\n"
        # TTV rows: when --ttv is NOT set, keep the commented-out stub for
        # reference. When --ttv IS set, the real per-transit rows are
        # appended after the lightcurve download (below) because we need
        # the observed time series to count transits with data coverage.
        if not ttv:
            for i, row in target_df.iterrows():
                pl = planets[i]
                text += f"#TTV companion {pl},,,,,\n"
                for i in range(5):
                    text += f"#{pl}_ttv_transit_{i+1},0,1,uniform -0.1 0.1,TTV$_\mathrm{{ttv;{i+1}}}$,d,\n"
        if debug:
            logger.info(text)
        fp = outdir.joinpath("params.csv")
        np.savetxt(fp, [text], fmt="%s")
        logger.info(f"Saved: {fp}")

        # =====Create params_star.csv===== #
        text3 = f"""#R_star,R_star_lerr,R_star_uerr,M_star,M_star_lerr,M_star_uerr,Teff_star,Teff_star_lerr,Teff_star_uerr
    #R_sun,R_sun,R_sun,M_sun,M_sun,M_sun,K,K,K
    {radius:.2f},{radius_err:.2f},{radius_err:.2f},{mass:.2f},{mass_err:.2f},{mass_err:.2f},{Teff:.0f},{Teff_err:.0f},{Teff_err:.0f}"""
        if debug:
            logger.info(text3)

        fp = outdir.joinpath("params_star.csv")
        np.savetxt(fp, [text3], fmt="%s")
        logger.info(f"Saved: {fp}")

        # =====Create run.py===== #
        text4 = """#!/usr/bin/env python
import allesfitter

fig = allesfitter.show_initial_guess('.')
#allesfitter.prepare_ttv_fit('.', style='tessplot')

# nested sampling
#allesfitter.ns_fit('.')
#allesfitter.ns_output('.')

# mcmc (if needed)
#allesfitter.mcmc_fit('.')
#allesfitter.mcmc_output('.')"""

        if debug:
            logger.info(text4)

        fp = outdir.joinpath("run.py")
        np.savetxt(fp, [text4], fmt="%s")
        logger.info(f"Saved: {fp}")

        query_name = name
        if toiid or ctoiid:
            query_name = f"TIC {ticid}"
        elif name.lower()[:2] == "k2":
            # search epic name or coordinates
            try:
                logger.info("Searching for EPIC name")
                query_name = get_name_aliases(name, key="epic")
            except Exception as e:
                logger.info(f"Error: {e}")

        # search all available data for reference
        all_lcs = lk.search_lightcurve(query_name, mission=mission)
        logger.info(all_lcs)  
        if len(all_lcs) > 0:
            pipelines = set([i.lower() for i in all_lcs.author])
            logger.info(f"Available Pipelines: {pipelines}")
        else:
            msg = "No light curves found."
            logger.error(msg); sys.exit()               
        unique_exptimes = all_lcs.table.to_pandas().exptime.unique()
        logger.info(f"Available Exp. times: {unique_exptimes}")
        idx = [i == pipeline.lower() for i in pipelines]
        if sum(idx) == 0:
            msg = f"pipeline={pipeline} not in {pipelines}"
            logger.error(msg); sys.exit()
        
        # search only requested data
        result = lk.search_lightcurve(
            query_name, author=pipeline, exptime=exptime, mission=mission
        )
        if result:
            sectors = list(map(int, [s.split()[-1] for s in result.mission]))
            unique_sectors = sorted(set(sectors))
            if sector_flag == "all_sector":
                # case: sector='all'
                # `sectors` mirrors `result.mission` (one row per exposure
                # time / flux column), so it has duplicates and is in search
                # order. Log the deduped+sorted view for human readability.
                logger.info(
                    f"Using {pipeline.upper()} pipeline in {len(unique_sectors)} sectors: {unique_sectors}"
                )
                unique_exptimes = result.table.to_pandas().exptime.unique()
                if len(unique_exptimes) > 1:
                    msg = f"Multiple exposure times are available for `all` sectors:\n{result}.\n"
                    msg += f"Try using -exp={unique_exptimes}"
                    logger.error(msg); sys.exit()
                exptime = unique_exptimes[0] if exptime is None else exptime
                lc = result.download_all(
                    flux_column=lc_type, quality_bitmask=quality_bitmask
                ).stitch()
                logger.info(
                    "The lightcurves were not flattened/de-trended to avoid removing transits."
                )
                _seg = getattr(lc, "sector", None) or getattr(lc, "campaign", None) or getattr(lc, "quarter", None)
                if _seg is not None:
                    assert int(_seg) == int(unique_sectors[-1])
                if pipeline == "spoc":
                    lc1 = result.download_all(
                        quality_bitmask=quality_bitmask, flux_column="pdcsap_flux"
                    ).stitch()
                    lc2 = result.download_all(
                        quality_bitmask=quality_bitmask, flux_column="sap_flux"
                    ).stitch()
            elif sector_flag == "multi_sector":
                # case: sector int or list
                idx = [str(i) in sector for i in sectors]
                msg = f"{pipeline.upper()} lightcurves for sector={sector} is not available. Try sector={unique_sectors}."
                assert sum(idx) > 0, logger.error(msg)

                filtered_result = result[idx]
                unique_exptimes = filtered_result.table.to_pandas().exptime.unique()
                msg = f"Using {pipeline.upper()} pipeline in {len(sector)} sectors: {sector} (exptime={unique_exptimes} sec).\n"
                if sector_flag != "all_sector":
                    msg += f"Otherwise use sector=({unique_sectors}, all))."
                logger.info(msg)
                if len(sector) > len(filtered_result):
                    msg = f"Not all sector={sector} have exptime={exptime} sec.\n"
                    msg = "Try to limit the sectors.\n"
                    logger.error(msg); sys.exit()
                elif len(sector) < len(filtered_result):
                    msg = f"Multiple exposure times are available for the given sector:\n{filtered_result}.\n"
                    msg += f"Try using -exp={unique_exptimes}"
                    logger.error(msg); sys.exit()
                assert len(sector) == len(filtered_result)
                exptime = unique_exptimes[0] if exptime is None else exptime
                lc = filtered_result.download_all(
                    quality_bitmask=quality_bitmask, flux_column=lc_type
                ).stitch()
                logger.info(
                    "The lightcurves were not flattened/de-trended to avoid removing transits."
                )
                msg = f"sector={lc.sector} in header not in requested sector={sector}"
                _seg = getattr(lc, "sector", None) or getattr(lc, "campaign", None) or getattr(lc, "quarter", None)
                if _seg is not None:
                    assert str(int(_seg)) in np.array(sector), logger.error(msg)
                if pipeline == "spoc":
                    lc1 = filtered_result.download_all(
                        quality_bitmask=quality_bitmask, flux_column="pdcsap_flux"
                    ).stitch()
                    lc2 = filtered_result.download_all(
                        quality_bitmask=quality_bitmask, flux_column="sap_flux"
                    ).stitch()
            else:
                if sector_flag == "first":
                    idx = 0
                    sector = sectors[idx]
                elif sector_flag == "last" or sector_flag == "default":
                    idx = -1
                    sector = sectors[idx]

                filtered_result = result[idx]
                lc = filtered_result.download(
                    quality_bitmask=quality_bitmask, flux_column=lc_type
                ).normalize()
                unique_exptimes = filtered_result.table.to_pandas().exptime.unique()
                # logger.info(f"Exp times: {unique_exptimes}")
                exptime = unique_exptimes[0] if exptime is None else exptime
                _seg = getattr(lc, "sector", None) or getattr(lc, "campaign", None) or getattr(lc, "quarter", None)
                if _seg is not None:
                    assert int(_seg) == int(sector)
                logger.info(f"Using {pipeline.upper()} pipeline in sector {sector}.")
                if pipeline == "spoc":
                    lc1 = filtered_result.download(
                        quality_bitmask=quality_bitmask, flux_column="pdcsap_flux"
                    ).normalize()
                    lc2 = filtered_result.download(
                        quality_bitmask=quality_bitmask, flux_column="sap_flux"
                    ).normalize()
            if sigma:
                nbefore = len(lc)
                lc = lc.remove_outliers(sigma=sigma)
                nafter = len(lc)
                if nbefore>nafter:
                    diff = nbefore-nafter
                    logger.info(f"Removed {diff} outliers using sigma={sigma}.")
                if pipeline == "spoc":
                    lc1 = lc1.remove_outliers(sigma=sigma)
                    lc2 = lc2.remove_outliers(sigma=sigma)
            if sector_flag == "all_sector":
                secs = "s".join(map(str, unique_sectors))
            else:
                if isinstance(sector, list):
                    secs = "s".join(map(str, sector))
                else:
                    secs = str(sector)
            if pipeline == "spoc" and len(lc1)==len(lc2):
                fig, axs = plt.subplots(2, 1, figsize=(8,6), sharex=True)
                _ = lc1.scatter(ax=axs[0], zorder=2, label="PDCSAP", c="C0")
                _ = lc2.scatter(ax=axs[0], zorder=1, label="SAP", c="C1")
                axs[0].set_title(f"Sector={secs}\nexptime={int(exptime)}s")
                (lc1-lc2).scatter(ax=axs[1], label='difference', c='k')
                fp = outdir.joinpath(
                    f"{target_name}_{mission}_{lc_type.split('_')[0]}_s{secs}_exp{int(exptime)}s"
                )
                fig.savefig(fp.with_suffix('.png'))
            else:
                ax = lc.scatter(label=pipeline)
                ax.set_title(f"Sector={secs}\nexptime={int(exptime)}s")
                fp = outdir.joinpath(
                    f"{target_name}_{mission}_{pipeline}_s{secs}_exp{int(exptime)}s"
                )
                ax.figure.savefig(fp.with_suffix('.png'))
            logger.info(f"Saved: {fp.with_suffix('.png')}")
            df = lc.to_pandas()
            msg = "Somehow, the lightcurve data is empty."
            assert len(df)>0, logger.error(msg)
            df["time"] = df.index + bjd_offset
            df = df.reset_index(drop=True).sort_values(by="time")
            cols = ["time", "flux", "flux_err"]
            msg = f"Somehow, `flux_err` is all NaN.\n{df[cols]}\n"
            if len(df['flux_err'].dropna(axis='index'))==0:
                df['flux_err'] = 1
                msg += "Setting flux error column = 1 (See allesfitter documentation)."
                logger.error(msg)
            df2 = df[cols].dropna(axis='index')
            msg = "Lightcurve is all NaN. No lightcurve downloaded."
            assert len(df2)>0, logger.error(msg)
            df2[cols].to_csv(fp.with_suffix('.csv'), sep=",", header=False, index=False)
            logger.info(f"Saved: {fp.with_suffix('.csv')}")
            # Also save under the instrument label allesfitter expects
            # (inst_phot in settings.csv). This is the file the fit and
            # allesclass load at runtime.
            inst_fp = outdir.joinpath(f"{fn}.csv")
            df2[cols].to_csv(inst_fp, sep=",", header=False, index=False)
            logger.info(f"Saved: {inst_fp}")
            logger.info(f"Ndata: {len(df):,}")
            logger.info(df[cols].head())
            if debug:
                logger.info(df.head())

            # TTV rows are appended at the very end of main() once all
            # config files and the {fn}.csv lightcurve are on disk — that
            # lets us use allesclass(outdir).BASEMENT.data to pick up the
            # same observed-transit counts the fit will compute.
        else:
            msg = "No lightcurve downloaded. Check inputs."
            logger.error(msg); sys.exit()

        # =====Create settings.csv===== #
        text2 = """#name,value
###############################################################################,
# General settings,
###############################################################################,\n"""

        text2 += f"companions_phot,{' '.join(planets[:len(target_df)])}"
        text2 += f"""
companions_rv,
inst_phot,{' '.join(fns)}
inst_rv,
time_format,BJD_TDB
#passbands,{' '.join(fns)}
###############################################################################,
# Fit performance settings,
###############################################################################,
multiprocess,True
multiprocess_cores,40
fast_fit,True
fast_fit_width,0.3333333333333333
#fast_fit_width,0.5
secondary_eclipse,False
phase_curve,False
shift_epoch,True\n"""
        for i, row in target_df.iterrows():
            pl = planets[i]
            text2 += f"inst_for_{pl}_epoch,all\n"
            text2 += f"#inst_for_{pl}_epoch,{' '.join(fns)}\n"
        text2 += f"""###############################################################################,
# MCMC settings,
###############################################################################,
mcmc_nwalkers,100
mcmc_total_steps,2000
mcmc_burn_steps,1000
mcmc_thin_by,2
###############################################################################,
# Nested Sampling settings,
###############################################################################,
ns_modus,dynamic
ns_nlive,1000
ns_bound,single
ns_sample,auto
ns_tol,100
###############################################################################,
# Limb darkening law per object and instrument,
# if 'quad' two corresponding parameter called 'ldc_q1_inst' and 'ldc_q2_inst' have to be given in params.csv,
###############################################################################,\n"""
        for inst in fns:
            text2 += f"host_ld_law_{inst},quad\n"
        text2 += """#####################################,
# Exposure interpolation settings,
#####################################,\n"""
        # Resolve exposure time in seconds. --exp is passed in seconds; if
        # absent, fall back to the median cadence (which is in days) and
        # convert. This integer value is the key for T_EXP_DAYS.
        if args.exptime is not None:
            exptime_sec = int(round(args.exptime))
        else:
            exptime_sec = int(round(float(np.median(np.diff(df2.time))) * 86400))
        exptime = exptime_sec / 86400.0  # keep the days value for downstream use
        assert exptime < 1, "exp time should be in days"
        # Prefer the preset value for this exposure time; else use the
        # computed days value directly.
        t_exp_val = T_EXP_DAYS.get(exptime_sec, exptime)
        text2 += "### crucial only for long (>600s) exposure times,\n"
        for inst in fns:
            text2 += f"t_exp_{inst},{t_exp_val:.6f}\n"
            text2 += f"#t_exp_{inst},0.020833\n"
            text2 += f"#t_exp_n_int_{inst},10\n"
        text2 += """###############################################################################,
# Baseline settings per instrument: sample / hybrid,
# if 'sample_offset' one corresponding parameter called 'baseline_offset_key_inst' has to be given in params.csv,
# if 'sample_linear' two corresponding parameters called 'baseline_offset_key_inst' and 'baseline_slope_key_inst' have to be given in params.csv,
# if 'sample_GP' two corresponding parameters called 'baseline_gp1_key_inst' and 'baseline_gp2_key_inst' have to be given in params.csv,
###############################################################################,\n"""
        for inst in fns:
            text2 += f"#baseline_flux_{inst},sample_offset\n"
            text2 += f"#baseline_flux_{inst},sample_linear\n"
            text2 += f"#baseline_flux_{inst},hybrid_spline\n"
            text2 += f"#baseline_flux_{inst},hybrid_poly_2\n"
            text2 += f"baseline_flux_{inst},sample_GP_Matern32\n"
        text2 += """###############################################################################,
# Error settings (overall scaling) per instrument: sample / hybrid,
# if 'sample' one corresponding parameter called 'ln_err_key_inst' (photometry) or 'ln_jitter_key_inst' (RV) has to be given in params.csv,
###############################################################################,\n"""
        for inst in fns:
            text2 += f"error_flux_{inst},sample\n"
        text2 += """###############################################################################,
# Flares,
# if N>0 4xN corresponding parameters has to be given in params.csv,
# See https://github.com/MNGuenther/allesfitter/blob/master/paper/GJ_1243/allesfit_0/params.csv,
###############################################################################,
#N_flares,0
###############################################################################,
# Number of spots per object and instrument,
# if N>0 3xN corresponding parameters has to be given in params.csv,
###############################################################################,\n"""
        for inst in fns:
            text2 += f"#host_N_spots_{inst},0\n"
        text2 += """###############################################################################,
# Host density prior,
###############################################################################,\n"""
        text2 += f"use_host_density_prior,{rhostar_prior}"
        text2 += f"""
###############################################################################,
# Stellar variability: sample_GP_SHO / _real / _complex / matern32,
# if 'sample_GP_SHO' three corresponding parameters has to be given in params.csv,
# See https://github.com/MNGuenther/allesfitter/blob/master/tutorials/06_transits_and_rvs_with_stellar_variability/allesfit/params.csv,
###############################################################################,
#stellar_var_flux,sample_GP_SHO
#stellar_var_rv,sample_GP_real
###################################################,
# Fit TTV,
###################################################,
fit_ttvs,False
###############################################################################,
# Stellar grid per object and instrument,
###############################################################################,\n"""
        for inst in fns:
            text2 += f"host_grid_{inst},very_sparse\n"
        for i, row in target_df.iterrows():
            pl = planets[i]
            for inst in fns:
                text2 += f"#{pl}_grid_{inst},very_sparse\n"
                text2 += f"#{pl}_shape_{inst},sphere\n"
                text2 += f"#{pl}_flux_weighted_{inst},False\n"

        if debug:
            logger.info(text2)

        fp = outdir.joinpath("settings.csv")
        np.savetxt(fp, [text2], fmt="%s")
        logger.info(f"Saved: {fp}")

        # ===== Append TTV rows to params.csv using allesclass ===== #
        # allesclass loads params.csv + settings.csv + {inst}.csv and,
        # during BASEMENT init, may populate
        #   BASEMENT.data[f'{c}_tmid_observed_transits']
        # for every companion in companions_phot. That list holds exactly
        # the transits the fit will see, so len(...) is the right count
        # for {c}_ttv_transit_N (N indexes the N-th observed transit).
        # If the cache is missing for a companion, replicate what
        # afplot_per_transit does in allesfitter/general_output.py
        # (lines ~887-892): build times_combined across inst_phot and
        # call get_tmid_observed_transits with a window of fast_fit_width
        # (= T_tra_tot equivalent in days).
        if ttv:
            try:
                alles = allesclass(str(outdir))
                _settings = alles.BASEMENT.settings
                _params = alles.BASEMENT.params
                _data = alles.BASEMENT.data
                _fast_fit_width = float(_settings.get("fast_fit_width", 0.3333333333333333))
                ttv_text = ""
                for _c in _settings.get("companions_phot", []):
                    _key = f"{_c}_tmid_observed_transits"
                    _tmids = _data.get(_key)
                    if _tmids is None or len(_tmids) == 0:
                        # Fallback: compute on the fly, mirroring
                        # afplot_per_transit in general_output.py.
                        _times = []
                        for _inst in _settings.get("inst_phot", []):
                            _times += list(_data[_inst]["time"])
                        _times = np.sort(np.asarray(_times, dtype=float))
                        _epoch = float(_params[f"{_c}_epoch"])
                        _period = float(_params[f"{_c}_period"])
                        _T_tra_tot = _fast_fit_width  # days
                        _tmids = get_tmid_observed_transits(
                            _times, _epoch, _period, _T_tra_tot,
                        )
                        logger.info(
                            f"TTV: {_c} cache miss; recomputed via "
                            f"get_tmid_observed_transits (width={_T_tra_tot:.4f} d)"
                        )
                    _n_obs = len(_tmids)
                    logger.info(f"TTV: {_c} has {_n_obs} transits with data")
                    if _n_obs == 0:
                        continue
                    ttv_text += f"#TTV companion {_c},,,,,\n"
                    for _j in range(_n_obs):
                        ttv_text += (
                            f"{_c}_ttv_transit_{_j+1},0,1,uniform -0.1 0.1,"
                            f"TTV$_\\mathrm{{ttv;{_j+1}}}$,d,\n"
                        )
                if ttv_text:
                    params_fp = outdir.joinpath("params.csv")
                    with open(params_fp, "a") as _f:
                        _f.write(ttv_text)
                    logger.info(f"Appended TTV rows to {params_fp}")
                else:
                    logger.error(
                        "TTV: no companion had observed transits; "
                        "nothing appended to params.csv"
                    )
            except Exception as e:
                logger.error(f"TTV append via allesclass failed: {e}")


if __name__ == "__main__":
    main()

