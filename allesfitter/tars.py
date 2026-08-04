#!/usr/bin/env python3
"""Reader for TARS (TESS All-Sky Rotation Survey) HLSP light curves.

lightkurve cannot ingest TARS products — ``lightkurve.read`` raises
``LightkurveError: Not recognized as a supported data product``. Its
``detect_filetype`` keys on ``ORIGIN`` / ``CREATOR`` / ``PROCVER`` or on
distinctive column sets, and a TARS file carries none of those: it is
identified solely by ``HLSPID='TARS'`` in the primary header, and its
LIGHTCURVE extension holds just ``TIME`` and ``FLUX``. Verified absent from
lightkurve 2.4.2 (pinned here), 2.6.0 (latest release), and ``main`` — there is
no ``tars.py`` reader and no "tars" string in ``detect.py``.

Usage::

    from allesfitter.tars import read_tars, tars_to_csv

    lc = read_tars("hlsp_tars_..._lc.fits")     # a normal TessLightCurve
    tars_to_csv(lc, "tars.csv")                 # allesfitter instrument CSV
"""

import os

import numpy as np
from astropy.io import fits

#::: TARS stores BTJD, i.e. BJD_TDB - 2457000 (BJDREFI in the LIGHTCURVE header)
TESS_BJD_OFFSET = 2457000

#::: allesfitter rejects NaN and non-positive uncertainties, so a light curve
#::: without an error column needs a placeholder. 1.0 matches what
#::: scripts/prepare_allesfit.py writes for any pipeline whose flux_err is all
#::: NaN (e.g. QLP): it makes every point equally weighted and lets the fitted
#::: `ln_err_flux_<inst>` term set the actual scale.
PLACEHOLDER_FLUX_ERR = 1.0


def read_tars(path, flux_err=None):
    """Read a TARS ``*_lc.fits`` HLSP file into a ``lightkurve.TessLightCurve``.

    Parameters
    ----------
    path : str
        Path to a TARS light-curve FITS file.
    flux_err : float, optional
        TARS provides no uncertainty column. By default ``flux_err`` is left as
        NaN, which is honest but is rejected by :meth:`Basement.load_data`; pass
        a value (or use :func:`tars_to_csv`, which substitutes
        ``PLACEHOLDER_FLUX_ERR``) to get a fittable light curve.

    Returns
    -------
    lightkurve.TessLightCurve
        Time in BTJD (TDB), flux dimensionless and already normalized by TARS.

    Raises
    ------
    ValueError
        If the file is not a TARS product.
    """
    #::: imported lazily: lightkurve is slow to import and the package keeps its
    #::: heavy dependencies out of `import allesfitter`
    import astropy.units as u
    from astropy.time import Time
    from lightkurve import TessLightCurve

    with fits.open(path) as hdulist:
        primary = hdulist[0].header
        hlspid = str(primary.get("HLSPID", "")).strip().upper()
        if hlspid != "TARS":
            raise ValueError(
                f"{os.path.basename(path)!r} is not a TARS light curve "
                f"(HLSPID={primary.get('HLSPID')!r}). Use lightkurve.read() for the "
                "pipelines it supports (SPOC, QLP, TASOC, CDIPS, TGLC, ...)."
            )
        lc_hdu = hdulist[1]
        missing = [c for c in ("TIME", "FLUX") if c not in lc_hdu.columns.names]
        if missing:
            raise ValueError(
                f"{os.path.basename(path)!r} is missing the {missing} column(s) "
                f"in its LIGHTCURVE extension; got {lc_hdu.columns.names}."
            )
        time_btjd = np.asarray(lc_hdu.data["TIME"], dtype=float)
        flux = np.asarray(lc_hdu.data["FLUX"], dtype=float)

    if flux_err is None:
        err = np.full_like(flux, np.nan)
    else:
        err = np.full_like(flux, float(flux_err))

    return TessLightCurve(
        time=Time(time_btjd, format="btjd", scale="tdb"),
        flux=flux * u.dimensionless_unscaled,
        flux_err=err * u.dimensionless_unscaled,
        meta={
            "MISSION": "TESS",
            "AUTHOR": "TARS",
            "FLUX_ORIGIN": "TARS",
            "OBJECT": primary.get("HLSPTARG"),
            "LABEL": primary.get("HLSPTARG"),
            "TARGETID": primary.get("TICID"),
            "TICID": primary.get("TICID"),
            "SECTOR": primary.get("SECTOR"),
            "CAMERA": primary.get("CAMERA"),
            "CCD": primary.get("CCD"),
            "EXPTIME": primary.get("EXPTIME"),
            "TESSMAG": primary.get("TESSMAG"),
            "RA": primary.get("RA_TARG"),
            "DEC": primary.get("DEC_TARG"),
            "HLSPVER": primary.get("HLSPVER"),
        },
    )


def tars_to_csv(lc, path, bjd_offset=TESS_BJD_OFFSET):
    """Write a light curve as a 3-column allesfitter instrument CSV.

    Applies the same uncertainty rule as ``scripts/prepare_allesfit.py``: when
    ``flux_err`` is *entirely* NaN it is replaced by
    :data:`PLACEHOLDER_FLUX_ERR`; otherwise rows containing any NaN are dropped.
    Times are converted from BTJD to full BJD_TDB, since allesfitter expects the
    same time system in the data files and in ``params.csv``.

    Returns
    -------
    int
        Number of rows written.
    """
    time = np.asarray(lc.time.value, dtype=float) + bjd_offset
    flux = np.asarray(lc.flux, dtype=float)
    flux_err = np.asarray(lc.flux_err, dtype=float)

    if not np.any(np.isfinite(flux_err)):
        #::: no uncertainty information at all (TARS, QLP): weight every point
        #::: equally and let the fitted `ln_err_flux_<inst>` set the scale
        flux_err = np.full_like(flux, PLACEHOLDER_FLUX_ERR)

    keep = np.isfinite(time) & np.isfinite(flux) & np.isfinite(flux_err)
    order = np.argsort(time[keep], kind="stable")
    rows = np.column_stack([time[keep][order], flux[keep][order], flux_err[keep][order]])

    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    np.savetxt(path, rows, delimiter=",", fmt="%.10f")
    return len(rows)
