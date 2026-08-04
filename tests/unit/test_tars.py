"""Reading TARS HLSP light curves and exporting them as allesfitter CSVs."""

import numpy as np
import pytest
from astropy.io import fits

from allesfitter.basement import _load_inst_csv
from allesfitter.tars import (
    PLACEHOLDER_FLUX_ERR,
    TESS_BJD_OFFSET,
    is_tars_file,
    patch_lightkurve,
    read_tars,
    tars_to_csv,
)

TIME_BTJD = np.array([2228.75, 2228.76, 2228.77, 2228.78])
FLUX = np.array([1.0001, 0.9998, 1.0003, 0.9995])


def write_tars_fits(path, time=TIME_BTJD, flux=FLUX, hlspid="TARS", columns=("TIME", "FLUX")):
    """Build a minimal TARS-shaped HLSP file: primary header + LIGHTCURVE table."""
    primary = fits.PrimaryHDU()
    primary.header["HLSPID"] = hlspid
    primary.header["HLSPNAME"] = "TESS All-Sky Rotation Survey"
    primary.header["HLSPTARG"] = "TIC 146413471"
    primary.header["HLSPVER"] = "1.0"
    primary.header["TICID"] = 146413471
    primary.header["SECTOR"] = 34
    primary.header["CAMERA"] = 1
    primary.header["CCD"] = 2
    primary.header["EXPTIME"] = 600
    primary.header["TESSMAG"] = 11.0427
    primary.header["RA_TARG"] = 123.4
    primary.header["DEC_TARG"] = -56.7
    primary.header["TELESCOP"] = "TESS"

    available = {"TIME": time, "FLUX": flux}
    cols = [
        fits.Column(
            name=name, format="D", unit="d" if name == "TIME" else None, array=available[name]
        )
        for name in columns
    ]
    table = fits.BinTableHDU.from_columns(cols, name="LIGHTCURVE")
    table.header["BJDREFI"] = TESS_BJD_OFFSET
    table.header["BJDREFF"] = 0.0
    fits.HDUList([primary, table]).writeto(path, overwrite=True)
    return str(path)


def test_read_tars_returns_a_tess_lightcurve_in_btjd(tmp_path):
    # Arrange
    path = write_tars_fits(tmp_path / "tars_lc.fits")

    # Act
    lc = read_tars(path)

    # Assert
    assert type(lc).__name__ == "TessLightCurve"
    assert lc.time.format == "btjd"
    assert lc.time.scale == "tdb"
    np.testing.assert_allclose(lc.time.value, TIME_BTJD)
    np.testing.assert_allclose(np.asarray(lc.flux, dtype=float), FLUX)
    assert lc.meta["SECTOR"] == 34
    assert lc.meta["TICID"] == 146413471
    assert lc.meta["AUTHOR"] == "TARS"


def test_read_tars_leaves_flux_err_nan_by_default(tmp_path):
    # Arrange: TARS ships no uncertainty column
    path = write_tars_fits(tmp_path / "tars_lc.fits")

    # Act
    lc = read_tars(path)

    # Assert
    assert not np.any(np.isfinite(np.asarray(lc.flux_err, dtype=float)))


def test_read_tars_honours_an_explicit_flux_err(tmp_path):
    # Arrange
    path = write_tars_fits(tmp_path / "tars_lc.fits")

    # Act
    lc = read_tars(path, flux_err=1.45e-3)

    # Assert
    np.testing.assert_allclose(np.asarray(lc.flux_err, dtype=float), 1.45e-3)


def test_read_tars_rejects_a_non_tars_product(tmp_path):
    # Arrange: a QLP file would carry a different HLSPID
    path = write_tars_fits(tmp_path / "qlp_lc.fits", hlspid="QLP")

    # Act / Assert
    with pytest.raises(ValueError, match="is not a TARS light curve"):
        read_tars(path)


def test_read_tars_rejects_a_file_without_a_flux_column(tmp_path):
    # Arrange
    path = write_tars_fits(tmp_path / "tars_lc.fits", columns=("TIME",))

    # Act / Assert
    with pytest.raises(ValueError, match="missing the \\['FLUX'\\] column"):
        read_tars(path)


def test_tars_to_csv_substitutes_one_when_flux_err_is_all_nan(tmp_path):
    # Arrange: same rule scripts/prepare_allesfit.py applies to QLP
    lc = read_tars(write_tars_fits(tmp_path / "tars_lc.fits"))
    out = tmp_path / "tars.csv"

    # Act
    n_rows = tars_to_csv(lc, str(out))

    # Assert
    assert n_rows == len(TIME_BTJD)
    _time, _flux, flux_err, _custom, _cov = _load_inst_csv(str(out))
    np.testing.assert_allclose(flux_err, PLACEHOLDER_FLUX_ERR)


def test_tars_to_csv_converts_btjd_to_full_bjd(tmp_path):
    # Arrange
    lc = read_tars(write_tars_fits(tmp_path / "tars_lc.fits"))
    out = tmp_path / "tars.csv"

    # Act
    tars_to_csv(lc, str(out))

    # Assert: allesfitter needs the same time system as params.csv
    time, _flux, _flux_err, _custom, _cov = _load_inst_csv(str(out))
    np.testing.assert_allclose(time, TIME_BTJD + TESS_BJD_OFFSET)


def test_tars_to_csv_drops_rows_instead_of_overwriting_partial_nans(tmp_path):
    # Arrange: real uncertainties present, two points missing
    lc = read_tars(write_tars_fits(tmp_path / "tars_lc.fits"), flux_err=1.45e-3)
    lc.flux_err[:2] = np.nan
    out = tmp_path / "tars.csv"

    # Act
    n_rows = tars_to_csv(lc, str(out))

    # Assert: the placeholder must not clobber genuine uncertainties
    assert n_rows == len(TIME_BTJD) - 2
    _time, _flux, flux_err, _custom, _cov = _load_inst_csv(str(out))
    np.testing.assert_allclose(flux_err, 1.45e-3)


def test_is_tars_file_discriminates_products(tmp_path):
    # Arrange
    tars = write_tars_fits(tmp_path / "tars_lc.fits")
    qlp = write_tars_fits(tmp_path / "qlp_lc.fits", hlspid="QLP")

    # Act / Assert
    assert is_tars_file(tars) is True
    assert is_tars_file(qlp) is False


def test_is_tars_file_is_false_for_unopenable_targets(tmp_path):
    # Arrange: URLs, S3 URIs and truncated downloads must not raise here
    not_fits = tmp_path / "notes.txt"
    not_fits.write_text("definitely not a FITS file")

    # Act / Assert
    assert is_tars_file(str(not_fits)) is False
    assert is_tars_file(str(tmp_path / "missing.fits")) is False
    assert is_tars_file("https://example.invalid/x_lc.fits") is False


def test_patch_lightkurve_makes_lk_read_accept_tars(tmp_path):
    # Arrange
    import lightkurve as lk

    path = write_tars_fits(tmp_path / "tars_lc.fits")
    patch_lightkurve()

    # Act
    lc = lk.read(path)

    # Assert
    assert type(lc).__name__ == "TessLightCurve"
    np.testing.assert_allclose(lc.time.value, TIME_BTJD)


def test_patch_lightkurve_rebinds_the_name_search_uses(tmp_path):
    # Arrange: SearchResult.download_all calls lightkurve.search's own `read`,
    # so patching only lightkurve.io.read would leave downloads broken
    import lightkurve
    import lightkurve.search

    patch_lightkurve()

    # Act / Assert
    assert lightkurve.search.read is lightkurve.read
    assert lightkurve.io.read is lightkurve.read


def test_patch_lightkurve_leaves_other_products_to_lightkurve(tmp_path):
    # Arrange: a non-TARS file must still take lightkurve's own code path
    import lightkurve as lk

    patch_lightkurve()
    path = write_tars_fits(tmp_path / "qlp_lc.fits", hlspid="QLP")

    # Act: lightkurve reads this stub through its own generic reader
    lc = lk.read(path)

    # Assert: it did not go through read_tars, which stamps its own provenance
    assert lc.meta.get("FLUX_ORIGIN") != "TARS"
    assert lc.meta.get("AUTHOR") != "TARS"


def test_patch_lightkurve_is_idempotent():
    # Arrange
    patch_lightkurve()

    # Act: a second call must not wrap `read` again
    applied_again = patch_lightkurve()

    # Assert
    assert applied_again is False


def test_tars_to_csv_output_passes_basement_load_data_validation(tmp_path):
    # Arrange
    lc = read_tars(write_tars_fits(tmp_path / "tars_lc.fits"))
    out = tmp_path / "tars.csv"
    tars_to_csv(lc, str(out))

    # Act
    time, flux, flux_err, custom_series, _cov = _load_inst_csv(str(out))

    # Assert: the exact guards Basement.load_data raises on
    assert not any(np.isnan(time * flux * flux_err * custom_series))
    assert not any(flux_err == 0)
    assert not any(flux_err < 0)
    assert all(np.diff(time) > 0)
