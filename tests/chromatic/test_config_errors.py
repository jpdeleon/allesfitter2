"""Error-case tests for chromatic config parser.

These pin the parser hardening added on the chromatic branch: it must raise
informative errors for inconsistent configurations rather than silently
producing a broken fit.
"""

from __future__ import annotations

import pytest

from allesfitter import config

from tests.chromatic._helpers import (
    TRUE_RR_KEPLER,
    TRUE_RR_TESS,
    common_orbital_rows as _common_orbital_rows,
    dilution_rows as _dilution_rows,
    ldc_rows as _ldc_rows,
)


def _chromatic_rr_rows():
    return [
        {"name": "b_rr_tess", "value": TRUE_RR_TESS, "fit": 1,
         "bounds": "uniform 0 0.3", "label": "rr_tess"},
        {"name": "b_rr_kepler", "value": TRUE_RR_KEPLER, "fit": 1,
         "bounds": "uniform 0 0.3", "label": "rr_kepler"},
    ]


# --------------------------------------------------------------------------- #
# Bandpass / inst_phot count mismatch
# --------------------------------------------------------------------------- #
class TestBandpassCountMismatch:
    def test_too_few_bandpasses_raises(self, make_datadir):
        # 2 instruments but only 1 bandpass entry — must raise, not broadcast.
        datadir = make_datadir(
            "bp_too_few",
            inst_phot=["tess", "kepler"],
            bandpass="tess",
            params_rows=_chromatic_rr_rows()
            + _common_orbital_rows()
            + _dilution_rows(["tess", "kepler"])
            + _ldc_rows("tess")
            + _ldc_rows("kepler"),
        )
        with pytest.raises(ValueError, match="bandpass.*entries.*inst_phot"):
            config.init(str(datadir), quiet=True)

    def test_too_many_bandpasses_raises(self, make_datadir):
        datadir = make_datadir(
            "bp_too_many",
            inst_phot=["tess"],
            bandpass="tess kepler",
            params_rows=[
                {"name": "b_rr_tess", "value": TRUE_RR_TESS, "fit": 1,
                 "bounds": "uniform 0 0.3", "label": "rr_tess"},
            ]
            + _common_orbital_rows()
            + _dilution_rows(["tess"])
            + _ldc_rows("tess"),
        )
        with pytest.raises(ValueError, match="bandpass.*entries.*inst_phot"):
            config.init(str(datadir), quiet=True)


# --------------------------------------------------------------------------- #
# Duplicate params.csv rows
# --------------------------------------------------------------------------- #
class TestDuplicateRows:
    def test_duplicate_rr_row_raises(self, make_datadir):
        # Two rows for b_rr_tess with conflicting values — last-wins is a
        # silent foot-gun, so we raise.
        rows = (
            _chromatic_rr_rows()
            + [{"name": "b_rr_tess", "value": 0.20, "fit": 1,
                "bounds": "uniform 0 0.3", "label": "rr_dup"}]
            + _common_orbital_rows()
            + _dilution_rows(["tess", "kepler"])
            + _ldc_rows("tess")
            + _ldc_rows("kepler")
        )
        datadir = make_datadir(
            "dup_rr",
            inst_phot=["tess", "kepler"],
            bandpass="tess kepler",
            params_rows=rows,
        )
        with pytest.raises(ValueError, match="duplicate rows"):
            config.init(str(datadir), quiet=True)

    def test_duplicate_orbital_row_raises(self, make_datadir):
        rows = (
            _chromatic_rr_rows()
            + _common_orbital_rows()
            + [{"name": "b_period", "value": 5.0, "fit": 1,
                "bounds": "uniform 4 6", "label": "period_dup", "unit": "d"}]
            + _dilution_rows(["tess", "kepler"])
            + _ldc_rows("tess")
            + _ldc_rows("kepler")
        )
        datadir = make_datadir(
            "dup_period",
            inst_phot=["tess", "kepler"],
            bandpass="tess kepler",
            params_rows=rows,
        )
        with pytest.raises(ValueError, match="duplicate rows.*b_period"):
            config.init(str(datadir), quiet=True)


# --------------------------------------------------------------------------- #
# Unknown bandpass referenced in params.csv
# --------------------------------------------------------------------------- #
class TestUnknownBandpass:
    def test_typo_in_rr_suffix_raises(self, make_datadir):
        # User types b_rr_tes (missing s) — must surface, not silently ignore.
        rows = (
            [{"name": "b_rr_tes", "value": TRUE_RR_TESS, "fit": 1,
              "bounds": "uniform 0 0.3", "label": "rr_typo"}]
            + [{"name": "b_rr_kepler", "value": TRUE_RR_KEPLER, "fit": 1,
               "bounds": "uniform 0 0.3", "label": "rr_kepler"}]
            + _common_orbital_rows()
            + _dilution_rows(["tess", "kepler"])
            + _ldc_rows("tess")
            + _ldc_rows("kepler")
        )
        datadir = make_datadir(
            "unknown_bp",
            inst_phot=["tess", "kepler"],
            bandpass="tess kepler",
            params_rows=rows,
        )
        with pytest.raises(ValueError, match="unknown bandpass"):
            config.init(str(datadir), quiet=True)

    def test_extra_unknown_bandpass_rr_row_raises(self, make_datadir):
        rows = (
            _chromatic_rr_rows()
            + [{"name": "b_rr_johnson", "value": 0.12, "fit": 1,
                "bounds": "uniform 0 0.3", "label": "rr_extra"}]
            + _common_orbital_rows()
            + _dilution_rows(["tess", "kepler"])
            + _ldc_rows("tess")
            + _ldc_rows("kepler")
        )
        datadir = make_datadir(
            "extra_bp",
            inst_phot=["tess", "kepler"],
            bandpass="tess kepler",
            params_rows=rows,
        )
        with pytest.raises(ValueError, match="unknown bandpass.*johnson"):
            config.init(str(datadir), quiet=True)


# --------------------------------------------------------------------------- #
# Chromatic settings vs. params.csv shape mismatch
#
# These pin the new check that catches silent fallback to the unsuffixed
# b_rr when chromatic mode is on but params.csv hasn't been migrated.
# --------------------------------------------------------------------------- #
class TestChromaticParamsShapeMismatch:
    def test_only_achromatic_b_rr_in_chromatic_settings_raises(self, make_datadir):
        # params.csv has b_rr but no b_rr_<bp> at all → would silently
        # fall back to b_rr for every band before this check existed.
        from tests.chromatic._helpers import TRUE_RR_TESS
        rows = (
            [{"name": "b_rr", "value": TRUE_RR_TESS, "fit": 1,
              "bounds": "uniform 0 0.3", "label": "rr"}]
            + _common_orbital_rows()
            + _dilution_rows(["tess", "kepler"])
            + _ldc_rows("tess")
            + _ldc_rows("kepler")
        )
        datadir = make_datadir(
            "achromatic_in_chromatic",
            inst_phot=["tess", "kepler"],
            bandpass="tess kepler",
            params_rows=rows,
        )
        with pytest.raises(ValueError, match="Chromatic configuration mismatch"):
            config.init(str(datadir), quiet=True)

    def test_partial_chromatic_rr_raises(self, make_datadir):
        # b_rr_tess given, b_rr_kepler missing — would silently default the
        # missing key to None and fall back to nonexistent b_rr.
        rows = (
            [{"name": "b_rr_tess", "value": TRUE_RR_TESS, "fit": 1,
              "bounds": "uniform 0 0.3", "label": "rr_tess"}]
            + _common_orbital_rows()
            + _dilution_rows(["tess", "kepler"])
            + _ldc_rows("tess")
            + _ldc_rows("kepler")
        )
        datadir = make_datadir(
            "partial_chromatic",
            inst_phot=["tess", "kepler"],
            bandpass="tess kepler",
            params_rows=rows,
        )
        with pytest.raises(ValueError, match="missing b_rr_kepler"):
            config.init(str(datadir), quiet=True)

    def test_mixed_achromatic_and_partial_chromatic_raises(self, make_datadir):
        # Both b_rr AND b_rr_tess present, but b_rr_kepler missing — the
        # ambiguous half-and-half shape that's neither valid achromatic nor
        # valid chromatic.
        rows = (
            [
                {"name": "b_rr", "value": TRUE_RR_TESS, "fit": 1,
                 "bounds": "uniform 0 0.3", "label": "rr"},
                {"name": "b_rr_tess", "value": TRUE_RR_TESS, "fit": 1,
                 "bounds": "uniform 0 0.3", "label": "rr_tess"},
            ]
            + _common_orbital_rows()
            + _dilution_rows(["tess", "kepler"])
            + _ldc_rows("tess")
            + _ldc_rows("kepler")
        )
        datadir = make_datadir(
            "mixed_rr",
            inst_phot=["tess", "kepler"],
            bandpass="tess kepler",
            params_rows=rows,
        )
        with pytest.raises(ValueError, match="mixes the achromatic key"):
            config.init(str(datadir), quiet=True)


# --------------------------------------------------------------------------- #
# Achromatic backward compatibility
# --------------------------------------------------------------------------- #
class TestAchromaticBackcompat:
    def test_achromatic_config_initializes_unchanged(self, achromatic_datadir):
        # No 'bandpass' setting at all — pre-chromatic behavior. Must succeed
        # with the exact key shapes from master: single b_rr, single ldc per inst.
        config.init(str(achromatic_datadir), quiet=True)
        b = config.BASEMENT
        assert b.settings["bandpass"] == {}
        assert b.settings["chromatic"] is False
        assert "b_rr" in b.params
        assert "host_ldc_q1_tess" in b.params
        # No bandpass-keyed rr leaked in
        chromatic_rr = [k for k in b.params if k.startswith("b_rr_")]
        assert chromatic_rr == [], f"unexpected chromatic rr keys: {chromatic_rr}"


# --------------------------------------------------------------------------- #
# Per-instrument settings suffix validation
# --------------------------------------------------------------------------- #
class TestOrphanPerInstSettings:
    """Regression for the user-reported silent failure:

    settings.csv had
        inst_phot,tglc1800_s10s11 tglc600_s37s38 tglc120_s63s64 tglc120_s90
        bandpass,tess tess tess tess
        host_ld_law_tess,quad

    No instrument is named 'tess', so host_ld_law_<actual_inst> defaulted
    to None and ellc silently received ldc_1=None — host_ldc_q1_tess values
    in params.csv had zero effect on the transit shape.

    The validator must catch this at config-load time."""

    def test_host_ld_law_keyed_by_bandpass_raises(self, make_datadir):
        insts = ["tglc1800_s10s11", "tglc600_s37s38", "tglc120_s63s64", "tglc120_s90"]
        rows = (
            [{"name": "b_rr_tess", "value": TRUE_RR_TESS, "fit": 1,
              "bounds": "uniform 0 0.3", "label": "rr_tess"}]
            + _common_orbital_rows()
            + _dilution_rows(insts)
            + [{"name": f"ln_err_flux_{i}", "value": -7.0, "fit": 0,
                "bounds": "uniform -15 0", "label": f"ln_err_{i}"} for i in insts]
            + [{"name": f"baseline_offset_flux_{i}", "value": 0.0, "fit": 0,
                "bounds": "uniform -0.05 0.05", "label": f"offset_{i}"} for i in insts]
            + _ldc_rows("tess")
        )
        datadir = make_datadir(
            "orphan_ld_law",
            inst_phot=insts,
            bandpass="tess tess tess tess",
            params_rows=rows,
            extra_settings=["host_ld_law_tess,quad", "host_ld_space_tess,q"],
        )
        with pytest.raises(ValueError, match=r"per-instrument keys whose suffix"):
            config.init(str(datadir), quiet=True)

    def test_error_message_hints_at_bandpass_confusion(self, make_datadir):
        insts = ["tglc120_s90", "tglc120_s63s64"]
        rows = (
            [{"name": "b_rr_tess", "value": TRUE_RR_TESS, "fit": 1,
              "bounds": "uniform 0 0.3", "label": "rr_tess"}]
            + _common_orbital_rows()
            + _dilution_rows(insts)
            + [{"name": f"ln_err_flux_{i}", "value": -7.0, "fit": 0,
                "bounds": "uniform -15 0", "label": f"ln_err_{i}"} for i in insts]
            + [{"name": f"baseline_offset_flux_{i}", "value": 0.0, "fit": 0,
                "bounds": "uniform -0.05 0.05", "label": f"offset_{i}"} for i in insts]
            + _ldc_rows("tess")
        )
        datadir = make_datadir(
            "orphan_hint",
            inst_phot=insts,
            bandpass="tess tess",
            params_rows=rows,
            extra_settings=["host_ld_law_tess,quad"],
        )
        with pytest.raises(ValueError) as exc:
            config.init(str(datadir), quiet=True)
        msg = str(exc.value)
        # The hint must name the affected instruments and call out the
        # bandpass/instrument confusion.
        assert "BANDPASS label" in msg
        assert "tglc120_s90" in msg or "tglc120_s63s64" in msg

    def test_missing_host_ld_law_defaults_to_quad(self, make_datadir):
        # When settings.csv does not specify host_ld_law_<inst> at all, the
        # default must be 'quad' (not None). The prior None default silently
        # disabled limb darkening and made host_ldc_q1/q2 in params.csv have
        # zero effect on the transit shape.
        insts = ["tess"]
        rows = (
            [{"name": "b_rr", "value": TRUE_RR_TESS, "fit": 1,
              "bounds": "uniform 0 0.3", "label": "rr"}]
            + _common_orbital_rows()
            + _dilution_rows(insts)
            + [{"name": f"ln_err_flux_{i}", "value": -7.0, "fit": 0,
                "bounds": "uniform -15 0", "label": f"ln_err_{i}"} for i in insts]
            + [{"name": f"baseline_offset_flux_{i}", "value": 0.0, "fit": 0,
                "bounds": "uniform -0.05 0.05", "label": f"offset_{i}"} for i in insts]
            + _ldc_rows("tess")
        )
        # make_datadir's write_settings auto-emits host_ld_law_<inst>,quad —
        # write the project manually here so the host_ld_law row is missing.
        import pathlib
        datadir = make_datadir(
            "default_quad",
            inst_phot=insts,
            bandpass=None,
            params_rows=rows,
        )
        # Strip the host_ld_law row to test the *default* code path.
        settings_path = pathlib.Path(datadir) / "settings.csv"
        kept = [
            line for line in settings_path.read_text().splitlines()
            if not line.startswith("host_ld_law_")
        ]
        settings_path.write_text("\n".join(kept) + "\n")

        config.init(str(datadir), quiet=True)
        assert config.BASEMENT.settings["host_ld_law_tess"] == "quad"

    def test_host_ld_law_none_string_disables_ld(self, make_datadir):
        # Users who explicitly want no limb darkening must still be able to
        # opt out via host_ld_law_<inst>,none — the default-to-quad change
        # must not override an explicit 'none'.
        insts = ["tess"]
        rows = (
            [{"name": "b_rr", "value": TRUE_RR_TESS, "fit": 1,
              "bounds": "uniform 0 0.3", "label": "rr"}]
            + _common_orbital_rows()
            + _dilution_rows(insts)
            + [{"name": f"ln_err_flux_{i}", "value": -7.0, "fit": 0,
                "bounds": "uniform -15 0", "label": f"ln_err_{i}"} for i in insts]
            + [{"name": f"baseline_offset_flux_{i}", "value": 0.0, "fit": 0,
                "bounds": "uniform -0.05 0.05", "label": f"offset_{i}"} for i in insts]
            + _ldc_rows("tess")
        )
        import pathlib
        datadir = make_datadir(
            "explicit_none",
            inst_phot=insts,
            bandpass=None,
            params_rows=rows,
        )
        # Replace the auto-emitted host_ld_law_tess,quad with ,none.
        settings_path = pathlib.Path(datadir) / "settings.csv"
        new_lines = [
            "host_ld_law_tess,none" if line.startswith("host_ld_law_tess,") else line
            for line in settings_path.read_text().splitlines()
        ]
        settings_path.write_text("\n".join(new_lines) + "\n")

        config.init(str(datadir), quiet=True)
        assert config.BASEMENT.settings["host_ld_law_tess"] is None

    def test_per_inst_rows_pass(self, make_datadir):
        # Sanity: when the user correctly writes one row per actual instrument
        # name, config.init succeeds. This locks the validator from over-firing.
        insts = ["tglc120_s90", "tglc120_s63s64"]
        rows = (
            [{"name": "b_rr_tess", "value": TRUE_RR_TESS, "fit": 1,
              "bounds": "uniform 0 0.3", "label": "rr_tess"}]
            + _common_orbital_rows()
            + _dilution_rows(insts)
            + [{"name": f"ln_err_flux_{i}", "value": -7.0, "fit": 0,
                "bounds": "uniform -15 0", "label": f"ln_err_{i}"} for i in insts]
            + [{"name": f"baseline_offset_flux_{i}", "value": 0.0, "fit": 0,
                "bounds": "uniform -0.05 0.05", "label": f"offset_{i}"} for i in insts]
            + _ldc_rows("tess")
        )
        datadir = make_datadir(
            "orphan_ok",
            inst_phot=insts,
            bandpass="tess tess",
            params_rows=rows,
            # no extra settings — write_settings already emits host_ld_law_<inst>
            # for each inst in inst_phot, which is the correct shape.
        )
        config.init(str(datadir), quiet=True)
        b = config.BASEMENT
        assert b.settings["chromatic"] is False
        for inst in insts:
            assert b.settings["host_ld_law_" + inst] == "quad"
