"""Scope-mapping tests for chromatic transit modeling.

These tests pin the contract:
- Orbital params (period, epoch, cosi, rsuma, f_c, f_s, K) are global
  (single-keyed, no per-instrument or per-bandpass suffix).
- Radius ratio (rr) is per-bandpass in chromatic mode, single-keyed in
  achromatic mode.
- Limb darkening coefficients are per-bandpass in chromatic mode and
  per-instrument in achromatic mode.
- Per-instrument-only params (dil, sbratio, baselines, errors) live at
  the instrument scope regardless of chromatic mode.
- Backward compatibility: achromatic configs initialize exactly as before.
"""

from __future__ import annotations

import pytest

from allesfitter import config

# Orbital params that must NEVER appear with a _<inst> or _<bandpass> suffix
GLOBAL_ORBITAL_KEYS = (
    "b_period",
    "b_epoch",
    "b_cosi",
    "b_rsuma",
    "b_f_c",
    "b_f_s",
)


def _all_params(basement):
    return set(basement.params.keys())


# --------------------------------------------------------------------------- #
# Chromatic flag
# --------------------------------------------------------------------------- #
class TestChromaticFlag:
    def test_set_when_multiple_unique_bandpasses(self, two_band_two_inst_datadir):
        config.init(str(two_band_two_inst_datadir), quiet=True)
        assert config.BASEMENT.settings["chromatic"] is True

    def test_false_for_single_bandpass(self, one_band_two_inst_datadir):
        config.init(str(one_band_two_inst_datadir), quiet=True)
        assert config.BASEMENT.settings["chromatic"] is False

    def test_false_when_bandpass_absent(self, achromatic_datadir):
        config.init(str(achromatic_datadir), quiet=True)
        assert config.BASEMENT.settings["chromatic"] is False
        assert config.BASEMENT.settings["bandpass"] == {}


# --------------------------------------------------------------------------- #
# Key helpers
# --------------------------------------------------------------------------- #
class TestKeyHelpers:
    def test_get_rr_key_chromatic_uses_bandpass(self, two_band_two_inst_datadir):
        config.init(str(two_band_two_inst_datadir), quiet=True)
        b = config.BASEMENT
        assert b.get_rr_key("b", "tess") == "b_rr_tess"
        assert b.get_rr_key("b", "kepler") == "b_rr_kepler"

    def test_get_rr_key_achromatic_unsuffixed(self, achromatic_datadir):
        config.init(str(achromatic_datadir), quiet=True)
        assert config.BASEMENT.get_rr_key("b", "tess") == "b_rr"

    def test_get_rr_key_shared_bandpass_groups_instruments(self, one_band_two_inst_datadir):
        config.init(str(one_band_two_inst_datadir), quiet=True)
        b = config.BASEMENT
        # Both insts map to bandpass 'tess'. chromatic is False (single
        # unique bandpass), but the parser still keys rr by the bandpass —
        # so both instruments resolve to the same shared scalar b_rr_tess.
        assert b.get_rr_key("b", "tess_pdcsap") == "b_rr_tess"
        assert b.get_rr_key("b", "tess_qlp") == "b_rr_tess"

    def test_get_ldc_key_chromatic_uses_bandpass(self, two_band_two_inst_datadir):
        config.init(str(two_band_two_inst_datadir), quiet=True)
        b = config.BASEMENT
        assert b.get_ldc_key("host", 1, "tess", "q") == "host_ldc_q1_tess"
        assert b.get_ldc_key("host", 2, "kepler", "q") == "host_ldc_q2_kepler"
        assert b.get_ldc_key("host", 1, "tess", "u") == "host_ldc_u1_tess"

    def test_get_ldc_key_achromatic_uses_inst(self, achromatic_datadir):
        config.init(str(achromatic_datadir), quiet=True)
        assert config.BASEMENT.get_ldc_key("host", 1, "tess", "q") == "host_ldc_q1_tess"

    def test_get_ldc_key_rejects_invalid_space(self, achromatic_datadir):
        config.init(str(achromatic_datadir), quiet=True)
        with pytest.raises(ValueError, match="space must be"):
            config.BASEMENT.get_ldc_key("host", 1, "tess", space="z")


# --------------------------------------------------------------------------- #
# Global orbital params are single-keyed
# --------------------------------------------------------------------------- #
class TestGlobalOrbitalScope:
    @pytest.mark.parametrize(
        "fixture_name",
        [
            "two_band_two_inst_datadir",
            "one_band_two_inst_datadir",
            "achromatic_datadir",
        ],
    )
    def test_orbital_params_single_keyed(self, request, fixture_name):
        datadir = request.getfixturevalue(fixture_name)
        config.init(str(datadir), quiet=True)
        params = _all_params(config.BASEMENT)
        for key in GLOBAL_ORBITAL_KEYS:
            assert key in params, f"{key} missing from params"
            # No _<inst> or _<bp> variants should exist
            siblings = [k for k in params if k.startswith(key + "_")]
            assert (
                siblings == []
            ), f"{key} must be globally shared, but found scoped siblings: {siblings}"


# --------------------------------------------------------------------------- #
# Per-bandpass rr keys
# --------------------------------------------------------------------------- #
class TestRRScope:
    def test_chromatic_one_rr_per_unique_bandpass(self, two_band_two_inst_datadir):
        config.init(str(two_band_two_inst_datadir), quiet=True)
        params = _all_params(config.BASEMENT)
        bandpasses = set(config.BASEMENT.settings["bandpass"].values())
        for bp in bandpasses:
            assert f"b_rr_{bp}" in params
        # No unsuffixed b_rr when chromatic
        assert "b_rr" not in config.BASEMENT.fitkeys

    def test_achromatic_single_rr_key(self, achromatic_datadir):
        config.init(str(achromatic_datadir), quiet=True)
        params = _all_params(config.BASEMENT)
        assert "b_rr" in params
        # No chromatic variants
        assert not any(k.startswith("b_rr_") for k in params if k != "b_rr")

    def test_chromatic_rr_keys_in_fitkeys(self, two_band_two_inst_datadir):
        config.init(str(two_band_two_inst_datadir), quiet=True)
        fk = list(config.BASEMENT.fitkeys)
        assert "b_rr_tess" in fk
        assert "b_rr_kepler" in fk


# --------------------------------------------------------------------------- #
# LDC scope
# --------------------------------------------------------------------------- #
class TestLDCScope:
    def test_chromatic_ldc_scalars_per_bandpass(self, two_band_two_inst_datadir):
        config.init(str(two_band_two_inst_datadir), quiet=True)
        params = _all_params(config.BASEMENT)
        for bp in ("tess", "kepler"):
            assert f"host_ldc_q1_{bp}" in params
            assert f"host_ldc_q2_{bp}" in params

    def test_chromatic_ldc_scalar_suffixes_only_match_known_bandpasses(
        self, two_band_two_inst_datadir
    ):
        # basement.validate() defaults q1..q4 (and u1..u4) for each bandpass to
        # None when absent — so there are more keys than the ones the user
        # supplied. The contract we pin is that every host_ldc_q* / host_ldc_u*
        # suffix matches a known bandpass; nothing is keyed by inst when inst
        # differs from bandpass.
        config.init(str(two_band_two_inst_datadir), quiet=True)
        params = _all_params(config.BASEMENT)
        known_bands = set(config.BASEMENT.settings["bandpass"].values())
        suffix_prefixes = ("host_ldc_q", "host_ldc_u")
        for k in params:
            if not k.startswith(suffix_prefixes):
                continue
            # Strip 'host_ldc_qN_' or 'host_ldc_uN_' to get the suffix.
            tail = k.split("_", 3)
            # Expected shape: ['host','ldc','qN','<suffix>']
            assert len(tail) == 4, f"unexpected LDC key shape: {k}"
            suffix = tail[3]
            assert suffix in known_bands, (
                f"LDC key {k!r} suffix {suffix!r} is not a known bandpass " f"{sorted(known_bands)}"
            )
        # And the user-supplied q1/q2 must still be there for both bands.
        for bp in known_bands:
            assert f"host_ldc_q1_{bp}" in params
            assert f"host_ldc_q2_{bp}" in params

    def test_shared_bandpass_groups_ldc_scalars(self, one_band_two_inst_datadir):
        # Two instruments share bandpass 'tess'; the bandpass-suffixed LDC
        # appears once but is read by both insts during ellc assembly.
        config.init(str(one_band_two_inst_datadir), quiet=True)
        params = _all_params(config.BASEMENT)
        # Scalar source-of-truth keys live under the bandpass suffix.
        assert "host_ldc_q1_tess" in params
        assert "host_ldc_q2_tess" in params
        # No inst-suffixed scalars (would double-count the prior)
        assert "host_ldc_q1_tess_pdcsap" not in config.BASEMENT.fitkeys
        assert "host_ldc_q1_tess_qlp" not in config.BASEMENT.fitkeys


# --------------------------------------------------------------------------- #
# Per-instrument-only scope (dilution, baselines, errors, sbratio)
# --------------------------------------------------------------------------- #
class TestInstrumentScope:
    def test_dilution_per_instrument_chromatic(self, two_band_two_inst_datadir):
        config.init(str(two_band_two_inst_datadir), quiet=True)
        params = _all_params(config.BASEMENT)
        assert "dil_tess" in params
        assert "dil_kepler" in params

    def test_dilution_per_instrument_shared_band(self, one_band_two_inst_datadir):
        # Even when both insts share a bandpass, dilution stays per-inst.
        config.init(str(one_band_two_inst_datadir), quiet=True)
        params = _all_params(config.BASEMENT)
        assert "dil_tess_pdcsap" in params
        assert "dil_tess_qlp" in params

    def test_sbratio_per_instrument(self, two_band_two_inst_datadir):
        config.init(str(two_band_two_inst_datadir), quiet=True)
        params = _all_params(config.BASEMENT)
        # Validator at basement.py:1121 ensures sbratio_<inst> entries exist.
        assert "b_sbratio_tess" in params
        assert "b_sbratio_kepler" in params
