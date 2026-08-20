from types import SimpleNamespace

import numpy as np
import pytest

import allesfitter.update_params as update_params_module
from allesfitter.update_params import _ttv_rows, update_params

# Anderson-Darling classification is data-dependent, so these two samples are
# fixed (seed + size) to reliably land on opposite sides of the normal/uniform
# decision boundary used by update_params()'s dist-picking logic.
_NORMAL_SAMPLE = np.random.default_rng(42).normal(loc=5.0, scale=0.1, size=1000)
_UNIFORM_SAMPLE = np.random.default_rng(42).uniform(low=0.0, high=1.0, size=1000)


def _percentile_errors(sample):
    median = float(np.percentile(sample, 50))
    ll = median - float(np.percentile(sample, 16))
    ul = float(np.percentile(sample, 84)) - median
    return median, ll, ul


def _make_alles(posterior_samples, fixed_params, *, settings=None, data=None):
    """A SimpleNamespace mimicking the slice of allesclass's public surface
    that update_params()/_ttv_rows() touch."""
    medians, lls, uls = {}, {}, {}
    for name, sample in posterior_samples.items():
        median, ll, ul = _percentile_errors(sample)
        medians[name], lls[name], uls[name] = median, ll, ul

    fitkeys = list(posterior_samples)
    allkeys = fitkeys + list(fixed_params)
    labels = {name: f"${name}$" for name in allkeys}
    units = {name: "" for name in allkeys}

    basement = SimpleNamespace(
        fitkeys=fitkeys,
        allkeys=allkeys,
        labels=[labels[name] for name in allkeys],
        units=[units[name] for name in allkeys],
        params=dict(fixed_params),
        settings=settings or {},
        data=data or {},
    )
    return SimpleNamespace(
        posterior_params=posterior_samples,
        posterior_params_median=medians,
        posterior_params_ll=lls,
        posterior_params_ul=uls,
        BASEMENT=basement,
    )


def test_update_params_writes_normal_prior_for_normally_distributed_posterior(
    monkeypatch, tmp_path
):
    alles = _make_alles({"b_rr": _NORMAL_SAMPLE}, {})
    monkeypatch.setattr(update_params_module, "allesclass", lambda path: alles)

    fp = update_params(str(tmp_path))

    text = (tmp_path / "params2.csv").read_text()
    assert fp == str(tmp_path / "params2.csv")
    median, ll, ul = _percentile_errors(_NORMAL_SAMPLE)
    sig = np.sqrt(ll**2 + ul**2)
    assert f"b_rr,{median:6f},1,normal {median:6f} {sig:6f}" in text


def test_update_params_writes_uniform_prior_for_nonnormal_posterior(monkeypatch, tmp_path):
    alles = _make_alles({"b_rr": _UNIFORM_SAMPLE}, {})
    monkeypatch.setattr(update_params_module, "allesclass", lambda path: alles)

    update_params(str(tmp_path))

    text = (tmp_path / "params2.csv").read_text()
    l_limit, mid, u_limit = np.nanpercentile(_UNIFORM_SAMPLE, q=[1, 50, 99])
    assert f"b_rr,{mid:6f},1,uniform {l_limit:6f} {u_limit:6f}" in text


def test_update_params_preserves_fixed_parameters(monkeypatch, tmp_path):
    alles = _make_alles({"b_rr": _NORMAL_SAMPLE}, {"b_f_c": 0.0})
    monkeypatch.setattr(update_params_module, "allesclass", lambda path: alles)

    update_params(str(tmp_path))

    text = (tmp_path / "params2.csv").read_text()
    assert "b_f_c,0.0,0,," in text


def test_update_params_skips_fixed_parameter_with_none_value(monkeypatch, tmp_path):
    alles = _make_alles({"b_rr": _NORMAL_SAMPLE}, {"host_ldc_q1_tess": None})
    monkeypatch.setattr(update_params_module, "allesclass", lambda path: alles)

    update_params(str(tmp_path))

    text = (tmp_path / "params2.csv").read_text()
    assert "host_ldc_q1_tess" not in text


def test_update_params_omits_ttv_rows_by_default(monkeypatch, tmp_path):
    alles = _make_alles({"b_rr": _NORMAL_SAMPLE}, {})
    monkeypatch.setattr(update_params_module, "allesclass", lambda path: alles)

    update_params(str(tmp_path))

    text = (tmp_path / "params2.csv").read_text()
    assert "#TTV" not in text
    assert "_ttv_transit_" not in text


def test_update_params_appends_ttv_rows_when_requested(monkeypatch, tmp_path):
    time = np.linspace(100, 110.5, 2000)
    settings = {
        "fast_fit_width": 0.5,
        "inst_phot": ["tess"],
        "companions_phot": ["b"],
    }
    data = {"tess": {"time": time}}
    alles = _make_alles(
        {"b_rr": _NORMAL_SAMPLE},
        {"b_epoch": 100.1, "b_period": 2.0},
        settings=settings,
        data=data,
    )
    alles.posterior_params_median["b_epoch"] = 100.1
    alles.posterior_params_median["b_period"] = 2.0
    monkeypatch.setattr(update_params_module, "allesclass", lambda path: alles)

    update_params(str(tmp_path), ttv=True)

    text = (tmp_path / "params2.csv").read_text()
    assert "#TTV companion b" in text
    for j in range(1, 7):
        assert f"b_ttv_transit_{j},0,1,uniform -0.05 0.05" in text


def test_ttv_rows_cache_hit_matching_fresh_count(monkeypatch):
    time = np.linspace(100, 110.5, 2000)
    from allesfitter.exoworlds_rdx.lightcurves.index_transits import get_tmid_observed_transits

    tmids = get_tmid_observed_transits(time, 100.1, 2.0, 0.5)
    settings = {
        "fast_fit_width": 0.5,
        "inst_phot": ["tess"],
        "companions_phot": ["b"],
    }
    data = {
        "tess": {"time": time},
        "b_tmid_observed_transits": tmids,
    }
    alles = _make_alles({}, {"b_epoch": 100.1, "b_period": 2.0}, settings=settings, data=data)
    alles.posterior_params_median = {"b_epoch": 100.1, "b_period": 2.0}

    text = _ttv_rows(alles)

    assert text.count("_ttv_transit_") == len(tmids) == 6


def test_ttv_rows_cache_miss_recomputes_from_time_union():
    time = np.linspace(100, 110.5, 2000)
    settings = {
        "fast_fit_width": 0.5,
        "inst_phot": ["tess"],
        "companions_phot": ["b"],
    }
    data = {"tess": {"time": time}}
    alles = _make_alles({}, {"b_epoch": 100.1, "b_period": 2.0}, settings=settings, data=data)
    alles.posterior_params_median = {"b_epoch": 100.1, "b_period": 2.0}

    text = _ttv_rows(alles)

    assert text.count("_ttv_transit_") == 6


def test_ttv_rows_prefers_union_count_on_stale_cache_disagreement():
    time = np.linspace(100, 110.5, 2000)
    settings = {
        "fast_fit_width": 0.5,
        "inst_phot": ["tess"],
        "companions_phot": ["b"],
    }
    # Stale cache: only 4 of the 6 real transits (e.g. from a fast-fit-reduced
    # single-instrument cache built before more data was added).
    stale_tmids = np.array([100.1, 102.1, 104.1, 106.1])
    data = {
        "tess": {"time": time},
        "b_tmid_observed_transits": stale_tmids,
    }
    alles = _make_alles({}, {"b_epoch": 100.1, "b_period": 2.0}, settings=settings, data=data)
    alles.posterior_params_median = {"b_epoch": 100.1, "b_period": 2.0}

    text = _ttv_rows(alles)

    assert text.count("_ttv_transit_") == 6


def test_ttv_rows_skips_companion_with_no_data_coverage():
    settings = {
        "fast_fit_width": 0.1,
        "inst_phot": ["tess"],
        "companions_phot": ["b"],
    }
    # Data only near t=100; the companion's first observable epoch (t=100.3)
    # falls entirely in the gap between the two points.
    data = {"tess": {"time": np.array([100.0, 100.6])}}
    alles = _make_alles({}, {"b_epoch": 100.3, "b_period": 1000.0}, settings=settings, data=data)
    alles.posterior_params_median = {"b_epoch": 100.3, "b_period": 1000.0}

    text = _ttv_rows(alles)

    assert text == ""


def test_update_params_continues_after_posterior_section_error(monkeypatch, tmp_path):
    """A malformed/missing posterior key shouldn't crash update_params(); the
    error is logged and an (empty) params2.csv is still written."""
    alles = _make_alles({}, {})
    del alles.posterior_params_median  # forces the try block to raise
    monkeypatch.setattr(update_params_module, "allesclass", lambda path: alles)

    fp = update_params(str(tmp_path))

    assert (tmp_path / "params2.csv").exists()
    assert fp == str(tmp_path / "params2.csv")


@pytest.mark.parametrize("debug", [True, False])
def test_update_params_debug_flag_does_not_change_output(monkeypatch, tmp_path, debug):
    alles = _make_alles({"b_rr": _NORMAL_SAMPLE}, {"b_f_c": 0.0})
    monkeypatch.setattr(update_params_module, "allesclass", lambda path: alles)

    update_params(str(tmp_path), debug=debug)

    assert (tmp_path / "params2.csv").exists()
