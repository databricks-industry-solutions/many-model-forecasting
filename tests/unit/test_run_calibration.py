"""Unit tests for one model's calibration and its output rows (prediction intervals).

Tests cover:
- calibrate_model: method resolution (conformal, CQR, native, fallbacks), per-model methods
  (method_by_model), agreement with the lower-level functions, clipping, and leave-one-window-out metrics
- calibrate_model on the golden M5 sample with the default settings
- backtest_cubes: window alignment, dropped series, native bands and their fallbacks
- Group labels, scale fallback, and the calibration_output / metric row builders
- Unsupported settings
"""
import pathlib

import numpy as np
import pandas as pd
import pytest

from mmf_sa.intervals.config import check_supported, resolve_prediction_intervals_conf
from mmf_sa.intervals.conformal import (
    calibrate_model, cqr_corrections, finalize_bounds, interval_metrics, residual_bounds, residual_quantiles,
)
from mmf_sa.intervals.features import backtest_cubes, classify_series, group_columns, group_labels, positive_sigma
from mmf_sa.intervals.scoring import calibration_rows, metric_rows
from mmf_sa.exceptions import ConfigurationError

GOLDEN = pathlib.Path(__file__).parent / "data" / "calibration_golden.npz"


def _pi(**overrides):
    base = {"enabled": True, "coverage": {"min_scores_per_group": 1}}
    for k, v in overrides.items():
        if k == "coverage":
            base["coverage"].update(v)
        else:
            base[k] = v
    return resolve_prediction_intervals_conf(base)


@pytest.fixture(scope="module")
def cubes():
    rng = np.random.default_rng(3)
    n, w, h = 30, 4, 5
    yhat = rng.gamma(2.0, 2.0, (n, w, h))
    y = np.maximum(yhat + rng.normal(0, 2, (n, w, h)), 0)
    sigma = rng.uniform(0.5, 2.0, n)
    labels = np.array(["A"] * 10 + ["B"] * 20, dtype=object)
    native = {lv: (yhat - 2.5 * lv / 80, yhat + 2.5 * lv / 80) for lv in (80, 95)}
    return y, yhat, sigma, labels, native


def test_conformal_matches_residual_quantiles(cubes):
    y, yhat, sigma, labels, _ = cubes
    res = calibrate_model(y, yhat, sigma, _pi(), labels=labels, clip_lower=0.0)
    assert set(res) == {80, 95}
    q_lo, q_hi, inf = residual_quantiles((y - yhat) / sigma[:, None, None], 80, "series", labels,
                                             shrinkage_kappa=10, min_scores_per_group=1)
    r = res[80]
    assert (r["method"], r["score"], r["target"]) == ("conformal", "signed_normalized", "series")
    np.testing.assert_array_equal(r["q_lo"], q_lo)
    np.testing.assert_array_equal(r["q_hi"], q_hi)
    assert r["shrinkage_weight"] == pytest.approx(4 / 14)
    assert r["native_metrics"] == {}


def test_auto_uses_cqr_with_native_bands(cubes):
    y, yhat, sigma, labels, native = cubes
    res = calibrate_model(y, yhat, sigma, _pi(), labels=labels, native=native, clip_lower=0.0)
    r = res[95]
    assert (r["method"], r["score"], r["target"]) == ("cqr", "asymmetric", "group")
    lo, hi = np.maximum(native[95][0], 0.0), native[95][1]
    q_lo, q_hi, _ = cqr_corrections(y, yhat, lo, hi, sigma, 95, pool="group", labels=labels,
                                        min_scores_per_group=1)
    np.testing.assert_array_equal(r["q_lo"], q_lo)
    np.testing.assert_array_equal(r["q_hi"], q_hi)
    assert r["shrinkage_weight"] is None
    assert set(r["native_metrics"]) == {"coverage", "msis", "width"}


def test_method_conformal_ignores_native_bands(cubes):
    y, yhat, sigma, labels, native = cubes
    res = calibrate_model(y, yhat, sigma, _pi(method="conformal"), labels=labels, native=native)
    assert res[80]["method"] == "conformal"
    assert res[80]["native_metrics"]  # the native band is still reported


def test_method_native_keeps_bands(cubes):
    y, yhat, sigma, labels, native = cubes
    res = calibrate_model(y, yhat, sigma, _pi(method="native"), labels=labels, native=native, clip_lower=0.0)
    r = res[80]
    assert r["method"] == "native"
    assert r["q_lo"].shape == r["q_hi"].shape == (30, 5)
    assert not r["q_lo"].any() and not r["q_hi"].any()
    for k in r["metrics"]:
        np.testing.assert_allclose(r["metrics"][k], r["native_metrics"][k])


@pytest.mark.parametrize("method", ["native", "cqr"])
def test_native_methods_fall_back_without_bands(cubes, method, caplog):
    y, yhat, sigma, labels, _ = cubes
    with caplog.at_level("WARNING"):
        res = calibrate_model(y, yhat, sigma, _pi(method=method), labels=labels)
    assert res[80]["method"] == "conformal"
    assert "falling back" in caplog.text


def test_method_by_model_sets_the_method_of_each_model(cubes):
    y, yhat, sigma, labels, native = cubes
    pi = _pi(method="conformal", method_by_model={"Chronos2": "cqr", "StatsForecastAutoETS": "native"})

    def method(model):
        return calibrate_model(y, yhat, sigma, pi, labels=labels, native=native, clip_lower=0.0,
                                   model=model)[95]["method"]

    assert method("Chronos2") == "cqr"
    assert method("StatsForecastAutoETS") == "native"
    assert method("StatsForecastAutoArima") == "conformal"  # not listed: the method of the block
    assert method(None) == "conformal"


def test_method_by_model_matches_the_global_method_it_names(cubes):
    y, yhat, sigma, labels, native = cubes
    per_model = calibrate_model(y, yhat, sigma, _pi(method="conformal", method_by_model={"m": "cqr"}),
                                    labels=labels, native=native, clip_lower=0.0, model="m")
    global_cqr = calibrate_model(y, yhat, sigma, _pi(method="cqr"), labels=labels, native=native, clip_lower=0.0)
    for level in per_model:
        np.testing.assert_array_equal(per_model[level]["q_lo"], global_cqr[level]["q_lo"])
        np.testing.assert_array_equal(per_model[level]["q_hi"], global_cqr[level]["q_hi"])


def test_method_by_model_native_falls_back_without_bands(cubes, caplog):
    y, yhat, sigma, labels, _ = cubes
    pi = _pi(method="auto", method_by_model={"Chronos2": "native"})
    with caplog.at_level("WARNING"):
        res = calibrate_model(y, yhat, sigma, pi, labels=labels, model="Chronos2")
    assert res[80]["method"] == "conformal"
    assert "falling back" in caplog.text and "Chronos2" in caplog.text


def test_leave_one_window_out_metrics(cubes):
    y, yhat, sigma, labels, _ = cubes
    pi = _pi(coverage={"target": "group"})
    res = calibrate_model(y, yhat, sigma, pi, labels=labels, clip_lower=0.0)
    m = res[80]["metrics"]
    assert m["coverage"].shape == (30, 4)
    # Window 2 is scored with quantiles calibrated on windows 0, 1 and 3 only.
    scores = (y - yhat) / sigma[:, None, None]
    q_lo, q_hi, _ = residual_quantiles(scores[:, [0, 1, 3]], 80, "group", labels, min_scores_per_group=1)
    lo, hi = finalize_bounds(*residual_bounds(yhat[:, 2:3], sigma, q_lo, q_hi), 0.0)
    expected = interval_metrics(y[:, 2:3], lo, hi, sigma, 80)
    for k in expected:
        np.testing.assert_allclose(m[k][:, 2], expected[k])


def test_single_window_has_no_metrics(cubes):
    y, yhat, sigma, labels, _ = cubes
    res = calibrate_model(y[:, :1], yhat[:, :1], sigma, _pi(coverage={"target": "marginal"}))
    assert res[80]["metrics"] == {}


def test_signed_score_is_unscaled(cubes):
    y, yhat, sigma, labels, _ = cubes
    res = calibrate_model(y, yhat, sigma, _pi(score="signed", coverage={"target": "marginal"}))
    q_lo, q_hi, _ = residual_quantiles(y - yhat, 80, "marginal")
    np.testing.assert_array_equal(res[80]["q_hi"], q_hi)


def test_quantiles_setting_calibrates_their_central_levels(cubes):
    y, yhat, sigma, labels, _ = cubes
    res = calibrate_model(y, yhat, sigma, _pi(quantiles=[0.05, 0.5, 0.95]), labels=labels)
    assert set(res) == {90}


def test_rejects_bad_inputs(cubes):
    y, yhat, sigma, labels, native = cubes
    with pytest.raises(ValueError, match="sigma"):
        calibrate_model(y, yhat, np.zeros(30), _pi())
    with pytest.raises(ValueError, match="shape"):
        calibrate_model(y, yhat, sigma, _pi(), native={80: (native[80][0][:, :2], native[80][1][:, :2])})


@pytest.mark.parametrize("overrides, message", [
    ({"recency_decay": 0.9}, "recency_decay"),
    ({"coverage": {"level_group_col": "a", "levels_by_group": {"A": [95]}}}, "levels_by_group"),
    ({"score": "abs"}, "abs"),
])
def test_unsupported_settings(overrides, message):
    with pytest.raises(ConfigurationError, match=message):
        check_supported(_pi(**overrides))


def test_golden_default_settings_reproduce_notebook():
    """The default settings (auto -> CQR asymmetric, adaptive split, ABC x XYZ pool) on AutoETS, W=12."""
    g = np.load(GOLDEN)
    sigma, labels = g["sigma"], g["abc_xyz"].astype(object)
    yhat = np.nan_to_num(np.maximum(g["AutoETS_cal"], 0.0))
    native = {}
    for lv in (80, 95):
        lo, hi = g[f"AutoETS_lo_{lv}_cal"], g[f"AutoETS_hi_{lv}_cal"]
        native[lv] = (np.where(np.isfinite(lo), lo, yhat), np.where(np.isfinite(hi), hi, yhat))
    res = calibrate_model(g["Y_cal"], yhat, sigma, _pi(), labels=labels, native=native, clip_lower=0.0)
    for lv in (80, 95):
        q_lo, q_hi, _ = cqr_corrections(g["Y_cal"], yhat, np.maximum(native[lv][0], 0.0), native[lv][1],
                                            sigma, lv, pool="group", labels=labels, min_scores_per_group=1)
        np.testing.assert_allclose(res[lv]["q_lo"], q_lo, atol=1e-12)
        np.testing.assert_allclose(res[lv]["q_hi"], q_hi, atol=1e-12)
        # Out-of-sample coverage is near nominal even on 30 series.
        assert abs(res[lv]["metrics"]["coverage"].mean() - lv / 100) < 0.08


def _eval_rows():
    dates = pd.to_datetime(["2024-01-01", "2024-01-08", "2024-01-15"])
    rows = []
    for sid, n_windows in (("a", 3), ("b", 3), ("c", 2)):
        for d in dates[-n_windows:]:
            f = np.array([1.0, 2.0]) + (sid == "b")
            rows.append({"uid": sid, "backtest_window_start_date": d, "forecast": f, "actual": f + 0.5,
                         "forecast_quantiles": {"0.1": f - 1, "0.9": f + 1, "0.5": f}})
    frame = pd.DataFrame(rows)
    # Series "b" has a missing 95% key everywhere and its last window as (key, value) pairs.
    frame.at[5, "forecast_quantiles"] = [("0.1", [1.0, 2.0]), ("0.9", [3.0, 4.0])]
    return frame


def test_backtest_cubes():
    out = backtest_cubes(_eval_rows(), "uid", levels=(80, 95))
    assert list(out["ids"]) == ["a", "b"]
    assert out["dropped"] == 1  # "c" has 2 windows instead of 3
    assert out["y"].shape == out["yhat"].shape == (2, 3, 2)
    assert out["dates"].shape == (2, 3)
    lo80, hi80 = out["native"][80]
    np.testing.assert_array_equal(lo80[0, 0], [0.0, 1.0])
    np.testing.assert_array_equal(hi80[1, 2], [3.0, 4.0])  # read from the pair list
    lo95, hi95 = out["native"][95]
    np.testing.assert_array_equal(lo95, out["yhat"])  # missing keys fall back to the point forecast


def test_backtest_cubes_without_quantiles():
    out = backtest_cubes(_eval_rows().drop(columns="forecast_quantiles"), "uid", levels=(80,))
    assert out["native"] is None


def test_backtest_cubes_drops_non_finite_and_short_arrays():
    frame = _eval_rows()
    frame.at[0, "actual"] = np.array([np.nan, 1.0])
    frame.at[3, "forecast"] = np.array([1.0])
    out = backtest_cubes(frame, "uid")
    assert len(out["ids"]) == 0 and out["dropped"] == 3


def _features():
    f = pd.DataFrame({"sigma": [1.0, 2.0, 0.5], "volume": [10.0, 5.0, 1.0], "cv": [0.5, 1.0, 2.0],
                      "adi": [1.0, 2.0, 3.0], "cv2": [0.1, 0.2, 0.9], "store": ["s1", "s2", "s1"]},
                     index=pd.Index(["a", "b", "c"], name="uid"))
    return classify_series(f)


def test_group_labels():
    f = _features()
    assert list(group_labels(f, _pi())) == list(f["abc_xyz"])
    assert list(group_labels(f, _pi(coverage={"group_col": ["store", "sb_class"]}))) == [
        "s1|smooth", "s2|intermittent", "s1|lumpy"]
    assert group_labels(f, _pi(cqr_pool="marginal", coverage={"target": "marginal", "group_col": None})) is None
    assert group_columns(_pi(coverage={"group_col": ["store", "abc_class"]})) == ["store"]
    with pytest.raises(ConfigurationError, match="not found"):
        group_labels(f, _pi(coverage={"group_col": "region"}))


def test_positive_sigma():
    sigma, n = positive_sigma(np.array([2.0, 0.0, np.nan, 4.0]))
    np.testing.assert_array_equal(sigma, [2.0, 3.0, 3.0, 4.0])
    assert n == 2
    np.testing.assert_array_equal(positive_sigma(np.zeros(2))[0], [1.0, 1.0])


def test_output_rows(cubes):
    y, yhat, sigma, labels, native = cubes
    ids = np.array([f"s{i}" for i in range(30)], dtype=object)
    feats = classify_series(pd.DataFrame(
        {"sigma": sigma, "volume": np.arange(30.0) + 1, "cv": np.linspace(0.1, 3, 30), "adi": 1.0, "cv2": 0.1},
        index=ids))
    res = calibrate_model(y, yhat, sigma, _pi(), labels=labels, native=native, clip_lower=0.0)
    rows = calibration_rows("AutoETS", "uid", ids, res, feats, labels, 4, 1, 0.0)
    assert len(rows) == 30 * 5 * 2
    assert set(rows["method"]) == {"cqr"} and set(rows["pool_key"]) == {"A", "B"}
    assert rows["step"].min() == 1 and rows["step"].max() == 5
    first = rows[(rows["uid"] == "s0") & (rows["level"] == 80.0)].sort_values("step")
    np.testing.assert_array_equal(first["q_hi"], res[80]["q_hi"][0])
    dates = np.tile(pd.date_range("2024-01-01", periods=4, freq="7D").to_numpy(), (30, 1))
    mrows = metric_rows("AutoETS", "uid", ids, dates, res)
    assert set(mrows["metric_name"]) == {f"{p}{m}_{lv}" for p in ("", "native_") for m in ("coverage", "msis", "width")
                                         for lv in (80, 95)}
    assert len(mrows) == 30 * 4 * 12
