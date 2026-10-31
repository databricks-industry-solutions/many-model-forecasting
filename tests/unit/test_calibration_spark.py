"""Equality of the distributed calibration (mmf_sa.intervals.spark) with the numpy reference.

Tests cover, against conformal.calibrate_model on the same data:
- every method / pool / score: CQR asymmetric (adaptive and equal split), symmetric, scaled; residual
  conformal marginal, group, series (pooled horizon or not), signed score; native
- per-model methods (method_by_model), with and without native bands
- small groups merged into __other__, infinite ranks, ties, dropped series, missing native quantiles
- the M5 golden fixture with the default settings
- Spark classification against features.classify_series, with ties

These tests need Spark Python workers (PySpark >= 3.5 on Python <= 3.11 locally); they are skipped otherwise.
"""
import pathlib

import numpy as np
import pandas as pd
import pytest
from pyspark.sql import functions as F

from mmf_sa.intervals.config import level_quantiles, resolve_prediction_intervals_conf
from mmf_sa.intervals.quantiles import quantile_key
from mmf_sa.intervals.conformal import calibrate_model
from mmf_sa.intervals.features import classify_series, series_features
from mmf_sa.intervals.scoring import _metric_name, correction_column
from mmf_sa.intervals import spark as cs

GOLDEN = pathlib.Path(__file__).parent / "data" / "calibration_golden.npz"
TOL = 1e-12


@pytest.fixture(scope="module")
def spark(spark_session):
    spark_session.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
    spark_session.conf.set("spark.sql.shuffle.partitions", "4")
    try:
        spark_session.createDataFrame(pd.DataFrame({"a": [1]})).groupBy("a").applyInPandas(
            lambda p: p, "a long").collect()
    except Exception as err:  # PySpark 3.4 workers cannot start on Python 3.12
        pytest.skip(f"Spark Python workers unavailable: {type(err).__name__}")
    return spark_session


def _pi(**overrides):
    base = {"enabled": True, "coverage": {"min_scores_per_group": 1}}
    for k, v in overrides.items():
        if k == "coverage":
            base["coverage"].update(v)
        else:
            base[k] = v
    return resolve_prediction_intervals_conf(base)


def _data(n=24, w=4, h=5, seed=0, ties=False):
    rng = np.random.default_rng(seed)
    yhat = rng.gamma(2.0, 2.0, (n, w, h))
    y = np.maximum(yhat + rng.normal(0, 2, (n, w, h)), 0)
    if ties:
        y = np.round(y)
        yhat = np.round(yhat * 2) / 2
    sigma = rng.uniform(0.5, 2.0, n)
    labels = np.array(["A"] * (n // 2) + ["B"] * (n // 2 - 2) + ["C"] * 2, dtype=object)
    native = {lv: (yhat - 2.5 * lv / 80, yhat + 2.5 * lv / 80) for lv in (80, 95)}
    ids = np.array([f"s{i:03d}" for i in range(n)], dtype=object)
    return ids, y, yhat, sigma, labels, native


def _frames(spark, ids, y, yhat, sigma, labels, native, extra_rows=()):
    n, w, h = y.shape
    dates = pd.date_range("2024-01-01", periods=w, freq="7D")
    rows = []
    for i, sid in enumerate(ids):
        for j in range(w):
            q = None
            if native is not None:
                q = {}
                for lv, (lo, hi) in native.items():
                    k_lo, k_hi = (quantile_key(x) for x in level_quantiles(lv))
                    q[k_lo], q[k_hi] = [float(x) for x in lo[i, j]], [float(x) for x in hi[i, j]]
            rows.append({"uid": sid, "backtest_window_start_date": dates[j], "forecast": [float(x) for x in yhat[i, j]],
                         "actual": [float(x) for x in y[i, j]], "forecast_quantiles": q})
    rows += list(extra_rows)
    schema = ("uid string, backtest_window_start_date timestamp, forecast array<double>, actual array<double>, "
              "forecast_quantiles map<string,array<double>>")
    eval_sdf = spark.createDataFrame(pd.DataFrame(rows), schema)
    feats = pd.DataFrame({"uid": ids, "sigma": sigma, "abc_xyz": labels, "abc_class": "A", "xyz_class": "X",
                          "sb_class": "smooth"})
    return eval_sdf, spark.createDataFrame(feats)


def _compare(spark, ids, y, yhat, sigma, labels, native, pi, clip_lower=0.0, extra_rows=(), model="m"):
    eval_sdf, feats = _frames(spark, ids, y, yhat, sigma, labels, native, extra_rows)
    result, summary = cs.calibrate_model_spark(spark, eval_sdf, feats, "uid", model, pi, clip_lower,
                                               cs.Materializer(spark, "default.pi_tmp"))
    out = result.toPandas()
    result.unpersist()
    ref = calibrate_model(y, yhat, sigma, pi, labels=labels if pi.coverage.group_col else None,
                              native=native, clip_lower=clip_lower, model=model)
    rows = out[out["kind"] == "calibration"]
    metrics = out[out["kind"] == "metric"]
    assert set(out["model"]) == {model}
    for lv, r in ref.items():
        got = rows[rows["level"] == float(lv)].sort_values(["uid", "step"])
        assert len(got) == len(ids) * y.shape[2]
        np.testing.assert_allclose(got["q_lo"].to_numpy().reshape(len(ids), -1), r["q_lo"], rtol=0, atol=TOL)
        np.testing.assert_allclose(got["q_hi"].to_numpy().reshape(len(ids), -1), r["q_hi"], rtol=0, atol=TOL)
        np.testing.assert_array_equal(got.groupby("uid")["infinite_bound"].first().to_numpy(), r["infinite"])
        assert set(got["method"]) == {r["method"]} and set(got["coverage_target"]) == {r["target"]}
        for native_metric, values in ((False, r["metrics"]), (True, r["native_metrics"])):
            for name, arr in values.items():
                m = metrics[metrics["metric_name"] == _metric_name(name, lv, native_metric)]
                m = m.sort_values(["uid", "window_date"])
                np.testing.assert_allclose(m["metric_value"].to_numpy().reshape(len(ids), -1), arr,
                                           rtol=0, atol=TOL)
    return summary


@pytest.mark.parametrize("overrides", [
    {},                                                                    # CQR asymmetric adaptive, group pool
    {"cqr_tail_split": "equal"},
    {"cqr_pool": "series"},
    {"cqr_pool": "marginal"},
    {"cqr_score": "symmetric"},
    {"cqr_score": "scaled"},
    {"cqr_score": "scaled", "cqr_pool": "marginal"},
    {"method": "native"},
    {"method": "conformal"},                                               # series target, pooled horizon
    {"method": "conformal", "coverage": {"pool_horizon": False}},
    {"method": "conformal", "coverage": {"target": "group"}},
    {"method": "conformal", "coverage": {"target": "marginal"}},
    {"method": "conformal", "score": "signed"},
    {"method": "conformal", "coverage": {"min_scores_per_group": 30}},      # small groups merged
    {"coverage": {"min_scores_per_group": 30}},
    {"quantiles": [0.05, 0.5, 0.95]},
])
def test_spark_matches_numpy(spark, overrides):
    ids, y, yhat, sigma, labels, native = _data()
    pi = _pi(**overrides)
    if pi.quantiles:
        native = {90: (yhat - 3.0, yhat + 3.0)}
    _compare(spark, ids, y, yhat, sigma, labels, native, pi)


def test_spark_matches_numpy_without_native_bands(spark):
    ids, y, yhat, sigma, labels, _ = _data(seed=1)
    _compare(spark, ids, y, yhat, sigma, labels, None, _pi())  # falls back to conformal


@pytest.mark.parametrize("model, method", [("Chronos2", "cqr"), ("StatsForecastAutoETS", "native"),
                                           ("StatsForecastAutoArima", "conformal")])
def test_spark_matches_numpy_with_method_by_model(spark, model, method):
    ids, y, yhat, sigma, labels, native = _data(seed=9)
    pi = _pi(method="conformal", method_by_model={"Chronos2": "cqr", "StatsForecastAutoETS": "native"})
    assert _compare(spark, ids, y, yhat, sigma, labels, native, pi, model=model)["method"] == method


def test_spark_method_by_model_without_native_bands_falls_back(spark):
    ids, y, yhat, sigma, labels, _ = _data(seed=10)
    pi = _pi(method="conformal", method_by_model={"Chronos2": "cqr"})
    assert _compare(spark, ids, y, yhat, sigma, labels, None, pi, model="Chronos2")["method"] == "conformal"


def test_spark_matches_numpy_with_ties_and_no_clipping(spark):
    ids, y, yhat, sigma, labels, native = _data(seed=2, ties=True)
    _compare(spark, ids, y, yhat, sigma, labels, native, _pi(), clip_lower=None)
    _compare(spark, ids, y, yhat, sigma, labels, native, _pi(method="conformal"), clip_lower=None)


def test_spark_matches_numpy_with_few_windows_and_infinite_ranks(spark):
    ids, y, yhat, sigma, labels, native = _data(n=6, w=2, h=3, seed=3)
    _compare(spark, ids, y, yhat, sigma, labels, native, _pi(cqr_pool="series"))
    _compare(spark, ids, y, yhat, sigma, labels, native, _pi(method="conformal"))


def test_spark_drops_incomplete_series(spark):
    ids, y, yhat, sigma, labels, native = _data(seed=4)
    bad = [{"uid": "zz", "backtest_window_start_date": pd.Timestamp("2024-01-01"), "forecast": [1.0] * 5,
            "actual": [1.0] * 5, "forecast_quantiles": None}]
    eval_sdf, feats = _frames(spark, ids, y, yhat, sigma, labels, native, bad)
    feats = feats.unionByName(spark.createDataFrame(pd.DataFrame(
        {"uid": ["zz"], "sigma": [1.0], "abc_xyz": ["A"], "abc_class": "A", "xyz_class": "X", "sb_class": "smooth"})))
    result, summary = cs.calibrate_model_spark(spark, eval_sdf, feats, "uid", "m", _pi(), 0.0,
                                               cs.Materializer(spark, "default.pi_tmp"))
    assert summary["dropped"] == 1
    assert "zz" not in set(result.select("uid").distinct().toPandas()["uid"])
    result.unpersist()


def test_spark_golden_default_settings(spark):
    g = np.load(GOLDEN)
    sigma, labels = g["sigma"], g["abc_xyz"].astype(object)
    yhat = np.nan_to_num(np.maximum(g["AutoETS_cal"], 0.0))
    native = {}
    for lv in (80, 95):
        lo, hi = g[f"AutoETS_lo_{lv}_cal"], g[f"AutoETS_hi_{lv}_cal"]
        native[lv] = (np.where(np.isfinite(lo), lo, yhat), np.where(np.isfinite(hi), hi, yhat))
    ids = np.array([f"s{i:03d}" for i in range(len(sigma))], dtype=object)
    _compare(spark, ids, g["Y_cal"], yhat, sigma, labels, native, _pi())


def test_spark_classes_match_pandas(spark):
    rng = np.random.default_rng(5)
    n = 41
    feats = pd.DataFrame({
        "uid": [f"s{i:03d}" for i in range(n)],
        "sigma": 1.0,
        "volume": np.round(rng.gamma(1.0, 10.0, n)),            # ties in volume
        "cv": np.round(rng.uniform(0.1, 2.0, n), 1),            # ties in cv
        "adi": rng.uniform(1.0, 2.0, n),
        "cv2": rng.uniform(0.0, 1.0, n),
    })
    feats.loc[3, "cv"] = np.inf
    expected = classify_series(feats.set_index("uid"))
    got = cs.classify(spark.createDataFrame(feats), "uid").toPandas().set_index("uid").loc[expected.index]
    for c in ("sb_class", "abc_class", "xyz_class", "abc_xyz"):
        np.testing.assert_array_equal(got[c].to_numpy(), expected[c].to_numpy(), err_msg=c)


def test_spark_single_window_adaptive_cqr(spark):
    ids, y, yhat, sigma, labels, native = _data(n=12, w=1, h=4, seed=6)
    _compare(spark, ids, y, yhat, sigma, labels, native, _pi())


def test_spark_dataframes_with_the_same_plan_are_not_separate_scoring_data(spark):
    from omegaconf import OmegaConf
    from mmf_sa import Forecaster
    spark.createDataFrame(pd.DataFrame({"a": [1]})).write.mode("overwrite").saveAsTable("default.pi_same_source")
    df = spark.table("default.pi_same_source")
    conf = OmegaConf.create({"freq": "D", "backtest_length": 7, "active_models": []})
    same = Forecaster(conf=conf, data_conf={"train_data": df, "scoring_data": spark.table("default.pi_same_source")},
                      spark=spark, experiment_id="e")
    assert not same.has_separate_scoring_data()
    other = Forecaster(conf=conf, data_conf={"train_data": df, "scoring_data": df.filter("a > 0")}, spark=spark,
                       experiment_id="e")
    assert other.has_separate_scoring_data()


def test_scoring_reads_only_the_latest_calibration_of_a_run(spark, tmp_path):
    from omegaconf import OmegaConf
    from mmf_sa import Forecaster
    rows = []
    for run_date, q in ((pd.Timestamp("2024-01-01"), 1.0), (pd.Timestamp("2024-01-02"), 2.0)):
        for step in (1, 2):
            rows.append({"model": "m", "method": "conformal", "score": "signed_normalized", "coverage_target": "series",
                         "uid": "a", "pool_key": "__all__", "step": step, "level": 80.0, "q_lo": -q, "q_hi": q * step,
                         "sigma": 1.0, "clip_lower": 0.0, "run_id": "r", "run_date": run_date})
    spark.createDataFrame(pd.DataFrame(rows)).write.mode("overwrite").saveAsTable("default.pi_cal_rerun")
    conf = OmegaConf.create({"freq": "D", "backtest_length": 7, "active_models": [], "group_id": "uid",
                             "evaluation_output": "default.pi_eval", "calibration_output": "default.pi_cal_rerun",
                             "prediction_intervals": {"enabled": True}})
    f = Forecaster(conf=conf, data_conf={}, spark=spark, experiment_id="e", run_id="r")
    cal_rows = f.calibration_rows_for("m")
    wide = f.wide_corrections(cal_rows, [80]).toPandas()
    np.testing.assert_array_equal(wide[correction_column(0, "q_hi")].iloc[0], [2.0, 4.0])


@pytest.mark.parametrize("overrides", [{}, {"method": "conformal"}, {"cqr_pool": "series"}])
def test_spark_matches_numpy_across_many_buckets(spark, monkeypatch, overrides):
    monkeypatch.setattr(cs, "SERIES_PER_BUCKET", 5)  # 24 series -> 5 buckets
    ids, y, yhat, sigma, labels, native = _data(seed=7)
    _compare(spark, ids, y, yhat, sigma, labels, native, _pi(**overrides))


def test_spark_series_features_match_numpy(spark):
    rng = np.random.default_rng(8)
    rows, firsts, expected = [], [], {}
    start = pd.Timestamp("2024-01-01")
    for i, (n_days, zero_share) in enumerate([(60, 0.0), (60, 0.7), (40, 0.95), (5, 0.0), (60, 1.0), (30, 0.3)]):
        sid = f"s{i}"
        y = np.where(rng.random(n_days) < zero_share, 0.0, rng.poisson(4, n_days).astype(float))
        if i == 1:
            y[3] = np.nan  # a missing value counts as 0
        first = start + pd.Timedelta(days=n_days - 7)
        for t in range(n_days):
            rows.append({"uid": sid, "ds": start + pd.Timedelta(days=t), "y": y[t],
                         "store": None if t < 2 else f"st{i % 2}"})
        firsts.append({"uid": sid, "_first_window": first})
        hist = np.nan_to_num(y[: max(n_days - 7, 0)])[-30:]
        expected[sid] = series_features(hist, season_length=7)
    # A series whose history starts at its first window: no history, sigma 0.
    rows.append({"uid": "late", "ds": start + pd.Timedelta(days=50), "y": 3.0, "store": "st9"})
    firsts.append({"uid": "late", "_first_window": start + pd.Timedelta(days=50)})
    expected["late"] = series_features(np.array([]), season_length=7)
    train = spark.createDataFrame(pd.DataFrame(rows), "uid string, ds timestamp, y double, store string")
    windows = spark.createDataFrame(pd.DataFrame(firsts), "uid string, _first_window timestamp")
    got = cs.series_features(train, windows, "uid", "ds", "y", 7, 30, ["store"]).toPandas().set_index("uid")
    assert set(got.index) == set(expected)
    for sid, e in expected.items():
        for k in ("sigma", "volume", "cv", "adi", "cv2"):
            assert got.loc[sid, k] == pytest.approx(e[k], rel=1e-12, abs=1e-12), (sid, k)
    assert got.loc["s0", "store"] == "st0" and got.loc["late", "store"] == "st9"
