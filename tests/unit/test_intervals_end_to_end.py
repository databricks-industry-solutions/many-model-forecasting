"""End-to-end prediction intervals through run_forecast, on a local Spark session.

Tests cover, with StatsForecastAutoETS (native bands, CQR) and StatsForecastCrostonSBA (conformal):
- evaluation_output: native quantiles only for the model that has them
- calibration_output: the method of each model, one row per series, step and level
- interval_metrics_output: coverage, MSIS and width, plus native_* only for native bands
- scoring_output: filled, clipped, uncrossed and nested interval columns that agree with the quantile map
- a scoring-only run reuses the calibration of its run_id, and writes null intervals without one
- a model whose calibration fails gets null intervals while the others are calibrated; method_by_model

These tests need Spark Python workers (PySpark >= 3.5 on Python <= 3.11 locally); they are skipped otherwise.
"""
import importlib.resources
import logging

import mlflow
import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf
from pyspark.sql import functions as F

from mmf_sa import Forecaster, run_forecast
from mmf_sa.intervals import spark as calibration_spark

H, LEVELS = 7, (80, 95)
NATIVE, CONFORMAL = "StatsForecastAutoETS", "StatsForecastCrostonSBA"


@pytest.fixture(scope="module")
def spark(spark_session):
    spark_session.conf.set("spark.sql.shuffle.partitions", "4")
    try:
        spark_session.createDataFrame(pd.DataFrame({"a": [1]})).groupBy("a").applyInPandas(
            lambda p: p, "a long").collect()
    except Exception as err:  # PySpark 3.4 workers cannot start on Python 3.12
        pytest.skip(f"Spark Python workers unavailable: {type(err).__name__}")
    return spark_session


def _train(n_series=10, n_days=98, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n_days, freq="D")
    frames = []
    for i in range(n_series):
        y = np.maximum(5 + i + 2 * np.sin(np.arange(n_days) * 2 * np.pi / 7) + rng.normal(0, 1, n_days), 0)
        if i % 3 == 0:
            y = np.where(rng.random(n_days) < 0.6, 0.0, y)  # intermittent series
        frames.append(pd.DataFrame({"unique_id": f"s{i}", "ds": dates, "y": y}))
    return pd.concat(frames, ignore_index=True)


def _kwargs(tmp, prefix, **intervals):
    return dict(
        train_data="pi_e2e_train", scoring_data="pi_e2e_train", scoring_output=f"default.{prefix}_scoring",
        evaluation_output=f"default.{prefix}_eval", group_id="unique_id", date_col="ds", target="y", freq="D",
        prediction_length=H, backtest_length=21, stride=7, train_predict_ratio=1, active_models=[NATIVE, CONFORMAL],
        experiment_path=f"{tmp}/{prefix}", use_case_name=prefix,
        prediction_intervals={"enabled": True, "output_format": "both", **intervals},
    )


@pytest.fixture(scope="module")
def tmp(tmp_path_factory, spark):
    path = tmp_path_factory.mktemp("pi_e2e")
    mlflow.set_tracking_uri(f"sqlite:///{path}/mlruns.db")
    spark.createDataFrame(_train()).createOrReplaceTempView("pi_e2e_train")
    return path


@pytest.fixture(scope="module")
def run(spark, tmp):
    kwargs = _kwargs(tmp, "pi_e2e")
    return kwargs, run_forecast(spark=spark, **kwargs)


def _rows(spark, table, run_id):
    return spark.table(table).filter(F.col("run_id") == run_id).toPandas()


def test_backtest_rows_carry_native_quantiles_only_for_native_bands(spark, run):
    kwargs, run_id = run
    ev = _rows(spark, kwargs["evaluation_output"], run_id)
    native = ev[ev["model"] == NATIVE]
    assert len(native) and native["forecast_quantiles"].notna().all()
    assert sorted(dict(native["forecast_quantiles"].iloc[0]), key=float) == ["0.025", "0.1", "0.5", "0.9", "0.975"]
    assert ev.loc[ev["model"] == CONFORMAL, "forecast_quantiles"].isna().all()


def test_calibration_output_has_the_method_of_each_model(spark, run):
    kwargs, run_id = run
    cal = _rows(spark, kwargs["evaluation_output"] + "_calibration", run_id)
    methods = cal.groupby("model")[["method", "score", "coverage_target"]].first()
    assert tuple(methods.loc[NATIVE]) == ("cqr", "asymmetric", "group")
    assert tuple(methods.loc[CONFORMAL]) == ("conformal", "signed_normalized", "series")
    for model in (NATIVE, CONFORMAL):
        m = cal[cal["model"] == model]
        assert len(m) == 10 * H * len(LEVELS) and m["sigma"].gt(0).all()
        assert not m.duplicated(["unique_id", "step", "level"]).any()


def test_interval_metrics_are_named_per_level_and_native_only_with_bands(spark, run):
    kwargs, run_id = run
    metrics = _rows(spark, kwargs["evaluation_output"] + "_interval_metrics", run_id)
    names = metrics.groupby("model")["metric_name"].unique().map(set)
    plain = {f"{m}_{lv}" for m in ("coverage", "msis", "width") for lv in LEVELS}
    assert names[CONFORMAL] == plain
    assert names[NATIVE] == plain | {f"native_{n}" for n in plain}
    coverage = metrics[metrics["metric_name"].str.contains("coverage")]["metric_value"]
    assert coverage.between(0, 1).all()


def test_scoring_output_intervals_are_filled_clipped_and_nested(spark, run):
    kwargs, run_id = run
    score = _rows(spark, kwargs["scoring_output"], run_id)
    assert set(score["model"]) == {NATIVE, CONFORMAL} and len(score) == 2 * 10
    for _, r in score.iterrows():
        lo80, hi80, lo95, hi95 = (np.asarray(r[c]) for c in ("y_lo_80", "y_hi_80", "y_lo_95", "y_hi_95"))
        assert all(len(a) == H for a in (lo80, hi80, lo95, hi95))
        assert (lo80 >= 0).all() and (lo80 <= hi80).all()  # clip_lower auto: the target is never negative
        assert (lo95 <= lo80 + 1e-9).all() and (hi95 >= hi80 - 1e-9).all()
        q = dict(r["y_quantiles"])
        np.testing.assert_allclose(q["0.1"], lo80)
        np.testing.assert_allclose(q["0.975"], hi95)
        # The native median is kept for models with native bands; conformal intervals have none.
        assert ("0.5" in q) == (r["model"] == NATIVE)


def _scoring_only(spark, kwargs, run_id):
    base = OmegaConf.create(importlib.resources.files("mmf_sa").joinpath("forecasting_conf_daily.yaml").read_text())
    conf = OmegaConf.merge(base, OmegaConf.create({**kwargs, "run_id": run_id, "metric": "smape", "resample": False}))
    Forecaster(conf=conf, data_conf={}, spark=spark, run_id=run_id).evaluate_score(evaluate=False)


def test_scoring_only_reuses_the_calibration_of_its_run_id(spark, run):
    kwargs, run_id = run
    first = _rows(spark, kwargs["scoring_output"], run_id)
    _scoring_only(spark, kwargs, run_id)
    both = _rows(spark, kwargs["scoring_output"], run_id)
    assert len(both) == 2 * len(first)
    second = both.sort_values("run_date").groupby(["model", "unique_id"]).last()
    first = first.set_index(["model", "unique_id"]).loc[second.index]
    for col in ("y_lo_80", "y_hi_95"):
        np.testing.assert_allclose(np.stack(second[col]), np.stack(first[col]))


def test_scoring_only_without_calibration_writes_null_intervals(spark, run):
    kwargs, _ = run
    kwargs = dict(kwargs, scoring_output="default.pi_e2e_uncalibrated_scoring")
    _scoring_only(spark, kwargs, "no-calibration-for-this-run")
    score = _rows(spark, kwargs["scoring_output"], "no-calibration-for-this-run")
    assert len(score) == 2 * 10
    assert score["y"].notna().all() and score["y_lo_80"].isna().all() and score["y_quantiles"].isna().all()


def test_failed_calibration_is_isolated_and_method_by_model_applies(spark, tmp, monkeypatch, caplog):
    real = calibration_spark.calibrate_model_spark

    def failing(spark_, eval_sdf, features, series_col, model, *args, **kw):
        if model == CONFORMAL:
            raise RuntimeError("calibration exploded")
        return real(spark_, eval_sdf, features, series_col, model, *args, **kw)

    monkeypatch.setattr(calibration_spark, "calibrate_model_spark", failing)
    kwargs = _kwargs(tmp, "pi_e2e_fail", method_by_model={NATIVE: "conformal"})
    with caplog.at_level(logging.ERROR):
        run_id = run_forecast(spark=spark, **kwargs)
    assert any("Calibration failed for StatsForecastCrostonSBA" in r.getMessage() for r in caplog.records)
    ev = _rows(spark, kwargs["evaluation_output"], run_id)
    assert ev.loc[ev["model"] == NATIVE, "forecast_quantiles"].isna().all()  # conformal: no native quantiles
    cal = _rows(spark, kwargs["evaluation_output"] + "_calibration", run_id)
    assert set(cal["model"]) == {NATIVE} and set(cal["method"]) == {"conformal"}
    score = _rows(spark, kwargs["scoring_output"], run_id).set_index("model")
    assert score.loc[NATIVE, "y_lo_80"].notna().all()
    assert score.loc[CONFORMAL, "y_lo_80"].isna().all() and score.loc[CONFORMAL, "y"].notna().all()
