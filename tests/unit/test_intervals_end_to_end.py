"""End-to-end native prediction intervals through run_forecast, on a local Spark session.

Tests cover, with StatsForecastAutoETS (native intervals) and StatsForecastCrostonSBA (none):
- evaluation_output: sorted native quantiles only for the model that has them
- scoring_output: interval columns that agree with the quantile map, are nested and never cross for the native
  model, and are NULL for the other one, with one warning
- no calibration tables are written
- with intervals disabled, the tables have no interval columns

These tests need Spark Python workers (PySpark >= 3.5 on Python <= 3.11 locally); they are skipped otherwise.
"""
import contextlib
import logging

import mlflow
import numpy as np
import pandas as pd
import pytest
from pyspark.sql import functions as F

from mmf_sa import run_forecast

H, LEVELS = 7, (80, 95)
NATIVE, NO_NATIVE = "StatsForecastAutoETS", "StatsForecastCrostonSBA"
QUANTILE_KEYS = ["0.025", "0.1", "0.5", "0.9", "0.975"]


@contextlib.contextmanager
def _capture_warnings():
    """Warning records of the models and the Forecaster (pytest's caplog is function-scoped)."""
    records = []
    handler = logging.Handler(logging.WARNING)
    handler.emit = records.append
    root = logging.getLogger()
    root.addHandler(handler)
    try:
        yield records
    finally:
        root.removeHandler(handler)


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
        prediction_length=H, backtest_length=21, stride=7, train_predict_ratio=1, active_models=[NATIVE, NO_NATIVE],
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
    with _capture_warnings() as records:
        run_id = run_forecast(spark=spark, **kwargs)
    return kwargs, run_id, records


def _rows(spark, table, run_id):
    return spark.table(table).filter(F.col("run_id") == run_id).toPandas()


def test_backtest_rows_carry_sorted_native_quantiles_only_for_native_models(spark, run):
    kwargs, run_id, _ = run
    ev = _rows(spark, kwargs["evaluation_output"], run_id)
    native = ev[ev["model"] == NATIVE]
    assert len(native) and native["forecast_quantiles"].notna().all()
    for qmap in native["forecast_quantiles"]:
        qmap = dict(qmap)
        assert sorted(qmap, key=float) == QUANTILE_KEYS
        cube = np.stack([qmap[k] for k in QUANTILE_KEYS])
        assert cube.shape == (5, H) and (np.diff(cube, axis=0) >= 0).all()
    assert ev.loc[ev["model"] == NO_NATIVE, "forecast_quantiles"].isna().all()


def test_scoring_output_native_intervals_match_the_map_and_are_nested(spark, run):
    kwargs, run_id, _ = run
    score = _rows(spark, kwargs["scoring_output"], run_id)
    assert set(score["model"]) == {NATIVE, NO_NATIVE} and len(score) == 2 * 10
    assert "forecast_quantiles" not in score.columns
    for _, r in score[score["model"] == NATIVE].iterrows():
        lo80, hi80, lo95, hi95 = (np.asarray(r[c]) for c in ("y_lo_80", "y_hi_80", "y_lo_95", "y_hi_95"))
        assert all(len(a) == H for a in (lo80, hi80, lo95, hi95))
        assert (lo95 <= lo80).all() and (lo80 <= hi80).all() and (hi80 <= hi95).all()
        q = dict(r["y_quantiles"])
        assert sorted(q, key=float) == QUANTILE_KEYS
        np.testing.assert_array_equal(q["0.1"], lo80)
        np.testing.assert_array_equal(q["0.975"], hi95)


def test_models_without_native_intervals_get_null_columns_and_one_warning(spark, run):
    kwargs, run_id, records = run
    score = _rows(spark, kwargs["scoring_output"], run_id)
    other = score[score["model"] == NO_NATIVE]
    assert other["y"].notna().all()
    for c in ("y_quantiles", "y_lo_80", "y_hi_80", "y_lo_95", "y_hi_95"):
        assert other[c].isna().all()
    warnings = [r.getMessage() for r in records if "has no native prediction intervals" in r.getMessage()]
    assert warnings == [f"{NO_NATIVE} has no native prediction intervals; its interval columns are NULL."]


def test_no_calibration_tables_are_written(spark, run):
    kwargs, _, _ = run
    for suffix in ("_calibration", "_interval_metrics"):
        assert not spark.catalog.tableExists(kwargs["evaluation_output"] + suffix)


def test_disabled_intervals_add_no_columns(spark, tmp):
    kwargs = _kwargs(tmp, "pi_e2e_off")
    kwargs["prediction_intervals"] = {"enabled": False}
    run_id = run_forecast(spark=spark, **kwargs)
    assert "forecast_quantiles" not in spark.table(kwargs["evaluation_output"]).columns
    score = _rows(spark, kwargs["scoring_output"], run_id)
    assert len(score) == 2 * 10
    assert not [c for c in score.columns if c.startswith("y_") or c == "forecast_quantiles"]
