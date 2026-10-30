"""Unit tests for native quantiles in the backtest (prediction intervals).

Tests cover:
- Quantile keys, requested quantiles and their central levels
- statsforecast: native quantiles in the backtest, unchanged point forecasts
- The forecast_quantiles column of the local evaluation UDF, with intervals on and off
- Which models receive native_quantiles (Forecaster.native_quantiles)
- Chronos UDF packing of medians and quantiles, and per-series quantile maps
- The Spark schema of evaluation rows written by backtest_global_model
"""
import mlflow
import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

from mmf_sa import Forecaster
from mmf_sa import calibration as cal
from mmf_sa.models.chronosforecast.ChronosPipeline import (
    decode_quantile_forecasts,
    encode_quantile_forecasts,
    quantile_maps,
)
from mmf_sa.models.statsforecast.StatsFcForecastingPipeline import StatsFcAutoETS

QUANTILES = [0.025, 0.1, 0.5, 0.9, 0.975]


def test_quantile_helpers():
    assert [cal.quantile_key(q) for q in QUANTILES] == ["0.025", "0.1", "0.5", "0.9", "0.975"]
    assert cal.level_quantiles(80) == (0.1, 0.9)
    pi = cal.resolve_prediction_intervals_conf({"enabled": True})
    assert cal.requested_quantiles(pi) == QUANTILES
    assert cal.central_levels(QUANTILES) == [80, 95]
    by_group = cal.resolve_prediction_intervals_conf(
        {"levels": [80], "coverage": {"level_group_col": "abc", "levels_by_group": {"A": [95], "C": [50]}}})
    assert cal.requested_quantiles(by_group) == [0.025, 0.1, 0.25, 0.5, 0.75, 0.9, 0.975]
    assert cal.requested_quantiles(cal.resolve_prediction_intervals_conf({"quantiles": [0.05, 0.95]})) == [
        0.05, 0.5, 0.95]
    assert cal.central_levels([0.0125]) == [97.5]


def _series(n=120, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"uid": "a", "date": pd.date_range("2023-01-01", periods=n),
                         "y": rng.poisson(3, n).astype(float)})


def _ets(quantiles=None):
    params = dict(group_id="uid", date_col="date", target="y", freq="D", prediction_length=7, metric="smape",
                  stride=7, backtest_length=14, model_spec={"season_length": 7, "model": "ZNZ"})
    if quantiles:
        params["native_quantiles"] = quantiles
    return StatsFcAutoETS(OmegaConf.create(params))


def test_statsforecast_backtest_native_quantiles():
    df = _series()
    start = df.date.max() - pd.Timedelta(days=14)
    plain = _ets().backtest(df, start=start, group_id="a")
    with_q = _ets(QUANTILES).backtest(df, start=start, group_id="a")
    assert plain["forecast_quantiles"].isna().all()
    for a, b in zip(plain["forecast"], with_q["forecast"]):
        np.testing.assert_array_equal(a, b)
    for fq, point in zip(with_q["forecast_quantiles"], with_q["forecast"]):
        assert list(fq) == ["0.025", "0.1", "0.5", "0.9", "0.975"]
        stacked = np.stack([fq[k] for k in fq])
        assert (np.diff(stacked, axis=0) >= 0).all()          # quantiles are ordered
        np.testing.assert_allclose(np.maximum(fq["0.5"], 0), point)  # the point forecast is the clipped median
        assert len(fq["0.1"]) == 7


@pytest.mark.parametrize("with_quantiles", [False, True])
def test_local_evaluation_udf_columns(with_quantiles):
    out = Forecaster.evaluate_one_local_model(_series(), _ets(QUANTILES), with_quantiles=with_quantiles)
    assert ("forecast_quantiles" in out.columns) == with_quantiles
    assert len(out) == 2


def _forecaster(**pi):
    conf = OmegaConf.create({"freq": "D", "backtest_length": 14, "evaluation_output": "t",
                             "active_models": [], "prediction_intervals": pi})
    return Forecaster(conf=conf, data_conf={}, spark=None, experiment_id="e")


def test_native_quantiles_per_model():
    native = OmegaConf.create({"name": "m", "native_intervals": True})
    plain = OmegaConf.create({"name": "m"})
    assert _forecaster(enabled=True).native_quantiles(native) == QUANTILES
    assert _forecaster(enabled=True).native_quantiles(plain) is None
    assert _forecaster(enabled=True, method="conformal").native_quantiles(native) is None
    assert _forecaster(enabled=False).native_quantiles(native) is None


def test_chronos_udf_packing_round_trip():
    rng = np.random.default_rng(1)
    medians = rng.random((3, 4))
    qs = np.sort(rng.random((3, 5, 4)), axis=1)
    frame = pd.DataFrame({"unique_id": ["a", "b", "c"], "y": encode_quantile_forecasts(medians, qs)})
    out = decode_quantile_forecasts(frame, "y", QUANTILES, prediction_length=4)
    np.testing.assert_array_equal(np.stack(out["y"]), medians)
    for i, key in enumerate(["0.025", "0.1", "0.5", "0.9", "0.975"]):
        np.testing.assert_array_equal(np.stack(out[cal.QUANTILE_COLUMN_PREFIX + key]), qs[:, i, :])
    maps = quantile_maps(out.rename(columns={"unique_id": "uid"}), "uid")
    np.testing.assert_array_equal(maps["b"]["0.9"], qs[1, 3, :])
    assert quantile_maps(pd.DataFrame({"uid": ["a"], "y": [[1.0]]}), "uid") == {}


class _CapturedWrite:
    def __init__(self, sdf):
        type(self).sdf = sdf

    def saveAsTable(self, name):
        pass


class _FakeModel:
    def __init__(self, res_pdf):
        self.res_pdf = res_pdf
        self.params = OmegaConf.create({"name": "m"})

    def backtest(self, df, start, spark=None):
        return self.res_pdf.copy()


@pytest.mark.parametrize("enabled", [False, True])
def test_global_backtest_schema(spark_session, monkeypatch, enabled):
    res_pdf = pd.DataFrame({
        "uid": ["a", "b"],
        "backtest_window_start_date": pd.to_datetime(["2024-01-01", "2024-01-01"]),
        "metric_name": ["smape", "smape"],
        "metric_value": [0.1, 0.2],
        "forecast": [np.array([1.0, 2.0]), np.array([3.0, 4.0])],
        "actual": [np.array([1.0, 2.0]), np.array([3.0, 5.0])],
        "model_pickle": [b"", b""],
        "forecast_quantiles": [{"0.1": np.array([0.5, 1.5]), "0.9": np.array([1.5, 2.5])}, None],
    })
    conf = OmegaConf.create({"freq": "D", "backtest_length": 2, "group_id": "uid", "date_col": "ds",
                             "evaluation_output": "t", "use_case_name": "u", "active_models": [],
                             "prediction_intervals": {"enabled": enabled}})
    f = Forecaster(conf=conf, data_conf={}, spark=spark_session, experiment_id="e")
    # Arrow builds the frame in the JVM, so no Python worker is needed (PySpark 3.4 workers fail on Python 3.12).
    spark_session.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
    spark_session.conf.set("spark.sql.execution.arrow.pyspark.fallback.enabled", "false")
    monkeypatch.setattr(Forecaster, "evaluation_writer", lambda self, sdf: _CapturedWrite(sdf))
    monkeypatch.setattr(mlflow, "log_metric", lambda *a, **k: None)
    train = pd.DataFrame({"uid": ["a", "b"], "ds": pd.to_datetime(["2023-12-31"] * 2), "y": [1.0, 2.0]})
    try:
        f.backtest_global_model(_FakeModel(res_pdf), train, train.iloc[:0], model_uri="", write=True)
        rows = {r["uid"]: r for r in _CapturedWrite.sdf.collect()}
    finally:
        spark_session.conf.unset("spark.sql.execution.arrow.pyspark.enabled")
        spark_session.conf.unset("spark.sql.execution.arrow.pyspark.fallback.enabled")
    assert ("forecast_quantiles" in _CapturedWrite.sdf.columns) == enabled
    if enabled:
        assert rows["a"]["forecast_quantiles"] == {"0.1": [0.5, 1.5], "0.9": [1.5, 2.5]}
        assert rows["b"]["forecast_quantiles"] is None


def test_forecast_levels_match_requested_quantiles():
    from mmf_sa.models.chronosforecast.ChronosPipeline import forecast_levels
    pi = cal.resolve_prediction_intervals_conf({"enabled": True})
    assert forecast_levels(cal.requested_quantiles(pi)) == QUANTILES
    assert forecast_levels([0.9, 0.1]) == [0.1, 0.5, 0.9]


class _FakePipeline:
    quantiles = [0.1, 0.5, 0.9]

    def predict_quantiles(self, inputs, prediction_length, quantile_levels):
        import torch
        n = len(inputs)
        q = torch.arange(n * prediction_length * len(quantile_levels), dtype=torch.float32)
        return q.reshape(n, prediction_length, len(quantile_levels)), None


def test_clamp_warning_logged_once(caplog):
    import torch
    from mmf_sa.models.chronosforecast import ChronosPipeline as cp
    cp._warned_clamped.clear()
    with caplog.at_level("WARNING", logger=cp.__name__):
        for _ in range(3):
            medians, qs, levels = cp.quantile_forecasts(_FakePipeline(), [torch.zeros(5)] * 2, 4, QUANTILES)
    assert sum("trained range" in r.message for r in caplog.records) == 1
    assert medians.shape == (2, 4) and qs.shape == (2, 5, 4) and levels == QUANTILES
    np.testing.assert_array_equal(medians, qs[:, 2, :])
