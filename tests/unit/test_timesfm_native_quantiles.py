"""Unit tests for native quantiles in the TimesFM pipeline.

timesfm is not installed in the test environment, so the model is faked. The fake returns outputs with the
shapes of TimesFM 2.5's forecast API: a point forecast (n, H) and a quantile forecast (n, H, 10), with the
mean first and then the deciles 0.1-0.9. Index k of the fake holds 100 * series + 10 * k + step, so deciles
are linear in k and interpolated values are easy to check.

Tests cover:
- quantile_forecasts: exact deciles, linear interpolation, clamping, mean excluded, warning logged once
- _predict_single: quantile columns, point forecast unchanged with and without quantiles, covariate path
- create_predict_udf: packing of the point and quantiles, round trip through decode_quantile_forecasts
- _predict_distributed: decoding of the packed UDF output on the driver
- calculate_metrics: 8th element carries the series' quantile map
"""
import sys
import types

import numpy as np
import pandas as pd
import pytest
import torch
from omegaconf import OmegaConf

from mmf_sa.intervals.quantiles import (
    QUANTILE_COLUMN_PREFIX, decode_quantile_forecasts, encode_quantile_forecasts, forecast_levels, quantile_key,
)
from mmf_sa.models.timesfmforecast import TimesFMPipeline as tfp
from mmf_sa.models.timesfmforecast.TimesFMPipeline import TimesFMForecaster, quantile_forecasts

QUANTILES = [0.025, 0.1, 0.5, 0.9, 0.975]
H = 4
N_OUT = 1 + len(tfp.TIMESFM_QUANTILE_LEVELS)  # the mean and the nine deciles
MEDIAN_COLUMN = QUANTILE_COLUMN_PREFIX + "0.5"


def _timesfm_output(n, horizon):
    """Quantile forecast of n series in TimesFM's (n, H, 10) layout: 100 * series + 10 * k + step."""
    s = np.arange(n)[:, None, None]
    k = np.arange(N_OUT)[None, None, :]
    h = np.arange(horizon)[None, :, None]
    return (100.0 * s + 10.0 * k + h).astype(np.float64)


def _expected(series, decile_index):
    """Expected values of a series at a position k (k = 10 * level) of the fake quantile output."""
    return 100.0 * series + 10.0 * decile_index + np.arange(H)


class _FakeTimesFM:
    """Stands in for TimesFM_2p5_200M_torch with the pinned API's forecast() and forecast_with_covariates()."""

    def __init__(self):
        self.model = types.SimpleNamespace(device=None, to=lambda device: None)
        self.compiled = []
        self.calls = []

    def compile(self, config):
        self.compiled.append(config)

    def forecast(self, horizon, inputs):
        self.calls.append("forecast")
        quantiles = _timesfm_output(len(inputs), horizon)
        return quantiles[..., 5], quantiles

    def forecast_with_covariates(self, inputs, **covariates):
        self.calls.append("forecast_with_covariates")
        quantiles = _timesfm_output(len(inputs), H)
        return [q[:, 5] for q in quantiles], list(quantiles)


@pytest.fixture
def fake_timesfm(monkeypatch):
    """Fake model, also installed as the timesfm module that the pipeline imports inside its functions."""
    model = _FakeTimesFM()
    module = types.ModuleType("timesfm")
    module.ForecastConfig = lambda **kwargs: types.SimpleNamespace(**kwargs)
    module.TimesFM_2p5_200M_torch = types.SimpleNamespace(from_pretrained=lambda repo: model)
    monkeypatch.setitem(sys.modules, "timesfm", module)
    return model


def _forecaster(model, native_quantiles=None, covariates=None):
    params = dict(group_id="uid", date_col="date", target="y", freq="D", prediction_length=H, metric="mae")
    if native_quantiles:
        params["native_quantiles"] = native_quantiles
    if covariates:
        params["dynamic_future_numerical"] = covariates
    forecaster = TimesFMForecaster(OmegaConf.create(params))
    forecaster.model = model  # _get_or_create_model returns an existing model without loading one
    return forecaster


def _history(n=20):
    frames = []
    for uid, base in [("a", 10.0), ("b", 50.0)]:
        dates = pd.date_range("2023-01-01", periods=n)
        frames.append(pd.DataFrame({"uid": uid, "date": dates, "y": base + np.arange(n, dtype=float)}))
    return pd.concat(frames, ignore_index=True)


def _future(n_hist=20, horizon=H):
    frames = []
    for uid in ["a", "b"]:
        dates = pd.date_range("2023-01-01", periods=n_hist + horizon)[n_hist:]
        frames.append(pd.DataFrame({"uid": uid, "date": dates, "y": 0.0}))
    return pd.concat(frames, ignore_index=True)


def _quantile_columns(levels):
    return [QUANTILE_COLUMN_PREFIX + quantile_key(level) for level in forecast_levels(levels)]


def test_quantile_forecasts_returns_exact_deciles():
    q = _timesfm_output(1, H)[0]
    out = quantile_forecasts(q, [0.1, 0.5, 0.9])
    np.testing.assert_array_equal(out, np.stack([q[:, 1], q[:, 5], q[:, 9]]))


def test_quantile_forecasts_interpolates_between_deciles():
    q = _timesfm_output(1, H)[0]
    out = quantile_forecasts(q, [0.15, 0.13, 0.85])
    np.testing.assert_allclose(out[0], q[:, 1] + 0.5 * (q[:, 2] - q[:, 1]))
    np.testing.assert_allclose(out[1], 0.7 * q[:, 1] + 0.3 * q[:, 2])
    np.testing.assert_allclose(out[2], q[:, 8] + 0.5 * (q[:, 9] - q[:, 8]))


def test_quantile_forecasts_clamps_levels_outside_deciles():
    q = _timesfm_output(1, H)[0]
    out = quantile_forecasts(q, [0.025, 0.1, 0.9, 0.975])
    np.testing.assert_array_equal(out, np.stack([q[:, 1], q[:, 1], q[:, 9], q[:, 9]]))


def test_quantile_forecasts_ignores_the_mean():
    q = _timesfm_output(1, H)[0]
    q[:, 0] = 1e6  # the mean (index 0) is not a quantile
    assert quantile_forecasts(q, [0.1, 0.5, 0.9]).max() < 1e3


def test_clamp_warning_logged_once(caplog):
    with caplog.at_level("WARNING"):
        for _ in range(3):
            assert _forecaster(None, QUANTILES).native_levels() == QUANTILES
        _forecaster(None, [0.1, 0.5, 0.9]).native_levels()  # inside the deciles: no warning
    warnings = [r.message for r in caplog.records if "clamped" in r.message]
    assert len(warnings) == 1
    assert "[0.1, 0.9]" in warnings[0] and "method 'auto'" in warnings[0]


def test_predict_single_point_forecast_unchanged_by_quantiles(fake_timesfm):
    hist = _history()
    plain, _ = _forecaster(fake_timesfm).predict(hist)
    with_q, _ = _forecaster(fake_timesfm, QUANTILES).predict(hist)
    assert not [c for c in plain.columns if str(c).startswith(QUANTILE_COLUMN_PREFIX)]
    assert list(with_q.columns) == list(plain.columns) + _quantile_columns(QUANTILES)
    for series in range(2):
        # The point forecast is the 0.5 decile (index 5) of the fake output, as in TimesFM.
        np.testing.assert_array_equal(plain["y"].iloc[series], _expected(series, 5))
        np.testing.assert_array_equal(with_q["y"].iloc[series], plain["y"].iloc[series])


def test_predict_single_quantile_columns_have_the_right_values(fake_timesfm):
    forecast_df, _ = _forecaster(fake_timesfm, QUANTILES + [0.15]).predict(_history())
    assert MEDIAN_COLUMN in forecast_df.columns
    for series in range(2):
        np.testing.assert_array_equal(forecast_df["__q_0.025"].iloc[series], _expected(series, 1))  # clamped to 0.1
        np.testing.assert_array_equal(forecast_df["__q_0.1"].iloc[series], _expected(series, 1))
        np.testing.assert_array_equal(forecast_df["__q_0.15"].iloc[series], _expected(series, 1.5))
        np.testing.assert_array_equal(forecast_df["__q_0.5"].iloc[series], _expected(series, 5))
        np.testing.assert_array_equal(forecast_df["__q_0.9"].iloc[series], _expected(series, 9))
        np.testing.assert_array_equal(forecast_df["__q_0.975"].iloc[series], _expected(series, 9))  # clamped to 0.9
        assert forecast_df["__q_0.9"].iloc[series].shape == (H,)


def test_predict_single_covariate_path_returns_quantiles(fake_timesfm):
    hist = _history()
    hist["x"] = np.arange(len(hist), dtype=float)
    future = _future()
    future["x"] = 1.0
    plain, _ = _forecaster(fake_timesfm, covariates=["x"]).predict(hist, future)
    with_q, _ = _forecaster(fake_timesfm, QUANTILES, covariates=["x"]).predict(hist, future)
    assert "forecast_with_covariates" in fake_timesfm.calls
    assert list(with_q.columns) == list(plain.columns) + _quantile_columns(QUANTILES)
    for series in range(2):
        np.testing.assert_array_equal(with_q["y"].iloc[series], plain["y"].iloc[series])
        np.testing.assert_array_equal(with_q["__q_0.9"].iloc[series], _expected(series, 9))


def _packed(levels, n=3):
    """UDF output for n series: point and quantiles packed by encode_quantile_forecasts."""
    quantiles = np.stack([quantile_forecasts(q, levels) for q in _timesfm_output(n, H)])
    return encode_quantile_forecasts(_timesfm_output(n, H)[..., 5], quantiles)


@pytest.mark.parametrize("with_covariates", [False, True])
def test_udf_packs_point_and_quantiles_round_trip(fake_timesfm, monkeypatch, with_covariates):
    monkeypatch.setattr(torch.cuda, "set_device", lambda device: None)
    forecaster = _forecaster(fake_timesfm, QUANTILES, covariates=["x"] if with_covariates else None)
    udf = forecaster.create_predict_udf(1, [("dyn_num", "x")] if with_covariates else None)
    y = pd.Series([np.arange(20, dtype=float) + 10 * s for s in range(3)])
    batch = (y, pd.Series([np.arange(20 + H, dtype=float)] * 3)) if with_covariates else y
    (packed,) = list(udf.func(iter([batch])))

    assert len(packed[0]) == (1 + len(forecast_levels(QUANTILES))) * H
    frame = decode_quantile_forecasts(pd.DataFrame({"y": list(packed)}), "y", forecast_levels(QUANTILES), H)
    for series in range(3):
        np.testing.assert_array_equal(frame["y"].iloc[series], _expected(series, 5))
        np.testing.assert_array_equal(frame["__q_0.025"].iloc[series], _expected(series, 1))
        np.testing.assert_array_equal(frame["__q_0.9"].iloc[series], _expected(series, 9))
    np.testing.assert_array_equal(np.stack(list(frame[MEDIAN_COLUMN])), np.stack(list(frame["y"])))


def test_udf_without_quantiles_returns_point_forecasts_only(fake_timesfm, monkeypatch):
    monkeypatch.setattr(torch.cuda, "set_device", lambda device: None)
    udf = _forecaster(fake_timesfm).create_predict_udf(1)
    (out,) = list(udf.func(iter([pd.Series([np.arange(20, dtype=float)] * 2)])))
    assert all(len(v) == H for v in out)
    np.testing.assert_array_equal(np.stack(list(out)), np.stack([_expected(s, 5) for s in range(2)]))


class _FakeColumn:
    def alias(self, name):
        return self


class _FakeSpark:
    """Enough of a Spark DataFrame for _predict_distributed: columns are placeholders and the
    repartition().select().toPandas() chain returns the packed UDF output."""

    def __init__(self, packed):
        self.packed = packed

    def __getattr__(self, name):
        return _FakeColumn()

    def __getitem__(self, name):
        return _FakeColumn()

    def repartition(self, *args):
        return self

    def select(self, *columns):
        return self

    def toPandas(self):
        return self.packed


def test_predict_distributed_decodes_packed_quantiles(fake_timesfm):
    forecaster = _forecaster(fake_timesfm, QUANTILES)
    packed = pd.DataFrame({
        "unique_id": ["a", "b"],
        "ds": [np.array([0] * H), np.array([0] * H)],
        "y": _packed(forecast_levels(QUANTILES), n=2),
    })
    forecaster.prepare_data_for_spark = lambda hist_df, spark, val_df=None: _FakeSpark(packed)
    forecaster.create_horizon_timestamps_udf = lambda: lambda column: _FakeColumn()
    forecaster.create_predict_udf = lambda device_count, col_map=None: lambda *columns: _FakeColumn()

    forecast_df, _ = forecaster._predict_distributed(_history(), spark=object())

    assert list(forecast_df["uid"]) == ["a", "b"]
    for series in range(2):
        np.testing.assert_array_equal(forecast_df["y"].iloc[series], _expected(series, 5))
        np.testing.assert_array_equal(forecast_df["__q_0.025"].iloc[series], _expected(series, 1))
        np.testing.assert_array_equal(forecast_df["__q_0.975"].iloc[series], _expected(series, 9))


def test_calculate_metrics_carries_quantile_map(fake_timesfm):
    val = _future()
    val["y"] = [1.0, 2.0, 3.0, 4.0] * 2
    metrics = _forecaster(fake_timesfm, QUANTILES).calculate_metrics(
        _history(), val, curr_date=pd.Timestamp("2023-01-21"))
    assert [m[0] for m in metrics] == ["a", "b"]
    for series, metric in enumerate(metrics):
        assert len(metric) == 8
        quantile_map = metric[7]
        assert set(quantile_map) == {quantile_key(q) for q in forecast_levels(QUANTILES)}
        np.testing.assert_array_equal(quantile_map["0.9"], _expected(series, 9))
        np.testing.assert_array_equal(quantile_map["0.5"], metric[4])

    plain = _forecaster(fake_timesfm).calculate_metrics(_history(), val, curr_date=pd.Timestamp("2023-01-21"))
    assert all(len(metric) == 8 and metric[7] is None for metric in plain)
