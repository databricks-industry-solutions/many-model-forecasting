"""Backtest metric rows shared by all models (ForecastingRegressor.series_metrics and backtest).

Tests cover:
- metric_function: every supported metric, and the error for an unknown one
- series_metrics on wide forecasts (one row per series, with native quantiles) and long forecasts
  (one row per series and step); series that cannot be scored are skipped
- backtest: EvaluationRow columns, and plain 7-tuples from custom models still accepted
"""
import logging

import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

from mmf_sa.intervals.quantiles import QUANTILE_COLUMN_PREFIX
from mmf_sa.exceptions import UnsupportedMetricError
from mmf_sa.models.abstract_model import EvaluationRow, ForecastingRegressor, metric_function

H = 3


def _model(metric="mae", **params):
    return ForecastingRegressor(OmegaConf.create({
        "freq": "D", "prediction_length": H, "group_id": "unique_id", "date_col": "ds", "target": "y",
        "metric": metric, "stride": 1, **params}))


def _actuals(ids=("a", "b")):
    dates = pd.date_range("2024-01-10", periods=H, freq="D")
    return pd.DataFrame({"unique_id": np.repeat(ids, H), "ds": np.tile(dates, len(ids)),
                         "y": np.arange(len(ids) * H, dtype=float)})


@pytest.mark.parametrize("name, expected", [
    ("mae", 1.0), ("mse", 1.0), ("rmse", 1.0), ("mape", 0.5), ("smape", (2 / 3 + 2 / 5) / 2),
])
def test_metric_function(name, expected):
    actual, forecast = np.array([2.0, 2.0]), np.array([1.0, 3.0])
    assert metric_function(name)(actual, forecast) == pytest.approx(expected)


def test_metric_function_rejects_unknown_metrics():
    with pytest.raises(UnsupportedMetricError, match="Metric wape not supported"):
        metric_function("wape")


def test_series_metrics_on_wide_forecasts_with_native_quantiles():
    val = _actuals()
    pred = pd.DataFrame({"unique_id": ["a", "b"], "y": [np.array([1.0, 1.0, 3.0]), np.array([3.0, 4.0, 5.0])],
                         QUANTILE_COLUMN_PREFIX + "0.9": [np.array([2.0, 2.0, 4.0]), np.array([4.0, 5.0, 6.0])]})
    rows = _model().series_metrics(pred, val, pd.Timestamp("2024-01-10"))
    assert [r.key for r in rows] == ["a", "b"]
    a, b = rows
    assert isinstance(a, EvaluationRow) and len(a) == 8
    assert a.metric_name == "mae" and a.metric_value == pytest.approx(np.mean([1, 0, 1]))
    assert b.metric_value == pytest.approx(0.0)
    np.testing.assert_array_equal(a.actual, [0.0, 1.0, 2.0])
    assert a.forecast.dtype == np.float64 and a.model_pickle == b""
    assert list(a.forecast_quantiles) == ["0.9"]
    np.testing.assert_allclose(a.forecast_quantiles["0.9"], [2.0, 2.0, 4.0])


def test_series_metrics_on_long_forecasts_uses_the_last_prediction_length_steps():
    val = _actuals()
    # Four steps per series: the first one is history and must be dropped.
    pred = pd.DataFrame({"unique_id": np.repeat(["a", "b"], H + 1), "y": [9.0, 0.0, 1.0, 2.0, 9.0, 3.0, 4.0, 5.0]})
    rows = _model().series_metrics(pred, val, pd.Timestamp("2024-01-10"), long_format=True)
    assert [r.metric_value for r in rows] == [0.0, 0.0]
    assert all(r.forecast_quantiles is None for r in rows)
    np.testing.assert_array_equal(rows[1].forecast, [3.0, 4.0, 5.0])


def test_series_without_actuals_or_with_bad_forecasts_are_skipped(caplog):
    val = _actuals(ids=("a",))
    pred = pd.DataFrame({"unique_id": ["a", "missing", "a2"], "y": [np.zeros(H), np.zeros(H), np.zeros(H)]})
    val = pd.concat([val, _actuals(ids=("a2",)).iloc[:1]])  # a2 has 1 actual for 3 forecasts
    with caplog.at_level(logging.WARNING):
        rows = _model().series_metrics(pred, val, pd.Timestamp("2024-01-10"))
    assert [r.key for r in rows] == ["a"]
    assert sum("missing" in r.getMessage() for r in caplog.records) == 1
    assert sum("a2" in r.getMessage() for r in caplog.records) == 1


def test_series_metrics_rejects_unknown_metrics():
    pred = pd.DataFrame({"unique_id": ["a"], "y": [np.zeros(H)]})
    with pytest.raises(UnsupportedMetricError):
        _model(metric="wape").series_metrics(pred, _actuals(ids=("a",)), pd.Timestamp("2024-01-10"))


class _TupleModel(ForecastingRegressor):
    """A custom model whose calculate_metrics returns plain 7-tuples, as before EvaluationRow existed."""

    def calculate_metrics(self, hist_df, val_df, curr_date, spark=None):
        return [(k, curr_date, "mae", 0.0, np.zeros(H), np.zeros(H), b"x") for k in val_df["unique_id"].unique()]


def test_backtest_accepts_seven_tuples_and_names_the_columns():
    dates = pd.date_range("2024-01-01", periods=12, freq="D")
    df = pd.DataFrame({"unique_id": "a", "ds": dates, "y": np.arange(12, dtype=float)})
    res = _TupleModel(_model().params).backtest(df, start=dates[-H - 2])
    assert list(res.columns) == ["unique_id", "backtest_window_start_date", "metric_name", "metric_value", "forecast",
                                 "actual", "model_pickle", "forecast_quantiles"]
    assert len(res) == 2
    assert res["forecast_quantiles"].isna().all() and (res["model_pickle"] == b"x").all()
