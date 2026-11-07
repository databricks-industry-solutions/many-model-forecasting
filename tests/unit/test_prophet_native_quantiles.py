"""Unit tests for native quantiles of SKTimeProphet (prediction intervals).

sktime's Prophet adapter is replaced by a fake that follows the adapter's contract: predict returns the
point forecast, and predict_interval reuses the fitted model for each coverage and returns a
(variable, coverage, lower|upper) MultiIndex frame. The real prophet package is not needed.

Tests cover:
- Quantile columns: keys, central coverages and values (lower/upper bounds of each central interval)
- The point forecast, unchanged with and without native quantiles
- One fit per predict, with the intervals read from the fitted model
- The backtest (calculate_metrics -> forecast_quantiles) and the local scoring path (forecast)
"""
import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

import mmf_sa.models.sktime.SKTimeForecastingPipeline as pipeline_module
from mmf_sa import Forecaster
from mmf_sa.intervals.quantiles import QUANTILE_COLUMN_PREFIX
from mmf_sa.models.abstract_model import ForecastingRegressor
from mmf_sa.models.sktime.SKTimeForecastingPipeline import SKTimeProphet

QUANTILES = [0.025, 0.1, 0.5, 0.9, 0.975]
KEYS = ["0.025", "0.1", "0.5", "0.9", "0.975"]
N_STEPS = 7
STEPS = np.arange(1, N_STEPS + 1)


def point(steps):
    """The fake point forecast: negative for the first steps, so that clipping is visible."""
    return steps - 3.0


class FakeProphet:
    """Stands in for sktime's Prophet adapter. The half-width of a coverage-c interval is 10 * c * step."""

    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.predict_calls = 0
        self.interval_calls = []
        FakeProphet.instances.append(self)

    def fit(self, x):
        self._last = x.index.max()
        return self

    def _index(self, steps):
        return pd.PeriodIndex([self._last + int(s) for s in steps], freq=self._last.freq)

    def predict(self, fh):
        self.predict_calls += 1
        steps = np.asarray(fh.to_pandas(), dtype=float)
        return pd.DataFrame({"y": point(steps)}, index=self._index(steps))

    def predict_interval(self, fh=None, X=None, coverage=0.90):
        assert isinstance(coverage, list), "sktime checks that coverage is a list of floats"
        self.interval_calls.append(list(coverage))
        steps = np.asarray(fh.to_pandas(), dtype=float)
        columns = pd.MultiIndex.from_product([["y"], coverage, ["lower", "upper"]])
        values = [arr for c in coverage for arr in (point(steps) - 10.0 * c * steps, point(steps) + 10.0 * c * steps)]
        return pd.DataFrame(np.column_stack(values), index=self._index(steps), columns=columns)


@pytest.fixture
def fake_prophet(monkeypatch):
    monkeypatch.setattr(FakeProphet, "instances", [])
    monkeypatch.setattr(pipeline_module, "Prophet", FakeProphet)
    return FakeProphet


def _params(quantiles=None):
    params = dict(group_id="uid", date_col="date", target="y", freq="D", prediction_length=N_STEPS,
                  metric="smape", stride=7, backtest_length=14,
                  model_spec={"growth": "linear", "yearly_seasonality": "auto", "weekly_seasonality": "auto",
                              "daily_seasonality": "auto", "seasonality_mode": "additive"})
    if quantiles:
        params["native_quantiles"] = quantiles
    return OmegaConf.create(params)


def _series(n=120, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"uid": "a", "date": pd.date_range("2023-01-01", periods=n),
                         "y": rng.poisson(3, n).astype(float)})


def test_quantile_columns_are_the_central_interval_bounds(fake_prophet):
    df = _series()
    forecast_df, _ = SKTimeProphet(_params(QUANTILES)).forecast(df)
    quantile_cols = [c for c in forecast_df.columns if str(c).startswith(QUANTILE_COLUMN_PREFIX)]
    assert quantile_cols == [QUANTILE_COLUMN_PREFIX + k for k in KEYS]
    assert len(forecast_df) == N_STEPS
    raw = point(STEPS)
    np.testing.assert_allclose(forecast_df[QUANTILE_COLUMN_PREFIX + "0.5"], raw)      # median = raw point forecast
    np.testing.assert_allclose(forecast_df[QUANTILE_COLUMN_PREFIX + "0.1"], raw - 8 * STEPS)    # 80% interval
    np.testing.assert_allclose(forecast_df[QUANTILE_COLUMN_PREFIX + "0.9"], raw + 8 * STEPS)
    np.testing.assert_allclose(forecast_df[QUANTILE_COLUMN_PREFIX + "0.025"], raw - 9.5 * STEPS)  # 95% interval
    np.testing.assert_allclose(forecast_df[QUANTILE_COLUMN_PREFIX + "0.975"], raw + 9.5 * STEPS)
    # The quantiles are the raw band: the negative median is not clipped, the point forecast is.
    assert (forecast_df[QUANTILE_COLUMN_PREFIX + "0.5"] < 0).any()
    np.testing.assert_array_equal(forecast_df["y"], np.maximum(raw, 0))


def test_intervals_are_read_from_the_fitted_model(fake_prophet):
    SKTimeProphet(_params(QUANTILES)).forecast(_series())
    assert len(fake_prophet.instances) == 1  # one fit per predict, no refit per coverage
    fitted = fake_prophet.instances[0]
    assert fitted.predict_calls == 1
    assert fitted.interval_calls == [[0.8, 0.95]]  # one call for both central intervals


def test_point_forecast_is_unchanged_with_and_without_quantiles(fake_prophet):
    df = _series()
    plain, _ = SKTimeProphet(_params()).forecast(df)
    with_q, _ = SKTimeProphet(_params(QUANTILES)).forecast(df)
    assert list(plain.columns) == ["date", "y"]
    pd.testing.assert_frame_equal(plain, with_q[["date", "y"]])
    np.testing.assert_array_equal(plain["y"], np.maximum(point(STEPS), 0))
    assert fake_prophet.instances[0].interval_calls == []  # without quantiles, no interval is computed
    assert fake_prophet.instances[1].interval_calls == [[0.8, 0.95]]


@pytest.mark.parametrize("quantiles, coverage", [
    ([0.05, 0.5, 0.95], [0.9]),   # one central interval gives both bounds
    ([0.5, 0.9], [0.8]),          # an upper-only request reads the upper bound of the 80% interval
    ([0.5], []),                  # the median alone needs no interval
])
def test_quantiles_read_only_the_intervals_they_need(fake_prophet, quantiles, coverage):
    forecast_df, _ = SKTimeProphet(_params(quantiles)).forecast(_series())
    assert fake_prophet.instances[0].interval_calls == ([coverage] if coverage else [])
    assert [c[len(QUANTILE_COLUMN_PREFIX):] for c in forecast_df.columns
            if str(c).startswith(QUANTILE_COLUMN_PREFIX)] == [f"{q:g}" for q in quantiles]
    for q in quantiles:
        if q == 0.5:
            expected = point(STEPS)
        else:
            width = 1 - 2 * min(q, 1 - q)  # coverage of the central interval whose bound is q
            expected = point(STEPS) + (-1 if q < 0.5 else 1) * 10.0 * width * STEPS
        np.testing.assert_allclose(forecast_df[QUANTILE_COLUMN_PREFIX + f"{q:g}"], expected)


def test_backtest_collects_native_quantiles(fake_prophet):
    df = _series()
    start = df.date.max() - pd.Timedelta(days=14)
    plain = SKTimeProphet(_params()).backtest(df, start=start, group_id="a")
    with_q = SKTimeProphet(_params(QUANTILES)).backtest(df, start=start, group_id="a")
    assert plain["forecast_quantiles"].isna().all()
    assert len(with_q) == 2
    np.testing.assert_allclose(plain["metric_value"], with_q["metric_value"])
    for a, b in zip(plain["forecast"], with_q["forecast"]):
        np.testing.assert_array_equal(a, b)
    for fq, point_forecast in zip(with_q["forecast_quantiles"], with_q["forecast"]):
        assert list(fq) == KEYS
        np.testing.assert_allclose(np.maximum(fq["0.5"], 0), point_forecast)
        np.testing.assert_allclose(fq["0.1"], np.asarray(fq["0.5"]) - 8 * STEPS)
        np.testing.assert_allclose(fq["0.975"], np.asarray(fq["0.5"]) + 9.5 * STEPS)
        assert len(fq["0.1"]) == N_STEPS


def test_calculate_metrics_forecast_quantiles(fake_prophet):
    df = _series()
    train, val = df.iloc[:-N_STEPS], df.iloc[-N_STEPS:]
    metrics = SKTimeProphet(_params(QUANTILES)).calculate_metrics(train, val, val.date.iloc[0])
    assert set(metrics["forecast_quantiles"]) == set(KEYS)
    np.testing.assert_allclose(metrics["forecast_quantiles"]["0.9"], np.asarray(point(STEPS)) + 8 * STEPS)


def test_forecast_returns_quantiles_for_scoring(fake_prophet):
    df = _series()
    model = SKTimeProphet(_params(QUANTILES))
    forecast_df, _ = model.forecast(df)
    assert set(ForecastingRegressor.forecast_quantiles(forecast_df)) == set(KEYS)
    scored = Forecaster.score_one_local_model(df.copy(), model, with_quantiles=True)
    fq = scored["forecast_quantiles"].iloc[0]
    assert fq is not None, "scoring failed; see the logged error"
    assert list(fq) == KEYS
    np.testing.assert_allclose(fq["0.025"], point(STEPS) - 9.5 * STEPS)
    np.testing.assert_allclose(fq["0.975"], point(STEPS) + 9.5 * STEPS)


def test_native_intervals_flag_passes_quantiles_to_prophet():
    conf = OmegaConf.create({"freq": "D", "backtest_length": 14, "evaluation_output": "t", "metric": "smape",
                             "date_col": "date", "target": "y", "group_id": "uid", "prediction_length": N_STEPS,
                             "stride": 7, "active_models": ["SKTimeProphet"],
                             "prediction_intervals": {"enabled": True}})
    forecaster = Forecaster(conf=conf, data_conf={}, spark=None, experiment_id="e")
    flagged = forecaster.get_model_with_native_quantiles(
        OmegaConf.create({"name": "SKTimeProphet", "native_intervals": True}))
    assert isinstance(flagged, SKTimeProphet)
    assert list(flagged.params["native_quantiles"]) == QUANTILES
    plain = forecaster.get_model_with_native_quantiles(OmegaConf.create({"name": "SKTimeProphet"}))
    assert plain.params.get("native_quantiles") is None
