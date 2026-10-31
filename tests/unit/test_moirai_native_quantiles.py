"""Unit tests for native quantiles of the Moirai pipeline (prediction intervals).

Moirai is sample based: its native quantiles are empirical quantiles of the sample paths. The tests use fake
sample paths and fake Spark frames, so they run without uni2ts, a GPU or a Spark session. Covers:
- Quantiles of the sample paths (match np.quantile) and the point forecast (the sample median)
- The UDF packing and the driver-side unpacking (round trip)
- predict and calculate_metrics with and without native_quantiles
- The one-per-process warning for small num_samples
"""
import logging

import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

from mmf_sa import calibration as cal
from mmf_sa.models.moiraiforecast import MoiraiPipeline as mp

QUANTILES = [0.025, 0.1, 0.5, 0.9, 0.975]
LEVELS = ["0.025", "0.1", "0.5", "0.9", "0.975"]
H = 4


def _samples(num_samples=200, seed=0):
    # float32 as the sample paths come out of torch
    return np.random.default_rng(seed).gamma(2.0, 3.0, size=(num_samples, H)).astype(np.float32)


def test_sample_quantiles_match_np_quantile():
    samples = _samples()
    median, qs, levels = mp.sample_quantile_forecasts(samples, QUANTILES)
    assert levels == QUANTILES
    assert qs.shape == (len(QUANTILES), H)
    np.testing.assert_allclose(qs, np.quantile(samples.astype(np.float64), QUANTILES, axis=0), rtol=1e-12)
    np.testing.assert_allclose(qs[0, 1], np.quantile(samples[:, 1].astype(np.float64), 0.025), rtol=1e-12)
    assert (np.diff(qs, axis=0) >= 0).all()
    # The median is always added to the levels, and it is the point forecast.
    assert mp.sample_quantile_forecasts(samples, [0.1, 0.9])[2] == [0.1, 0.5, 0.9]


def test_point_forecast_is_the_sample_median():
    samples = _samples()
    median, _, _ = mp.sample_quantile_forecasts(samples, QUANTILES)
    np.testing.assert_array_equal(median, np.median(samples, axis=0).astype(np.float64))
    # Without native quantiles the UDF output is the median itself, as before.
    np.testing.assert_array_equal(mp.pack_sample_forecasts(samples, []), np.median(samples, axis=0))
    np.testing.assert_array_equal(mp.pack_sample_forecasts(samples, None), np.median(samples, axis=0))


def test_udf_pack_unpack_round_trip():
    series = [_samples(seed=s) for s in range(3)]
    frame = pd.DataFrame({"unique_id": ["a", "b", "c"], "y": [mp.pack_sample_forecasts(s, QUANTILES) for s in series]})
    out = cal.decode_quantile_forecasts(frame, "y", cal.forecast_levels(QUANTILES), H)
    np.testing.assert_allclose(
        np.stack(out["y"]), np.stack([np.median(s, axis=0).astype(np.float64) for s in series]), rtol=1e-12)
    for i, key in enumerate(LEVELS):
        expected = np.stack([np.quantile(s.astype(np.float64), QUANTILES[i], axis=0) for s in series])
        np.testing.assert_allclose(np.stack(out[cal.QUANTILE_COLUMN_PREFIX + key]), expected, rtol=1e-12)


class _Col:
    """Stands in for a Spark column. The fake UDF outputs carry their packed values in `values`."""

    def __init__(self, values=None):
        self.values = values

    def alias(self, name):
        return self


class _Selected:
    def __init__(self, uids, ds, y):
        self.frame = pd.DataFrame({"unique_id": uids, "ds": ds, "y": y})

    def toPandas(self):
        return self.frame


class _SparkFrame:
    """Stands in for the Spark frame that predict repartitions and selects from."""
    unique_id = _Col()
    ds = _Col()
    y = _Col()

    def __init__(self, uids):
        self.uids = uids

    def repartition(self, *args):
        return self

    def select(self, unique_id, ds, y):
        return _Selected(self.uids, ds.values, y.values)


def _moirai(native_quantiles=None, num_samples=200):
    params = {"freq": "D", "prediction_length": H, "patch_size": 32, "num_samples": num_samples,
              "group_id": "uid", "date_col": "date", "target": "y", "metric": "mae"}
    if native_quantiles:
        params["native_quantiles"] = native_quantiles
    return mp.MoiraiSmall(OmegaConf.create(params))


def _fake_spark(monkeypatch, model, samples_by_series):
    """Routes predict through fake Spark frames; the UDF output is packed by the real pack_sample_forecasts."""
    quantiles = [float(q) for q in model.params.get("native_quantiles") or []]
    uids = list(samples_by_series)
    packed = [mp.pack_sample_forecasts(samples_by_series[u], quantiles) for u in uids]
    monkeypatch.setattr(model, "prepare_data", lambda df, future=False, spark=None: _SparkFrame(uids))
    monkeypatch.setattr(model, "create_horizon_timestamps_udf",
                        lambda: (lambda col: _Col([np.arange(H)] * len(uids))))
    monkeypatch.setattr(model, "create_predict_udf", lambda device_count: (lambda col: _Col(packed)))
    return model


def _predict(monkeypatch, native_quantiles, samples):
    frame, _ = _fake_spark(monkeypatch, _moirai(native_quantiles), samples).predict(pd.DataFrame())
    return frame


def test_predict_keeps_point_forecast_and_adds_quantile_columns(monkeypatch):
    samples = {"a": _samples(seed=0), "b": _samples(seed=1)}
    plain = _predict(monkeypatch, None, samples)
    with_q = _predict(monkeypatch, QUANTILES, samples)

    assert not [c for c in plain.columns if str(c).startswith(cal.QUANTILE_COLUMN_PREFIX)]
    assert list(with_q.columns) == list(plain.columns) + [cal.QUANTILE_COLUMN_PREFIX + k for k in LEVELS]
    assert np.stack(with_q["y"]).shape == (2, H)
    np.testing.assert_array_equal(np.stack(with_q["y"]), np.stack(plain["y"]))
    for i, key in enumerate(LEVELS):
        expected = np.stack([np.quantile(samples[u].astype(np.float64), QUANTILES[i], axis=0) for u in samples])
        np.testing.assert_allclose(np.stack(with_q[cal.QUANTILE_COLUMN_PREFIX + key]), expected, rtol=1e-12)


def test_calculate_metrics_appends_quantile_map(monkeypatch):
    samples = {"a": _samples(seed=0), "b": _samples(seed=1)}
    val_df = pd.DataFrame({"uid": ["a"] * H + ["b"] * H, "y": np.arange(2 * H, dtype=float) + 1})

    metrics = _fake_spark(monkeypatch, _moirai(QUANTILES), samples).calculate_metrics(
        pd.DataFrame(), val_df, "2024-01-01")
    assert len(metrics) == 2 and all(len(m) == 8 for m in metrics)
    row = {m[0]: m for m in metrics}["b"]
    np.testing.assert_allclose(row[4], np.median(samples["b"], axis=0))  # the forecast is still the median
    assert list(row[7]) == LEVELS
    np.testing.assert_allclose(row[7]["0.9"], np.quantile(samples["b"].astype(np.float64), 0.9, axis=0), rtol=1e-12)

    plain = _fake_spark(monkeypatch, _moirai(), samples).calculate_metrics(pd.DataFrame(), val_df, "2024-01-01")
    assert all(len(m) == 8 and m[7] is None for m in plain)


@pytest.fixture
def fresh_warnings(monkeypatch):
    # pandas_udf needs a live SparkContext; the warning is raised before the UDF is built.
    monkeypatch.setattr(mp, "pandas_udf", lambda *args, **kwargs: (lambda fn: fn))


def _warnings(caplog):
    return [r for r in caplog.records if "sample paths" in r.getMessage()]


def test_few_samples_warning_logged_once(fresh_warnings, caplog):
    with caplog.at_level(logging.WARNING, logger=mp.__name__):
        for _ in range(3):
            _moirai(QUANTILES, num_samples=10).create_predict_udf(1)
    records = _warnings(caplog)
    assert len(records) == 1
    assert "num_samples" in records[0].getMessage() and "100" in records[0].getMessage()


@pytest.mark.parametrize("num_samples, quantiles", [(100, QUANTILES), (10, None), (10, [])])
def test_no_few_samples_warning_when_not_needed(fresh_warnings, caplog, num_samples, quantiles):
    with caplog.at_level(logging.WARNING, logger=mp.__name__):
        _moirai(quantiles, num_samples=num_samples).create_predict_udf(1)
    assert _warnings(caplog) == []
