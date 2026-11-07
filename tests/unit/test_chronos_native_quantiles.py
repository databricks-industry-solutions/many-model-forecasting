"""Native quantiles through the distributed Chronos predict UDFs (prediction intervals).

The UDF functions run directly (udf.func) with a fake chronos library, as on a Spark worker.
Tests cover:
- Chronos-Bolt: packing of medians and quantiles across batches, NaN for the levels outside the trained range with
  one warning naming the model, and the plain path (no native_quantiles) unchanged
- Chronos-Bolt and Chronos-2: the point forecast equals the plain path's when the quantile outputs cross, and levels
  between the trained quantiles are interpolated
- the round trip UDF output -> _predict_distributed, which decodes the quantile columns
- Chronos-2, with and without known-future covariates
The fake quantile value of series s at step h and level q is s + 100 q + h, so a wrong series, step or level shows.
"""
import logging
import sys
import types

import numpy as np
import pandas as pd
import pytest
import torch
from omegaconf import OmegaConf

from mmf_sa.intervals.quantiles import decode_quantile_forecasts, forecast_levels
from mmf_sa.models.chronosforecast.ChronosPipeline import Chronos2, ChronosBoltSmall

from .test_timesfm_native_quantiles import _FakeColumn, _FakeSpark

H = 4
QUANTILES = [0.025, 0.1, 0.5, 0.9, 0.975]
TRAINED = [round(0.1 * k, 1) for k in range(1, 10)]


def _expected(base, level):
    return base + 100 * level + np.arange(H)


def _bases(inputs):
    return torch.tensor([float(np.asarray(x["target"] if isinstance(x, dict) else x)[0]) for x in inputs])


class _FakeBolt:
    """Chronos-Bolt: predict_quantiles returns one (n, H, Q) tensor, predict (n, trained quantiles, H)."""
    quantiles = TRAINED

    def __init__(self):
        self.calls = []

    def _cube(self, inputs, levels, prediction_length):
        steps = torch.arange(prediction_length, dtype=torch.float32)
        return (_bases(inputs)[:, None, None] + 100 * torch.tensor(levels)[None, None, :] + steps[None, :, None])

    def predict_quantiles(self, inputs, prediction_length, quantile_levels):
        self.calls.append(("predict_quantiles", len(inputs), list(quantile_levels), inputs))
        q = self._cube(inputs, quantile_levels, prediction_length)
        return q, q.mean(-1)

    def predict(self, inputs, prediction_length):
        self.calls.append(("predict", len(inputs), None, inputs))
        return self._cube(inputs, self.quantiles, prediction_length).transpose(1, 2)


class _FakeChronos2(_FakeBolt):
    """Chronos-2: one (n_variates, H, Q) tensor per series from predict_quantiles, (n_variates, Q, H) from predict."""

    def predict_quantiles(self, inputs, prediction_length, quantile_levels):
        q, _ = super().predict_quantiles(inputs, prediction_length, quantile_levels)
        return [x[None] for x in q], None

    def predict(self, inputs, prediction_length):
        return [x[None] for x in super().predict(inputs, prediction_length)]


@pytest.fixture
def chronos(monkeypatch):
    """Installs a fake chronos library; returns the loaded pipelines and their from_pretrained arguments."""
    loaded = []

    def install(pipeline):
        def from_pretrained(repo, device_map, dtype):
            loaded.append({"repo": repo, "device_map": device_map, "dtype": dtype})
            return pipeline
        module = types.ModuleType("chronos")
        module.BaseChronosPipeline = types.SimpleNamespace(from_pretrained=from_pretrained)
        monkeypatch.setitem(sys.modules, "chronos", module)
        monkeypatch.setattr(torch.cuda, "set_device", lambda device: None)
        return loaded

    return install


def _model(cls, native_quantiles=None, **params):
    conf = dict(name=cls.__name__, group_id="uid", date_col="date", target="y", freq="D", prediction_length=H,
                batch_size=2, metric="mae", **params)
    if native_quantiles:
        conf["native_quantiles"] = native_quantiles
    return cls(OmegaConf.create(conf))


def _histories(n=3, length=10):
    """Series s starts at 10 * s, so its fake forecasts are 10 s + 100 q + h."""
    return pd.Series([np.full(length, 10.0 * s) for s in range(n)])


def test_bolt_udf_packs_medians_and_quantiles_across_batches(chronos, caplog):
    pipeline = _FakeBolt()
    loaded = chronos(pipeline)
    udf = _model(ChronosBoltSmall, QUANTILES).create_predict_udf(1)
    with caplog.at_level(logging.WARNING):
        (packed,) = list(udf.func(iter([_histories()])))

    assert loaded == [{"repo": "amazon/chronos-bolt-small", "device_map": "cuda:0", "dtype": torch.bfloat16}]
    assert [c[:3] for c in pipeline.calls] == [("predict_quantiles", 2, TRAINED), ("predict_quantiles", 1, TRAINED)]
    levels = forecast_levels(QUANTILES)
    frame = decode_quantile_forecasts(pd.DataFrame({"y": list(packed)}), "y", levels, H)
    for s in range(3):
        np.testing.assert_allclose(frame["y"].iloc[s], _expected(10 * s, 0.5))
        for q in levels:
            if q in (0.025, 0.975):  # outside Chronos-Bolt's trained range [0.1, 0.9]
                assert np.isnan(frame[f"__q_{q}"].iloc[s]).all()
            else:
                np.testing.assert_allclose(frame[f"__q_{q}"].iloc[s], _expected(10 * s, q), rtol=1e-6)
    messages = [r.getMessage() for r in caplog.records if "native range" in r.getMessage()]
    assert len(messages) == 1 and "ChronosBoltSmall" in messages[0] and "[0.1, 0.9]" in messages[0]


def test_bolt_udf_without_native_quantiles_returns_the_median_only(chronos):
    pipeline = _FakeBolt()
    chronos(pipeline)
    (out,) = list(_model(ChronosBoltSmall).create_predict_udf(1).func(iter([_histories()])))
    assert [c[0] for c in pipeline.calls] == ["predict", "predict"]
    assert all(len(v) == H for v in out)
    for s in range(3):
        np.testing.assert_allclose(out[s], _expected(10 * s, 0.5), rtol=1e-6)


class _FakeCrossingBolt(_FakeBolt):
    """Chronos-Bolt whose 0.5 output is above its 0.6 and 0.7 outputs: the median of its trained quantiles is the 0.6 one."""

    def _cube(self, inputs, levels, prediction_length):
        bump = torch.tensor([25.0 if round(level, 6) == 0.5 else 0.0 for level in levels])
        return super()._cube(inputs, levels, prediction_length) + bump[None, None, :]


def test_bolt_point_forecast_is_unchanged_when_quantiles_cross(chronos):
    pipeline = _FakeCrossingBolt()
    chronos(pipeline)
    (plain,) = list(_model(ChronosBoltSmall).create_predict_udf(1).func(iter([_histories()])))
    (packed,) = list(_model(ChronosBoltSmall, QUANTILES).create_predict_udf(1).func(iter([_histories()])))
    frame = decode_quantile_forecasts(pd.DataFrame({"y": list(packed)}), "y", forecast_levels(QUANTILES), H)
    for s in range(3):
        np.testing.assert_allclose(frame["y"].iloc[s], np.asarray(plain[s], dtype=np.float64))
        np.testing.assert_allclose(frame["y"].iloc[s], _expected(10 * s, 0.6), rtol=1e-6)
        np.testing.assert_array_equal(frame["__q_0.5"].iloc[s], frame["y"].iloc[s])
    assert all(c[2] == TRAINED for c in pipeline.calls if c[0] == "predict_quantiles")


class _FakeCrossingChronos2(_FakeChronos2):
    """Chronos-2 whose 0.5 output is above its 0.6 and 0.7 outputs."""

    def _cube(self, inputs, levels, prediction_length):
        bump = torch.tensor([25.0 if round(level, 6) == 0.5 else 0.0 for level in levels])
        return super()._cube(inputs, levels, prediction_length) + bump[None, None, :]


def test_chronos2_point_forecast_is_unchanged_when_quantiles_cross(chronos):
    pipeline = _FakeCrossingChronos2()
    chronos(pipeline)
    (plain,) = list(_model(Chronos2).create_predict_udf(1).func(iter([_histories()])))
    (packed,) = list(_model(Chronos2, QUANTILES).create_predict_udf(1).func(iter([_histories()])))
    frame = decode_quantile_forecasts(pd.DataFrame({"y": list(packed)}), "y", forecast_levels(QUANTILES), H)
    for s in range(3):
        np.testing.assert_allclose(frame["y"].iloc[s], np.asarray(plain[s], dtype=np.float64))
        np.testing.assert_allclose(frame["y"].iloc[s], _expected(10 * s, 0.5) + 25, rtol=1e-6)
        np.testing.assert_array_equal(frame["__q_0.5"].iloc[s], frame["y"].iloc[s])


def test_bolt_levels_between_trained_quantiles_are_interpolated(chronos):
    chronos(_FakeBolt())
    (packed,) = list(_model(ChronosBoltSmall, [0.25, 0.75]).create_predict_udf(1).func(iter([_histories()])))
    frame = decode_quantile_forecasts(pd.DataFrame({"y": list(packed)}), "y", forecast_levels([0.25, 0.75]), H)
    for s in range(3):
        for q in (0.25, 0.75):
            np.testing.assert_allclose(frame[f"__q_{q}"].iloc[s], _expected(10 * s, q), rtol=1e-6)


def test_bolt_udf_output_round_trips_through_predict_distributed(chronos, caplog):
    chronos(_FakeBolt())
    model = _model(ChronosBoltSmall, QUANTILES)
    (packed,) = list(model.create_predict_udf(1).func(iter([_histories(n=2)])))
    spark_output = pd.DataFrame({"unique_id": ["a", "b"], "ds": [np.zeros(H)] * 2, "y": list(packed)})
    model.prepare_data = lambda df, future=False, spark=None: _FakeSpark(spark_output)
    model.create_horizon_timestamps_udf = lambda: (lambda column: _FakeColumn())
    model.create_predict_udf = lambda device_count: (lambda column: _FakeColumn())

    caplog.clear()
    with caplog.at_level(logging.WARNING):
        forecast_df, _ = model._predict_distributed(pd.DataFrame(), spark=object())

    assert list(forecast_df["uid"]) == ["a", "b"]
    for s in range(2):
        np.testing.assert_allclose(forecast_df["y"].iloc[s], _expected(10 * s, 0.5))
        np.testing.assert_allclose(forecast_df["__q_0.1"].iloc[s], _expected(10 * s, 0.1), rtol=1e-6)
        np.testing.assert_allclose(forecast_df["__q_0.9"].iloc[s], _expected(10 * s, 0.9), rtol=1e-6)
        # Outside Chronos-Bolt's trained range: NaN, so the quantile maps leave them out.
        assert np.isnan(forecast_df["__q_0.025"].iloc[s]).all() and np.isnan(forecast_df["__q_0.975"].iloc[s]).all()
    # The UDF's warning is logged on the workers; the driver logs its own.
    messages = [r.getMessage() for r in caplog.records if "trained range" in r.getMessage()]
    assert messages == ["ChronosBoltSmall: quantiles ['0.025', '0.975'] are outside the trained range of the "
                        "checkpoint, so the interval levels that need them are NULL."]


@pytest.mark.parametrize("with_covariates", [False, True])
def test_chronos2_udf_packs_quantiles(chronos, with_covariates):
    pipeline = _FakeChronos2()
    loaded = chronos(pipeline)
    params = {"dynamic_future_numerical": ["promo"]} if with_covariates else {}
    model = _model(Chronos2, QUANTILES, **params)
    col_map = [("dyn_num", "promo")] if with_covariates else None
    histories = _histories()
    batch = (histories, pd.Series([np.arange(10 + H, dtype=float)] * 3)) if with_covariates else histories
    (packed,) = list(model.create_predict_udf(1, col_map).func(iter([batch])))

    assert loaded[0]["repo"] == model.repo
    inputs = pipeline.calls[0][3]
    if with_covariates:
        assert set(inputs[0]) == {"target", "past_covariates", "future_covariates"}
        np.testing.assert_array_equal(inputs[0]["past_covariates"]["promo"], np.arange(10))
        np.testing.assert_array_equal(inputs[0]["future_covariates"]["promo"], np.arange(10, 10 + H))
    levels = forecast_levels(QUANTILES)
    frame = decode_quantile_forecasts(pd.DataFrame({"y": list(packed)}), "y", levels, H)
    for s in range(3):
        np.testing.assert_allclose(frame["y"].iloc[s], _expected(10 * s, 0.5))
        np.testing.assert_allclose(frame["__q_0.9"].iloc[s], _expected(10 * s, 0.9), rtol=1e-6)
