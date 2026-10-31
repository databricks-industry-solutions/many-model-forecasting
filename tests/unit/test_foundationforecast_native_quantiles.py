"""Native quantiles of the FoundationForecast models (prediction intervals).

Tests cover:
- Quantile columns named after the MMF keys, carrying the library's values, for a supporting model
- The point forecast and the library call with and without native_quantiles
- calculate_metrics' 8th element: each series' quantile map, None without native quantiles
- Adapters that cannot return quantiles, and library output that lacks some levels
- Levels with more than 3 decimals, which foundationforecast rejects, are requested rounded outward
- Warnings: levels outside the native range, unsupported adapters, missing levels, few sample paths; each logged once
"""
import logging

import numpy as np
import pandas as pd
import pytest

from mmf_sa.intervals.quantiles import QUANTILE_COLUMN_PREFIX
from mmf_sa.models.foundationforecast import FoundationForecastPipeline as ffp

from .test_foundationforecast_pipeline import (  # noqa: F401 (fixtures and helpers shared with the FF tests)
    FakeFFModel,
    H,
    fake_foundationforecast,
    make_df,
    make_model,
    spark_session,
    split_at,
)

LEVELS = [0.025, 0.1, 0.5, 0.9, 0.975]
KEYS = ["0.025", "0.1", "0.5", "0.9", "0.975"]
DATES = pd.date_range("2024-01-01", periods=30, freq="D")
_library_forecast = FakeFFModel.forecast


@pytest.fixture
def quantile_library(monkeypatch):
    """Installs a fake forecast() that returns the library's quantile columns (``ff_forecast-q-*``).

    Each quantile sits (q - 0.5) * 10 above the point forecast, so expected values are easy to read.
    ``drop`` lists levels the library does not return.
    """
    def install(drop=()):
        def forecast(self, df, h, freq, quantiles=None, **kwargs):
            out = _library_forecast(self, df, h, freq, **kwargs)
            if quantiles is not None:
                self.calls[-1]["quantiles"] = quantiles
            for q in quantiles or []:
                if q not in drop:
                    out[ffp.ff_quantile_column(q)] = out[self.alias] + (q - 0.5) * 10
            return out

        monkeypatch.setattr(FakeFFModel, "forecast", forecast)

    return install


def make_native_model(name):
    """A model with native_quantiles set, as Forecaster.get_model_with_native_quantiles sets them."""
    model = make_model(name)
    model.params["native_quantiles"] = LEVELS
    return model


def quantile_columns(frame):
    return sorted(c for c in frame.columns if c.startswith(QUANTILE_COLUMN_PREFIX))


@pytest.mark.parametrize("q, column", [
    (0.025, "ff_forecast-q-2.5"),
    (0.1, "ff_forecast-q-10"),
    (0.5, "ff_forecast-q-50"),
    (0.975, "ff_forecast-q-97.5"),
    (0.151, "ff_forecast-q-15.1"),
])
def test_quantile_columns_follow_foundationforecast_naming(q, column):
    assert ffp.ff_quantile_column(q) == column


def test_native_quantile_columns_carry_the_library_values(quantile_library):
    quantile_library()
    pred, _ = make_native_model("FFChronos2").predict(make_df(DATES))
    assert quantile_columns(pred) == sorted(QUANTILE_COLUMN_PREFIX + k for k in KEYS)
    assert FakeFFModel.instances[-1].calls[-1]["quantiles"] == LEVELS
    for _, row in pred.iterrows():
        assert all(len(row[QUANTILE_COLUMN_PREFIX + k]) == H for k in KEYS)
        np.testing.assert_allclose(row[QUANTILE_COLUMN_PREFIX + "0.5"], row["y"])
        np.testing.assert_allclose(row[QUANTILE_COLUMN_PREFIX + "0.975"], row["y"] + 4.75)
        np.testing.assert_allclose(row[QUANTILE_COLUMN_PREFIX + "0.025"], row["y"] - 4.75)
        np.testing.assert_allclose(row[QUANTILE_COLUMN_PREFIX + "0.9"], row["y"] + 4.0)


def test_point_forecast_is_the_same_with_native_quantiles(quantile_library):
    quantile_library()
    df = make_df(DATES)
    plain, _ = make_model("FFChronos2").predict(df)
    native, _ = make_native_model("FFChronos2").predict(df)
    assert list(native["unique_id"]) == list(plain["unique_id"])
    np.testing.assert_array_equal(np.stack(native["y"]), np.stack(plain["y"]))


def test_without_native_quantiles_the_library_is_called_as_before(quantile_library):
    quantile_library()
    pred, _ = make_model("FFChronos2").predict(make_df(DATES))
    call = FakeFFModel.instances[-1].calls[-1]
    assert "quantiles" not in call
    assert list(pred.columns) == ["unique_id", "ds", "y"]


def test_scoring_forecast_returns_native_quantile_columns(quantile_library):
    quantile_library()
    pred, _ = make_native_model("FFTimesFM_2_5_200m").forecast(make_df(DATES))
    assert quantile_columns(pred) == sorted(QUANTILE_COLUMN_PREFIX + k for k in KEYS)


def test_calculate_metrics_appends_each_series_quantile_map(quantile_library):
    quantile_library()
    hist, val = split_at(make_df(DATES), DATES[-H - 1])
    rows = make_native_model("FFChronos2").calculate_metrics(hist, val, DATES[-H - 1])
    assert len(rows) == 3
    for row in rows:
        assert len(row) == 8
        assert list(row[7]) == KEYS
        np.testing.assert_allclose(row[7]["0.975"], row[4] + 4.75)
        np.testing.assert_allclose(row[7]["0.5"], row[4])


def test_calculate_metrics_quantile_map_is_none_without_native_quantiles(quantile_library):
    quantile_library()
    hist, val = split_at(make_df(DATES), DATES[-H - 1])
    rows = make_model("FFChronos2").calculate_metrics(hist, val, DATES[-H - 1])
    assert len(rows) == 3 and all(len(row) == 8 and row[7] is None for row in rows)


def test_adapter_that_cannot_return_quantiles_is_unaffected(quantile_library, monkeypatch, caplog):
    quantile_library()
    monkeypatch.setattr(ffp.FFChronos2, "native_quantiles_supported", False)
    df = make_df(DATES)
    with caplog.at_level(logging.WARNING, logger=ffp.__name__):
        plain, _ = make_model("FFChronos2").predict(df)
        model = make_native_model("FFChronos2")
        native, _ = model.predict(df)
        model.predict(df)
    call = FakeFFModel.instances[-1].calls[-1]
    assert "quantiles" not in call
    assert list(native.columns) == list(plain.columns) == ["unique_id", "ds", "y"]
    np.testing.assert_array_equal(np.stack(native["y"]), np.stack(plain["y"]))
    messages = [r.message for r in caplog.records if "cannot return quantiles" in r.message]
    assert len(messages) == 1 and "FFChronos2" in messages[0]


def test_levels_the_library_omits_are_left_out_with_one_warning(quantile_library, caplog):
    quantile_library(drop=(0.025, 0.975))
    df = make_df(DATES)
    with caplog.at_level(logging.WARNING, logger=ffp.__name__):
        model = make_native_model("FFPatchTSTFM_R2")
        pred, _ = model.predict(df)
        model.predict(df)
    assert quantile_columns(pred) == sorted(QUANTILE_COLUMN_PREFIX + k for k in ["0.1", "0.5", "0.9"])
    messages = [r.message for r in caplog.records if "did not return the quantiles" in r.message]
    assert len(messages) == 1 and "0.025" in messages[0] and "0.975" in messages[0]


def test_library_without_quantile_columns_keeps_the_point_forecast(caplog):
    df = make_df(DATES)
    plain, _ = make_model("FFPatchTSTFM_R2").predict(df)
    with caplog.at_level(logging.WARNING, logger=ffp.__name__):
        native, _ = make_native_model("FFPatchTSTFM_R2").predict(df)
    assert quantile_columns(native) == []
    np.testing.assert_array_equal(np.stack(native["y"]), np.stack(plain["y"]))
    assert len([r for r in caplog.records if "did not return the quantiles" in r.message]) == 1


def test_levels_outside_the_native_range_warn_once(quantile_library, caplog):
    quantile_library()
    df = make_df(DATES)
    with caplog.at_level(logging.WARNING, logger=ffp.__name__):
        make_model("FFTimesFM_2_5_200m").predict(df)
        model = make_native_model("FFTimesFM_2_5_200m")
        model.predict(df)
        model.predict(df)
    messages = [r.message for r in caplog.records if "native range" in r.message]
    assert len(messages) == 1
    assert "FFTimesFM_2_5_200m" in messages[0] and "[0.1, 0.9]" in messages[0] and "too narrow" in messages[0]


def test_levels_inside_the_native_range_do_not_warn(quantile_library, caplog):
    quantile_library()
    with caplog.at_level(logging.WARNING, logger=ffp.__name__):
        make_native_model("FFPatchTSTFM_R2").predict(make_df(DATES))
    assert not [r for r in caplog.records if "native range" in r.message]


@pytest.mark.parametrize("q, expected", [
    (0.0125, 0.012), (0.9875, 0.988), (0.025, 0.025), (0.1, 0.1), (0.5, 0.5), (0.975, 0.975), (0.0005, 0.001), (0.9999, 0.999),
])
def test_ff_level_rounds_outward_to_3_decimals(q, expected):
    assert ffp.ff_level(q) == expected


def test_levels_with_more_than_3_decimals_are_requested_rounded_and_warn_once(quantile_library, caplog):
    quantile_library()
    df = make_df(DATES)
    with caplog.at_level(logging.WARNING, logger=ffp.__name__):
        model = make_model("FFChronos2")
        model.params["native_quantiles"] = [0.0125, 0.1, 0.5, 0.9, 0.9875]
        pred, _ = model.predict(df)
        model.predict(df)
    assert FakeFFModel.instances[-1].calls[-1]["quantiles"] == [0.012, 0.1, 0.5, 0.9, 0.988]
    keys = ["0.0125", "0.1", "0.5", "0.9", "0.9875"]
    assert quantile_columns(pred) == sorted(QUANTILE_COLUMN_PREFIX + k for k in keys)
    for _, row in pred.iterrows():
        np.testing.assert_allclose(row[QUANTILE_COLUMN_PREFIX + "0.0125"], row["y"] + (0.012 - 0.5) * 10)
        np.testing.assert_allclose(row[QUANTILE_COLUMN_PREFIX + "0.9875"], row["y"] + (0.988 - 0.5) * 10)
    messages = [r.message for r in caplog.records if "at most 3 decimals" in r.message]
    assert len(messages) == 1 and "0.0125" in messages[0] and "0.988" in messages[0]


def test_few_sample_paths_warn_once(quantile_library, caplog):
    quantile_library()
    df = make_df(DATES)
    with caplog.at_level(logging.WARNING, logger=ffp.__name__):
        make_model("FFToto").predict(df)  # no native quantiles: no warning
        model = make_native_model("FFToto")
        model.params["num_samples"] = 32
        model.predict(df)
        model.predict(df)
    messages = [r.message for r in caplog.records if "sample paths" in r.message]
    assert len(messages) == 1
    assert "FFToto" in messages[0] and "32" in messages[0] and "num_samples" in messages[0] and "100" in messages[0]


def test_enough_sample_paths_or_no_sampling_do_not_warn(quantile_library, caplog):
    quantile_library()
    with caplog.at_level(logging.WARNING, logger=ffp.__name__):
        model = make_native_model("FFToto")
        model.params["num_samples"] = 256
        model.predict(make_df(DATES))
        make_native_model("FFToto2_22m").predict(make_df(DATES))
    assert not [r for r in caplog.records if "sample paths" in r.message]
