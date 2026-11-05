import importlib.resources as pkg_resources

import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf
from statsforecast.models import AutoMFLES

import mmf_sa
from mmf_sa.exceptions import ConfigurationError
from mmf_sa.models import ModelRegistry

FREQ_CONFS = {
    "H": "forecasting_conf_hourly.yaml",
    "D": "forecasting_conf_daily.yaml",
    "W": "forecasting_conf_weekly.yaml",
    "M": "forecasting_conf_monthly.yaml",
}
STATSFORECAST_MODELS = [
    name for name, conf in ModelRegistry.load_models_conf()["models"].items()
    if conf.get("framework") == "StatsForecast"
]


def make_conf(freq="M", active_models=None, models=None, prediction_length=12):
    """The conf run_forecast builds: the frequency's base conf merged with the user's."""
    base = OmegaConf.create(pkg_resources.read_text(mmf_sa, FREQ_CONFS[freq]))
    user = {
        "freq": freq,
        "prediction_length": prediction_length,
        "group_id": "unique_id",
        "date_col": "ds",
        "target": "y",
        "metric": "smape",
        "active_models": active_models or STATSFORECAST_MODELS,
    }
    if models is not None:
        user["models"] = models
    conf = OmegaConf.merge(base, OmegaConf.create(user))
    conf["active_models"] = user["active_models"]
    return conf


def monthly_series(n=96, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    y = 100 + 0.5 * t + 10 * np.sin(2 * np.pi * t / 12) + rng.normal(0, 2, n)
    return pd.DataFrame({
        "unique_id": "s1",
        "ds": pd.date_range("2015-01-31", periods=n, freq="ME"),
        "y": y,
    })


@pytest.mark.parametrize("freq", list(FREQ_CONFS))
def test_shipped_configs_build_the_same_models_as_explicit_arguments(freq):
    conf = make_conf(freq)
    registry = ModelRegistry(conf)
    for name in STATSFORECAST_MODELS:
        model = registry.get_model(name).model_spec
        spec = registry.get_model_conf(name).get("model_spec")
        kwargs = {k: v for k, v in (OmegaConf.to_container(spec) if spec is not None else {}).items() if k != "level"}
        if type(model) is AutoMFLES:
            kwargs["test_size"] = conf.prediction_length
        assert vars(model) == vars(type(model)(**kwargs)), (freq, name)


@pytest.mark.parametrize("freq,season_length", [("H", 24), ("D", 7), ("W", 4), ("M", 12)])
def test_seasonal_models_use_the_frequency_season_length(freq, season_length):
    registry = ModelRegistry(make_conf(freq))
    for name in STATSFORECAST_MODELS:
        spec = registry.get_model_conf(name).get("model_spec")
        if spec is not None and "season_length" in spec:
            assert spec.season_length == season_length, (freq, name)


def test_auto_arima_keeps_approximation_none_from_frequency_conf():
    model = ModelRegistry(make_conf("M", ["StatsForecastAutoArima"])).get_model("StatsForecastAutoArima")
    assert model.model_spec.approximation is None
    assert model.model_spec.season_length == 12


def test_auto_arima_constrained_exhaustive_search():
    spec = {
        "season_length": 12, "max_p": 1, "max_q": 1, "max_P": 1, "max_Q": 1,
        "D": 1, "max_d": 1, "stepwise": False, "ic": "aic", "allowdrift": True, "approximation": False,
    }
    conf = make_conf("M", ["StatsForecastAutoArima"], {"StatsForecastAutoArima": {"model_spec": spec}})
    model = ModelRegistry(conf).get_model("StatsForecastAutoArima")
    for key, value in spec.items():
        assert getattr(model.model_spec, key) == value, key

    forecast_df, fitted = model.forecast(monthly_series())
    assert len(forecast_df) == 12
    assert np.isfinite(forecast_df["y"]).all()
    p, q, P, Q, s, d, D = fitted.fitted_[0, 0].model_["arma"]
    assert max(p, q, P, Q) <= 1 and D == 1 and d <= 1 and s == 12


def test_auto_ets_restricted_model():
    conf = make_conf("M", ["StatsForecastAutoETS"],
                     {"StatsForecastAutoETS": {"model_spec": {"model": "ZZA", "damped": True}}})
    model = ModelRegistry(conf).get_model("StatsForecastAutoETS")
    assert model.model_spec.model == "ZZA"
    assert model.model_spec.damped is True


def test_unknown_key_raises():
    conf = make_conf("M", ["StatsForecastAutoArima"],
                     {"StatsForecastAutoArima": {"model_spec": {"maxp": 1}}})
    with pytest.raises(ConfigurationError, match="maxp"):
        ModelRegistry(conf).get_model("StatsForecastAutoArima")


def test_unknown_key_left_empty_is_ignored():
    conf = make_conf("M", ["StatsForecastAutoETS"],
                     {"StatsForecastAutoETS": {"model_spec": {"level": None}}})
    ModelRegistry(conf).get_model("StatsForecastAutoETS")


def test_auto_mfles_test_size_defaults_to_prediction_length():
    conf = make_conf("M", ["StatsForecastAutoMfles"], prediction_length=6)
    assert ModelRegistry(conf).get_model("StatsForecastAutoMfles").model_spec.test_size == 6

    conf = make_conf("M", ["StatsForecastAutoMfles"],
                     {"StatsForecastAutoMfles": {"model_spec": {"test_size": 3, "n_windows": 3}}},
                     prediction_length=6)
    model = ModelRegistry(conf).get_model("StatsForecastAutoMfles")
    assert model.model_spec.test_size == 3
    assert model.model_spec.n_windows == 3


def test_user_registered_variant():
    variant = {
        "module": "mmf_sa.models.statsforecast.StatsFcForecastingPipeline",
        "model_class": "StatsFcAutoArima",
        "framework": "StatsForecast",
        "model_type": "local",
        "model_spec": {"season_length": 12, "max_p": 1, "max_q": 1, "d": 0, "D": 1, "stepwise": False},
    }
    conf = make_conf("M", ["StatsForecastAutoArima", "StatsForecastAutoArimaD0"],
                     {"StatsForecastAutoArimaD0": variant})
    registry = ModelRegistry(conf)
    default = registry.get_model("StatsForecastAutoArima").model_spec
    custom = registry.get_model("StatsForecastAutoArimaD0").model_spec
    assert (default.d, default.stepwise) == (None, True)
    assert (custom.d, custom.D, custom.stepwise) == (0, 1, False)
