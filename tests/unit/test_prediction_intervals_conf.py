"""Unit tests for the prediction_intervals configuration (mmf_sa.intervals.config).

Tests cover:
- Defaults, and the identical block shipped in every forecasting_conf_*.yaml
- Merging a partial user block
- `quantiles` replacing `levels`, and the interval levels of each
- Rejection of unknown keys (including the calibration settings, which are not part of this block) and invalid values
- The native_intervals capability flags in models_conf.yaml
- Plumbing: run_forecast(prediction_intervals=...) and the Forecaster
"""
import importlib.resources as pkg_resources

import pytest
from omegaconf import OmegaConf

import mmf_sa
from mmf_sa.intervals.config import DEFAULT_PREDICTION_INTERVALS, interval_levels, resolve_prediction_intervals_conf
from mmf_sa.exceptions import ConfigurationError
from mmf_sa.models import ModelRegistry


def _resolve(conf):
    return OmegaConf.to_container(resolve_prediction_intervals_conf(conf))


def test_defaults_when_block_is_missing():
    assert _resolve(None) == DEFAULT_PREDICTION_INTERVALS
    assert _resolve({}) == DEFAULT_PREDICTION_INTERVALS
    assert _resolve(None)["enabled"] is False


@pytest.mark.parametrize("freq", ["hourly", "daily", "weekly", "monthly"])
def test_shipped_yaml_block_matches_defaults(freq):
    conf = OmegaConf.create(pkg_resources.read_text(mmf_sa, f"forecasting_conf_{freq}.yaml"))
    assert _resolve(conf.prediction_intervals) == DEFAULT_PREDICTION_INTERVALS


def test_partial_override_keeps_other_defaults():
    pi = _resolve({"enabled": True, "levels": [90]})
    assert pi == {**DEFAULT_PREDICTION_INTERVALS, "enabled": True, "levels": [90]}


def test_quantiles_replace_levels():
    pi = _resolve({"levels": [80, 95], "quantiles": [0.1, 0.5, 0.9]})
    assert pi["levels"] is None
    assert pi["quantiles"] == [0.1, 0.5, 0.9]


def test_interval_levels():
    assert interval_levels(resolve_prediction_intervals_conf({"levels": [95, 80]})) == [80, 95]
    assert interval_levels(resolve_prediction_intervals_conf({"quantiles": [0.025, 0.5, 0.975, 0.1, 0.9]})) == [80, 95]


def test_accepts_dictconfig():
    pi = _resolve(OmegaConf.create({"enabled": True, "output_format": "both"}))
    assert pi["output_format"] == "both"


@pytest.mark.parametrize("conf, message", [
    ({"enable": True}, "Unknown key"),
    ({"coverage": {"kappa": 10}}, "Unknown key"),
    ({"enabled": "yes"}, "enabled"),
    ({"levels": []}, "levels"),
    ({"levels": [100]}, "levels"),
    ({"levels": [0]}, "levels"),
    ({"levels": None}, "levels"),
    ({"quantiles": [1.0]}, "quantiles"),
    ({"quantiles": [0.1, 0.5, 0.95]}, "central pairs"),
    # Calibration settings are not part of the block.
    ({"method": "conformal"}, "Unknown key"),
    ({"method_by_model": {"Chronos2": "native"}}, "Unknown key"),
    ({"coverage": {"target": "group"}}, "Unknown key"),
    ({"clip_lower": 0}, "Unknown key"),
    ({"output_format": "wide"}, "output_format"),
])
def test_rejects_invalid_values(conf, message):
    with pytest.raises(ConfigurationError, match=message):
        resolve_prediction_intervals_conf(conf)


def test_native_interval_flags():
    models = ModelRegistry.load_models_conf().models
    flagged = {name for name, m in models.items() if m.get("native_intervals")}
    assert flagged == {
        "StatsForecastBaselineNaive", "StatsForecastBaselineSeasonalNaive", "StatsForecastAutoArima",
        "StatsForecastAutoETS", "StatsForecastAutoCES", "StatsForecastAutoTheta", "StatsForecastAutoTbats",
        "ChronosBoltTiny", "ChronosBoltMini", "ChronosBoltSmall", "ChronosBoltBase",
        "Chronos2", "Chronos2Small", "Chronos2Synth",
        "SKTimeProphet", "TimesFM_2_5_200m",
        "MoiraiSmall", "MoiraiBase", "MoiraiLarge", "MoiraiMoESmall", "MoiraiMoEBase", "MoiraiMoELarge",
        "FFChronos2", "FFChronos2Small", "FFTimesFM_2_5_200m", "FFTimesFM_3_0", "FFTiRex2", "FFToto",
        "FFToto2_22m", "FFToto2_313m", "FFToto2_1B", "FFFlowState", "FFPatchTSTFM_R2", "FFPatchTSTFM_R1",
        "FFTafsutBase", "FFT0Beta", "FFT0Gamma", "FFTimestral_5M", "FFTimestral_28M", "FFTimestral_123M",
        "FFTimestral_209M",
    }


def test_native_interval_flags_match_the_model_classes():
    """Models flagged native_intervals declare native_quantiles_supported, and no other model does."""
    import importlib
    models = ModelRegistry.load_models_conf().models
    checked, mismatched = set(), []
    for name, m in models.items():
        try:
            cls = getattr(importlib.import_module(m.module), m.model_class)
        except ImportError:
            continue  # the model's library is not installed in this environment
        checked.add(name)
        if bool(m.get("native_intervals")) != cls.native_quantiles_supported:
            mismatched.append(name)
    assert mismatched == []
    flagged = {name for name, m in models.items() if m.get("native_intervals")}
    assert flagged <= checked  # every flagged model's module loads without its library


class _CapturingForecaster:
    captured = None

    def __init__(self, conf, data_conf, spark, run_id):
        type(self).captured = conf

    def evaluate_score(self, evaluate, score):
        return "run"


def _run_forecast(monkeypatch, **kwargs):
    monkeypatch.setattr(mmf_sa, "Forecaster", _CapturingForecaster)
    mmf_sa.run_forecast(
        spark=None, train_data="catalog.schema.train", group_id="unique_id", date_col="ds",
        target="y", freq="D", prediction_length=7, backtest_length=14, stride=7, **kwargs)
    return _CapturingForecaster.captured


def test_run_forecast_keeps_disabled_block_by_default(monkeypatch):
    conf = _run_forecast(monkeypatch)
    assert _resolve(conf.prediction_intervals) == DEFAULT_PREDICTION_INTERVALS


def test_run_forecast_argument_overrides_yaml(monkeypatch):
    conf = _run_forecast(monkeypatch, prediction_intervals={"enabled": True, "levels": [90]})
    pi = _resolve(conf.prediction_intervals)
    assert pi["enabled"] is True
    assert pi["levels"] == [90]
    assert pi["output_format"] == "quantile_map"


def _forecaster_conf(**pi):
    return OmegaConf.create({"freq": "D", "backtest_length": 14, "active_models": [],
                             "prediction_intervals": pi})


def test_forecaster_does_not_require_evaluation_output():
    f = mmf_sa.Forecaster(conf=_forecaster_conf(enabled=True), data_conf={}, spark=None, experiment_id="e")
    assert f.intervals_enabled


def test_forecaster_resolves_block():
    f = mmf_sa.Forecaster(conf=_forecaster_conf(levels=[90]), data_conf={}, spark=None, experiment_id="e")
    assert f.conf.prediction_intervals.levels == [90]
    assert f.conf.prediction_intervals.enabled is False
    with pytest.raises(ConfigurationError, match="output_format"):
        mmf_sa.Forecaster(conf=_forecaster_conf(output_format="x"), data_conf={}, spark=None, experiment_id="e")
