"""Unit tests for the prediction_intervals configuration (mmf_sa.calibration).

Tests cover:
- Defaults, and the identical block shipped in every forecasting_conf_*.yaml
- Merging a partial user block, including nested coverage keys
- `quantiles` replacing `levels`
- Rejection of unknown keys and invalid values
- The native_intervals capability flags in models_conf.yaml
- Plumbing: run_forecast(prediction_intervals=...) and the Forecaster checks
"""
import importlib.resources as pkg_resources

import pytest
from omegaconf import OmegaConf

import mmf_sa
from mmf_sa.calibration import DEFAULT_PREDICTION_INTERVALS, resolve_prediction_intervals_conf
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
    pi = _resolve({"enabled": True, "levels": [90], "coverage": {"shrinkage_kappa": 20}})
    assert pi["enabled"] is True
    assert pi["levels"] == [90]
    assert pi["coverage"]["shrinkage_kappa"] == 20
    assert pi["coverage"]["target"] == "series"
    assert pi["coverage"]["group_col"] == "auto"
    assert pi["cqr_pool"] == "group"


def test_quantiles_replace_levels():
    pi = _resolve({"levels": [80, 95], "quantiles": [0.1, 0.5, 0.9]})
    assert pi["levels"] is None
    assert pi["quantiles"] == [0.1, 0.5, 0.9]


def test_accepts_dictconfig_and_valid_alternatives():
    pi = _resolve(OmegaConf.create({
        "method": "conformal",
        "coverage": {"target": "group", "group_col": ["store", "category"],
                     "level_group_col": "abc_class", "levels_by_group": {"A": [95], "C": [80]}},
        "clip_lower": 0,
        "recency_decay": 0.9,
        "output_format": "both",
    }))
    assert pi["coverage"]["group_col"] == ["store", "category"]
    assert pi["clip_lower"] == 0


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
    ({"method": "bootstrap"}, "method"),
    ({"cqr_score": "abs"}, "cqr_score"),
    ({"cqr_tail_split": "upper"}, "cqr_tail_split"),
    ({"cqr_pool": "store"}, "cqr_pool"),
    ({"cqr_score": "scaled", "cqr_pool": "series"}, "scaled"),
    ({"coverage": {"target": "store"}}, "coverage.target"),
    ({"coverage": {"target": "group", "group_col": None}}, "group_col"),
    ({"coverage": {"group_col": None}}, "group_col"),
    ({"coverage": {"group_col": 3}}, "group_col"),
    ({"coverage": {"min_scores_per_group": 0}}, "min_scores_per_group"),
    ({"coverage": {"shrinkage_kappa": 0}}, "shrinkage_kappa"),
    ({"coverage": {"pool_horizon": "yes"}}, "pool_horizon"),
    ({"coverage": {"levels_by_group": {"A": [95]}}}, "level_group_col"),
    ({"coverage": {"level_group_col": "abc", "levels_by_group": {"A": [150]}}}, "levels_by_group"),
    ({"recency_decay": 0}, "recency_decay"),
    ({"clip_lower": "zero"}, "clip_lower"),
    ({"output_format": "wide"}, "output_format"),
])
def test_rejects_invalid_values(conf, message):
    with pytest.raises(ConfigurationError, match=message):
        resolve_prediction_intervals_conf(conf)


def test_marginal_pools_do_not_need_group_col():
    pi = _resolve({"cqr_pool": "marginal", "coverage": {"target": "marginal", "group_col": None}})
    assert pi["coverage"]["group_col"] is None


def test_native_interval_flags():
    models = ModelRegistry.load_models_conf().models
    flagged = {name for name, m in models.items() if m.get("native_intervals")}
    assert flagged == {
        "StatsForecastBaselineNaive", "StatsForecastBaselineSeasonalNaive", "StatsForecastAutoArima",
        "StatsForecastAutoETS", "StatsForecastAutoCES", "StatsForecastAutoTheta", "StatsForecastAutoTbats",
        "ChronosBoltTiny", "ChronosBoltMini", "ChronosBoltSmall", "ChronosBoltBase",
        "Chronos2", "Chronos2Small", "Chronos2Synth",
    }


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
    conf = _run_forecast(monkeypatch, prediction_intervals={"enabled": True, "levels": [90],
                                                             "coverage": {"target": "group"}})
    pi = _resolve(conf.prediction_intervals)
    assert pi["enabled"] is True
    assert pi["levels"] == [90]
    assert pi["coverage"]["target"] == "group"
    assert pi["coverage"]["shrinkage_kappa"] == 10


def _forecaster_conf(**pi):
    return OmegaConf.create({"freq": "D", "backtest_length": 14, "active_models": [],
                             "prediction_intervals": pi})


def test_forecaster_requires_evaluation_output_when_enabled():
    with pytest.raises(ConfigurationError, match="evaluation_output"):
        mmf_sa.Forecaster(conf=_forecaster_conf(enabled=True), data_conf={}, spark=None, experiment_id="e")


@pytest.mark.parametrize("pi", [{"enabled": True, "recency_decay": 0.9}, {"enabled": True, "score": "abs"}])
def test_forecaster_rejects_unsupported_settings_before_the_backtest(pi):
    conf = OmegaConf.create({"freq": "D", "backtest_length": 14, "active_models": [], "evaluation_output": "t",
                             "prediction_intervals": pi})
    with pytest.raises(ConfigurationError, match="not supported"):
        mmf_sa.Forecaster(conf=conf, data_conf={}, spark=None, experiment_id="e")


def test_forecaster_resolves_block():
    f = mmf_sa.Forecaster(conf=_forecaster_conf(levels=[90]), data_conf={}, spark=None, experiment_id="e")
    assert f.conf.prediction_intervals.levels == [90]
    assert f.conf.prediction_intervals.enabled is False
    with pytest.raises(ConfigurationError, match="method"):
        mmf_sa.Forecaster(conf=_forecaster_conf(method="x"), data_conf={}, spark=None, experiment_id="e")
