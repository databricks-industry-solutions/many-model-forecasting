"""Unit tests for model logging and registration (register_models).

Tests cover:
- The UC name of a model, and no name when register_models is False
- Global evaluation: the final model is logged (and registered when enabled), the backtest model is always logged and
  never registered, and the backtest metrics are logged to the backtest model
- Foundation evaluation: the logged model URI is written in both modes
- Scoring: the final model logged by the evaluation is used in both modes, without the registry
- Scoring without evaluation: the latest registered version, or the latest final model logged in the experiment
- Backtest models are never used for scoring
- FoundationForecast register() returns the model info and passes the tags
"""
import contextlib
from types import SimpleNamespace

import mlflow
import pandas as pd
import pytest
from omegaconf import OmegaConf

from mmf_sa import Forecaster
from mmf_sa.exceptions import ModelRegistryError


def _forecaster(register_models=None):
    conf = {"freq": "D", "backtest_length": 2, "group_id": "uid", "date_col": "ds", "target": "y",
            "metric": "smape", "model_output": "cat.sch", "use_case_name": "uc", "active_models": []}
    if register_models is not None:
        conf["register_models"] = register_models
    return Forecaster(conf=OmegaConf.create(conf), data_conf={}, spark=None, experiment_id="e", run_id="r1")


def _info(model_id):
    return SimpleNamespace(model_id=model_id, model_uri=f"models:/{model_id}")


class _FakeModel:
    """A global or foundation model recording its fit and register calls."""

    def __init__(self, model_id, name="M"):
        self.model_id = model_id
        self.params = OmegaConf.create({"name": name})
        self.fitted_rows = None
        self.register_kwargs = None

    def fit(self, df):
        self.fitted_rows = len(df)

    def register(self, **kwargs):
        self.register_kwargs = kwargs
        return _info(self.model_id)

    def get_params(self):
        return {"params": "p"}


@pytest.fixture
def no_tracking(monkeypatch):
    """Stubs the MLflow calls of the evaluation and records the logged-model tags and metrics."""
    calls = {"logged_model_tags": [], "metrics": []}
    run = SimpleNamespace(info=SimpleNamespace(run_id="run"))
    monkeypatch.setattr(mlflow, "start_run", lambda **k: contextlib.nullcontext(run))
    monkeypatch.setattr(mlflow, "autolog", lambda **k: None)
    monkeypatch.setattr(mlflow, "set_tag", lambda *a, **k: None)
    monkeypatch.setattr(mlflow, "log_params", lambda *a, **k: None)
    monkeypatch.setattr(mlflow, "set_logged_model_tags", lambda model_id, tags: calls["logged_model_tags"].append(
        (model_id, tags)))
    monkeypatch.setattr(mlflow, "log_metric", lambda key, value, **k: calls["metrics"].append((key, k.get("model_id"))))
    return calls


def _history():
    return pd.DataFrame({"uid": ["a"] * 6 + ["b"] * 6,
                         "ds": list(pd.date_range("2024-01-01", periods=6)) * 2,
                         "y": [float(i) for i in range(12)]})


def _prepare(monkeypatch, f, models):
    monkeypatch.setattr(f, "prepare_data_for_global_model", lambda mode=None: (_history(), []))
    monkeypatch.setattr(f.model_registry, "get_model", lambda name, override_conf=None: models.pop(0))
    backtests = []
    monkeypatch.setattr(f, "backtest_global_model", lambda **k: backtests.append(k) or 0.1)
    return backtests


def test_registered_model_name():
    conf = OmegaConf.create({"name": "M"})
    assert _forecaster().registered_model_name(conf) == "cat.sch.M_uc"
    assert _forecaster(True).registered_model_name(conf) == "cat.sch.M_uc"
    assert _forecaster(False).registered_model_name(conf) is None


@pytest.mark.parametrize("register_models,registered_name", [(True, "cat.sch.M_uc"), (False, None)])
def test_global_evaluation_logs_final_and_backtest_models(monkeypatch, no_tracking, register_models, registered_name):
    f = _forecaster(register_models)
    final, backtest = _FakeModel("m-final"), _FakeModel("m-backtest")
    backtests = _prepare(monkeypatch, f, [final, backtest])
    f.evaluate_global_model(OmegaConf.create({"name": "M", "model_type": "global"}))

    assert final.fitted_rows == 12 and backtest.fitted_rows == 8
    assert final.register_kwargs["registered_model_name"] == registered_name
    assert final.register_kwargs.get("name", "model") == "model"
    assert final.register_kwargs["tags"] == {
        "mmf_model_name": "M", "mmf_use_case": "uc", "mmf_run_id": "r1", "mmf_model_role": "final"}
    assert backtest.register_kwargs["registered_model_name"] is None
    assert backtest.register_kwargs["name"] == "backtest_model"
    assert backtest.register_kwargs["tags"]["mmf_model_role"] == "backtest"
    assert no_tracking["logged_model_tags"] == [
        ("m-final", {"mmf_backtest_model_id": "m-backtest"}), ("m-backtest", {"mmf_final_model_id": "m-final"})]
    [bt] = backtests
    assert bt["model"] is backtest
    assert bt["model_uri"] == "models:/m-backtest"
    assert bt["model_id"] == "m-backtest"
    assert f.logged_model_uris == {"M": "models:/m-final"}


@pytest.mark.parametrize("register_models,registered_name", [(True, "cat.sch.M_uc"), (False, None)])
@pytest.mark.parametrize("framework", ["Chronos", "FoundationForecast"])
def test_foundation_evaluation_uses_the_logged_model_uri(
        monkeypatch, no_tracking, framework, register_models, registered_name):
    f = _forecaster(register_models)
    model = _FakeModel("m-fm")
    backtests = _prepare(monkeypatch, f, [model])
    f.evaluate_foundation_model(OmegaConf.create({"name": "M", "model_type": "foundation", "framework": framework}))

    assert model.register_kwargs["registered_model_name"] == registered_name
    assert model.register_kwargs["tags"]["mmf_model_role"] == "final"
    assert ("input_example" in model.register_kwargs) == (framework == "FoundationForecast")
    [bt] = backtests
    assert bt["model_uri"] == "models:/m-fm" and bt["model_id"] == "m-fm"
    assert no_tracking["metrics"] == [("smape", "m-fm")]
    assert f.logged_model_uris == {"M": "models:/m-fm"}


def _patch_load_model(monkeypatch, load_model):
    # mlflow.pyfunc is a lazy loader that replaces itself with the module on first use; patch the module.
    mlflow.pyfunc.load_model
    monkeypatch.setattr(mlflow.pyfunc, "load_model", load_model)


def _logged_models(monkeypatch, roles):
    """Stubs mlflow.get_logged_model with the given model_id -> role tags."""
    monkeypatch.setattr(mlflow, "get_logged_model", lambda model_id: SimpleNamespace(
        model_id=model_id, tags={} if roles[model_id] is None else {"mmf_model_role": roles[model_id]}))


def _no_registry(monkeypatch):
    def fail(*a, **k):
        raise AssertionError("the model registry must not be used")
    monkeypatch.setattr(Forecaster, "get_latest_model_info", staticmethod(fail))
    monkeypatch.setattr(mlflow, "search_logged_models", fail)


@pytest.mark.parametrize("register_models", [True, False])
def test_scoring_uses_the_final_model_of_the_evaluation(monkeypatch, register_models):
    f = _forecaster(register_models)
    f.logged_model_uris = {"G": "models:/m-final", "F": "models:/m-fm"}
    _no_registry(monkeypatch)
    _logged_models(monkeypatch, {"m-final": "final", "m-fm": "final"})
    loaded = []
    _patch_load_model(monkeypatch, lambda uri: loaded.append(uri) or "pyfunc")

    assert f.get_model_for_scoring(OmegaConf.create({"name": "G", "model_type": "global"})) == (
        "pyfunc", "models:/m-final")
    assert f.get_model_for_scoring(OmegaConf.create({"name": "F", "model_type": "foundation"})) == (
        None, "models:/m-fm")
    assert loaded == ["models:/m-final"]


def test_global_evaluation_then_scoring_never_uses_the_backtest_model(monkeypatch, no_tracking):
    f = _forecaster(False)
    _prepare(monkeypatch, f, [_FakeModel("m-final"), _FakeModel("m-backtest")])
    f.evaluate_global_model(OmegaConf.create({"name": "M", "model_type": "global"}))
    _no_registry(monkeypatch)
    _logged_models(monkeypatch, {"m-final": "final", "m-backtest": "backtest"})
    _patch_load_model(monkeypatch, lambda uri: uri)
    model, uri = f.get_model_for_scoring(OmegaConf.create({"name": "M", "model_type": "global"}))
    assert model == uri == "models:/m-final"


def test_backtest_models_are_not_used_for_scoring(monkeypatch):
    _logged_models(monkeypatch, {"m-backtest": "backtest"})
    with pytest.raises(ModelRegistryError, match="backtest model"):
        Forecaster.check_final_model("models:/m-backtest")


def test_registered_versions_and_untagged_models_are_accepted(monkeypatch):
    _logged_models(monkeypatch, {"m-old": None})
    Forecaster.check_final_model("models:/m-old")
    monkeypatch.setattr(mlflow, "get_logged_model", lambda model_id: pytest.fail("not a logged model URI"))
    Forecaster.check_final_model("models:/cat.sch.M_uc/3")
    Forecaster.check_final_model("models:/cat.sch.M_uc@champion")
    Forecaster.check_final_model("runs:/run/model")


@pytest.mark.parametrize("version,expected", [
    (SimpleNamespace(version="3", model_id="m-v3"), "models:/m-v3"),
    (SimpleNamespace(version="3", model_id=None), "models:/cat.sch.M_uc/3"),
])
def test_scoring_without_evaluation_uses_the_latest_registered_version(monkeypatch, version, expected):
    f = _forecaster(True)
    names = []
    monkeypatch.setattr(Forecaster, "get_latest_model_info", staticmethod(
        lambda client, name: names.append(name) or version))
    monkeypatch.setattr(mlflow, "search_logged_models", lambda **k: pytest.fail("the experiment must not be searched"))
    assert f.find_final_model_uri(OmegaConf.create({"name": "M"})) == expected
    assert names == ["cat.sch.M_uc"]


def test_scoring_without_evaluation_uses_the_latest_final_model_of_the_experiment(monkeypatch, caplog):
    f = _forecaster(False)
    searches = []
    monkeypatch.setattr(Forecaster, "get_latest_model_info", staticmethod(lambda *a: pytest.fail("no registry")))
    monkeypatch.setattr(mlflow, "search_logged_models", lambda **k: searches.append(k) or [
        SimpleNamespace(model_id="m-latest", tags={"mmf_run_id": "r0"})])
    with caplog.at_level("WARNING"):
        assert f.find_final_model_uri(OmegaConf.create({"name": "M"})) == "models:/m-latest"
    [search] = searches
    assert search["experiment_ids"] == ["e"]
    assert search["filter_string"] == (
        "tags.mmf_model_name = 'M' AND tags.mmf_use_case = 'uc' AND tags.mmf_model_role = 'final'")
    assert search["order_by"] == [{"field_name": "creation_time", "ascending": False}]
    assert search["max_results"] == 1
    assert "m-latest" in caplog.text and "r0" in caplog.text


def test_scoring_without_evaluation_fails_without_a_logged_final_model(monkeypatch):
    f = _forecaster(False)
    monkeypatch.setattr(mlflow, "search_logged_models", lambda **k: [])
    with pytest.raises(ModelRegistryError, match="No final model of M"):
        f.find_final_model_uri(OmegaConf.create({"name": "M"}))


def test_local_models_do_not_use_mlflow(monkeypatch):
    f = _forecaster(False)
    _no_registry(monkeypatch)
    monkeypatch.setattr(f.model_registry, "get_model", lambda name: f"local {name}")
    assert f.get_model_for_scoring(OmegaConf.create({"name": "L", "model_type": "local"})) == ("local L", None)


def test_foundationforecast_register_returns_the_model_info_and_passes_tags(monkeypatch):
    from tests.unit.test_foundationforecast_pipeline import make_model
    import mmf_sa.models.foundationforecast.FoundationForecastPipeline as ffp
    captured = {}
    monkeypatch.setattr(ffp.mlflow.pyfunc, "log_model", lambda **k: captured.update(k) or _info("m-ff"))
    info = make_model("FFChronos2").register(None, tags={"mmf_model_role": "final"})
    assert info.model_id == "m-ff"
    assert captured["registered_model_name"] is None
    assert captured["name"] == "model" and captured["tags"] == {"mmf_model_role": "final"}
