import logging
import pickle
import sys
import types

import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

from mmf_sa.exceptions import DataPreparationError, ModelInitializationError, ModelPredictionError
from mmf_sa.models import ModelRegistry
from mmf_sa.models.foundationforecast import _runtime
from mmf_sa.models.foundationforecast import FoundationForecastPipeline as ffp

EXPECTED_REPOS = {
    "FFChronos2": "amazon/chronos-2",
    "FFChronos2Small": "autogluon/chronos-2-small",
    "FFTimesFM_2_5_200m": "google/timesfm-2.5-200m-pytorch",
    "FFTimesFM_3_0": "google/timesfm-3.0-pytorch",
    "FFTiRex2": "NX-AI/TiRex-2",
    "FFToto": "Datadog/Toto-Open-Base-1.0",
    "FFToto2_22m": "Datadog/Toto-2.0-22m",
    "FFToto2_313m": "Datadog/Toto-2.0-313m",
    "FFToto2_1B": "Datadog/Toto-2.0-1B",
    "FFFlowState": "ibm-research/flowstate",
    "FFPatchTSTFM_R2": "ibm-granite/granite-timeseries-patchtst-fm-r2",
    "FFPatchTSTFM_R1": "ibm-research/patchtst-fm-r1",
    "FFTafsutBase": "Tafsut-FM/tafsut-univariate-base",
    "FFT0Beta": "theforecastingcompany/t0-beta",
}
NON_COMMERCIAL_MODELS = {"FFTimesFM_3_0", "FFPatchTSTFM_R1"}
MMF_ENTRY_KEYS = {"module", "model_class", "framework", "model_type", "license"}
H = 4


class FakeFFModel:
    """Mimics foundationforecast's forecast() output, including its quirks."""

    instances = []

    def __init__(self, ff_class, repo_id, alias, **kwargs):
        self.ff_class = ff_class
        self.repo_id = repo_id
        self.alias = alias
        self.kwargs = kwargs
        self.calls = []
        self.steps = None
        FakeFFModel.instances.append(self)

    def forecast(self, df, h, freq):
        self.calls.append({"columns": list(df.columns), "h": h, "freq": freq})
        rows = []
        # Series come back in reverse order to check that results are joined by key.
        for uid, grp in reversed(list(df.groupby("unique_id", sort=True))):
            last = grp["ds"].max()
            # pandas date_range snaps to the alias anchor (e.g. "W" -> Sundays), like foundationforecast.
            future = pd.date_range(start=last, periods=h + 1, freq=freq)
            future = future[future > last][:h]
            steps = self.steps or h
            for i in range(steps):
                rows.append({"unique_id": uid, "ds": future[i % h], self.alias: float(grp["y"].iloc[-1] + i + 1)})
        return pd.DataFrame(rows)


class FakeFFModelsModule(types.ModuleType):
    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return lambda **kwargs: FakeFFModel(ff_class=name, **kwargs)


class FakeWeightCache:
    clears = 0

    def clear(self):
        FakeWeightCache.clears += 1


@pytest.fixture(scope="module", autouse=True)
def spark_session():
    # Overrides the autouse Spark session from conftest; these tests never use Spark.
    yield None


@pytest.fixture(autouse=True)
def fake_foundationforecast(monkeypatch):
    FakeFFModel.instances = []
    models = FakeFFModelsModule("foundationforecast.models")
    package = types.ModuleType("foundationforecast")
    package.models = models
    monkeypatch.setitem(sys.modules, "foundationforecast", package)
    monkeypatch.setitem(sys.modules, "foundationforecast.models", models)
    FakeWeightCache.clears = 0
    cache_module = types.ModuleType("foundationforecast.core.model_weight_cache")
    cache_module.get_model_weight_cache = FakeWeightCache
    monkeypatch.setitem(sys.modules, "foundationforecast.core", types.ModuleType("foundationforecast.core"))
    monkeypatch.setitem(sys.modules, "foundationforecast.core.model_weight_cache", cache_module)
    fake_torch = types.ModuleType("torch")
    fake_torch.float32, fake_torch.bfloat16, fake_torch.float16 = "f32", "bf16", "f16"
    fake_torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(ffp, "_warned", set())
    monkeypatch.setattr(ffp, "_current_checkpoint", None)
    yield


def base_models_conf():
    return ModelRegistry.load_models_conf()["models"]


def ff_entries():
    return {
        name: entry for name, entry in base_models_conf().items()
        if entry.get("framework") == "FoundationForecast"
    }


def make_model(name, **overrides):
    conf = {
        "active_models": [name],
        "prediction_length": H,
        "group_id": "unique_id",
        "date_col": "ds",
        "target": "y",
        "metric": "smape",
        "freq": "D",
        "backtest_length": 8,
        "stride": H,
    }
    conf.update(overrides)
    return ModelRegistry(OmegaConf.create(conf)).get_model(name)


def make_df(dates, n_series=3, ids=None):
    ids = ids if ids is not None else [f"s{i}" for i in range(n_series)]
    return pd.concat(
        [pd.DataFrame({"unique_id": uid, "ds": dates, "y": np.arange(len(dates), dtype=float) + 10 * k})
         for k, uid in enumerate(ids)],
        ignore_index=True,
    )


def test_ff_model_list_matches_plan():
    assert set(ff_entries()) == set(EXPECTED_REPOS)


@pytest.mark.parametrize("name", sorted(EXPECTED_REPOS))
def test_registry_resolves_each_model_to_its_checkpoint(name):
    model = make_model(name)
    assert isinstance(model, ffp.FoundationForecastForecaster)
    assert model.repo == EXPECTED_REPOS[name]
    assert ff_entries()[name]["model_type"] == "foundation"


@pytest.mark.parametrize("name", sorted(EXPECTED_REPOS))
def test_every_yaml_key_is_read_by_the_model(name):
    entry = ff_entries()[name]
    model_keys = set(entry) - MMF_ENTRY_KEYS
    model_class = getattr(ffp, entry["model_class"])
    assert model_keys == set(model_class.hparams)


@pytest.mark.parametrize("name", sorted(EXPECTED_REPOS))
def test_yaml_hyperparameters_reach_the_constructor(name):
    model = make_model(name)
    model.predict(make_df(pd.date_range("2024-01-01", periods=30, freq="D")))
    ff_model = FakeFFModel.instances[-1]
    entry = ff_entries()[name]
    expected = {k: entry[k] for k in model.hparams if entry[k] is not None}
    if "dtype" in expected:
        expected["dtype"] = {"float32": "f32", "bfloat16": "bf16", "float16": "f16"}[expected["dtype"]]
    assert ff_model.ff_class == model.ff_class
    assert ff_model.repo_id == EXPECTED_REPOS[name]
    assert ff_model.kwargs == expected


def test_weight_cache_is_cleared_only_when_switching_checkpoints():
    df = make_df(pd.date_range("2024-01-01", periods=30, freq="D"))
    make_model("FFChronos2").predict(df)
    make_model("FFChronos2").predict(df)
    assert FakeWeightCache.clears == 0
    make_model("FFToto2_22m").predict(df)
    assert FakeWeightCache.clears == 1
    make_model("FFChronos2").predict(df)
    assert FakeWeightCache.clears == 2


def test_timesfm_3_does_not_receive_per_core_batch_size():
    assert "per_core_batch_size" in ffp.FFTimesFM_2_5_200m.hparams
    assert "per_core_batch_size" not in ffp.FFTimesFM_3_0.hparams


def test_every_model_declares_a_license_and_nc_models_are_flagged():
    entries = ff_entries()
    assert all(entry.get("license") for entry in entries.values())
    flagged = {n for n, e in entries.items() if e["license"] in ffp.NON_COMMERCIAL_LICENSES}
    assert flagged == NON_COMMERCIAL_MODELS


def test_predict_returns_one_row_per_series_with_original_ids():
    model = make_model("FFChronos2")
    df = make_df(pd.date_range("2024-01-01", periods=20, freq="D"), ids=[3, 1, 2])
    pred, _ = model.predict(df)
    assert sorted(pred["unique_id"]) == [1, 2, 3]
    assert pred["unique_id"].map(type).eq(int).all()
    for _, row in pred.iterrows():
        assert len(row["ds"]) == H and len(row["y"]) == H
        last_y = df.loc[df.unique_id == row["unique_id"], "y"].iloc[-1]
        np.testing.assert_allclose(row["y"], last_y + np.arange(1, H + 1))


def test_ff_receives_only_univariate_columns():
    model = make_model("FFChronos2")
    df = make_df(pd.date_range("2024-01-01", periods=20, freq="D"))
    df["promo"] = 1.0
    model.predict(df)
    assert FakeFFModel.instances[-1].calls[-1]["columns"] == ["unique_id", "ds", "y"]


@pytest.mark.parametrize(
    "freq, dates, ff_freq",
    [
        ("H", pd.date_range("2024-03-01", periods=48, freq="h"), "h"),
        ("D", pd.date_range("2024-01-01", periods=30, freq="D"), "D"),
        ("W", pd.date_range("2023-01-01", periods=30, freq="W-SUN"), "W"),
        ("W", pd.date_range("2023-01-02", periods=30, freq="W-MON"), "W"),
        ("M", pd.date_range("2020-01-31", periods=30, freq="ME"), "ME"),
    ],
    ids=["hourly", "daily", "weekly_sunday", "weekly_monday", "monthly"],
)
def test_horizon_dates_follow_mmf_offsets(freq, dates, ff_freq):
    model = make_model("FFFlowState", freq=freq)
    pred, _ = model.predict(make_df(dates))
    assert FakeFFModel.instances[-1].calls[-1]["freq"] == ff_freq
    expected = [dates[-1] + model.one_ts_offset * k for k in range(1, H + 1)]
    for got in pred["ds"]:
        assert got.dtype == np.dtype("datetime64[ns]")
        assert list(pd.to_datetime(got)) == expected


def test_monthly_data_must_be_month_end():
    model = make_model("FFChronos2", freq="M")
    with pytest.raises(DataPreparationError, match="month-end"):
        model.predict(make_df(pd.date_range("2020-01-01", periods=24, freq="MS")))


def test_missing_target_values_are_rejected():
    model = make_model("FFChronos2")
    df = make_df(pd.date_range("2024-01-01", periods=20, freq="D"))
    df.loc[5, "y"] = np.nan
    with pytest.raises(DataPreparationError, match="missing values"):
        model.predict(df)


def test_wrong_forecast_length_raises():
    model = make_model("FFChronos2")
    model._get_ff_model().steps = H - 1
    with pytest.raises(ModelPredictionError):
        model.predict(make_df(pd.date_range("2024-01-01", periods=20, freq="D")))


def test_forecast_uses_only_rows_with_observed_target():
    model = make_model("FFChronos2")
    dates = pd.date_range("2024-01-01", periods=24, freq="D")
    df = make_df(dates)
    df.loc[df["ds"] > dates[19], "y"] = np.nan
    pred, _ = model.forecast(df)
    for got in pred["ds"]:
        assert pd.Timestamp(got[0]) == dates[20]


def test_predict_uses_driver_path_even_with_spark():
    model = make_model("FFChronos2")
    pred, _ = model.predict(make_df(pd.date_range("2024-01-01", periods=20, freq="D")), spark=object())
    assert len(pred) == 3
    with pytest.raises(NotImplementedError):
        model._predict_distributed(pd.DataFrame(), spark=object())


def test_covariates_warn_once_and_are_ignored(caplog):
    with caplog.at_level(logging.WARNING, logger=ffp.__name__):
        make_model("FFChronos2", dynamic_future_numerical=["promo"])
        make_model("FFChronos2", dynamic_future_numerical=["promo"])
    messages = [r.message for r in caplog.records if "univariate" in r.message]
    assert len(messages) == 1 and "dynamic_future_numerical" in messages[0]


def test_non_commercial_license_warns_once(caplog):
    with caplog.at_level(logging.WARNING, logger=ffp.__name__):
        make_model("FFTimesFM_3_0")
        make_model("FFTimesFM_3_0")
        make_model("FFChronos2")
    messages = [r.message for r in caplog.records if "non-commercial" in r.message]
    assert len(messages) == 1 and "FFTimesFM_3_0" in messages[0]


def test_backtest_reports_metrics_per_series_and_window():
    model = make_model("FFChronos2")
    df = make_df(pd.date_range("2024-01-01", periods=40, freq="D"))
    res = model.backtest(df, start=pd.Timestamp("2024-01-31"))
    assert list(res.columns) == [
        "unique_id", "backtest_window_start_date", "metric_name", "metric_value",
        "forecast", "actual", "model_pickle",
    ]
    assert len(res) == 3 * 2
    assert res["metric_name"].eq("smape").all()
    assert res["forecast"].map(len).eq(H).all() and res["actual"].map(len).eq(H).all()


def test_mlflow_model_returns_long_frame_and_pickles_without_weights():
    mlflow_model = ffp.FoundationForecastMLflowModel(
        ff_class="Chronos", repo="amazon/chronos-2", ff_kwargs={"batch_size": 16},
        prediction_length=H, freq="W",
    )
    dates = pd.date_range("2023-01-02", periods=20, freq="W-MON")
    out = mlflow_model.predict(None, make_df(dates, n_series=2))
    assert list(out.columns) == ["unique_id", "ds", "y"]
    assert len(out) == 2 * H
    assert pd.Timestamp(out["ds"].iloc[0]) == dates[-1] + pd.DateOffset(weeks=1)
    assert mlflow_model._model is not None
    assert pickle.loads(pickle.dumps(mlflow_model))._model is None


@pytest.fixture
def fake_tirex2(monkeypatch):
    module = types.ModuleType(_runtime._TIREX2_SLSTM_MODULE)
    module._flashrnn_backend = lambda device: {"cpu": "vanilla", "cuda": "cuda"}[device]
    monkeypatch.setitem(sys.modules, _runtime._TIREX2_SLSTM_MODULE, module)
    return module


def test_tirex2_patch_uses_vanilla_on_cuda_and_is_idempotent(fake_tirex2):
    _runtime.patch_tirex2_slstm_backend()
    patched = fake_tirex2._flashrnn_backend
    _runtime.patch_tirex2_slstm_backend()
    assert fake_tirex2._flashrnn_backend is patched
    assert patched("cuda") == "vanilla"
    assert patched("cpu") == "vanilla"


def test_tirex2_patch_fails_loudly_when_target_is_missing(fake_tirex2):
    del fake_tirex2._flashrnn_backend
    with pytest.raises(ModelInitializationError, match="timecopilot-tirex2"):
        _runtime.patch_tirex2_slstm_backend()


def test_ensure_runtime_runs_once(monkeypatch):
    calls = []
    monkeypatch.setattr(_runtime, "_runtime_ready", False)
    monkeypatch.setattr(_runtime, "patch_tirex2_slstm_backend", lambda: calls.append(1))
    _runtime.ensure_runtime()
    _runtime.ensure_runtime()
    assert calls == [1]
