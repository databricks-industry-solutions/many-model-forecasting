import logging
import pathlib
import pickle
import sys
import types

import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

from mmf_sa.exceptions import (
    DataPreparationError,
    MissingFeatureError,
    ModelInitializationError,
    ModelPredictionError,
)
from mmf_sa.models import ModelRegistry
from mmf_sa.models.foundationforecast import _runtime
from mmf_sa.models.foundationforecast import FoundationForecastPipeline as ffp

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]

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
    "FFT0Gamma": "theforecastingcompany/t0-gamma",
    "FFTimestral_5M": "mistralai/Timestral-5M",
    "FFTimestral_28M": "mistralai/Timestral-28M",
    "FFTimestral_123M": "mistralai/Timestral-123M",
    "FFTimestral_209M": "mistralai/Timestral-209M",
}
NON_COMMERCIAL_MODELS = {"FFTimesFM_3_0", "FFPatchTSTFM_R1"}
REVENUE_CAPPED_MODELS = {"FFTimestral_5M", "FFTimestral_28M", "FFTimestral_123M", "FFTimestral_209M"}
MMF_ENTRY_KEYS = {"module", "model_class", "framework", "model_type", "license", "native_intervals"}
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

    def forecast(self, df, h, freq, **kwargs):
        self.calls.append({"columns": list(df.columns), "h": h, "freq": freq, "df": df, **kwargs})
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
    # The real setup imports tirex2, which would bind the fake torch for later test modules.
    monkeypatch.setattr(_runtime, "_runtime_ready", True)
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


def make_cov_df(dates, n_series=3, ids=None):
    """make_df plus a numeric ``promo`` and an integer ``dow`` covariate."""
    df = make_df(dates, n_series=n_series, ids=ids)
    step = df.groupby("unique_id").cumcount()
    df["promo"] = (step % 3 == 0).astype(float) + df.groupby("unique_id").ngroup() * 10
    df["dow"] = df["ds"].dt.dayofweek
    return df


def split_at(df, cutoff):
    return df[df.ds <= cutoff].copy(), df[df.ds > cutoff].copy()


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
    capped = {n for n, e in entries.items() if e["license"] in ffp.REVENUE_CAPPED_LICENSES}
    assert capped == REVENUE_CAPPED_MODELS


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


def test_ff_receives_only_univariate_columns_without_covariate_config():
    model = make_model("FFChronos2")
    df = make_df(pd.date_range("2024-01-01", periods=20, freq="D"))
    df["promo"] = 1.0
    model.predict(df)
    call = FakeFFModel.instances[-1].calls[-1]
    assert call["columns"] == ["unique_id", "ds", "y"]
    assert "X_df" not in call and "futr_exog_list" not in call


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


COVARIATE_MODELS = {
    "FFChronos2", "FFChronos2Small", "FFTimesFM_3_0", "FFTiRex2", "FFT0Beta", "FFT0Gamma",
    "FFTimestral_5M", "FFTimestral_28M", "FFTimestral_123M", "FFTimestral_209M",
}
DAILY = pd.date_range("2024-01-01", periods=24, freq="D")
CUTOFF = DAILY[19]


def test_covariate_support_matches_foundationforecast_docs():
    supported = {name for name in EXPECTED_REPOS if getattr(ffp, name).supports_future_covariates}
    assert supported == COVARIATE_MODELS
    string_ok = {name for name in EXPECTED_REPOS if getattr(ffp, name).accepts_categorical_strings}
    assert string_ok == {"FFChronos2", "FFChronos2Small"}


@pytest.mark.parametrize("name", sorted(COVARIATE_MODELS))
def test_supported_model_passes_history_and_horizon_covariates(name):
    model = make_model(name, dynamic_future_numerical=["promo"], dynamic_future_categorical=["dow"])
    df = make_cov_df(DAILY)
    hist, future = split_at(df, CUTOFF)
    model.predict(hist, future)
    call = FakeFFModel.instances[-1].calls[-1]
    assert call["columns"] == ["unique_id", "ds", "y", "promo", "dow"]
    assert call["futr_exog_list"] == ["promo", "dow"]
    X_df = call["X_df"]
    assert list(X_df.columns) == ["unique_id", "ds", "promo", "dow"]
    assert X_df.groupby("unique_id").size().eq(H).all()
    expected = future.astype({"unique_id": str}).sort_values(["unique_id", "ds"])
    pd.testing.assert_frame_equal(
        X_df.reset_index(drop=True),
        expected[["unique_id", "ds", "promo", "dow"]].astype({"dow": float}).reset_index(drop=True),
    )
    assert X_df["dow"].dtype == np.float64 and call["df"]["dow"].dtype == np.float64


def test_duplicate_covariate_names_are_passed_once():
    model = make_model("FFChronos2", dynamic_future_numerical=["promo"], dynamic_future_categorical=["promo"])
    df = make_cov_df(DAILY)
    model.predict(*split_at(df, CUTOFF))
    assert FakeFFModel.instances[-1].calls[-1]["futr_exog_list"] == ["promo"]


def test_forecast_takes_each_series_next_steps_from_future_rows():
    model = make_model("FFT0Beta", dynamic_future_numerical=["promo"])
    df = make_cov_df(DAILY)
    # s1 ends two days earlier; the rows after the horizon must be ignored.
    df.loc[(df.unique_id == "s1") & (df.ds > DAILY[17]), "y"] = np.nan
    df.loc[(df.unique_id != "s1") & (df.ds > CUTOFF), "y"] = np.nan
    df = pd.concat([df, make_cov_df(pd.date_range(DAILY[-1] + pd.Timedelta(days=1), periods=5))
                    .assign(y=np.nan)], ignore_index=True)
    pred, _ = model.forecast(df)
    X_df = FakeFFModel.instances[-1].calls[-1]["X_df"]
    starts = X_df.groupby("unique_id")["ds"].min()
    assert starts["s1"] == DAILY[18] and starts["s0"] == DAILY[20]
    assert X_df.groupby("unique_id").size().eq(H).all()
    s1 = df[(df.unique_id == "s1") & df.ds.between(DAILY[18], DAILY[21])]
    np.testing.assert_allclose(X_df.loc[X_df.unique_id == "s1", "promo"], s1["promo"])
    assert pd.Timestamp(pred.set_index("unique_id").loc["s1", "ds"][0]) == DAILY[18]


def test_series_without_all_future_rows_is_rejected():
    model = make_model("FFChronos2", dynamic_future_numerical=["promo"])
    hist, future = split_at(make_cov_df(DAILY), CUTOFF)
    future = future.drop(future[(future.unique_id == "s2") & (future.ds == DAILY[22])].index)
    with pytest.raises(DataPreparationError, match="1 series are missing future rows.*'s2'"):
        model.predict(hist, future)


def test_duplicate_future_rows_are_rejected():
    model = make_model("FFChronos2", dynamic_future_numerical=["promo"])
    hist, future = split_at(make_cov_df(DAILY), CUTOFF)
    with pytest.raises(DataPreparationError, match="duplicate dates"):
        model.predict(hist, pd.concat([future, future.head(1)]))


def test_missing_covariate_column_raises():
    model = make_model("FFChronos2", dynamic_future_numerical=["price"])
    with pytest.raises(MissingFeatureError, match="price"):
        model.predict(*split_at(make_cov_df(DAILY), CUTOFF))


@pytest.mark.parametrize("where", ["history", "horizon"])
def test_null_covariates_are_rejected(where):
    model = make_model("FFChronos2", dynamic_future_numerical=["promo"])
    hist, future = split_at(make_cov_df(DAILY), CUTOFF)
    target = hist if where == "history" else future
    target.loc[target.index[1], "promo"] = np.nan
    with pytest.raises(DataPreparationError, match="Covariate 'promo' contains missing values"):
        model.predict(hist, future)


def test_null_covariates_after_the_horizon_are_ignored():
    model = make_model("FFChronos2", dynamic_future_numerical=["promo"])
    df = make_cov_df(pd.date_range("2024-01-01", periods=26, freq="D"))
    hist, future = split_at(df, CUTOFF)
    future.loc[future.ds == future.ds.max(), "promo"] = np.nan
    model.predict(hist, future)
    assert FakeFFModel.instances[-1].calls[-1]["X_df"]["promo"].notna().all()


def test_string_categoricals_reach_chronos_2_unchanged():
    model = make_model("FFChronos2", dynamic_future_categorical=["event"])
    df = make_cov_df(DAILY).assign(event=lambda d: np.where(d.promo % 10 == 1, "sale", "none"))
    model.predict(*split_at(df, CUTOFF))
    call = FakeFFModel.instances[-1].calls[-1]
    assert set(call["X_df"]["event"]) <= {"sale", "none"}
    assert set(call["df"]["event"]) == {"sale", "none"}


@pytest.mark.parametrize("name", sorted(COVARIATE_MODELS - {"FFChronos2", "FFChronos2Small"}))
def test_string_categoricals_must_be_encoded_for_other_models(name):
    model = make_model(name, dynamic_future_categorical=["event"])
    df = make_cov_df(DAILY).assign(event="none")
    with pytest.raises(DataPreparationError, match="Label-encode"):
        model.predict(*split_at(df, CUTOFF))


def test_boolean_covariates_are_cast_to_float():
    model = make_model("FFT0Beta", dynamic_future_categorical=["holiday"])
    df = make_cov_df(DAILY).assign(holiday=lambda d: d.dow == 6)
    model.predict(*split_at(df, CUTOFF))
    call = FakeFFModel.instances[-1].calls[-1]
    assert call["df"]["holiday"].dtype == np.float64 and call["X_df"]["holiday"].dtype == np.float64


@pytest.mark.parametrize("covariates, expected", [([], "W"), (["promo"], "W-MON")])
def test_weekly_covariates_use_the_data_weekday(covariates, expected):
    model = make_model("FFChronos2", freq="W", dynamic_future_numerical=covariates)
    dates = pd.date_range("2023-01-02", periods=30, freq="W-MON")
    model.predict(*split_at(make_cov_df(dates), dates[25]))
    assert FakeFFModel.instances[-1].calls[-1]["freq"] == expected


def test_weekly_covariates_require_one_weekday():
    model = make_model("FFChronos2", freq="W", dynamic_future_numerical=["promo"])
    dates = pd.date_range("2023-01-02", periods=30, freq="W-MON")
    df = make_cov_df(dates)
    df.loc[df.unique_id == "s1", "ds"] += pd.Timedelta(days=1)
    with pytest.raises(DataPreparationError, match="same weekday"):
        model.predict(*split_at(df, dates[25] + pd.Timedelta(days=1)))


def test_monthly_horizon_rows_are_matched_by_month_end():
    model = make_model("FFTiRex2", freq="M", dynamic_future_numerical=["promo"])
    dates = pd.date_range("2020-01-31", periods=30, freq="ME")
    model.predict(*split_at(make_cov_df(dates), dates[25]))
    X_df = FakeFFModel.instances[-1].calls[-1]["X_df"]
    assert list(X_df.loc[X_df.unique_id == "s0", "ds"]) == list(dates[26:30])


def test_integer_ids_are_strings_in_covariate_frames():
    model = make_model("FFChronos2", dynamic_future_numerical=["promo"])
    pred, _ = model.predict(*split_at(make_cov_df(DAILY, ids=[3, 1, 2]), CUTOFF))
    call = FakeFFModel.instances[-1].calls[-1]
    assert set(call["X_df"]["unique_id"]) == {"1", "2", "3"} == set(call["df"]["unique_id"])
    assert pred["unique_id"].map(type).eq(int).all()


def test_backtest_passes_each_window_covariates():
    model = make_model("FFTimestral_5M", dynamic_future_numerical=["promo"])
    df = make_cov_df(pd.date_range("2024-01-01", periods=40, freq="D"))
    res = model.backtest(df, start=pd.Timestamp("2024-01-31"))
    calls = FakeFFModel.instances[-1].calls
    assert len(calls) == 2 and len(res) == 3 * 2
    for call in calls:
        X_df = call["X_df"]
        window = df.astype({"unique_id": str}).merge(X_df[["unique_id", "ds"]])
        np.testing.assert_allclose(X_df["promo"], window["promo"])
        assert X_df["ds"].min() > call["df"]["ds"].max()


def test_missing_future_rows_fall_back_to_univariate_with_warning(caplog):
    model = make_model("FFChronos2", dynamic_future_numerical=["promo"])
    df = make_cov_df(DAILY)
    with caplog.at_level(logging.WARNING, logger=ffp.__name__):
        model.forecast(df)
        model.forecast(df)
    call = FakeFFModel.instances[-1].calls[-1]
    assert call["columns"] == ["unique_id", "ds", "y"] and "X_df" not in call
    assert len([r for r in caplog.records if "forecasting without covariates" in r.message]) == 1


def test_unsupported_model_warns_once_and_ignores_covariates(caplog):
    dates = pd.date_range("2024-01-01", periods=24, freq="D")
    df = make_cov_df(dates)
    with caplog.at_level(logging.WARNING, logger=ffp.__name__):
        make_model("FFToto2_22m", dynamic_future_numerical=["promo"])
        model = make_model("FFToto2_22m", dynamic_future_numerical=["promo"])
        model.predict(df[df.ds <= dates[19]], df[df.ds > dates[19]])
    messages = [r.message for r in caplog.records if "does not support covariates" in r.message]
    assert len(messages) == 1 and "dynamic_future_numerical" in messages[0]
    call = FakeFFModel.instances[-1].calls[-1]
    assert call["columns"] == ["unique_id", "ds", "y"] and "X_df" not in call


def test_supported_model_warns_only_about_static_and_historical_covariates(caplog):
    with caplog.at_level(logging.WARNING, logger=ffp.__name__):
        make_model(
            "FFChronos2",
            dynamic_future_numerical=["promo"],
            static_features=["store_type"],
            dynamic_historical_numerical=["traffic"],
        )
    messages = [r.message for r in caplog.records if "ignoring" in r.message]
    assert len(messages) == 1
    assert "static_features" in messages[0] and "dynamic_historical_numerical" in messages[0]
    assert "dynamic_future_numerical" not in messages[0]


def test_non_commercial_license_warns_once(caplog):
    with caplog.at_level(logging.WARNING, logger=ffp.__name__):
        make_model("FFTimesFM_3_0")
        make_model("FFTimesFM_3_0")
        make_model("FFChronos2")
    messages = [r.message for r in caplog.records if "non-commercial" in r.message]
    assert len(messages) == 1 and "FFTimesFM_3_0" in messages[0]


def test_revenue_capped_license_warns_once(caplog):
    with caplog.at_level(logging.WARNING, logger=ffp.__name__):
        make_model("FFTimestral_5M")
        make_model("FFTimestral_5M")
        make_model("FFT0Gamma")
    messages = [r.message for r in caplog.records if "revenue threshold" in r.message]
    assert len(messages) == 1 and "FFTimestral_5M" in messages[0]
    assert "mistral-modified-mit" in messages[0]


def test_t0_gamma_uses_published_inference_flags_without_cross_series_grouping():
    make_model("FFT0Gamma").predict(make_df(pd.date_range("2024-01-01", periods=30, freq="D")))
    kwargs = FakeFFModel.instances[-1].kwargs
    assert kwargs["cross_inference"] == 0
    assert all(kwargs[k] for k in ("trim_leading_missing", "symmetric_averaging", "clip_nonnegative", "snap_integers"))


def test_backtest_reports_metrics_per_series_and_window():
    model = make_model("FFChronos2")
    df = make_df(pd.date_range("2024-01-01", periods=40, freq="D"))
    res = model.backtest(df, start=pd.Timestamp("2024-01-31"))
    assert list(res.columns) == [
        "unique_id", "backtest_window_start_date", "metric_name", "metric_value",
        "forecast", "actual", "model_pickle", "forecast_quantiles",
    ]
    assert res["forecast_quantiles"].isna().all()  # no native quantiles for FoundationForecast models yet
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


def covariate_mlflow_model(**overrides):
    kwargs = dict(
        ff_class="Chronos", repo="amazon/chronos-2", ff_kwargs={}, prediction_length=H, freq="D",
        futr_exog_list=["promo"], name="FFChronos2",
    )
    kwargs.update(overrides)
    return ffp.FoundationForecastMLflowModel(**kwargs)


def history_and_future_rows(n_series=2):
    df = make_cov_df(DAILY, n_series=n_series)
    df.loc[df.ds > CUTOFF, "y"] = np.nan
    return df


def test_mlflow_model_takes_covariates_from_future_rows():
    mlflow_model = covariate_mlflow_model()
    model_input = history_and_future_rows()
    out = mlflow_model.predict(None, model_input)
    call = mlflow_model._model.calls[-1]
    assert call["futr_exog_list"] == ["promo"]
    assert call["df"]["ds"].max() == CUTOFF and call["df"]["y"].notna().all()
    expected = model_input[model_input.y.isna()].sort_values(["unique_id", "ds"])
    np.testing.assert_allclose(call["X_df"]["promo"], expected["promo"])
    assert list(out.columns) == ["unique_id", "ds", "y"] and len(out) == 2 * H
    assert pd.Timestamp(out["ds"].iloc[0]) == DAILY[20]
    assert pickle.loads(pickle.dumps(mlflow_model)).futr_exog_list == ["promo"]


def test_mlflow_model_without_future_rows_forecasts_without_covariates(caplog):
    mlflow_model = covariate_mlflow_model()
    with caplog.at_level(logging.WARNING, logger=ffp.__name__):
        mlflow_model.predict(None, make_cov_df(DAILY, n_series=2))
    call = mlflow_model._model.calls[-1]
    assert call["columns"] == ["unique_id", "ds", "y"] and "X_df" not in call
    assert any("forecasting without covariates" in r.message for r in caplog.records)


def test_mlflow_model_validates_future_covariates():
    model_input = history_and_future_rows()
    model_input = model_input.drop(model_input[(model_input.unique_id == "s1") & (model_input.ds == DAILY[21])].index)
    with pytest.raises(DataPreparationError, match="missing future rows"):
        covariate_mlflow_model().predict(None, model_input)
    with pytest.raises(DataPreparationError, match="Label-encode"):
        covariate_mlflow_model(ff_class="T0", name="FFT0Beta").predict(
            None, history_and_future_rows().assign(promo="x")
        )


def test_mlflow_model_registered_before_covariate_support_still_predicts():
    mlflow_model = covariate_mlflow_model()
    for attr in ("futr_exog_list", "accepts_categorical_strings", "name"):
        delattr(mlflow_model, attr)
    out = mlflow_model.predict(None, make_cov_df(DAILY, n_series=2))
    assert mlflow_model._model.calls[-1]["columns"] == ["unique_id", "ds", "y"]
    assert len(out) == 2 * H


@pytest.fixture
def logged(monkeypatch):
    captured = {}
    monkeypatch.setattr(ffp.mlflow.pyfunc, "log_model", lambda *args, **kwargs: captured.update(kwargs))
    return captured


def schema_types(schema):
    return {col.name: col.type.name for col in schema.inputs}


def test_register_without_covariates_keeps_the_univariate_schema(logged):
    make_model("FFChronos2").register("cat.schema.model")
    signature = logged["signature"]
    assert schema_types(signature.inputs) == {"unique_id": "string", "ds": "datetime", "y": "double"}
    assert signature.inputs == signature.outputs
    assert logged["python_model"].futr_exog_list == []


def test_register_with_covariates_uses_training_data_types(logged):
    from mlflow.models.utils import _enforce_schema

    model = make_model("FFChronos2", dynamic_future_numerical=["promo"], dynamic_future_categorical=["dow"])
    train = make_cov_df(DAILY)
    model.register("cat.schema.model", input_example=train)
    signature, example = logged["signature"], logged["input_example"]
    assert schema_types(signature.inputs) == {
        "unique_id": "string", "ds": "datetime", "y": "double", "promo": "double", "dow": "integer",
    }
    assert schema_types(signature.outputs) == {"unique_id": "string", "ds": "datetime", "y": "double"}
    assert example["unique_id"].nunique() == 1 and example["y"].isna().sum() == H
    assert example["y"].tail(H).isna().all()
    pyfunc = logged["python_model"]
    assert pyfunc.futr_exog_list == ["promo", "dow"] and pyfunc.accepts_categorical_strings
    out = pyfunc.predict(None, _enforce_schema(example, signature.inputs))
    assert len(out) == H and pd.Timestamp(out["ds"].iloc[0]) == example["ds"].iloc[-H]


def test_register_with_covariates_and_no_training_data_uses_a_synthetic_example(logged):
    make_model("FFT0Beta", dynamic_future_numerical=["promo"]).register("cat.schema.model")
    example = logged["input_example"]
    assert schema_types(logged["signature"].inputs)["promo"] == "double"
    out = logged["python_model"].predict(None, example)
    assert len(out) == H


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


def test_install_pins_agree_across_requirements_registry_and_extra():
    import tomllib
    from mmf_sa.models.abstract_model import MMF_PACKAGE, MODEL_PIP_REQUIREMENTS

    lines = [
        line.strip()
        for line in (REPO_ROOT / "requirements-foundationforecast.txt").read_text().splitlines()
        if line.strip() and not line.startswith("-r ")
    ]
    registered = [r for r in MODEL_PIP_REQUIREMENTS["foundationforecast"] if r != MMF_PACKAGE]
    assert registered == lines
    assert "--extra-index-url https://download.pytorch.org/whl/cu128" in lines
    assert {"torch==2.11.0+cu128", "torchvision==0.26.0+cu128", "torchaudio==2.11.0+cu128"} <= set(lines)

    extra = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())["project"]["optional-dependencies"]["foundationforecast"]
    assert extra == [r.split("+")[0] for r in lines if not r.startswith("--")]


def test_ensure_runtime_runs_once(monkeypatch):
    calls = []
    monkeypatch.setattr(_runtime, "_runtime_ready", False)
    monkeypatch.setattr(_runtime, "patch_tirex2_slstm_backend", lambda: calls.append(1))
    _runtime.ensure_runtime()
    _runtime.ensure_runtime()
    assert calls == [1]
