"""Checks MMF against the installed foundationforecast package. No weights are loaded:
the native covariate forecast, the only step that needs a checkpoint, is replaced."""
import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

pytest.importorskip("foundationforecast")

from foundationforecast import models as ff_models  # noqa: E402

from mmf_sa.models import ModelRegistry  # noqa: E402
from mmf_sa.models.foundationforecast import FoundationForecastPipeline as ffp  # noqa: E402

from .test_foundationforecast_pipeline import EXPECTED_REPOS, H, make_cov_df, split_at  # noqa: E402

COVARIATE_MODELS = sorted(n for n in EXPECTED_REPOS if getattr(ffp, n).supports_future_covariates)


@pytest.fixture(scope="module", autouse=True)
def spark_session():
    # Overrides the autouse Spark session from conftest; these tests never use Spark.
    yield None


def make_model(name, **overrides):
    conf = {
        "active_models": [name], "prediction_length": H, "group_id": "unique_id", "date_col": "ds",
        "target": "y", "metric": "smape", "freq": "D", "backtest_length": 8, "stride": H,
    }
    conf.update(overrides)
    return ModelRegistry(OmegaConf.create(conf)).get_model(name)


@pytest.mark.parametrize("name", sorted(EXPECTED_REPOS))
def test_mmf_support_flag_matches_foundationforecast(name):
    model_cls = getattr(ffp, name)
    ff_model = getattr(ff_models, model_cls.ff_class)(repo_id=EXPECTED_REPOS[name])
    assert ff_model.supports_native_futr_exog() == model_cls.supports_future_covariates


def chronos2_timestamp_check(ff_model, df, h, freq, horizon_df, futr_exog_list):
    """Runs the Chronos-2 input validation of Chronos2Pipeline.predict_df."""
    from chronos.df_utils import make_future_df, validate_and_normalize_df

    context = ff_model._to_chronos2_df(df, futr_exog_list)
    future = horizon_df[["unique_id", "ds", *futr_exog_list]].rename(
        columns={"unique_id": "item_id", "ds": "timestamp"}
    )
    context, future = validate_and_normalize_df(context, future, ["target"], h)
    expected = make_future_df(context, h, freq=freq)
    assert (pd.DatetimeIndex(future["timestamp"]) == pd.DatetimeIndex(expected["timestamp"])).all()


@pytest.mark.parametrize("freq, dates", [
    ("D", pd.date_range("2024-01-01", periods=40, freq="D")),
    ("W", pd.date_range("2023-01-02", periods=40, freq="W-MON")),
    ("W", pd.date_range("2023-01-01", periods=40, freq="W-SUN")),
    ("M", pd.date_range("2020-01-31", periods=40, freq="ME")),
    ("H", pd.date_range("2024-03-01", periods=40, freq="h")),
], ids=["daily", "weekly_monday", "weekly_sunday", "monthly", "hourly"])
@pytest.mark.parametrize("name", COVARIATE_MODELS)
def test_mmf_frames_pass_foundationforecast_validation(name, freq, dates, monkeypatch):
    model = make_model(name, freq=freq, dynamic_future_numerical=["promo"], dynamic_future_categorical=["dow"])
    captured = {}

    def native(self, df, h, freq, level, quantiles, panel, horizon_df, futr_exog_list):
        captured.update(df=df, horizon_df=horizon_df, futr_exog_list=futr_exog_list, freq=freq)
        if isinstance(self, ff_models.Chronos):
            chronos2_timestamp_check(self, df, h, freq, horizon_df, futr_exog_list)
        return horizon_df[["unique_id", "ds"]].assign(**{self.alias: 1.0})

    # TimesFM's constructor returns a version-specific subclass, so patch the built instance's class.
    monkeypatch.setattr(type(model._get_ff_model()), "_forecast_native_futr_exog", native)
    hist, future = split_at(make_cov_df(dates), dates[-H - 1])
    pred, _ = model.predict(hist, future)

    assert captured["futr_exog_list"] == ["promo", "dow"]
    assert captured["horizon_df"].groupby("unique_id").size().eq(H).all()
    assert len(pred) == 3
    for values in pred["y"]:
        np.testing.assert_allclose(values, np.ones(H))
