import pandas as pd
import pytest

from mmf_sa.models import ModelRegistry
from omegaconf import OmegaConf
from .fixtures import m4_df_exogenous


@pytest.fixture
def base_config():
    return OmegaConf.create(
        {
            "date_col": "ds",
            "target": "y",
            "group_id": "unique_id",
            "freq": "D",
            "prediction_length": 10,
            "stride": 10,
            "metric": "smape",
            "active_models": ["StatsForecastAutoArima"],
            "dynamic_future_numerical": ["feature1", "feature2"]
        }
    )


def test_exogenous_regressors(base_config, m4_df_exogenous):
    model_registry = ModelRegistry(base_config)
    model = model_registry.get_model("StatsForecastAutoArima")
    _df = m4_df_exogenous[m4_df_exogenous.unique_id == "D8"]
    _hist_df = _df[:-10]
    _val_df = _df[-10:]
    forecast_df, _ = model.predict(_hist_df, _val_df)
    assert len(forecast_df) == 10
    assert forecast_df["y"].notnull().all()

    res_df = model.backtest(_df, start=_df.ds.max() - pd.DateOffset(days=35), group_id="D8")
    assert len(res_df) == 3
    assert res_df["metric_value"].notnull().all()
