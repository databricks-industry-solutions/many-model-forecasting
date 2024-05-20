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
            "metric": "smape",
            "active_models": ["StatsForecastAutoArima"],
            "dynamic_reals": ["feature1", "feature2"]
        }
    )


def test_exogenous_regressors(base_config, m4_df_exogenous):
    model_registry = ModelRegistry(base_config)
    model = model_registry.get_model("StatsForecastAutoArima")
    _df = m4_df_exogenous[m4_df_exogenous.unique_id == "D8"]
    _hist_df = _df[:-10]
    _val_df = _df[-10:]
    res_df = model.predict(_hist_df, _val_df)
    print(
        model.backtest(
            _df, start=_df.ds.max() - pd.DateOffset(days=35), stride=10, retrain=True
        )
    )
