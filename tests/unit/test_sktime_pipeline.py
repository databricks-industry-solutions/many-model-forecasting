import pandas as pd
import pytest

from forecasting_sa.models import ModelRegistry
from omegaconf import OmegaConf
from .fixtures import m4_df


@pytest.fixture
def base_config():
    return OmegaConf.create(
        {
            "date_col": "ds",
            "target": "y",
            "freq": "D",
            "prediction_length": 10,
            "active_models": ["SKTimeLgbmDsDt", "SKTimeTBats"],
        }
    )


def test_sktime_lgbm_ds_dt(base_config, m4_df):
    model_registry = ModelRegistry(base_config)
    model = model_registry.get_model("SKTimeLgbmDsDt")
    _df = m4_df[m4_df.unique_id == "D8"]
    model.fit(_df)
    res_df = model.predict(_df)
    print(
        model.backtest(
            _df, start=_df.ds.max() - pd.DateOffset(days=35), stride=10, retrain=True
        )
    )


def test_sktime_tbats(base_config, m4_df):
    model_registry = ModelRegistry(base_config)
    model = model_registry.get_model("SKTimeTBats")
    _df = m4_df[m4_df.unique_id == "D8"]
    model.fit(_df)
    res_df = model.predict(_df)
    print(
        model.backtest(
            _df, start=_df.ds.max() - pd.DateOffset(days=35), stride=10, retrain=False
        )
    )
