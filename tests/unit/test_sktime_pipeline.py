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
            "active_models": ["SKTimeLgbmDsDt"],
        }
    )


def test_sktime(base_config, m4_df):
    model_registry = ModelRegistry(base_config)
    model = model_registry.get_model("SKTimeLgbmDsDt")
    model.param_grid = {
        "deseasonalise__model": ["additive", "multiplicative"],
        "detrend__forecaster__degree": [1, 2],
    }
    _df = m4_df[m4_df.unique_id == "D2"]
    model.fit(_df)
    res_df = model.predict(_df)
    print(
        model.backtest(
            _df, start=_df.ds.max() - pd.DateOffset(days=35), stride=10, retrain=True
        )
    )
