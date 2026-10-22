import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

pytest.importorskip("torch")


def _history_and_future(n_series=2, n_months=12, horizon=3):
    hist = pd.DataFrame([
        {"unique_id": f"s{i}", "ds": ds, "y": float(t)}
        for i in range(n_series)
        for t, ds in enumerate(pd.date_range("2020-01-31", periods=n_months, freq="ME"))
    ])
    future = hist.groupby("unique_id").tail(horizon).copy()
    future["ds"] = future["ds"] + pd.offsets.MonthEnd(horizon)
    future["y"] = np.nan
    return hist, pd.concat([hist, future], ignore_index=True)


@pytest.mark.parametrize("module, cls", [
    ("mmf_sa.models.chronosforecast.ChronosPipeline", "ChronosBoltTiny"),
    ("mmf_sa.models.moiraiforecast.MoiraiPipeline", "MoiraiSmall"),
])
def test_forecast_uses_only_observed_history(module, cls, monkeypatch):
    """Scoring passes the future rows (target is null); forecast() must start after the last observation."""
    model_cls = getattr(pytest.importorskip(module), cls)
    model = model_cls(OmegaConf.create({
        "group_id": "unique_id", "date_col": "ds", "target": "y", "freq": "M",
        "prediction_length": 3, "batch_size": 8, "metric": "smape",
    }))
    hist, scoring_df = _history_and_future()
    seen = {}

    def fake_predict(hist_df, val_df=None, curr_date=None, spark=None):
        seen["hist_df"] = hist_df
        return pd.DataFrame(), None

    monkeypatch.setattr(model, "predict", fake_predict)
    model.forecast(scoring_df)

    assert seen["hist_df"]["y"].notnull().all()
    assert seen["hist_df"]["ds"].max() == hist["ds"].max()
    assert len(seen["hist_df"]) == len(hist)
