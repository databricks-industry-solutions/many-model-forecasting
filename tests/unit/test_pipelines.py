import yaml
import pathlib
import pandas as pd
import pytest
import mlflow
from datasetsforecast.m4 import M4
from forecasting_sa import run_forecast

from .fixtures import temp_dir, spark_session


def _transform_group(df):
    unique_id = df.unique_id.iloc[0]
    _start = pd.Timestamp("2020-01-01")
    _end = _start + pd.DateOffset(days=int(df.count()[0]) - 1)
    date_idx = pd.date_range(start=_start, end=_end, freq="D", name="ds")
    res_df = pd.DataFrame(data=[], index=date_idx).reset_index()
    res_df["unique_id"] = unique_id
    res_df["y"] = df.y.values
    return res_df


@pytest.fixture
def m4_df():
    y_df, _, _ = M4.load(directory=str(pathlib.Path.home()), group="Daily")
    _ids = [f"D{i}" for i in range(1, 10)]
    y_df = (
        y_df.groupby("unique_id")
        .filter(lambda x: x.unique_id.iloc[0] in _ids)
        .groupby("unique_id")
        .apply(_transform_group)
        .reset_index(drop=True)
    )
    return y_df


def test_api_func(temp_dir, spark_session, m4_df):
    mlflow.set_tracking_uri(f"sqlite:///mlruns.db")
    spark_session.createDataFrame(
        m4_df[m4_df.ds > "2021-12-31"]
    ).createOrReplaceTempView("train")

    active_models = [
        "StatsForecastArima",
        "StatsForecastETS",
        "StatsForecastCES",
        "StatsForecastTSB",
        "StatsForecastADIDA",
        "StatsForecastIMAPA",
        "StatsForecastCrostonSBA",
        "StatsForecastCrostonOptimized",
        "StatsForecastCrostonClassic",
        "StatsForecastBaselineWindowAverage",
        "StatsForecastBaselineSeasonalWindowAverage",
        "StatsForecastBaselineNaive",
        "StatsForecastBaselineSeasonalNaive",
        "GluonTSTorchDeepAR",
    ]

    run_forecast(
        spark=spark_session,
        conf={"temp_path": f"{str(temp_dir)}/temp"},
        train_data="train",
        scoring_data="train",
        scoring_output="scoring_out",
        metrics_output="metrics",
        group_id="unique_id",
        date_col="ds",
        target="y",
        freq="D",
        train_predict_ratio=2,
        active_models=active_models,
        experiment_path=f"{str(temp_dir)}/fsa_experiment",
        use_case_name="fsa",
    )
