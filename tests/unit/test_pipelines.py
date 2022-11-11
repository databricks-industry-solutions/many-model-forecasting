import mlflow
from forecasting_sa import run_forecast

from .fixtures import temp_dir, spark_session, m4_df


def test_api_func(temp_dir, spark_session, m4_df):
    mlflow.set_tracking_uri(f"sqlite:///mlruns.db")
    spark_session.createDataFrame(m4_df).createOrReplaceTempView("train")

    active_models = [
        "SKTimeLgbmDsDt",
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
        active_models=["SKTimeLgbmDsDt"],  # active_models,
        ensemble=True,
        ensemble_metric="smape",
        ensemble_metric_avg=0.3,
        ensemble_metric_max=0.5,
        ensemble_scoring_output="ensemble_output",
        experiment_path=f"{str(temp_dir)}/fsa_experiment",
        use_case_name="fsa",
    )
