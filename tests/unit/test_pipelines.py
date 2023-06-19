import mlflow
import datetime
from forecasting_sa import run_forecast

from .fixtures import temp_dir, spark_session, m4_df, m4_df_exogenous


def test_api_func(temp_dir, spark_session, m4_df, m4_df_exogenous):
    mlflow.set_tracking_uri(f"sqlite:///mlruns.db")
    spark_session.createDataFrame(m4_df).createOrReplaceTempView("train")

    active_models = [
        "StatsForecastAutoArima",
        #"GluonTSDeepAR",
        #"SKTimeLgbmDsDt",
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
        dynamic_reals=[],
        dynamic_future=[],
        dynamic_historical=[],
        static_features=[],
        train_predict_ratio=2,
        active_models=active_models,
        data_quality_check=True,
        ensemble=True,
        ensemble_metric="smape",
        ensemble_metric_avg=0.3,
        ensemble_metric_max=0.5,
        ensemble_scoring_output="ensemble_output",
        experiment_path=f"{str(temp_dir)}/fsa_experiment",
        use_case_name="fsa",
    )

    score = m4_df_exogenous
    last = max(score["ds"]) + datetime.timedelta(days=-35)
    score["y"] = score.apply(lambda x: None if x["ds"] > last else x["y"], axis=1)
    train = score.dropna()

    spark_session.createDataFrame(train).createOrReplaceTempView("train")
    spark_session.createDataFrame(score).createOrReplaceTempView("score")

    active_models = [
        "StatsForecastAutoArima",
    ]

    run_forecast(
        spark=spark_session,
        conf={"temp_path": f"{str(temp_dir)}/temp"},
        train_data="train",
        scoring_data="score",
        scoring_output="scoring_out",
        metrics_output="metrics",
        group_id="unique_id",
        date_col="ds",
        target="y",
        freq="D",
        dynamic_reals=["feature1", "feature2"],
        dynamic_future=[],
        dynamic_historical=[],
        static_features=[],
        train_predict_ratio=2,
        active_models=active_models,
        data_quality_check=True,
        ensemble=False,
        ensemble_metric="smape",
        ensemble_metric_avg=0.3,
        ensemble_metric_max=0.5,
        ensemble_scoring_output="ensemble_output",
        experiment_path=f"{str(temp_dir)}/fsa_experiment",
        use_case_name="fsa",
    )
