import mlflow
import datetime
from mmf_sa import run_forecast

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
        evaluation_output="evaluation_output",
        group_id="unique_id",
        date_col="ds",
        target="y",
        freq="D",
        dynamic_future_numerical=[],
        dynamic_future_categorical=[],
        dynamic_historical_numerical=[],
        dynamic_historical_categorical=[],
        static_features=[],
        train_predict_ratio=2,
        active_models=active_models,
        data_quality_check=True,
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
        evaluation_output="evaluation_output",
        group_id="unique_id",
        date_col="ds",
        target="y",
        freq="D",
        dynamic_future_categorical=["feature1", "feature2"],
        dynamic_historical_categorical=[],
        static_features=[],
        train_predict_ratio=2,
        active_models=active_models,
        data_quality_check=True,
        experiment_path=f"{str(temp_dir)}/fsa_experiment",
        use_case_name="fsa",
    )
