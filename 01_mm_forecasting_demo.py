# Databricks notebook source
# MAGIC %md
# MAGIC # Many Models Forecasting SA (MMFSA) Demo
# MAGIC This demo highlights how to configure MMF SA to use M4 competition data

# COMMAND ----------

# MAGIC %pip install -r requirements.txt
# MAGIC %pip install datasetsforecast

# COMMAND ----------

import logging

logger = spark._jvm.org.apache.log4j
logging.getLogger("py4j.java_gateway").setLevel(logging.ERROR)
logging.getLogger("py4j.clientserver").setLevel(logging.ERROR)

# COMMAND ----------

import pathlib
import pandas as pd
from datasetsforecast.m4 import M4
from forecasting_sa import run_forecast

# COMMAND ----------

# MAGIC %md
# MAGIC ### Data preparation steps 
# MAGIC We are using `datasetsforecast` package to download M4 data. 
# MAGIC 
# MAGIC M4 dataset contains a set of time series which we use for testing of MMF SA. 
# MAGIC 
# MAGIC Below we have developed a number of functions to convert M4 time series to the expected format. 

# COMMAND ----------


def _transform_group(df):
    unique_id = df.unique_id.iloc[0]
    _start = pd.Timestamp("2020-01-01")
    _end = _start + pd.DateOffset(days=int(df.count()[0]) - 1)
    date_idx = pd.date_range(start=_start, end=_end, freq="D", name="ds")
    res_df = pd.DataFrame(data=[], index=date_idx).reset_index()
    res_df["unique_id"] = unique_id
    res_df["y"] = df.y.values
    return res_df


def create_m4_df():
    y_df, _, _ = M4.load(directory=str(pathlib.Path.home()), group="Daily")
    _ids = [f"D{i}" for i in range(1, 100)]
    y_df = (
        y_df.groupby("unique_id")
        .filter(lambda x: x.unique_id.iloc[0] in _ids)
        .groupby("unique_id")
        .apply(_transform_group)
        .reset_index(drop=True)
    )
    return y_df


# COMMAND ----------

# MAGIC %md ### Now the dataset looks in the following way:

# COMMAND ----------

m4_df = spark.createDataFrame(create_m4_df())
m4_df.createOrReplaceTempView("mmf_sa_train")
display(m4_df)

# COMMAND ----------

# MAGIC %sql select * from mmf_sa_train where unique_id in ('D1', 'D2', 'D6', 'D7')

# COMMAND ----------

# MAGIC %md ### Let's configure the list of models we are going to use for training:

# COMMAND ----------

active_models = [
    "StatsForecastAutoArima",
    "StatsForecastAutoETS",
    "StatsForecastAutoCES",
    "StatsForecastAutoTheta",
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

# COMMAND ----------

# MAGIC %md ### Now we can run the forecasting process using `run_forecast` function.

# COMMAND ----------

run_forecast(
    spark=spark,
    # conf={"temp_path": f"{str(temp_dir)}/temp"},
    train_data="mmf_sa_train",
    scoring_data="mmf_sa_train",
    scoring_output="mmf_sa_forecast_scoring_out",
    metrics_output="mmf_sa_metrics",
    group_id="unique_id",
    date_col="ds",
    target="y",
    freq="D",
    #data_quality_check=False,
    ensemble=True,
    ensemble_metric="smape",
    ensemble_metric_avg=0.3,
    ensemble_metric_max=0.5,
    ensemble_scoring_output="mmf_sa_forecast_ensemble_out",
    train_predict_ratio=2,
    active_models=active_models,
    experiment_path=f"/Shared/fsa_cicd_pr_experiment",
    use_case_name="fsa",
)

# COMMAND ----------

# MAGIC %md ### Metrics output
# MAGIC In the metrics output table, the metrics for all backtest windows and all models are stored. This info can be used to monitor model performance or decide which models should be taken into the final aggregated forecast.

# COMMAND ----------

# MAGIC %sql select * from mmf_sa_metrics

# COMMAND ----------

# MAGIC %md ### Forecast output
# MAGIC In the Forecast output table, the final forecast for each model and each time series is stored. 

# COMMAND ----------

# MAGIC %sql select * from mmf_sa_forecast_scoring_out

# COMMAND ----------

# MAGIC %md ### Final Ensemble Output
# MAGIC In the final ensemble output table, we store the averaged forecast. The models which meet the threshold defined using the ensembling parameters are taken into consideration

# COMMAND ----------

# MAGIC %sql select * from mmf_sa_forecast_ensemble_out

# COMMAND ----------


