# Databricks notebook source
# MAGIC %md
# MAGIC # Many Models Forecasting Integration Test

# COMMAND ----------

# MAGIC %pip install -r ../requirements.txt --quiet
dbutils.library.restartPython()
# COMMAND ----------

import logging
from tqdm.autonotebook import tqdm
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

def transform_group(df):
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
    _ids = [f"D{i}" for i in range(1, 10)]
    y_df = (
        y_df.groupby("unique_id")
        .filter(lambda x: x.unique_id.iloc[0] in _ids)
        .groupby("unique_id")
        .apply(transform_group)
        .reset_index(drop=True)
    )
    return y_df

# COMMAND ----------

m4_df = spark.createDataFrame(create_m4_df())
m4_df.createOrReplaceTempView("mmf_train")

active_models = [
    #"StatsForecastBaselineWindowAverage",
    #"StatsForecastBaselineSeasonalWindowAverage",
    #"StatsForecastBaselineNaive",
    #"StatsForecastBaselineSeasonalNaive",
    "StatsForecastAutoArima",
    #"StatsForecastAutoETS",
    #"StatsForecastAutoCES",
    #"StatsForecastAutoTheta",
    #"StatsForecastTSB",
    #"StatsForecastADIDA",
    #"StatsForecastIMAPA",
    #"StatsForecastCrostonClassic",
    #"StatsForecastCrostonOptimized",
    #"StatsForecastCrostonSBA",
    "RFableArima",
    #"RFableETS",
    #"RFableNNETAR",
    #"RFableEnsemble",
    #"RDynamicHarmonicRegression",
    "SKTimeLgbmDsDt",
    #"SKTimeTBats",
    #"NeuralForecastRNN",
    #"NeuralForecastLSTM",
    #"NeuralForecastNBEATSx",
    #"NeuralForecastNHITS",
    #"NeuralForecastAutoRNN",
    #"NeuralForecastAutoLSTM",
    #"NeuralForecastAutoNBEATSx",
    #"NeuralForecastAutoNHITS",
]

# COMMAND ----------


# Make sure that the catalog and the schema exist
catalog = "main" # Name of the catalog we use to manage our assets
db = "mmf" # Name of the schema we use to manage our assets (e.g. datasets)

_ = spark.sql(f"CREATE CATALOG IF NOT EXISTS {catalog}")
_ = spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{db}")

# COMMAND ----------

run_forecast(
    spark=spark,
    train_data="mmf_train",
    scoring_data="mmf_train",
    scoring_output=f"{catalog}.{db}.daily_scoring_output",
    metrics_output=f"{catalog}.{db}.daily_metrics_output",
    group_id="unique_id",
    date_col="ds",
    target="y",
    freq="D",
    prediction_length=10,
    backtest_months=1,
    stride=10,
    train_predict_ratio=2,
    data_quality_check=True,
    resample=False,
    ensemble=True,
    ensemble_metric="smape",
    ensemble_metric_avg=0.3,
    ensemble_metric_max=0.5,
    ensemble_scoring_output=f"{catalog}.{db}.daily_ensemble_output",
    active_models=active_models,
    experiment_path=f"/Shared/mmf_experiment",
    use_case_name="mmf",
)

# COMMAND ----------

# MAGIC %sql drop table main.mmf.daily_metrics_output
# MAGIC %sql drop table main.mmf.daily_scoring_output
# MAGIC %sql drop table main.mmf.daily_ensemble_output

# COMMAND ----------


