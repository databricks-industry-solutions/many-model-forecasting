# Databricks notebook source
# MAGIC %md
# MAGIC # Many Models Forecasting SA (MMFSA) Demo
# MAGIC This demo highlights how to configure MMF SA to use M4 competition data

# COMMAND ----------

# MAGIC %pip install -r requirements.txt --quiet
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

# Number of time series
n = 100

def load_m4():
    y_df, _, _ = M4.load(directory=str(pathlib.Path.home()), group="Monthly")
    _ids = [f"M{i}" for i in range(1, n + 1)]
    y_df = (
        y_df.groupby("unique_id")
        .filter(lambda x: x.unique_id.iloc[0] in _ids)
        .groupby("unique_id")
        .apply(transform_group)
        .reset_index(drop=True)
    )
    return y_df

def transform_group(df):
    unique_id = df.unique_id.iloc[0]
    _cnt = 60  # df.count()[0]
    _start = pd.Timestamp("2018-01-01")
    _end = _start + pd.DateOffset(months=_cnt)
    date_idx = pd.date_range(start=_start, end=_end, freq="M", name="date")
    _df = (
        pd.DataFrame(data=[], index=date_idx)
        .reset_index()
        .rename(columns={"index": "date"})
    )
    _df["unique_id"] = unique_id
    _df["y"] = df[:60].y.values
    return _df

train = spark.createDataFrame(load_m4())
train.createOrReplaceTempView("mmf_train")

display(spark.read.table("mmf_train"))

# COMMAND ----------

# MAGIC %md ### Now the dataset looks in the following way:

# COMMAND ----------

# MAGIC %sql select unique_id, count(date) as count from mmf_train group by unique_id order by unique_id

# COMMAND ----------

# MAGIC %sql select count(distinct(unique_id)) from mmf_train

# COMMAND ----------

# MAGIC %md ### Let's configure the list of models we are going to use for training:

# COMMAND ----------

active_models = [
    "StatsForecastBaselineWindowAverage",
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
    "SKTimeTBats",
    #"SKTimeLgbmDsDt",
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

# MAGIC %md ### Now we can run the forecasting process using `run_forecast` function.

# COMMAND ----------

# Make sure that the catalog and the schema exist
catalog = "solacc_uc" # Name of the catalog we use to manage our assets
db = "mmf" # Name of the schema we use to manage our assets (e.g. datasets)

_ = spark.sql(f"CREATE CATALOG IF NOT EXISTS {catalog}")
_ = spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{db}")

# COMMAND ----------

run_forecast(
    spark=spark,
    train_data="mmf_train",
    scoring_data="mmf_train",
    scoring_output=f"{catalog}.{db}.monthly_scoring_output",
    metrics_output=f"{catalog}.{db}.monthly_metrics_output",
    group_id="unique_id",
    date_col="date",
    target="y",
    freq="M",
    prediction_length=3,
    backtest_months=12,
    stride=1,
    train_predict_ratio=2,
    data_quality_check=True,
    resample=False,
    ensemble=True,
    ensemble_metric="smape",
    ensemble_metric_avg=0.3,
    ensemble_metric_max=0.5,
    ensemble_scoring_output=f"{catalog}.{db}.monthly_ensemble_output",
    active_models=active_models,
    experiment_path=f"/Shared/mmf_experiment_monthly",
    use_case_name="mmf",
)

# COMMAND ----------

# MAGIC %md ### Metrics output
# MAGIC In the metrics output table, the metrics for all backtest windows and all models are stored. This info can be used to monitor model performance or decide which models should be taken into the final aggregated forecast.

# COMMAND ----------

# MAGIC %sql select * from solacc_uc.mmf.monthly_metrics_output order by unique_id, model, backtest_window_start_date

# COMMAND ----------

# MAGIC %md ### Forecast output
# MAGIC In the Forecast output table, the final forecast for each model and each time series is stored. 

# COMMAND ----------

# MAGIC %sql select * from solacc_uc.mmf.monthly_scoring_output order by unique_id, model, date

# COMMAND ----------

# MAGIC %md ### Final Ensemble Output
# MAGIC In the final ensemble output table, we store the averaged forecast. The models which meet the threshold defined using the ensembling parameters are taken into consideration

# COMMAND ----------

# MAGIC %sql select * from solacc_uc.mmf.monthly_ensemble_output order by unique_id, model, date

# COMMAND ----------

# MAGIC %sql delete from solacc_uc.mmf.monthly_metrics_output

# COMMAND ----------

# MAGIC %sql delete from solacc_uc.mmf.monthly_scoring_output

# COMMAND ----------

# MAGIC %sql delete from solacc_uc.mmf.monthly_ensemble_output

# COMMAND ----------


