# Databricks notebook source
# MAGIC %pip install -r requirements.txt --quiet

# COMMAND ----------

import logging
from tqdm.autonotebook import tqdm
logger = spark._jvm.org.apache.log4j
logging.getLogger("py4j.java_gateway").setLevel(logging.ERROR)
logging.getLogger("py4j.clientserver").setLevel(logging.ERROR)

# COMMAND ----------

# Make sure that the catalog and the schema exist
catalog = "solacc_uc" # Name of the catalog we use to manage our assets
db = "mmf" # Name of the schema we use to manage our assets (e.g. datasets)
volume = "rossmann" # Name of the schema where you have your rossmann dataset csv sotred

_ = spark.sql(f"CREATE CATALOG IF NOT EXISTS {catalog}")
_ = spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{db}")
_ = spark.sql(f"CREATE VOLUME IF NOT EXISTS {catalog}.{db}.{volume}")

# COMMAND ----------

# MAGIC %md Download the dataset from [Kaggle](kaggle.com/competitions/rossmann-store-sales/data) and store them in the volume.

# COMMAND ----------

# Randomly select 100 stores to forecast
import random
random.seed(100)
sample = False
stores = sorted(random.sample(range(0, 1000), 100))

train = spark.read.csv(f"/Volumes/{catalog}/{db}/{volume}/train.csv", header=True, inferSchema=True)
test = spark.read.csv(f"/Volumes/{catalog}/{db}/{volume}/test.csv", header=True, inferSchema=True)

if sample:
    train = train.filter(train.Store.isin(stores))
    test = test.filter(test.Store.isin(stores))

train.write.mode("overwrite").option("mergeSchema", "true").saveAsTable(f"{catalog}.{db}.rossmann_train")
test.write.mode("overwrite").option("mergeSchema", "true").saveAsTable(f"{catalog}.{db}.rossmann_test")

# COMMAND ----------

# Set the number of shuffle partitions larger than the total number of cores
sqlContext.setConf("spark.sql.shuffle.partitions", "1000")

# COMMAND ----------

display(spark.sql(f"select * from {catalog}.{db}.rossmann_train where Store=49 order by Date"))
display(spark.sql(f"select * from {catalog}.{db}.rossmann_test where Store=49 order by Date"))

# COMMAND ----------

import pathlib
import pandas as pd
from forecasting_sa import run_forecast

active_models = [
    "StatsForecastBaselineWindowAverage",
    "StatsForecastBaselineSeasonalWindowAverage",
    "StatsForecastBaselineNaive",
    "StatsForecastBaselineSeasonalNaive",
    "StatsForecastAutoArima",
    "StatsForecastAutoETS",
    "StatsForecastAutoCES",
    "StatsForecastAutoTheta",
    "StatsForecastTSB",
    "StatsForecastADIDA",
    "StatsForecastIMAPA",
    "StatsForecastCrostonClassic",
    "StatsForecastCrostonOptimized",
    "StatsForecastCrostonSBA",
    "RFableArima",
    "RFableETS",
    "RFableNNETAR",
    "RFableEnsemble",
    "RDynamicHarmonicRegression",
    #"SKTimeLgbmDsDt",
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

run_id = run_forecast(
    spark=spark,
    train_data=f"{catalog}.{db}.rossmann_train",
    scoring_data=f"{catalog}.{db}.rossmann_test",
    scoring_output=f"{catalog}.{db}.rossmann_scoring_output",
    metrics_output=f"{catalog}.{db}.rossmann_metrics_output",
    group_id="Store",
    date_col="Date",
    target="Sales",
    freq="D",
    dynamic_reals=["DayOfWeek", "Open", "Promo", "SchoolHoliday"],
    prediction_length=10,
    backtest_months=1,
    stride=10,
    train_predict_ratio=2,
    active_models=active_models,
    data_quality_check=True,
    resample=False,
    ensemble=True,
    ensemble_metric="smape",
    ensemble_metric_avg=0.3,
    ensemble_metric_max=0.5,
    ensemble_scoring_output=f"{catalog}.{db}.rossmann_ensemble_output",
    experiment_path=f"/Shared/mmf_rossmann",
    use_case_name="mmf_rossmann",
)
print(run_id)

# COMMAND ----------

# MAGIC %sql select * from solacc_uc.mmf.rossmann_metrics_output order by Store, model, backtest_window_start_date

# COMMAND ----------

# MAGIC %sql select * from solacc_uc.mmf.rossmann_scoring_output order by Store, model

# COMMAND ----------

# MAGIC %sql select * from solacc_uc.mmf.rossmann_ensemble_output order by Store

# COMMAND ----------

# MAGIC %sql delete from solacc_uc.mmf.rossmann_metrics_output

# COMMAND ----------

# MAGIC %sql delete from solacc_uc.mmf.rossmann_scoring_output

# COMMAND ----------

# MAGIC %sql delete from solacc_uc.mmf.rossmann_ensemble_output

# COMMAND ----------



