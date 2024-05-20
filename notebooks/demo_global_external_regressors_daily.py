# Databricks notebook source
# MAGIC %pip install -r ../requirements.txt --quiet

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
random.seed(7)

# Number of time series to sample
sample = True
size = 100
stores = sorted(random.sample(range(0, 1000), size))

train = spark.read.csv(f"/Volumes/{catalog}/{db}/{volume}/train.csv", header=True, inferSchema=True)
test = spark.read.csv(f"/Volumes/{catalog}/{db}/{volume}/test.csv", header=True, inferSchema=True)

if sample:
    train = train.filter(train.Store.isin(stores))
    test = test.filter(test.Store.isin(stores))

train.write.mode("overwrite").option("mergeSchema", "true").saveAsTable(f"{catalog}.{db}.rossmann_daily_train")
test.write.mode("overwrite").option("mergeSchema", "true").saveAsTable(f"{catalog}.{db}.rossmann_daily_test")

# COMMAND ----------

# Set the number of shuffle partitions larger than the total number of cores
#sqlContext.setConf("spark.sql.shuffle.partitions", "1000")

# COMMAND ----------

display(spark.sql(f"select * from {catalog}.{db}.rossmann_daily_train where Store=49 order by Date"))
display(spark.sql(f"select * from {catalog}.{db}.rossmann_daily_test where Store=49 order by Date"))

# COMMAND ----------

import pathlib
import pandas as pd
from forecasting_sa import run_forecast

active_models = [
    "NeuralForecastRNN",
    "NeuralForecastLSTM",
    "NeuralForecastNBEATSx",
    "NeuralForecastNHITS",
    "NeuralForecastAutoRNN",
    "NeuralForecastAutoLSTM",
    "NeuralForecastAutoNBEATSx",
    "NeuralForecastAutoNHITS",
    "NeuralForecastAutoTiDE",
    "NeuralForecastAutoPatchTST",
]

# COMMAND ----------

# MAGIC %md ### Now we can run the forecasting process using `run_forecast` function.

# COMMAND ----------

# MAGIC %md
# MAGIC We have to loop through the model in the following way else cuda will throw an error.

# COMMAND ----------

for model in active_models:
  dbutils.notebook.run(
    "run_global_external_regressors_daily",
    timeout_seconds=0,
    arguments={"catalog": catalog, "db": db, "model": model})

# COMMAND ----------

# MAGIC %sql select * from solacc_uc.mmf.rossmann_daily_evaluation_output order by Store, model, backtest_window_start_date

# COMMAND ----------

# MAGIC %sql select * from solacc_uc.mmf.rossmann_daily_scoring_output order by Store, model

# COMMAND ----------

# MAGIC %sql select * from solacc_uc.mmf.rossmann_daily_ensemble_output order by Store

# COMMAND ----------

# MAGIC %md ### Delete Tables

# COMMAND ----------

# MAGIC #%sql delete from solacc_uc.mmf.rossmann_daily_evaluation_output

# COMMAND ----------

# MAGIC #%sql delete from solacc_uc.mmf.rossmann_daily_scoring_output

# COMMAND ----------

# MAGIC #%sql delete from solacc_uc.mmf.rossmann_daily_ensemble_output
