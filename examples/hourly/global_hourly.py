# Databricks notebook source
# MAGIC %md
# MAGIC # Many Models Forecasting Demo
# MAGIC This notebook showcases how to run MMF with global models on multiple time series of hourly resolution. We will use [M4 competition](https://www.sciencedirect.com/science/article/pii/S0169207019301128#sec5) data. The descriptions here are mostly the same as the case with the [daily resolution](https://github.com/databricks-industry-solutions/many-model-forecasting/blob/main/examples/daily/global_daily.py), so we will skip the redundant parts and focus only on the essentials.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Cluster setup
# MAGIC
# MAGIC We recommend using a cluster with [Databricks Runtime 16.4 LTS for ML](https://docs.databricks.com/en/release-notes/runtime/16.4lts-ml.html). The cluster should be single-node with one or more GPU instances: e.g. [g4dn.12xlarge [T4]](https://aws.amazon.com/ec2/instance-types/g4/) on AWS or [Standard_NC64as_T4_v3](https://learn.microsoft.com/en-us/azure/virtual-machines/nct4-v3-series) on Azure.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Install and import packages
# MAGIC Check out [requirements-global.txt](https://github.com/databricks-industry-solutions/many-model-forecasting/blob/main/requirements-global.txt) if you're interested in the libraries we use.

# COMMAND ----------

# MAGIC %pip install datasetsforecast==0.0.8 --quiet
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import logging
logger = spark._jvm.org.apache.log4j
logging.getLogger("py4j.java_gateway").setLevel(logging.ERROR)
logging.getLogger("py4j.clientserver").setLevel(logging.ERROR)

# COMMAND ----------

import uuid
import pathlib
import pandas as pd
from datasetsforecast.m4 import M4

# COMMAND ----------

# MAGIC %md
# MAGIC ### Prepare data 
# MAGIC We are using [`datasetsforecast`](https://github.com/Nixtla/datasetsforecast/tree/main/) package to download M4 data.

# COMMAND ----------

# Number of time series
n = 100


def create_m4_hourly():
    y_df, _, _ = M4.load(directory=str(pathlib.Path.home()), group="Hourly")
    _ids = [f"H{i}" for i in range(1, n)]
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
    if len(df) > 720:
        df = df.iloc[-720:]
    _start = pd.Timestamp("2025-01-01 00:00")
    _end = _start + pd.DateOffset(hours=len(df)-1)
    date_idx = pd.date_range(start=_start, end=_end, freq="H", name="ds")
    res_df = pd.DataFrame(data=[], index=date_idx).reset_index()
    res_df["unique_id"] = unique_id
    res_df["y"] = df.y.values
    return res_df

# COMMAND ----------

# MAGIC %md
# MAGIC We are going to save this data in a delta lake table. Provide catalog and database names where you want to store the data.

# COMMAND ----------

catalog = "mmf" # Name of the catalog we use to manage our assets
db = "m4" # Name of the schema we use to manage our assets (e.g. datasets)
user = spark.sql('select current_user() as user').collect()[0]['user'] # User email address

# COMMAND ----------

# Making sure that the catalog and the schema exist
_ = spark.sql(f"CREATE CATALOG IF NOT EXISTS {catalog}")
_ = spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{db}")

(
    spark.createDataFrame(create_m4_hourly())
    .write.format("delta").mode("overwrite")
    .saveAsTable(f"{catalog}.{db}.m4_hourly_train")
)

# COMMAND ----------

# MAGIC %md Let's take a peak at the dataset:

# COMMAND ----------

display(spark.sql(f"select unique_id, count(ds) as count from {catalog}.{db}.m4_hourly_train group by unique_id order by unique_id"))

# COMMAND ----------

display(
  spark.sql(f"select * from {catalog}.{db}.m4_hourly_train where unique_id in ('H1', 'H2', 'H3', 'H4', 'H5') order by unique_id, ds")
  )

# COMMAND ----------

# MAGIC %md ### Models
# MAGIC Let's configure a list of models we are going to apply to our time series for evaluation and forecasting. A comprehensive list of all supported models is available in [mmf_sa/models/README.md](https://github.com/databricks-industry-solutions/many-model-forecasting/blob/main/mmf_sa/models/README.md). Look for the models where `model_type: global`; these are the global models we import from [neuralforecast](https://github.com/Nixtla/neuralforecast). Check their documentation for the detailed description of each model. 

# COMMAND ----------

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

# MAGIC %md ### Run MMF
# MAGIC
# MAGIC Now, we can run the evaluation and forecasting using `run_forecast` function defined in [mmf_sa/models/__init__.py](https://github.com/databricks-industry-solutions/many-model-forecasting/blob/main/mmf_sa/models/__init__.py). Refer to [README.md](https://github.com/databricks-industry-solutions/many-model-forecasting/blob/main/README.md#parameters-description) for a comprehensive description of each parameter. Make sure to set `freq="H"` in `run_forecast` function called in [examples/run_hourly.py](https://github.com/databricks-industry-solutions/many-model-forecasting/blob/main/examples/run_hourly.py).

# COMMAND ----------

# The same run_id will be assigned to all the models. This makes it easier to run the post evaluation analysis later.
run_id = str(uuid.uuid4())

for model in active_models:
  dbutils.notebook.run(
    "../run_hourly",
    timeout_seconds=0,
    arguments={"catalog": catalog, "db": db, "model": model, "run_id": run_id, "user": user})

# COMMAND ----------

# MAGIC %md ### Evaluate
# MAGIC In `evaluation_output` table, the we store all evaluation results for all backtesting trials from all models.

# COMMAND ----------

display(spark.sql(f"""
    select * from {catalog}.{db}.hourly_evaluation_output 
    where unique_id = 'H1'
    order by unique_id, model, backtest_window_start_date
    """))

# COMMAND ----------

# MAGIC %md ### Forecast
# MAGIC In `scoring_output` table, forecasts for each time series from each model are stored.

# COMMAND ----------

display(spark.sql(f"""
    select * from {catalog}.{db}.hourly_scoring_output 
    where unique_id = 'H1'
    order by unique_id, model, ds
    """))

# COMMAND ----------

# MAGIC %md
# MAGIC Refer to the [notebook](https://github.com/databricks-industry-solutions/many-model-forecasting/blob/main/examples/post-evaluation-analysis.ipynb) for guidance on performing fine-grained model selection after running `run_forecast`.

# COMMAND ----------

# MAGIC %md ### Delete Tables
# MAGIC Let's clean up the tables.

# COMMAND ----------

#display(spark.sql(f"delete from {catalog}.{db}.hourly_evaluation_output"))

# COMMAND ----------

#display(spark.sql(f"delete from {catalog}.{db}.hourly_scoring_output"))
