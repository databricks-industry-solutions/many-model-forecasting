# Databricks notebook source
# MAGIC %md
# MAGIC # Many Models Forecasting Demo
# MAGIC This notebook showcases how to run MMF with foundation models on multiple time series of monthly resolution. We will use [M4 competition](https://www.sciencedirect.com/science/article/pii/S0169207019301128#sec5) data. The descriptions here are mostly the same as the case with the [daily resolution](https://github.com/databricks-industry-solutions/many-model-forecasting/blob/main/examples/foundation_daily.py), so we will skip the redundant parts and focus only on the essentials.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Cluster setup
# MAGIC
# MAGIC We recommend using a cluster with [Databricks Runtime 14.3 LTS for ML](https://docs.databricks.com/en/release-notes/runtime/14.3lts-ml.html) or above. The cluster should be single-node with one or more GPU instances: e.g. [g4dn.12xlarge [T4]](https://aws.amazon.com/ec2/instance-types/g4/) on AWS or [Standard_NC64as_T4_v3](https://learn.microsoft.com/en-us/azure/virtual-machines/nct4-v3-series) on Azure. MMF leverages [Pandas UDF](https://docs.databricks.com/en/udf/pandas.html) for distributing the inference tasks and utilizing all the available resource.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Install and import packages
# MAGIC Check out [requirements.txt](https://github.com/databricks-industry-solutions/many-model-forecasting/blob/main/requirements.txt) if you're interested in the libraries we use.

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


def create_m4_monthly():
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


# COMMAND ----------

# MAGIC %md
# MAGIC We are going to save this data in a delta lake table. Provide catalog and database names where you want to store the data.__

# COMMAND ----------

catalog = "mmf" # Name of the catalog we use to manage our assets
db = "m4" # Name of the schema we use to manage our assets (e.g. datasets)
user = spark.sql('select current_user() as user').collect()[0]['user'] # User email address

# COMMAND ----------

# Making sure that the catalog and the schema exist
_ = spark.sql(f"CREATE CATALOG IF NOT EXISTS {catalog}")
_ = spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{db}")

(
    spark.createDataFrame(create_m4_monthly())
    .write.format("delta").mode("overwrite")
    .saveAsTable(f"{catalog}.{db}.m4_monthly_train")
)

# COMMAND ----------

# MAGIC %md Let's take a peak at the dataset:

# COMMAND ----------

display(spark.sql(f"select unique_id, count(date) as count from {catalog}.{db}.m4_monthly_train group by unique_id order by unique_id"))

# COMMAND ----------

display(
  spark.sql(f"select * from {catalog}.{db}.m4_monthly_train where unique_id in ('M1', 'M2', 'M3', 'M4', 'M5') order by unique_id, date")
  )

# COMMAND ----------

# MAGIC %md ### Models
# MAGIC Let's configure a list of models we are going to apply to our time series for evaluation and forecasting. A comprehensive list of all supported models is available in [mmf_sa/models/models_conf.yaml](https://github.com/databricks-industry-solutions/many-model-forecasting/blob/main/mmf_sa/models/models_conf.yaml). Look for the models where `model_type: foundation`; these are the foundation models we import from [chronos](https://github.com/amazon-science/chronos-forecasting/tree/main), [uni2ts](https://github.com/SalesforceAIResearch/uni2ts) and [moment](https://github.com/moment-timeseries-foundation-model/moment). Check their documentation for the detailed description of each model. 

# COMMAND ----------

active_models = [
    "ChronosT5Tiny",
    "ChronosT5Mini",
    "ChronosT5Small",
    "ChronosT5Base",
    "ChronosT5Large",
    "ChronosBoltTiny",
    "ChronosBoltMini",
    "ChronosBoltSmall",
    "ChronosBoltBase",
    "MoiraiSmall",
    "MoiraiBase",
    "MoiraiLarge",
    "TimesFM_1_0_200m",
    "TimesFM_2_0_500m",
]

# COMMAND ----------

# MAGIC %md ### Run MMF
# MAGIC
# MAGIC Now, we can run the evaluation and forecasting using `run_forecast` function defined in [mmf_sa/models/__init__.py](https://github.com/databricks-industry-solutions/many-model-forecasting/blob/main/mmf_sa/models/__init__.py). Refer to [README.md](https://github.com/databricks-industry-solutions/many-model-forecasting/blob/main/README.md#parameters-description) for a comprehensive description of each parameter.

# COMMAND ----------

# The same run_id will be assigned to all the models. This makes it easier to run the post evaluation analysis later.
run_id = str(uuid.uuid4())

for model in active_models:
  dbutils.notebook.run(
    "run_monthly",
    timeout_seconds=0,
    arguments={"catalog": catalog, "db": db, "model": model, "run_id": run_id, "user": user})

# COMMAND ----------

# MAGIC %md ### Evaluate
# MAGIC In `evaluation_output` table, the we store all evaluation results for all backtesting trials from all models. This information can be used to understand which models performed well on which time series on which periods of backtesting. This is very important for selecting the final model for forecasting or models for ensembling. Maybe, it's faster to take a look at the table:

# COMMAND ----------

display(spark.sql(f"""
    select * from {catalog}.{db}.monthly_evaluation_output 
    where unique_id = 'M1'
    order by unique_id, model, backtest_window_start_date
    """))

# COMMAND ----------

# MAGIC %md ### Forecast
# MAGIC In `scoring_output` table, forecasts for each time series from each model are stored.

# COMMAND ----------

display(spark.sql(f"""
    select * from {catalog}.{db}.monthly_scoring_output 
    where unique_id = 'M1'
    order by unique_id, model, date
    """))

# COMMAND ----------

# MAGIC %md ### Delete Tables
# MAGIC Let's clean up the tables.

# COMMAND ----------

#display(spark.sql(f"delete from {catalog}.{db}.monthly_evaluation_output"))

# COMMAND ----------

#display(spark.sql(f"delete from {catalog}.{db}.monthly_scoring_output"))
