# Databricks notebook source
# MAGIC %pip install datasetsforecast==0.0.8 --quiet
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import pathlib
import pandas as pd
from datasetsforecast.m4 import M4
import logging
logger = spark._jvm.org.apache.log4j
logging.getLogger("py4j.java_gateway").setLevel(logging.ERROR)
logging.getLogger("py4j.clientserver").setLevel(logging.ERROR)

# COMMAND ----------

dbutils.widgets.text("catalog", "")
dbutils.widgets.text("db", "")
dbutils.widgets.text("n", "")
dbutils.widgets.text("volume", "rossmann")

catalog = dbutils.widgets.get("catalog")  # Name of the catalog we use to manage our assets
db = dbutils.widgets.get("db") # Name of the schema we use to manage our assets (e.g. datasets)
volume = dbutils.widgets.get("volume")  # Name of the schema where you have your rossmann dataset csv sotred
n = int(dbutils.widgets.get("n"))  # Number of time series to sample

#  Make sure the catalog, schema and volume exist
_ = spark.sql(f"CREATE CATALOG IF NOT EXISTS {catalog}")
_ = spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{db}")
_ = spark.sql(f"CREATE VOLUME IF NOT EXISTS {catalog}.{db}.{volume}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Daily M4 Data

# COMMAND ----------

def create_m4_daily():
    y_df, _, _ = M4.load(directory=str(pathlib.Path.home()), group="Daily")
    _ids = [f"D{i}" for i in range(1, n)]
    y_df = (
        y_df.groupby("unique_id")
        .filter(lambda x: x.unique_id.iloc[0] in _ids)
        .groupby("unique_id")
        .apply(transform_group_daily)
        .reset_index(drop=True)
    )
    return y_df


def transform_group_daily(df):
    unique_id = df.unique_id.iloc[0]
    _start = pd.Timestamp("2020-01-01")
    _end = _start + pd.DateOffset(days=int(df.count()[0]) - 1)
    date_idx = pd.date_range(start=_start, end=_end, freq="D", name="ds")
    res_df = pd.DataFrame(data=[], index=date_idx).reset_index()
    res_df["unique_id"] = unique_id
    res_df["y"] = df.y.values
    return res_df

 
(
    spark.createDataFrame(create_m4_daily())
    .write.format("delta").mode("overwrite")
    .saveAsTable(f"{catalog}.{db}.m4_daily_train")
)

print(f"Saved data to {catalog}.{db}.m4_daily_train")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Monthly M4 Data

# COMMAND ----------

def create_m4_monthly():
    y_df, _, _ = M4.load(directory=str(pathlib.Path.home()), group="Monthly")
    _ids = [f"M{i}" for i in range(1, n + 1)]
    y_df = (
        y_df.groupby("unique_id")
        .filter(lambda x: x.unique_id.iloc[0] in _ids)
        .groupby("unique_id")
        .apply(transform_group_monthly)
        .reset_index(drop=True)
    )
    return y_df


def transform_group_monthly(df):
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


(
    spark.createDataFrame(create_m4_monthly())
    .write.format("delta").mode("overwrite")
    .saveAsTable(f"{catalog}.{db}.m4_monthly_train")
)

print(f"Saved data to {catalog}.{db}.m4_monthly_train")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Daily Rossmann with Exogenous Regressors

# COMMAND ----------

# MAGIC %md Download the dataset from [Kaggle](kaggle.com/competitions/rossmann-store-sales/data) and store them in the volume.

# COMMAND ----------

# Randomly select 100 stores to forecast
import random
random.seed(7)

# Number of time series to sample
sample = True
stores = sorted(random.sample(range(0, 1000), n))

train = spark.read.csv(f"/Volumes/{catalog}/{db}/{volume}/train.csv", header=True, inferSchema=True)
test = spark.read.csv(f"/Volumes/{catalog}/{db}/{volume}/test.csv", header=True, inferSchema=True)

if sample:
    train = train.filter(train.Store.isin(stores))
    test = test.filter(test.Store.isin(stores))

train.write.mode("overwrite").option("mergeSchema", "true").saveAsTable(f"{catalog}.{db}.rossmann_daily_train")
test.write.mode("overwrite").option("mergeSchema", "true").saveAsTable(f"{catalog}.{db}.rossmann_daily_test")

print(f"Saved data to {catalog}.{db}.rossmann_daily_train and {catalog}.{db}.rossmann_daily_test")
