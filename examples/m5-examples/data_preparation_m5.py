# Databricks notebook source
# MAGIC %pip install datasetsforecast==0.0.8 --quiet
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import pathlib
import pandas as pd
from datasetsforecast.m5 import M5
import logging
logger = spark._jvm.org.apache.log4j
logging.getLogger("py4j.java_gateway").setLevel(logging.ERROR)
logging.getLogger("py4j.clientserver").setLevel(logging.ERROR)

# COMMAND ----------

catalog = "mmf"
db = "m5"

# COMMAND ----------

_ = spark.sql(f"CREATE CATALOG IF NOT EXISTS {catalog}")
_ = spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{db}")

# COMMAND ----------

df_target, df_exogenous, static_features = M5.load(directory=str(pathlib.Path.home()))
daily_train = pd.merge(df_target, df_exogenous, on=['unique_id','ds'], how='inner')

# COMMAND ----------

# MAGIC %md
# MAGIC #### Write out the entire dataset to a delta table

# COMMAND ----------

(
    spark.createDataFrame(daily_train)
    .write.format("delta").mode("overwrite")
    .saveAsTable(f"{catalog}.{db}.daily_train")
)
print(f"Saved data to {catalog}.{db}.daily_train_full")

# COMMAND ----------

# MAGIC %md
# MAGIC #### Write out the sampled dataset to a delta table

# COMMAND ----------

import random
random.seed(7)

unique_ids = list(daily_train["unique_id"].unique())
unique_id_1000 = sorted(random.sample(unique_ids, 1000))
unique_id_10000 = sorted(random.sample(unique_ids, 10000))

daily_train_1000 = daily_train[daily_train["unique_id"].isin(unique_id_1000)]
daily_train_10000 = daily_train[daily_train["unique_id"].isin(unique_id_10000)]

# COMMAND ----------

(
    spark.createDataFrame(daily_train_1000)
    .write.format("delta").mode("overwrite")
    .saveAsTable(f"{catalog}.{db}.daily_train_1000")
)
print(f"Saved data to {catalog}.{db}.daily_train_1000")

# COMMAND ----------

(
    spark.createDataFrame(daily_train_10000)
    .write.format("delta").mode("overwrite")
    .saveAsTable(f"{catalog}.{db}.daily_train_10000")
)
print(f"Saved data to {catalog}.{db}.daily_train_10000")

# COMMAND ----------

display(spark.sql(f"select * from {catalog}.{db}.daily_train_1000"))

# COMMAND ----------


