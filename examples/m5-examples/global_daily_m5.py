# Databricks notebook source
# DBTITLE 1,Install the necessary libraries
# MAGIC %pip install -r ../../requirements.txt --quiet
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
from mmf_sa import run_forecast

# COMMAND ----------

catalog = "mmf" # Name of the catalog we use to manage our assets
db = "m5" # Name of the schema we use to manage our assets (e.g. datasets)
n = 1000  # Number of items: choose from [1000, 10000, 'full']. full is 35k
table = f"daily_train_{n}" # Training table name
user_email = spark.sql('select current_user() as user').collect()[0]['user'] # User email

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

# The same run_id will be assigned to all the models. This makes it easier to run the post evaluation analysis later.
run_id = str(uuid.uuid4())

for model in active_models:
  dbutils.notebook.run(
    "run_daily_m5",
    timeout_seconds=0, 
    arguments={
      "catalog": catalog, 
      "db": db, 
      "model": model, 
      "run_id": run_id, 
      "table": table , 
      "user_email": user_email,
      }
    )

# COMMAND ----------

display(
  spark.sql(f"select * from {catalog}.{db}.daily_evaluation_output order by unique_id, model, backtest_window_start_date limit 1000")
  )

# COMMAND ----------

display(spark.sql(f"select * from {catalog}.{db}.daily_scoring_output order by unique_id, model, ds limit 1000"))

# COMMAND ----------

display(spark.sql(f"delete from {catalog}.{db}.daily_evaluation_output"))

# COMMAND ----------

display(spark.sql(f"delete from {catalog}.{db}.daily_scoring_output"))

# COMMAND ----------


