# Databricks notebook source
import uuid
import pathlib
import pandas as pd

# COMMAND ----------

catalog = "mmf" # Name of the catalog we use to manage our assets
db = "m5" # Name of the schema we use to manage our assets (e.g. datasets)
n = 100  # Number of items: choose from [100, 1000, 10000, 'full']. full is 35k
table = f"daily_train_{n}" # Training table name
user = spark.sql('select current_user() as user').collect()[0]['user'] # User email

# COMMAND ----------

active_models = [
    "ChronosBoltTiny",
    "ChronosBoltMini",
    "ChronosBoltSmall",
    "ChronosBoltBase",
    "Chronos2",
    "Chronos2Small",
    "Chronos2Synth",
    "TimesFM_2_5_200m",
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
      "user": user,
      }
    )

# COMMAND ----------

display(spark.sql(f"""
                  select * from {catalog}.{db}.daily_evaluation_output 
                  order by unique_id, model, backtest_window_start_date
                  limit 10
                  """))

# COMMAND ----------

display(spark.sql(f"""
                  select * from {catalog}.{db}.daily_scoring_output 
                  order by unique_id, model, ds
                  limit 10
                  """))

# COMMAND ----------

#display(spark.sql(f"delete from {catalog}.{db}.daily_evaluation_output"))

# COMMAND ----------

#display(spark.sql(f"delete from {catalog}.{db}.daily_scoring_output"))
