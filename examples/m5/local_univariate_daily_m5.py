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

import pathlib
import pandas as pd
from mmf_sa import run_forecast

# COMMAND ----------

catalog = "mmf" # Name of the catalog we use to manage our assets
db = "m5" # Name of the schema we use to manage our assets (e.g. datasets)
user = spark.sql('select current_user() as user').collect()[0]['user'] # User email address

n = 100  # Number of items: choose from [1000, 10000, 'full']. full is 35k
taining_table = f"daily_train_{n}"

# COMMAND ----------

display(
  spark.sql(f"""
            select * from {catalog}.{db}.{taining_table} 
            where unique_id in ('FOODS_1_001_WI_1', 'FOODS_1_004_TX_2', 'FOODS_1_006_WI_1', 'FOODS_1_008_CA_3', 'FOODS_1_012_WI_1') 
            order by unique_id, ds
            """))

# COMMAND ----------

active_models = [
    "StatsForecastBaselineWindowAverage",
    "StatsForecastBaselineSeasonalWindowAverage",
    "StatsForecastBaselineNaive",
    "StatsForecastBaselineSeasonalNaive",
    "StatsForecastAutoArima",
    "StatsForecastAutoETS",
    "StatsForecastAutoCES",
    "StatsForecastAutoTheta",
    "StatsForecastAutoTbats",
    "StatsForecastAutoMfles",
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
    "SKTimeTBats",
    "SKTimeProphet",
    "SKTimeLgbmDsDt",
]

# COMMAND ----------

if n > sc.defaultParallelism:
    sqlContext.setConf("spark.sql.shuffle.partitions", sc.defaultParallelism)

# COMMAND ----------

run_forecast(
    spark=spark,
    train_data=f"{catalog}.{db}.{taining_table}",
    scoring_data=f"{catalog}.{db}.{taining_table}",
    scoring_output=f"{catalog}.{db}.daily_scoring_output",
    evaluation_output=f"{catalog}.{db}.daily_evaluation_output",
    group_id="unique_id",
    date_col="ds",
    target="y",
    freq="D",
    prediction_length=28,
    backtest_length=90,
    stride=7,
    metric="smape",
    train_predict_ratio=1,
    data_quality_check=True,
    resample=False,
    active_models=active_models,
    experiment_path=f"/Users/{user}/mmf/m5_daily",
    use_case_name="m5_daily",
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
