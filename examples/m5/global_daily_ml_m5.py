# Databricks notebook source
# MAGIC %md
# MAGIC # Many Models Forecasting Demo — M5 Global ML Models (MLForecast + LightGBM)
# MAGIC
# MAGIC This notebook runs MMF with CPU-based global models (`MLForecastLGBM`, `MLForecastAutoLGBM`) on the [M5 competition](https://www.kaggle.com/competitions/m5-forecasting-accuracy) daily panel. M5 includes future-known regressors (`sell_price`, `snap_CA`, `snap_TX`, `snap_WI`) which we feed to the models.
# MAGIC
# MAGIC Run [`data_preparation_m5.py`](https://github.com/databricks-industry-solutions/many-model-forecasting/blob/main/examples/m5/data_preparation_m5.py) first to materialize `mmf.m5.daily_train_<n>`.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Cluster setup
# MAGIC
# MAGIC We recommend a **single-node CPU cluster** with [Databricks Runtime 17.3 LTS for ML](https://docs.databricks.com/en/release-notes/runtime/17.3lts-ml.html). The pinned versions in [`requirements-global.txt`](https://github.com/databricks-industry-solutions/many-model-forecasting/blob/main/requirements-global.txt) (`mlforecast==1.0.31`, `lightgbm==4.6.0`, `optuna==3.6.1`) match the DBR ML preinstalled versions exactly.

# COMMAND ----------

# MAGIC %pip install -r ../requirements-global.txt --quiet
# MAGIC %pip install datasetsforecast==0.0.8 --quiet
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import logging
import pathlib
import pandas as pd
from mmf_sa import run_forecast

logging.getLogger("py4j.clientserver").setLevel(logging.WARNING)
logging.getLogger("py4j.java_gateway").setLevel(logging.WARNING)

# COMMAND ----------

catalog = "mmf"  # Name of the catalog we use to manage our assets
db = "m5"  # Name of the schema we use to manage our assets (e.g. datasets)
n = 100  # Number of items: choose from [100, 1000, 10000, 'full']. full is 35k
table = f"daily_train_{n}"  # Training table name (created by data_preparation_m5.py)
user = spark.sql('select current_user() as user').collect()[0]['user']  # User email

# COMMAND ----------

# MAGIC %md
# MAGIC ### Models
# MAGIC
# MAGIC `MLForecastLGBM` runs with fixed hyperparameters and `MLForecastAutoLGBM` performs joint LightGBM + feature-pipeline HPO via Optuna. Both models consume the daily-frequency search-space defaults from [`mmf_sa/forecasting_conf_daily.yaml`](https://github.com/databricks-industry-solutions/many-model-forecasting/blob/main/mmf_sa/forecasting_conf_daily.yaml).

# COMMAND ----------

active_models = [
    "MLForecastLGBM",
    "MLForecastAutoLGBM",
]

# COMMAND ----------

# MAGIC %md
# MAGIC ### Run MMF
# MAGIC
# MAGIC We pass M5's numeric future-known regressors via `dynamic_future_numerical` (`sell_price`) and `dynamic_future_categorical` (`snap_CA`, `snap_TX`, `snap_WI`). String columns (`event_name_*`, `event_type_*`) are excluded because LightGBM does not consume strings without explicit categorical encoding.
# MAGIC
# MAGIC We set `scoring_data=None` so MMF only runs the evaluation phase. M5's `train_data == scoring_data` pattern combined with future regressors causes the scoring path to fail (no future-regressor rows beyond the training window); the evaluation path handles future regressors correctly via its rolling-origin backtest.

# COMMAND ----------

run_forecast(
    spark=spark,
    train_data=f"{catalog}.{db}.{table}",
    evaluation_output=f"{catalog}.{db}.daily_evaluation_output",
    model_output=f"{catalog}.{db}",
    group_id="unique_id",
    date_col="ds",
    target="y",
    freq="D",
    prediction_length=28,
    backtest_length=90,
    stride=7,
    metric="smape",
    dynamic_future_numerical=["sell_price"],
    dynamic_future_categorical=["snap_CA", "snap_TX", "snap_WI"],
    train_predict_ratio=1,
    data_quality_check=True,
    resample=False,
    active_models=active_models,
    experiment_path=f"/Users/{user}/mmf/m5_daily",
    use_case_name="m5_daily",
    accelerator="cpu",
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Evaluate
# MAGIC The `evaluation_output` table aggregates per-series, per-backtest-window metrics across all models you've run with `use_case_name="m5_daily"`.

# COMMAND ----------

display(
    spark.sql(f"""
        select * from {catalog}.{db}.daily_evaluation_output
        where model in ('MLForecastLGBM', 'MLForecastAutoLGBM')
        order by unique_id, model, backtest_window_start_date
        limit 20
        """)
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Delete Tables
# MAGIC Optional cleanup.

# COMMAND ----------

#display(spark.sql(f"delete from {catalog}.{db}.daily_evaluation_output where model in ('MLForecastLGBM', 'MLForecastAutoLGBM')"))
