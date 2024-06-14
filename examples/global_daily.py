# Databricks notebook source
# MAGIC %md
# MAGIC # Many Models Forecasting Demo
# MAGIC This notebook showcases how to run MMF with global models on multiple time series of daily resolution. We will use [M4 competition](https://www.sciencedirect.com/science/article/pii/S0169207019301128#sec5) data.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Cluster setup
# MAGIC
# MAGIC We recommend using a cluster with [Databricks Runtime 14.3 LTS for ML](https://docs.databricks.com/en/release-notes/runtime/14.3lts-ml.html) or above. The cluster should be single-node with one or more GPU instances: e.g. [g4dn.12xlarge [T4]](https://aws.amazon.com/ec2/instance-types/g4/) on AWS or [Standard_NC64as_T4_v3](https://learn.microsoft.com/en-us/azure/virtual-machines/nct4-v3-series) on Azure. MMF leverages [neuralforecast](https://nixtlaverse.nixtla.io/neuralforecast/index.html) which is built on top of [pytorch](https://lightning.ai/docs/pytorch/stable/common/trainer.html) and can therefore utilize all the [available resources](https://lightning.ai/docs/pytorch/stable/common/trainer.html). 

# COMMAND ----------

# MAGIC %md
# MAGIC ### Install and import packages
# MAGIC Check out [requirements.txt](https://github.com/databricks-industry-solutions/many-model-forecasting/blob/main/requirements.txt) if you're interested in the libraries we use.

# COMMAND ----------

# DBTITLE 1,Install the necessary libraries
# MAGIC %pip install -r ../requirements.txt --quiet
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
from mmf_sa import run_forecast

# COMMAND ----------

# MAGIC %md
# MAGIC ### Prepare data 
# MAGIC We are using [`datasetsforecast`](https://github.com/Nixtla/datasetsforecast/tree/main/) package to download M4 data. M4 dataset contains a set of time series which we use for testing MMF. Below we have written a number of custome functions to convert M4 time series to an expected format.

# COMMAND ----------

# Number of time series
n = 100


def create_m4_daily():
    y_df, _, _ = M4.load(directory=str(pathlib.Path.home()), group="Daily")
    _ids = [f"D{i}" for i in range(1, n+1)]
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
    if len(df) > 1020:
        df = df.iloc[-1020:]
    _start = pd.Timestamp("2020-01-01")
    _end = _start + pd.DateOffset(days=int(df.count()[0]) - 1)
    date_idx = pd.date_range(start=_start, end=_end, freq="D", name="ds")
    res_df = pd.DataFrame(data=[], index=date_idx).reset_index()
    res_df["unique_id"] = unique_id
    res_df["y"] = df.y.values
    return res_df


# COMMAND ----------

# MAGIC %md
# MAGIC We are going to save this data in a delta lake table. Provide catalog and database names where you want to store the data.

# COMMAND ----------

catalog = "mmf"  # Name of the catalog we use to manage our assets
db = "m4"  # Name of the schema we use to manage our assets (e.g. datasets)
user = spark.sql('select current_user() as user').collect()[0]['user']  # User email address

# COMMAND ----------

# Making sure that the catalog and the schema exist
_ = spark.sql(f"CREATE CATALOG IF NOT EXISTS {catalog}")
_ = spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{db}")

(
    spark.createDataFrame(create_m4_daily())
    .write.format("delta").mode("overwrite")
    .saveAsTable(f"{catalog}.{db}.m4_daily_train")
)

# COMMAND ----------

# MAGIC %md Let's take a peak at the dataset:

# COMMAND ----------

display(
  spark.sql(f"select * from {catalog}.{db}.m4_daily_train where unique_id in ('D1', 'D2', 'D3', 'D4', 'D5') order by unique_id, ds")
  )

# COMMAND ----------

# MAGIC %md ### Models
# MAGIC Let's configure a list of models we are going to apply to our time series for evaluation and forecasting. A comprehensive list of all supported models is available in [mmf_sa/models/models_conf.yaml](https://github.com/databricks-industry-solutions/many-model-forecasting/blob/main/mmf_sa/models/models_conf.yaml). Look for the models where `model_type: global`; these are the global models we import from [neuralforecast](https://github.com/Nixtla/neuralforecast). Check their documentation for the detailed description of each model. 
# MAGIC
# MAGIC Some of these models perform [hyperparameter optimization](https://nixtlaverse.nixtla.io/neuralforecast/examples/automatic_hyperparameter_tuning.html) on its own to search for the best parameters. You can specify the range of the search or fix the values in [mmf_sa/models/models_conf.yaml](https://github.com/databricks-industry-solutions/many-model-forecasting/blob/main/mmf_sa/models/models_conf.yaml) or overwrite the default values in [mmf_sa/forecasting_conf.yaml](https://github.com/databricks-industry-solutions/many-model-forecasting/blob/main/mmf_sa/forecasting_conf.yaml). You can also introduce new hyperparameters that are supported by the base models. To do this, first add those hyperparameters under the model specification in [mmf_sa/models/models_conf.yaml](https://github.com/databricks-industry-solutions/many-model-forecasting/blob/main/mmf_sa/models/models_conf.yaml). Then, include these hyperparameters inside the model instantiation which happens in the model pipeline script: e.g. `NeuralFcAutoNBEATSx` class in [mmf_sa/models/neuralforecast/NeuralForecastPipeline.py](https://github.com/databricks-industry-solutions/many-model-forecasting/blob/main/mmf_sa/models/neuralforecast/NeuralForecastPipeline.py).

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
# MAGIC Now, we can run the evaluation and forecasting using `run_forecast` function defined in [mmf_sa/models/__init__.py](https://github.com/databricks-industry-solutions/many-model-forecasting/blob/main/mmf_sa/models/__init__.py). Refer to [README.md](https://github.com/databricks-industry-solutions/many-model-forecasting/blob/main/README.md#parameters-description) for a comprehensive description of each parameter. 
# MAGIC
# MAGIC Note that we are not providing any covariate field (i.e. `static_features`, `dynamic_future` or `dynamic_historical`) yet in this example. We will look into how we can add exogenous regressors to help our models in a different notebook: [examples/global_external_regressors_daily.py](https://github.com/databricks-industry-solutions/many-model-forecasting/blob/main/examples/global_external_regressors_daily.py).
# MAGIC
# MAGIC While the following cell is running, you can check the status of your run on Experiments. Make sure you look for the experiments with the path you provided as `experiment_path` within `run_forecast`. On the Experiments page, you see one entry per one model (i.e. NeuralForecastAutoNBEATSx). The metric provided here is a simple average over all back testing trials and all time series. This is intended to give you an initial feeling of how good each model performs on your entire data mix. But we will look into how you can scrutinize the evaluation using the `evaluation_output` table in a bit. 
# MAGIC
# MAGIC If you are interested in how MMF achieves distributed training and inference, have a look at the two methods `evaluate_global_model` and `evaluate_global_model` defined in the source code [`Forecaster.py`](https://github.com/databricks-industry-solutions/many-model-forecasting/blob/main/mmf_sa/Forecaster.py).
# MAGIC
# MAGIC A small difference here in running `run_forecast` from the local model case is that we have to iterate through the `active_models` and  call the function in a [separate notebook](https://github.com/databricks-industry-solutions/many-model-forecasting/blob/main/examples/run_daily.py). This is to avoid the CUDA out of memory issue by freeing up the GPU memory after each model. Make sure to provide `accelerator="gpu"` as an input parameter to `run_forecast` function. Also, set the parameter `data_quality_check=True` or provide a complete dataset without missing entries to avoid issues with skipped dates.

# COMMAND ----------

# The same run_id will be assigned to all the models. This makes it easier to run the post evaluation analysis later.
run_id = str(uuid.uuid4())

for model in active_models:
  dbutils.notebook.run(
    "run_daily",
    timeout_seconds=0, 
    arguments={"catalog": catalog, "db": db, "model": model, "run_id": run_id, "user": user})

# COMMAND ----------

# MAGIC %md ### Evaluate
# MAGIC In `evaluation_output` table, the we store all evaluation results for all backtesting trials from all models. This information can be used to understand which models performed well on which time series on which periods of backtesting. This is very important for selecting the final model for forecasting or models for ensembling. Maybe, it's faster to take a look at the table:

# COMMAND ----------

display(spark.sql(f"select * from {catalog}.{db}.daily_evaluation_output order by unique_id, model, backtest_window_start_date"))

# COMMAND ----------

# MAGIC %md
# MAGIC For global models, we train the model once using the training dataset excluding `backtest_months`. We then use the same fitted model to produce the as-if forecasts for all back testing periods. We do this to make sure that there is no data leakage. See how MMF implements backtesting in `backtest` method in [mmf_sa/models/abstract_model.py](https://github.com/databricks-industry-solutions/many-model-forecasting/blob/main/mmf_sa/models/abstract_model.py). But the model that gets registered in Unity Catalog is trained using the full history.
# MAGIC
# MAGIC We store the as-if forecasts together with the actuals for each backtesting period, so you can construct any metric of your interest. We provide a few out-of-the-box metrics for you (e.g. smape), but the idea here is that you construct your own metrics reflecting your business requirements and evaluate models based on those. For example, maybe you care more about the accuracy of the near-horizon forecasts than the far-horizon ones. In such case, you can apply a decreasing wieght to compute weighted aggregated metrics.
# MAGIC
# MAGIC Note that if you run local and/or global models against the same time series with the same input parameters (except for those specifying global and foundation models), you will get the entries from those models in the same table and will be able to compare across all types models, which is the biggest benefit of having all models integrated in one solution.
# MAGIC
# MAGIC We also register the model in Unity Catalog and store each model's URI in this table (`model_uri`). You can use MLFlow to [load the models](https://mlflow.org/docs/latest/python_api/mlflow.sklearn.html#mlflow.sklearn.load_model) and access their specifications or produce forecasts. 

# COMMAND ----------

# MAGIC %md ### Forecast
# MAGIC In `scoring_output` table, forecasts for each time series from each model are stored. Based on the evaluation exercised performed on `evaluation_output` table, you can select the forecasts from the best performing models or a mix of models. We again store each model's URI in this table (`model_uri`). You can use MLFlow to [load the models](https://mlflow.org/docs/latest/python_api/mlflow.sklearn.html#mlflow.sklearn.load_model) and access their specifications or produce forecasts. 

# COMMAND ----------

display(spark.sql(f"select * from {catalog}.{db}.daily_scoring_output order by unique_id, model, ds"))

# COMMAND ----------

# MAGIC %md ### Delete Tables
# MAGIC Let's clean up the tables.

# COMMAND ----------

display(spark.sql(f"delete from {catalog}.{db}.daily_evaluation_output"))

# COMMAND ----------

display(spark.sql(f"delete from {catalog}.{db}.daily_scoring_output"))
