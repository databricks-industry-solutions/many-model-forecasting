# Databricks notebook source
# MAGIC %md
# MAGIC # Many Models Forecasting (MMF)
# MAGIC This notebook showcases how to run MMF with local models on multiple univariate time series of daily resolution. We will use [M4 competition](https://www.sciencedirect.com/science/article/pii/S0169207019301128#sec5) data.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Cluster setup
# MAGIC
# MAGIC We recommend using a cluster with [Databricks Runtime 14.3 LTS for ML](https://docs.databricks.com/en/release-notes/runtime/14.3lts-ml.html) or above. The cluster can be either a single-node or multi-node CPU cluster. MMF leverages [Pandas UDF](https://docs.databricks.com/en/udf/pandas.html) under the hood and utilizes all the available resource. Make sure to set the following Spark configurations before you start your cluster: [`spark.sql.execution.arrow.enabled true`](https://spark.apache.org/docs/3.0.1/sql-pyspark-pandas-with-arrow.html#enabling-for-conversion-tofrom-pandas) and [`spark.sql.adaptive.enabled false`](https://spark.apache.org/docs/latest/sql-performance-tuning.html#adaptive-query-execution). You can do this by specifying [Spark configuration](https://docs.databricks.com/en/compute/configure.html#spark-configuration) in the advanced options on the cluster creation page.

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

import pathlib
import pandas as pd
from datasetsforecast.m4 import M4
from mmf_sa import run_forecast

# COMMAND ----------

# MAGIC %md
# MAGIC #### Install R packages
# MAGIC If you want to use the R fable models, you need to [install the R dependecies](https://docs.databricks.com/en/libraries/index.html#r-library-support). See [RUNME.py](https://github.com/databricks-industry-solutions/many-model-forecasting/blob/main/RUNME.py) for the full list of required R libraries and their versions.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Prepare data 
# MAGIC We are using [`datasetsforecast`](https://github.com/Nixtla/datasetsforecast/tree/main/) package to download M4 data. M4 dataset contains a set of time series which we use for testing MMF. Below we have written a number of custome functions to convert M4 time series to an expected format.

# COMMAND ----------

# Number of time series
n = 1000


def create_m4_daily():
    y_df, _, _ = M4.load(directory=str(pathlib.Path.home()), group="Daily")
    _ids = [f"D{i}" for i in range(1, n)]
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

catalog = "mmf" # Name of the catalog we use to manage our assets
db = "m4" # Name of the schema we use to manage our assets (e.g. datasets)
user = spark.sql('select current_user() as user').collect()[0]['user'] # User email address

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

# MAGIC %md
# MAGIC If the number of time series is larger than the number of total cores, we set `spark.sql.shuffle.partitions` to the number of cores (can also be a multiple) so that we don't under-utilize the resource.

# COMMAND ----------

if n > sc.defaultParallelism:
    sqlContext.setConf("spark.sql.shuffle.partitions", sc.defaultParallelism)

# COMMAND ----------

# MAGIC %md ### Models
# MAGIC Let's configure a list of models we are going to apply to our time series for evaluation and forecasting. A comprehensive list of all supported models is available in [mmf_sa/models/models_conf.yaml](https://github.com/databricks-industry-solutions/many-model-forecasting/blob/main/mmf_sa/models/models_conf.yaml). Look for the models where `model_type: local`; these are the local models we import from [statsforecast](https://github.com/Nixtla/statsforecast), [r fable](https://cran.r-project.org/web/packages/fable/vignettes/fable.html) and [sktime](https://github.com/sktime/sktime). Check their documentations for the detailed description of each model. 
# MAGIC
# MAGIC Some of these models perform hyperparameter optimization ([statsforecast Automatic Forecasting](https://nixtlaverse.nixtla.io/statsforecast/index.html#automatic-forecasting)) on its own for some hyperparameters. For other hyperparameters or models, you can modify the hyperparameters in [mmf_sa/models/models_conf.yaml](https://github.com/databricks-industry-solutions/many-model-forecasting/blob/main/mmf_sa/models/models_conf.yaml) or overwrite the default values in [mmf_sa/forecasting_conf.yaml](https://github.com/databricks-industry-solutions/many-model-forecasting/blob/main/mmf_sa/forecasting_conf.yaml). You can also introduce new hyperparameters that are supported by the base models. To do this, first add those hyperparameters under the model specification in [mmf_sa/models/models_conf.yaml](https://github.com/databricks-industry-solutions/many-model-forecasting/blob/main/mmf_sa/models/models_conf.yaml). Then, include these hyperparameters inside the model instantiation which happens in the model pipeline script: e.g. `StatsFcAutoArima` class in [mmf_sa/models/statsforecast/StatsFcForecastingPipeline.py](https://github.com/databricks-industry-solutions/many-model-forecasting/blob/main/mmf_sa/models/statsforecast/StatsFcForecastingPipeline.py).

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
    #"SKTimeTBats",
    #"SKTimeLgbmDsDt",
    ]

# COMMAND ----------

# MAGIC %md ### Run MMF
# MAGIC
# MAGIC Now, we can run the evaluation and forecasting using `run_forecast` function defined in [mmf_sa/models/__init__.py](https://github.com/databricks-industry-solutions/many-model-forecasting/blob/main/mmf_sa/models/__init__.py).
# MAGIC
# MAGIC Refer to [README.md](https://github.com/databricks-industry-solutions/many-model-forecasting/blob/main/README.md#parameters-description) for a comprehensive explanation of each parameter. Note that we are not providing any covariate field (i.e. `static_features`, `dynamic_future` or `dynamic_historical`) yet in this example. We will look into how we can add exogenous regressors to help our models in a different notebook: [examples/local_univariate_external_regressors_daily.py](https://github.com/databricks-industry-solutions/many-model-forecasting/blob/main/examples/local_univariate_external_regressors_daily.py).
# MAGIC
# MAGIC While the following cell is running, you can check the status of your run on Experiments. Make sure you look for the experiments with the path you provided as `experiment_path` within `run_forecast`. On the Experiments page, you see one entry per one model (i.e. StatsForecastAutoArima). The metric provided here is a simple average over all back testing trials and all time series. This is intended to give you an initial feeling of how good each model performs on your entire data mix. But we will look into how you can scrutinize the evaluation using the `evaluation_output` table in a bit. 
# MAGIC
# MAGIC If you are interested in how Pandas UDF achieves parallel fitting and forecasting of multiple time series by distributing them across multiple executors, have a look at the two methods `evaluate_local_model` and `score_local_model` defined in the source code [`Forecaster.py`](https://github.com/databricks-industry-solutions/many-model-forecasting/blob/main/mmf_sa/Forecaster.py).

# COMMAND ----------

run_forecast(
    spark=spark,
    train_data=f"{catalog}.{db}.m4_daily_train",
    scoring_data=f"{catalog}.{db}.m4_daily_train",
    scoring_output=f"{catalog}.{db}.daily_scoring_output",
    evaluation_output=f"{catalog}.{db}.daily_evaluation_output",
    group_id="unique_id",
    date_col="ds",
    target="y",
    freq="D",
    prediction_length=10,
    backtest_months=1,
    stride=10,
    metric="smape",
    train_predict_ratio=1,
    data_quality_check=True,
    resample=False,
    active_models=active_models,
    experiment_path=f"/Users/{user}/mmf/m4_daily",
    use_case_name="m4_daily",
)

# COMMAND ----------

# MAGIC %md ### Evaluate
# MAGIC In `evaluation_output` table, the we store all evaluation results for all backtesting trials from all models. This information can be used to understand which models performed well on which time series on which periods of backtesting. This is very important for selecting the final model for forecasting or models for ensembling. Maybe, it's faster to take a look at the table:

# COMMAND ----------

display(
  spark.sql(f"""
    select * from {catalog}.{db}.daily_evaluation_output 
    where unique_id = 'D1'
    order by unique_id, model, backtest_window_start_date
    """))

# COMMAND ----------

# MAGIC %md
# MAGIC For each backtesting trial, we train the model on the training dataset only up to `backtesting_window_start_date`. We then use that fitted model to generate the forecasts for that specific horizon. We then move `backtesting_window_start_date` forward by `stride` and do the same exercise. This continues until `backtesting_window_start_date` reaches the last day of the time series given in the training dataset. See how MMF implements backtesting in `backtest` method in [mmf_sa/models/abstract_model.py](https://github.com/databricks-industry-solutions/many-model-forecasting/blob/main/mmf_sa/models/abstract_model.py).
# MAGIC > 
# MAGIC We store the **as-if** forecasts together with the actuals for each backtesting period, so you can construct any metric of your interest. We provide a few out-of-the-box metrics for you (e.g. smape), but the idea here is that you construct your own metrics reflecting your business requirements and evaluate models based on those. For example, maybe you care more about the accuracy of the near-horizon forecasts than the far-horizon ones. In such case, you can apply a decreasing wieght to compute weighted aggregated metrics.
# MAGIC
# MAGIC Note that if you run other global and foundation models against the same time series with the same input parameters (except for those specifying global and foundation models), you will get the entries from those models in the same table and will be able to compare across all types models, which is the biggest benefit of having all models integrated in one solution.
# MAGIC
# MAGIC We also store each model in a binary format in this table (`model_pickle`). You can unpickle the models and access their specifications or produce forecasts. 

# COMMAND ----------

# MAGIC %md ### Forecast
# MAGIC In `scoring_output` table, forecasts for each time series from each model are stored. Based on the evaluation exercised performed on `evaluation_output` table, you can select the forecasts from the best performing models or a mix of models. We are again storing each model in a binary format in this table.

# COMMAND ----------

display(spark.sql(f"""
                  select * from {catalog}.{db}.daily_scoring_output 
                  where unique_id = 'D1'
                  order by unique_id, model, ds
                  """))

# COMMAND ----------

# MAGIC %md ### Delete Tables
# MAGIC Let's clean up the tables.

# COMMAND ----------

#display(spark.sql(f"delete from {catalog}.{db}.daily_evaluation_output"))

# COMMAND ----------

#display(spark.sql(f"delete from {catalog}.{db}.daily_scoring_output"))
