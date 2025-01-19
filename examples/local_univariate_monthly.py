# Databricks notebook source
# MAGIC %md
# MAGIC # Many Models Forecasting Demo
# MAGIC
# MAGIC This notebook showcases how to run MMF with local models on multiple univariate time series of monthly resolution. We will use [M4 competition](https://www.sciencedirect.com/science/article/pii/S0169207019301128#sec5) data. The descriptions here are mostly the same as the case with the [daily resolution](https://github.com/databricks-industry-solutions/many-model-forecasting/blob/main/examples/local_univariate_daily.py), so we will skip the redundant parts and focus only on the essentials.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Cluster setup
# MAGIC
# MAGIC We recommend using a cluster with [Databricks Runtime 14.3 LTS for ML](https://docs.databricks.com/en/release-notes/runtime/14.3lts-ml.html) or above. The cluster can be either a single-node or multi-node CPU cluster. Make sure to set the following Spark configurations before you start your cluster: [`spark.sql.execution.arrow.enabled true`](https://spark.apache.org/docs/3.0.1/sql-pyspark-pandas-with-arrow.html#enabling-for-conversion-tofrom-pandas) and [`spark.sql.adaptive.enabled false`](https://spark.apache.org/docs/latest/sql-performance-tuning.html#adaptive-query-execution). You can do this by specifying [Spark configuration](https://docs.databricks.com/en/compute/configure.html#spark-configuration) in the advanced options on the cluster creation page.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Install and import packages

# COMMAND ----------

# DBTITLE 1,Install the necessary libraries
# MAGIC %pip install -r ../requirements.txt --quiet
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import logging
from tqdm.autonotebook import tqdm
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
# MAGIC We are going to save this data in a delta lake table. Provide catalog and database names where you want to store the data.

# COMMAND ----------

catalog = "mmf"  # Name of the catalog we use to manage our assets
db = "m4"  # Name of the schema we use to manage our assets (e.g. datasets)
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

display(
  spark.sql(f"select unique_id, count(date) as count from {catalog}.{db}.m4_monthly_train group by unique_id order by unique_id")
  )

# COMMAND ----------

display(
  spark.sql(f"select * from {catalog}.{db}.m4_monthly_train where unique_id in ('M1', 'M2', 'M3', 'M4', 'M5') order by unique_id, date")
  )

# COMMAND ----------

if n > sc.defaultParallelism:
    sqlContext.setConf("spark.sql.shuffle.partitions", sc.defaultParallelism)

# COMMAND ----------

# MAGIC %md
# MAGIC Note that monthly forecasting requires the timestamp column to represent the last day of each month.

# COMMAND ----------

# MAGIC %md ### Models
# MAGIC Let's configure a list of models we are going to apply to our time series for evaluation and forecasting. A comprehensive list of all supported models is available in [mmf_sa/models/models_conf.yaml](https://github.com/databricks-industry-solutions/many-model-forecasting/blob/main/mmf_sa/models/models_conf.yaml). Look for the models where `model_type: local`; these are the local models we import from [statsforecast](https://github.com/Nixtla/statsforecast), [r fable](https://cran.r-project.org/web/packages/fable/vignettes/fable.html) and [sktime](https://github.com/sktime/sktime). Check their documentations for the description of each model. 

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
    "SKTimeTBats",
    "SKTimeLgbmDsDt",
]

# COMMAND ----------

# MAGIC %md ### Run MMF
# MAGIC
# MAGIC Now, we can run the evaluation and forecasting using `run_forecast` function defined in [mmf_sa/models/__init__.py](https://github.com/databricks-industry-solutions/many-model-forecasting/blob/main/mmf_sa/models/__init__.py). Make sure to set `freq="M"` in `run_forecast` function.

# COMMAND ----------

run_forecast(
    spark=spark,
    train_data=f"{catalog}.{db}.m4_monthly_train",
    scoring_data=f"{catalog}.{db}.m4_monthly_train",
    scoring_output=f"{catalog}.{db}.monthly_scoring_output",
    evaluation_output=f"{catalog}.{db}.monthly_evaluation_output",
    group_id="unique_id",
    date_col="date",
    target="y",
    freq="M",
    prediction_length=3,
    backtest_months=12,
    stride=1,
    metric="smape",
    train_predict_ratio=1,
    data_quality_check=True,
    resample=False,
    active_models=active_models,
    experiment_path=f"/Users/{user}/mmf/m4_monthly",
    use_case_name="m4_monthly",
)

# COMMAND ----------

# MAGIC %md ### Evaluate
# MAGIC In `evaluation_output` table, the we store all evaluation results for all backtesting trials from all models.

# COMMAND ----------

display(
  spark.sql(f"""
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
