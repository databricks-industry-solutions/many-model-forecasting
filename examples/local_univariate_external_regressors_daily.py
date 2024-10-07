# Databricks notebook source
# MAGIC %md
# MAGIC # Many Models Forecasting Demo
# MAGIC
# MAGIC This notebook showcases how to run MMF with local models on multiple time series of daily resolution using exogenous regressors. We will use [Rossmann Store](https://www.kaggle.com/competitions/rossmann-store-sales/data) data. To be able to run this notebook, you need to register on [Kaggle](https://www.kaggle.com/) and download the dataset. The descriptions here are mostly the same as the case [without exogenous regressors](https://github.com/databricks-industry-solutions/many-model-forecasting/blob/main/examples/local_univariate_daily.py), so we will skip the redundant parts and focus only on the essentials. 

# COMMAND ----------

# MAGIC %md
# MAGIC ### Cluster setup
# MAGIC
# MAGIC We recommend using a cluster with [Databricks Runtime 14.3 LTS for ML](https://docs.databricks.com/en/release-notes/runtime/14.3lts-ml.html) or above. The cluster can be either a single-node or multi-node CPU cluster. Make sure to set the following Spark configurations before you start your cluster: [`spark.sql.execution.arrow.enabled true`](https://spark.apache.org/docs/3.0.1/sql-pyspark-pandas-with-arrow.html#enabling-for-conversion-tofrom-pandas) and [`spark.sql.adaptive.enabled false`](https://spark.apache.org/docs/latest/sql-performance-tuning.html#adaptive-query-execution). You can do this by specifying [Spark configuration](https://docs.databricks.com/en/compute/configure.html#spark-configuration) in the advanced options on the cluster creation page.

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

import pandas as pd
from mmf_sa import run_forecast

# COMMAND ----------

# MAGIC %md
# MAGIC ### Prepare data 
# MAGIC Before running this notebook, download the dataset from [Kaggle](https://www.kaggle.com/competitions/rossmann-store-sales/data) and store them in Unity Catalog as a [volume](https://docs.databricks.com/en/connect/unity-catalog/volumes.html).

# COMMAND ----------

catalog = "mmf" # Name of the catalog we use to manage our assets
db = "rossmann" # Name of the schema we use to manage our assets (e.g. datasets)
volume = "csv" # Name of the volume where you have your rossmann dataset csv stored
user = spark.sql('select current_user() as user').collect()[0]['user'] # User email address

# COMMAND ----------

# Make sure that the catalog and the schema exist
_ = spark.sql(f"CREATE CATALOG IF NOT EXISTS {catalog}")
_ = spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{db}")
_ = spark.sql(f"CREATE VOLUME IF NOT EXISTS {catalog}.{db}.{volume}")

# COMMAND ----------

# Randomly select 100 stores to forecast
import random
random.seed(7)

# Number of time series to sample
sample = True
size = 100
stores = sorted(random.sample(range(0, 1000), size))

train = spark.read.csv(f"/Volumes/{catalog}/{db}/{volume}/train.csv", header=True, inferSchema=True)
test = spark.read.csv(f"/Volumes/{catalog}/{db}/{volume}/test.csv", header=True, inferSchema=True)

if sample:
    train = train.filter(train.Store.isin(stores))
    test = test.filter(test.Store.isin(stores))

# COMMAND ----------

# MAGIC %md
# MAGIC We are going to save this data in a delta lake table. Provide catalog and database names where you want to store the data.

# COMMAND ----------

train.write.mode("overwrite").option("mergeSchema", "true").saveAsTable(f"{catalog}.{db}.rossmann_daily_train")
test.write.mode("overwrite").option("mergeSchema", "true").saveAsTable(f"{catalog}.{db}.rossmann_daily_test")

# COMMAND ----------

# MAGIC %md Let's take a peak at the dataset:

# COMMAND ----------

display(spark.sql(f"select * from {catalog}.{db}.rossmann_daily_train where Store=49 order by Date"))
display(spark.sql(f"select * from {catalog}.{db}.rossmann_daily_test where Store=49 order by Date"))

# COMMAND ----------

# MAGIC %md
# MAGIC Note that in `rossmann_daily_train` we have our target variable `Sales` but not in `rossmann_daily_test`. This is because `rossmann_daily_test` is going to be used as our `scoring_data` that stores `dynamic_future` variables of the future dates. When you adapt this notebook to your use case, make sure to comply with these datasets formats. See statsforecast's [documentation](https://nixtlaverse.nixtla.io/statsforecast/docs/how-to-guides/exogenous.html) for more detail on exogenous regressors.

# COMMAND ----------

if sample and size > sc.defaultParallelism:
    sqlContext.setConf("spark.sql.shuffle.partitions", sc.defaultParallelism)

# COMMAND ----------

# MAGIC %md ### Models
# MAGIC Let's configure a list of models we are going to apply to our time series for evaluation and forecasting. A comprehensive list of all supported models is available in [mmf_sa/models/models_conf.yaml](https://github.com/databricks-industry-solutions/many-model-forecasting/blob/main/mmf_sa/models/models_conf.yaml). Look for the models where `model_type: local`; these are the local models we import from [statsforecast](https://github.com/Nixtla/statsforecast), [r fable](https://cran.r-project.org/web/packages/fable/vignettes/fable.html) and [sktime](https://github.com/sktime/sktime). Check their documentations for the description of each model. 
# MAGIC
# MAGIC Exogenous regressors are currently only supported for [some models](https://nixtlaverse.nixtla.io/statsforecast/index.html#models) from statsforecast (e.g. `StatsForecastAutoArima`). But including non-supported models in the active model list doesn't harm: models that can't use exogenous regressors will simply ignore them.

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
# MAGIC Now, we run the evaluation and forecasting using `run_forecast` function. We are providing the training table and the scoring table names. If `scoring_data` is not provided or if the same name as `train_data` is provided, the models will ignore the `dynamic_future` regressors. Note that we are providing a covariate field (i.e. `dynamic_future`) this time. There are also other convariate fields, namely `static_features`, and `dynamic_historical`, but these are only relevant with the global models. 

# COMMAND ----------

run_forecast(
    spark=spark,
    train_data=f"{catalog}.{db}.rossmann_daily_train",
    scoring_data=f"{catalog}.{db}.rossmann_daily_test",
    scoring_output=f"{catalog}.{db}.rossmann_daily_scoring_output",
    evaluation_output=f"{catalog}.{db}.rossmann_daily_evaluation_output",
    group_id="Store",
    date_col="Date",
    target="Sales",
    freq="D",
    dynamic_future=["DayOfWeek", "Open", "Promo", "SchoolHoliday"],
    prediction_length=10,
    backtest_months=1,
    stride=10,
    metric="smape",
    train_predict_ratio=1,
    active_models=active_models,
    data_quality_check=False,
    resample=False,
    experiment_path=f"/Users/{user}/mmf/rossmann_daily",
    use_case_name="rossmann_daily",
)

# COMMAND ----------

# MAGIC %md ### Evaluate
# MAGIC In `evaluation_output` table, the we store all evaluation results for all backtesting trials from all models.

# COMMAND ----------

display(
  spark.sql(f"select * from {catalog}.{db}.rossmann_daily_evaluation_output order by Store, model, backtest_window_start_date")
  )

# COMMAND ----------

# MAGIC %md ### Forecast
# MAGIC In `scoring_output` table, forecasts for each time series from each model are stored.

# COMMAND ----------

display(spark.sql(f"select * from {catalog}.{db}.rossmann_daily_scoring_output order by Store, model"))

# COMMAND ----------

# MAGIC %md ### Delete Tables
# MAGIC Let's clean up the tables.

# COMMAND ----------

#display(spark.sql(f"delete from {catalog}.{db}.rossmann_daily_evaluation_output"))

# COMMAND ----------

#display(spark.sql(f"delete from {catalog}.{db}.rossmann_daily_scoring_output"))

# COMMAND ----------


