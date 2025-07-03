# Databricks notebook source
# MAGIC %md
# MAGIC # Many Models Forecasting Demo
# MAGIC
# MAGIC This notebook showcases how to run MMF with global models on multiple time series of daily resolution using exogenous regressors. We will use [Rossmann Store](https://www.kaggle.com/competitions/rossmann-store-sales/data) data. To be able to run this notebook, you need to register on [Kaggle](https://www.kaggle.com/) and download the dataset. The descriptions here are mostly the same as the case [without exogenous regressors](https://github.com/databricks-industry-solutions/many-model-forecasting/blob/main/examples/daily/global_daily.py), so we will skip the redundant parts and focus only on the essentials. 

# COMMAND ----------

# MAGIC %md
# MAGIC ### Cluster setup
# MAGIC
# MAGIC We recommend using a cluster with [Databricks Runtime 16.4 LTS for ML](https://docs.databricks.com/en/release-notes/runtime/16.4lts-ml.html). The cluster should be single-node with one or more GPU instances: e.g. [g4dn.12xlarge [T4]](https://aws.amazon.com/ec2/instance-types/g4/) on AWS or [Standard_NC64as_T4_v3](https://learn.microsoft.com/en-us/azure/virtual-machines/nct4-v3-series) on Azure. MMF leverages [neuralforecast](https://nixtlaverse.nixtla.io/neuralforecast/index.html) which is built on top of [pytorch](https://lightning.ai/docs/pytorch/stable/common/trainer.html) and can therefore utilize all the [available resources](https://lightning.ai/docs/pytorch/stable/common/trainer.html). 

# COMMAND ----------

# MAGIC %md
# MAGIC ### Install and import packages
# MAGIC Check out [requirements-global.txt](https://github.com/databricks-industry-solutions/many-model-forecasting/blob/main/requirements-global.txt) if you're interested in the libraries we use.

# COMMAND ----------

# MAGIC %pip install datasetsforecast==0.0.8 --quiet

# COMMAND ----------

import logging
from tqdm.autonotebook import tqdm
logger = spark._jvm.org.apache.log4j
logging.getLogger("py4j.java_gateway").setLevel(logging.ERROR)
logging.getLogger("py4j.clientserver").setLevel(logging.ERROR)
import uuid

# COMMAND ----------

# MAGIC %md
# MAGIC ### Prepare data 
# MAGIC Before running this notebook, download the dataset from [Kaggle](https://www.kaggle.com/competitions/rossmann-store-sales/data) and store them in Unity Catalog as a [volume](https://docs.databricks.com/en/connect/unity-catalog/volumes.html).

# COMMAND ----------

catalog = "mmf" # Name of the catalog we use to manage our assets
db = "rossmann" # Name of the schema we use to manage our assets (e.g. datasets)
volume = "csv" # Name of the volume where you have your rossmann dataset csv sotred
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
# MAGIC Note that in `rossmann_daily_train` we have our target variable `Sales` but not in `rossmann_daily_test`. This is because `rossmann_daily_test` is going to be used as our `scoring_data` that stores `dynamic_future_categorical` variables of the future dates. When you adapt this notebook to your use case, make sure to comply with these datasets formats. See neuralforecast's [documentation](https://nixtlaverse.nixtla.io/neuralforecast/examples/exogenous_variables.html) for more detail on exogenous regressors.

# COMMAND ----------

# MAGIC %md ### Models
# MAGIC Let's configure a list of models we are going to apply to our time series for evaluation and forecasting. A comprehensive list of all supported models is available in [mmf_sa/models/README.md](https://github.com/databricks-industry-solutions/many-model-forecasting/blob/main/mmf_sa/models/README.md). Look for the models where `model_type: global`; these are the global models we import from [neuralforecast](https://github.com/Nixtla/neuralforecast). Check their documentation for the detailed description of each model. 
# MAGIC
# MAGIC Exogenous regressors are currently only supported for [some models](https://nixtlaverse.nixtla.io/neuralforecast/models.html) (e.g. `NeuralForecastAutoNBEATSx`). The following list of models support exogenous regressors.

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
]

# COMMAND ----------

# MAGIC %md ### Run MMF
# MAGIC
# MAGIC Now, we run the evaluation and forecasting using `run_forecast` function. We are providing the training table and the scoring table names. If `scoring_data` is not provided or if the same name as `train_data` is provided, the models will ignore the `dynamic_future_numerical` and `dynamic_future_categorical` regressors. Note that we are providing a covariate field (i.e. `dynamic_future_numerical` or `dynamic_future_categorical`) this time in `run_forecast` function called in [examples/run_external_regressors_daily.py](https://github.com/databricks-industry-solutions/many-model-forecasting/blob/main/examples/run_external_regressors_daily.py). There are also other convariate fields, namely `static_features`, and `dynamic_historical_numerical` and `dynamic_historical_categorical`, which you can provide. Read more about these covariates in [neuralforecast's documentation](https://nixtlaverse.nixtla.io/neuralforecast/examples/exogenous_variables.html).

# COMMAND ----------

# The same run_id will be assigned to all the models. This makes it easier to run the post evaluation analysis later.
run_id = str(uuid.uuid4())

for model in active_models:
  dbutils.notebook.run(
    "../run_external_regressors_daily",
    timeout_seconds=0,
    arguments={"catalog": catalog, "db": db, "model": model, "run_id": run_id, "user": user})

# COMMAND ----------

# MAGIC %md ### Evaluate
# MAGIC In `evaluation_output` table, the we store all evaluation results for all backtesting trials from all models.

# COMMAND ----------

display(
  spark.sql(f"""
            select * from {catalog}.{db}.rossmann_daily_evaluation_output 
            where Store=49
            order by Store, model, backtest_window_start_date
            """))

# COMMAND ----------

# MAGIC %md ### Forecast
# MAGIC In `scoring_output` table, forecasts for each time series from each model are stored.

# COMMAND ----------

display(spark.sql(f"""
                  select * from {catalog}.{db}.rossmann_daily_scoring_output 
                  where Store=49 
                  order by Store, model
                  """))

# COMMAND ----------

# MAGIC %md
# MAGIC Refer to the [notebook](https://github.com/databricks-industry-solutions/many-model-forecasting/blob/main/examples/post-evaluation-analysis.ipynb) for guidance on performing fine-grained model selection after running `run_forecast`.

# COMMAND ----------

# MAGIC %md ### Delete Tables
# MAGIC Let's clean up the tables.

# COMMAND ----------

#display(spark.sql(f"delete from {catalog}.{db}.rossmann_daily_evaluation_output"))

# COMMAND ----------

#display(spark.sql(f"delete from {catalog}.{db}.rossmann_daily_scoring_output"))
