# Databricks notebook source
# MAGIC %md
# MAGIC This is an example notebook that shows how to use [chronos](https://github.com/amazon-science/chronos-forecasting/tree/main) models on Databricks. The notebook loads the model, distributes the inference, registers the model, deploys the model and makes online forecasts.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cluster setup
# MAGIC
# MAGIC We recommend using a cluster with [Databricks Runtime 14.3 LTS for ML](https://docs.databricks.com/en/release-notes/runtime/14.3lts-ml.html) or above. The cluster can be single-node or multi-node with one or more GPU instances on each worker: e.g. [g5.12xlarge [A10G]](https://aws.amazon.com/ec2/instance-types/g5/) on AWS or [Standard_NV72ads_A10_v5](https://learn.microsoft.com/en-us/azure/virtual-machines/nva10v5-series) on Azure. This notebook will leverage [Pandas UDF](https://docs.databricks.com/en/udf/pandas.html) for distributing the inference tasks and utilizing all the available resource.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Install package

# COMMAND ----------

# MAGIC %pip install git+https://github.com/amazon-science/chronos-forecasting.git --quiet
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Prepare data 
# MAGIC We use [`datasetsforecast`](https://github.com/Nixtla/datasetsforecast/tree/main/) package to download M4 data. M4 dataset contains a set of time series which we use for testing MMF. Below we have written a number of custome functions to convert M4 time series to an expected format.
# MAGIC
# MAGIC Make sure that the catalog and the schema already exist.

# COMMAND ----------

catalog = "mmf"  # Name of the catalog we use to manage our assets
db = "m4" # Name of the schema we use to manage our assets (e.g. datasets)
n = 100  # Number of time series to sample

# COMMAND ----------

# This cell will create tables: 
# 1. {catalog}.{db}.m4_daily_train
# 2. {catalog}.{db}.m4_monthly_train
dbutils.notebook.run("../data_preparation", timeout_seconds=0, arguments={"catalog": catalog, "db": db, "n": n})

# COMMAND ----------

from pyspark.sql.functions import collect_list

# Make sure that the data exists
df = spark.table(f'{catalog}.{db}.m4_daily_train')
df = df.groupBy('unique_id').agg(collect_list('ds').alias('ds'), collect_list('y').alias('y'))
display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Distribute Inference
# MAGIC We use [Pandas UDF](https://docs.databricks.com/en/udf/pandas.html#iterator-of-series-to-iterator-of-series-udf) to distribute the inference.

# COMMAND ----------

import pandas as pd
import numpy as np
import torch
from typing import Iterator
from pyspark.sql.functions import pandas_udf

# Function to create a Pandas UDF to generate horizon timestamps
def create_get_horizon_timestamps(freq, prediction_length):

  @pandas_udf('array<timestamp>')
  def get_horizon_timestamps(batch_iterator: Iterator[pd.Series]) -> Iterator[pd.Series]:   
      one_ts_offset = pd.offsets.MonthEnd(1) if freq == "M" else pd.DateOffset(days=1)
      barch_horizon_timestamps = []
      for batch in batch_iterator:
          for series in batch:
              timestamp = last = series.max()
              horizon_timestamps = []
              for i in range(prediction_length):
                  timestamp = timestamp + one_ts_offset
                  horizon_timestamps.append(timestamp.to_numpy())
              barch_horizon_timestamps.append(np.array(horizon_timestamps))
      yield pd.Series(barch_horizon_timestamps)

  return get_horizon_timestamps


# Function to create a Pandas UDF to generate forecasts
def create_forecast_udf(repository, prediction_length, num_samples, batch_size):

  @pandas_udf('array<double>')
  def forecast_udf(bulk_iterator: Iterator[pd.Series]) -> Iterator[pd.Series]:
    
    ## initialization step
    import numpy as np
    import pandas as pd
    import torch
    from chronos import ChronosPipeline
    pipeline = ChronosPipeline.from_pretrained(
        repository,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        )
    
    ## inference
    for bulk in bulk_iterator:
      median = []
      for i in range(0, len(bulk), batch_size):
        batch = bulk[i:i+batch_size]
        contexts = [torch.tensor(list(series)) for series in batch]
        forecasts = pipeline.predict(context=contexts, prediction_length=prediction_length, num_samples=num_samples)
        median.extend([np.median(forecast, axis=0) for forecast in forecasts])

    yield pd.Series(median)
  
  return forecast_udf

# COMMAND ----------

# MAGIC %md
# MAGIC We specify the requirements of our forecasts. 

# COMMAND ----------

chronos_model = "chronos-t5-tiny"  # Alternatively: chronos-t5-mini, chronos-t5-small, chronos-t5-base, chronos-t5-large
prediction_length = 10  # Time horizon for forecasting
num_samples = 10  # Number of forecast to generate. We will take median as our final forecast.
batch_size = 4  # Number of time series to process simultaneously 
freq = "D" # Frequency of the time series
device_count = torch.cuda.device_count()  # Number of GPUs available

# COMMAND ----------

# MAGIC %md
# MAGIC Let's generate the forecasts.

# COMMAND ----------

get_horizon_timestamps = create_get_horizon_timestamps(freq=freq, prediction_length=prediction_length)

forecast_udf = create_forecast_udf(
  repository=f"amazon/{chronos_model}", 
  prediction_length=prediction_length,
  num_samples=num_samples,
  batch_size=batch_size,
  )

forecasts = df.repartition(device_count).select(
  df.unique_id,  
  get_horizon_timestamps(df.ds).alias("ds"),
  forecast_udf(df.y).alias("forecast")
  )

display(forecasts)

# COMMAND ----------

# MAGIC %md
# MAGIC ##Register Model
# MAGIC We will package our model using [`mlflow.pyfunc.PythonModel`](https://mlflow.org/docs/latest/python_api/mlflow.pyfunc.html) and register this in Unity Catalog.

# COMMAND ----------

import mlflow
import torch
import numpy as np
from mlflow.models.signature import ModelSignature
from mlflow.types import DataType, Schema, TensorSpec
mlflow.set_registry_uri("databricks-uc")


class ChronosModel(mlflow.pyfunc.PythonModel):
  def __init__(self, repository):
    import torch
    from chronos import ChronosPipeline
    self.pipeline = ChronosPipeline.from_pretrained(
      repository,
      device_map="cuda",
      torch_dtype=torch.bfloat16,
      )  
  
  def predict(self, context, input_data, params=None):
    history = [torch.tensor(list(series)) for series in input_data]
    forecast = self.pipeline.predict(
        context=history,
        prediction_length=10,
        num_samples=10,
    )
    return forecast.numpy()

pipeline = ChronosModel(f"amazon/{chronos_model}")

# Need model signature to be able to register the model
input_schema = Schema([TensorSpec(np.dtype(np.double), (-1, -1))])
output_schema = Schema([TensorSpec(np.dtype(np.uint8), (-1, -1, -1))])
signature = ModelSignature(inputs=input_schema, outputs=output_schema)
input_example = np.random.rand(1, 52)
registered_model_name=f"{catalog}.{db}.{chronos_model}"

# Log and register the model
with mlflow.start_run() as run:
  mlflow.pyfunc.log_model(
    "model",
    python_model=pipeline,
    registered_model_name=registered_model_name,
    signature=signature,
    input_example=input_example,
    pip_requirements=[
      f"git+https://github.com/amazon-science/chronos-forecasting.git",
    ],
  )

# COMMAND ----------

# MAGIC %md
# MAGIC ##Reload Model
# MAGIC Once the registration is complete, we will reload the model and generate forecasts.

# COMMAND ----------

from mlflow import MlflowClient
client = MlflowClient()

# Get the latest model version
def get_latest_model_version(client, registered_model_name):
    latest_version = 1
    for mv in client.search_model_versions(f"name='{registered_model_name}'"):
        version_int = int(mv.version)
        if version_int > latest_version:
            latest_version = version_int
    return latest_version

model_version = get_latest_model_version(client, registered_model_name)
logged_model = f"models:/{registered_model_name}/{model_version}"

# Load model as a PyFuncModel
loaded_model = mlflow.pyfunc.load_model(logged_model)

# Create random input data
input_data = np.random.rand(5, 52) # (batch, series)

# Generate forecasts
loaded_model.predict(input_data)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Deploy Model
# MAGIC We will deploy our model behind a real-time endpoint of [Databricks Mosaic AI Model Serving](https://www.databricks.com/product/model-serving).

# COMMAND ----------

# With the token, you can create our authorization header for our subsequent REST calls
token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().getOrElse(None)
headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

# Next you need an endpoint at which to execute your request which you can get from the notebook's tags collection
java_tags = dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags()

# This object comes from the Java CM - Convert the Java Map opject to a Python dictionary
tags = sc._jvm.scala.collection.JavaConversions.mapAsJavaMap(java_tags)

# Lastly, extract the Databricks instance (domain name) from the dictionary
instance = tags["browserHostName"]

# COMMAND ----------

import requests

model_serving_endpoint_name = chronos_model

# auto_capture_config specifies where the inference logs should be written
my_json = {
    "name": model_serving_endpoint_name,
    "config": {
        "served_models": [
            {
                "model_name": registered_model_name,
                "model_version": model_version,
                "workload_type": "GPU_SMALL",
                "workload_size": "Small",
                "scale_to_zero_enabled": "true",
            }
        ],
        "auto_capture_config": {
            "catalog_name": catalog,
            "schema_name": db,
            "table_name_prefix": model_serving_endpoint_name,
        },
    },
}

# Make sure to drop the inference table of it exists
_ = spark.sql(
    f"DROP TABLE IF EXISTS {catalog}.{db}.`{model_serving_endpoint_name}_payload`"
)

# COMMAND ----------

def func_create_endpoint(model_serving_endpoint_name):
    # get endpoint status
    endpoint_url = f"https://{instance}/api/2.0/serving-endpoints"
    url = f"{endpoint_url}/{model_serving_endpoint_name}"
    r = requests.get(url, headers=headers)
    if "RESOURCE_DOES_NOT_EXIST" in r.text:
        print(
            "Creating this new endpoint: ",
            f"https://{instance}/serving-endpoints/{model_serving_endpoint_name}/invocations",
        )
        re = requests.post(endpoint_url, headers=headers, json=my_json)
    else:
        new_model_version = (my_json["config"])["served_models"][0]["model_version"]
        print(
            "This endpoint existed previously! We are updating it to a new config with new model version: ",
            new_model_version,
        )
        # update config
        url = f"{endpoint_url}/{model_serving_endpoint_name}/config"
        re = requests.put(url, headers=headers, json=my_json["config"])
        # wait till new config file in place
        import time, json

        # get endpoint status
        url = f"https://{instance}/api/2.0/serving-endpoints/{model_serving_endpoint_name}"
        retry = True
        total_wait = 0
        while retry:
            r = requests.get(url, headers=headers)
            assert (
                r.status_code == 200
            ), f"Expected an HTTP 200 response when accessing endpoint info, received {r.status_code}"
            endpoint = json.loads(r.text)
            if "pending_config" in endpoint.keys():
                seconds = 10
                print("New config still pending")
                if total_wait < 6000:
                    # if less the 10 mins waiting, keep waiting
                    print(f"Wait for {seconds} seconds")
                    print(f"Total waiting time so far: {total_wait} seconds")
                    time.sleep(10)
                    total_wait += seconds
                else:
                    print(f"Stopping,  waited for {total_wait} seconds")
                    retry = False
            else:
                print("New config in place now!")
                retry = False

    assert (
        re.status_code == 200
    ), f"Expected an HTTP 200 response, received {re.status_code}"


def func_delete_model_serving_endpoint(model_serving_endpoint_name):
    endpoint_url = f"https://{instance}/api/2.0/serving-endpoints"
    url = f"{endpoint_url}/{model_serving_endpoint_name}"
    response = requests.delete(url, headers=headers)
    if response.status_code != 200:
        raise Exception(
            f"Request failed with status {response.status_code}, {response.text}"
        )
    else:
        print(model_serving_endpoint_name, "endpoint is deleted!")
    return response.json()

# COMMAND ----------

# Create an endpoint. This may take some time.
func_create_endpoint(model_serving_endpoint_name)

# COMMAND ----------

import time, mlflow


def wait_for_endpoint():
    endpoint_url = f"https://{instance}/api/2.0/serving-endpoints"
    while True:
        url = f"{endpoint_url}/{model_serving_endpoint_name}"
        response = requests.get(url, headers=headers)
        assert (
            response.status_code == 200
        ), f"Expected an HTTP 200 response, received {response.status_code}\n{response.text}"

        status = response.json().get("state", {}).get("ready", {})
        # print("status",status)
        if status == "READY":
            print(status)
            print("-" * 80)
            return
        else:
            print(f"Endpoint not ready ({status}), waiting 5 miutes")
            time.sleep(300)  # Wait 300 seconds


api_url = mlflow.utils.databricks_utils.get_webapp_url()

wait_for_endpoint()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Online Forecast
# MAGIC Once the endpoint is ready, let's send a request to the model and generate an online forecast.

# COMMAND ----------

import os
import requests
import pandas as pd
import json
import matplotlib.pyplot as plt

# Replace URL with the end point invocation url you get from Model Seriving page.
endpoint_url = f"https://{instance}/serving-endpoints/{model_serving_endpoint_name}/invocations"
token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
def forecast(input_data, url=endpoint_url, databricks_token=token):
    headers = {
        "Authorization": f"Bearer {databricks_token}",
        "Content-Type": "application/json",
    }
    body = {"inputs": input_data.tolist()}
    data = json.dumps(body)
    response = requests.request(method="POST", headers=headers, url=url, data=data)
    if response.status_code != 200:
        raise Exception(
            f"Request failed with status {response.status_code}, {response.text}"
        )
    return response.json()

# COMMAND ----------

input_data = np.random.rand(5, 52) # (batch, series)
forecast(input_data)

# COMMAND ----------

# Delete the serving endpoint
func_delete_model_serving_endpoint(model_serving_endpoint_name)

# COMMAND ----------


