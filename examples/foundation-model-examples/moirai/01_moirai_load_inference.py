# Databricks notebook source
# MAGIC %md
# MAGIC This is an example notebook that shows how to use [Moirai](https://github.com/SalesforceAIResearch/uni2ts) models on Databricks. 
# MAGIC
# MAGIC The notebook loads the model, distributes the inference, registers the model, deploys the model and makes online forecasts.

# COMMAND ----------

# MAGIC %pip install git+https://github.com/SalesforceAIResearch/uni2ts.git --quiet
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Prepare Data
# MAGIC Make sure that the catalog and the schema already exist.

# COMMAND ----------

catalog = "mmf"  # Name of the catalog we use to manage our assets
db = "m4"  # Name of the schema we use to manage our assets (e.g. datasets)
n = 100  # Number of time series to sample

# COMMAND ----------

# This cell will create tables: 
# 1. {catalog}.{db}.m4_daily_train, 
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
from einops import rearrange
from typing import Iterator
from pyspark.sql.functions import pandas_udf


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


def create_forecast_udf(repository, prediction_length, patch_size, num_samples):

  @pandas_udf('array<double>')
  def forecast_udf(bulk_iterator: Iterator[pd.Series]) -> Iterator[pd.Series]:
    
    ## initialization step
    import torch
    import numpy as np
    import pandas as pd
    from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
    module = MoiraiModule.from_pretrained(repository)

    ## inference
    for bulk in bulk_iterator:
      median = []
      for series in bulk:
        model = MoiraiForecast(
          module=module,
          prediction_length=prediction_length,
          context_length=len(series),
          patch_size=patch_size,
          num_samples=num_samples,
          target_dim=1,
          feat_dynamic_real_dim=0,
          past_feat_dynamic_real_dim=0,
        )
        # Time series values. Shape: (batch, time, variate)
        past_target = rearrange(
            torch.as_tensor(series, dtype=torch.float32), "t -> 1 t 1"
        )
        # 1s if the value is observed, 0s otherwise. Shape: (batch, time, variate)
        past_observed_target = torch.ones_like(past_target, dtype=torch.bool)
        # 1s if the value is padding, 0s otherwise. Shape: (batch, time)
        past_is_pad = torch.zeros_like(past_target, dtype=torch.bool).squeeze(-1)
        forecast = model(
            past_target=past_target,
            past_observed_target=past_observed_target,
            past_is_pad=past_is_pad,
        )
        median.append(np.median(forecast[0], axis=0))
    yield pd.Series(median)  
  return forecast_udf

# COMMAND ----------

model = "moirai-1.0-R-small"  # Alternatibely moirai-1.0-R-base, moirai-1.0-R-large
prediction_length = 10  # Time horizon for forecasting
num_samples = 10  # Number of forecast to generate. We will take median as our final forecast.
patch_size = 32  # Patch size: choose from {"auto", 8, 16, 32, 64, 128}
freq = "D" # Frequency of the time series
device_count = torch.cuda.device_count()  # Number of GPUs available

# COMMAND ----------

get_horizon_timestamps = create_get_horizon_timestamps(freq=freq, prediction_length=prediction_length)

forecast_udf = create_forecast_udf(
  repository=f"Salesforce/{model}", 
  prediction_length=prediction_length,
  patch_size=patch_size,
  num_samples=num_samples,
  )

forecasts = df.repartition(device_count).select(
  df.unique_id, 
  get_horizon_timestamps(df.ds).alias("ds"),
  forecast_udf(df.y).alias("forecast"),
  )

display(forecasts)

# COMMAND ----------

# MAGIC %md
# MAGIC ##Register Model

# COMMAND ----------

import mlflow
import torch
import numpy as np
from mlflow.models.signature import ModelSignature
from mlflow.types import DataType, Schema, TensorSpec
mlflow.set_registry_uri("databricks-uc")


class MoiraiModel(mlflow.pyfunc.PythonModel):
  def __init__(self, repository):
    import torch
    from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
    self.module = MoiraiModule.from_pretrained(repository) 
  
  def predict(self, context, input_data, params=None):
    from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
    model = MoiraiForecast(
          module=self.module,
          prediction_length=10,
          context_length=len(input_data),
          patch_size=32,
          num_samples=10,
          target_dim=1,
          feat_dynamic_real_dim=0,
          past_feat_dynamic_real_dim=0,
        )
    
    # Time series values. Shape: (batch, time, variate)
    past_target = rearrange(
        torch.as_tensor(input_data, dtype=torch.float32), "t -> 1 t 1"
    )
    # 1s if the value is observed, 0s otherwise. Shape: (batch, time, variate)
    past_observed_target = torch.ones_like(past_target, dtype=torch.bool)
    # 1s if the value is padding, 0s otherwise. Shape: (batch, time)
    past_is_pad = torch.zeros_like(past_target, dtype=torch.bool).squeeze(-1)
    forecast = model(
        past_target=past_target,
        past_observed_target=past_observed_target,
        past_is_pad=past_is_pad,
    )
    return np.median(forecast[0], axis=0)

pipeline = MoiraiModel(f"Salesforce/{model}")
input_schema = Schema([TensorSpec(np.dtype(np.double), (-1,))])
output_schema = Schema([TensorSpec(np.dtype(np.uint8), (-1,))])
signature = ModelSignature(inputs=input_schema, outputs=output_schema)
input_example = np.random.rand(52)
registered_model_name=f"{catalog}.{db}.moirai-1-r-small"

with mlflow.start_run() as run:
  mlflow.pyfunc.log_model(
    "model",
    python_model=pipeline,
    registered_model_name=registered_model_name,
    signature=signature,
    input_example=input_example,
    pip_requirements=[
      "git+https://github.com/SalesforceAIResearch/uni2ts.git",
    ],
  )

# COMMAND ----------

# MAGIC %md
# MAGIC ##Reload Model

# COMMAND ----------

from mlflow import MlflowClient
mlflow_client = MlflowClient()

def get_latest_model_version(mlflow_client, registered_model_name):
    latest_version = 1
    for mv in mlflow_client.search_model_versions(f"name='{registered_model_name}'"):
        version_int = int(mv.version)
        if version_int > latest_version:
            latest_version = version_int
    return latest_version

model_version = get_latest_model_version(mlflow_client, registered_model_name)
logged_model = f"models:/{registered_model_name}/{model_version}"

# Load model as a PyFuncModel
loaded_model = mlflow.pyfunc.load_model(logged_model)

# COMMAND ----------

input_data = np.random.rand(52)
loaded_model.predict(input_data)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Deploy Model on Databricks Model Serving

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

model_serving_endpoint_name = "moirai-1-r-small"

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

input_data = np.random.rand(52)
forecast(input_data)

# COMMAND ----------

func_delete_model_serving_endpoint(model_serving_endpoint_name)

# COMMAND ----------


