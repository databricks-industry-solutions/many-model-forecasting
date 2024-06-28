# Databricks notebook source
# MAGIC %md
# MAGIC This is an example notebook that shows how to use Foundational Model Time-Series [TimeGPT](https://docs.nixtla.io/) models on Databricks and fine-tune it on the fly.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Pre-requisites to URL and API key for AzureAI
# MAGIC Here are the prerequisites:
# MAGIC 1. If you don’t have an Azure subscription, get one here: https://azure.microsoft.com/en-us/pricing/purchase-options/pay-as-you-go
# MAGIC 2. Create an Azure AI Studio hub and project. Supported regions are: East US 2, Sweden Central, North Central US, East US, West US, West US3, South Central US. Make sure you pick one these as the Azure region for the hub.
# MAGIC Next, you need to create a deployment to obtain the inference API and key:
# MAGIC
# MAGIC 3. Open the TimeGEN-1 model card in the model catalog: https://aka.ms/aistudio/landing/nixtlatimegen1
# MAGIC 4. Click on Deploy and select the Pay-as-you-go option.
# MAGIC 5. Subscribe to the Marketplace offer and deploy. You can also review the API pricing at this step.
# MAGIC 6. You should land on the deployment page that shows you the API key and URL in less than a minute.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cluster setup
# MAGIC
# MAGIC TimeGPT is accessible through an API as a service, so the actual compute for inference or fine-tuning will not take place on Databricks. For this reason a GPU cluster is not necessary and we recommend using a cluster with [Databricks Runtime 14.3 LTS for ML](https://docs.databricks.com/en/release-notes/runtime/14.3lts-ml.html) or above with CPUs. This notebook will leverage [Pandas UDF](https://docs.databricks.com/en/udf/pandas.html) for distributing the inference tasks and utilizing all the available resource.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Install package

# COMMAND ----------

# DBTITLE 1,Import Libraries
# MAGIC %pip install nixtla --quiet
# MAGIC %pip install --upgrade mlflow --quiet
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Add the API key as a secret

# COMMAND ----------

key_name = f'api_key'
scope_name = f'time-gpt'

# COMMAND ----------

# MAGIC %md
# MAGIC If this is your first time running the notebook and you still don't have your credential managed in the secret, uncomment and run the following cell. Read more about Databricks secrets management [here](https://docs.databricks.com/en/security/secrets/index.html).

# COMMAND ----------

#import time
#from databricks.sdk import WorkspaceClient

#w = WorkspaceClient()

# put the key in secret 
#w.secrets.create_scope(scope=scope_name)
#w.secrets.put_secret(scope=scope_name, key=key_name, string_value=f'<input api key here>')

# cleanup
#w.secrets.delete_secret(scope=scope_name, key=key_name)
## w.secrets.delete_secret(scope=scope_name, key=key_name)
## w.secrets.delete_scope(scope=scope_name)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Prepare data 
# MAGIC We use [`datasetsforecast`](https://github.com/Nixtla/datasetsforecast/tree/main/) package to download M4 data. M4 dataset contains a set of time series which we use for testing MMF. Below we have written a number of custome functions to convert M4 time series to an expected format.
# MAGIC
# MAGIC Make sure that the catalog and the schema already exist.

# COMMAND ----------

catalog = "mmf"  # Name of the catalog we use to manage our assets
db = "m4"  # Name of the schema we use to manage our assets (e.g. datasets)
n = 10  # Number of time series to sample

# COMMAND ----------

# This cell will create tables: 
# 1. {catalog}.{db}.m4_daily_train, 
# 2. {catalog}.{db}.m4_monthly_train
dbutils.notebook.run("../data_preparation", timeout_seconds=0, arguments={"catalog": catalog, "db": db, "n": n})

# COMMAND ----------

from pyspark.sql.functions import collect_list,size

# Make sure that the data exists
df = spark.table(f'{catalog}.{db}.m4_daily_train')
df = df.groupBy('unique_id').agg(collect_list('ds').alias('ds'), collect_list('y').alias('y'))
df = df.filter(size(df.ds) >= 300)
display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Distribute Fine-Tuning and Inference
# MAGIC We use [Pandas UDF](https://docs.databricks.com/en/udf/pandas.html#iterator-of-series-to-iterator-of-series-udf) to distribute fine-tuning and inference.

# COMMAND ----------

import pandas as pd
import numpy as np
import torch
from typing import Iterator,Tuple
from pyspark.sql.functions import pandas_udf


## function to create a Pandas UDF to generate fine-tune and generate forecasts given a time series history
def create_forecast_udf(model_url, api_key, prediction_length=12, ft_steps=10):

  @pandas_udf('struct<timestamp:array<string>,forecast:array<double>>')
  def forecast_udf(iterator: Iterator[Tuple[pd.Series, pd.Series]]) -> Iterator[pd.DataFrame]:
    
    ## initialization step
    import numpy as np
    import pandas as pd
    from nixtla import NixtlaClient

    model = NixtlaClient(
    base_url=model_url,
    api_key=api_key)

    ## inference
    for timeseries, past_values in iterator:
      median = []
      for ts, y in zip(timeseries, past_values):
        tdf = pd.DataFrame({"timestamp":ts,
                            "value" :y})
        
        pred = model.forecast(
                      df=tdf,
                      h=prediction_length,
                      finetune_steps=ft_steps,
                      time_col="timestamp",
                      target_col="value")

        median.append({'timestamp' : list(pred['timestamp'].astype(str).    values),
                       'forecast' : list(pred['TimeGPT'].values) })
    yield pd.DataFrame(median)  
  return forecast_udf

# COMMAND ----------

# MAGIC %md
# MAGIC We specify the requirements of our forecasts. 

# COMMAND ----------

model_url = "https://TimeGEN-1-pj-serverless.eastus2.inference.ai.azure.com"
prediction_length = 10  # Time horizon for forecasting
ft_steps = 10  # Number of training interations to perform for fientuning
api_key = dbutils.secrets.get(scope =scope_name,key = key_name)
freq = "D" # Frequency of the time series

# COMMAND ----------

# MAGIC %md
# MAGIC Let's fine-tune and generate forecasts.

# COMMAND ----------

forecast_udf = create_forecast_udf(
  model_url=model_url, 
  api_key=api_key,
  )

forecasts = df.select(
  df.unique_id,
  forecast_udf("ds", "y").alias("forecast"),
  ).select("unique_id", "forecast.timestamp", "forecast.forecast")

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
from mlflow.types import DataType, Schema, TensorSpec ,ColSpec, ParamSpec,ParamSchema
mlflow.set_registry_uri("databricks-uc")


class TimeGPTPipeline(mlflow.pyfunc.PythonModel):
  def __init__(self, model_url, api_key):
    import numpy as np
    import pandas as pd
    from nixtla import NixtlaClient
    self.model_url = model_url
    self.api_key = api_key
  
  def predict(self, context, input_data, params=None):
    from nixtla import NixtlaClient
    model = NixtlaClient(
        base_url=self.model_url,
        api_key=self.api_key)
    
    pred = model.forecast(
              df=input_data,
              h=params['h'],
              finetune_steps = params['finetune_steps'],
              time_col="timestamp",
              target_col="value")
    pred.rename(columns={'TimeGPT': 'forecast'},
                inplace=True)
    return pred

pipeline = TimeGPTPipeline(model_url = model_url,api_key=api_key)

# Need model signature to be able to register the model
input_schema = Schema([ColSpec.from_json_dict(**{"type": "datetime", "name": "timestamp", "required": True}),
                       ColSpec.from_json_dict(**{"type": "double", "name": "value", "required": True})])
output_schema = Schema([ColSpec.from_json_dict(**{"type": "datetime", "name": "timestamp", "required": True}),
                       ColSpec.from_json_dict(**{"type": "double", "name": "forecast", "required": True})])
param_schema = ParamSchema([ParamSpec.from_json_dict(**{"type": "integer", "name": "h", "default": 12}),
                            ParamSpec.from_json_dict(**{"type": "integer", "name": "finetune_steps", "default": 10})])
signature = ModelSignature(inputs=input_schema, outputs=output_schema,params = param_schema)
registered_model_name=f"{catalog}.{db}.time_gpt_ft"

pdf = df.filter(df.unique_id == 'D7').toPandas()
pdf = {
  "timestamp" : list(pdf['ds'][0]),
  "value" : list(pdf['y'][0])
}
pdf = pd.DataFrame(pdf)

# Log and register the model
with mlflow.start_run() as run:
  mlflow.pyfunc.log_model(
    "model",
    python_model=pipeline,
    registered_model_name=registered_model_name,
    signature=signature,
    input_example=pdf,
    pip_requirements=[
      "nixtla"
    ]
  )

# COMMAND ----------

# MAGIC %md
# MAGIC ##Reload Model
# MAGIC Once the registration is complete, we will reload the model and generate forecasts.

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

# MAGIC %md
# MAGIC ###Test the endpoint before deployment

# COMMAND ----------

# Test the endpoint before deployment
loaded_model.predict(pdf,params = {'h' :20})

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

model_serving_endpoint_name = "timegpt_ft"

my_json = {
    "name": model_serving_endpoint_name,
    "config": {
        "served_models": [
            {
                "model_name": registered_model_name,
                "model_version": model_version,
                "workload_type": "CPU_SMALL",
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
    body = {'dataframe_split': input_data.to_dict(orient='split'),"params" :{'h':20}}
    data = json.dumps(body)
    response = requests.request(method="POST", headers=headers, url=url, data=data)
    if response.status_code != 200:
        raise Exception(
            f"Request failed with status {response.status_code}, {response.text}"
        )
    return response.json()

# COMMAND ----------

pdf['timestamp'] = pdf['timestamp'].astype('str')
forecast(pdf)

# COMMAND ----------

# Delete the serving endpoint
func_delete_model_serving_endpoint(model_serving_endpoint_name)

# COMMAND ----------


