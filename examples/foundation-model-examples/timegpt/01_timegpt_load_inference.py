# Databricks notebook source
# MAGIC %md
# MAGIC This is an example notebook that shows how to use Foundational Model Time-Series [TimeGPT](https://docs.nixtla.io/) models on Databricks. 
# MAGIC The notebook loads the model, distributes the inference, registers the model, deploys the model and makes online forecasts.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Pre-requisites to URL and API key for AzureAI
# MAGIC Here are the prerequisites:
# MAGIC 1. If you don’t have an Azure subscription, get one here: https://azure.microsoft.com/en-us/pricing/purchase-options/pay-as-you-go
# MAGIC 2. Create an Azure AI Studio hub and project. Supported regions are: East US 2, Sweden Central, North Central US, East US, West US, West US3, South Central US. Make sure you pick one these as the Azure region for the hub.
# MAGIC Next, you need to create a deployment to obtain the inference API and key.
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

# This cell runs the notebook ../data_preparation and creates the following tables with M4 data: 
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
from typing import Iterator,Tuple
from pyspark.sql.functions import pandas_udf


## Function to select a single time series from the prepared dataset
def get_single_time_series(unique_id):
  # Filter the DataFrame to get records with the specified unique_id and convert to a pandas DataFrame
  pdf = df.filter(df.unique_id == unique_id).toPandas()
  # Create a dictionary with timestamp and value columns
  pdf = {
    "timestamp" : list(pdf['ds'][0]),
    "value" : list(pdf['y'][0])
  }
  # Return a new pandas DataFrame created from the dictionary
  return pd.DataFrame(pdf)


## Function to create a Pandas UDF to generate forecasts given a time series history
def create_forecast_udf(model_url, api_key, prediction_length=12):

  @pandas_udf('struct<timestamp:array<string>,forecast:array<double>>')
  def forecast_udf(iterator: Iterator[Tuple[pd.Series, pd.Series]]) -> Iterator[pd.DataFrame]:
    
    ## Initialization step
    import numpy as np
    import pandas as pd
    from nixtla import NixtlaClient  # Import NixtlaClient from the nixtla library

    # Initialize the NixtlaClient with the provided model URL and API key
    model = NixtlaClient(
      base_url=model_url,
      api_key=api_key)

    ## Inference step
    for timeseries, past_values in iterator:
      median = []  # Initialize a list to store the forecast results
      for ts, y in zip(timeseries, past_values):
        # Create a DataFrame from the time series and past values
        tdf = pd.DataFrame({"timestamp": ts,
                            "value": y})
        # Generate a forecast using the NixtlaClient model
        pred = model.forecast(
                      df=tdf,
                      h=prediction_length,
                      time_col="timestamp",
                      target_col="value")
        
        # Append the forecast results to the median list
        median.append({'timestamp': list(pred['timestamp'].astype(str).values),
                       'forecast': list(pred['TimeGPT'].values)})
    # Yield the results as a pandas DataFrame
    yield pd.DataFrame(median)  
    
  return forecast_udf  # Return the forecast UDF

# COMMAND ----------

# MAGIC %md
# MAGIC We specify the requirements of our forecasts. 

# COMMAND ----------

# DBTITLE 1,Forecasting with TimeGEN on Azure AI
model_url = "https://TimeGEN-1-pj-serverless.eastus2.inference.ai.azure.com" # Put your model url
prediction_length = 12  # Time horizon for forecasting
api_key = dbutils.secrets.get(scope=scope_name, key=key_name) # Get credential from secrets 
freq = "D" # Frequency of the time series

# COMMAND ----------

# MAGIC %md
# MAGIC Let's generate the forecasts.

# COMMAND ----------

# Create Pandas UDF
forecast_udf = create_forecast_udf(
  model_url=model_url, 
  api_key=api_key,
  )

# Apply Pandas UDF to the dataframe
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
from mlflow.types import DataType, Schema, TensorSpec, ColSpec, ParamSpec, ParamSchema

mlflow.set_registry_uri("databricks-uc")  # Set the MLflow registry URI to Databricks Unity Catalog.

# Define a custom MLflow Python model class for TimeGPTPipeline
class TimeGPTPipeline(mlflow.pyfunc.PythonModel):
  def __init__(self, model_url, api_key):
    import numpy as np
    import pandas as pd
    from nixtla import NixtlaClient  # Import NixtlaClient from the nixtla library
    self.model_url = model_url  # Store the model URL
    self.api_key = api_key  # Store the API key
  
  def predict(self, context, input_data, params=None):
    from nixtla import NixtlaClient  # Import NixtlaClient from the nixtla library
    model = NixtlaClient(
        base_url=self.model_url,
        api_key=self.api_key)  # Initialize the NixtlaClient with the stored model URL and API key
    
    # Generate a forecast using the NixtlaClient model
    pred = model.forecast(
              df=input_data,
              h=params['h'],  # Use the horizon length from the params
              time_col="timestamp",
              target_col="value")
    # Rename the forecast column to 'forecast'
    pred.rename(columns={'TimeGPT': 'forecast'},
                inplace=True)
    return pred  # Return the prediction DataFrame

# Initialize the custom TimeGPTPipeline with the specified model URL and API key
pipeline = TimeGPTPipeline(model_url=model_url, api_key=api_key)

# Define the input and output schema for the model
input_schema = Schema([ColSpec.from_json_dict(**{"type": "datetime", "name": "timestamp", "required": True}),
                       ColSpec.from_json_dict(**{"type": "double", "name": "value", "required": True})])
output_schema = Schema([ColSpec.from_json_dict(**{"type": "datetime", "name": "timestamp", "required": True}),
                       ColSpec.from_json_dict(**{"type": "double", "name": "forecast", "required": True})])
param_schema = ParamSchema([ParamSpec.from_json_dict(**{"type": "integer", "name": "h", "default": 12})])
# Create a ModelSignature object to represent the input, output, and parameter schema
signature = ModelSignature(inputs=input_schema, outputs=output_schema, params=param_schema)

# Define the registered model name using variables for catalog and database
registered_model_name = f"{catalog}.{db}.time_gpt"

# Filter the DataFrame to get records with the specified unique_id and convert to a pandas DataFrame
pdf = df.filter(df.unique_id == 'D7').toPandas()
pdf = {
  "timestamp" : list(pdf['ds'][0]),
  "value" : list(pdf['y'][0])
}
# Get a single time series from the dataset
pdf = get_single_time_series('D4')

# Log and register the model with MLflow
with mlflow.start_run() as run:
  mlflow.pyfunc.log_model(
    "model",  # The artifact path where the model is logged
    python_model=pipeline,  # The custom Python model to log
    registered_model_name=registered_model_name,  # The name to register the model under
    signature=signature,  # The model signature
    input_example=pdf[:10],  # An example input to log with the model
    pip_requirements=[
      "nixtla"  # Python package requirements
    ]
  )


# COMMAND ----------

# MAGIC %md
# MAGIC ##Reload Model
# MAGIC Once the registration is complete, we will reload the model and generate forecasts.

# COMMAND ----------

from mlflow import MlflowClient
mlflow_client = MlflowClient()

# Define a function to get the latest version number of a registered model
def get_latest_model_version(mlflow_client, registered_model_name):
    latest_version = 1  # Initialize the latest version number to 1
    # Iterate through all model versions of the specified registered model
    for mv in mlflow_client.search_model_versions(f"name='{registered_model_name}'"):
        version_int = int(mv.version)  # Convert the version number to an integer
        if version_int > latest_version:  # Check if the current version is greater than the latest version
            latest_version = version_int  # Update the latest version number
    return latest_version  # Return the latest version number

# Get the latest version number of the specified registered model
model_version = get_latest_model_version(mlflow_client, registered_model_name)
# Construct the model URI using the registered model name and the latest version number
logged_model = f"models:/{registered_model_name}/{model_version}"

# Load the model as a PyFuncModel using the constructed model URI
loaded_model = mlflow.pyfunc.load_model(logged_model)


# COMMAND ----------

# MAGIC %md
# MAGIC ###Test the endpoint before deployment

# COMMAND ----------

# Get random data points
pdf = get_single_time_series('D4')

# Generate forecasts
loaded_model.predict(pdf, params = {'h': 20})

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

model_serving_endpoint_name = "time-gpt"

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

# Function to create an endpoint in Model Serving and deploy the model behind it
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

# Function to delete the endpoint from Model Serving
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

import time
import mlflow

# Define a function to wait for a serving endpoint to be ready
def wait_for_endpoint():
    endpoint_url = f"https://{instance}/api/2.0/serving-endpoints"  # Construct the base URL for the serving endpoints API
    while True:  # Infinite loop to repeatedly check the status of the endpoint
        url = f"{endpoint_url}/{model_serving_endpoint_name}"  # Construct the URL for the specific model serving endpoint
        response = requests.get(url, headers=headers)  # Send a GET request to the endpoint URL with the necessary headers
        
        # Ensure the response status code is 200 (OK)
        assert (
            response.status_code == 200
        ), f"Expected an HTTP 200 response, received {response.status_code}\n{response.text}"

        # Extract the status of the endpoint from the response JSON
        status = response.json().get("state", {}).get("ready", {})
        # print("status",status)  # Optional: Print the status for debugging purposes
        
        # Check if the endpoint status is "READY"
        if status == "READY":
            print(status)  # Print the status if the endpoint is ready
            print("-" * 80)  # Print a separator line for clarity
            return  # Exit the function when the endpoint is ready
        else:
            # Print a message indicating the endpoint is not ready and wait for 5 minutes
            print(f"Endpoint not ready ({status}), waiting 5 minutes")
            time.sleep(300)  # Wait for 300 seconds (5 minutes) before checking again

# Get the Databricks web application URL using an MLflow utility function
api_url = mlflow.utils.databricks_utils.get_webapp_url()

# Call the wait_for_endpoint function to wait for the serving endpoint to be ready
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

# MAGIC %md
# MAGIC ### Test online forecast

# COMMAND ----------

# Send request to the endpoint
pdf = get_single_time_series('D3')
pdf['timestamp'] = pdf['timestamp'].astype(str)
forecast(pdf)

# COMMAND ----------

# Delete the serving endpoint
func_delete_model_serving_endpoint(model_serving_endpoint_name)

# COMMAND ----------


