# Databricks notebook source
# MAGIC %md
# MAGIC This is an example notebook that shows how to use [TimesFM](https://github.com/google-research/timesfm) models on Databricks. The notebook loads the model, distributes the inference, registers the model, deploys the model and makes online forecasts.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cluster setup
# MAGIC
# MAGIC **As of June 5, 2024, TimesFM supports python version below 3.10. So make sure your cluster is below DBR ML 14.3.**
# MAGIC
# MAGIC We recommend using a cluster with [Databricks Runtime 14.3 LTS for ML](https://docs.databricks.com/en/release-notes/runtime/14.3lts-ml.html). The cluster can be single-node or multi-node with one or more GPU instances on each worker: e.g. [g5.12xlarge [A10G]](https://aws.amazon.com/ec2/instance-types/g5/) on AWS or [Standard_NV72ads_A10_v5](https://learn.microsoft.com/en-us/azure/virtual-machines/nva10v5-series) on Azure. This notebook will leverage [Pandas UDF](https://docs.databricks.com/en/udf/pandas.html) for distributing the inference tasks and utilizing all the available resource.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Install package

# COMMAND ----------

# MAGIC %pip install jax[cuda12]==0.4.26 --quiet
# MAGIC %pip install protobuf==3.20.* --quiet
# MAGIC %pip install utilsforecast --quiet
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import sys
import subprocess
package = "git+https://github.com/google-research/timesfm.git"
subprocess.check_call([sys.executable, "-m", "pip", "install", package, "--quiet"])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Prepare data 
# MAGIC We use [`datasetsforecast`](https://github.com/Nixtla/datasetsforecast/tree/main/) package to download M4 data. M4 dataset contains a set of time series which we use for testing MMF. Below we have written a number of custome functions to convert M4 time series to an expected format.
# MAGIC
# MAGIC Make sure that the catalog and the schema already exist.

# COMMAND ----------

catalog = "mmf"  # Name of the catalog we use to manage our assets
db = "m4"  # Name of the schema we use to manage our assets (e.g. datasets)
n = 100  # Number of time series to sample

# COMMAND ----------

# This cell runs the notebook ../data_preparation and creates the following tables with M4 data: 
# 1. {catalog}.{db}.m4_daily_train
# 2. {catalog}.{db}.m4_monthly_train
dbutils.notebook.run("../data_preparation", timeout_seconds=0, arguments={"catalog": catalog, "db": db, "n": n})

# COMMAND ----------

# Make sure that the data exists
df = spark.table(f'{catalog}.{db}.m4_daily_train').toPandas()
display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Distribute Inference

# COMMAND ----------

# MAGIC %md
# MAGIC Distribution of the inference is managed by TimesFM so we don't need to use Pandas UDF. See the [github repository](https://github.com/google-research/timesfm/tree/master?tab=readme-ov-file#initialize-the-model-and-load-a-checkpoint) of TimesFM for detailed description of the input parameters. 

# COMMAND ----------

import timesfm

# Initialize the TimesFm model with specified parameters.
tfm = timesfm.TimesFm(
    context_len=512,  # Max context length of the model. It must be a multiple of input_patch_len, which is 32.
    horizon_len=10,  # Forecast horizon length. It can be set to any value, recommended to be the largest needed.
    input_patch_len=32,  # Length of the input patch.
    output_patch_len=128,  # Length of the output patch.
    num_layers=20,
    model_dims=1280,
    backend="gpu",  # Backend for computation, set to use GPU for faster processing.
)

# Load the pre-trained model from the specified checkpoint.
tfm.load_from_checkpoint(repo_id="google/timesfm-1.0-200m")

# Generate forecasts on the input DataFrame.
forecast_df = tfm.forecast_on_df(
    inputs=df,  # The input DataFrame containing the time series data.
    freq="D",  # Frequency of the time series data, set to daily.
    value_name="y",  # Column name in the DataFrame containing the values to forecast.
    num_jobs=-1,  # Number of parallel jobs to run, set to -1 to use all available processors.
)

# Display the forecast DataFrame.
display(forecast_df)


# COMMAND ----------

# MAGIC %md
# MAGIC ##Register Model

# COMMAND ----------

# MAGIC %md
# MAGIC We should ensure that any non-serializable attributes (like the timesfm model in TimesFMModel class) are not included in the serialization process. One common approach is to override the __getstate__ and __setstate__ methods in the class to manage what gets pickled. This modification ensures that the timesfm model is not included in the serialization process, thus avoiding the error. The load_model method is called to load the model when needed, such as during prediction or after deserialization.
# MAGIC
# MAGIC We will package our model using [`mlflow.pyfunc.PythonModel`](https://mlflow.org/docs/latest/python_api/mlflow.pyfunc.html) and register this in Unity Catalog.

# COMMAND ----------

import mlflow
import torch
import numpy as np
from mlflow.models import infer_signature
from mlflow.models.signature import ModelSignature
from mlflow.types import DataType, Schema, TensorSpec

# Set the MLflow registry URI to Databricks Unity Catalog
mlflow.set_registry_uri("databricks-uc")

# Define a custom MLflow Python model class for TimesFM
class TimesFMModel(mlflow.pyfunc.PythonModel):
    def __init__(self, repository):
        self.repository = repository  # Store the repository ID for the model checkpoint
        self.tfm = None  # Initialize the model attribute to None

    def load_model(self):
        import timesfm
        # Initialize the TimesFm model with specified parameters
        self.tfm = timesfm.TimesFm(
            context_len=512,  # Max context length of the model, must be a multiple of input_patch_len (32).
            horizon_len=10,  # Horizon length for the forecast.
            input_patch_len=32,  # Length of the input patch.
            output_patch_len=128,  # Length of the output patch.
            num_layers=20,
            model_dims=1280,
            backend="gpu",  # Backend for computation, set to GPU.
        )
        # Load the pre-trained model from the specified checkpoint
        self.tfm.load_from_checkpoint(repo_id=self.repository)

    def predict(self, context, input_df, params=None):
        # Load the model if it hasn't been loaded yet
        if self.tfm is None:
            self.load_model()
        # Generate forecasts on the input DataFrame
        forecast_df = self.tfm.forecast_on_df(
            inputs=input_df,  # Input DataFrame containing the time series data.
            freq="D",  # Frequency of the time series data, set to daily.
            value_name="y",  # Column name in the DataFrame containing the values to forecast.
            num_jobs=-1,  # Number of parallel jobs to run, set to -1 to use all available processors.
        )
        return forecast_df  # Return the forecast DataFrame

    def __getstate__(self):
        state = self.__dict__.copy()  # Copy the instance's state
        # Remove the tfm attribute from the state, as it's not serializable
        del state['tfm']
        return state

    def __setstate__(self, state):
        # Restore instance attributes
        self.__dict__.update(state)
        # Reload the model since it was not stored in the state
        self.load_model()

# Initialize the custom TimesFM model with the specified repository ID
pipeline = TimesFMModel("google/timesfm-1.0-200m")
# Infer the model signature based on input and output DataFrames
signature = infer_signature(
    model_input=df,  # Input DataFrame for the model
    model_output=pipeline.predict(None, df),  # Output DataFrame from the model
)

# Define the registered model name using variables for catalog and database
registered_model_name = f"{catalog}.{db}.timesfm-1-200m"

# Start an MLflow run to log and register the model
with mlflow.start_run() as run:
    mlflow.pyfunc.log_model(
        "model",  # The artifact path where the model is logged
        python_model=pipeline,  # The custom Python model to log
        registered_model_name=registered_model_name,  # The name to register the model under
        signature=signature,  # The model signature
        input_example=df,  # An example input to log with the model
        pip_requirements=[
            "jax[cuda12]==0.4.26",  # Required Python packages
            "utilsforecast==0.1.10",
            "git+https://github.com/google-research/timesfm.git",
        ],
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ##Reload Model
# MAGIC Once the registration is complete, we will reload the model and generate forecasts.

# COMMAND ----------

from mlflow import MlflowClient
client = MlflowClient()

# Define a function to get the latest version number of a registered model
def get_latest_model_version(client, registered_model_name):
    latest_version = 1  # Initialize the latest version number to 1
    # Iterate through all model versions of the specified registered model
    for mv in client.search_model_versions(f"name='{registered_model_name}'"):
        version_int = int(mv.version)  # Convert the version number to an integer
        if version_int > latest_version:  # Check if the current version is greater than the latest version
            latest_version = version_int  # Update the latest version number
    return latest_version  # Return the latest version number

# Get the latest version number of the specified registered model
model_version = get_latest_model_version(client, registered_model_name)
# Construct the model URI using the registered model name and the latest version number
logged_model = f"models:/{registered_model_name}/{model_version}"

# Load the model as a PyFuncModel
loaded_model = mlflow.pyfunc.load_model(logged_model)

# Generate forecasts using the loaded model on the input DataFrame
loaded_model.predict(df)  # Use the loaded model to make predictions on the input DataFrame

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

model_serving_endpoint_name = "timesfm-1-200m"

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

# Create an endpoint. This may take some time.
func_create_endpoint(model_serving_endpoint_name)

# COMMAND ----------

import time, mlflow

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
            time.sleep(300)  # Wait for 300 seconds before checking again

# Get the Databricks web application URL using MLflow utility function
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
    body = {"inputs": input_data.tolist()}
    data = json.dumps(body)
    response = requests.request(method="POST", headers=headers, url=url, data=data)
    if response.status_code != 200:
        raise Exception(
            f"Request failed with status {response.status_code}, {response.text}"
        )
    return response.json()

# COMMAND ----------

# Send request to the endpoint
forecast(df)

# COMMAND ----------

# Delete the serving endpoint
func_delete_model_serving_endpoint(model_serving_endpoint_name)
