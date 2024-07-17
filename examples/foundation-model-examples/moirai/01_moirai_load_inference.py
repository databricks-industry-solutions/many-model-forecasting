# Databricks notebook source
# MAGIC %md
# MAGIC This is an example notebook that shows how to use [Moirai](https://github.com/SalesforceAIResearch/uni2ts) models on Databricks. The notebook loads the model, distributes the inference, registers the model, deploys the model and makes online forecasts.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cluster setup
# MAGIC
# MAGIC We recommend using a cluster with [Databricks Runtime 14.3 LTS for ML](https://docs.databricks.com/en/release-notes/runtime/14.3lts-ml.html) or above. The cluster can be single-node or multi-node with one or more GPU instances on each worker: e.g. [g5.12xlarge [A10G]](https://aws.amazon.com/ec2/instance-types/g5/) on AWS or [Standard_NV72ads_A10_v5](https://learn.microsoft.com/en-us/azure/virtual-machines/nva10v5-series) on Azure. This notebook will leverage [Pandas UDF](https://docs.databricks.com/en/udf/pandas.html) for distributing the inference tasks and utilizing all the available resource.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Install package

# COMMAND ----------

# MAGIC %pip install git+https://github.com/SalesforceAIResearch/uni2ts.git --quiet
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Prepare Data
# MAGIC We are using [`datasetsforecast`](https://github.com/Nixtla/datasetsforecast/tree/main/) package to download M4 data. M4 dataset contains a set of time series which we use for testing MMF. Below we have written a number of custome functions to convert M4 time series to an expected format.
# MAGIC
# MAGIC Make sure that the catalog and the schema already exist.

# COMMAND ----------

catalog = "mmf"  # Name of the catalog we use to manage our assets
db = "m4"  # Name of the schema we use to manage our assets (e.g. datasets)
n = 100  # Number of time series to sample

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
from einops import rearrange
from typing import Iterator
from pyspark.sql.functions import pandas_udf

# Function to create a Pandas UDF to generate horizon timestamps
def create_get_horizon_timestamps(freq, prediction_length):
    """
    Creates a Pandas UDF to generate horizon timestamps based on the given frequency and prediction length.

    Parameters:
    - freq (str): The frequency of the time series ('M' for monthly, 'D' for daily, etc.).
    - prediction_length (int): The number of future timestamps to generate.

    Returns:
    - get_horizon_timestamps (function): A Pandas UDF function that generates horizon timestamps.
    """
    
    @pandas_udf('array<timestamp>')
    def get_horizon_timestamps(batch_iterator: Iterator[pd.Series]) -> Iterator[pd.Series]:
        # Determine the offset for timestamp increments based on the frequency
        one_ts_offset = pd.offsets.MonthEnd(1) if freq == "M" else pd.DateOffset(days=1)
        
        barch_horizon_timestamps = []
        # Iterate over batches of series in the batch iterator
        for batch in batch_iterator:
            for series in batch:
                timestamp = last = series.max()
                horizon_timestamps = []
                # Generate future timestamps based on the prediction length
                for i in range(prediction_length):
                    timestamp = timestamp + one_ts_offset
                    horizon_timestamps.append(timestamp.to_numpy())
                barch_horizon_timestamps.append(np.array(horizon_timestamps))
        # Yield the generated horizon timestamps as a Pandas Series
        yield pd.Series(barch_horizon_timestamps)

    return get_horizon_timestamps

# Function to create a Pandas UDF to generate forecasts
def create_forecast_udf(repository, prediction_length, patch_size, num_samples):
    """
    Creates a Pandas UDF to generate forecasts using a pre-trained model.

    Parameters:
    - repository (str): The path to the pre-trained model repository.
    - prediction_length (int): The length of the forecast horizon.
    - patch_size (int): The size of the patches for the model input.
    - num_samples (int): The number of samples to generate for each forecast.

    Returns:
    - forecast_udf (function): A Pandas UDF function that generates forecasts.
    """
    
    @pandas_udf('array<double>')
    def forecast_udf(bulk_iterator: Iterator[pd.Series]) -> Iterator[pd.Series]:
        ## Initialization step
        import torch
        import numpy as np
        import pandas as pd
        from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
        
        # Load the pre-trained model module from the repository
        module = MoiraiModule.from_pretrained(repository)

        ## Inference
        for bulk in bulk_iterator:
            median = []
            for series in bulk:
                # Initialize the forecast model with the loaded module and given parameters
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
                # Prepare the past target tensor. Shape: (batch, time, variate)
                past_target = rearrange(
                    torch.as_tensor(series, dtype=torch.float32), "t -> 1 t 1"
                )
                # Create a tensor indicating observed values. Shape: (batch, time, variate)
                past_observed_target = torch.ones_like(past_target, dtype=torch.bool)
                # Create a tensor indicating padding values. Shape: (batch, time)
                past_is_pad = torch.zeros_like(past_target, dtype=torch.bool).squeeze(-1)
                
                # Generate the forecast
                forecast = model(
                    past_target=past_target,
                    past_observed_target=past_observed_target,
                    past_is_pad=past_is_pad,
                )
                # Append the median forecast of the first sample to the list
                median.append(np.median(forecast[0], axis=0))
        # Yield the generated forecasts as a Pandas Series
        yield pd.Series(median)
        
    return forecast_udf


# COMMAND ----------

# MAGIC %md
# MAGIC We specify the requirements of our forecasts. 

# COMMAND ----------

model = "moirai-1.0-R-small"  # Alternatibely moirai-1.0-R-base, moirai-1.0-R-large
prediction_length = 10  # Time horizon for forecasting
num_samples = 10  # Number of forecast to generate. We will take median as our final forecast.
patch_size = 32  # Patch size: choose from {"auto", 8, 16, 32, 64, 128}
freq = "D" # Frequency of the time series
device_count = torch.cuda.device_count()  # Number of GPUs available

# COMMAND ----------

# MAGIC %md
# MAGIC Let's generate the forecasts.

# COMMAND ----------

# Create the Pandas UDF for generating horizon timestamps using the specified frequency and prediction length
get_horizon_timestamps = create_get_horizon_timestamps(freq=freq, prediction_length=prediction_length)

# Create the Pandas UDF for generating forecasts using the specified model repository and forecast parameters
forecast_udf = create_forecast_udf(
  repository=f"Salesforce/{model}",  # Path to the pre-trained model repository
  prediction_length=prediction_length,  # Length of the forecast horizon
  patch_size=patch_size,  # Size of the patches for the model input
  num_samples=num_samples,  # Number of samples to generate for each forecast
)

# Repartition the DataFrame to match the number of devices (for parallel processing) and select the required columns
forecasts = df.repartition(device_count).select(
  df.unique_id,  # Select the unique identifier for each time series
  get_horizon_timestamps(df.ds).alias("ds"),  # Generate horizon timestamps and alias as 'ds'
  forecast_udf(df.y).alias("forecast"),  # Generate forecasts and alias as 'forecast'
)

# Display the resulting DataFrame with unique_id, horizon timestamps, and forecasts
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

# Set the MLflow registry URI to Databricks Unity Catalog
mlflow.set_registry_uri("databricks-uc")

class MoiraiModel(mlflow.pyfunc.PythonModel):
    def __init__(self, repository):
        """
        Initialize the MoiraiModel class by loading the pre-trained model from the given repository.
        
        Parameters:
        - repository (str): The path to the pre-trained model repository.
        """
        import torch
        from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
        
        # Load the pre-trained model module from the repository
        self.module = MoiraiModule.from_pretrained(repository)
  
    def predict(self, context, input_data, params=None):
        """
        Generate forecasts using the loaded model.
        
        Parameters:
        - context: The context in which the model is being run.
        - input_data: The input data for prediction, expected to be a time series.
        - params: Additional parameters for prediction (not used here).
        
        Returns:
        - forecast: The median forecast result as a NumPy array.
        """
        from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
        
        # Initialize the forecast model with the loaded module and given parameters
        model = MoiraiForecast(
            module=self.module,
            prediction_length=10,  # Length of the forecast horizon
            context_length=len(input_data),  # Context length is the length of the input data
            patch_size=32,  # Size of the patches for the model input
            num_samples=10,  # Number of samples to generate for each forecast
            target_dim=1,  # Dimension of the target variable
            feat_dynamic_real_dim=0,  # No dynamic real features
            past_feat_dynamic_real_dim=0,  # No past dynamic real features
        )
        
        # Prepare the past target tensor. Shape: (batch, time, variate)
        past_target = rearrange(
            torch.as_tensor(input_data, dtype=torch.float32), "t -> 1 t 1"
        )
        # Create a tensor indicating observed values. Shape: (batch, time, variate)
        past_observed_target = torch.ones_like(past_target, dtype=torch.bool)
        # Create a tensor indicating padding values. Shape: (batch, time)
        past_is_pad = torch.zeros_like(past_target, dtype=torch.bool).squeeze(-1)
        
        # Generate the forecast
        forecast = model(
            past_target=past_target,
            past_observed_target=past_observed_target,
            past_is_pad=past_is_pad,
        )
        
        # Return the median forecast of the first sample
        return np.median(forecast[0], axis=0)

# Initialize the MoiraiModel with the specified model repository
pipeline = MoiraiModel(f"Salesforce/{model}")

# Define the input and output schema for the model
input_schema = Schema([TensorSpec(np.dtype(np.double), (-1,))])
output_schema = Schema([TensorSpec(np.dtype(np.uint8), (-1,))])
signature = ModelSignature(inputs=input_schema, outputs=output_schema)

# Example input data for model registration
input_example = np.random.rand(52)

# Define the registered model name
registered_model_name = f"{catalog}.{db}.moirai-1-r-small"

# Log and register the model with MLflow
with mlflow.start_run() as run:
    mlflow.pyfunc.log_model(
        "model",
        python_model=pipeline,  # The custom Python model
        registered_model_name=registered_model_name,  # The name under which to register the model
        signature=signature,  # The model signature
        input_example=input_example,  # An example of the input data
        pip_requirements=[
            "git+https://github.com/SalesforceAIResearch/uni2ts.git",
        ],
    )


# COMMAND ----------

# MAGIC %md
# MAGIC ##Reload Model
# MAGIC Once the registration is complete, we will reload the model and generate forecasts.

# COMMAND ----------

from mlflow import MlflowClient

# Create an instance of the MlflowClient to interact with the MLflow tracking server
mlflow_client = MlflowClient()

def get_latest_model_version(mlflow_client, registered_model_name):
    """
    Retrieve the latest version number of a registered model.
    
    Parameters:
    - mlflow_client (MlflowClient): The MLflow client instance.
    - registered_model_name (str): The name of the registered model.
    
    Returns:
    - latest_version (int): The latest version number of the registered model.
    """
    # Initialize the latest version to 1 (assuming at least one version exists)
    latest_version = 1
    
    # Iterate over all model versions for the given registered model
    for mv in mlflow_client.search_model_versions(f"name='{registered_model_name}'"):
        # Convert the version to an integer
        version_int = int(mv.version)
        
        # Update the latest version if a higher version is found
        if version_int > latest_version:
            latest_version = version_int
            
    # Return the latest version number
    return latest_version

# Get the latest version of the registered model
model_version = get_latest_model_version(mlflow_client, registered_model_name)

# Construct the URI for the logged model using the registered model name and latest version
logged_model = f"models:/{registered_model_name}/{model_version}"

# Load the model as a PyFuncModel from the logged model URI
loaded_model = mlflow.pyfunc.load_model(logged_model)


# Create random input data (52 data points)
input_data = np.random.rand(52)

# Generate forecasts using the loaded model
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

model_serving_endpoint_name = "moirai-1-r-small"

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

import time
import mlflow
import requests

def wait_for_endpoint():
    """
    Waits for a model serving endpoint to become ready.

    This function continuously polls the serving endpoint's status and waits until the endpoint is ready.
    """
    # Construct the base URL for the serving endpoint API
    endpoint_url = f"https://{instance}/api/2.0/serving-endpoints"
    
    while True:
        # Construct the full URL for the specific model serving endpoint
        url = f"{endpoint_url}/{model_serving_endpoint_name}"
        
        # Send a GET request to the endpoint URL with the required headers
        response = requests.get(url, headers=headers)
        
        # Assert that the response status code is 200 (OK)
        assert (
            response.status_code == 200
        ), f"Expected an HTTP 200 response, received {response.status_code}\n{response.text}"
        
        # Extract the 'ready' status from the JSON response
        status = response.json().get("state", {}).get("ready", {})
        
        # Check if the status is "READY"
        if status == "READY":
            # Print the status and a separator line, then exit the function
            print(status)
            print("-" * 80)
            return
        else:
            # Print a message indicating the endpoint is not ready and wait for 5 minutes (300 seconds)
            print(f"Endpoint not ready ({status}), waiting 5 minutes")
            time.sleep(300)

# Get the API URL for the current Databricks instance
api_url = mlflow.utils.databricks_utils.get_webapp_url()

# Call the function to wait for the endpoint to become ready
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

# Construct the endpoint URL for model invocation using the provided instance and model serving endpoint name.
# This URL is used to send data to the model and get predictions.
endpoint_url = f"https://{instance}/serving-endpoints/{model_serving_endpoint_name}/invocations"

# Retrieve the Databricks API token using dbutils (a utility available in Databricks notebooks).
# This token is used for authentication when making requests to the endpoint.
token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

def forecast(input_data, url=endpoint_url, databricks_token=token):
    """
    Send input data to the model serving endpoint and retrieve the forecast.

    Parameters:
    - input_data (numpy.ndarray): The input data to be sent to the model.
    - url (str): The endpoint URL for model invocation.
    - databricks_token (str): The Databricks API token for authentication.

    Returns:
    - dict: The JSON response from the model containing the forecast.
    """
    # Set the request headers, including the authorization token and content type.
    headers = {
        "Authorization": f"Bearer {databricks_token}",
        "Content-Type": "application/json",
    }
    
    # Convert the input data to a list and create the request body.
    body = {"inputs": input_data.tolist()}
    
    # Serialize the request body to a JSON formatted string.
    data = json.dumps(body)
    
    # Send a POST request to the endpoint URL with the headers and serialized data.
    response = requests.request(method="POST", headers=headers, url=url, data=data)
    
    # Check if the response status code is not 200 (OK), raise an exception if the request failed.
    if response.status_code != 200:
        raise Exception(
            f"Request failed with status {response.status_code}, {response.text}"
        )
    
    # Return the JSON response from the model containing the forecast.
    return response.json()


# COMMAND ----------

# Send request to the endpoint
input_data = np.random.rand(52)
forecast(input_data)

# COMMAND ----------

# Delete the serving endpoint
func_delete_model_serving_endpoint(model_serving_endpoint_name)

# COMMAND ----------


