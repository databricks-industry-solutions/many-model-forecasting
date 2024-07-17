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

# This cell runs the notebook ../data_preparation and creates the following tables with M4 data: 
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
    """
    Creates a Pandas UDF to generate future timestamps based on the given frequency and prediction length.

    Parameters:
    freq (str): Frequency of the timestamps ('M' for month-end, otherwise daily).
    prediction_length (int): Number of future timestamps to generate.

    Returns:
    function: A Pandas UDF that generates an array of future timestamps for each input time series.
    """
    
    @pandas_udf('array<timestamp>')
    def get_horizon_timestamps(batch_iterator: Iterator[pd.Series]) -> Iterator[pd.Series]:
        # Determine the offset for the next timestamp based on the frequency
        one_ts_offset = pd.offsets.MonthEnd(1) if freq == "M" else pd.DateOffset(days=1)
        
        barch_horizon_timestamps = []  # List to hold the arrays of future timestamps
        
        # Iterate over batches of input time series
        for batch in batch_iterator:
            for series in batch:
                timestamp = last = series.max()  # Get the last timestamp in the series
                horizon_timestamps = []  # List to hold future timestamps for the current series
                
                # Generate future timestamps
                for i in range(prediction_length):
                    timestamp = timestamp + one_ts_offset
                    horizon_timestamps.append(timestamp.to_numpy())
                
                barch_horizon_timestamps.append(np.array(horizon_timestamps))
        
        yield pd.Series(barch_horizon_timestamps)  # Yield the result as a Pandas Series

    return get_horizon_timestamps


# Function to create a Pandas UDF to generate forecasts
def create_forecast_udf(repository, prediction_length, num_samples, batch_size):
    """
    Creates a Pandas UDF to generate forecasts using a pretrained model from the given repository.

    Parameters:
    repository (str): Path or identifier for the model repository.
    prediction_length (int): Number of future values to predict.
    num_samples (int): Number of samples to generate for each prediction.
    batch_size (int): Number of time series to process in each batch.

    Returns:
    function: A Pandas UDF that generates an array of forecasted values for each input time series.
    """
    
    @pandas_udf('array<double>')
    def forecast_udf(bulk_iterator: Iterator[pd.Series]) -> Iterator[pd.Series]:
        
        # Initialization step
        import numpy as np
        import pandas as pd
        import torch
        from chronos import ChronosPipeline
        
        # Load the pretrained model from the repository
        pipeline = ChronosPipeline.from_pretrained(repository, device_map="auto", torch_dtype=torch.bfloat16)
        
        # Inference step
        for bulk in bulk_iterator:
            median = []  # List to hold the median forecast for each series
            
            # Process the time series in batches
            for i in range(0, len(bulk), batch_size):
                batch = bulk[i:i+batch_size]
                contexts = [torch.tensor(list(series)) for series in batch]  # Convert series to tensors
                
                # Generate forecasts using the pretrained model
                forecasts = pipeline.predict(context=contexts, prediction_length=prediction_length, num_samples=num_samples)
                
                # Calculate the median forecast for each series
                median.extend([np.median(forecast, axis=0) for forecast in forecasts])
            
            yield pd.Series(median)  # Yield the result as a Pandas Series
        
    return forecast_udf


# COMMAND ----------

# MAGIC %md
# MAGIC We specify the requirements for our forecasts. 

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

# Create a Pandas UDF to generate horizon timestamps with the specified frequency and prediction length
get_horizon_timestamps = create_get_horizon_timestamps(freq=freq, prediction_length=prediction_length)

# Create a Pandas UDF to generate forecasts using a pretrained model from the specified repository
forecast_udf = create_forecast_udf(
    repository=f"amazon/{chronos_model}",  # Model repository path or identifier
    prediction_length=prediction_length,   # Number of future values to predict
    num_samples=num_samples,               # Number of samples to generate for each prediction
    batch_size=batch_size,                 # Number of time series to process in each batch
)

# Apply the UDFs to the DataFrame and select the relevant columns
forecasts = df.repartition(device_count).select(
    df.unique_id,                             # Select the unique identifier for each time series
    get_horizon_timestamps(df.ds).alias("ds"), # Generate and alias the horizon timestamps for each series
    forecast_udf(df.y).alias("forecast")       # Generate and alias the forecasted values for each series
)

# Display the resulting DataFrame containing the forecasts
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

# Set the MLflow registry URI to use Databricks Unity Catalog
mlflow.set_registry_uri("databricks-uc")

# Define a custom MLflow model class for the Chronos pipeline
class ChronosModel(mlflow.pyfunc.PythonModel):
    def __init__(self, repository):
        import torch
        from chronos import ChronosPipeline
        # Initialize the ChronosPipeline with a pretrained model from the specified repository
        self.pipeline = ChronosPipeline.from_pretrained(
            repository,
            device_map="cuda",          # Use GPU for inference
            torch_dtype=torch.bfloat16, # Use bfloat16 precision
        )  
    
    def predict(self, context, input_data, params=None):
        # Convert input data to a list of PyTorch tensors
        history = [torch.tensor(list(series)) for series in input_data]
        # Generate forecasts using the ChronosPipeline
        forecast = self.pipeline.predict(
            context=history,
            prediction_length=10,  # Length of the prediction horizon
            num_samples=10,        # Number of samples to generate
        )
        return forecast.numpy()  # Convert the forecast to a NumPy array

# Instantiate the custom model with the specified repository
pipeline = ChronosModel(f"amazon/{chronos_model}")

# Define the input and output schema for the model signature
input_schema = Schema([TensorSpec(np.dtype(np.double), (-1, -1))])      # Input: 2D array of doubles
output_schema = Schema([TensorSpec(np.dtype(np.uint8), (-1, -1, -1))])  # Output: 3D array of unsigned 8-bit integers
signature = ModelSignature(inputs=input_schema, outputs=output_schema)

# Create an example input for the model (1 sample, 52 features)
input_example = np.random.rand(1, 52)

# Define the registered model name in the format: catalog.database.model_name
registered_model_name = f"{catalog}.{db}.{chronos_model}"

# Log and register the model with MLflow
with mlflow.start_run() as run:
    mlflow.pyfunc.log_model(
      "model",                         # Model artifact path
      python_model=pipeline,           # Custom model class instance
      registered_model_name=registered_model_name, # Name to register the model under
      signature=signature,             # Model signature
      input_example=input_example,     # Example input
      pip_requirements=[               # List of pip requirements
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

# Function to get the latest version of a registered model
def get_latest_model_version(client, registered_model_name):
    latest_version = 1  # Initialize the latest version to 1
    # Iterate through all model versions for the given registered model name
    for mv in client.search_model_versions(f"name='{registered_model_name}'"):
        version_int = int(mv.version)  # Convert version string to integer
        # Update the latest version if a higher version is found
        if version_int > latest_version:
            latest_version = version_int
    return latest_version  # Return the latest version number

# Get the latest version of the specified registered model
model_version = get_latest_model_version(client, registered_model_name)
# Construct the model URI using the registered model name and its latest version
logged_model = f"models:/{registered_model_name}/{model_version}"

# Load the model as a PyFuncModel from the specified URI
loaded_model = mlflow.pyfunc.load_model(logged_model)

# Create random input data (5 samples, each with 52 data points)
input_data = np.random.rand(5, 52)  # (batch, series)

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

def wait_for_endpoint():
    # Construct the base URL for the serving endpoints API
    endpoint_url = f"https://{instance}/api/2.0/serving-endpoints"
    
    while True:
        # Construct the URL for the specific model serving endpoint
        url = f"{endpoint_url}/{model_serving_endpoint_name}"
        
        # Send a GET request to the endpoint URL
        response = requests.get(url, headers=headers)
        
        # Assert that the response status code is 200 (OK)
        assert (
            response.status_code == 200
        ), f"Expected an HTTP 200 response, received {response.status_code}\n{response.text}"
        
        # Extract the status of the endpoint from the response
        status = response.json().get("state", {}).get("ready", {})
        
        # If the endpoint is ready, print the status and return
        if status == "READY":
            print(status)
            print("-" * 80)
            return
        else:
            # If the endpoint is not ready, print the status and wait for 5 minutes
            print(f"Endpoint not ready ({status}), waiting 5 minutes")
            time.sleep(300)  # Wait 300 seconds (5 minutes)

# Get the API URL for the Databricks instance
api_url = mlflow.utils.databricks_utils.get_webapp_url()

# Call the function to wait for the endpoint to be ready
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

# Replace URL with the endpoint invocation URL you get from the Model Serving page.
endpoint_url = f"https://{instance}/serving-endpoints/{model_serving_endpoint_name}/invocations"

# Get the Databricks API token
token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

# Define a function to send input data to the model serving endpoint and get the forecast
def forecast(input_data, url=endpoint_url, databricks_token=token):
    # Set up the headers for the POST request, including the authorization token
    headers = {
        "Authorization": f"Bearer {databricks_token}",
        "Content-Type": "application/json",
    }
    # Prepare the body of the request with the input data
    body = {"inputs": input_data.tolist()}
    # Convert the body to a JSON string
    data = json.dumps(body)
    # Send a POST request to the model serving endpoint
    response = requests.request(method="POST", headers=headers, url=url, data=data)
    # Check if the response status code is not 200 (OK)
    if response.status_code != 200:
        # Raise an exception if the request failed
        raise Exception(
            f"Request failed with status {response.status_code}, {response.text}"
        )
    # Return the response JSON as a Python dictionary
    return response.json()


# COMMAND ----------

# Send request to the endpoint
input_data = np.random.rand(5, 52) # (batch, series)
forecast(input_data)

# COMMAND ----------

# Delete the serving endpoint
func_delete_model_serving_endpoint(model_serving_endpoint_name)

# COMMAND ----------


