# Databricks notebook source
# MAGIC %md
# MAGIC This is an example notebook that shows how to use [Moirai](https://github.com/SalesforceAIResearch/uni2ts) models on Databricks. The notebook loads, fine-tunes, and registers the model.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cluster setup
# MAGIC We recommend using a cluster with [Databricks Runtime 14.3 LTS for ML](https://docs.databricks.com/en/release-notes/runtime/14.3lts-ml.html) or above. The cluster can be single-node or multi-node with one or more GPU instances on each worker: e.g. [g5.12xlarge [A10G]](https://aws.amazon.com/ec2/instance-types/g5/) on AWS or [Standard_NV72ads_A10_v5](https://learn.microsoft.com/en-us/azure/virtual-machines/nva10v5-series) on Azure.

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
db = "random"  # Name of the schema we use to manage our assets (e.g. datasets)
volume = "moirai_fine_tune" # Name of the volume we store the data and the weigts
model = "moirai-1.0-R-small"  # Alternatibely: moirai-1.0-R-base, moirai-1.0-R-large
n = 100  # Number of time series to sample

# COMMAND ----------

# Make sure that the database exists.
_ = spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{db}")

# Make sure that the volume exists. We stored the fine-tuned weights here.
_ = spark.sql(f"CREATE VOLUME IF NOT EXISTS {catalog}.{db}.{volume}")

# COMMAND ----------

# MAGIC %md
# MAGIC We synthesize `n` number of time series (randomly sampled) at daily resolution and store it as a csv file in UC Volume. 

# COMMAND ----------

import pandas as pd
import numpy as np

df_dict = {}

for i in range(n):

    # Create a date range for the index
    date_range = pd.date_range(start='2021-01-01', end='2023-12-31', freq='D')

    # Create a DataFrame with a date range index and two columns: 'item_id' and 'target'
    df = pd.DataFrame({
        'item_id': str(f"item_{i}"),
        'target': np.random.randn(len(date_range))
    }, index=date_range)

    # Set 'item_id' as the second level of the MultiIndex
    df.set_index('item_id', append=True, inplace=True)

    # Sort the index
    df.sort_index(inplace=True)

    df_dict[i] = df


pdf = pd.concat([df_dict[i] for i in range(n)])
pdf.to_csv(f"/Volumes/{catalog}/{db}/{volume}/random.csv", index=True)
pdf

# COMMAND ----------

# MAGIC %md
# MAGIC This dotenv file is needed to call the [`uni2ts.data.builder.simple`](https://github.com/SalesforceAIResearch/uni2ts/blob/main/src/uni2ts/data/builder/simple.py) function from the [`uni2ts`](https://github.com/SalesforceAIResearch/uni2ts) library to build a dataset. 

# COMMAND ----------

import os
import site

# Construct the path to the 'uni2ts' directory within the site-packages directory.
# site.getsitepackages()[0] returns the path to the first directory in the list of site-packages directories.
uni2ts = os.path.join(site.getsitepackages()[0], "uni2ts")

# Construct the path to the '.env' file within the 'uni2ts' directory.
dotenv = os.path.join(uni2ts, ".env")

# Set the 'DOTENV' environment variable to the path of the '.env' file.
# This tells the system where to find the '.env' file.
os.environ['DOTENV'] = dotenv

# Set the 'CUSTOM_DATA_PATH' environment variable to a path constructed using the provided 'catalog', 'db', and 'volume'.
# This sets a custom data path for the application to use.
os.environ['CUSTOM_DATA_PATH'] = f"/Volumes/{catalog}/{db}/{volume}"


# COMMAND ----------

# MAGIC %sh 
# MAGIC rm -f $DOTENV
# MAGIC touch $DOTENV
# MAGIC echo "CUSTOM_DATA_PATH=$CUSTOM_DATA_PATH" >> $DOTENV

# COMMAND ----------

# MAGIC %md
# MAGIC We convert the dataset into the Uni2TS format. `random` is the name we give to the training dataset, which we load from our volume's location. See the [README](https://github.com/SalesforceAIResearch/uni2ts/tree/main?tab=readme-ov-file#fine-tuning) of Uni2TS for more information on the parameters. 

# COMMAND ----------

# MAGIC %sh python -m uni2ts.data.builder.simple random /Volumes/mmf/random/moirai_fine_tune/random.csv \
# MAGIC     --dataset_type long \
# MAGIC     --offset 640

# COMMAND ----------

# MAGIC %md
# MAGIC ##Run Fine-tuning
# MAGIC
# MAGIC In this example, we wil fine-tune `moirai-1.0-R-small` for max 100 epochs with early stopping (can be specified here: [`examples/foundation-model-examples/moirai/conf/finetune/default.yaml`](https://github.com/databricks-industry-solutions/many-model-forecasting/blob/main/examples/foundation-model-examples/moirai/conf/finetune/default.yaml)). The learning rate is set to 1e-3, which you can modify in the model specific configuration file: [`examples/foundation-model-examples/moirai/conf/finetune/model/moirai_1.0_R_small.yaml`](https://github.com/databricks-industry-solutions/many-model-forecasting/blob/main/examples/foundation-model-examples/moirai/conf/finetune/model/moirai_1.0_R_small.yaml). 
# MAGIC
# MAGIC Make sure that you have the configuration yaml files placed inside the [`conf`](examples/foundation-model-examples/moirai/conf) folder and the [`train.py`](https://github.com/databricks-industry-solutions/many-model-forecasting/blob/main/examples/foundation-model-examples/moirai/train.py) script in the same directory. These two assets are taken directly from and [cli/conf](https://github.com/SalesforceAIResearch/uni2ts/tree/main/cli/conf) and [cli/train.py](https://github.com/SalesforceAIResearch/uni2ts/blob/main/cli/train.py). They are subject to change as the Moirai' team develops the framework further. Keep your eyes on the latest changes (we will try too) and use the latest versions as needed.
# MAGIC
# MAGIC The key configuration files to be customized for you use case are [`examples/foundation-model-examples/moirai/conf/finetune/default.yaml`](https://github.com/databricks-industry-solutions/many-model-forecasting/blob/main/examples/foundation-model-examples/moirai/conf/finetune/default.yaml), [`examples/foundation-model-examples/moirai/conf/finetune/data/random.yaml`](https://github.com/databricks-industry-solutions/many-model-forecasting/blob/main/examples/foundation-model-examples/moirai/conf/finetune/data/random.yaml) and [`examples/foundation-model-examples/moirai/conf/finetune/val_data/random.yaml`](https://github.com/databricks-industry-solutions/many-model-forecasting/blob/main/examples/foundation-model-examples/moirai/conf/finetune/val_data/random.yaml). Read through the Moirai [documentation](https://github.com/SalesforceAIResearch/uni2ts) for more detail.

# COMMAND ----------

# MAGIC %sh python train.py \
# MAGIC   -cp conf/finetune \
# MAGIC   run_name=random_run \
# MAGIC   model=moirai_1.0_R_small \
# MAGIC   data=random \
# MAGIC   val_data=random

# COMMAND ----------

# MAGIC %md
# MAGIC ##Register Model
# MAGIC We get the fine-tuned weights from the run from the UC volume, wrap the pipeline with [`mlflow.pyfunc.PythonModel`](https://mlflow.org/docs/latest/python_api/mlflow.pyfunc.html) and register this on Unity Catalog.

# COMMAND ----------

import mlflow
import torch
import numpy as np
from mlflow.models.signature import ModelSignature  # Used to define the model input and output schema.
from mlflow.types import DataType, Schema, TensorSpec  # Used to define the data types and structure for model inputs and outputs.

# Set the MLflow registry URI to Databricks Unity Catalog
mlflow.set_registry_uri("databricks-uc")

# Define a custom MLflow Python model class
class FineTunedMoiraiModel(mlflow.pyfunc.PythonModel):  
    def predict(self, context, input_data, params=None):
        from einops import rearrange  # Einops is a library for tensor operations.
        from uni2ts.model.moirai import MoiraiForecast, MoiraiModule  # Import the required classes from the Moirai model.
        
        # Determine the device to run the model on (GPU if available, otherwise CPU)
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        
        # Load the pre-trained Moirai model from the checkpoint
        model = MoiraiForecast.load_from_checkpoint(
            prediction_length=10,
            context_length=len(input_data),
            patch_size=32,
            num_samples=10,
            target_dim=1,
            feat_dynamic_real_dim=0,
            past_feat_dynamic_real_dim=0,
            checkpoint_path=context.artifacts["weights"],
        ).to(device)
        
        # Prepare the input data for the model
        # Time series values. Shape: (batch, time, variate)
        past_target = rearrange(
            torch.as_tensor(input_data, dtype=torch.float32), "t -> 1 t 1"
        )
        # 1s if the value is observed, 0s otherwise. Shape: (batch, time, variate)
        past_observed_target = torch.ones_like(past_target, dtype=torch.bool)
        # 1s if the value is padding, 0s otherwise. Shape: (batch, time)
        past_is_pad = torch.zeros_like(past_target, dtype=torch.bool).squeeze(-1)
        
        # Generate the forecast using the model
        forecast = model(
            past_target=past_target.to(device),
            past_observed_target=past_observed_target.to(device),
            past_is_pad=past_is_pad.to(device),
        )
        
        # Return the median forecast
        return np.median(forecast.cpu()[0], axis=0)

# Define the input schema for the model
input_schema = Schema([TensorSpec(np.dtype(np.double), (-1,))])
# Define the output schema for the model
output_schema = Schema([TensorSpec(np.dtype(np.uint8), (-1,))])
# Create a ModelSignature object to represent the input and output schema
signature = ModelSignature(inputs=input_schema, outputs=output_schema)
# Create an example input to log with the model
input_example = np.random.rand(52)

# Define the registered model name using variables for catalog, database, and volume
registered_model_name = f"{catalog}.{db}.moirai-1-r-small_finetuned"

# Define the path to the model weights
weights = f"/Volumes/{catalog}/{db}/{volume}/outputs/moirai_1.0_R_small/random/random_run/checkpoints/epoch=0-step=100.ckpt"

# Log and register the model with MLflow
with mlflow.start_run() as run:
    mlflow.pyfunc.log_model(
        "model",  # The artifact path where the model is logged
        python_model=FineTunedMoiraiModel(),  # The custom Python model to log
        registered_model_name=registered_model_name,  # The name to register the model under
        artifacts={"weights": weights},  # The model artifacts to log
        signature=signature,  # The model signature
        input_example=input_example,  # An example input to log with the model
        pip_requirements=[
            "git+https://github.com/SalesforceAIResearch/uni2ts.git",
        ],  # The Python packages required to run the model
    )


# COMMAND ----------

# MAGIC %md
# MAGIC ##Reload Model
# MAGIC We reload the model from the registry and perform forecasting on a randomly generated time series (for testing purpose). You can also go ahead and deploy this model behind a Model Serving's real-time endpoint. See the previous notebook: [`01_moirai_load_inference`](https://github.com/databricks-industry-solutions/many-model-forecasting/blob/main/examples/foundation-model-examples/chronos/02_moirai_load_inference.py) for more information.

# COMMAND ----------

from mlflow import MlflowClient
client = MlflowClient()

# Function to get the latest version number of a registered model
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

# Create input data for the model
input_data = np.random.rand(52)  # Generate random input data of shape (52,)

# Generate forecasts using the loaded model
loaded_model.predict(input_data)  # Use the loaded model to make predictions on the input data


# COMMAND ----------


