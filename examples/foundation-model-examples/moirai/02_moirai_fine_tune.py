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

uni2ts = os.path.join(site.getsitepackages()[0], "uni2ts")
dotenv = os.path.join(uni2ts, ".env")
os.environ['DOTENV'] = dotenv
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
from mlflow.models.signature import ModelSignature
from mlflow.types import DataType, Schema, TensorSpec
mlflow.set_registry_uri("databricks-uc")


class FineTunedMoiraiModel(mlflow.pyfunc.PythonModel):  
  def predict(self, context, input_data, params=None):
    from einops import rearrange
    from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
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
    
    # Time series values. Shape: (batch, time, variate)
    past_target = rearrange(
        torch.as_tensor(input_data, dtype=torch.float32), "t -> 1 t 1"
    )
    # 1s if the value is observed, 0s otherwise. Shape: (batch, time, variate)
    past_observed_target = torch.ones_like(past_target, dtype=torch.bool)
    # 1s if the value is padding, 0s otherwise. Shape: (batch, time)
    past_is_pad = torch.zeros_like(past_target, dtype=torch.bool).squeeze(-1)
    forecast = model(
        past_target=past_target.to(device),
        past_observed_target=past_observed_target.to(device),
        past_is_pad=past_is_pad.to(device),
    )
    return np.median(forecast.cpu()[0], axis=0)

input_schema = Schema([TensorSpec(np.dtype(np.double), (-1,))])
output_schema = Schema([TensorSpec(np.dtype(np.uint8), (-1,))])
signature = ModelSignature(inputs=input_schema, outputs=output_schema)
input_example = np.random.rand(52)
registered_model_name=f"{catalog}.{db}.moirai-1-r-small_finetuned"
weights = f"/Volumes/{catalog}/{db}/{volume}/outputs/moirai_1.0_R_small/random/random_run/checkpoints/epoch=0-step=100.ckpt"

# Log and register the model
with mlflow.start_run() as run:
  mlflow.pyfunc.log_model(
    "model",
    python_model=FineTunedMoiraiModel(),
    registered_model_name=registered_model_name,
    artifacts={"weights": weights},
    signature=signature,
    input_example=input_example,
    pip_requirements=[
      "git+https://github.com/SalesforceAIResearch/uni2ts.git",
    ],
  )

# COMMAND ----------

# MAGIC %md
# MAGIC ##Reload Model
# MAGIC We reload the model from the registry and perform forecasting on a randomly generated time series (for testing purpose). You can also go ahead and deploy this model behind a Model Serving's real-time endpoint. See the previous notebook: [`01_moirai_load_inference`](https://github.com/databricks-industry-solutions/many-model-forecasting/blob/main/examples/foundation-model-examples/chronos/02_moirai_load_inference.py) for more information.

# COMMAND ----------

from mlflow import MlflowClient
client = MlflowClient()

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

# Create input data
input_data = np.random.rand(52)

# Generate forecasts
loaded_model.predict(input_data)

# COMMAND ----------


