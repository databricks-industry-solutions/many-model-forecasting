# Databricks notebook source
# MAGIC %pip install git+https://github.com/SalesforceAIResearch/uni2ts.git --quiet
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Distributed Inference

# COMMAND ----------

import pandas as pd
import numpy as np
import torch
from einops import rearrange
from typing import Iterator
from pyspark.sql.functions import collect_list, pandas_udf


def create_forecast_udf():
  @pandas_udf('array<double>')
  def forecast_udf(bulk_iterator: Iterator[pd.Series]) -> Iterator[pd.Series]:
    
    ## initialization step
    import torch
    import numpy as np
    import pandas as pd
    from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
    module = MoiraiModule.from_pretrained("Salesforce/moirai-1.0-R-small")

    ## inference
    for bulk in bulk_iterator:
      median = []
      for series in bulk:
        model = MoiraiForecast(
          module=module,
          prediction_length=10,
          context_length=len(series),
          patch_size=32,
          num_samples=100,
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
        #print(f"median.append: {np.median(forecast[0], axis=0)}")
        median.append(np.median(forecast[0], axis=0))
    yield pd.Series(median)  
  return forecast_udf

forecast_udf = create_forecast_udf()
df = spark.table('solacc_uc.mmf.m4_daily_train')
df = df.groupBy('unique_id').agg(collect_list('ds').alias('ds'), collect_list('y').alias('y'))
device_count = torch.cuda.device_count()
forecasts = df.repartition(device_count).select(df.unique_id, forecast_udf(df.y).alias("forecast"))

display(forecasts)

# COMMAND ----------

# MAGIC %md
# MAGIC ##Register Model

# COMMAND ----------

import mlflow
import torch
import numpy as np
from mlflow.models import infer_signature
from mlflow.models.signature import ModelSignature
from mlflow.types import DataType, Schema, TensorSpec

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
          num_samples=100,
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
    #print(f"median.append: {np.median(forecast[0], axis=0)}")
    return np.median(forecast[0], axis=0)

pipeline = MoiraiModel("Salesforce/moirai-1.0-R-small")
input_schema = Schema([TensorSpec(np.dtype(np.double), (-1,))])
output_schema = Schema([TensorSpec(np.dtype(np.uint8), (-1,))])
signature = ModelSignature(inputs=input_schema, outputs=output_schema)
input_example = np.random.rand(52)

with mlflow.start_run() as run:
  mlflow.pyfunc.log_model(
    "model",
    python_model=pipeline,
    registered_model_name="solacc_uc.mmf.moirai_test",
    signature=signature,
    input_example=input_example,
    pip_requirements=[
      "git+https://github.com/SalesforceAIResearch/uni2ts.git",
    ],
  )

# COMMAND ----------

pipeline.predict(None, input_example)

# COMMAND ----------

import mlflow
from mlflow import MlflowClient
import pandas as pd

mlflow.set_registry_uri("databricks-uc")
mlflow_client = MlflowClient()

def get_latest_model_version(mlflow_client, registered_name):
    latest_version = 1
    for mv in mlflow_client.search_model_versions(f"name='{registered_name}'"):
        version_int = int(mv.version)
        if version_int > latest_version:
            latest_version = version_int
    return latest_version

model_name = "moirai_test"
registered_name = f"solacc_uc.mmf.{model_name}"
model_version = get_latest_model_version(mlflow_client, registered_name)
logged_model = f"models:/{registered_name}/{model_version}"

# Load model as a PyFuncModel
loaded_model = mlflow.pyfunc.load_model(logged_model)

# COMMAND ----------

import numpy as np
input_data = np.random.rand(52)
loaded_model.predict(input_data)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Online Inference

# COMMAND ----------

import os
import requests
import pandas as pd
import json
import matplotlib.pyplot as plt

# Replace URL with the end point invocation url you get from Model Seriving page.
endpoint_url = "https://e2-demo-field-eng.cloud.databricks.com/serving-endpoints/moirai-test/invocations"
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

forecast(input_data)
