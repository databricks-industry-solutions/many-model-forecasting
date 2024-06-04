# Databricks notebook source
# MAGIC %pip install git+https://github.com/moment-timeseries-foundation-model/moment.git --quiet
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Distributed Inference

# COMMAND ----------

import pandas as pd
import numpy as np
import torch
from typing import Iterator
from pyspark.sql.functions import collect_list, pandas_udf

@pandas_udf('array<double>')
def forecast_udf(batch_iterator: Iterator[pd.Series]) -> Iterator[pd.Series]:
  ## initialization step
  from momentfm import MOMENTPipeline
  model = MOMENTPipeline.from_pretrained(
    "AutonLab/MOMENT-1-large", 
    device_map="cuda",
    model_kwargs={
      "task_name": "forecasting",
      "forecast_horizon": 10},
    )
  model.init()
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model = model.to(device)

  ## inference
  for batch in batch_iterator:
    batch_forecast = []
    for series in batch:
      # takes in tensor of shape [batchsize, n_channels, context_length]
      context = list(series)
      if len(context) < 512:
        input_mask = [1] * len(context) + [0] * (512 - len(context))
        context = context + [0] * (512 - len(context))
      else:
        input_mask = [1] * 512
        context = context[-512:]
      
      input_mask = torch.reshape(torch.tensor(input_mask),(1, 512)).to(device)
      context = torch.reshape(torch.tensor(context),(1, 1, 512)).to(dtype=torch.float32).to(device)
      output = model(context, input_mask=input_mask)
      
      forecast = output.forecast.squeeze().tolist()
      batch_forecast.append(forecast)

  yield pd.Series(batch_forecast)

df = spark.table('solacc_uc.mmf.m4_daily_train')
df = df.groupBy('unique_id').agg(collect_list('ds').alias('ds'), collect_list('y').alias('y'))
device_count = torch.cuda.device_count()
forecasts = df.repartition(device_count).select(df.unique_id, df.ds, forecast_udf(df.y).alias("forecast"))

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

class MomentModel(mlflow.pyfunc.PythonModel):
  def __init__(self, repository):
    from momentfm import MOMENTPipeline
    self.model = MOMENTPipeline.from_pretrained(
      repository, 
      device_map="cuda",
      model_kwargs={
        "task_name": "forecasting",
        "forecast_horizon": 10},
      )
    self.model.init()
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self.model = self.model.to(self.device)

  def predict(self, context, input_data, params=None):
    series = list(input_data)
    if len(series) < 512:
      input_mask = [1] * len(series) + [0] * (512 - len(series))
      series = series + [0] * (512 - len(series))
    else:
      input_mask = [1] * 512
      series = series[-512:]
    input_mask = torch.reshape(torch.tensor(input_mask),(1, 512)).to(self.device)
    series = torch.reshape(torch.tensor(series),(1, 1, 512)).to(dtype=torch.float32).to(self.device)
    output = self.model(series, input_mask=input_mask)
    forecast = output.forecast.squeeze().tolist()
    return forecast

pipeline = MomentModel("AutonLab/MOMENT-1-large")
input_schema = Schema([TensorSpec(np.dtype(np.double), (-1,))])
output_schema = Schema([TensorSpec(np.dtype(np.uint8), (-1,))])
signature = ModelSignature(inputs=input_schema, outputs=output_schema)
input_example = np.random.rand(52)

with mlflow.start_run() as run:
  mlflow.pyfunc.log_model(
    "model",
    python_model=pipeline,
    registered_model_name="solacc_uc.mmf.moment_test",
    signature=signature,
    input_example=input_example,
    pip_requirements=[
      "git+https://github.com/moment-timeseries-foundation-model/moment.git",
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

model_name = "moment_test"
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


