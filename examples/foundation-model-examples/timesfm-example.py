# Databricks notebook source
# MAGIC %pip install -r ../../requirements.txt --quiet
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import sys
import subprocess
package = "git+https://github.com/google-research/timesfm.git"
subprocess.check_call([sys.executable, "-m", "pip", "install", package, "--quiet"])

# COMMAND ----------

import timesfm
tfm = timesfm.TimesFm(
    context_len=512,
    horizon_len=10,
    input_patch_len=32,
    output_patch_len=128,
    num_layers=20,
    model_dims=1280,
    backend="gpu",
)

# COMMAND ----------

tfm.load_from_checkpoint(repo_id="google/timesfm-1.0-200m")

# COMMAND ----------

import pandas as pd
df = spark.table('solacc_uc.mmf.m4_daily_train').toPandas()
forecast_df = tfm.forecast_on_df(
    inputs=df,
    freq="D",  # monthly
    value_name="y",
    num_jobs=-1,
)

# COMMAND ----------

display(forecast_df)
