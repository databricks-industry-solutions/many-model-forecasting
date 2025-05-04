# Databricks notebook source
dbutils.widgets.text("catalog", "")
dbutils.widgets.text("db", "")
dbutils.widgets.text("model", "")
dbutils.widgets.text("run_id", "")
dbutils.widgets.text("user", "")

# COMMAND ----------

model_class = "global" if "NeuralForecast" in dbutils.widgets.get("model") else "foundation" 
%pip install -r ../requirements-{model_class}.txt --quiet
dbutils.library.restartPython()

# COMMAND ----------

catalog = dbutils.widgets.get("catalog")
db = dbutils.widgets.get("db")
model = dbutils.widgets.get("model")
run_id = dbutils.widgets.get("run_id")
user = dbutils.widgets.get("user")

# COMMAND ----------

from mmf_sa import run_forecast
import logging
logger = spark._jvm.org.apache.log4j
logging.getLogger("py4j.java_gateway").setLevel(logging.ERROR)
logging.getLogger("py4j.clientserver").setLevel(logging.ERROR)


run_forecast(
    spark=spark,
    train_data=f"{catalog}.{db}.m4_hourly_train",
    scoring_data=f"{catalog}.{db}.m4_hourly_train",
    scoring_output=f"{catalog}.{db}.hourly_scoring_output",
    evaluation_output=f"{catalog}.{db}.hourly_evaluation_output",
    model_output=f"{catalog}.{db}",
    group_id="unique_id",
    date_col="ds",
    target="y",
    freq="H",
    prediction_length=24,
    backtest_length=168,
    stride=24,
    metric="smape",
    train_predict_ratio=1,
    data_quality_check=True,
    resample=False,
    active_models=[model],
    experiment_path=f"/Users/{user}/mmf/m4_hourly",
    use_case_name="m4_hourly",
    run_id=run_id,
    accelerator="gpu",
)
