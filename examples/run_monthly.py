# Databricks notebook source
# MAGIC %pip install -r ../requirements.txt --quiet
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

dbutils.widgets.text("catalog", "")
dbutils.widgets.text("db", "")
dbutils.widgets.text("model", "")
dbutils.widgets.text("run_id", "")

catalog = dbutils.widgets.get("catalog")
db = dbutils.widgets.get("db")
model = dbutils.widgets.get("model")
run_id = dbutils.widgets.get("run_id")

# COMMAND ----------

from mmf_sa import run_forecast
import logging
logger = spark._jvm.org.apache.log4j
logging.getLogger("py4j.java_gateway").setLevel(logging.ERROR)
logging.getLogger("py4j.clientserver").setLevel(logging.ERROR)


run_forecast(
    spark=spark,
    train_data=f"{catalog}.{db}.m4_monthly_train",
    scoring_data=f"{catalog}.{db}.m4_monthly_train",
    scoring_output=f"{catalog}.{db}.monthly_scoring_output",
    evaluation_output=f"{catalog}.{db}.monthly_evaluation_output",
    model_output=f"{catalog}.{db}",
    group_id="unique_id",
    date_col="date",
    target="y",
    freq="M",
    prediction_length=3,
    backtest_months=12,
    stride=1,
    metric="smape",
    train_predict_ratio=1,
    data_quality_check=True,
    resample=False,
    active_models=[model],
    experiment_path=f"/Shared/mmf_experiment_monthly",
    use_case_name="m4_monthly",
    run_id=run_id,
    accelerator="gpu",
)
