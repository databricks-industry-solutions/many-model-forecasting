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

n = 1000  # Number of items: choose from [1000, 10000, 'full']. full is 35k
taining_table = f"daily_train_{n}"
user_email = spark.sql('select current_user() as user').collect()[0]['user']

run_forecast(
    spark=spark,
    train_data=f"{catalog}.{db}.{taining_table}",
    scoring_data=f"{catalog}.{db}.{taining_table}",
    scoring_output=f"{catalog}.{db}.daily_scoring_output",
    evaluation_output=f"{catalog}.{db}.daily_evaluation_output",
    model_output=f"{catalog}.{db}",
    group_id="unique_id",
    date_col="ds",
    target="y",
    freq="D",
    prediction_length=28,
    backtest_months=3,
    stride=7,
    train_predict_ratio=1,
    data_quality_check=True,
    resample=False,
    active_models=[model],
    experiment_path=f"/Users/{user_email}/mmf/m5",
    use_case_name="m5_daily",
    run_id=run_id,
    accelerator="gpu",
)
