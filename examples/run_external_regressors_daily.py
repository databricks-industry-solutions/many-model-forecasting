# Databricks notebook source
# MAGIC %pip install -r ../requirements-global.txt --quiet
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

dbutils.widgets.text("catalog", "")
dbutils.widgets.text("db", "")
dbutils.widgets.text("model", "")
dbutils.widgets.text("run_id", "")
dbutils.widgets.text("user", "")

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
    train_data=f"{catalog}.{db}.rossmann_daily_train",
    scoring_data=f"{catalog}.{db}.rossmann_daily_test",
    scoring_output=f"{catalog}.{db}.rossmann_daily_scoring_output",
    evaluation_output=f"{catalog}.{db}.rossmann_daily_evaluation_output",
    model_output=f"{catalog}.{db}",
    group_id="Store",
    date_col="Date",
    target="Sales",
    freq="D",
    dynamic_future_categorical=["DayOfWeek", "Open", "Promo", "SchoolHoliday"],
    prediction_length=10,
    backtest_length=30,
    stride=10,
    metric="smape",
    train_predict_ratio=1,
    active_models=[model],
    data_quality_check=True,
    resample=False,
    experiment_path=f"/Users/{user}/mmf/rossmann_daily",
    use_case_name="rossmann_daily",
    run_id=run_id,
    accelerator="gpu",
)
