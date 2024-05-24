# Databricks notebook source
# MAGIC %pip install -r ../requirements.txt --quiet
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

dbutils.widgets.text("catalog", "")
dbutils.widgets.text("db", "")
dbutils.widgets.text("model", "")

catalog = dbutils.widgets.get("catalog")
db = dbutils.widgets.get("db")
model = dbutils.widgets.get("model")

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
    dynamic_future=["DayOfWeek", "Open", "Promo", "SchoolHoliday"],
    prediction_length=10,
    backtest_months=1,
    stride=10,
    train_predict_ratio=2,
    active_models=[model],
    data_quality_check=True,
    resample=False,
    ensemble=True,
    ensemble_metric="smape",
    ensemble_metric_avg=0.3,
    ensemble_metric_max=0.5,
    ensemble_scoring_output=f"{catalog}.{db}.rossmann_daily_ensemble_output",
    experiment_path=f"/Shared/mmf_rossmann",
    use_case_name="rossmann_daily",
    accelerator="gpu",
)
