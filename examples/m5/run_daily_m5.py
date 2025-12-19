# Databricks notebook source
dbutils.widgets.text("catalog", "")
dbutils.widgets.text("db", "")
dbutils.widgets.text("model", "")
dbutils.widgets.text("run_id", "")
dbutils.widgets.text("table", "")
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

run_forecast(
    spark=spark,
    train_data=f"{catalog}.{db}.{table}",
    scoring_data=f"{catalog}.{db}.{table}",
    scoring_output=f"{catalog}.{db}.daily_scoring_output",
    evaluation_output=f"{catalog}.{db}.daily_evaluation_output",
    model_output=f"{catalog}.{db}",
    group_id="unique_id",
    date_col="ds",
    target="y",
    freq="D",
    prediction_length=28,
    backtest_length=90,
    stride=7,
    metric="smape",
    train_predict_ratio=1,
    data_quality_check=True,
    resample=False,
    active_models=[model],
    experiment_path=f"/Users/{user}/mmf/m5_daily",
    use_case_name="m5_daily",
    run_id=run_id,
    accelerator="gpu",
)
