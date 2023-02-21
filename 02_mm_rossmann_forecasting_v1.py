# Databricks notebook source
# MAGIC %pip install -r requirements.txt

# COMMAND ----------

import logging
logger = spark._jvm.org.apache.log4j
logging.getLogger("py4j.java_gateway").setLevel(logging.ERROR)
logging.getLogger("py4j.clientserver").setLevel(logging.ERROR)

# COMMAND ----------

import pathlib
import pandas as pd
from forecasting_sa import run_forecast

active_models = [
    #"StatsForecastBaselineWindowAverage",
    #"StatsForecastBaselineSeasonalWindowAverage",
    #"StatsForecastBaselineNaive",
    #"StatsForecastBaselineSeasonalNaive",
    "StatsForecastAutoArima",
    "StatsForecastAutoETS",
    "StatsForecastAutoCES",
    "StatsForecastAutoTheta",
    #"StatsForecastTSB",
    #"StatsForecastADIDA",
    #"StatsForecastIMAPA",
    #"StatsForecastCrostonClassic",
    #"StatsForecastCrostonOptimized",
    #"StatsForecastCrostonSBA",
    #"RFableArima",
    #"RFableETS",
    #"RFableNNETAR",
    #"RFableEnsemble",
    #"RDynamicHarmonicRegression",
    #"GluonTSSimpleFeedForward",
    #"GluonTSSeasonalNaive",
    #"GluonTSNBEATS",
    #"GluonTSDeepAR",
    #"GluonTSProphet",
    #"GluonTSTorchDeepAR",
    #"GluonTSTransformer",
    #"NeuralForecastMQNHiTS",
    #"SKTimeLgbmDsDt",
    #"SKTimeTBats",
]

run_id = run_forecast(
    spark=spark,
    # conf={"temp_path": f"{str(temp_dir)}/temp"},
    train_data="rossmann.rossmann_sales_and_stores_train",
    scoring_data="rossmann.rossmann_sales_and_stores_train",
    scoring_output="rossmann.rossmann_sales_forecast_out_all_stores",
    metrics_output="rossmann.forecasting_metrics_all_stores",
    group_id="Store",
    date_col="Date",
    target="Sales",
    freq="D",
    prediction_length=10,
    backtest_months=1,
    stride=10,
    train_predict_ratio=2,
    active_models=active_models,
    data_quality_check=False,
    ensemble=True,
    ensemble_metric="smape",
    ensemble_metric_avg=0.3,
    ensemble_metric_max=0.5,
    ensemble_scoring_output="rossmann.forecast_ensemble_out_all_stores",
    experiment_path=f"/Shared/fsa_rossmann_temp",
    use_case_name="fsa_rossmann",
)
print(run_id)

# COMMAND ----------

# MAGIC %sql
# MAGIC select run_id, run_date
# MAGIC from rossmann.forecasting_metrics_all_stores
# MAGIC group by run_id, run_date

# COMMAND ----------

# MAGIC %sql
# MAGIC select
# MAGIC   model,
# MAGIC   avg(metric_value)
# MAGIC from
# MAGIC   rossmann.forecasting_metrics_all_stores
# MAGIC   where
# MAGIC   run_id="a4dc77cb-fe72-4de2-a920-f321e812730f"
# MAGIC group by
# MAGIC   model
# MAGIC --having
# MAGIC  -- avg(metric_value) > 0.4

# COMMAND ----------

# MAGIC %sql select count(distinct Store) from rossmann.forecast_ensemble_out_100_stores where  run_id="a6e464f0-69d7-431b-a184-ae1f4042dbf3" 

# COMMAND ----------

import pandas as pd
from forecasting_sa.models import ModelRegistry
from omegaconf import OmegaConf
from sktime.performance_metrics.forecasting import mean_absolute_percentage_error

model_registry = ModelRegistry(
    OmegaConf.create(
        {
            "date_col": "ds",
            "target": "y",
            "freq": "D",
            "prediction_length": 10,
            "backtest_months": 1,
            "stride": 10,
            "active_models": ["SKTimeLgbmDsDt", "SKTimeTBats"],
        }
    )
)
model = model_registry.get_model("SKTimeLgbmDsDt")
print(model)
prediction_length = 10
sdf = spark.sql("select * from rossmann.rossmann_sales_and_stores_train_100_stores order by Date")
sdf = (
    sdf.withColumnRenamed("Date", "ds")
    .withColumnRenamed("Sales", "y")
    .withColumnRenamed("Store", "unique_id")
)
pdf = sdf.where("unique_id='1'").cache().toPandas()
pdf["ds"] = pd.to_datetime(pdf["ds"])
date_idx = pd.date_range(
    start=pdf["ds"].min(), end=pdf["ds"].max(), freq="D", name="ds"
)
pdf.set_index("ds", inplace=True)
pdf.sort_index(inplace=True)
pdf = pdf.reindex(date_idx, method="backfill").reset_index()
pdf = pdf.set_index("ds", drop=False)
pdf = pdf.fillna(0.1)
pdf["y"] = pdf.y.clip(0.1)
# pdf.index.max() - pd.DateOffset(days=prediction_length * 5)
#train_df = pdf[pdf.index < pd.Timestamp("2015-06-09")]
#val_df = pdf[pdf.index >= pd.Timestamp("2015-06-09")][:prediction_length]
#print(val_df.y)
#model.fit(train_df)
#res_df = model.predict(train_df)
#print(res_df)
#smape = mean_absolute_percentage_error(val_df.y.values, res_df.y.values, symmetric=True)
#print(smape)



# COMMAND ----------

model.backtest(df=pdf.reset_index(drop=True), start=pdf.index.max() - pd.DateOffset(months=1), retrain=True)

# COMMAND ----------

model.model.get_params()

# COMMAND ----------

# MAGIC %sql select count(distinct Store) from rossmann.forecast_ensemble_out_100_stores 

# COMMAND ----------

# MAGIC %sql select * from rossmann.rossmann_sales_forecast_out_100_stores

# COMMAND ----------

# MAGIC %sql select * from rossmann.forecast_ensemble_out_100_stores

# COMMAND ----------

# MAGIC 
# MAGIC 
# MAGIC %sql select model,  avg(metric_value) metric_value_avg
# MAGIC from rossmann.forecasting_metrics_100_stores 
# MAGIC where run_id="adc0b67c-a5f4-49d0-8e87-c97b0e54d428" 
# MAGIC group by model

# COMMAND ----------

# MAGIC %sql select model, Store, avg(metric_value) metric_value
# MAGIC from rossmann.forecasting_metrics_100_stores 
# MAGIC where run_id="adc0b67c-a5f4-49d0-8e87-c97b0e54d428" and Store>84 and metric_value<0.5
# MAGIC group by model, Store

# COMMAND ----------


