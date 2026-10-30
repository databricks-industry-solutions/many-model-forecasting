# Databricks notebook source
# MAGIC %md
# MAGIC # Prediction-interval scale test: calibration and scoring only (task 7, spec section 11.6)
# MAGIC
# MAGIC The backtest of local models runs one statsforecast fit per series and window, which takes hours at
# MAGIC 1M series and does not test the prediction-interval code. This notebook writes **synthetic backtest rows**
# MAGIC for `n_series` series directly to `evaluation_output` (3 windows, h = 7), for one model with native 80/95%
# MAGIC bands (labelled `StatsForecastAutoETS`, so CQR) and one without (labelled `StatsForecastCrostonSBA`, so
# MAGIC residual conformal). It then times `Forecaster.run_calibration` and the scoring step
# MAGIC (`write_scoring_output`) on synthetic forecasts, and checks the output row counts. Everything is
# MAGIC generated in Spark, so the driver only holds what MMF itself collects.

# COMMAND ----------

# MAGIC %pip install "mmf_sa[local] @ git+https://github.com/lbruand-db/many-model-forecasting.git@feat/pi-p0-config" --quiet

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import json
import math
import time
from datetime import datetime

from omegaconf import OmegaConf
from pyspark.sql import functions as F
from pyspark.sql.types import ArrayType, BinaryType, DoubleType, MapType, StringType

from mmf_sa import Forecaster

dbutils.widgets.text("n_series", "1000000")
dbutils.widgets.text("catalog", "main")
dbutils.widgets.text("schema", "mmf_pi_validation")
N = int(dbutils.widgets.get("n_series"))
CATALOG, SCHEMA = dbutils.widgets.get("catalog"), dbutils.widgets.get("schema")
PREFIX = f"{CATALOG}.{SCHEMA}.scale_{N}"
H, W, HISTORY = 7, 3, 60
DAYS = HISTORY + W * H + H            # history, backtest windows, then the forecast horizon
NATIVE, RESIDUAL = "StatsForecastAutoETS", "StatsForecastCrostonSBA"
RUN_ID = f"scale-{N}-{int(time.time())}"
Z = {80: 1.2816, 95: 1.96}
timings = {}

# COMMAND ----------

# MAGIC %md ## Synthetic data, generated in Spark

# COMMAND ----------

t0 = time.time()
series = spark.range(N).select(F.format_string("s%07d", "id").alias("unique_id"),
                               (F.lit(0.5) + (F.col("id") % 50) / 5).alias("lam"))
days = spark.range(DAYS).select(F.col("id").cast("int").alias("t"))
mu = F.col("lam") * (1 + 0.5 * F.sin(2 * math.pi * F.col("t") / 7))
daily = (series.crossJoin(days)
         .withColumn("mu", mu)
         .withColumn("y", F.greatest(F.lit(0.0), F.round(F.col("mu") + F.sqrt("mu") * F.randn(7))))
         .withColumn("ds", F.to_timestamp(F.date_add(F.lit("2023-01-01"), F.col("t")))))
daily.filter(F.col("t") < HISTORY + W * H).select("unique_id", "ds", "y").write.mode("overwrite") \
    .saveAsTable(f"{PREFIX}_train")
daily = spark.table(f"{PREFIX}_train").join(series, "unique_id") \
    .withColumn("t", F.datediff(F.col("ds"), F.lit("2023-01-01")))
timings["generate_train"] = round(time.time() - t0, 1)


def arrays(frame, key_cols):
    """One row per key with yhat / actual arrays ordered by t."""
    ordered = frame.groupBy(*key_cols).agg(F.array_sort(F.collect_list(F.struct("t", "yhat", "y"))).alias("_r"))
    return ordered.select(*key_cols, F.col("_r.yhat").alias("forecast"), F.col("_r.y").alias("actual"))


def quantile_map(forecast, lam):
    keys, values = [], []
    for level, z in Z.items():
        a = (100 - level) / 200
        keys += [f"{a:g}", f"{1 - a:g}"]
        values += [F.transform(forecast, lambda x: x - z * F.sqrt(lam) * 0.9),
                   F.transform(forecast, lambda x: x + z * F.sqrt(lam) * 0.9)]
    return F.map_from_arrays(F.array(*[F.lit(k) for k in keys]), F.array(*values))


t0 = time.time()
bt = (daily.filter(F.col("t") >= HISTORY)
      .withColumn("window", ((F.col("t") - HISTORY) / H).cast("int"))
      .withColumn("yhat", F.col("lam") * (1 + 0.5 * F.sin(2 * math.pi * F.col("t") / 7))))
windows = arrays(bt, ["unique_id", "lam", "window"]).withColumn(
    "backtest_window_start_date",
    F.to_timestamp(F.date_add(F.lit("2023-01-01"), (F.lit(HISTORY) + F.col("window") * H).cast("int"))))
run_date = datetime.now()
base = (windows.withColumn("metric_name", F.lit("smape")).withColumn("metric_value", F.lit(0.0))
        .withColumn("model_pickle", F.lit(None).cast(BinaryType()))
        .withColumn("run_id", F.lit(RUN_ID)).withColumn("run_date", F.lit(run_date))
        .withColumn("use_case", F.lit("scale")).withColumn("model_uri", F.lit("")))
cols = ["unique_id", "backtest_window_start_date", "metric_name", "metric_value", "forecast", "actual",
        "model_pickle", "forecast_quantiles", "run_id", "run_date", "model", "use_case"]
native_rows = base.withColumn("forecast_quantiles", quantile_map(F.col("forecast"), F.col("lam"))) \
    .withColumn("model", F.lit(NATIVE))
residual_rows = base.withColumn("forecast_quantiles",
                                F.lit(None).cast(MapType(StringType(), ArrayType(DoubleType())))) \
    .withColumn("model", F.lit(RESIDUAL))
for t in ("evaluation_output", "evaluation_output_calibration", "evaluation_output_interval_metrics",
          "scoring_output"):
    spark.sql(f"DROP TABLE IF EXISTS {PREFIX}_{t}")
native_rows.select(*cols, "model_uri").unionByName(residual_rows.select(*cols, "model_uri")) \
    .write.mode("overwrite").saveAsTable(f"{PREFIX}_evaluation_output")
timings["generate_backtest"] = round(time.time() - t0, 1)
print(timings)

# COMMAND ----------

# MAGIC %md ## Calibration

# COMMAND ----------

conf = OmegaConf.create({
    "group_id": "unique_id", "date_col": "ds", "target": "y", "freq": "D", "prediction_length": H,
    "backtest_length": W * H, "stride": H, "metric": "smape", "use_case_name": "scale",
    "train_data": f"{PREFIX}_train", "evaluation_output": f"{PREFIX}_evaluation_output",
    "scoring_output": f"{PREFIX}_scoring_output", "active_models": [NATIVE, RESIDUAL],
    "prediction_intervals": {"enabled": True, "output_format": "both"},
})
f = Forecaster(conf=conf, data_conf={}, spark=spark, experiment_id="scale-test", run_id=RUN_ID)
f.run_date = run_date
t0 = time.time()
f.run_calibration()
timings["calibration"] = round(time.time() - t0, 1)
print(timings)

# COMMAND ----------

# MAGIC %md ## Scoring (applying the calibration to synthetic forecasts)

# COMMAND ----------

future = (daily.select("unique_id", "lam").distinct().crossJoin(
    spark.range(HISTORY + W * H, DAYS).select(F.col("id").cast("int").alias("t")))
    .withColumn("yhat", F.col("lam") * (1 + 0.5 * F.sin(2 * math.pi * F.col("t") / 7)))
    .withColumn("ds", F.to_timestamp(F.date_add(F.lit("2023-01-01"), F.col("t")))))
fc = future.groupBy("unique_id", "lam").agg(F.array_sort(F.collect_list(F.struct("t", "ds", "yhat"))).alias("_r")) \
    .select("unique_id", "lam", F.col("_r.ds").alias("ds"), F.col("_r.yhat").alias("y"))
for model, with_q in ((NATIVE, True), (RESIDUAL, False)):
    sdf = fc.withColumn("forecast_quantiles", quantile_map(F.col("y"), F.col("lam")) if with_q else
                        F.lit(None).cast(MapType(StringType(), ArrayType(DoubleType())))) \
        .drop("lam").withColumn("model_pickle", F.lit(None).cast(BinaryType())) \
        .withColumn("run_id", F.lit(RUN_ID)).withColumn("run_date", F.lit(run_date)) \
        .withColumn("use_case", F.lit("scale")).withColumn("model", F.lit(model)).withColumn("model_uri", F.lit(""))
    t0 = time.time()
    f.write_scoring_output(sdf, model)
    timings[f"scoring_{model}"] = round(time.time() - t0, 1)
print(timings)

# COMMAND ----------

cal = spark.table(f"{PREFIX}_evaluation_output_calibration").filter(F.col("run_id") == RUN_ID)
met = spark.table(f"{PREFIX}_evaluation_output_interval_metrics").filter(F.col("run_id") == RUN_ID)
score = spark.table(f"{PREFIX}_scoring_output").filter(F.col("run_id") == RUN_ID)
checks = {
    "calibration_rows": {r["model"]: r["count"] for r in cal.groupBy("model").count().collect()},
    "calibration_rows_expected_per_model": N * H * 2,
    "metric_rows": {r["model"]: r["count"] for r in met.groupBy("model").count().collect()},
    "scoring_rows": {r["model"]: r["count"] for r in score.groupBy("model").count().collect()},
    "scoring_null_intervals": {r["model"]: r["count"] for r in
                               score.filter(F.col("y_quantiles").isNull()).groupBy("model").count().collect()},
    "methods": {r["model"]: f"{r['method']}/{r['score']}/{r['coverage_target']}" for r in
                cal.select("model", "method", "score", "coverage_target").distinct().collect()},
    "coverage": {f"{r['model']} {r['metric_name']}": round(r["v"], 4) for r in
                 met.filter(F.col("metric_name").startswith("coverage_")).groupBy("model", "metric_name")
                 .agg(F.avg("metric_value").alias("v")).collect()},
}
print(json.dumps(checks, indent=1))
dbutils.notebook.exit(json.dumps({"n_series": N, "run_id": RUN_ID, "timings": timings, "checks": checks}))
