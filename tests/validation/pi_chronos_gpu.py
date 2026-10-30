# Databricks notebook source
# MAGIC %md
# MAGIC # Prediction intervals with Chronos on GPU
# MAGIC
# MAGIC Runs MMF with prediction intervals for `ChronosBoltSmall` and `Chronos2` on an M5 sample, on a GPU cluster,
# MAGIC through both foundation-model predict paths:
# MAGIC
# MAGIC * `distributed`: Pandas UDF per GPU (`serverless=False`), the default on classic GPU clusters;
# MAGIC * `driver`: driver-only predict (`serverless=True`), the path used on serverless GPU.
# MAGIC
# MAGIC For each, it checks that the backtest writes native quantiles, that calibration uses CQR, the
# MAGIC leave-one-window-out and held-out coverage, and that Chronos-Bolt's clamped 95% band (its trained quantiles
# MAGIC stop at 10-90%) is corrected. The experiment folder does not exist beforehand, which also checks that MMF
# MAGIC creates it. Results are returned with `dbutils.notebook.exit(json)`.

# COMMAND ----------

# MAGIC %pip install "mmf_sa @ git+https://github.com/lbruand-db/many-model-forecasting.git@feat/pi-p0-config" chronos-forecasting==2.2.2 utilsforecast==0.2.15 datasetsforecast==0.0.8 --quiet

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import json
import time
import uuid

import numpy as np
import pandas as pd
import torch
from datasetsforecast.m5 import M5
from pyspark.sql import functions as F

from mmf_sa import run_forecast

dbutils.widgets.text("n_series", "200")
dbutils.widgets.text("catalog", "main")
dbutils.widgets.text("schema", "lbruand_mmf_pi")
N_SERIES = int(dbutils.widgets.get("n_series"))
CATALOG, SCHEMA = dbutils.widgets.get("catalog"), dbutils.widgets.get("schema")
H, STRIDE, W, TEST_WINDOWS = 28, 14, 3, 4
MODELS = ["ChronosBoltSmall", "Chronos2"]
print("GPUs:", torch.cuda.device_count(), [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
assert torch.cuda.is_available(), "this notebook must run on a GPU cluster"

# COMMAND ----------

# MAGIC %md ## Data: an M5 sample, with the last 28 days before the test period held out

# COMMAND ----------

M5.download(directory="/tmp/m5")
m5_path = "/tmp/m5/m5/datasets"
calendar = pd.read_csv(f"{m5_path}/calendar.csv", usecols=["date"], parse_dates=["date"])
calendar["d"] = [f"d_{i + 1}" for i in range(len(calendar))]
wide = pd.read_csv(f"{m5_path}/sales_train_evaluation.csv")
wide["unique_id"] = wide["item_id"] + "_" + wide["store_id"]
wide = wide.sample(n=N_SERIES, random_state=7)
day_cols = [c for c in wide.columns if c.startswith("d_")]
Y = wide.melt(id_vars=["unique_id"], value_vars=day_cols, var_name="d", value_name="y")
Y = Y.merge(calendar, on="d").rename(columns={"date": "ds"}).drop(columns="d")
Y["y"] = Y["y"].astype(float)
Y = Y.sort_values(["unique_id", "ds"])
Y = Y[Y.groupby("unique_id")["y"].cumsum() > 0].reset_index(drop=True)
test_cutoff = Y["ds"].max() - pd.Timedelta(days=TEST_WINDOWS * H)
train = Y[Y["ds"] <= test_cutoff]
test = Y[(Y["ds"] > test_cutoff) & (Y["ds"] <= test_cutoff + pd.Timedelta(days=H))]
PREFIX = f"{CATALOG}.{SCHEMA}.chronos_gpu"
spark.createDataFrame(train).write.mode("overwrite").saveAsTable(f"{PREFIX}_train")
actual = {u: g.sort_values("ds")["y"].to_numpy() for u, g in test.groupby("unique_id")}
print(f"{train['unique_id'].nunique()} series, train to {test_cutoff.date()}")

# COMMAND ----------

# MAGIC %md ## Two runs: distributed (Pandas UDF per GPU) and driver-only predict

# COMMAND ----------

user = spark.sql("select current_user() as user").collect()[0]["user"]
experiment_root = f"/Users/{user}/mmf_pi_chronos_{uuid.uuid4().hex[:8]}"   # does not exist yet
results = {}
for mode, serverless in (("distributed", False), ("driver", True)):
    p = f"{PREFIX}_{mode}"
    for t in ("scoring_output", "evaluation_output", "evaluation_output_calibration",
              "evaluation_output_interval_metrics"):
        spark.sql(f"DROP TABLE IF EXISTS {p}_{t}")
    t0 = time.time()
    run_id = run_forecast(
        spark=spark, train_data=f"{PREFIX}_train", scoring_data=f"{PREFIX}_train",
        scoring_output=f"{p}_scoring_output", evaluation_output=f"{p}_evaluation_output",
        model_output=f"{CATALOG}.{SCHEMA}",
        group_id="unique_id", date_col="ds", target="y", freq="D",
        prediction_length=H, backtest_length=H + (W - 1) * STRIDE, stride=STRIDE, metric="smape",
        train_predict_ratio=1, active_models=MODELS, accelerator="gpu", serverless=serverless,
        experiment_path=f"{experiment_root}/{mode}", use_case_name=f"pi_chronos_{mode}",
        prediction_intervals={"enabled": True, "output_format": "both"},
    )
    runtime = round(time.time() - t0, 1)

    ev = spark.table(f"{p}_evaluation_output").filter((F.col("run_id") == run_id) & (F.col("metric_name") == "smape"))
    native = {r["model"]: (r["n"], r["with_q"]) for r in ev.groupBy("model").agg(
        F.count(F.lit(1)).alias("n"), F.count("forecast_quantiles").alias("with_q")).collect()}
    keys = {r["model"]: sorted(r["k"]) for r in ev.filter(F.col("forecast_quantiles").isNotNull())
            .select("model", F.map_keys("forecast_quantiles").alias("k")).dropDuplicates(["model"]).collect()}
    cal = spark.table(f"{p}_evaluation_output_calibration").filter(F.col("run_id") == run_id)
    methods = {r["model"]: f"{r['method']}/{r['score']}/{r['coverage_target']}" for r in
               cal.select("model", "method", "score", "coverage_target").distinct().collect()}
    sigma = {(r["model"], r["unique_id"]): r["sigma"] for r in
             cal.select("model", "unique_id", "sigma").distinct().collect()}
    lowo = {f"{r['model']} {r['metric_name']}": round(r["v"], 3) for r in
            spark.table(f"{p}_evaluation_output_interval_metrics").filter(
                (F.col("run_id") == run_id) & F.col("metric_name").rlike("^(native_)?coverage_"))
            .groupBy("model", "metric_name").agg(F.avg("metric_value").alias("v")).collect()}
    score = spark.table(f"{p}_scoring_output").filter(F.col("run_id") == run_id).toPandas()
    heldout = {}
    for model, g in score.groupby("model"):
        for level in (80, 95):
            a = 1 - level / 100
            cov, msis, nulls = [], [], 0
            for _, r in g.iterrows():
                y, lo, hi = actual.get(r["unique_id"]), r[f"y_lo_{level}"], r[f"y_hi_{level}"]
                if lo is None:
                    nulls += 1
                    continue
                if y is None or len(y) != H:
                    continue
                lo, hi = np.asarray(lo), np.asarray(hi)
                cov.append(np.mean((y >= lo) & (y <= hi)))
                msis.append(np.mean((hi - lo) + (2 / a) * (lo - y) * (y < lo) + (2 / a) * (y - hi) * (y > hi))
                            / sigma[(model, r["unique_id"])])
            heldout[f"{model} {level}"] = {"coverage": round(float(np.mean(cov)), 3),
                                          "msis": round(float(np.mean(msis)), 2), "series": len(cov), "null": nulls}
    results[mode] = {"run_id": run_id, "runtime_s": runtime, "backtest_rows_with_quantiles": native,
                     "quantile_keys": keys, "methods": methods, "lowo_coverage": lowo, "heldout": heldout}
    print(mode, json.dumps(results[mode], indent=1, default=str))

# COMMAND ----------

experiment_created = True
try:
    dbutils.fs.ls(f"file:/Workspace{experiment_root}")
except Exception:
    experiment_created = False
dbutils.notebook.exit(json.dumps({"gpus": torch.cuda.device_count(), "series": int(train["unique_id"].nunique()),
                                  "experiment_root_created": experiment_created, "results": results}, default=str))
