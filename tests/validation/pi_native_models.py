# Databricks notebook source
# MAGIC %md
# MAGIC # Native prediction intervals, model by model
# MAGIC
# MAGIC Runs MMF with prediction intervals on an M5 sample for each model of the `models` widget, with the model's
# MAGIC real library and weights, one model at a time, and checks for each:
# MAGIC
# MAGIC * the backtest: every row has native quantiles, with the expected keys, finite, non-crossing, and a median
# MAGIC   near the point forecast; how often bands are degenerate (zero width) and the 95% / 80% width ratio, which
# MAGIC   is about 1 when the model clamps the 95% levels to its native range;
# MAGIC * the point forecast: the same as in a run without prediction intervals;
# MAGIC * calibration: CQR is used, and the leave-one-window-out coverage before and after it;
# MAGIC * scoring: interval columns are filled, with the held-out coverage of the 28 days after the training data;
# MAGIC * the warnings MMF logs on the driver about native quantiles.
# MAGIC
# MAGIC `modes` lists the predict paths: `distributed` (`serverless=False`: Pandas UDFs where the model has them) and
# MAGIC `driver` (`serverless=True`: driver-only predict). `requirements` holds the lines of a requirements file,
# MAGIC separated by `|`. Results are returned with `dbutils.notebook.exit(json)`.

# COMMAND ----------

dbutils.widgets.text("requirements", "")
dbutils.widgets.text("models", "")
dbutils.widgets.text("modes", "distributed")
dbutils.widgets.text("accelerator", "gpu")
dbutils.widgets.text("n_series", "200")
dbutils.widgets.text("tag", "native")
dbutils.widgets.text("compare_point", "true")
dbutils.widgets.text("catalog", "main")
dbutils.widgets.text("schema", "lbruand_mmf_pi")
with open("/tmp/pi_native_requirements.txt", "w") as f:
    f.write("\n".join(line.strip() for line in dbutils.widgets.get("requirements").split("|") if line.strip()) + "\n")
print(open("/tmp/pi_native_requirements.txt").read())

# COMMAND ----------

# MAGIC %pip install -r /tmp/pi_native_requirements.txt datasetsforecast==0.0.8 --quiet

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import json
import logging
import time
import traceback

import numpy as np
import pandas as pd
from datasetsforecast.m5 import M5
from pyspark.sql import functions as F
from mmf_sa import run_forecast

MODELS = [m.strip() for m in dbutils.widgets.get("models").split(",") if m.strip()]
MODES = [m.strip() for m in dbutils.widgets.get("modes").split(",") if m.strip()]
ACCELERATOR = dbutils.widgets.get("accelerator")
N_SERIES = int(dbutils.widgets.get("n_series"))
TAG = dbutils.widgets.get("tag")
COMPARE_POINT = dbutils.widgets.get("compare_point") == "true"
CATALOG, SCHEMA = dbutils.widgets.get("catalog"), dbutils.widgets.get("schema")
H, STRIDE, W, TEST_WINDOWS = 28, 14, 3, 4
KEYS = ["0.025", "0.1", "0.5", "0.9", "0.975"]
env = {"models": MODELS, "modes": MODES, "accelerator": ACCELERATOR}
if ACCELERATOR == "gpu":
    import torch
    env["gpus"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    env["torch"] = torch.__version__
    assert torch.cuda.is_available(), "accelerator=gpu needs a GPU cluster"
print(env)

# COMMAND ----------

# MAGIC %md ## Data: an M5 sample, with the 28 days after the training data held out

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
PREFIX = f"{CATALOG}.{SCHEMA}.pi_{TAG}"
spark.createDataFrame(train).write.mode("overwrite").saveAsTable(f"{PREFIX}_train")
actual = {u: g.sort_values("ds")["y"].to_numpy() for u, g in test.groupby("unique_id")}
print(f"{train['unique_id'].nunique()} series, train to {test_cutoff.date()}")

# COMMAND ----------

# MAGIC %md ## Checks

# COMMAND ----------


class WarningCollector(logging.Handler):
    """Warnings logged on the driver by mmf_sa that mention quantiles, native bands or samples."""

    def __init__(self):
        super().__init__(logging.WARNING)
        self.messages = []

    def emit(self, record):
        msg = record.getMessage()
        if record.name.startswith("mmf_sa") and any(w in msg.lower() for w in ("quantile", "native", "sample")):
            self.messages.append(msg[:400])


def run(model, mode, intervals):
    p = f"{PREFIX}_{mode}" + ("" if intervals else "_nopi")
    kwargs = {"prediction_intervals": {"enabled": True, "output_format": "both"}} if intervals else {}
    t0 = time.time()
    run_id = run_forecast(
        spark=spark, train_data=f"{PREFIX}_train", scoring_data=f"{PREFIX}_train",
        scoring_output=f"{p}_scoring_output", evaluation_output=f"{p}_evaluation_output",
        model_output=f"{CATALOG}.{SCHEMA}",
        group_id="unique_id", date_col="ds", target="y", freq="D",
        prediction_length=H, backtest_length=H + (W - 1) * STRIDE, stride=STRIDE, metric="smape",
        train_predict_ratio=1, active_models=[model], accelerator=ACCELERATOR, serverless=mode == "driver",
        experiment_path=f"/Users/{USER}/mmf_pi_{TAG}/{mode}", use_case_name=f"pi_{TAG}_{mode}", **kwargs,
    )
    return p, run_id, round(time.time() - t0, 1)


def backtest_checks(ev):
    """Checks of the native quantiles in one model's backtest rows (pandas, one row per series and window)."""
    out = {"rows": len(ev), "rows_with_quantiles": int(ev["forecast_quantiles"].notna().sum())}
    q = ev[ev["forecast_quantiles"].notna()]
    if q.empty:
        return out
    maps = [dict(m) for m in q["forecast_quantiles"]]
    out["keys"] = sorted({k for m in maps for k in m}, key=float)
    keys = [k for k in KEYS if all(k in m for m in maps)]
    cube = np.stack([np.stack([np.asarray(m[k], dtype=float) for k in keys]) for m in maps])  # rows, keys, H
    point = np.stack([np.asarray(f, dtype=float) for f in q["forecast"]])
    out["finite_share"] = round(float(np.isfinite(cube).mean()), 4)
    out["non_crossing_share"] = round(float((np.diff(cube, axis=1) >= -1e-6).all(axis=1).mean()), 4)
    scale = np.abs(point).mean() + 1e-9
    if "0.5" in keys:
        out["median_minus_point_rel"] = round(float(np.abs(cube[:, keys.index("0.5")] - point).mean() / scale), 4)
    widths = {}
    for lo, hi in (("0.1", "0.9"), ("0.025", "0.975")):
        if lo in keys and hi in keys:
            w = cube[:, keys.index(hi)] - cube[:, keys.index(lo)]
            widths[hi] = w
            out[f"degenerate_share_{lo}_{hi}"] = round(float((w <= 1e-9).mean()), 4)
            out[f"point_outside_{lo}_{hi}_share"] = round(float(
                ((point < cube[:, keys.index(lo)] - 1e-6) | (point > cube[:, keys.index(hi)] + 1e-6)).mean()), 4)
    if len(widths) == 2:
        out["width_ratio_95_80"] = round(float(widths["0.975"].mean() / (widths["0.9"].mean() + 1e-12)), 3)
    return out


def point_diff(p, run_id, p_nopi, run_id_nopi, model):
    def forecasts(table, rid):
        rows = spark.table(table).filter((F.col("run_id") == rid) & (F.col("model") == model)
                                         & (F.col("metric_name") == "smape")).select(
            "unique_id", "backtest_window_start_date", "forecast").toPandas()
        return {(r.unique_id, r.backtest_window_start_date): np.asarray(r.forecast, dtype=float)
                for r in rows.itertuples()}
    a, b = forecasts(f"{p}_evaluation_output", run_id), forecasts(f"{p_nopi}_evaluation_output", run_id_nopi)
    common = sorted(set(a) & set(b))
    diffs = np.array([np.abs(a[k] - b[k]).max() for k in common]) if common else np.array([np.nan])
    return {"rows_compared": len(common), "max_abs_diff": float(np.nanmax(diffs)),
            "rows_different": int((diffs > 1e-6).sum()),
            "mean_abs_point": round(float(np.mean([np.abs(a[k]).mean() for k in common])), 4) if common else None}


def interval_checks(p, run_id, model):
    cal_rows = spark.table(f"{p}_evaluation_output_calibration").filter(
        (F.col("run_id") == run_id) & (F.col("model") == model))
    out = {"methods": sorted(f"{r['method']}/{r['score']}/{r['coverage_target']}" for r in
                             cal_rows.select("method", "score", "coverage_target").distinct().collect())}
    sigma = {r["unique_id"]: r["sigma"] for r in cal_rows.select("unique_id", "sigma").distinct().collect()}
    out["lowo_coverage"] = {r["metric_name"]: round(r["v"], 3) for r in
                            spark.table(f"{p}_evaluation_output_interval_metrics").filter(
                                (F.col("run_id") == run_id) & (F.col("model") == model)
                                & F.col("metric_name").rlike("^(native_)?coverage_"))
                            .groupBy("metric_name").agg(F.avg("metric_value").alias("v")).collect()}
    score = spark.table(f"{p}_scoring_output").filter((F.col("run_id") == run_id) & (F.col("model") == model)).toPandas()
    out["scoring_rows"] = len(score)
    out["scoring_quantile_keys"] = sorted({k for m in score["y_quantiles"] if m is not None for k in dict(m)}, key=float)
    for level in (80, 95):
        a = 1 - level / 100
        cov, msis, nulls = [], [], 0
        for _, r in score.iterrows():
            y, lo, hi = actual.get(r["unique_id"]), r[f"y_lo_{level}"], r[f"y_hi_{level}"]
            if lo is None or hi is None:
                nulls += 1
                continue
            if y is None or len(y) != H or r["unique_id"] not in sigma:
                continue
            lo, hi = np.asarray(lo), np.asarray(hi)
            cov.append(np.mean((y >= lo) & (y <= hi)))
            msis.append(np.mean((hi - lo) + (2 / a) * (lo - y) * (y < lo) + (2 / a) * (y - hi) * (y > hi))
                        / sigma[r["unique_id"]])
        out[f"heldout_{level}"] = {"coverage": round(float(np.mean(cov)), 3) if cov else None,
                                   "msis": round(float(np.mean(msis)), 2) if msis else None,
                                   "series": len(cov), "null": nulls}
    return out


# COMMAND ----------

# MAGIC %md ## One model at a time

# COMMAND ----------

USER = spark.sql("select current_user() as user").collect()[0]["user"]
for mode in MODES:
    for suffix in ("", "_nopi"):
        for t in ("scoring_output", "evaluation_output", "evaluation_output_calibration",
                  "evaluation_output_interval_metrics"):
            spark.sql(f"DROP TABLE IF EXISTS {PREFIX}_{mode}{suffix}_{t}")

collector = WarningCollector()
logging.getLogger().addHandler(collector)
results = {}
for model in MODELS:
    results[model] = {}
    for i, mode in enumerate(MODES):
        res = {}
        collector.messages = []
        try:
            p, run_id, res["runtime_s"] = run(model, mode, intervals=True)
            res["run_id"] = run_id
            ev = spark.table(f"{p}_evaluation_output").filter(
                (F.col("run_id") == run_id) & (F.col("model") == model) & (F.col("metric_name") == "smape")
            ).select("unique_id", "backtest_window_start_date", "forecast", "forecast_quantiles").toPandas()
            res["backtest"] = backtest_checks(ev)
            res["intervals"] = interval_checks(p, run_id, model)
            res["warnings"] = sorted(set(collector.messages))
            if COMPARE_POINT and i == 0:
                p_nopi, run_id_nopi, _ = run(model, mode, intervals=False)
                res["point_vs_no_intervals"] = point_diff(p, run_id, p_nopi, run_id_nopi, model)
        except Exception as err:
            res["error"] = f"{type(err).__name__}: {err}"[:1500]
            res["traceback"] = traceback.format_exc()[-3000:]
        results[model][mode] = res
        print(model, mode, json.dumps(res, indent=1, default=str))
logging.getLogger().removeHandler(collector)

# COMMAND ----------

dbutils.notebook.exit(json.dumps({"env": env, "series": int(train["unique_id"].nunique()), "results": results},
                                 default=str))
