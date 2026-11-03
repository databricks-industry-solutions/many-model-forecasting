# Databricks notebook source
# MAGIC %md
# MAGIC # Held-out coverage of calibrated prediction intervals
# MAGIC
# MAGIC Holds out the last horizon of each series of a dataset, runs `run_forecast` with native prediction intervals on
# MAGIC the rest, calibrates the intervals with `calibrate_intervals` in three configurations, and scores every
# MAGIC configuration on the held-out horizon:
# MAGIC
# MAGIC * `default`: one marginal pool per horizon step, each series' tail quantiles shrunk toward it (CQR pooled over
# MAGIC   all series);
# MAGIC * `abc_xyz`: the same, pooled per ABC x XYZ class (`pool="group", group_col="auto"`);
# MAGIC * `series_scale`: one pool, each series' scale (mean |e|) shrunk toward it instead of its tail quantiles.
# MAGIC
# MAGIC For each model and level: held-out coverage, MSIS (scaled by each series' seasonal-naive error) and width, overall,
# MAGIC by decile of the series' scale (sigma) and by ABC x XYZ class, next to the raw native bands. `dataset` is `m5`
# MAGIC (daily Walmart sales, retail) or `m4_monthly` (M4 monthly: macro, micro, finance, industry, demographic).
# MAGIC Results are returned with `dbutils.notebook.exit(json)`.

# COMMAND ----------

dbutils.widgets.text("requirements", "")
dbutils.widgets.text("dataset", "m5")
dbutils.widgets.text("n_series", "2000")
dbutils.widgets.text("windows", "3")
dbutils.widgets.text("catalog", "main")
dbutils.widgets.text("schema", "mmf_pi_validation")
with open("/tmp/pi_cov_requirements.txt", "w") as f:
    f.write("\n".join(line.strip() for line in dbutils.widgets.get("requirements").split("|") if line.strip()) + "\n")

# COMMAND ----------

# MAGIC %pip install -r /tmp/pi_cov_requirements.txt datasetsforecast==0.0.8 --quiet

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import json
import pathlib
import time

import numpy as np
import pandas as pd
from pyspark.sql import functions as F

from mmf_sa import calibrate_intervals, run_forecast

DATASET = dbutils.widgets.get("dataset")
N_SERIES, W = int(dbutils.widgets.get("n_series")), int(dbutils.widgets.get("windows"))
CATALOG, SCHEMA = dbutils.widgets.get("catalog"), dbutils.widgets.get("schema")
PREFIX = f"{CATALOG}.{SCHEMA}.pi_cov_{DATASET}_w{W}"
USER = spark.sql("select current_user() as user").collect()[0]["user"]
LEVELS = [80, 95]
SETTINGS = {
    "m5": {"freq": "D", "h": 28, "stride": 28,
           "models": ["StatsForecastAutoETS", "StatsForecastBaselineSeasonalNaive", "StatsForecastCrostonSBA"]},
    "m4_monthly": {"freq": "M", "h": 18, "stride": 6,
                   "models": ["StatsForecastAutoETS", "StatsForecastBaselineSeasonalNaive",
                              "StatsForecastBaselineWindowAverage"]},
}[DATASET]
H = SETTINGS["h"]
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")

# COMMAND ----------

# MAGIC %md ## Data, with the last horizon of each series held out

# COMMAND ----------

if DATASET == "m5":
    from datasetsforecast.m5 import M5
    M5.download(directory="/tmp/m5")
    path = "/tmp/m5/m5/datasets"
    calendar = pd.read_csv(f"{path}/calendar.csv", usecols=["date"], parse_dates=["date"])
    calendar["d"] = [f"d_{i + 1}" for i in range(len(calendar))]
    wide = pd.read_csv(f"{path}/sales_train_evaluation.csv")
    wide["unique_id"] = wide["item_id"] + "_" + wide["store_id"]
    wide = wide.sample(n=N_SERIES, random_state=7)
    day_cols = [c for c in wide.columns if c.startswith("d_")]
    Y = wide.melt(id_vars=["unique_id"], value_vars=day_cols, var_name="d", value_name="y")
    Y = Y.merge(calendar, on="d").rename(columns={"date": "ds"}).drop(columns="d")
    Y["y"] = Y["y"].astype(float)
    Y = Y.sort_values(["unique_id", "ds"])
    Y = Y[Y.groupby("unique_id")["y"].cumsum() > 0]  # drop leading zeros
else:
    from datasetsforecast.m4 import M4
    raw, _, _ = M4.load(directory=str(pathlib.Path.home()), group="Monthly")
    lengths = raw.groupby("unique_id").size()
    ids = lengths[lengths >= 96].sample(n=N_SERIES, random_state=7).index
    frames = []
    for uid, g in raw[raw["unique_id"].isin(ids)].groupby("unique_id"):
        g = g.sort_values("ds").iloc[-144:]
        frames.append(pd.DataFrame({"unique_id": uid, "y": g["y"].to_numpy(dtype=float),
                                    "ds": pd.date_range(end="2019-12-31", periods=len(g), freq="ME")}))
    Y = pd.concat(frames, ignore_index=True)
Y = Y.reset_index(drop=True)
step = Y.groupby("unique_id").cumcount(ascending=False)  # 0 for the last observation
train, test = Y[step >= H], Y[step < H]
spark.createDataFrame(train[["unique_id", "ds", "y"]]).write.mode("overwrite").saveAsTable(f"{PREFIX}_train")
actual = {u: g.sort_values("ds")["y"].to_numpy() for u, g in test.groupby("unique_id")}
print(f"{DATASET}: {train['unique_id'].nunique()} series, horizon {H}, {W} backtest windows")

# COMMAND ----------

# MAGIC %md ## Forecasts with native intervals

# COMMAND ----------

for t in ("evaluation_output", "scoring_output"):
    spark.sql(f"DROP TABLE IF EXISTS {PREFIX}_{t}")
t0 = time.time()
run_id = run_forecast(
    spark=spark, train_data=f"{PREFIX}_train", scoring_data=f"{PREFIX}_train",
    scoring_output=f"{PREFIX}_scoring_output", evaluation_output=f"{PREFIX}_evaluation_output",
    group_id="unique_id", date_col="ds", target="y", freq=SETTINGS["freq"], prediction_length=H,
    backtest_length=H + (W - 1) * SETTINGS["stride"], stride=SETTINGS["stride"], metric="smape",
    train_predict_ratio=1, active_models=SETTINGS["models"],
    experiment_path=f"/Users/{USER}/mmf_pi_cov/{DATASET}", use_case_name=f"pi_cov_{DATASET}",
    prediction_intervals={"enabled": True, "levels": LEVELS, "output_format": "both"},
)
timings = {"run_forecast": round(time.time() - t0, 1)}

# COMMAND ----------

# MAGIC %md ## Calibration, three configurations

# COMMAND ----------

CONFIGS = {
    "default": {},
    "abc_xyz": {"pool": "group", "group_col": "auto"},
    "series_scale": {"settings": {"coverage": {"target": "series_scale"}}},
}
calibration = {}
for name, kwargs in CONFIGS.items():
    for t in ("evaluation_output_calibration", "evaluation_output_interval_metrics", "scoring_output_intervals"):
        spark.sql(f"DROP TABLE IF EXISTS {PREFIX}_{name}_{t}")
    t0 = time.time()
    out = calibrate_intervals(
        spark, train_data=f"{PREFIX}_train", evaluation_output=f"{PREFIX}_evaluation_output",
        scoring_output=f"{PREFIX}_scoring_output", output_prefix=f"{PREFIX}_{name}", freq=SETTINGS["freq"],
        levels=LEVELS, output_format="lo_hi", **kwargs)
    timings[f"calibrate_{name}"] = round(time.time() - t0, 1)
    calibration[name] = {m: {k: s.get(k) for k in ("methods", "coverage", "error")} for m, s in out["models"].items()}
print(json.dumps(calibration, indent=1))

# COMMAND ----------

# MAGIC %md ## Held-out scores

# COMMAND ----------

classes = (spark.table(f"{PREFIX}_default_evaluation_output_calibration")
           .select("unique_id", "sigma", F.concat("abc_class", "xyz_class").alias("abc_xyz")).distinct().toPandas()
           .drop_duplicates("unique_id").set_index("unique_id"))
classes["sigma_decile"] = pd.qcut(classes["sigma"].rank(method="first"), 10, labels=False) + 1


def scores(frame, lo_col, hi_col, level):
    """Per-series held-out coverage, MSIS (scaled by sigma) and width of one level's bounds."""
    a = 1 - level / 100
    rows = []
    for r in frame.itertuples(index=False):
        y, lo, hi = actual.get(r.unique_id), getattr(r, lo_col), getattr(r, hi_col)
        if y is None or lo is None or hi is None or len(y) != H or r.unique_id not in classes.index:
            continue
        lo, hi, s = np.asarray(lo), np.asarray(hi), classes.at[r.unique_id, "sigma"]
        rows.append({"unique_id": r.unique_id, "model": r.model, "coverage": np.mean((y >= lo) & (y <= hi)),
                     "msis": np.mean((hi - lo) + 2 / a * np.maximum(lo - y, 0) + 2 / a * np.maximum(y - hi, 0)) / s,
                     "width": np.mean(hi - lo) / s})
    return pd.DataFrame(rows)


def summarize(per_series):
    if per_series.empty:
        return {}
    df = per_series.join(classes, on="unique_id")
    out = {}
    for model, g in df.groupby("model"):
        out[model] = {
            "series": int(len(g)),
            "overall": {k: round(float(g[k].mean()), 4) for k in ("coverage", "msis", "width")},
            "coverage_by_sigma_decile": {int(d): round(float(v), 3) for d, v in g.groupby("sigma_decile")["coverage"].mean().items()},
            "coverage_by_abc_xyz": {c: [round(float(v["coverage"].mean()), 3), int(len(v))] for c, v in g.groupby("abc_xyz")},
        }
    return out


native = spark.table(f"{PREFIX}_scoring_output").filter(F.col("run_id") == run_id).select(
    "unique_id", "model", *[f"y_{s}_{lv}" for lv in LEVELS for s in ("lo", "hi")]).toPandas()
results = {"native": {lv: summarize(scores(native, f"y_lo_{lv}", f"y_hi_{lv}", lv)) for lv in LEVELS}}
for name in CONFIGS:
    cal = spark.table(f"{PREFIX}_{name}_scoring_output_intervals").select(
        "unique_id", "model", *[f"y_{s}_{lv}" for lv in LEVELS for s in ("lo", "hi")]).toPandas()
    results[name] = {lv: summarize(scores(cal, f"y_lo_{lv}", f"y_hi_{lv}", lv)) for lv in LEVELS}

for lv in LEVELS:
    print(f"--- level {lv}: held-out coverage / msis")
    for name, res in results.items():
        print(name.ljust(8), {m: (v["overall"]["coverage"], v["overall"]["msis"]) for m, v in res[lv].items()})

# COMMAND ----------

dbutils.notebook.exit(json.dumps({
    "dataset": DATASET, "series": int(train["unique_id"].nunique()), "h": H, "windows": W, "run_id": run_id,
    "timings": timings, "calibration": calibration, "heldout": results}, default=str))
