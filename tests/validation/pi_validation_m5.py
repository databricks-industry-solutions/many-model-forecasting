# Databricks notebook source
# MAGIC %md
# MAGIC # Prediction-interval validation on M5
# MAGIC
# MAGIC Runs MMF with prediction intervals on the experiment's M5 sample (same seed, same series), and checks:
# MAGIC
# MAGIC 1. **Held-out coverage.** MMF is trained on data up to the experiment's test cutoff, with backtest
# MAGIC    windows equal to the experiment's last `W` calibration windows (h = 28, stride 14). Its forecast of
# MAGIC    the next 28 days is scored against the actuals, and compared with the experiment's numbers.
# MAGIC 2. **Reference equality at scale.** MMF's `calibration_output` and interval metrics must equal
# MAGIC    `mmf_sa.intervals.conformal.calibrate_model` (the numpy reference) run on MMF's own backtest.
# MAGIC
# MAGIC Results are returned with `dbutils.notebook.exit(json)`.

# COMMAND ----------

# MAGIC %pip install "mmf_sa[local] @ git+https://github.com/lbruand-db/many-model-forecasting.git@feat/pi-p0-config" datasetsforecast==0.0.8 --quiet

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import json
import time

import numpy as np
import pandas as pd
from datasetsforecast.m5 import M5
from pyspark.sql import functions as F

from mmf_sa.intervals.config import resolve_prediction_intervals_conf
from mmf_sa.intervals.conformal import calibrate_model
from mmf_sa.intervals.features import backtest_cubes
from mmf_sa import run_forecast

dbutils.widgets.text("n_series", "5000")
dbutils.widgets.text("seed", "7")
dbutils.widgets.text("windows", "3")
dbutils.widgets.text("catalog", "main")
dbutils.widgets.text("schema", "mmf_pi_validation")
N_SERIES = int(dbutils.widgets.get("n_series"))
SEED = int(dbutils.widgets.get("seed"))
W = int(dbutils.widgets.get("windows"))
CATALOG, SCHEMA = dbutils.widgets.get("catalog"), dbutils.widgets.get("schema")
H, CAL_STEP, W_LONG, TEST_WINDOWS, TEST_STEP, FEATURE_DAYS = 28, 14, 12, 4, 28, 365
PREFIX = f"{CATALOG}.{SCHEMA}.m5_s{SEED}_w{W}"
MODELS = ["StatsForecastBaselineSeasonalNaive", "StatsForecastAutoETS", "StatsForecastCrostonSBA"]
t0 = time.time()

# COMMAND ----------

# MAGIC %md ## Data: the experiment's sample (same seed and filters as the experiment that chose the defaults)

# COMMAND ----------

M5.download(directory="/tmp/m5")
m5_path = "/tmp/m5/m5/datasets"
calendar = pd.read_csv(f"{m5_path}/calendar.csv", usecols=["date"], parse_dates=["date"])
calendar["d"] = [f"d_{i + 1}" for i in range(len(calendar))]
wide = pd.read_csv(f"{m5_path}/sales_train_evaluation.csv")
wide["unique_id"] = wide["item_id"] + "_" + wide["store_id"]
day_cols = [c for c in wide.columns if c.startswith("d_")]
rng = np.random.default_rng(SEED)
pick = rng.choice(len(wide), size=min(len(wide), int(1.5 * N_SERIES)), replace=False)
wide = wide.iloc[np.sort(pick)]
Y = wide.melt(id_vars=["unique_id"], value_vars=day_cols, var_name="d", value_name="y")
Y = Y.merge(calendar, on="d").rename(columns={"date": "ds"}).drop(columns="d")
Y["y"] = Y["y"].astype(float)
Y = Y.sort_values(["unique_id", "ds"]).reset_index(drop=True)
del wide
Y = Y[Y.groupby("unique_id")["y"].cumsum() > 0]

T = Y["ds"].max()
test_cutoff = T - pd.Timedelta(days=TEST_WINDOWS * TEST_STEP)
first_cal_cutoff = test_cutoff - pd.Timedelta(days=H + (W_LONG - 1) * CAL_STEP)
min_start = first_cal_cutoff - pd.Timedelta(days=FEATURE_DAYS + 28)
starts = Y.groupby("unique_id")["ds"].min()
eligible = starts[starts <= min_start].index.to_numpy()
ids = np.sort(rng.choice(eligible, size=min(N_SERIES, len(eligible)), replace=False))
Y = Y[Y["unique_id"].isin(ids)]
# The experiment also dropped series with a zero seasonal-naive scale over the year before calibration.
hist = Y[(Y["ds"] <= first_cal_cutoff) & (Y["ds"] > first_cal_cutoff - pd.Timedelta(days=FEATURE_DAYS))]
mae = hist.groupby("unique_id")["y"].apply(lambda s: np.mean(np.abs(s.to_numpy()[7:] - s.to_numpy()[:-7])))
mean = hist.groupby("unique_id")["y"].apply(lambda s: np.mean(np.abs(s.to_numpy())))
keep = mae.index[(mae > 0) | (mean > 0)]
Y = Y[Y["unique_id"].isin(keep)].reset_index(drop=True)
train = Y[Y["ds"] <= test_cutoff]
test = Y[(Y["ds"] > test_cutoff) & (Y["ds"] <= test_cutoff + pd.Timedelta(days=H))]
print(f"{Y['unique_id'].nunique()} series, train to {test_cutoff.date()}, test {len(test)} rows")
spark.createDataFrame(train).write.mode("overwrite").saveAsTable(f"{PREFIX}_train")

# COMMAND ----------

# MAGIC %md ## MMF run: W windows of 28 days with stride 14 = the experiment's last W calibration windows

# COMMAND ----------

user = spark.sql("select current_user() as user").collect()[0]["user"]
backtest_length = H + (W - 1) * CAL_STEP
for t in ("scoring_output", "evaluation_output", "evaluation_output_calibration",
          "evaluation_output_interval_metrics"):
    spark.sql(f"DROP TABLE IF EXISTS {PREFIX}_{t}")
run_id = run_forecast(
    spark=spark,
    train_data=f"{PREFIX}_train",
    scoring_data=f"{PREFIX}_train",
    scoring_output=f"{PREFIX}_scoring_output",
    evaluation_output=f"{PREFIX}_evaluation_output",
    group_id="unique_id", date_col="ds", target="y", freq="D",
    prediction_length=H, backtest_length=backtest_length, stride=CAL_STEP, metric="smape",
    train_predict_ratio=1, active_models=MODELS,
    experiment_path=f"/Users/{user}/mmf_pi_experiment/validation",
    use_case_name=f"m5_s{SEED}_w{W}",
    prediction_intervals={"enabled": True, "output_format": "both"},
)
t_mmf = time.time() - t0
print(f"run_forecast done in {t_mmf:.0f}s, run_id {run_id}")

# COMMAND ----------

# MAGIC %md ## 1. Held-out coverage of the next 28 days

# COMMAND ----------

score = spark.table(f"{PREFIX}_scoring_output").filter(F.col("run_id") == run_id).toPandas()
calib = spark.table(f"{PREFIX}_evaluation_output_calibration").filter(F.col("run_id") == run_id).toPandas()
sigma = calib.drop_duplicates(["model", "unique_id"]).set_index(["model", "unique_id"])["sigma"]
actual = {u: g.sort_values("ds")["y"].to_numpy() for u, g in test.groupby("unique_id")}

def crossing_share(maps):
    """Share of scoring rows whose quantile map crosses (a lower level above a higher one) at some step."""
    rows = [dict(m) for m in maps if m is not None]
    bad = [(np.diff(np.stack([np.asarray(m[k]) for k in sorted(m, key=float)]), axis=0) < -1e-9).any() for m in rows]
    return round(float(np.mean(bad)), 4) if bad else None


crossing = {model: crossing_share(g["y_quantiles"]) for model, g in score.groupby("model")}
heldout = []
for model, g in score.groupby("model"):
    for level in (80, 95):
        a = 1 - level / 100
        cov, msis, width, n = [], [], [], 0
        for _, r in g.iterrows():
            y = actual.get(r["unique_id"])
            lo, hi = r[f"y_lo_{level}"], r[f"y_hi_{level}"]
            if y is None or lo is None or len(y) != H:
                continue
            lo, hi = np.asarray(lo), np.asarray(hi)
            s = sigma[(model, r["unique_id"])]
            cov.append(np.mean((y >= lo) & (y <= hi)))
            msis.append(np.mean((hi - lo) + (2 / a) * (lo - y) * (y < lo) + (2 / a) * (y - hi) * (y > hi)) / s)
            width.append(np.mean(hi - lo) / s)
            n += 1
        heldout.append({"model": model, "level": level, "series": n, "coverage": float(np.mean(cov)),
                        "msis": float(np.mean(msis)), "width": float(np.mean(width))})
heldout = pd.DataFrame(heldout)
display(heldout)

# Experiment, seed 7, W=3, default methods, 4 test windows.
experiment = pd.DataFrame([
    ("StatsForecastAutoETS", 80, 0.788, 6.448), ("StatsForecastAutoETS", 95, 0.944, 9.834),
    ("StatsForecastBaselineSeasonalNaive", 80, 0.799, 7.388), ("StatsForecastBaselineSeasonalNaive", 95, 0.946, 10.724),
    ("StatsForecastCrostonSBA", 80, 0.783, 7.195), ("StatsForecastCrostonSBA", 95, 0.944, 14.479),
], columns=["model", "level", "exp_coverage", "exp_msis"])
comparison = heldout.merge(experiment, on=["model", "level"], how="left")
display(comparison)

# COMMAND ----------

# MAGIC %md ## 2. MMF (Spark) equals the numpy reference on MMF's own backtest

# COMMAND ----------


pi = resolve_prediction_intervals_conf({"enabled": True, "output_format": "both"})
evals = spark.table(f"{PREFIX}_evaluation_output").filter(
    (F.col("run_id") == run_id) & (F.col("metric_name") == "smape")).select(
    "unique_id", "backtest_window_start_date", "model", "forecast", "actual", "forecast_quantiles").toPandas()
metrics = spark.table(f"{PREFIX}_evaluation_output_interval_metrics").filter(F.col("run_id") == run_id).toPandas()
clip = calib["clip_lower"].dropna()
clip = float(clip.iloc[0]) if len(clip) else None
equality = []
for model, frame in evals.groupby("model"):
    c = calib[calib["model"] == model]
    feats = c.drop_duplicates("unique_id").set_index("unique_id")
    cubes = backtest_cubes(frame[frame["unique_id"].isin(feats.index)], "unique_id", (80, 95))
    f = feats.loc[cubes["ids"]]
    labels = (f["abc_class"] + f["xyz_class"]).to_numpy(dtype=object)
    ref = calibrate_model(cubes["y"], cubes["yhat"], f["sigma"].to_numpy(), pi, labels=labels,
                              native=cubes["native"], clip_lower=clip)
    m = metrics[metrics["model"] == model]
    for level, r in ref.items():
        got = c[c["level"] == float(level)].set_index(["unique_id", "step"]).sort_index()
        q_lo = got["q_lo"].to_numpy().reshape(len(cubes["ids"]), -1)
        q_hi = got["q_hi"].to_numpy().reshape(len(cubes["ids"]), -1)
        row = {"model": model, "level": level, "series": len(cubes["ids"]),
               "max_abs_q_diff": float(max(np.abs(q_lo - r["q_lo"]).max(), np.abs(q_hi - r["q_hi"]).max()))}
        for name, arr in r["metrics"].items():
            mm = m[m["metric_name"] == f"{name}_{level}"].sort_values(["unique_id", "backtest_window_start_date"])
            row[f"max_abs_{name}_diff"] = float(np.abs(mm["metric_value"].to_numpy().reshape(arr.shape) - arr).max())
            row[f"lowo_{name}"] = float(arr.mean())
        equality.append(row)
equality = pd.DataFrame(equality)
display(equality)

# COMMAND ----------

summary = {
    "run_id": run_id, "windows": W, "series": int(Y["unique_id"].nunique()), "runtime_s": round(t_mmf, 1),
    "heldout": comparison.to_dict(orient="records"), "equality": equality.to_dict(orient="records"),
    "scoring_crossing_share": crossing,
}
dbutils.notebook.exit(json.dumps(summary, default=float))
