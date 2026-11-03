# Databricks notebook source
# MAGIC %md
# MAGIC # Candidates for the per-series scale of the default conformal target
# MAGIC
# MAGIC The default conformal target (`series_scale`) divides each series' residual scores by a scale factor
# MAGIC c_i = (n_i d_i + kappa d_pool) / ((n_i + kappa) d_pool). With d = mean |e| and n_i = W x H scores it under-covered
# MAGIC on M4 monthly (`pi_calibration_coverage.py`). This notebook reuses the backtest and forecasts of that run and, with
# MAGIC the numpy reference (`intervals.conformal.calibrate_model`, method conformal for every model), compares candidate
# MAGIC scale statistics and weights on the held-out horizon:
# MAGIC
# MAGIC * statistic d: `mean` |e|, `rms` (root mean square of e), `q80` (80% quantile of |e|);
# MAGIC * weight n_i: `scores` (W x H) or `windows` (W);
# MAGIC * references: `none` (c_i = 1: one pool, no per-series scale) and `former` (tail quantiles shrunk, ABC x XYZ).
# MAGIC
# MAGIC Results are returned with `dbutils.notebook.exit(json)`.

# COMMAND ----------

dbutils.widgets.text("requirements", "")
dbutils.widgets.text("datasets", "m5,m4_monthly")
dbutils.widgets.text("n_series", "2000")
dbutils.widgets.text("windows", "3")
dbutils.widgets.text("catalog", "main")
dbutils.widgets.text("schema", "lbruand_mmf_pi_ci_local")
with open("/tmp/pi_cand_requirements.txt", "w") as f:
    f.write("\n".join(line.strip() for line in dbutils.widgets.get("requirements").split("|") if line.strip()) + "\n")

# COMMAND ----------

# MAGIC %pip install -r /tmp/pi_cand_requirements.txt datasetsforecast==0.0.8 --quiet

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import json
import pathlib

import numpy as np
import pandas as pd
from pyspark.sql import functions as F

from mmf_sa.intervals import conformal
from mmf_sa.intervals.calibration_config import resolve_prediction_intervals_conf
from mmf_sa.intervals.features import backtest_cubes

DATASETS = [d.strip() for d in dbutils.widgets.get("datasets").split(",") if d.strip()]
N_SERIES, W = int(dbutils.widgets.get("n_series")), int(dbutils.widgets.get("windows"))
CATALOG, SCHEMA = dbutils.widgets.get("catalog"), dbutils.widgets.get("schema")
H_OF = {"m5": 28, "m4_monthly": 18}
LEVELS = [80, 95]
ORIGINAL = {"pool_mean_abs": conformal.pool_mean_abs, "series_scale_factors": conformal.series_scale_factors}

# COMMAND ----------

# MAGIC %md ## Held-out actuals (the same samples as pi_calibration_coverage.py)

# COMMAND ----------


def holdout(dataset, h):
    if dataset == "m5":
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
        Y = Y[Y.groupby("unique_id")["y"].cumsum() > 0]
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
    step = Y.groupby("unique_id").cumcount(ascending=False)
    test = Y[step < h]
    return {u: g.sort_values("ds")["y"].to_numpy() for u, g in test.groupby("unique_id")}, float(Y[step >= h]["y"].min())


# COMMAND ----------

# MAGIC %md ## Candidates


# COMMAND ----------


def stat(a, how, axis=None):
    """Scale statistic of absolute scores a."""
    if how == "mean":
        return a.mean(axis=axis)
    if how == "rms":
        return np.sqrt((a ** 2).mean(axis=axis))
    return np.quantile(a, 0.8, axis=axis)


def install(how, weight):
    """Replaces the scale of the series_scale target in intervals.conformal (numpy reference)."""
    def pool_stat(e, pools):
        return {p: float(stat(np.abs(e[pools == p]).ravel(), how)) for p in pd.unique(pools)}

    def factors(e, pools, pool_values, kappa):
        n = e.shape[0]
        a = np.abs(e.reshape(n, -1))
        n_i = a.shape[1] if weight == "scores" else e.shape[1]
        d_i = stat(a, how, axis=1)
        d_pool = np.array([pool_values[p] for p in pools], dtype=float)
        out = np.ones(n)
        ok = d_pool > 0
        out[ok] = (n_i * d_i[ok] + kappa * d_pool[ok]) / ((n_i + kappa) * d_pool[ok])
        return out

    conformal.pool_mean_abs, conformal.series_scale_factors = pool_stat, factors


def restore():
    conformal.pool_mean_abs = ORIGINAL["pool_mean_abs"]
    conformal.series_scale_factors = ORIGINAL["series_scale_factors"]


CANDIDATES = {f"{how}_{weight}": (how, weight) for how in ("mean", "rms", "q80") for weight in ("scores", "windows")}


def conf(**coverage):
    return resolve_prediction_intervals_conf({"enabled": True, "method": "conformal", "levels": LEVELS,
                                             "coverage": coverage})


# COMMAND ----------

# MAGIC %md ## Calibration and held-out scores

# COMMAND ----------


def scores(y_test, yhat_test, sigma, q_lo, q_hi, level, clip_lower):
    lo, hi = conformal.finalize_bounds(yhat_test + sigma[:, None] * q_lo, yhat_test + sigma[:, None] * q_hi, clip_lower)
    a = 1 - level / 100
    cov = ((y_test >= lo) & (y_test <= hi)).mean(axis=1)
    msis = ((hi - lo) + 2 / a * np.maximum(lo - y_test, 0) + 2 / a * np.maximum(y_test - hi, 0)).mean(axis=1) / sigma
    return cov, msis


def summary(cov, msis, decile, abc_xyz):
    by_class = pd.Series(cov).groupby(abc_xyz).agg(["mean", "size"])
    by_class = by_class[by_class["size"] >= 30]
    by_dec = pd.Series(cov).groupby(decile).mean()
    return {"coverage": round(float(cov.mean()), 4), "msis": round(float(msis.mean()), 3),
            "worst_class": ([by_class["mean"].idxmin(), round(float(by_class["mean"].min()), 3)]
                            if len(by_class) else None),
            "decile_1": round(float(by_dec.loc[1]), 3), "decile_10": round(float(by_dec.loc[10]), 3),
            "decile_spread": round(float(by_dec.max() - by_dec.min()), 3)}


results = {}
for dataset in DATASETS:
    h = H_OF[dataset]
    prefix = f"{CATALOG}.{SCHEMA}.pi_cov_{dataset}_w{W}"
    actual, min_train = holdout(dataset, h)
    clip_lower = 0.0 if min_train >= 0 else None
    feats = (spark.table(f"{prefix}_default_evaluation_output_calibration")
             .select("unique_id", "sigma", F.concat("abc_class", "xyz_class").alias("abc_xyz")).distinct().toPandas()
             .drop_duplicates("unique_id").set_index("unique_id"))
    ev = spark.table(f"{prefix}_evaluation_output").filter(F.col("metric_name") == "smape").select(
        "unique_id", "model", "backtest_window_start_date", "forecast", "actual").toPandas()
    sc = spark.table(f"{prefix}_scoring_output").select("unique_id", "model", "y").toPandas()
    results[dataset] = {}
    for model, rows in ev.groupby("model"):
        cubes = backtest_cubes(rows, "unique_id")
        point = sc[sc["model"] == model].set_index("unique_id")["y"]
        keep = [i for i, u in enumerate(cubes["ids"]) if u in feats.index and u in point.index and u in actual
                and len(actual[u]) == h]
        ids = np.asarray(cubes["ids"], dtype=object)[keep]
        y, yhat = cubes["y"][keep], cubes["yhat"][keep]
        sigma = feats.loc[ids, "sigma"].to_numpy(dtype=float)
        labels = feats.loc[ids, "abc_xyz"].to_numpy(dtype=object)
        y_test = np.stack([actual[u] for u in ids])
        yhat_test = np.stack([np.asarray(point.loc[u], dtype=float) for u in ids])
        decile = pd.qcut(pd.Series(sigma).rank(method="first"), 10, labels=False).to_numpy() + 1
        out = {}
        runs = {name: (conf(), cand) for name, cand in CANDIDATES.items()}
        runs["none"] = (conf(target="marginal"), None)
        runs["former"] = (conf(target="series", group_col="auto"), None)
        for name, (pi, cand) in runs.items():
            if cand is not None:
                install(*cand)
            try:
                res = conformal.calibrate_model(y, yhat, sigma, pi, labels=labels, clip_lower=clip_lower, model=model)
            finally:
                restore()
            out[name] = {}
            for lv in LEVELS:
                cov, msis = scores(y_test, yhat_test, sigma, res[lv]["q_lo"], res[lv]["q_hi"], lv, clip_lower)
                out[name][lv] = {**summary(cov, msis, decile, labels),
                                 "backtest_coverage": round(float(res[lv]["metrics"]["coverage"].mean()), 4)}
        results[dataset][model] = out
        print(dataset, model, len(ids), "series")
        for lv in LEVELS:
            print(f"  level {lv}:", {k: (v[lv]["coverage"], v[lv]["msis"]) for k, v in out.items()})

# COMMAND ----------

dbutils.notebook.exit(json.dumps(results, default=str))
