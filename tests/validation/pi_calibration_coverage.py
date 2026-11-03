# Databricks notebook source
# MAGIC %md
# MAGIC # Held-out coverage of calibrated prediction intervals, per pooling
# MAGIC
# MAGIC The pooling study of `POOLING_VALIDATION.md`. For one dataset of the registry below: holds out the last horizon of
# MAGIC each series, runs `run_forecast` with native prediction intervals on the rest, calibrates the intervals with
# MAGIC `calibrate_intervals` in each pooling configuration, and scores every configuration on the held-out horizon:
# MAGIC
# MAGIC * `native`: the models' raw bands (models with native intervals only);
# MAGIC * `default`: one marginal pool per horizon step, each series' tail quantiles shrunk toward it;
# MAGIC * `abc_xyz`: the same, pooled per ABC x XYZ class (`pool="group", group_col="auto"`);
# MAGIC * `sb_class`: pooled per Syntetos-Boylan class (smooth, erratic, intermittent, lumpy);
# MAGIC * `keys`: pooled per the dataset's natural keys, for datasets that have them.
# MAGIC
# MAGIC For each model and level: held-out coverage, MSIS (scaled by each series' seasonal-naive error), width and WIS;
# MAGIC per grouping (ABC x XYZ class, series-scale decile, Syntetos-Boylan class, natural keys), the worst-group shortfall
# MAGIC below nominal over groups of at least 30 series, and the share of series more than 10 points below nominal; the
# MAGIC share of series merged into the "other" pool; and bootstrap intervals (over series) of the paired differences with
# MAGIC `default`. Results are returned with `dbutils.notebook.exit(json)`.

# COMMAND ----------

dbutils.widgets.text("requirements", "")
dbutils.widgets.text("dataset", "m5")
dbutils.widgets.text("n_series", "2000")
dbutils.widgets.text("windows", "3")
dbutils.widgets.text("seed", "7")
dbutils.widgets.text("bootstrap", "1000")
dbutils.widgets.text("catalog", "main")
dbutils.widgets.text("schema", "mmf_pi_validation")
with open("/tmp/pi_cov_requirements.txt", "w") as f:
    f.write("\n".join(line.strip() for line in dbutils.widgets.get("requirements").split("|") if line.strip()) + "\n")

# COMMAND ----------

# MAGIC %pip install -r /tmp/pi_cov_requirements.txt datasetsforecast==0.0.8 openpyxl --quiet

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import io
import json
import pathlib
import time
import urllib.request
import zipfile

import numpy as np
import pandas as pd
from pyspark.sql import functions as F

from mmf_sa import calibrate_intervals, run_forecast

DATASET = dbutils.widgets.get("dataset")
N_SERIES, W = int(dbutils.widgets.get("n_series")), int(dbutils.widgets.get("windows"))
SEED, N_BOOT = int(dbutils.widgets.get("seed")), int(dbutils.widgets.get("bootstrap"))
CATALOG, SCHEMA = dbutils.widgets.get("catalog"), dbutils.widgets.get("schema")
PREFIX = f"{CATALOG}.{SCHEMA}.pi_cov_{DATASET}_w{W}_s{SEED}"
USER = spark.sql("select current_user() as user").collect()[0]["user"]
LEVELS = [80, 95]
MIN_GROUP = 30
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")

# COMMAND ----------

# MAGIC %md ## Loaders

# COMMAND ----------


def download(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (mmf-validation)"})
    with urllib.request.urlopen(request, timeout=600) as r:
        return r.read()


TSF_FREQ = {"yearly": "YS", "quarterly": "QS", "monthly": "MS", "weekly": "W", "daily": "D", "hourly": "h",
            "half_hourly": "30min", "minutely": "min", "10_minutes": "10min", "4_seconds": "4s"}


def _tsf_time(value):
    """Monash start timestamps look like '2015-01-01 00-00-01'."""
    if value is None:
        return pd.Timestamp("2000-01-01")
    date, _, clock = value.partition(" ")
    return pd.Timestamp(f"{date} {clock.replace('-', ':')}" if clock else date)


def read_tsf(url: str, sample: int = None, rng=None) -> pd.DataFrame:
    """
    A Monash .tsf file (zipped on Zenodo) as a long frame unique_id, ds, y (plus the series' other attributes);
    '?' values are NaN. With sample, only that many series, drawn at random, are expanded.
    """
    raw = download(url)
    if raw[:2] == b"PK":  # a zip (Zenodo file URLs end in /content)
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            raw = z.read([n for n in z.namelist() if n.endswith(".tsf")][0])
    attributes, freq, data = [], None, []
    lines = raw.decode("utf-8", errors="replace").splitlines()
    del raw
    for i, line in enumerate(lines):
        low = line.strip().lower()
        if low.startswith("@attribute"):
            attributes.append(line.split()[1])
        elif low.startswith("@frequency"):
            freq = TSF_FREQ[line.split()[1].lower()]
        elif low.startswith("@data"):
            data = [ln for ln in lines[i + 1:] if ln.strip() and not ln.startswith("#")]
            break
    del lines
    if sample is not None and len(data) > sample:
        data = [data[j] for j in sorted(np.random.default_rng(rng).choice(len(data), sample, replace=False))]
    rows = []
    for line in data:
        parts = line.strip().split(":")
        meta = dict(zip(attributes, parts[:len(attributes)]))
        y = np.array([np.nan if v == "?" else float(v) for v in parts[len(attributes)].split(",")])
        frame = pd.DataFrame({"unique_id": meta.get("series_name", f"T{len(rows)}"),
                              "ds": pd.date_range(_tsf_time(meta.get("start_timestamp")), periods=len(y),
                                                  freq=freq or "D"), "y": y})
        for a in attributes:
            if a not in ("series_name", "start_timestamp"):
                frame[a] = meta.get(a)
        rows.append(frame)
    return pd.concat(rows, ignore_index=True)


ZENODO = "https://zenodo.org/api/records/{record}/files/{name}.zip/content"


def monash(record, name, sample=None, rng=None, month_end=False, floor=None):
    df = read_tsf(ZENODO.format(record=record, name=name), sample, rng)
    if month_end:
        df["ds"] = df["ds"] + pd.offsets.MonthEnd(0)  # MMF's monthly data uses month-end timestamps
    if floor:
        df["ds"] = df["ds"].dt.floor(floor)  # Monash hourly series start at 00:00:01
    return df


def load_car_parts(rng):
    return monash(4656021, "car_parts_dataset_without_missing_values", month_end=True)


def load_hospital(rng):
    return monash(4656014, "hospital_dataset", month_end=True)


def load_web_traffic(rng):
    return monash(4656075, "kaggle_web_traffic_dataset_without_missing_values", sample=int(1.5 * N_SERIES), rng=rng)


def load_london_smart_meters(rng):
    df = monash(4656091, "london_smart_meters_dataset_without_missing_values", sample=int(1.2 * N_SERIES), rng=rng)
    df["ds"] = df["ds"].dt.normalize()
    return df.groupby(["unique_id", "ds"], as_index=False)["y"].sum()  # half-hourly kWh -> daily kWh


def load_electricity(rng):
    return monash(4656140, "electricity_hourly_dataset", floor="h")


def load_traffic(rng):
    return monash(4656132, "traffic_hourly_dataset", floor="h")


def load_online_retail(rng):
    raw = download("https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip")
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        xlsx = z.read([n for n in z.namelist() if n.endswith(".xlsx")][0])
    sheets = pd.read_excel(io.BytesIO(xlsx), sheet_name=None, dtype={"Invoice": str, "StockCode": str},
                           engine="openpyxl")
    df = pd.concat(sheets.values(), ignore_index=True).drop_duplicates()  # the two sheets overlap
    df = df[~df["Invoice"].str.startswith("C") & (df["Quantity"] > 0)]  # no returns
    df = df[df["StockCode"].str.match(r"^\d{5}")]  # products only (not POST, BANK CHARGES, AMAZONFEE...)
    daily = (df.assign(ds=df["InvoiceDate"].dt.normalize()).groupby(["StockCode", "ds"], as_index=False)["Quantity"]
             .sum().rename(columns={"StockCode": "unique_id", "Quantity": "y"}))
    # A dense daily grid from each product's first sale to the last day of the data, zeros on days without sales.
    end = daily["ds"].max()
    frames = []
    for uid, g in daily.groupby("unique_id"):
        days = pd.date_range(g["ds"].min(), end, freq="D")
        frames.append(pd.DataFrame({"unique_id": uid, "ds": days,
                                    "y": g.set_index("ds")["y"].reindex(days, fill_value=0).to_numpy(dtype=float)}))
    out = pd.concat(frames, ignore_index=True)
    sold_days = out[out["y"] > 0].groupby("unique_id").size()
    return out[out["unique_id"].isin(sold_days[sold_days >= 60].index)]  # at least 60 days with sales


def load_tourism(rng):
    from datasetsforecast.hierarchical import HierarchicalData
    Y_df, S_df, _ = HierarchicalData.load("/tmp/hier", "TourismLarge")
    bottom = Y_df[Y_df["unique_id"].isin(S_df.columns)].copy()  # the 304 region x purpose series
    bottom["ds"] = pd.to_datetime(bottom["ds"]) + pd.offsets.MonthEnd(0)
    # Bottom ids are positional: state letter, zone letter, region letter, then the purpose (Hol, Vis, Bus, Oth).
    bottom["state"] = bottom["unique_id"].str[0]
    bottom["zone"] = bottom["unique_id"].str[:2]
    bottom["purpose"] = bottom["unique_id"].str[3:]
    return bottom


def load_epf(rng):
    frames = []
    for market in ("NP", "PJM", "BE", "FR", "DE"):
        df = pd.read_csv(io.BytesIO(download(f"https://zenodo.org/api/records/4624805/files/{market}.csv/content")))
        df = df.iloc[:, :2]  # the timestamp, then the price, in every file (DE's timestamp column has no header)
        df.columns = ["ds", "y"]
        df["ds"] = pd.to_datetime(df["ds"])
        # One daily series per market and hour of the day.
        df["market"], df["hour"] = market, df["ds"].dt.hour.map("{:02d}".format)
        df["unique_id"] = market + "_h" + df["hour"]
        df["ds"] = df["ds"].dt.normalize()
        frames.append(df[["unique_id", "ds", "y", "market", "hour"]])
    return pd.concat(frames, ignore_index=True)


def load_fred_md(rng):
    raw = pd.read_csv(io.BytesIO(download(
        "https://files.stlouisfed.org/files/htdocs/fred-md/monthly/current.csv")))
    df = raw.iloc[1:].copy()  # row 0 holds the transformation codes; the values are raw levels
    df["ds"] = pd.to_datetime(df["sasdate"], format="%m/%d/%Y") + pd.offsets.MonthEnd(0)
    long = df.drop(columns="sasdate").melt(id_vars="ds", var_name="unique_id", value_name="y")
    long["y"] = pd.to_numeric(long["y"], errors="coerce")
    return long


def load_m5(rng):
    from datasetsforecast.m5 import M5
    M5.download(directory="/tmp/m5")
    path = "/tmp/m5/m5/datasets"
    calendar = pd.read_csv(f"{path}/calendar.csv", usecols=["date"], parse_dates=["date"])
    calendar["d"] = [f"d_{i + 1}" for i in range(len(calendar))]
    wide = pd.read_csv(f"{path}/sales_train_evaluation.csv")
    wide["unique_id"] = wide["item_id"] + "_" + wide["store_id"]
    wide = wide.sample(n=min(N_SERIES, len(wide)), random_state=rng)
    day_cols = [c for c in wide.columns if c.startswith("d_")]
    keys = wide[["unique_id", "store_id", "cat_id"]]
    Y = wide.melt(id_vars=["unique_id"], value_vars=day_cols, var_name="d", value_name="y")
    Y = Y.merge(calendar, on="d").rename(columns={"date": "ds"}).drop(columns="d")
    Y["y"] = Y["y"].astype(float)
    Y = Y.sort_values(["unique_id", "ds"])
    Y = Y[Y.groupby("unique_id")["y"].cumsum() > 0]  # drop leading zeros
    return Y.merge(keys, on="unique_id")


def load_m4_monthly(rng):
    from datasetsforecast.m4 import M4
    raw, _, _ = M4.load(directory=str(pathlib.Path.home()), group="Monthly")
    lengths = raw.groupby("unique_id").size()
    ids = lengths[lengths >= 96].sample(n=N_SERIES, random_state=rng).index
    frames = []
    for uid, g in raw[raw["unique_id"].isin(ids)].groupby("unique_id"):
        g = g.sort_values("ds")
        frames.append(pd.DataFrame({"unique_id": uid, "y": g["y"].to_numpy(dtype=float),
                                    "ds": pd.date_range(end="2019-12-31", periods=len(g), freq="ME")}))
    return pd.concat(frames, ignore_index=True)


# Each entry: loader(random_state) -> long frame (unique_id, ds, y, natural key columns); freq; horizon h; stride;
# history cap in periods (the last max_len are kept); natural keys; volume (True: sales, counts, demand); the model
# without native intervals (calibrated with conformal intervals).
CROSTON, WINDOW_AVERAGE = "StatsForecastCrostonSBA", "StatsForecastBaselineWindowAverage"
REGISTRY = {
    # Volume targets
    "m5": {"load": load_m5, "freq": "D", "h": 28, "stride": 28, "max_len": 730, "keys": ["store_id", "cat_id"],
           "volume": True, "conformal_model": CROSTON},
    "car_parts": {"load": load_car_parts, "freq": "M", "h": 6, "stride": 3, "max_len": 51, "keys": [],
                  "volume": True, "conformal_model": CROSTON},
    "hospital": {"load": load_hospital, "freq": "M", "h": 12, "stride": 6, "max_len": 84, "keys": [],
                 "volume": True, "conformal_model": WINDOW_AVERAGE},
    "web_traffic": {"load": load_web_traffic, "freq": "D", "h": 28, "stride": 28, "max_len": 365, "keys": [],
                    "volume": True, "conformal_model": WINDOW_AVERAGE},
    "london_smart_meters": {"load": load_london_smart_meters, "freq": "D", "h": 28, "stride": 28, "max_len": 365,
                            "keys": [], "volume": True, "conformal_model": WINDOW_AVERAGE},
    "electricity": {"load": load_electricity, "freq": "H", "h": 24, "stride": 24, "max_len": 24 * 7 * 8, "keys": [],
                    "volume": True, "conformal_model": WINDOW_AVERAGE},
    "online_retail": {"load": load_online_retail, "freq": "D", "h": 14, "stride": 14, "max_len": 730, "keys": [],
                      "volume": True, "conformal_model": CROSTON},
    "tourism": {"load": load_tourism, "freq": "M", "h": 12, "stride": 6, "max_len": 228,
                "keys": ["purpose"], "volume": True, "conformal_model": WINDOW_AVERAGE},
    # Targets that are not volumes
    "m4_monthly": {"load": load_m4_monthly, "freq": "M", "h": 18, "stride": 6, "max_len": 144, "keys": [],
                   "volume": False, "conformal_model": WINDOW_AVERAGE},
    "traffic": {"load": load_traffic, "freq": "H", "h": 24, "stride": 24, "max_len": 24 * 7 * 8, "keys": [],
                "volume": False, "conformal_model": WINDOW_AVERAGE},
    "epf_prices": {"load": load_epf, "freq": "D", "h": 7, "stride": 7, "max_len": 728, "keys": ["market"],
                   "volume": False, "conformal_model": WINDOW_AVERAGE},
    "fred_md": {"load": load_fred_md, "freq": "M", "h": 12, "stride": 6, "max_len": 240, "keys": [],
                "volume": False, "conformal_model": WINDOW_AVERAGE},
}
SET = REGISTRY[DATASET]
H, FREQ = SET["h"], SET["freq"]
MODELS = ["StatsForecastAutoETS", "StatsForecastBaselineSeasonalNaive", SET["conformal_model"]]

# COMMAND ----------

# MAGIC %md ## Data, with the last horizon of each series held out

# COMMAND ----------

Y = SET["load"](SEED)
Y = Y.dropna(subset=["y"]).sort_values(["unique_id", "ds"]).reset_index(drop=True)
Y = Y[Y.groupby("unique_id").cumcount(ascending=False) < SET["max_len"]]
need = H + (H + (W - 1) * SET["stride"]) + 2 * {"H": 24, "D": 7, "W": 52, "M": 12}[FREQ]
lengths = Y.groupby("unique_id").size()
Y = Y[Y["unique_id"].isin(lengths[lengths >= need].index)]
ids = pd.Series(Y["unique_id"].unique())
if len(ids) > N_SERIES:
    Y = Y[Y["unique_id"].isin(ids.sample(n=N_SERIES, random_state=SEED))]
step = Y.groupby("unique_id").cumcount(ascending=False)  # 0 for the last observation
train, test = Y[step >= H], Y[step < H]
keys = [k for k in SET["keys"] if k in Y.columns]
# run_forecast reads a three-column table; the calibration also gets the natural keys, as group columns.
spark.createDataFrame(train[["unique_id", "ds", "y"]]).write.mode("overwrite").option(
    "overwriteSchema", "true").saveAsTable(f"{PREFIX}_train")
calibration_train = spark.table(f"{PREFIX}_train")
if keys:
    key_table = Y.drop_duplicates("unique_id")[["unique_id", *keys]].astype(str)
    calibration_train = calibration_train.join(spark.createDataFrame(key_table), on="unique_id", how="left")
actual = {u: g.sort_values("ds")["y"].to_numpy() for u, g in test.groupby("unique_id")}
key_of = (Y.drop_duplicates("unique_id").set_index("unique_id")[keys].astype(str).agg("|".join, axis=1)
          if keys else None)
print(f"{DATASET}: {train['unique_id'].nunique()} series, horizon {H}, {W} backtest windows, keys {keys}")

# COMMAND ----------

# MAGIC %md ## Forecasts with native intervals

# COMMAND ----------

for t in ("evaluation_output", "scoring_output"):
    spark.sql(f"DROP TABLE IF EXISTS {PREFIX}_{t}")
t0 = time.time()
run_id = run_forecast(
    spark=spark, train_data=f"{PREFIX}_train", scoring_data=f"{PREFIX}_train",
    scoring_output=f"{PREFIX}_scoring_output", evaluation_output=f"{PREFIX}_evaluation_output",
    group_id="unique_id", date_col="ds", target="y", freq=FREQ, prediction_length=H,
    backtest_length=H + (W - 1) * SET["stride"], stride=SET["stride"], metric="smape",
    train_predict_ratio=1, active_models=MODELS,
    experiment_path=f"/Users/{USER}/mmf_pi_cov/{DATASET}", use_case_name=f"pi_cov_{DATASET}",
    prediction_intervals={"enabled": True, "levels": LEVELS, "output_format": "both"},
)
timings = {"run_forecast": round(time.time() - t0, 1)}

# COMMAND ----------

# MAGIC %md ## Calibration, one per pooling

# COMMAND ----------

CONFIGS = {"default": {}, "abc_xyz": {"pool": "group", "group_col": "auto"},
           "sb_class": {"pool": "group", "group_col": "sb_class"}}
if keys:
    CONFIGS["keys"] = {"pool": "group", "group_col": keys}
calibration, merged = {}, {}
for name, kwargs in CONFIGS.items():
    for t in ("evaluation_output_calibration", "evaluation_output_interval_metrics", "scoring_output_intervals"):
        spark.sql(f"DROP TABLE IF EXISTS {PREFIX}_{name}_{t}")
    t0 = time.time()
    out = calibrate_intervals(
        spark, train_data=calibration_train, evaluation_output=f"{PREFIX}_evaluation_output",
        scoring_output=f"{PREFIX}_scoring_output", output_prefix=f"{PREFIX}_{name}", freq=FREQ, levels=LEVELS,
        output_format="lo_hi", **kwargs)
    timings[f"calibrate_{name}"] = round(time.time() - t0, 1)
    calibration[name] = {m: {k: s.get(k) for k in ("methods", "coverage", "error")} for m, s in out["models"].items()}
    pools = spark.table(f"{PREFIX}_{name}_evaluation_output_calibration").select("model", "unique_id", "pool_key") \
        .distinct().toPandas()
    merged[name] = {m: round(float((g["pool_key"] == "__other__").mean()), 3) for m, g in pools.groupby("model")}
print(json.dumps(calibration, indent=1))

# COMMAND ----------

# MAGIC %md ## Held-out scores

# COMMAND ----------

classes = (spark.table(f"{PREFIX}_default_evaluation_output_calibration")
           .select("unique_id", "sigma", F.concat("abc_class", "xyz_class").alias("abc_xyz"), "sb_class")
           .distinct().toPandas().drop_duplicates("unique_id").set_index("unique_id"))
classes["sigma_decile"] = (pd.qcut(classes["sigma"].rank(method="first"), 10, labels=False) + 1).astype(str)
if keys:
    classes["keys"] = key_of.reindex(classes.index)
GROUPINGS = ["abc_xyz", "sigma_decile", "sb_class"] + (["keys"] if keys else [])
point = spark.table(f"{PREFIX}_scoring_output").filter(F.col("run_id") == run_id).select(
    "unique_id", "model", "y").toPandas().set_index(["model", "unique_id"])["y"]


def per_series(frame):
    """Per-series held-out coverage, MSIS and width of each level, the point forecast's scaled error, and WIS."""
    rows = []
    for r in frame.itertuples(index=False):
        y = actual.get(r.unique_id)
        if y is None or len(y) != H or r.unique_id not in classes.index:
            continue
        s = classes.at[r.unique_id, "sigma"]
        row, ok = {"unique_id": r.unique_id, "model": r.model}, True
        for lv in LEVELS:
            lo, hi = getattr(r, f"y_lo_{lv}"), getattr(r, f"y_hi_{lv}")
            if lo is None or hi is None:
                ok = False
                break
            lo, hi, a = np.asarray(lo), np.asarray(hi), 1 - lv / 100
            row[f"coverage_{lv}"] = np.mean((y >= lo) & (y <= hi))
            row[f"msis_{lv}"] = np.mean((hi - lo) + 2 / a * np.maximum(lo - y, 0) + 2 / a * np.maximum(y - hi, 0)) / s
            row[f"width_{lv}"] = np.mean(hi - lo) / s
        if not ok:
            continue
        yhat = point.get((r.model, r.unique_id))
        row["scaled_ae"] = np.mean(np.abs(y - np.asarray(yhat))) / s if yhat is not None else np.nan
        row["wis"] = (0.5 * row["scaled_ae"] + sum((1 - lv / 100) / 2 * row[f"msis_{lv}"] for lv in LEVELS)) \
            / (len(LEVELS) + 0.5)
        rows.append(row)
    return pd.DataFrame(rows).join(classes, on="unique_id") if rows else pd.DataFrame()


def worst_gap(df, grouping, lv):
    """Largest shortfall of a group's coverage below nominal, over groups of at least MIN_GROUP series."""
    g = df.groupby(grouping)[f"coverage_{lv}"].agg(["mean", "size"])
    g = g[g["size"] >= MIN_GROUP]
    return round(float(max(0.0, lv / 100 - g["mean"].min())), 4) if len(g) else None


def summarize(df):
    out = {}
    if df.empty:
        return out
    for model, g in df.groupby("model"):
        res = {"series": int(len(g)), "wis": round(float(g["wis"].mean()), 4)}
        for lv in LEVELS:
            res[str(lv)] = {
                "coverage": round(float(g[f"coverage_{lv}"].mean()), 4),
                "msis": round(float(g[f"msis_{lv}"].mean()), 4),
                "width": round(float(g[f"width_{lv}"].mean()), 4),
                "share_below_nominal_by_10": round(float((g[f"coverage_{lv}"] < lv / 100 - 0.10).mean()), 4),
                "worst_gap": {gr: worst_gap(g, gr, lv) for gr in GROUPINGS},
                "by_group": {gr: {str(k): [round(float(v["mean"]), 3), int(v["size"])] for k, v in
                                  g.groupby(gr)[f"coverage_{lv}"].agg(["mean", "size"]).iterrows()}
                             for gr in GROUPINGS},
            }
        out[model] = res
    return out


def bootstrap(df, base, rng):
    """95% bootstrap intervals over series of the paired differences (configuration - default), per model."""
    out = {}
    if df.empty:
        return out
    for model, g in df.groupby("model"):
        b = base[base["model"] == model].set_index("unique_id")
        g = g.set_index("unique_id").join(b[[c for c in b.columns if c.startswith(("coverage_", "wis"))]],
                                          rsuffix="_base", how="inner")
        if g.empty:
            continue
        n = len(g)
        idx = rng.integers(0, n, size=(N_BOOT, n))
        res = {}
        for metric in [f"coverage_{lv}" for lv in LEVELS] + ["wis"]:
            d = (g[metric] - g[f"{metric}_base"]).to_numpy()
            means = d[idx].mean(axis=1)
            res[metric] = [round(float(d.mean()), 4), round(float(np.quantile(means, 0.025)), 4),
                           round(float(np.quantile(means, 0.975)), 4)]
        for grouping in ("abc_xyz", "sigma_decile"):
            labels = g[grouping].to_numpy()
            for lv in LEVELS:
                cov, cov_b, gaps = g[f"coverage_{lv}"].to_numpy(), g[f"coverage_{lv}_base"].to_numpy(), []
                for row in idx[:200]:  # the worst-group gap is costlier: 200 resamples
                    agg = pd.DataFrame({"k": labels[row], "c": cov[row], "b": cov_b[row]}).groupby("k").agg(
                        c=("c", "mean"), b=("b", "mean"), n=("c", "size"))
                    agg = agg[agg["n"] >= MIN_GROUP]
                    if len(agg):
                        gaps.append(max(0.0, lv / 100 - agg["c"].min()) - max(0.0, lv / 100 - agg["b"].min()))
                if gaps:
                    res[f"worst_gap_{grouping}_{lv}"] = [round(float(np.mean(gaps)), 4),
                                                         round(float(np.quantile(gaps, 0.025)), 4),
                                                         round(float(np.quantile(gaps, 0.975)), 4)]
        out[model] = res
    return out


cols = ["unique_id", "model", *[f"y_{s}_{lv}" for lv in LEVELS for s in ("lo", "hi")]]
frames = {"native": per_series(spark.table(f"{PREFIX}_scoring_output").filter(F.col("run_id") == run_id)
                               .select(*cols).toPandas())}
for name in CONFIGS:
    frames[name] = per_series(spark.table(f"{PREFIX}_{name}_scoring_output_intervals").select(*cols).toPandas())
rng = np.random.default_rng(SEED)
results = {name: summarize(df) for name, df in frames.items()}
paired = {name: bootstrap(df, frames["default"], rng) for name, df in frames.items() if name != "default"}

for lv in LEVELS:
    print(f"--- level {lv}: coverage / msis / worst ABC x XYZ gap")
    for name, res in results.items():
        print(name.ljust(9), {m: (v[str(lv)]["coverage"], v[str(lv)]["msis"], v[str(lv)]["worst_gap"]["abc_xyz"])
                              for m, v in res.items()})

# COMMAND ----------

dbutils.notebook.exit(json.dumps({
    "dataset": DATASET, "volume": SET["volume"], "keys": keys, "series": int(train["unique_id"].nunique()), "h": H,
    "windows": W, "seed": SEED, "run_id": run_id, "timings": timings, "calibration": calibration,
    "merged_into_other": merged, "heldout": results, "paired_vs_default": paired}, default=str))
