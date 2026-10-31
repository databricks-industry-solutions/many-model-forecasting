"""Distributed prediction-interval calibration.

Spark counterpart of intervals.conformal.calibrate_model. Nothing that grows with the number of series is
collected to the driver: only window/horizon counts, group-label counts, pool order statistics
(pools x steps x requests), the adaptive-split miss rates, and scalars.

- Backtest rows are aligned and exploded to one row per series, window and step (long_backtest).
- Pool quantiles are exact order statistics: row_number() over (fold, pool, step) ordered by score,
  joined to the conformal ranks of each pool (pool_statistics). They equal numpy_pool_statistics.
- The per-series combination, the corrections and the leave-one-window-out metrics run per series in
  applyInPandas, with intervals.conformal.fit_from_statistics, as in the numpy reference.
"""
import logging
import uuid

import numpy as np
import pandas as pd
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    ArrayType, BooleanType, DoubleType, IntegerType, LongType, StringType, StructField, StructType, TimestampType,
)
from pyspark.sql.window import Window

from mmf_sa.intervals.config import calibration_levels, level_quantiles
from mmf_sa.intervals.conformal import (
    MARGINAL_POOL, OTHER_GROUP, conformal_rank, cqr_lower_from_statistics, finalize_bounds, fit_from_statistics,
    fit_plan, interval_metrics, score_cubes, stat_requests, window_bounds,
)
from mmf_sa.intervals.features import ABC_SHARES, SB_ADI, SB_CV2
from mmf_sa.intervals.quantiles import quantile_key
from mmf_sa.intervals.scoring import _metric_name

_logger = logging.getLogger(__name__)


FULL = "full"  # fold label of the calibration on all windows (fold index W)


class Materializer:
    """
    Caches intermediate DataFrames of the prediction-interval steps, which run many Spark actions on them.
    Uses df.persist() where available. Where it is not (serverless clients before environment version 5
    reject it with CACHE_REQUIRES_NEWER_CLIENT_ON_SERVERLESS_GC), each DataFrame is written to a temporary
    table named <scratch_prefix>_<id> and read back, so that it is not recomputed by every action; the
    tables are dropped by uncache() and close().
    """

    def __init__(self, spark: SparkSession, scratch_prefix: str):
        self.spark = spark
        self.scratch_prefix = scratch_prefix
        self.can_persist = None
        self.tables = {}
        self.persisted = []

    def cache(self, df: DataFrame) -> DataFrame:
        if self.can_persist is not False:
            try:
                out = df.persist()
                self.can_persist = True
                self.persisted.append(out)
                return out
            except Exception as err:
                self.can_persist = False
                _logger.warning(f"DataFrame caching is not available on this compute ({type(err).__name__}); "
                                f"prediction-interval steps materialize intermediate results to temporary tables "
                                f"{self.scratch_prefix}_* instead, dropped at the end.")
        name = f"{self.scratch_prefix}_{uuid.uuid4().hex[:12]}"
        df.write.mode("overwrite").saveAsTable(name)
        out = self.spark.table(name)
        self.tables[id(out)] = name
        return out

    def uncache(self, df: DataFrame) -> None:
        name = self.tables.pop(id(df), None)
        if name is not None:
            self._drop(name)
            return
        try:
            df.unpersist()
        except Exception:
            pass
        self.persisted = [p for p in self.persisted if p is not df]

    def close(self) -> None:
        """Releases everything still cached: unpersists DataFrames, drops the temporary tables."""
        for df in self.persisted:
            try:
                df.unpersist()
            except Exception:
                pass
        self.persisted = []
        for name in list(self.tables.values()):
            self._drop(name)
        self.tables.clear()

    def _drop(self, name: str) -> None:
        try:
            self.spark.sql(f"DROP TABLE IF EXISTS {name}")
        except Exception as err:
            _logger.warning(f"Could not drop temporary table {name}: {err}")


def _mode(sdf: DataFrame, column: str) -> int:
    """Most frequent value of an integer column; the smallest on ties (as pandas Series.mode().iloc[0])."""
    row = sdf.groupBy(column).count().orderBy(F.col("count").desc(), F.col(column).asc()).limit(1).collect()
    return int(row[0][0])


def _finite(c):
    return ~(F.isnan(c) | c.isNull() | (c == float("inf")) | (c == float("-inf")))


def _all_finite(arr):
    return F.forall(arr, lambda x: _finite(x))


def long_backtest(eval_sdf: DataFrame, series_col: str, levels, has_quantiles: bool):
    """
    One model's backtest rows aligned as intervals.features.backtest_cubes does, exploded to one row per
    series, window and step.
    Parameters:
        eval_sdf (DataFrame): The model's evaluation_output rows (series_col as string,
            backtest_window_start_date, forecast, actual, optionally forecast_quantiles).
        levels: Central levels whose native bands to extract.
        has_quantiles (bool): Whether native quantiles are read (else no lo/hi columns).
    Returns: (DataFrame, int, int, int, int): Rows (series, window, window_date, step, y, yhat, lo_<i>, hi_<i>),
        W, H, the number of series dropped, and the number kept.
    """
    w = _mode(eval_sdf.groupBy(series_col).count(), "count")
    h = _mode(eval_sdf.select(F.coalesce(F.size("forecast"), F.lit(0)).alias("h")), "h")
    ranked = eval_sdf.withColumn(
        "_rn", F.row_number().over(Window.partitionBy(series_col).orderBy(F.col("backtest_window_start_date").desc())))
    kept = ranked.filter(F.col("_rn") <= w).withColumn("window", F.lit(w) - F.col("_rn"))
    valid = (
        F.col("forecast").isNotNull() & F.col("actual").isNotNull()
        & (F.size("forecast") == h) & (F.size("actual") == h)
        & _all_finite(F.col("forecast")) & _all_finite(F.col("actual"))
    )
    good = (kept.groupBy(series_col)
            .agg(F.count(F.lit(1)).alias("_n"), F.min(F.when(valid, 1).otherwise(0)).alias("_ok"))
            .filter((F.col("_n") == w) & (F.col("_ok") == 1)).select(series_col))
    n_all = eval_sdf.select(series_col).distinct().count()
    rows = kept.join(good, on=series_col, how="inner")
    n_kept = good.count()
    dropped = n_all - n_kept

    arrays = {"y": F.col("actual"), "yhat": F.col("forecast")}
    if has_quantiles:
        for i, lv in enumerate(levels):
            k_lo, k_hi = (quantile_key(q) for q in level_quantiles(lv))
            lo = F.coalesce(F.element_at("forecast_quantiles", k_lo), F.col("forecast"))
            hi = F.coalesce(F.element_at("forecast_quantiles", k_hi), F.col("forecast"))
            same = (F.size(lo) == h) & (F.size(hi) == h)
            lo, hi = F.when(same, lo).otherwise(F.col("forecast")), F.when(same, hi).otherwise(F.col("forecast"))
            # Non-finite native values fall back to the point forecast.
            arrays[f"lo_{i}"] = F.zip_with(lo, F.col("forecast"), lambda x, f: F.when(_finite(x), x).otherwise(f))
            arrays[f"hi_{i}"] = F.zip_with(hi, F.col("forecast"), lambda x, f: F.when(_finite(x), x).otherwise(f))
    wide = rows.select(series_col, "window", F.col("backtest_window_start_date").alias("window_date"),
                       *[c.alias(n) for n, c in arrays.items()])
    long = wide.withColumn("step", F.explode(F.sequence(F.lit(1), F.lit(h)))).select(
        series_col, "window", "window_date", "step",
        *[F.element_at(n, F.col("step")).cast(DoubleType()).alias(n) for n in arrays])
    return long, w, h, dropped, n_kept


def series_features(train: DataFrame, first_window: DataFrame, series_col: str, date_col: str, target: str,
                    season_length: int, periods: int, extra_cols=()) -> DataFrame:
    """
    Per-series scale and classification features in Spark SQL, as intervals.features.series_features over the last
    `periods` points before each series' first backtest window (missing target values count as 0).
    Parameters:
        train (DataFrame): series_col (string), date_col (timestamp), target (double) and extra_cols.
        first_window (DataFrame): series_col and _first_window (timestamp).
        extra_cols: Group columns; each takes the series' first non-null value by date, as a string.
    Returns: DataFrame: series_col, sigma, volume, cv, adi, cv2 and extra_cols, one row per series of first_window
        that has training rows.
    """
    base = train.join(first_window, on=series_col, how="inner")
    ids = base.select(series_col).distinct()
    recent = Window.partitionBy(series_col).orderBy(F.col(date_col).desc())
    in_order = Window.partitionBy(series_col).orderBy(date_col)
    y = F.coalesce(F.col(target), F.lit(0.0))
    hist = (base.filter(F.col(date_col) < F.col("_first_window"))
            .withColumn("_y", y)
            .withColumn("_rn", F.row_number().over(recent))
            .filter(F.col("_rn") <= periods)
            .withColumn("_diff", F.abs(F.col("_y") - F.lag("_y", season_length).over(in_order))))
    positive = F.when(F.col("_y") > 0, F.col("_y"))
    agg = hist.groupBy(series_col).agg(
        F.count(F.lit(1)).alias("_n"), F.sum("_y").alias("volume"), F.avg("_y").alias("_mean"),
        F.stddev_pop("_y").alias("_std"), F.avg(F.abs("_y")).alias("_mean_abs"), F.avg("_diff").alias("_snaive"),
        F.count(positive).alias("_nz"), F.avg(positive).alias("_nz_mean"), F.stddev_pop(positive).alias("_nz_std"))
    out = ids.join(agg, on=series_col, how="left").fillna({"_n": 0, "volume": 0.0, "_nz": 0})
    snaive = F.when(F.col("_n") > season_length, F.col("_snaive")).otherwise(F.lit(0.0))
    out = out.select(
        series_col,
        F.when(snaive > 0, snaive).otherwise(F.when(F.col("_n") > 0, F.col("_mean_abs")).otherwise(F.lit(0.0)))
        .alias("sigma"),
        F.col("volume").cast(DoubleType()).alias("volume"),
        F.when(F.col("_mean") > 0, F.col("_std") / F.col("_mean")).otherwise(F.lit(float("inf"))).alias("cv"),
        (F.col("_n") / F.greatest(F.col("_nz"), F.lit(1))).cast(DoubleType()).alias("adi"),
        F.when(F.col("_nz") > 1, F.pow(F.col("_nz_std") / F.col("_nz_mean"), 2)).otherwise(F.lit(0.0)).alias("cv2"),
    )
    for c in extra_cols:
        firsts = (base.filter(F.col(c).isNotNull())
                  .withColumn("_rn", F.row_number().over(in_order))
                  .filter(F.col("_rn") == 1)
                  .select(series_col, F.col(c).cast(StringType()).alias(c)))
        out = out.join(firsts, on=series_col, how="left")
    return out


def classify(features: DataFrame, series_col: str) -> DataFrame:
    """Syntetos-Boylan, ABC, XYZ and ABC x XYZ classes, as intervals.features.classify_series (ties by series id)."""
    sb = (F.when((F.col("adi") < SB_ADI) & (F.col("cv2") < SB_CV2), "smooth")
          .when(F.col("adi") < SB_ADI, "erratic")
          .when(F.col("cv2") < SB_CV2, "intermittent").otherwise("lumpy"))
    everyone = Window.partitionBy()
    by_volume = Window.orderBy(F.col("volume").desc(), F.col(series_col).asc()).rowsBetween(
        Window.unboundedPreceding, Window.currentRow)
    by_cv = Window.orderBy(F.col("cv").asc(), F.col(series_col).asc())
    out = (
        features.withColumn("sb_class", sb)
        .withColumn("_total", F.sum("volume").over(everyone))
        .withColumn("_n", F.count(F.lit(1)).over(everyone))
        .withColumn("_share", F.sum("volume").over(by_volume) / F.col("_total"))
        .withColumn("_rank", F.row_number().over(by_cv))
    )
    abc = (F.when(F.col("_total") == 0, "nan").when(F.col("_share") <= ABC_SHARES[0], "A")
           .when(F.col("_share") <= ABC_SHARES[1], "B").otherwise("C"))
    e1 = 1 + (F.col("_n") - 1) / 3
    e2 = 1 + 2 * (F.col("_n") - 1) / 3
    xyz = (F.when(F.col("_n") < 3, "X").when(F.col("_rank") <= e1, "X").when(F.col("_rank") <= e2, "Y")
           .otherwise("Z"))
    return (out.withColumn("abc_class", abc).withColumn("xyz_class", xyz)
            .withColumn("abc_xyz", F.concat("abc_class", "xyz_class"))
            .drop("_total", "_n", "_share", "_rank"))


def label_column(pi_conf):
    """Spark expression of the group label (intervals.features.group_labels), or None."""
    group_col = pi_conf["coverage"]["group_col"]
    if group_col is None:
        return None
    cols = ["abc_xyz"] if group_col == "auto" else ([group_col] if isinstance(group_col, str) else list(group_col))
    return F.concat_ws("|", *[F.coalesce(F.col(c).cast(StringType()), F.lit("None")) for c in cols])


def merged_pools(spark: SparkSession, labels: DataFrame, plan: dict, w: int, has_labels: bool) -> DataFrame:
    """Small table label -> pool_full / pool_lowo (merged for W and W - 1 windows), as intervals.conformal.pools_for."""
    counts = labels.groupBy("_label").agg(F.count(F.lit(1)).alias("_count")).toPandas()
    rows = []
    for label, count in zip(counts["_label"], counts["_count"]):
        def merged(n_windows):
            if plan["target"] in ("marginal", "native") or not has_labels:
                return MARGINAL_POOL
            return label if count * n_windows >= plan["min_scores_per_group"] else OTHER_GROUP
        rows.append((label, merged(w), merged(w - 1)))
    schema = StructType([StructField("_label", StringType()), StructField("pool_full", StringType()),
                         StructField("pool_lowo", StringType())])
    return spark.createDataFrame(pd.DataFrame(rows, columns=["_label", "pool_full", "pool_lowo"]), schema)


def score_columns(plan: dict, level_index: int, clip_lower):
    """Spark expressions of intervals.conformal.score_cubes (same operations, so the same doubles)."""
    y, yhat, sigma = F.col("y"), F.col("yhat"), F.col("sigma")
    if plan["kind"] == "conformal":
        scale = sigma if plan["score"] == "signed_normalized" else F.lit(1.0)
        return {"e": (y - yhat) / scale}
    if plan["kind"] == "native":
        return {}
    lo, hi = F.col(f"lo_{level_index}"), F.col(f"hi_{level_index}")
    if clip_lower is not None:
        lo = F.greatest(lo, F.lit(float(clip_lower)))
    if plan["score"] == "symmetric":
        return {"sym": F.greatest(lo - y, y - hi) / sigma}
    if plan["score"] == "scaled":
        return {"scl": F.abs(y - yhat) / F.greatest((hi - lo) / 2, F.lit(1e-3) * sigma)}
    return {"slo": (lo - y) / sigma, "shi": (y - hi) / sigma}


def folded(long: DataFrame, w: int) -> DataFrame:
    """Rows of every fold: fold W (all windows) and fold j < W (leave window j out), with the fold's pool."""
    folds = F.filter(F.sequence(F.lit(0), F.lit(w)), lambda f: f != F.col("window"))
    return (long.withColumn("fold", F.explode(folds))
            .withColumn("pool", F.when(F.col("fold") == w, F.col("pool_full")).otherwise(F.col("pool_lowo"))))


def pool_statistics(spark: SparkSession, scored: DataFrame, requests: dict) -> dict:
    """
    Exact pool order statistics of every fold, as intervals.conformal.numpy_pool_statistics.
    Parameters:
        scored (DataFrame): Folded rows with fold, pool, step and the score columns.
        requests (dict): fold -> list of (score, q, upper, steps_pooled) requests.
    Returns: dict fold -> {(pool, step, request): (value, infinite)}, step 0 for steps-pooled requests.
    """
    out = {fold: {} for fold in requests}
    by_id = {repr(r): r for reqs in requests.values() for r in reqs}
    groups = sorted({(r[0], r[3]) for reqs in requests.values() for r in reqs})
    for score, pooled in groups:
        keys = ["fold", "pool"] + ([] if pooled else ["step"])
        counts = scored.groupBy(*keys).agg(F.count(F.lit(1)).alias("_count")).toPandas()
        targets = []
        for rec in counts.to_dict("records"):
            for req in requests.get(int(rec["fold"]), []):
                if (req[0], req[3]) != (score, pooled):
                    continue
                k, inf = conformal_rank(int(rec["_count"]), req[1], upper=req[2])
                targets.append({**{c: rec[c] for c in keys}, "_rn": k, "_req": repr(req), "_inf": inf})
        if not targets:
            continue
        target_schema = StructType(
            [StructField("fold", IntegerType()), StructField("pool", StringType())]
            + ([] if pooled else [StructField("step", IntegerType())])
            + [StructField("_rn", IntegerType()), StructField("_req", StringType()), StructField("_inf", BooleanType())])
        target_pdf = pd.DataFrame(targets)[[f.name for f in target_schema.fields]]
        target_pdf = target_pdf.astype({c: "int32" for c in ("fold", "step", "_rn") if c in target_pdf.columns})
        target_df = spark.createDataFrame(target_pdf, target_schema)
        ranked = scored.select(*keys, F.col(score).alias("_s")).withColumn(
            "_rn", F.row_number().over(Window.partitionBy(*keys).orderBy("_s")))
        values = ranked.join(F.broadcast(target_df), on=keys + ["_rn"], how="inner").toPandas()
        for rec in values.to_dict("records"):
            step = 0 if pooled else int(rec["step"])
            out[int(rec["fold"])][(rec["pool"], step, by_id[rec["_req"]])] = (float(rec["_s"]), bool(rec["_inf"]))
    return out


SERIES_PER_BUCKET = 2000


def bucket_column(series_col: str, n_series: int):
    """Hash bucket of a series, so that the per-series numpy work runs on ~SERIES_PER_BUCKET series per UDF call."""
    n_buckets = max(1, -(-int(n_series) // SERIES_PER_BUCKET))
    return F.pmod(F.xxhash64(F.col(series_col)), F.lit(n_buckets))


def _batch_cubes(pdf: pd.DataFrame, series_col: str, n_levels: int, has_band: bool, w: int, h: int):
    """Cubes (n, W, H) of the series of one batch (every series has exactly W x H rows), and their first rows."""
    pdf = pdf.sort_values([series_col, "window", "step"], kind="mergesort")
    n = len(pdf) // (w * h)

    def cube(c):
        return pdf[c].to_numpy(dtype=float).reshape(n, w, h)

    bands = {i: (cube(f"lo_{i}"), cube(f"hi_{i}")) for i in range(n_levels)} if has_band else {}
    firsts = pdf.iloc[:: w * h]
    dates = pdf["window_date"].to_numpy().reshape(n, w, h)[:, :, 0]
    pools = {"full": firsts["pool_full"].to_numpy(dtype=object), "lowo": firsts["pool_lowo"].to_numpy(dtype=object)}
    return cube("y"), cube("yhat"), bands, dates, firsts, pools


def _folds(w: int):
    # Fold W uses all windows; the leave-one-window-out folds only exist with 2 windows or more.
    return range(w + 1) if w >= 2 else [w]


def batch_udf_lower(pdf, *, plan, level, w, h, clip_lower, stats, series_col):
    """Phase 1 of the adaptive CQR split: per fold, the lower-tail misses of the corrected band over a batch."""
    y, yhat, bands, _, firsts, pools = _batch_cubes(pdf, series_col, 1, True, w, h)
    sigma = firsts["sigma"].to_numpy(dtype=float)
    band = bands[0]
    if clip_lower is not None:
        band = (np.maximum(band[0], clip_lower), band[1])
    cubes = score_cubes(plan, y, yhat, sigma, band)
    rows = []
    for fold in _folds(w):
        idx = np.arange(w) if fold == w else np.delete(np.arange(w), fold)
        sub = {k: v[:, idx] for k, v in cubes.items()}
        q_lo, _ = cqr_lower_from_statistics(plan, level, sub, pools["full" if fold == w else "lowo"],
                                                stats[fold])
        rows.append((fold, int((sub["slo"] > q_lo[:, None, :]).sum()), int(sub["slo"].size)))
    return pd.DataFrame(rows, columns=["fold", "misses", "n"])


def batch_udf_final(pdf, *, plan, levels, has_band, w, h, clip_lower, stats, m_lo, series_col, model):
    """calibration_output rows (fold W) and leave-one-window-out metric rows (folds j < W) of a batch of series."""
    y, yhat, bands, dates, firsts, pools = _batch_cubes(pdf, series_col, len(levels), has_band, w, h)
    n = len(firsts)
    ids = firsts[series_col].to_numpy(dtype=object)
    sigma = firsts["sigma"].to_numpy(dtype=float)
    pool_key = (np.full(n, "", dtype=object) if plan["kind"] == "native"
                else np.full(n, MARGINAL_POOL, dtype=object) if plan["target"] == "marginal"
                else pools["full"])
    cal_frames, metric_frames = [], []
    for i, lv in enumerate(levels):
        band = bands.get(i)
        if band is not None and clip_lower is not None:
            band = (np.maximum(band[0], clip_lower), band[1])
        cubes = score_cubes(plan, y, yhat, sigma, band)

        def fit(fold):
            idx = np.arange(w) if fold == w else np.delete(np.arange(w), fold)
            sub = {k: v[:, idx] for k, v in cubes.items()}
            return fit_from_statistics(plan, lv, sub, pools["full" if fold == w else "lowo"], stats[lv][fold],
                                           m_lo.get((lv, fold)), horizon=h)

        q_lo, q_hi, inf = fit(w)
        cal_frames.append(pd.DataFrame({
            "model": model, "method": plan["kind"], "score": plan["score"], "coverage_target": plan["target"],
            series_col: np.repeat(ids, h), "pool_key": np.repeat(pool_key, h),
            "step": np.tile(np.arange(1, h + 1), n), "level": float(lv),
            "q_lo": q_lo.ravel(), "q_hi": q_hi.ravel(), "n_windows": w,
            "shrinkage_weight": (w / (w + plan["kappa"])) if plan["kappa"] else None,
            "infinite_bound": np.repeat(np.asarray(inf, dtype=bool), h), "clip_lower": clip_lower,
            "sigma": np.repeat(sigma, h),
            "abc_class": np.repeat(firsts["abc_class"].to_numpy(dtype=object), h),
            "xyz_class": np.repeat(firsts["xyz_class"].to_numpy(dtype=object), h),
            "sb_class": np.repeat(firsts["sb_class"].to_numpy(dtype=object), h),
        }))

        def add_metrics(j, lo, hi, native):
            for name, value in interval_metrics(y[:, j:j + 1], lo, hi, sigma, lv).items():
                metric_frames.append(pd.DataFrame({
                    series_col: ids, "window_date": dates[:, j],
                    "metric_name": _metric_name(name, lv, native), "metric_value": value}))

        if w >= 2:
            for j in range(w):
                fq_lo, fq_hi, _ = fit(j)
                b = None if band is None else (band[0][:, j:j + 1], band[1][:, j:j + 1])
                add_metrics(j, *finalize_bounds(*window_bounds(plan, yhat[:, j:j + 1], b, sigma, fq_lo, fq_hi),
                                                    clip_lower), False)
        if band is not None:
            for j in range(w):
                add_metrics(j, *finalize_bounds(band[0][:, j:j + 1], band[1][:, j:j + 1], clip_lower), True)
    cal_df = pd.concat(cal_frames, ignore_index=True)
    cal_df["kind"] = "calibration"
    met_df = (pd.concat(metric_frames, ignore_index=True) if metric_frames else
              pd.DataFrame(columns=[series_col, "window_date", "metric_name", "metric_value"]))
    met_df["kind"] = "metric"
    met_df["model"] = model
    return pd.concat([cal_df, met_df], ignore_index=True)


def lower_miss_rates(long: DataFrame, series_col: str, plan: dict, level: float, level_index: int, w: int, h: int,
                     clip_lower, lower_stats: dict) -> dict:
    """Adaptive CQR split: realised lower miss rate m_lo of every fold, aggregated over all series."""
    schema = StructType([StructField("fold", IntegerType()), StructField("misses", LongType()),
                         StructField("n", LongType())])

    def fn(pdf):
        out = batch_udf_lower(pdf, plan=plan, level=level, w=w, h=h, clip_lower=clip_lower, stats=lower_stats,
                              series_col=series_col)
        return out.astype({"fold": "int32", "misses": "int64", "n": "int64"})

    totals = (
        long.select(series_col, "_bucket", "window", "window_date", "step", "y", "yhat",
                    F.col(f"lo_{level_index}").alias("lo_0"), F.col(f"hi_{level_index}").alias("hi_0"),
                    "sigma", "pool_full", "pool_lowo")
        .groupBy("_bucket").applyInPandas(fn, schema)
        .groupBy("fold").agg(F.sum("misses").alias("misses"), F.sum("n").alias("n"))
        .toPandas()
    )
    return {(level, int(f)): float(m) / float(n) for f, m, n in zip(totals["fold"], totals["misses"], totals["n"])}


def output_schema(series_col: str) -> StructType:
    """Union schema of batch_udf_final: calibration_output rows and metric rows, told apart by kind."""
    return StructType(
        [StructField(c, StringType()) for c in ("kind", "model", "method", "score", "coverage_target", series_col,
                                                "pool_key")]
        + [StructField("step", IntegerType()), StructField("level", DoubleType()),
           StructField("q_lo", DoubleType()), StructField("q_hi", DoubleType()),
           StructField("n_windows", IntegerType()), StructField("shrinkage_weight", DoubleType()),
           StructField("infinite_bound", BooleanType()), StructField("clip_lower", DoubleType()),
           StructField("sigma", DoubleType())]
        + [StructField(c, StringType()) for c in ("abc_class", "xyz_class", "sb_class")]
        + [StructField("window_date", TimestampType()), StructField("metric_name", StringType()),
           StructField("metric_value", DoubleType())]
    )


def calibrate_model_spark(spark: SparkSession, eval_sdf: DataFrame, features: DataFrame, series_col: str,
                          model: str, pi_conf, clip_lower, materializer: Materializer):
    """
    Calibrates one model in Spark, as intervals.conformal.calibrate_model.
    Parameters:
        eval_sdf (DataFrame): The model's evaluation_output rows for the run (series_col as string).
        features (DataFrame): Per-series sigma, classes and group columns (series_col as string).
        materializer (Materializer): Caches the intermediate results; the returned result is cached with it.
    Returns: (DataFrame or None, dict): Rows of output_schema (kind "calibration" or "metric"), and a
        summary (dropped series, W, H, method) for logging. None when there is nothing to calibrate.
    """
    levels = calibration_levels(pi_conf)
    has_quantiles = ("forecast_quantiles" in eval_sdf.columns
                     and eval_sdf.filter(F.col("forecast_quantiles").isNotNull()).limit(1).count() > 0)
    plan = fit_plan(pi_conf, has_quantiles, model)
    eval_sdf = eval_sdf.join(features.select(series_col), on=series_col, how="left_semi")
    if eval_sdf.limit(1).count() == 0:
        return None, {"dropped": 0}
    long, w, h, dropped, n_kept = long_backtest(eval_sdf, series_col, levels, has_quantiles)

    label = label_column(pi_conf)
    feats = features.withColumn("_label", label if label is not None else F.lit(MARGINAL_POOL))
    long = long.join(feats.select(series_col, "sigma", "_label", "abc_class", "xyz_class", "sb_class"),
                     on=series_col, how="inner")
    pools = merged_pools(spark, long.select(series_col, "_label").distinct(), plan, w, label is not None)
    long = materializer.cache(long.join(F.broadcast(pools), on="_label", how="left").drop("_label")
                              .withColumn("_bucket", bucket_column(series_col, n_kept)))
    try:
        adaptive = plan["kind"] == "cqr" and plan["score"] == "asymmetric" and plan["tail_split"] == "adaptive"
        folds = list(_folds(w))
        stats, m_lo = {}, {}
        if plan["kind"] == "conformal":
            # Residual scores do not depend on the level: fold, sort and query them once for all levels.
            cols = score_columns(plan, 0, clip_lower)
            scored = materializer.cache(
                folded(long, w).select("fold", "pool", "step", *[c.alias(n) for n, c in cols.items()]))
            try:
                shared = pool_statistics(spark, scored, {
                    f: [r for lv in levels for r in stat_requests(plan, lv, "main")] for f in folds})
            finally:
                materializer.uncache(scored)
            stats = {lv: shared for lv in levels}
        for i, lv in enumerate([] if plan["kind"] == "conformal" else levels):
            cols = score_columns(plan, i, clip_lower)
            if not cols:
                stats[lv] = {f: {} for f in folds}
                continue
            scored = materializer.cache(folded(long, w).select("fold", "pool", "step", *[c.alias(n) for n, c in cols.items()]))
            try:
                if adaptive:
                    lower = pool_statistics(spark, scored, {f: stat_requests(plan, lv, "lower") for f in folds})
                    m_lo.update(lower_miss_rates(long, series_col, plan, lv, i, w, h, clip_lower, lower))
                stats[lv] = pool_statistics(spark, scored, {
                    f: stat_requests(plan, lv, "main", m_lo.get((lv, f))) for f in folds})
            finally:
                materializer.uncache(scored)

        def final_fn(pdf):
            out = batch_udf_final(pdf, plan=plan, levels=levels, has_band=has_quantiles, w=w, h=h,
                                  clip_lower=clip_lower, stats=stats, m_lo=m_lo, series_col=series_col, model=model)
            for c, t in (("step", "Int32"), ("n_windows", "Int32")):
                if c in out.columns:
                    out[c] = out[c].astype(t)
            return out.reindex(columns=[f.name for f in output_schema(series_col).fields])

        cols = [series_col, "_bucket", "window", "window_date", "step", "y", "yhat", "sigma", "pool_full",
                "pool_lowo", "abc_class", "xyz_class", "sb_class"]
        if has_quantiles:
            cols += [f"{side}_{i}" for i in range(len(levels)) for side in ("lo", "hi")]
        result = long.select(*cols).groupBy("_bucket").applyInPandas(final_fn, output_schema(series_col))
        result = materializer.cache(result)
        result.count()
    finally:
        materializer.uncache(long)
    return result, {"dropped": dropped, "w": w, "h": h, "method": plan["kind"], "score": plan["score"],
                    "target": plan["target"]}
