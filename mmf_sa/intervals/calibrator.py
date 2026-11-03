"""Calibration of prediction intervals from the tables that run_forecast writes (calibrate_intervals).

The calibration reads the backtest rows of evaluation_output, with the native quantiles that run_forecast writes for
models with native intervals; the training data, for each series' scale; and the forecasts of scoring_output. It
writes three tables of its own and never modifies the tables it reads, so it runs on its own CPU compute after the
forecast jobs, and can be re-run with other settings without re-running the models. Nothing here collects data that
grows with the number of series to the driver: the calibration runs in intervals.spark.
"""
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

from omegaconf import DictConfig, OmegaConf
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import (
    array_sort, avg, coalesce, col, collect_list, count, create_map, explode, expr, first, isnan, lit, map_keys, max,
    min, row_number, struct, sum as sum_, to_timestamp, when,
)
from pyspark.sql.types import ArrayType, DoubleType, MapType, StringType, StructField, StructType
from pyspark.sql.window import Window

from mmf_sa.exceptions import ConfigurationError
from mmf_sa.intervals import spark as calibration_spark
from mmf_sa.intervals.calibration_config import (
    calibration_levels, check_supported, level_quantiles, model_method, resolve_clip_lower,
    resolve_prediction_intervals_conf,
)
from mmf_sa.intervals.calibration_scoring import (
    CORRECTION_FIELDS, apply_joined_corrections, correction_column, interval_column, interval_columns,
    median_corrections, quantiles_column,
)
from mmf_sa.intervals.features import FEATURE_PERIODS, SEASON_LENGTH, group_columns
from mmf_sa.intervals.quantiles import quantile_key

_logger = logging.getLogger(__name__)

# Column of the forecasts holding the native quantile map, as in evaluation_output and the calibration functions.
NATIVE_FIELD = "forecast_quantiles"


def default_output_prefix(evaluation_output: str) -> str:
    """The output_prefix of calibrate_intervals by default: evaluation_output without its _evaluation_output suffix."""
    suffix = "_evaluation_output"
    return evaluation_output[:-len(suffix)] if evaluation_output.endswith(suffix) else evaluation_output


class IntervalCalibrator:
    """
    Calibrates the prediction intervals of the latest run of each model, from the tables of run_forecast.
    Parameters:
        spark (SparkSession): The Spark session.
        settings (DictConfig): train_data (table name or DataFrame), evaluation_output, scoring_output,
            output_prefix, group_id, date_col, target, freq, models (or None), run_ids ({model: run_id} or None)
            and prediction_intervals (the calibration block, see calibration_config).
    """

    def __init__(self, spark: SparkSession, settings: DictConfig, train_data=None):
        self.spark = spark
        self.settings = settings
        self.train_data = train_data if train_data is not None else settings.train_data
        self.calibration_date = datetime.now()

    @property
    def pi(self):
        return self.settings.prediction_intervals

    def table(self, name: str) -> str:
        """Name of an output table: <output_prefix>_<name>."""
        return f"{self.settings.output_prefix}_{name}"

    def materializer(self) -> "calibration_spark.Materializer":
        """Cache of the calibration steps, with temporary tables next to the outputs as fallback."""
        return calibration_spark.Materializer(self.spark, self.table("pi_tmp"))

    def train(self) -> DataFrame:
        if isinstance(self.train_data, DataFrame):
            return self.train_data
        if isinstance(self.train_data, str):
            return self.spark.table(self.train_data)
        return self.spark.createDataFrame(self.train_data)  # a pandas DataFrame

    # --- Run selection -------------------------------------------------------------------------------------

    def latest_runs(self) -> Dict[str, Dict[str, Any]]:
        """
        The run to calibrate for each model: run_ids[model] if given, else the model's latest run in
        evaluation_output, by run_date. Within a run, the latest run_date (a re-run with the same run_id appends)
        and one metric_name (the backtest rows repeat per metric). Models not in `models` are left out.
        Returns: {model: {"run_id", "run_date", "metric_name"}}.
        """
        ev = self.spark.table(self.settings.evaluation_output)
        models, run_ids = self.settings.models, self.settings.run_ids or {}
        if models:
            ev = ev.filter(col("model").isin(list(models)))
        runs = ev.groupBy("model", "run_id").agg(max("run_date").alias("run_date"),
                                                 min("metric_name").alias("metric_name"))
        rank = row_number().over(Window.partitionBy("model").orderBy(col("run_date").desc(), col("run_id").desc()))
        chosen = {}
        for r in runs.withColumn("_rank", rank).collect():
            wanted = run_ids.get(r["model"])
            if (wanted is None and r["_rank"] == 1) or (wanted is not None and r["run_id"] == wanted):
                chosen[r["model"]] = {"run_id": r["run_id"], "run_date": r["run_date"],
                                      "metric_name": r["metric_name"]}
        missing = sorted((set(models or []) | set(run_ids)) - set(chosen))
        if missing:
            _logger.warning(f"No backtest rows in {self.settings.evaluation_output} for {', '.join(missing)}; "
                            "they are not calibrated.")
        return chosen

    def backtest_rows(self, model: str, run: dict) -> DataFrame:
        """The model's backtest rows of its run: series (string), window, forecast, actual and native quantiles."""
        gid = self.settings.group_id
        ev = self.spark.table(self.settings.evaluation_output).filter(
            (col("model") == model) & (col("run_id") == run["run_id"]) & (col("run_date") == lit(run["run_date"]))
            & (col("metric_name") == run["metric_name"]))
        cols = [col(gid).cast(StringType()).alias(gid), "backtest_window_start_date", "forecast", "actual"]
        if NATIVE_FIELD in ev.columns:
            cols.append(NATIVE_FIELD)
        return ev.select(*cols)

    # --- Calibration ---------------------------------------------------------------------------------------

    def calibration_features(self, first_window: DataFrame, materializer) -> Tuple[DataFrame, float]:
        """
        Per-series features (sigma, classes, user group columns) as a Spark DataFrame, and the minimum of
        the training target.
        Parameters:
            first_window (DataFrame): Series id (string) and first backtest window start date per series.
            materializer (calibration_spark.Materializer): Caches the features (returned cached).
        """
        gid, date_col, target = self.settings.group_id, self.settings.date_col, self.settings.target
        freq = self.settings.freq
        extra_cols = group_columns(self.pi)
        train = self.train()
        missing = [c for c in extra_cols if c not in train.columns]
        if missing:
            raise ConfigurationError(f"prediction_intervals.coverage.group_col: column(s) {missing} "
                                     "not found in train_data")
        min_target = train.agg(min(col(target))).collect()[0][0]
        train = train.select(col(gid).cast(StringType()).alias(gid), to_timestamp(col(date_col)).alias(date_col),
                             col(target).cast(DoubleType()).alias(target), *extra_cols)
        windows = first_window.withColumnRenamed("backtest_window_start_date", "_first_window")
        feats = materializer.cache(calibration_spark.series_features(
            train, windows, gid, date_col, target, SEASON_LENGTH[freq],
            FEATURE_PERIODS[freq], extra_cols))
        cached = feats
        good = col("sigma").isNotNull() & ~isnan(col("sigma")) & (col("sigma") > 0)
        n_bad = feats.filter(~good).count()
        if n_bad:
            median = feats.filter(good).agg(expr("percentile(sigma, 0.5)")).collect()[0][0]
            _logger.warning(f"{n_bad} series have no positive seasonal-naive scale (e.g. an all-zero history); "
                            "they use the median scale of the other series.")
            feats = feats.withColumn("sigma", when(good, col("sigma")).otherwise(lit(1.0 if median is None
                                                                                       else float(median))))
        classified = materializer.cache(calibration_spark.classify(feats, gid))
        classified.count()
        materializer.uncache(cached)
        return classified, 0.0 if min_target is None else float(min_target)

    def run(self) -> Dict[str, Any]:
        """
        Calibrates each model's latest run and writes, for each, its calibration_output rows, its leave-one-window-out
        interval metrics and its calibrated scoring intervals. One model's failure is logged and does not stop the
        others.
        Returns: {"tables": {...}, "models": {model: summary}}.
        """
        pi = self.pi
        check_supported(pi)
        gid = self.settings.group_id
        tables = {"calibration": self.table("evaluation_output_calibration"),
                  "interval_metrics": self.table("evaluation_output_interval_metrics"),
                  "scoring_intervals": self.table("scoring_output_intervals")}
        out = {"tables": tables, "models": {}}
        runs = self.latest_runs()
        if not runs:
            _logger.warning(f"No backtest rows in {self.settings.evaluation_output}; nothing to calibrate.")
            return out
        known = set(runs)
        unknown_methods = set(pi.method_by_model) - known
        if unknown_methods:
            _logger.warning("prediction_intervals.method_by_model names models that are not calibrated, ignored: "
                            f"{', '.join(sorted(unknown_methods))}.")
        materializer = self.materializer()
        try:
            backtests = {m: self.backtest_rows(m, r) for m, r in runs.items()}
            first_window = None
            for rows in backtests.values():
                fw = rows.groupBy(gid).agg(min("backtest_window_start_date").alias("backtest_window_start_date"))
                first_window = fw if first_window is None else first_window.unionByName(fw)
            first_window = first_window.groupBy(gid).agg(
                min("backtest_window_start_date").alias("backtest_window_start_date"))
            features, min_target = self.calibration_features(first_window, materializer)
            clip_lower = resolve_clip_lower(pi.clip_lower, min_target)
            print(f"Prediction intervals: clip_lower resolved to {clip_lower}")
            for model in sorted(runs):
                try:
                    out["models"][model] = self.calibrate_one(model, runs[model], backtests[model], features,
                                                              clip_lower, materializer)
                except Exception as err:
                    # Its tables may already hold some of its rows of this run: the log says so.
                    _logger.error(f"Calibration of {model} failed, its output tables may hold part of its rows for "
                                  f"run {runs[model]['run_id']}; the other models continue: {repr(err)}", exc_info=err)
                    out["models"][model] = {"run_id": runs[model]["run_id"], "methods": {},
                                            "error": repr(err)[:500]}
        finally:
            materializer.close()
        return out

    def native_levels(self, backtest: DataFrame, levels) -> list:
        """The levels whose two native quantiles are in every backtest row that has native quantiles."""
        if NATIVE_FIELD not in backtest.columns:
            return []
        with_map = backtest.filter(col(NATIVE_FIELD).isNotNull())
        n = with_map.count()
        if not n:
            return []
        counts = with_map.select(explode(map_keys(col(NATIVE_FIELD))).alias("key")).groupBy("key").count().collect()
        complete = {r["key"] for r in counts if r["count"] == n}
        return [lv for lv in levels if all(quantile_key(q) in complete for q in level_quantiles(lv))]

    def level_blocks(self, model: str, backtest: DataFrame) -> list:
        """
        The (calibration block, backtest rows) pairs to calibrate a model with. The levels whose native quantiles
        the model returned get native-band methods (CQR by default); the levels it has none for (e.g. 95% for a model
        whose native range is 10-90%, left out by run_forecast) get conformal intervals from the backtest errors,
        which need no native band at scoring.
        """
        pi = self.pi
        levels = calibration_levels(pi)
        if model_method(pi, model) == "conformal":
            return [(pi, backtest)]
        native = self.native_levels(backtest, levels)
        rest = [lv for lv in levels if lv not in native]
        if not rest:
            return [(pi, backtest)]
        plain = backtest.drop(NATIVE_FIELD) if NATIVE_FIELD in backtest.columns else backtest
        if not native:
            return [(pi, plain)]  # calibrate_model warns that the native method falls back to conformal
        _logger.warning(f"{model}: no native quantiles for levels {rest}; they get conformal intervals, and "
                        f"levels {native} get calibrated native bands.")
        with_levels = lambda lv, extra: OmegaConf.merge(pi, OmegaConf.create({"levels": lv, "quantiles": None, **extra}))
        return [(with_levels(native, {}), backtest),
                (with_levels(rest, {"method_by_model": {**pi.method_by_model, model: "conformal"}}), plain)]

    def combined_wis(self, metrics: DataFrame) -> DataFrame:
        """
        wis over all levels from the metric rows of several calibration passes (one row per series, window and
        metric): (0.5 scaled_ae + sum over levels of alpha/2 msis) / (K + 0.5), as conformal.wis_from_metrics.
        """
        gid = self.settings.group_id
        levels = calibration_levels(self.pi)
        keys = [gid, "window_date", "model"]
        terms = metrics.filter(col("metric_name").isin(["scaled_ae"] + [f"msis_{lv:g}" for lv in levels]))
        weight = when(col("metric_name") == "scaled_ae", lit(0.5))
        for lv in levels:
            weight = weight.when(col("metric_name") == f"msis_{lv:g}", lit((1 - lv / 100) / 2))
        return (terms.groupBy(*keys)
                .agg((sum_(weight * col("metric_value")) / lit(len(levels) + 0.5)).alias("metric_value"),
                     count(lit(1)).alias("_terms"))
                .filter(col("_terms") == len(levels) + 1)
                .select(*keys, lit("wis").alias("metric_name"), "metric_value"))

    def calibrate_one(self, model: str, run: dict, backtest: DataFrame, features: DataFrame, clip_lower,
                      materializer) -> Dict[str, Any]:
        """Calibrates one model and writes its rows to the three output tables. Returns a summary."""
        gid = self.settings.group_id
        summary = {"run_id": run["run_id"], "methods": {}}
        backtest = materializer.cache(backtest)
        blocks = self.level_blocks(model, backtest)
        results = []
        try:
            for pi, rows in blocks:
                try:
                    result, info = calibration_spark.calibrate_model_spark(
                        self.spark, rows, features, gid, model, pi, clip_lower, materializer)
                except Exception as err:
                    _logger.error(f"Calibration failed for {model}: {repr(err)}", exc_info=err)
                    return {**summary, "error": repr(err)[:500]}
                if info.get("dropped"):
                    _logger.warning(f"{model}: {info['dropped']} series dropped from calibration "
                                    "(incomplete or non-finite backtest windows).")
                if result is None:
                    _logger.warning(f"{model}: no series to calibrate.")
                    return {**summary, "error": "no series to calibrate"}
                results.append(result)
                for lv in calibration_levels(pi):
                    summary["methods"][f"{lv:g}"] = f"{info.get('method')}/{info.get('score')}/{info.get('target')}"
            cal = results[0].filter(col("kind") == "calibration")
            metrics = results[0].filter(col("kind") == "metric")
            for extra in results[1:]:
                cal = cal.unionByName(extra.filter(col("kind") == "calibration"))
                # Level-free metrics are recomputed over all levels below; scaled_ae comes from the first pass.
                metrics = metrics.unionByName(extra.filter((col("kind") == "metric") & ~col("metric_name").isin(
                    "scaled_ae", "wis", "native_wis")))
            metric_cols = [gid, "window_date", "model", "metric_name", "metric_value"]
            metrics = metrics.select(*metric_cols)
            if len(results) > 1:
                metrics = metrics.filter(~col("metric_name").isin("wis", "native_wis"))
                metrics = metrics.unionByName(self.combined_wis(metrics).select(*metric_cols))
            self.write_calibration_output(cal, run)
            self.write_interval_metrics(metrics, run)
            summary["coverage"] = {}
            for r in (metrics.filter(col("metric_name").rlike("^(native_)?coverage_"))
                      .groupBy("metric_name").agg(avg("metric_value").alias("v"))
                      .orderBy("metric_name").collect()):
                summary["coverage"][r["metric_name"]] = round(float(r["v"]), 4)
                print(f"{model} ({summary['methods']}): leave-one-window-out {r['metric_name']} {r['v']:.3f}")
            summary["scoring_rows"] = self.write_scoring_intervals(model, run, cal)
        finally:
            for result in results:
                materializer.uncache(result)
            materializer.uncache(backtest)
        return summary

    def append(self, sdf: DataFrame, table: str) -> None:
        """
        Appends to an output table with mergeSchema. Columns of the table that the rows lack (e.g. a re-run with
        another output_format) are added as typed NULLs, which tables that are not Delta also need.
        """
        if self.spark.catalog.tableExists(table):
            for field in self.spark.table(table).schema.fields:
                if field.name not in sdf.columns:
                    sdf = sdf.withColumn(field.name, lit(None).cast(field.dataType))
        sdf.write.mode("append").option("mergeSchema", "true").saveAsTable(table)

    def _run_columns(self, sdf: DataFrame, run: dict) -> DataFrame:
        return (sdf.withColumn("run_id", lit(run["run_id"]))
                .withColumn("run_date", lit(run["run_date"]))
                .withColumn("calibration_date", lit(self.calibration_date)))

    def write_calibration_output(self, rows: DataFrame, run: dict):
        gid = self.settings.group_id
        names = ["model", "method", "score", "coverage_target", gid, "pool_key", "step", "level", "q_lo", "q_hi",
                 "n_windows", "shrinkage_weight", "infinite_bound", "clip_lower", "sigma", "abc_class",
                 "xyz_class", "sb_class"]
        sdf = (
            rows.select(*names)
            .withColumn("clip_lower", when(isnan(col("clip_lower")), lit(None)).otherwise(col("clip_lower")))
            .withColumn("shrinkage_weight",
                        when(isnan(col("shrinkage_weight")), lit(None)).otherwise(col("shrinkage_weight")))
        )
        self.append(self._run_columns(sdf, run), self.table("evaluation_output_calibration"))

    def write_interval_metrics(self, rows: DataFrame, run: dict):
        """Appends interval metric rows (series, window, metric_name, metric_value, model)."""
        gid = self.settings.group_id
        sdf = rows.select(gid, col("window_date").alias("backtest_window_start_date"), "metric_name",
                          "metric_value", "model")
        self.append(self._run_columns(sdf, run), self.table("evaluation_output_interval_metrics"))

    # --- Scoring ---------------------------------------------------------------------------------------------

    def native_scoring_map(self, sdf: DataFrame):
        """
        The native quantile map of scoring_output rows, as a column: <target>_quantiles where the row has one, else
        the map rebuilt from the <target>_lo_<L> / <target>_hi_<L> columns (output_format lo_hi); None when there is
        neither column. Per row, so that an all-NULL <target>_quantiles column does not hide the bounds.
        """
        target = self.settings.target
        pairs = []
        for level in calibration_levels(self.pi):
            for side, q in zip(("lo", "hi"), level_quantiles(level)):
                name = interval_column(target, side, level)
                if name in sdf.columns:
                    pairs += [lit(quantile_key(q)), col(name)]
        from_bounds = create_map(*pairs) if pairs else None
        if quantiles_column(target) not in sdf.columns:
            return from_bounds
        if from_bounds is None:
            return col(quantiles_column(target))
        return coalesce(col(quantiles_column(target)), from_bounds)

    def scoring_rows(self, model: str, run: dict) -> DataFrame:
        """The model's forecasts of its run in scoring_output, with the native quantile map as NATIVE_FIELD."""
        gid, date_col, target = self.settings.group_id, self.settings.date_col, self.settings.target
        sc = self.spark.table(self.settings.scoring_output).filter(
            (col("model") == model) & (col("run_id") == run["run_id"]))
        latest = sc.agg(max("run_date")).collect()[0][0]
        sc = sc.filter(col("run_date") == lit(latest))
        native = self.native_scoring_map(sc)
        cols = [col(gid).cast(StringType()).alias(gid), date_col, target, "model"]
        if "use_case" in sc.columns:
            cols.append("use_case")
        out = sc.select(*cols, (native if native is not None
                                else lit(None).cast(MapType(StringType(), ArrayType(DoubleType())))).alias(NATIVE_FIELD))
        # NULL bounds (levels outside a model's native range) are dropped from the map.
        return out.withColumn(NATIVE_FIELD, expr(f"map_filter({NATIVE_FIELD}, (k, v) -> v IS NOT NULL)"))

    def wide_corrections(self, cal: DataFrame, levels: list) -> DataFrame:
        """
        One row per series with its corrections in correction_column(i, field) columns: q_lo and
        q_hi as arrays ordered by step, sigma, method and score, for each level i.
        """
        gid = self.settings.group_id
        per_level = cal.groupBy(gid, "level").agg(
            array_sort(collect_list(struct("step", "q_lo", "q_hi"))).alias("_steps"),
            first("sigma").alias("sigma"), first("method").alias("method"), first("score").alias("score"),
        ).select(gid, "level", col("_steps.q_lo").alias("q_lo"), col("_steps.q_hi").alias("q_hi"),
                 "sigma", "method", "score")
        aggs = []
        for i, level in enumerate(levels):
            for field in CORRECTION_FIELDS:
                aggs.append(first(when(col("level") == float(level), col(field)), ignorenulls=True)
                            .alias(correction_column(i, field)))
        return per_level.groupBy(gid).agg(*aggs)

    def fallback_corrections(self, cal: DataFrame) -> dict:
        """Per-level median corrections for series without calibration rows (small: levels x horizon rows)."""
        steps = cal.groupBy("level", "step").agg(
            expr("percentile(q_lo, 0.5)").alias("q_lo"), expr("percentile(q_hi, 0.5)").alias("q_hi"))
        sigma = cal.filter(col("step") == 1).groupBy("level").agg(expr("percentile(sigma, 0.5)").alias("sigma"))
        kinds = (cal.filter(col("step") == 1).groupBy("level", "method", "score").count()
                 .withColumn("_rank", row_number().over(Window.partitionBy("level").orderBy(col("count").desc())))
                 .filter(col("_rank") == 1).select("level", "method", "score"))
        stats = steps.join(sigma, "level").join(kinds, "level").toPandas()
        return median_corrections(stats)

    def write_scoring_intervals(self, model: str, run: dict, cal: DataFrame) -> int:
        """
        Applies the model's corrections to its forecasts in scoring_output and appends the calibrated interval
        columns to scoring_output_intervals. The corrections are joined to the forecasts and applied per partition,
        so nothing is collected to the driver except the per-level fallback corrections. Returns the row count.
        """
        gid, target = self.settings.group_id, self.settings.target
        levels = calibration_levels(self.pi)
        output_format = self.pi.output_format
        sdf = self.scoring_rows(model, run)
        fallback = self.fallback_corrections(cal)
        clip_row = (cal.select("clip_lower").filter(col("clip_lower").isNotNull() & ~isnan(col("clip_lower")))
                    .limit(1).collect())
        clip_lower = float(clip_row[0][0]) if clip_row else None
        wide = self.wide_corrections(cal, levels)
        n_missing = sdf.select(gid).join(wide.select(gid), on=gid, how="left_anti").count()
        if n_missing:
            _logger.warning(f"{model}: {n_missing} series without calibration rows use the model's "
                            "median correction.")
        joined = sdf.join(wide, on=gid, how="left")

        fields = [f for f in sdf.schema.fields if f.name != NATIVE_FIELD]
        for name in interval_columns(target, levels, output_format):
            dtype = (MapType(StringType(), ArrayType(DoubleType())) if name == quantiles_column(target)
                     else ArrayType(DoubleType()))
            fields.append(StructField(name, dtype))
        schema = StructType(fields)
        names = [f.name for f in fields]

        def apply_batches(batches):
            for batch in batches:
                out, _ = apply_joined_corrections(
                    batch.reset_index(drop=True), gid, target, levels, output_format, clip_lower, fallback)
                yield out[names]

        calibrated = self._run_columns(joined.mapInPandas(apply_batches, schema), run)
        self.append(calibrated, self.table("scoring_output_intervals"))
        return self.spark.table(self.table("scoring_output_intervals")).filter(
            (col("model") == model) & (col("calibration_date") == lit(self.calibration_date))).count()


def _argument_or_setting(name: str, value, default, block: dict):
    """
    The value of a calibrate_intervals argument that is also a key of `settings` (block). An argument left at its
    default takes the value in settings; an argument set to another value than settings' is a conflict.
    """
    given = block.get(name)
    if not given:
        return value
    if value == default:
        return given
    if value != given:
        raise ConfigurationError(f"calibrate_intervals: {name} is set both as an argument ({value!r}) and in settings "
                                 f"({given!r}); set it in one place")
    return value


def calibrate_intervals(
    spark: SparkSession,
    train_data: Union[str, DataFrame, Any],
    evaluation_output: str,
    scoring_output: str,
    group_id: str = "unique_id",
    date_col: str = "ds",
    target: str = "y",
    freq: str = "D",
    levels: Optional[List[float]] = None,
    quantiles: Optional[List[float]] = None,
    output_prefix: Optional[str] = None,
    models: Optional[List[str]] = None,
    run_ids: Optional[Dict[str, str]] = None,
    method: str = "auto",
    method_by_model: Optional[Dict[str, str]] = None,
    pool: str = "marginal",
    group_col: Union[str, List[str], None] = None,
    shrinkage_kappa: float = 10,
    output_format: str = "quantile_map",
    settings: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Calibrates the prediction intervals of the latest run of each model, from the tables that run_forecast wrote.
    Run it after the forecast jobs, on CPU compute: it does not use the models, only their backtest and forecasts.
    Models whose backtest has native quantiles (prediction_intervals enabled in run_forecast) get their native bands
    corrected (CQR); the others get conformal intervals from their backtest errors.
    Parameters:
        spark (SparkSession): The Spark session.
        train_data: The training data of run_forecast (table name or DataFrame), for each series' scale.
        evaluation_output (str): The evaluation_output table of run_forecast (backtest rows).
        scoring_output (str): The scoring_output table of run_forecast (forecasts).
        group_id, date_col, target, freq (str): As in run_forecast.
        levels (List[float]): Central interval levels in percent. Default [80, 95]. Or set `quantiles`.
        quantiles (List[float]): Quantile levels in central pairs, e.g. [0.1, 0.5, 0.9]; replaces `levels`.
        output_prefix (str): Prefix of the output tables. Default: evaluation_output without its
            "_evaluation_output" suffix, e.g. "cat.sch.uc" for "cat.sch.uc_evaluation_output".
        models (List[str]): Models to calibrate. Default: every model in evaluation_output.
        run_ids (Dict[str, str]): The run to calibrate per model. Default: each model's latest run.
        method (str): auto (CQR on native bands where available, else conformal), conformal, native or cqr.
        method_by_model (Dict[str, str]): Per-model override of `method`.
        pool (str): "marginal" (default): the CQR corrections are pooled over all series, per horizon step (one pool,
            no per-series shrinkage). The conformal intervals pool their scores the same way, and shrink each series'
            own tail toward that pool (coverage.target "series", the default). "group": pooled per group of group_col.
        group_col: With pool="group", the groups: a column of train_data, a list of columns, or "auto" for the ABC x
            XYZ classes that MMF computes (volume and volatility classes, validated on M5 retail demand only).
        shrinkage_kappa (float): Strength of the shrinkage of each series toward its pool: weight W / (W + kappa) on
            the series' own tail quantiles, W its number of backtest windows. It applies to the conformal intervals,
            and to the CQR corrections only with cqr_pool "series".
        output_format (str): quantile_map, lo_hi or both, for scoring_output_intervals.
        settings (Dict[str, Any]): Other calibration settings (cqr_score, cqr_tail_split, cqr_pool, coverage,
            score, clip_lower), see calibration_config. The settings that are also arguments (method, method_by_model,
            output_format, levels, quantiles, cqr_pool for pool, and coverage's group_col and shrinkage_kappa) take the
            argument when it is not the default, and the settings' value otherwise; different values for both raise.
    Returns: Dict with the output table names ("tables") and a summary per model ("models").
    Writes, appending, with run_id, run_date (of the forecast run) and calibration_date:
        <output_prefix>_evaluation_output_calibration: the corrections per model, series, step and level;
        <output_prefix>_evaluation_output_interval_metrics: leave-one-window-out coverage, miss rates, MSIS, width,
            scaled_ae and WIS per series and backtest window;
        <output_prefix>_scoring_output_intervals: the calibrated interval columns of each forecast.
    """
    if pool not in ("marginal", "group"):
        raise ConfigurationError(f"calibrate_intervals: pool must be 'marginal' or 'group', got {pool!r}")
    if (pool == "group") != (group_col is not None):
        raise ConfigurationError("calibrate_intervals: pool='group' needs group_col, and group_col needs pool='group'")
    block = dict(settings or {})
    coverage = dict(block.pop("coverage", None) or {})
    block["cqr_pool"] = _argument_or_setting("cqr_pool", pool, "marginal", block)
    coverage["group_col"] = _argument_or_setting("group_col", group_col, None, coverage)
    coverage["shrinkage_kappa"] = _argument_or_setting("shrinkage_kappa", shrinkage_kappa, 10, coverage)
    block["enabled"] = True
    block["method"] = _argument_or_setting("method", method, "auto", block)
    block["method_by_model"] = _argument_or_setting("method_by_model", method_by_model, None, block) or {}
    block["output_format"] = _argument_or_setting("output_format", output_format, "quantile_map", block)
    block["coverage"] = coverage
    levels = _argument_or_setting("levels", levels, None, block)
    quantiles = _argument_or_setting("quantiles", quantiles, None, block)
    if levels is not None and quantiles is not None:
        raise ConfigurationError("calibrate_intervals: give levels or quantiles, not both")
    if quantiles is not None:
        block["quantiles"] = quantiles
    if levels is not None:
        block["levels"] = levels
    conf = OmegaConf.create({
        "evaluation_output": evaluation_output, "scoring_output": scoring_output,
        "train_data": train_data if isinstance(train_data, str) else None,
        "output_prefix": output_prefix or default_output_prefix(evaluation_output),
        "group_id": group_id, "date_col": date_col, "target": target, "freq": freq,
        "models": list(models) if models else None, "run_ids": dict(run_ids) if run_ids else None,
    })
    conf["prediction_intervals"] = resolve_prediction_intervals_conf(block)
    return IntervalCalibrator(spark, conf, train_data=train_data).run()
