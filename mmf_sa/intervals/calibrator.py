"""Prediction intervals of a Forecaster: the native-quantile settings of its models, the calibration of this run's
backtest, and the interval columns of the scoring output.

The Forecaster creates an IntervalCalibrator and delegates its prediction-interval methods to it. Nothing here
collects data that grows with the number of series to the driver: the calibration runs in intervals.spark.
"""
import logging
from typing import Tuple

import pandas as pd
from omegaconf import OmegaConf
from pyspark.sql import DataFrame
from pyspark.sql.functions import (
    array_sort, avg, col, collect_list, expr, first, isnan, lit, max, min, row_number, struct, to_timestamp, when,
)
from pyspark.sql.types import (
    ArrayType, DoubleType, LongType, MapType, StringType, StructField, StructType, TimestampType,
)
from pyspark.sql.window import Window

from mmf_sa.exceptions import ConfigurationError
from mmf_sa.intervals import spark as calibration_spark
from mmf_sa.intervals.config import (
    calibration_levels, check_supported, model_method, requested_quantiles, resolve_clip_lower,
)
from mmf_sa.intervals.features import FEATURE_PERIODS, SEASON_LENGTH, group_columns
from mmf_sa.intervals.scoring import (
    CORRECTION_FIELDS, apply_joined_corrections, correction_column, interval_columns, median_corrections,
    quantiles_column,
)
from mmf_sa.models.abstract_model import ForecastingRegressor

_logger = logging.getLogger(__name__)


class IntervalCalibrator:
    """
    The prediction intervals of a Forecaster: the native-quantile settings of its models, the calibration of this
    run's backtest (run_calibration), and the interval columns of the scoring output (add_prediction_intervals).

    The calibrator reads the state of its Forecaster when it is used (conf, spark, model_registry, run_id, run_date,
    resolve_source), so a value that the Forecaster gets after construction, such as run_date, applies here too.
    Parameters:
        forecaster (Forecaster): The Forecaster that owns the run.
    """

    def __init__(self, forecaster):
        self._forecaster = forecaster
        # Known models that are not active are skipped silently: the GPU orchestrator runs one model per Forecaster.
        unknown_methods = set(self.conf.prediction_intervals.method_by_model) - set(
            self.model_registry.all_models_conf)
        if unknown_methods:
            _logger.warning("prediction_intervals.method_by_model names unknown models, ignored: "
                            f"{', '.join(sorted(unknown_methods))}.")

    @property
    def conf(self):
        return self._forecaster.conf

    @property
    def spark(self):
        return self._forecaster.spark

    @property
    def model_registry(self):
        return self._forecaster.model_registry

    @property
    def run_id(self) -> str:
        return self._forecaster.run_id

    @property
    def run_date(self):
        return self._forecaster.run_date

    @property
    def intervals_enabled(self) -> bool:
        return self._forecaster.intervals_enabled

    def resolve_source(self, key: str) -> DataFrame:
        return self._forecaster.resolve_source(key)

    def native_quantiles(self, model_conf) -> list:
        """
        Quantile levels a model must produce in its backtest, or None.
        Only models flagged native_intervals produce them, and only when the model's interval method
        (method_by_model, else method) uses native bands: auto, native or cqr.
        """
        pi = self.conf.prediction_intervals
        if not pi.enabled or not model_conf.get("native_intervals", False):
            return None
        if model_method(pi, model_conf["name"]) == "conformal":
            return None
        return requested_quantiles(pi)

    def get_model_with_native_quantiles(self, model_conf) -> ForecastingRegressor:
        """Instantiates a model for evaluation or scoring, passing it native_quantiles when it must produce
        native bands."""
        quantiles = self.native_quantiles(model_conf)
        override = OmegaConf.create({"native_quantiles": quantiles}) if quantiles else None
        return self.model_registry.get_model(model_conf["name"], override_conf=override)

    def evaluation_quantile_fields(self) -> list:
        """The forecast_quantiles field of evaluation rows, present only when intervals are enabled."""
        if not self.intervals_enabled:
            return []
        return [StructField("forecast_quantiles", MapType(StringType(), ArrayType(DoubleType())))]

    def evaluation_writer(self, sdf: DataFrame):
        """Append writer for evaluation_output. With intervals enabled, existing tables gain forecast_quantiles."""
        writer = sdf.write.mode("append")
        if self.intervals_enabled:
            writer = writer.option("mergeSchema", "true")
        return writer

    def calibration_output_table(self) -> str:
        """Name of the calibration_output table: the calibration_output setting, or <evaluation_output>_calibration."""
        return self.conf.get("calibration_output") or f"{self.conf['evaluation_output']}_calibration"

    def interval_metrics_table(self) -> str:
        """Name of the interval metrics table: the interval_metrics_output setting, or
        <evaluation_output>_interval_metrics. Kept apart from evaluation_output, whose readers average
        metric_value per series and model without filtering metric_name."""
        return self.conf.get("interval_metrics_output") or f"{self.conf['evaluation_output']}_interval_metrics"

    def materializer(self) -> "calibration_spark.Materializer":
        """Cache of the prediction-interval steps, with temporary tables next to evaluation_output as fallback."""
        return calibration_spark.Materializer(self.spark, f"{self.conf['evaluation_output']}_pi_tmp")

    def calibration_features(self, first_window: DataFrame, materializer) -> Tuple[DataFrame, float]:
        """
        Per-series features (sigma, classes, user group columns) as a Spark DataFrame, and the minimum of
        the training target.
        Parameters:
            first_window (DataFrame): Series id (string) and first backtest window start date per series.
            materializer (calibration_spark.Materializer): Caches the features (returned cached).
        """
        gid, date_col, target = self.conf["group_id"], self.conf["date_col"], self.conf["target"]
        freq = self.conf["freq"]
        extra_cols = group_columns(self.conf.prediction_intervals)
        train = self.resolve_source("train_data")
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

    def run_calibration(self):
        """
        Calibrates prediction intervals from the backtest results of this run in evaluation_output
        in Spark: nothing that grows with the
        number of series is collected to the driver. Writes one calibration_output row per model, series,
        horizon step and level, and the leave-one-window-out interval metrics to the interval metrics
        table (interval_metrics_table).
        Parameters: self (Forecaster): A Forecaster object.
        """
        pi = self.conf.prediction_intervals
        check_supported(pi)
        gid = self.conf["group_id"]
        print("Starting run_calibration")
        # Only the models of this call: the GPU orchestrator runs one model per run_forecast call with a shared
        # run_id, and earlier calls have already calibrated theirs.
        active = list(self.model_registry.get_active_model_keys())
        eval_sdf = self.spark.table(self.conf["evaluation_output"]).filter(
            (col("run_id") == self.run_id) & (col("metric_name") == self.conf["metric"]) & col("model").isin(active))
        # A re-run with the same run_id appends a second backtest; keep each model's latest one.
        eval_sdf = eval_sdf.withColumn("_latest", max("run_date").over(Window.partitionBy("model"))) \
            .filter(col("run_date") == col("_latest"))
        cols = [col(gid).cast(StringType()).alias(gid), "backtest_window_start_date", "model", "forecast", "actual"]
        if "forecast_quantiles" in eval_sdf.columns:
            cols.append("forecast_quantiles")
        materializer = self.materializer()
        eval_sdf = materializer.cache(eval_sdf.select(*cols))
        models = [r[0] for r in eval_sdf.select("model").distinct().collect()]
        if not models:
            materializer.close()
            _logger.warning(f"No backtest rows for run_id {self.run_id} in {self.conf['evaluation_output']}; "
                            "nothing to calibrate.")
            return
        features, min_target = self.calibration_features(
            eval_sdf.groupBy(gid).agg(min("backtest_window_start_date").alias("backtest_window_start_date")),
            materializer)
        clip_lower = resolve_clip_lower(pi.clip_lower, min_target)
        print(f"Prediction intervals: clip_lower resolved to {clip_lower}")
        try:
            for model_name in sorted(models):
                try:
                    result, summary = calibration_spark.calibrate_model_spark(
                        self.spark, eval_sdf.filter(col("model") == model_name).drop("model"), features, gid,
                        model_name, pi, clip_lower, materializer)
                except Exception as err:
                    # One model's failure must not stop the others, nor scoring (its intervals will be null).
                    _logger.error(f"Calibration failed for {model_name}: {repr(err)}", exc_info=err)
                    continue
                if summary.get("dropped"):
                    _logger.warning(f"{model_name}: {summary['dropped']} series dropped from calibration "
                                    "(incomplete or non-finite backtest windows).")
                if result is None:
                    _logger.warning(f"{model_name}: no series to calibrate.")
                    continue
                try:
                    self.write_calibration_output(result.filter(col("kind") == "calibration"))
                    metrics = result.filter(col("kind") == "metric")
                    self.write_interval_metrics(metrics)
                    for r in (metrics.filter(col("metric_name").rlike("^(coverage|msis)_"))
                              .groupBy("metric_name").agg(avg("metric_value").alias("v"))
                              .orderBy("metric_name").collect()):
                        print(f"{model_name} ({summary['method']}, {summary['score']}, {summary['target']}): "
                              f"leave-one-window-out {r['metric_name']} {r['v']:.3f}")
                finally:
                    materializer.uncache(result)
        finally:
            materializer.close()
        print("Finished run_calibration")

    def write_calibration_output(self, rows: DataFrame):
        gid = self.conf["group_id"]
        names = ["model", "method", "score", "coverage_target", gid, "pool_key", "step", "level", "q_lo", "q_hi",
                 "n_windows", "shrinkage_weight", "infinite_bound", "clip_lower", "sigma", "abc_class",
                 "xyz_class", "sb_class"]
        (
            rows.select(*names)
            .withColumn("clip_lower", when(isnan(col("clip_lower")), lit(None)).otherwise(col("clip_lower")))
            .withColumn("shrinkage_weight",
                        when(isnan(col("shrinkage_weight")), lit(None)).otherwise(col("shrinkage_weight")))
            .withColumn("run_id", lit(self.run_id))
            .withColumn("run_date", lit(self.run_date))
            .withColumn("use_case", lit(self.conf["use_case_name"]))
            .write.mode("append").option("mergeSchema", "true")
            .saveAsTable(self.calibration_output_table())
        )

    def write_interval_metrics(self, rows: DataFrame):
        """Appends interval metric rows (series, window, metric_name, metric_value, model) to the interval metrics table."""
        gid = self.conf["group_id"]
        (
            rows.select(gid, col("window_date").alias("backtest_window_start_date"), "metric_name",
                        "metric_value", "model")
            .withColumn("run_id", lit(self.run_id))
            .withColumn("run_date", lit(self.run_date))
            .withColumn("use_case", lit(self.conf["use_case_name"]))
            .write.mode("append").option("mergeSchema", "true")
            .saveAsTable(self.interval_metrics_table())
        )

    def foundation_scoring_schema(self, prediction_df: pd.DataFrame) -> StructType:
        """Explicit schema of a foundation model's predictions with forecast_quantiles (map inference is unreliable)."""
        fields = []
        for c in prediction_df.columns:
            if c == "forecast_quantiles":
                fields.append(StructField(c, MapType(StringType(), ArrayType(DoubleType()))))
            elif c == self.conf["date_col"]:
                fields.append(StructField(c, ArrayType(TimestampType())))
            elif c == self.conf["target"]:
                fields.append(StructField(c, ArrayType(DoubleType())))
            elif c == "index":
                fields.append(StructField(c, LongType()))
            else:
                fields.append(StructField(c, StringType()))
        return StructType(fields)

    def calibration_rows_for(self, model_name: str):
        """This run's calibration_output rows of one model, as a Spark DataFrame, or None when there are none."""
        try:
            cal = self.spark.table(self.calibration_output_table()).filter(
                (col("run_id") == self.run_id) & (col("model") == model_name))
            latest = cal.agg(max("run_date")).collect()[0][0]
            if latest is None:
                return None
            # A re-run with the same run_id appends a second calibration; keep only the latest one.
            return cal.filter(col("run_date") == lit(latest))
        except Exception as err:
            _logger.warning(f"Cannot read calibration_output ({err}).")
            return None

    def wide_corrections(self, cal: DataFrame, levels: list) -> DataFrame:
        """
        One row per series with its corrections in correction_column(i, field) columns: q_lo and
        q_hi as arrays ordered by step, sigma, method and score, for each level i.
        """
        gid = self.conf["group_id"]
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
        gid = self.conf["group_id"]
        steps = cal.groupBy("level", "step").agg(
            expr("percentile(q_lo, 0.5)").alias("q_lo"), expr("percentile(q_hi, 0.5)").alias("q_hi"))
        sigma = cal.filter(col("step") == 1).groupBy("level").agg(expr("percentile(sigma, 0.5)").alias("sigma"))
        kinds = (cal.filter(col("step") == 1).groupBy("level", "method", "score").count()
                 .withColumn("_rank", row_number().over(Window.partitionBy("level").orderBy(col("count").desc())))
                 .filter(col("_rank") == 1).select("level", "method", "score"))
        stats = steps.join(sigma, "level").join(kinds, "level").toPandas()
        return median_corrections(stats)

    def add_prediction_intervals(self, sdf: DataFrame, model_name: str) -> DataFrame:
        """
        Applies this run's calibration_output of one model to its scoring rows (one row per series). The
        corrections are joined to the forecasts and applied per partition, so nothing is collected to the
        driver except the per-level fallback corrections.
        """
        pi = self.conf.prediction_intervals
        gid, target = self.conf["group_id"], self.conf["target"]
        levels = calibration_levels(pi)
        output_format = pi.output_format
        cal = self.calibration_rows_for(model_name)
        if cal is None:
            _logger.warning(f"{model_name}: no calibration rows for run_id {self.run_id}; "
                            "interval columns will be null.")
            fallback, clip_lower, joined = {}, None, sdf
        else:
            fallback = self.fallback_corrections(cal)
            clip_row = (cal.select("clip_lower").filter(col("clip_lower").isNotNull() & ~isnan(col("clip_lower")))
                        .limit(1).collect())
            clip_lower = float(clip_row[0][0]) if clip_row else None
            wide = self.wide_corrections(cal, levels)
            n_missing = sdf.select(gid).join(wide.select(gid), on=gid, how="left_anti").count()
            if n_missing:
                _logger.warning(f"{model_name}: {n_missing} series without calibration rows use the model's "
                                "median correction.")
            joined = sdf.join(wide, on=gid, how="left")

        fields = [f for f in sdf.schema.fields if f.name != "forecast_quantiles"]
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

        return joined.mapInPandas(apply_batches, schema)
