"""Distributed prediction-interval calibration, moved to mmf_sa.intervals.spark; this module re-exports it.

Existing imports keep working. New code should import from mmf_sa.intervals.spark. Module constants that
are read at run time (such as SERIES_PER_BUCKET) must be patched in mmf_sa.intervals.spark.
"""
from mmf_sa.intervals.spark import (
    FULL, Materializer, SERIES_PER_BUCKET, batch_udf_final, batch_udf_lower, bucket_column, calibrate_model_spark,
    classify, folded, label_column, long_backtest, lower_miss_rates, merged_pools, output_schema, pool_statistics,
    score_columns, series_features,
)

__all__ = [
    "FULL",
    "Materializer",
    "SERIES_PER_BUCKET",
    "batch_udf_final",
    "batch_udf_lower",
    "bucket_column",
    "calibrate_model_spark",
    "classify",
    "folded",
    "label_column",
    "long_backtest",
    "lower_miss_rates",
    "merged_pools",
    "output_schema",
    "pool_statistics",
    "score_columns",
    "series_features",
]
