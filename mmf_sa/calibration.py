"""Prediction-interval code, moved to the mmf_sa.intervals package; this module re-exports it.

Existing imports keep working. The names live in:
- mmf_sa.intervals.config: the prediction_intervals configuration and the level helpers
- mmf_sa.intervals.quantiles: quantile keys, column names and packing used by the models
- mmf_sa.intervals.conformal: the conformal and CQR math, and calibrate_model
- mmf_sa.intervals.features: series scale and classes, group labels, backtest cubes
- mmf_sa.intervals.scoring: applying the calibration at scoring, and the calibration and metric rows

The distributed calibration is in mmf_sa.intervals.spark (see calibration_spark.py). New code should import
from the new modules.
"""
from mmf_sa.intervals.config import (
    DEFAULT_PREDICTION_INTERVALS, calibration_levels, central_levels, check_supported, level_quantiles, model_method,
    requested_quantiles, resolve_clip_lower, resolve_prediction_intervals_conf,
)
from mmf_sa.intervals.quantiles import (
    MIN_SAMPLES_FOR_QUANTILES, QUANTILE_COLUMN_PREFIX, decode_quantile_forecasts, encode_quantile_forecasts,
    forecast_levels, quantile_columns, quantile_key, quantile_maps,
)
from mmf_sa.intervals.conformal import (
    MARGINAL_POOL, OTHER_GROUP, adaptive_upper_level, calibrate_model, conformal_rank, cqr_bounds, cqr_corrections,
    cqr_lower_from_statistics, finalize_bounds, fit_from_statistics, fit_plan, group_tail_quantiles, interval_metrics,
    merge_small_groups, native_half_width, numpy_pool_statistics, pooled_upper_quantile, pools_for, residual_bounds,
    residual_quantiles, score_cubes, stat_requests, tail_quantiles, upper_quantile, window_bounds,
)
from mmf_sa.intervals.features import (
    ABC_SHARES, COMPUTED_GROUP_COLS, FEATURE_PERIODS, SB_ADI, SB_CV2, SEASON_LENGTH, backtest_cubes, classify_series,
    group_columns, group_labels, positive_sigma, series_features,
)
from mmf_sa.intervals.scoring import (
    CORRECTION_FIELDS, _metric_name, apply_calibration, apply_joined_corrections, calibration_rows, correction_column,
    interval_column, interval_columns, median_corrections, metric_rows, quantiles_column, series_bounds,
)

__all__ = [
    "DEFAULT_PREDICTION_INTERVALS",
    "resolve_prediction_intervals_conf",
    "level_quantiles",
    "requested_quantiles",
    "central_levels",
    "resolve_clip_lower",
    "calibration_levels",
    "check_supported",
    "model_method",
    "QUANTILE_COLUMN_PREFIX",
    "MIN_SAMPLES_FOR_QUANTILES",
    "quantile_key",
    "quantile_columns",
    "quantile_maps",
    "forecast_levels",
    "encode_quantile_forecasts",
    "decode_quantile_forecasts",
    "OTHER_GROUP",
    "conformal_rank",
    "tail_quantiles",
    "upper_quantile",
    "merge_small_groups",
    "group_tail_quantiles",
    "residual_quantiles",
    "pooled_upper_quantile",
    "native_half_width",
    "cqr_corrections",
    "residual_bounds",
    "cqr_bounds",
    "finalize_bounds",
    "interval_metrics",
    "fit_plan",
    "score_cubes",
    "pools_for",
    "stat_requests",
    "adaptive_upper_level",
    "numpy_pool_statistics",
    "cqr_lower_from_statistics",
    "fit_from_statistics",
    "window_bounds",
    "calibrate_model",
    "MARGINAL_POOL",
    "SB_ADI",
    "SB_CV2",
    "ABC_SHARES",
    "series_features",
    "classify_series",
    "SEASON_LENGTH",
    "FEATURE_PERIODS",
    "COMPUTED_GROUP_COLS",
    "group_columns",
    "group_labels",
    "positive_sigma",
    "backtest_cubes",
    "_metric_name",
    "calibration_rows",
    "metric_rows",
    "quantiles_column",
    "interval_column",
    "interval_columns",
    "series_bounds",
    "correction_column",
    "CORRECTION_FIELDS",
    "apply_calibration",
    "apply_joined_corrections",
    "median_corrections",
]
