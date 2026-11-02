"""The interval columns of scoring_output, built from the native quantile map of each forecast.

Spark column expressions only: no Python UDF runs on the workers, which on serverless GPU do not have all of mmf_sa's
dependencies. The quantile maps are already sorted per step by the models (quantiles.sorted_quantiles), so the
intervals of a series are nested.
"""
from pyspark.sql import Column, DataFrame
from pyspark.sql.functions import col, expr, lit, when
from pyspark.sql.types import ArrayType, DoubleType, MapType, StringType

from mmf_sa.intervals.config import level_quantiles
from mmf_sa.intervals.quantiles import quantile_key

# Column of a model's forecasts holding its native quantile map, before the interval columns replace it.
QUANTILES_FIELD = "forecast_quantiles"

QUANTILE_MAP_TYPE = MapType(StringType(), ArrayType(DoubleType()))


def quantiles_column(target: str) -> str:
    """Name of the quantile-map column of scoring_output (output_format quantile_map or both)."""
    return f"{target}_quantiles"


def interval_column(target: str, side: str, level: float) -> str:
    """Name of a bound column of scoring_output (output_format lo_hi or both), e.g. y_lo_80 or y_hi_97_5."""
    return f"{target}_{side}_{level:g}".replace(".", "_")


def interval_columns(target: str, levels, output_format: str) -> list:
    """Names of the interval columns that scoring_output gets for an output_format."""
    cols = []
    if output_format in ("quantile_map", "both"):
        cols.append(quantiles_column(target))
    if output_format in ("lo_hi", "both"):
        cols += [interval_column(target, side, lv) for lv in levels for side in ("lo", "hi")]
    return cols


def _bound(key: str) -> Column:
    # try_element_at: NULL for a missing key, also under ANSI mode (serverless), where map[key] raises.
    return expr(f"try_element_at({QUANTILES_FIELD}, '{key}')")


def add_interval_columns(sdf: DataFrame, target: str, levels, output_format: str) -> DataFrame:
    """
    Replaces the native quantile map of one model's forecasts (QUANTILES_FIELD, absent for models without native
    intervals) with the interval columns of scoring_output (interval_columns). A level's bounds are NULL when the
    model did not return both of its quantiles.
    """
    has_map = QUANTILES_FIELD in sdf.columns
    null_bound = lit(None).cast(ArrayType(DoubleType()))
    out = sdf
    if output_format in ("quantile_map", "both"):
        out = out.withColumn(quantiles_column(target),
                             col(QUANTILES_FIELD) if has_map else lit(None).cast(QUANTILE_MAP_TYPE))
    if output_format in ("lo_hi", "both"):
        for lv in sorted(levels):
            lo, hi = (_bound(quantile_key(q)) for q in level_quantiles(lv)) if has_map else (null_bound, null_bound)
            both = lo.isNotNull() & hi.isNotNull()
            out = (out.withColumn(interval_column(target, "lo", lv), when(both, lo).otherwise(null_bound))
                   .withColumn(interval_column(target, "hi", lv), when(both, hi).otherwise(null_bound)))
    return out.drop(QUANTILES_FIELD) if has_map else out
