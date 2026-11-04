"""Unit tests for the native quantile maps and the interval columns of scoring_output.

Tests cover:
- sorted_quantiles: quantiles sorted per step (they never cross), missing or non-finite keys dropped
- the quantile maps of local (forecast_quantiles) and global/foundation (quantile_maps) forecasts
- add_interval_columns: output_format columns and their names, NULL bounds for a missing quantile or a model
  without native intervals, also under ANSI mode
- The local scoring UDF with native quantiles, and statsforecast forecast() quantiles
"""
import functools

import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

from mmf_sa import Forecaster
from mmf_sa.intervals.quantiles import QUANTILE_COLUMN_PREFIX, quantile_maps, sorted_quantiles
from mmf_sa.intervals.scoring import QUANTILE_MAP_TYPE, add_interval_columns, interval_columns
from mmf_sa.models.abstract_model import ForecastingRegressor
from mmf_sa.models.statsforecast.StatsFcForecastingPipeline import StatsFcAutoETS
from pyspark.sql import functions as F
from pyspark.sql.types import ArrayType, DoubleType


# --- Quantile maps ------------------------------------------------------------------------------------------


def test_sorted_quantiles_sorts_each_step_across_levels():
    out = sorted_quantiles({"0.9": [3.0, 1.0], "0.1": [1.0, 2.0], "0.5": [2.0, 3.0]})
    assert list(out) == ["0.1", "0.5", "0.9"]
    assert out == {"0.1": [1.0, 1.0], "0.5": [2.0, 2.0], "0.9": [3.0, 3.0]}


def test_sorted_quantiles_drops_missing_and_non_finite_keys():
    out = sorted_quantiles({"0.025": [np.nan, np.nan], "0.1": [1.0, 2.0], "0.5": [2.0, 3.0], "0.9": None,
                            "0.975": [9.0, np.inf]})
    assert out == {"0.1": [1.0, 2.0], "0.5": [2.0, 3.0]}
    assert sorted_quantiles({"0.1": [np.nan]}) is None and sorted_quantiles({}) is None


def test_forecast_quantiles_of_a_local_forecast_are_sorted():
    frame = pd.DataFrame({"y": [1.0, 2.0], f"{QUANTILE_COLUMN_PREFIX}0.1": [1.5, 0.0],
                          f"{QUANTILE_COLUMN_PREFIX}0.9": [1.0, 4.0]})
    assert ForecastingRegressor.forecast_quantiles(frame) == {"0.1": [1.0, 0.0], "0.9": [1.5, 4.0]}
    assert ForecastingRegressor.forecast_quantiles(frame[["y"]]) is None


def test_quantile_maps_per_series():
    frame = pd.DataFrame({"uid": ["a", "b"], "y": [[1.0], [2.0]],
                          f"{QUANTILE_COLUMN_PREFIX}0.1": [[0.5], [np.nan]],
                          f"{QUANTILE_COLUMN_PREFIX}0.9": [[1.5], [np.nan]]})
    assert quantile_maps(frame, "uid") == {"a": {"0.1": [0.5], "0.9": [1.5]}, "b": None}
    assert quantile_maps(frame[["uid", "y"]], "uid") == {}


# --- Interval columns of scoring_output ---------------------------------------------------------------------


QMAP = {"0.025": [0.0, 1.0], "0.1": [1.0, 2.0], "0.5": [2.0, 3.0], "0.9": [3.0, 4.0], "0.975": [4.0, 5.0]}


def _forecasts(spark, maps):
    """One forecast row per quantile map (None: NULL map), built from column expressions, so that neither a Python
    worker nor the SQL parser is needed."""
    def map_column(m):
        if m is None:
            return F.lit(None).cast(QUANTILE_MAP_TYPE)
        pairs = [c for k, v in m.items() for c in (F.lit(k), F.array(*[F.lit(float(x)) for x in v]))]
        return F.create_map(*pairs).cast(QUANTILE_MAP_TYPE)
    frames = [spark.range(1).select(F.lit(f"s{i}").alias("uid"), F.array(F.lit(2.0), F.lit(3.0)).alias("y"),
                                    map_column(m).alias("forecast_quantiles")) for i, m in enumerate(maps)]
    return functools.reduce(lambda a, b: a.unionByName(b), frames)


@pytest.mark.parametrize("fmt, columns", [
    ("quantile_map", ["y_quantiles"]),
    ("lo_hi", ["y_lo_80", "y_hi_80", "y_lo_97_5", "y_hi_97_5"]),
    ("both", ["y_quantiles", "y_lo_80", "y_hi_80", "y_lo_97_5", "y_hi_97_5"]),
])
def test_output_formats(spark_session, fmt, columns):
    assert interval_columns("y", [80, 97.5], fmt) == columns
    qmap = {**QMAP, "0.0125": [-1.0, 0.0], "0.9875": [5.0, 6.0]}
    out = add_interval_columns(_forecasts(spark_session, [qmap]), "y", [97.5, 80], fmt)
    assert out.columns == ["uid", "y"] + columns
    row = out.collect()[0].asDict()
    if fmt != "lo_hi":
        assert dict(row["y_quantiles"]) == qmap
    if fmt != "quantile_map":
        assert (row["y_lo_80"], row["y_hi_80"]) == (QMAP["0.1"], QMAP["0.9"])
        assert (row["y_lo_97_5"], row["y_hi_97_5"]) == ([-1.0, 0.0], [5.0, 6.0])


@pytest.mark.parametrize("ansi", ["true", "false"])
def test_a_level_without_both_quantiles_gets_null_bounds(spark_session, ansi):
    spark_session.conf.set("spark.sql.ansi.enabled", ansi)
    try:
        # The model left out 0.025 (outside its native range): the 95% bounds are NULL, both of them.
        no_tails = {k: v for k, v in QMAP.items() if k != "0.025"}
        rows = add_interval_columns(_forecasts(spark_session, [no_tails, None]), "y", [80, 95], "both").collect()
    finally:
        spark_session.conf.unset("spark.sql.ansi.enabled")
    assert rows[0]["y_lo_80"] == QMAP["0.1"] and rows[0]["y_lo_95"] is None and rows[0]["y_hi_95"] is None
    assert all(rows[1][c] is None for c in ("y_quantiles", "y_lo_80", "y_hi_80", "y_lo_95", "y_hi_95"))


def test_a_model_without_native_intervals_gets_null_columns(spark_session):
    sdf = _forecasts(spark_session, [QMAP]).drop("forecast_quantiles")
    out = add_interval_columns(sdf, "y", [80], "both")
    assert out.schema["y_quantiles"].dataType == QUANTILE_MAP_TYPE
    assert out.schema["y_lo_80"].dataType == ArrayType(DoubleType())
    row = out.collect()[0]
    assert row["y"] == [2.0, 3.0] and row["y_quantiles"] is None and row["y_hi_80"] is None


# --- Local models ------------------------------------------------------------------------------------------


def _ets(quantiles=None):
    params = dict(group_id="uid", date_col="date", target="y", freq="D", prediction_length=7, metric="smape",
                  stride=7, backtest_length=14, model_spec={"season_length": 7, "model": "ZNZ"})
    if quantiles:
        params["native_quantiles"] = quantiles
    return StatsFcAutoETS(OmegaConf.create(params))


def _series(n=120):
    rng = np.random.default_rng(0)
    return pd.DataFrame({"uid": "a", "date": pd.date_range("2023-01-01", periods=n),
                         "y": rng.poisson(3, n).astype(float)})


def test_statsforecast_forecast_quantiles():
    plain, _ = _ets().forecast(_series())
    with_q, _ = _ets([0.1, 0.5, 0.9]).forecast(_series())
    np.testing.assert_array_equal(plain["y"], with_q["y"])
    assert [c for c in with_q.columns if c.startswith(QUANTILE_COLUMN_PREFIX)] == ["__q_0.1", "__q_0.5", "__q_0.9"]


@pytest.mark.parametrize("with_quantiles", [False, True])
def test_local_scoring_udf(with_quantiles):
    out = Forecaster.score_one_local_model(_series(), _ets([0.1, 0.5, 0.9]), with_quantiles=with_quantiles)
    assert ("forecast_quantiles" in out.columns) == with_quantiles
    if with_quantiles:
        fq = out["forecast_quantiles"].iloc[0]
        assert list(fq) == ["0.1", "0.5", "0.9"] and len(fq["0.9"]) == 7
        cube = np.stack([fq[k] for k in fq])
        assert (np.diff(cube, axis=0) >= 0).all()
