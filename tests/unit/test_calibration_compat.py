"""The former modules mmf_sa.calibration and mmf_sa.calibration_spark keep re-exporting the code moved to
mmf_sa.intervals, so existing imports keep working.

Tests cover:
- every name they export is the object of the mmf_sa.intervals module that defines it
- the main entry points that external code used are among them
"""
import importlib

import pytest

from mmf_sa import calibration, calibration_spark

INTERVAL_MODULES = [importlib.import_module(f"mmf_sa.intervals.{m}")
                    for m in ("config", "quantiles", "conformal", "features", "scoring")]


def _home(name):
    homes = [m for m in INTERVAL_MODULES if name in vars(m)]
    return homes[0] if homes else None


@pytest.mark.parametrize("name", calibration.__all__)
def test_calibration_names_are_the_moved_objects(name):
    home = _home(name)
    assert home is not None, f"{name} is not defined in mmf_sa.intervals"
    assert getattr(calibration, name) is getattr(home, name)


@pytest.mark.parametrize("name", calibration_spark.__all__)
def test_calibration_spark_names_are_the_moved_objects(name):
    spark = importlib.import_module("mmf_sa.intervals.spark")
    assert getattr(calibration_spark, name) is getattr(spark, name)


def test_main_entry_points_are_still_importable():
    for name in ("resolve_prediction_intervals_conf", "calibrate_model", "apply_calibration",
                 "apply_joined_corrections", "series_features", "classify_series", "quantile_maps",
                 "QUANTILE_COLUMN_PREFIX", "forecast_levels"):
        assert hasattr(calibration, name), name
    assert hasattr(calibration_spark, "calibrate_model_spark") and hasattr(calibration_spark, "Materializer")
