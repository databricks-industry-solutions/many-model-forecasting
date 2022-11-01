import os
from pathlib import Path
import shutil
import tempfile
from pyspark.sql import SparkSession
import pathlib
import pandas as pd
from datasetsforecast.m4 import M4

import pytest


@pytest.fixture
def temp_dir(tmp_path) -> Path:
    path = tmp_path.joinpath("test_root")
    path.mkdir(parents=True)
    os.chdir(str(path))
    yield path
    shutil.rmtree(path)


@pytest.fixture(scope="module", autouse=True)
def spark_session():
    spark_warehouse_path = os.path.abspath(tempfile.mkdtemp())
    session = (
        SparkSession.builder.master("local[*]")
        .config("spark.jars.packages", "io.delta:delta-core_2.12:1.2.1")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config(
            "spark.sql.catalog.spark_catalog",
            "org.apache.spark.sql.delta.catalog.DeltaCatalog",
        )
        .config("spark.sql.warehouse.dir", str(spark_warehouse_path))
        .getOrCreate()
    )
    yield session
    session.stop()


def _transform_group(df):
    unique_id = df.unique_id.iloc[0]
    _start = pd.Timestamp("2020-01-01")
    _end = _start + pd.DateOffset(days=int(df.count()[0]) - 1)
    date_idx = pd.date_range(start=_start, end=_end, freq="D", name="ds")
    res_df = pd.DataFrame(data=[], index=date_idx).reset_index()
    res_df["unique_id"] = unique_id
    res_df["y"] = df.y.values
    return res_df


@pytest.fixture
def m4_df():
    y_df, _, _ = M4.load(directory=str(pathlib.Path.home()), group="Daily")
    _ids = [f"D{i}" for i in range(1, 10)]
    y_df = (
        y_df.groupby("unique_id")
        .filter(lambda x: x.unique_id.iloc[0] in _ids)
        .groupby("unique_id")
        .apply(_transform_group)
        .reset_index(drop=True)
    )
    return y_df
