import os
from pathlib import Path
import shutil
import tempfile
from pyspark.sql import SparkSession

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
