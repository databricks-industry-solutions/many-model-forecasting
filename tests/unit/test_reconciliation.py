import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch

from .fixtures import spark_session


@pytest.fixture
def hierarchical_df():
    """Synthetic leaf-level data with hierarchy columns."""
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=24, freq="MS")
    records = []
    hierarchy = [
        ("USA", "California", "Store1"),
        ("USA", "California", "Store2"),
        ("USA", "NewYork", "Store3"),
        ("Europe", "France", "Store4"),
        ("Europe", "Germany", "Store5"),
    ]
    for country, region, store in hierarchy:
        for ds in dates:
            records.append({
                "country": country,
                "region": region,
                "store": store,
                "unique_id": store,
                "ds": ds,
                "y": np.random.uniform(100, 1000),
                "model": "StatsForecastAutoArima",
                "forecast_source": "main_pipeline",
            })
    return pd.DataFrame(records)


@pytest.fixture
def aggregated_df(hierarchical_df):
    """Data already aggregated at all hierarchy levels."""
    from hierarchicalforecast.utils import aggregate
    spec = [["country"], ["country", "region"], ["country", "region", "store"]]
    Y_hier_df, _, _ = aggregate(df=hierarchical_df, spec=spec)
    Y_hier_df["model"] = "StatsForecastAutoArima"
    Y_hier_df["forecast_source"] = "main_pipeline"
    return Y_hier_df


def test_aggregate_builds_all_levels(hierarchical_df):
    from hierarchicalforecast.utils import aggregate
    spec = [["country"], ["country", "region"], ["country", "region", "store"]]
    Y_hier_df, S_df, tags = aggregate(df=hierarchical_df, spec=spec)

    unique_ids = Y_hier_df["unique_id"].unique()
    assert "USA" in unique_ids
    assert "Europe" in unique_ids
    assert "USA/California" in unique_ids
    assert "USA/California/Store1" in unique_ids
    assert "Europe/France/Store4" in unique_ids
    assert set(tags.keys()) == {"country", "region", "store"}


def test_coherence_after_reconciliation(hierarchical_df, spark_session):
    from mmf_sa.reconciliation import run_reconciliation

    sdf = spark_session.createDataFrame(hierarchical_df)
    sdf.createOrReplaceTempView("best_models_test")

    run_reconciliation(
        spark=spark_session,
        best_models_table="best_models_test",
        reconciliation_output="reconciliation_output_test",
        hierarchy_cols=["country", "region", "store"],
        date_col="ds",
        target="y",
        method="MinTrace",
    )

    result = spark_session.table("reconciliation_output_test").toPandas()

    # Verify coherence: sum of children == parent for each date
    for ds in result["ds"].unique():
        day = result[result["ds"] == ds]
        usa_reconciled = day[day["unique_id"] == "USA"]["y_reconciled"].values[0]
        usa_children = day[day["unique_id"].str.startswith("USA/California") | day["unique_id"].str.startswith("USA/NewYork")]
        usa_children_top = day[day["unique_id"].isin(["USA/California", "USA/NewYork"])]["y_reconciled"].sum()
        assert abs(usa_reconciled - usa_children_top) < 1e-4, \
            f"Coherence failed for USA on {ds}: {usa_reconciled} != {usa_children_top}"


def test_all_methods_run(hierarchical_df, spark_session):
    from mmf_sa.reconciliation import run_reconciliation

    sdf = spark_session.createDataFrame(hierarchical_df)
    sdf.createOrReplaceTempView("best_models_methods_test")

    for method in ["BottomUp", "TopDown", "MinTrace", "ERM"]:
        run_reconciliation(
            spark=spark_session,
            best_models_table="best_models_methods_test",
            reconciliation_output=f"reconciliation_output_{method.lower()}",
            hierarchy_cols=["country", "region", "store"],
            date_col="ds",
            target="y",
            method=method,
        )
        result = spark_session.table(f"reconciliation_output_{method.lower()}").toPandas()
        assert "y_reconciled" in result.columns
        assert result["reconciliation_method"].iloc[0] == method


def test_unsupported_method_raises(hierarchical_df, spark_session):
    from mmf_sa.reconciliation import run_reconciliation

    sdf = spark_session.createDataFrame(hierarchical_df)
    sdf.createOrReplaceTempView("best_models_invalid_test")

    with pytest.raises(ValueError, match="Unsupported method"):
        run_reconciliation(
            spark=spark_session,
            best_models_table="best_models_invalid_test",
            reconciliation_output="reconciliation_output_invalid",
            hierarchy_cols=["country", "region", "store"],
            method="InvalidMethod",
        )


def test_missing_hierarchy_cols_raises(hierarchical_df, spark_session):
    from mmf_sa.reconciliation import run_reconciliation

    sdf = spark_session.createDataFrame(hierarchical_df)
    sdf.createOrReplaceTempView("best_models_missing_test")

    with pytest.raises(ValueError, match="Hierarchy columns not found"):
        run_reconciliation(
            spark=spark_session,
            best_models_table="best_models_missing_test",
            reconciliation_output="reconciliation_output_missing",
            hierarchy_cols=["nonexistent_col"],
        )


def test_output_schema(hierarchical_df, spark_session):
    from mmf_sa.reconciliation import run_reconciliation

    sdf = spark_session.createDataFrame(hierarchical_df)
    sdf.createOrReplaceTempView("best_models_schema_test")

    run_reconciliation(
        spark=spark_session,
        best_models_table="best_models_schema_test",
        reconciliation_output="reconciliation_output_schema",
        hierarchy_cols=["country", "region", "store"],
        method="BottomUp",
    )

    result = spark_session.table("reconciliation_output_schema").toPandas()
    expected_cols = {"unique_id", "ds", "y_base", "y_reconciled", "hierarchy_level", "reconciliation_method"}
    assert expected_cols.issubset(set(result.columns))
    assert result["hierarchy_level"].notna().all()


def test_import_error_without_library():
    with patch.dict("sys.modules", {"hierarchicalforecast": None, "hierarchicalforecast.core": None, "hierarchicalforecast.utils": None}):
        with pytest.raises(ImportError, match="mmf_sa\\[hierarchical\\]"):
            from mmf_sa.reconciliation import run_reconciliation
            run_reconciliation(
                spark=None,
                best_models_table="x",
                reconciliation_output="y",
                hierarchy_cols=["a"],
            )
