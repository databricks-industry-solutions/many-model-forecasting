import numpy as np
import pandas as pd

from mmf_sa.models.abstract_model import collect_sorted
from .fixtures import spark_session


def test_collect_sorted_keeps_date_order(spark_session):
    """Each series comes back in date order, with its values aligned, even when the rows are shuffled."""
    n_ids, n_days = 50, 60
    dates = pd.date_range("2024-01-01", periods=n_days, freq="D")
    pdf = pd.DataFrame({
        "unique_id": np.repeat([f"id{i}" for i in range(n_ids)], n_days),
        "ds": np.tile(dates, n_ids),
        "y": np.tile(np.arange(n_days, dtype=float), n_ids),
        "x": np.tile(np.arange(n_days, dtype=float) * 10, n_ids),
    })
    # Round-robin repartition scatters every series over the partitions, so collect_list alone
    # would return the values in shuffle order
    sdf = spark_session.createDataFrame(pdf).repartition(8)

    out = collect_sorted(sdf, "unique_id", "ds", ["y", "x"]).toPandas()

    assert sorted(out.columns) == ["ds", "unique_id", "x", "y"]
    assert len(out) == n_ids
    for _, row in out.iterrows():
        assert list(pd.to_datetime(row["ds"])) == list(dates)
        assert list(row["y"]) == list(range(n_days))
        assert list(row["x"]) == [10.0 * d for d in range(n_days)]
