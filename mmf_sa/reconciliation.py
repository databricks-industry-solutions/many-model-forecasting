import logging
from typing import List

import pandas as pd
from pyspark.sql import SparkSession

logger = logging.getLogger(__name__)

SUPPORTED_METHODS = {"BottomUp", "TopDown", "MiddleOut", "MinTrace", "ERM"}


def _build_reconciler(method: str):
    try:
        from hierarchicalforecast.methods import BottomUp, TopDown, MiddleOut, MinTrace, ERM
    except ImportError:
        raise ImportError(
            "hierarchicalforecast is required for reconciliation. "
            "Install with: pip install mmf_sa[hierarchical]"
        )
    reconcilers = {
        "BottomUp": BottomUp(),
        "TopDown": TopDown(method="forecast_proportions"),
        "MiddleOut": MiddleOut(middle_level=None, top_down_method="forecast_proportions"),
        "MinTrace": MinTrace(method="mint_shrink"),
        "ERM": ERM(method="closed"),
    }
    if method not in reconcilers:
        raise ValueError(f"Unsupported method '{method}'. Supported: {sorted(SUPPORTED_METHODS)}")
    return reconcilers[method]


def run_reconciliation(
    spark: SparkSession,
    best_models_table: str,
    hierarchy_s_table: str,
    hierarchy_tags_table: str,
    reconciliation_output: str,
    fitted_output: str = None,
    date_col: str = "ds",
    target: str = "y",
    method: str = "MinTrace",
) -> None:
    try:
        from hierarchicalforecast.core import HierarchicalReconciliation
    except ImportError:
        raise ImportError(
            "hierarchicalforecast is required for reconciliation. "
            "Install with: pip install mmf_sa[hierarchical]"
        )

    logger.info(f"Starting hierarchical reconciliation with method={method}")

    # Read best_models output from Skill 5.
    # scoring_output stores ds and y as arrays — explode to individual rows if needed.
    best_models_sdf = spark.table(best_models_table)
    if isinstance(best_models_sdf.schema[date_col].dataType, __import__("pyspark.sql.types", fromlist=["ArrayType"]).ArrayType):
        from pyspark.sql.functions import explode, arrays_zip, col as scol
        best_models_sdf = (
            best_models_sdf
            .withColumn("_zipped", arrays_zip(scol(date_col), scol(target)))
            .withColumn("_row", explode("_zipped"))
            .withColumn(date_col, scol("_row")[date_col])
            .withColumn(target, scol("_row")[target])
            .drop("_zipped", "_row")
        )
    Y_df = best_models_sdf.toPandas()
    Y_df[date_col] = pd.to_datetime(Y_df[date_col])
    Y_df = Y_df.rename(columns={date_col: "ds", target: "y"})

    # Read tags from Delta table (written by Skill 1) and reconstruct dict
    tags_df = spark.table(hierarchy_tags_table).toPandas()
    tags = {
        level: list(group["unique_id"])
        for level, group in tags_df.groupby("level_name")
    }

    # Reconstruct S_df in memory from tags — avoids Delta roundtrip dtype issues
    # and narwhals type conversion problems in hierarchicalforecast 1.5.x.
    # tags has all the information needed: leaves = level with most series.
    # Aggregated levels first, leaves last — required for the identity matrix check.
    _bottom_series = max(tags.values(), key=len)
    _levels_ordered = sorted(tags.values(), key=len)  # top (fewest) → bottom (most)
    _all_series = [uid for level in _levels_ordered for uid in level]
    _rows = [
        [1.0 if (uid == leaf or leaf.startswith(uid + "/")) else 0.0 for leaf in _bottom_series]
        for uid in _all_series
    ]
    S_df = pd.DataFrame(_rows, columns=list(_bottom_series), dtype="float64")
    S_df.insert(0, "unique_id", _all_series)

    # Build Y_hat_df — hierarchicalforecast expects one column per model
    Y_hat_df = Y_df[["unique_id", "ds", "y"]].copy()
    Y_hat_df = Y_hat_df.rename(columns={"y": "BestModel"})

    # Load fitted values for MinTrace mint_shrink — in-sample predictions needed to estimate error covariance
    Y_fitted_df = None
    if fitted_output is not None:
        fitted_raw = spark.table(fitted_output).toPandas()
        if not fitted_raw.empty and "fitted_ds" in fitted_raw.columns:
            rows = []
            for _, row in fitted_raw.iterrows():
                if row["fitted_ds"] is not None and row["fitted_y_hat"] is not None:
                    uid = row.get("unique_id", None)
                    y_actuals = row.get("fitted_y", None)
                    for i, (ds, y_hat) in enumerate(zip(row["fitted_ds"], row["fitted_y_hat"])):
                        entry = {
                            "unique_id": uid,
                            "ds": pd.Timestamp(ds),
                            "y": float(y_actuals[i]) if y_actuals is not None else float("nan"),
                            "BestModel": float(y_hat),
                        }
                        rows.append(entry)
            if rows:
                Y_fitted_df = pd.DataFrame(rows)
                logger.info(f"Loaded {len(Y_fitted_df)} fitted value rows for MinTrace")

    # Reconcile
    reconciler = _build_reconciler(method)
    hrec = HierarchicalReconciliation(reconcilers=[reconciler])
    Y_rec_df = hrec.reconcile(
        Y_hat_df=Y_hat_df,
        Y_df=Y_fitted_df if Y_fitted_df is not None else Y_df[["unique_id", "ds", "y"]],
        S_df=S_df,
        tags=tags,
    )

    # Build output with y_base and y_reconciled for auditability.
    # hrec.reconcile() returns both the original forecast (BestModel) and reconciled (BestModel/{method}).
    # We identify the reconciled column by looking for the '/' separator in column names.
    reconciled_col = [c for c in Y_rec_df.columns if "/" in str(c)]
    if not reconciled_col:
        reconciled_col = [c for c in Y_rec_df.columns if c not in ("unique_id", "ds", "BestModel")]
    reconciled_col = reconciled_col[0]
    Y_out = Y_rec_df[["unique_id", "ds", "BestModel", reconciled_col]].copy()
    Y_out = Y_out.rename(columns={"BestModel": "y_base", reconciled_col: "y_reconciled"})

    # Add hierarchy level label
    level_map = {uid: level for level, ids in tags.items() for uid in ids}
    Y_out["hierarchy_level"] = Y_out["unique_id"].map(level_map)
    Y_out["reconciliation_method"] = method

    # Write to Delta table
    spark.createDataFrame(Y_out).write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(reconciliation_output)
    logger.info(f"Reconciliation output written to {reconciliation_output}")
