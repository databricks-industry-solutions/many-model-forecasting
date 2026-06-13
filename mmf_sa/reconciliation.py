from __future__ import annotations

import logging
from typing import Dict, List, Optional

import pandas as pd
from pyspark.sql import SparkSession

try:
    import polars as pl
    _POLARS_AVAILABLE = True
except ImportError:
    _POLARS_AVAILABLE = False

try:
    import numpy as np
    from scipy.sparse import csr_matrix
    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False

logger = logging.getLogger(__name__)

SUPPORTED_METHODS = {"BottomUp", "TopDown", "MiddleOut", "MinTrace", "ERM"}
SUPPORTED_MINTRACE_METHODS = {"mint_shrink", "mint_cov", "wls_var", "wls_struct"}

# Maps MMF freq codes to pandas offset aliases (used in monthly ds reconstruction)
_MMF_FREQ_TO_PANDAS = {"H": "h", "D": "D", "W": "W", "M": "MS"}


def _spark_to_polars(spark: SparkSession, table_name: str) -> pl.DataFrame:
    return pl.from_arrow(spark.table(table_name).toArrow())


def _build_reconciler(method: str, mintrace_method: str = "mint_shrink"):
    try:
        from hierarchicalforecast.methods import (
            BottomUp, BottomUpSparse, ERM, MiddleOut,
            MinTrace, MinTraceSparse, TopDown, TopDownSparse,
        )
    except ImportError:
        raise ImportError(
            "hierarchicalforecast is required for reconciliation. "
            "Install with: pip install mmf_sa[hierarchical]"
        )
    if method not in SUPPORTED_METHODS:
        raise ValueError(f"Unsupported method '{method}'. Supported: {sorted(SUPPORTED_METHODS)}")
    if method == "MinTrace" and mintrace_method not in SUPPORTED_MINTRACE_METHODS:
        raise ValueError(
            f"Unsupported mintrace_method '{mintrace_method}'. "
            f"Supported: {sorted(SUPPORTED_MINTRACE_METHODS)}"
        )
    # mint_shrink and mint_cov require full dense covariance estimation — only MinTrace (dense) supports them.
    # wls_var and wls_struct are diagonal/structural and work with MinTraceSparse.
    _MINTRACE_SPARSE_METHODS = {"wls_var", "wls_struct", "ols"}
    if method == "MinTrace":
        mintrace_cls = MinTraceSparse if mintrace_method in _MINTRACE_SPARSE_METHODS else MinTrace
    reconcilers = {
        "BottomUp": BottomUpSparse(),
        "TopDown": TopDownSparse(method="forecast_proportions"),
        "MiddleOut": MiddleOut(middle_level=None, top_down_method="forecast_proportions"),
        "MinTrace": mintrace_cls(method=mintrace_method),
        "ERM": ERM(method="closed"),
    }
    return reconcilers[method]


def _add_ds_from_window_start(df: pl.DataFrame, freq: str) -> pl.DataFrame:
    """Add 'ds' column by offsetting backtest_window_start_date by _step periods at freq."""
    # Ensure backtest_window_start_date is Datetime for duration arithmetic
    df = df.with_columns(
        pl.col("backtest_window_start_date").cast(pl.Datetime("us")).alias("backtest_window_start_date")
    )
    if freq == "H":
        return df.with_columns(
            (pl.col("backtest_window_start_date") + pl.duration(hours=pl.col("_step"))).alias("ds")
        )
    elif freq == "D":
        return df.with_columns(
            (pl.col("backtest_window_start_date") + pl.duration(days=pl.col("_step"))).alias("ds")
        )
    elif freq == "W":
        return df.with_columns(
            (pl.col("backtest_window_start_date") + pl.duration(weeks=pl.col("_step"))).alias("ds")
        )
    elif freq == "M":
        return df.with_columns(
            pl.col("backtest_window_start_date").dt.offset_by(
                pl.concat_str(pl.col("_step").cast(pl.String), pl.lit("mo"))
            ).alias("ds")
        )
    else:
        raise ValueError(f"Unsupported freq '{freq}'. Supported: {sorted(_MMF_FREQ_TO_PANDAS.keys())}")


def build_residual_Y_df_from_evaluation(
    eval_frame: pl.DataFrame,
    best_models_frame: Optional[pl.DataFrame],
    freq: str,
    model_col: str = "BestModel",
) -> pl.DataFrame:
    """Build a residual frame from evaluation_output, optionally filtered by best model per series.

    Args:
        eval_frame: Polars frame of evaluation_output — must have columns:
            unique_id, backtest_window_start_date, forecast (List[Float64]),
            actual (List[Float64]), model
        best_models_frame: Polars frame with at least (unique_id, model) columns,
            or None to use all residuals without filtering by model.
        freq: MMF frequency code — H | D | W | M
        model_col: column name to assign to the forecast values in the output

    Returns:
        Polars DataFrame with columns: unique_id, ds, y, {model_col}
        where y = backtest actual and {model_col} = backtest forecast.
        Residual = y − {model_col}.
    """
    if freq not in _MMF_FREQ_TO_PANDAS:
        raise ValueError(f"Unsupported freq '{freq}'. Supported: {sorted(_MMF_FREQ_TO_PANDAS.keys())}")

    # Filter by best model only when the forecast table carries a model column
    has_model = (
        best_models_frame is not None
        and "model" in best_models_frame.columns
        and "model" in eval_frame.columns
    )
    if has_model:
        bm = best_models_frame.select(["unique_id", "model"]).unique()
        joined = eval_frame.join(bm, on=["unique_id", "model"], how="inner")
        if joined.is_empty():
            raise ValueError(
                "No matching rows after joining evaluation_output with best_models on (unique_id, model). "
                "Verify that the model names in both tables match."
            )
    else:
        joined = eval_frame

    # Add step index [0, 1, ..., prediction_length-1] per row, then explode arrays
    exploded = (
        joined
        .with_columns(
            pl.int_ranges(pl.col("forecast").list.len()).alias("_step")
        )
        .explode(["forecast", "actual", "_step"])
    )

    # Reconstruct ds from backtest_window_start_date + _step * freq
    exploded = _add_ds_from_window_start(exploded, freq)

    # Deduplicate overlapping backtest windows — for the same (unique_id, ds) keep the
    # residual from the most recent window (largest backtest_window_start_date)
    result = (
        exploded
        .sort(["unique_id", "ds", "backtest_window_start_date"])
        .unique(subset=["unique_id", "ds"], keep="last", maintain_order=False)
        .rename({"actual": "y", "forecast": model_col})
        .select(["unique_id", "ds", "y", model_col])
    )

    return result


def reconcile_core(
    Y_hat_df: pl.DataFrame,
    Y_df: pl.DataFrame,
    S_df: pl.DataFrame,
    tags: Dict[str, List[str]],
    method: str = "MinTrace",
    mintrace_method: str = "mint_shrink",
) -> pl.DataFrame:
    """Apply hierarchical reconciliation — pure function, no Spark, no Delta.

    Args:
        Y_hat_df: Polars DataFrame (unique_id, ds, BestModel) — future forecasts to reconcile
        Y_df: Polars DataFrame (unique_id, ds, y, BestModel) — backtest residuals for W estimation.
              y = actual, BestModel = backtest forecast; residual = y − BestModel
        S_df: Polars DataFrame (unique_id, leaf1, leaf2, ...) — summation matrix
        tags: dict mapping level_name → list of unique_ids at that level
        method: reconciliation method — BottomUp | TopDown | MiddleOut | MinTrace | ERM
        mintrace_method: MinTrace sub-method — mint_shrink (default) | wls_var | wls_struct | mint_cov

    Returns:
        Polars DataFrame (unique_id, ds, y_base, y_reconciled, hierarchy_level, reconciliation_method)
    """
    try:
        from hierarchicalforecast.core import HierarchicalReconciliation
    except ImportError:
        raise ImportError(
            "hierarchicalforecast is required. Install with: pip install mmf_sa[hierarchical]"
        )

    reconciler = _build_reconciler(method, mintrace_method)
    hrec = HierarchicalReconciliation(reconcilers=[reconciler])

    # hierarchicalforecast works on pandas; S_df may already be pandas (sparse-backed)
    Y_hat_pd = Y_hat_df.to_pandas()
    Y_pd = Y_df.to_pandas()
    S_pd = S_df if isinstance(S_df, pd.DataFrame) else S_df.to_pandas()

    Y_rec_pd = hrec.reconcile(
        Y_hat_df=Y_hat_pd,
        Y_df=Y_pd,
        S_df=S_pd,
        tags=tags,
    )

    # Identify the reconciled column — named 'BestModel/{method}' by hrec.reconcile()
    reconciled_col = next(
        (c for c in Y_rec_pd.columns if "/" in str(c)),
        next((c for c in Y_rec_pd.columns if c not in ("unique_id", "ds", "BestModel")), None),
    )
    if reconciled_col is None:
        raise RuntimeError(
            f"Could not identify reconciled column in output. Columns: {list(Y_rec_pd.columns)}"
        )

    Y_out = Y_rec_pd[["unique_id", "ds", "BestModel", reconciled_col]].copy()
    Y_out = Y_out.rename(columns={"BestModel": "y_base", reconciled_col: "y_reconciled"})

    level_map = {uid: level for level, ids in tags.items() for uid in ids}
    Y_out["hierarchy_level"] = Y_out["unique_id"].map(level_map)
    Y_out["reconciliation_method"] = method

    return pl.from_pandas(Y_out)


def run_aggregation(
    spark: SparkSession,
    train_table: str,
    hierarchy_s_table: str,
    hierarchy_tags_table: str,
    hierarchy_cols: List[str],
    source_table: Optional[str] = None,
) -> None:
    # Reads hierarchy columns from source_table if provided (original raw table),
    # otherwise reads from train_table. Aggregated result is always written to train_table.
    try:
        from hierarchicalforecast.utils import aggregate
    except ImportError:
        raise ImportError(
            "hierarchicalforecast is required for aggregation. "
            "Install with: pip install mmf_sa[hierarchical]"
        )

    read_table = source_table if source_table else train_table
    # Load via Arrow → Polars so aggregate() uses Polars backend via Narwhals (no pandas materialization)
    train_df = _spark_to_polars(spark, read_table)
    train_df = train_df.with_columns(pl.col("ds").cast(pl.Datetime))

    spec = [[hierarchy_cols[0]]]
    for i in range(2, len(hierarchy_cols) + 1):
        spec.append(hierarchy_cols[:i])

    already_aggregated = train_df["unique_id"].str.contains("/").any()
    if already_aggregated:
        raise ValueError(
            "run_aggregation() requires leaf-level data without '/' in unique_id. "
            "The train table already contains aggregated series. "
            "For data that already has all hierarchy levels, use derive_hierarchy_from_unique_ids() instead."
        )

    Y_hier_df, S_df, tags = aggregate(
        df=train_df.drop("unique_id"), spec=spec
    )
    # Y_hier_df and S_df are Polars DataFrames (Narwhals preserves backend)
    spark.createDataFrame(Y_hier_df.to_pandas()).write.format("delta").mode("overwrite") \
        .option("overwriteSchema", "true").saveAsTable(train_table)

    if "unique_id" not in S_df.columns:
        S_df = S_df.with_row_index("unique_id")
    spark.createDataFrame(S_df.to_pandas()).write.format("delta").mode("overwrite") \
        .option("overwriteSchema", "true").saveAsTable(hierarchy_s_table)

    tags_rows = [{"level_name": level, "unique_id": uid} for level, ids in tags.items() for uid in ids]
    spark.createDataFrame(pd.DataFrame(tags_rows)).write.format("delta").mode("overwrite") \
        .option("overwriteSchema", "true").saveAsTable(hierarchy_tags_table)

    logger.info(f"Aggregation complete: {hierarchy_s_table}, {hierarchy_tags_table}")


def derive_hierarchy_from_unique_ids(
    spark: SparkSession,
    train_table: str,
    hierarchy_s_table: str,
    hierarchy_tags_table: str,
    hierarchy_cols: Optional[List[str]] = None,
) -> None:
    """Build S_df and tags from unique_ids that already use slash notation.

    Reads only distinct unique_ids — does not load time series values.
    Use this for data that already has all hierarchy levels aggregated
    (unique_ids like USA/California/Store1).

    Args:
        spark: active SparkSession
        train_table: Delta table with unique_id column (slash-separated)
        hierarchy_s_table: Delta table to write summation matrix
        hierarchy_tags_table: Delta table to write level→unique_id mapping
        hierarchy_cols: optional level names top→bottom (e.g. ["country","region","store"]).
            If None, uses generic names ("level_1", "level_2", ...).
    """
    uid_list = sorted([
        row["unique_id"]
        for row in spark.sql(f"SELECT DISTINCT unique_id FROM {train_table}").collect()
    ])

    tags: Dict[str, List[str]] = {}
    for uid in uid_list:
        depth = len(uid.split("/"))
        if hierarchy_cols and depth <= len(hierarchy_cols):
            level_name = "/".join(hierarchy_cols[:depth])
        else:
            level_name = f"level_{depth}"
        tags.setdefault(level_name, []).append(uid)

    bottom_series = max(tags.values(), key=len)
    levels_ordered = sorted(tags.values(), key=len)
    all_series = [uid for level in levels_ordered for uid in level]

    if _SCIPY_AVAILABLE:
        uid_index = {uid: i for i, uid in enumerate(all_series)}
        rows_idx, cols_idx = [], []
        for j, leaf in enumerate(bottom_series):
            if leaf in uid_index:
                rows_idx.append(uid_index[leaf])
                cols_idx.append(j)
            parts = leaf.split("/")
            for k in range(1, len(parts)):
                ancestor = "/".join(parts[:k])
                if ancestor in uid_index:
                    rows_idx.append(uid_index[ancestor])
                    cols_idx.append(j)
        S_csr = csr_matrix(
            (np.ones(len(rows_idx), dtype=np.float64), (rows_idx, cols_idx)),
            shape=(len(all_series), len(bottom_series)),
        )
        s_df = pd.DataFrame.sparse.from_spmatrix(S_csr, columns=list(bottom_series))
        s_df.insert(0, "unique_id", all_series)
    else:
        rows = [
            [1.0 if (uid == leaf or leaf.startswith(uid + "/")) else 0.0 for leaf in bottom_series]
            for uid in all_series
        ]
        s_df = pd.DataFrame(
            {leaf: [row[i] for row in rows] for i, leaf in enumerate(bottom_series)},
        )
        s_df.insert(0, "unique_id", all_series)

    spark.createDataFrame(s_df).write.format("delta").mode("overwrite") \
        .option("overwriteSchema", "true").saveAsTable(hierarchy_s_table)

    tags_rows = [{"level_name": level, "unique_id": uid} for level, ids in tags.items() for uid in ids]
    spark.createDataFrame(pd.DataFrame(tags_rows)).write.format("delta").mode("overwrite") \
        .option("overwriteSchema", "true").saveAsTable(hierarchy_tags_table)

    logger.info(f"Hierarchy derived from unique_ids: {hierarchy_s_table}, {hierarchy_tags_table}")


def run_reconciliation(
    spark: SparkSession,
    best_models_table: str,
    evaluation_output_table: str,
    hierarchy_s_table: str,
    hierarchy_tags_table: str,
    reconciliation_output: str,
    freq: str,
    date_col: str = "ds",
    target: str = "y",
    method: str = "MinTrace",
    mintrace_method: str = "mint_shrink",
) -> None:
    """Reconcile hierarchical forecasts using out-of-sample backtest residuals.

    Args:
        spark: active SparkSession
        best_models_table: Delta table with best-model forecasts per series.
            Expected columns: unique_id, {date_col} (Array[Timestamp] or Timestamp),
            {target} (Array[Double] or Double), model
        evaluation_output_table: Delta table with backtest results.
            Expected columns: unique_id, backtest_window_start_date, forecast (Array[Double]),
            actual (Array[Double]), model
        hierarchy_s_table: Delta table with the summation matrix (from run_aggregation)
        hierarchy_tags_table: Delta table with level→unique_id mapping (from run_aggregation)
        reconciliation_output: Delta table to write reconciled forecasts
        freq: MMF frequency code — H | D | W | M
        date_col: date column name in best_models_table (default: ds)
        target: target column name in best_models_table (default: y)
        method: reconciliation method — BottomUp | TopDown | MiddleOut | MinTrace | ERM
        mintrace_method: MinTrace sub-method — mint_shrink (default) | wls_var | wls_struct | mint_cov
    """
    if not _POLARS_AVAILABLE:
        raise ImportError(
            "polars is required for reconciliation. "
            "Install with: pip install mmf_sa[hierarchical]"
        )
    logger.info(f"Starting hierarchical reconciliation with method={method}")

    # Load Delta tables via Spark → Arrow → Polars
    best_models_pl = _spark_to_polars(spark, best_models_table)
    eval_pl = _spark_to_polars(spark, evaluation_output_table)
    tags_pl = _spark_to_polars(spark, hierarchy_tags_table)

    # Build Y_hat_df (forecasts to reconcile)
    # best_models_table may have array-typed date/target columns (from scoring_output)
    is_list_col = "list" in str(best_models_pl.schema.get(date_col, "")).lower()
    if is_list_col:
        y_hat_df = (
            best_models_pl
            .explode([date_col, target])
            .rename({date_col: "ds", target: "BestModel"})
            .select(["unique_id", "ds", "BestModel"])
        )
    else:
        y_hat_df = (
            best_models_pl
            .rename({date_col: "ds", target: "BestModel"})
            .select(["unique_id", "ds", "BestModel"])
        )

    # Build Y_df — filter by model only when the forecast table is from the MMF pipeline
    bm_for_residuals = best_models_pl if "model" in best_models_pl.columns else None
    y_df = build_residual_Y_df_from_evaluation(eval_pl, bm_for_residuals, freq)

    # Reconstruct tags dict
    tags: Dict[str, List[str]] = {}
    for row in tags_pl.to_dicts():
        tags.setdefault(row["level_name"], []).append(row["unique_id"])

    # Build S_df as sparse matrix — O(L × depth) instead of O(N × L)
    bottom_series = max(tags.values(), key=len)
    levels_ordered = sorted(tags.values(), key=len)
    all_series = [uid for level in levels_ordered for uid in level]

    if _SCIPY_AVAILABLE:
        uid_index = {uid: i for i, uid in enumerate(all_series)}
        rows_idx, cols_idx = [], []
        for j, leaf in enumerate(bottom_series):
            if leaf in uid_index:
                rows_idx.append(uid_index[leaf])
                cols_idx.append(j)
            parts = leaf.split("/")
            for k in range(1, len(parts)):
                ancestor = "/".join(parts[:k])
                if ancestor in uid_index:
                    rows_idx.append(uid_index[ancestor])
                    cols_idx.append(j)
        S_csr = csr_matrix(
            (np.ones(len(rows_idx), dtype=np.float64), (rows_idx, cols_idx)),
            shape=(len(all_series), len(bottom_series)),
        )
        s_df = pd.DataFrame.sparse.from_spmatrix(S_csr, columns=list(bottom_series))
        s_df.insert(0, "unique_id", all_series)
    else:
        # Fallback to dense if scipy not available
        dense_rows = [
            [1.0 if (uid == leaf or leaf.startswith(uid + "/")) else 0.0 for leaf in bottom_series]
            for uid in all_series
        ]
        s_df = pd.DataFrame(dense_rows, columns=list(bottom_series))
        s_df.insert(0, "unique_id", all_series)

    result = reconcile_core(y_hat_df, y_df, s_df, tags, method, mintrace_method)

    spark.createDataFrame(result.to_pandas()).write.format("delta").mode("overwrite") \
        .option("overwriteSchema", "true").saveAsTable(reconciliation_output)
    logger.info(f"Reconciliation output written to {reconciliation_output}")
