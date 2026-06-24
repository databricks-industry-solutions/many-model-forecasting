from __future__ import annotations

import logging
from datetime import datetime, timezone
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


def _build_reconciler(method: str, mintrace_method: str = "mint_shrink", middle_level: Optional[str] = None):
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
    if method == "MiddleOut" and not middle_level:
        raise ValueError("middle_level is required when method='MiddleOut'.")
    # mint_shrink and mint_cov require full dense covariance estimation — only MinTrace (dense) supports them.
    # wls_var and wls_struct are diagonal/structural and work with MinTraceSparse.
    _MINTRACE_SPARSE_METHODS = {"wls_var", "wls_struct"}
    if method == "MinTrace":
        mintrace_cls = MinTraceSparse if mintrace_method in _MINTRACE_SPARSE_METHODS else MinTrace
        return mintrace_cls(method=mintrace_method)
    if method == "BottomUp":
        return BottomUpSparse()
    if method == "TopDown":
        return TopDownSparse(method="forecast_proportions")
    if method == "MiddleOut":
        return MiddleOut(middle_level=middle_level, top_down_method="forecast_proportions")
    if method == "ERM":
        return ERM(method="closed")


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
        join_keys = ["unique_id", "model"]
        # If both tables carry run_id, also join on it so residuals are taken from the
        # exact run that produced the best-model forecasts. This pins the covariance
        # estimation to a single run and removes the cross-run ambiguity in the dedup
        # below. Falls back to (unique_id, model) for tables without run_id.
        if "run_id" in best_models_frame.columns and "run_id" in eval_frame.columns:
            join_keys.append("run_id")
        bm = best_models_frame.select(join_keys).unique()
        joined = eval_frame.join(bm, on=join_keys, how="inner")
        if joined.is_empty():
            raise ValueError(
                f"No matching rows after joining evaluation_output with best_models on {join_keys}. "
                "Verify that the model names (and run_id, if present) in both tables match."
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
    middle_level: Optional[str] = None,
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

    reconciler = _build_reconciler(method, mintrace_method, middle_level)
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


def _load_membership(spark: SparkSession, hierarchy_table: str) -> pl.DataFrame:
    df = _spark_to_polars(spark, hierarchy_table)
    required_cols = {"unique_id", "level_name", "parent_unique_id"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"Membership table '{hierarchy_table}' missing required columns: {sorted(missing)}. "
            f"Required: unique_id (string), level_name (string), parent_unique_id (string, nullable)."
        )
    df = (
        df.select(["unique_id", "level_name", "parent_unique_id"])
        .with_columns([
            pl.col("unique_id").cast(pl.Utf8).str.strip_chars().alias("unique_id"),
            pl.col("level_name").cast(pl.Utf8).str.strip_chars().alias("level_name"),
            pl.col("parent_unique_id").cast(pl.Utf8).str.strip_chars().alias("parent_unique_id"),
        ])
    )
    if df.is_empty():
        raise ValueError(f"Membership table '{hierarchy_table}' is empty.")
    logger.info(f"Loaded membership table: {len(df)} series, {df['level_name'].n_unique()} levels.")
    return df


def _validate_tree(membership_df: pl.DataFrame) -> None:
    roots = membership_df.filter(pl.col("parent_unique_id").is_null())
    if len(roots) == 0:
        raise ValueError("Tree validation failed: no root (no series with NULL parent_unique_id).")
    if len(roots) > 1:
        raise ValueError(
            f"Tree validation failed: multiple roots found: {roots['unique_id'].to_list()}. "
            "Exactly one series must have NULL parent_unique_id."
        )
    all_ids = set(membership_df["unique_id"].to_list())
    non_null_parents = membership_df.filter(
        pl.col("parent_unique_id").is_not_null()
    )["parent_unique_id"].to_list()
    dangling = [p for p in non_null_parents if p not in all_ids]
    if dangling:
        raise ValueError(
            f"Tree validation failed: parent_unique_id values not found as unique_id: {sorted(set(dangling))}."
        )
    parent_map = dict(zip(
        membership_df["unique_id"].to_list(),
        membership_df["parent_unique_id"].to_list(),
    ))
    for uid in membership_df["unique_id"].to_list():
        visited: set = set()
        current: Optional[str] = uid
        while current is not None:
            if current in visited:
                raise ValueError(
                    f"Tree validation failed: cycle detected involving series '{current}'."
                )
            visited.add(current)
            current = parent_map.get(current)
    parent_set = set(non_null_parents)
    n_leaves = len(all_ids - parent_set)
    logger.info(
        f"Tree validated: {len(all_ids)} series, {len(parent_set)} aggregates, {n_leaves} leaves."
    )


def _leaves(membership_df: pl.DataFrame) -> List[str]:
    parent_set = set(
        membership_df.filter(pl.col("parent_unique_id").is_not_null())["parent_unique_id"].to_list()
    )
    return membership_df.filter(~pl.col("unique_id").is_in(parent_set))["unique_id"].to_list()


def _build_tags(membership_df: pl.DataFrame) -> Dict[str, List[str]]:
    parent_map = dict(zip(
        membership_df["unique_id"].to_list(),
        membership_df["parent_unique_id"].to_list(),
    ))

    def _depth(uid: str) -> int:
        d = 0
        current: Optional[str] = uid
        while parent_map.get(current) is not None:
            current = parent_map[current]
            d += 1
        return d

    uid_depth = {uid: _depth(uid) for uid in membership_df["unique_id"].to_list()}
    level_ids: Dict[str, List[str]] = {}
    level_max_depth: Dict[str, int] = {}
    for row in membership_df.to_dicts():
        lname, uid = row["level_name"], row["unique_id"]
        level_ids.setdefault(lname, []).append(uid)
        level_max_depth[lname] = max(level_max_depth.get(lname, 0), uid_depth[uid])
    return {
        level: level_ids[level]
        for level in sorted(level_max_depth, key=lambda l: level_max_depth[l])
    }


def _build_S_from_adjacency(
    tags: Dict[str, List[str]],
    leaves: List[str],
    membership_df: pl.DataFrame,
) -> pd.DataFrame:
    all_series = [uid for ids in tags.values() for uid in ids]
    parent_map = dict(zip(
        membership_df["unique_id"].to_list(),
        membership_df["parent_unique_id"].to_list(),
    ))
    if _SCIPY_AVAILABLE:
        uid_index = {uid: i for i, uid in enumerate(all_series)}
        rows_idx, cols_idx = [], []
        for j, leaf in enumerate(leaves):
            current: Optional[str] = leaf
            while current is not None:
                if current in uid_index:
                    rows_idx.append(uid_index[current])
                    cols_idx.append(j)
                current = parent_map.get(current)
        S_csr = csr_matrix(
            (np.ones(len(rows_idx), dtype=np.float64), (rows_idx, cols_idx)),
            shape=(len(all_series), len(leaves)),
        )
        s_df = pd.DataFrame.sparse.from_spmatrix(S_csr, columns=list(leaves))
    else:
        def _is_ancestor_or_self(uid: str, leaf: str) -> bool:
            current: Optional[str] = leaf
            while current is not None:
                if current == uid:
                    return True
                current = parent_map.get(current)
            return False

        rows = [
            [1.0 if _is_ancestor_or_self(uid, leaf) else 0.0 for leaf in leaves]
            for uid in all_series
        ]
        s_df = pd.DataFrame(rows, columns=list(leaves))
    s_df.insert(0, "unique_id", all_series)
    return s_df


def _build_Y_hat(
    spark: SparkSession,
    levels: List[Dict],
    date_col: str,
    target: str,
) -> pl.DataFrame:
    frames = []
    for level in levels:
        df = _spark_to_polars(spark, level["best_models_table"])
        if "list" in str(df.schema.get(date_col, "")).lower():
            df = df.explode([date_col, target])
        df = (
            df.rename({date_col: "ds", target: "BestModel"})
            .select(["unique_id", "ds", "BestModel"])
        )
        frames.append(df)
    return pl.concat(frames)


def _build_residuals(
    spark: SparkSession,
    levels: List[Dict],
    freq: str,
) -> pl.DataFrame:
    frames = []
    for level in levels:
        eval_df = _spark_to_polars(spark, level["evaluation_table"])
        bm_df = _spark_to_polars(spark, level["best_models_table"])
        bm_for_residuals = bm_df if "model" in bm_df.columns else None
        frames.append(build_residual_Y_df_from_evaluation(eval_df, bm_for_residuals, freq))
    return pl.concat(frames)


def _validate_alignment(
    Y_hat: pl.DataFrame,
    Y_resid: pl.DataFrame,
    tags: Dict[str, List[str]],
    S: pd.DataFrame,
    levels: List[Dict],
) -> None:
    # 1. unique_id consistency: Y_hat ids == membership ids
    yhat_ids = set(Y_hat["unique_id"].unique().to_list())
    membership_ids = {uid for ids in tags.values() for uid in ids}
    extras = yhat_ids - membership_ids
    missing = membership_ids - yhat_ids
    if extras or missing:
        msg = "Alignment failed: unique_id mismatch between best_models tables and membership table."
        if extras:
            msg += f"\n  In best_models but not in membership: {sorted(extras)[:10]}"
        if missing:
            msg += f"\n  In membership but not in best_models: {sorted(missing)[:10]}"
        raise ValueError(msg)

    # 2. Level name consistency: levels list names must match level_name values in membership
    level_names_in_config = {lvl["name"] for lvl in levels}
    level_names_in_membership = set(tags.keys())
    config_not_in_membership = level_names_in_config - level_names_in_membership
    membership_not_in_config = level_names_in_membership - level_names_in_config
    if config_not_in_membership or membership_not_in_config:
        raise ValueError(
            f"Alignment failed: level name mismatch.\n"
            f"  In levels config but not in membership: {sorted(config_not_in_membership)}\n"
            f"  In membership but not in levels config: {sorted(membership_not_in_config)}"
        )

    # 3. All membership series must be present in Y_resid (reconciliation needs residuals for every series)
    resid_ids = set(Y_resid["unique_id"].unique().to_list())
    missing_resid = membership_ids - resid_ids
    if missing_resid:
        raise ValueError(
            f"Alignment failed: series missing from residuals (evaluation_output): {sorted(missing_resid)[:10]}."
        )

    # 4. All series share the same future ds values
    ds_sets = (
        Y_hat.group_by("unique_id")
        .agg(pl.col("ds").sort().alias("ds_list"))
    )
    # Convert to tuples for comparison — avoids ambiguous truth-value on Polars Series
    def _to_tuple(x):
        return tuple(x.to_list() if hasattr(x, "to_list") else x)

    ds_tuples = [_to_tuple(v) for v in ds_sets["ds_list"].to_list()]
    reference = ds_tuples[0]
    misaligned = [
        ds_sets["unique_id"][i]
        for i, t in enumerate(ds_tuples)
        if t != reference
    ]
    if misaligned:
        raise ValueError(
            f"Alignment failed: series do not share the same forecast horizon (ds values). "
            f"Misaligned (first 5): {misaligned[:5]}."
        )
    logger.info("Alignment validated: unique_ids, level names, leaf set, and forecast horizon consistent.")


def _coherence_precheck(
    Y_resid: pl.DataFrame,
    membership_df: pl.DataFrame,
    tolerance: float = 0.05,
) -> None:
    parent_map = dict(zip(
        membership_df["unique_id"].to_list(),
        membership_df["parent_unique_id"].to_list(),
    ))
    parents_with_children: Dict[str, List[str]] = {}
    for uid, parent in parent_map.items():
        if parent is not None:
            parents_with_children.setdefault(parent, []).append(uid)

    all_dates = sorted(Y_resid["ds"].unique().to_list())
    n_sample = max(10, int(len(all_dates) * 0.1))
    step = max(1, len(all_dates) // n_sample)
    sample_dates = all_dates[::step][:n_sample]

    actuals = Y_resid.select(["unique_id", "ds", "y"]).filter(pl.col("ds").is_in(sample_dates))
    actuals_map = {(row["unique_id"], row["ds"]): row["y"] for row in actuals.to_dicts()}

    violations = 0
    for ds in sample_dates:
        for parent, children in parents_with_children.items():
            parent_val = actuals_map.get((parent, ds))
            children_vals = [actuals_map.get((c, ds)) for c in children]
            if parent_val is None or any(v is None for v in children_vals):
                continue
            if abs(parent_val) > 1e-10:
                rel_err = abs(sum(children_vals) - parent_val) / abs(parent_val)
                if rel_err > tolerance:
                    violations += 1

    if violations > 0:
        logger.warning(
            f"Coherence pre-check: {violations} parent–children pairs exceeded "
            f"{tolerance * 100:.0f}% additive tolerance across {len(sample_dates)} sampled dates. "
            f"Reconciliation assumes additive coherence (parent = sum of children). "
            f"Verify that your target variable is additive across hierarchy levels."
        )
    else:
        logger.info(
            f"Coherence pre-check passed: actuals are additively coherent within {tolerance * 100:.0f}%."
        )


def _warn_mintrace_stability(
    Y_resid: pl.DataFrame,
    tags: Dict[str, List[str]],
    method: str,
    mintrace_method: str,
) -> None:
    if method != "MinTrace" or mintrace_method not in ("mint_shrink", "mint_cov"):
        return
    n_series = sum(len(ids) for ids in tags.values())
    n_time_points = Y_resid["ds"].n_unique()
    if n_time_points < n_series:
        logger.warning(
            f"MinTrace stability warning: {n_time_points} residual time points < {n_series} series. "
            f"Covariance estimation for '{mintrace_method}' may be unstable. "
            f"Consider using mintrace_method='wls_struct' or 'wls_var' instead."
        )


def run_reconciliation_multilevel(
    spark: SparkSession,
    levels: List[Dict],
    hierarchy_table: str,
    reconciliation_output: str,
    freq: str,
    date_col: str = "ds",
    target: str = "y",
    method: str = "MinTrace",
    mintrace_method: str = "mint_shrink",
    middle_level: Optional[str] = None,
) -> None:
    """Reconcile hierarchical forecasts from independent per-level MMF runs.

    Args:
        spark: active SparkSession
        levels: list of dicts, each with keys:
            - name: level name (must match level_name in hierarchy_table)
            - best_models_table: fully-qualified Delta table of best-model forecasts (Skill 5 output)
            - evaluation_table: fully-qualified Delta table of backtest results (Skill 4 output)
        hierarchy_table: fully-qualified Delta table with columns:
            unique_id (string), level_name (string), parent_unique_id (string, nullable)
        reconciliation_output: fully-qualified Delta table to write reconciled forecasts
        freq: MMF frequency code — H | D | W | M
        date_col: date column name in best_models tables (default: ds)
        target: target column name in best_models tables (default: y)
        method: reconciliation method — BottomUp | TopDown | MiddleOut | MinTrace | ERM
        mintrace_method: MinTrace sub-method — mint_shrink | wls_var | wls_struct | mint_cov
        middle_level: level name to anchor at when method='MiddleOut' (required for MiddleOut)
    """
    if not _POLARS_AVAILABLE:
        raise ImportError(
            "polars is required for reconciliation. "
            "Install with: pip install mmf_sa[hierarchical]"
        )
    if not levels:
        raise ValueError(
            "levels must be a non-empty list of dicts with keys: name, best_models_table, evaluation_table."
        )
    logger.info(f"Starting multilevel reconciliation: {len(levels)} levels, method={method}")

    membership_df = _load_membership(spark, hierarchy_table)
    _validate_tree(membership_df)
    leaf_list = _leaves(membership_df)
    tags = _build_tags(membership_df)
    S = _build_S_from_adjacency(tags, leaf_list, membership_df)
    Y_hat = _build_Y_hat(spark, levels, date_col, target)
    Y_resid = _build_residuals(spark, levels, freq)
    _validate_alignment(Y_hat, Y_resid, tags, S, levels)
    _coherence_precheck(Y_resid, membership_df)
    _warn_mintrace_stability(Y_resid, tags, method, mintrace_method)

    result = reconcile_core(Y_hat, Y_resid, S, tags, method, mintrace_method, middle_level)

    # Stamp every row with the time this reconciliation was generated. The table is still
    # written with mode("overwrite") so it always holds the current coherent forecast; the
    # timestamp records when that current snapshot was produced.
    result = result.with_columns(
        pl.lit(datetime.now(timezone.utc)).alias("reconciliation_timestamp")
    )

    spark.createDataFrame(result.to_pandas()).write.format("delta").mode("overwrite") \
        .option("overwriteSchema", "true").saveAsTable(reconciliation_output)
    logger.info(f"Reconciliation output written to {reconciliation_output}")


