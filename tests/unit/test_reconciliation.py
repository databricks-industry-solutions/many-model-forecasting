"""Unit tests for mmf_sa.reconciliation — new multilevel design.

Tests cover:
- Tree validation (valid tree, multiple roots, dangling parent, cycle)
- Leaf identification and tag construction
- S matrix construction from adjacency
- Residual frame construction from evaluation_output
- Alignment validation (id mismatch, horizon mismatch)
- Coherence pre-check (passing and violation)
- MinTrace stability warning
- reconcile_core output schema and coherence for all methods
- Graceful ImportError when optional dependencies are absent
"""
import logging
import pytest
import numpy as np
import pandas as pd

pl = pytest.importorskip("polars", reason="polars not installed")

from mmf_sa.reconciliation import (
    _validate_tree,
    _leaves,
    _build_tags,
    _build_S_from_adjacency,
    _validate_alignment,
    _coherence_precheck,
    _warn_mintrace_stability,
    build_residual_Y_df_from_evaluation,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

HIERARCHY = {
    # unique_id -> (level_name, parent_unique_id)
    "Total":  ("total",   None),
    "North":  ("region",  "Total"),
    "South":  ("region",  "Total"),
    "S1":     ("store",   "North"),
    "S2":     ("store",   "North"),
    "S3":     ("store",   "South"),
}

DATES_HIST   = pd.date_range("2023-01-01", periods=12, freq="MS").tolist()
DATES_FUTURE = pd.date_range("2024-01-01", periods=3,  freq="MS").tolist()


@pytest.fixture
def membership_df():
    rows = [
        {"unique_id": uid, "level_name": lvl, "parent_unique_id": parent}
        for uid, (lvl, parent) in HIERARCHY.items()
    ]
    return pl.DataFrame(rows)


@pytest.fixture
def tags_and_leaves(membership_df):
    leaf_list = _leaves(membership_df)
    tags = _build_tags(membership_df)
    return tags, leaf_list


@pytest.fixture
def S_df(tags_and_leaves, membership_df):
    tags, leaf_list = tags_and_leaves
    return _build_S_from_adjacency(tags, leaf_list, membership_df)


@pytest.fixture
def coherent_Y_resid():
    """Residuals where S1 + S2 = North, S3 = South, North + South = Total."""
    rows = []
    for ds in DATES_HIST:
        s1, s2, s3 = 10.0, 20.0, 15.0
        for uid, y, forecast in [
            ("S1", s1, s1 * 1.05),
            ("S2", s2, s2 * 1.05),
            ("S3", s3, s3 * 1.05),
            ("North", s1 + s2, (s1 + s2) * 1.05),
            ("South", s3, s3 * 1.05),
            ("Total", s1 + s2 + s3, (s1 + s2 + s3) * 1.05),
        ]:
            rows.append({"unique_id": uid, "ds": ds, "y": y, "BestModel": forecast})
    return pl.DataFrame(rows).with_columns(pl.col("ds").cast(pl.Datetime("us")))


@pytest.fixture
def Y_hat():
    rows = []
    for ds in DATES_FUTURE:
        for uid in HIERARCHY:
            rows.append({"unique_id": uid, "ds": ds, "BestModel": 50.0})
    return pl.DataFrame(rows).with_columns(pl.col("ds").cast(pl.Datetime("us")))


# ---------------------------------------------------------------------------
# Tree validation
# ---------------------------------------------------------------------------

def test_validate_tree_passes_valid(membership_df):
    _validate_tree(membership_df)  # must not raise


def test_validate_tree_multiple_roots():
    df = pl.DataFrame([
        {"unique_id": "A", "level_name": "total", "parent_unique_id": None},
        {"unique_id": "B", "level_name": "total", "parent_unique_id": None},
        {"unique_id": "C", "level_name": "leaf",  "parent_unique_id": "A"},
    ])
    with pytest.raises(ValueError, match="multiple roots"):
        _validate_tree(df)


def test_validate_tree_no_root():
    df = pl.DataFrame([
        {"unique_id": "A", "level_name": "lvl", "parent_unique_id": "B"},
        {"unique_id": "B", "level_name": "lvl", "parent_unique_id": "A"},
    ])
    with pytest.raises(ValueError, match="no root"):
        _validate_tree(df)


def test_validate_tree_dangling_parent():
    df = pl.DataFrame([
        {"unique_id": "Root", "level_name": "total", "parent_unique_id": None},
        {"unique_id": "Child", "level_name": "leaf", "parent_unique_id": "NonExistent"},
    ])
    with pytest.raises(ValueError, match="not found as unique_id"):
        _validate_tree(df)


def test_validate_tree_cycle():
    df = pl.DataFrame([
        {"unique_id": "Root",  "level_name": "total", "parent_unique_id": None},
        {"unique_id": "A",     "level_name": "mid",   "parent_unique_id": "B"},
        {"unique_id": "B",     "level_name": "mid",   "parent_unique_id": "A"},
    ])
    with pytest.raises(ValueError, match="cycle"):
        _validate_tree(df)


# ---------------------------------------------------------------------------
# Leaves and tags
# ---------------------------------------------------------------------------

def test_leaves_are_series_never_used_as_parent(membership_df):
    leaf_list = _leaves(membership_df)
    assert set(leaf_list) == {"S1", "S2", "S3"}


def test_build_tags_ordered_top_to_bottom(tags_and_leaves):
    tags, _ = tags_and_leaves
    levels = list(tags.keys())
    # "total" must come before "region", "region" before "store"
    assert levels.index("total") < levels.index("region")
    assert levels.index("region") < levels.index("store")


def test_build_tags_contains_all_series(tags_and_leaves, membership_df):
    tags, _ = tags_and_leaves
    all_tagged = {uid for ids in tags.values() for uid in ids}
    all_in_membership = set(membership_df["unique_id"].to_list())
    assert all_tagged == all_in_membership


# ---------------------------------------------------------------------------
# S matrix
# ---------------------------------------------------------------------------

def test_S_columns_are_leaves(S_df):
    s_leaves = set(c for c in S_df.columns if c != "unique_id")
    assert s_leaves == {"S1", "S2", "S3"}


def test_S_leaf_has_one_in_own_column(S_df):
    s = S_df.set_index("unique_id")
    for leaf in ["S1", "S2", "S3"]:
        assert s.loc[leaf, leaf] == 1.0


def test_S_ancestor_membership(S_df):
    s = S_df.set_index("unique_id")
    # North = S1 + S2 → North row must have 1s in S1 and S2 columns, 0 in S3
    assert s.loc["North", "S1"] == 1.0
    assert s.loc["North", "S2"] == 1.0
    assert s.loc["North", "S3"] == 0.0
    # South = S3
    assert s.loc["South", "S3"] == 1.0
    assert s.loc["South", "S1"] == 0.0
    # Total = all leaves
    for leaf in ["S1", "S2", "S3"]:
        assert s.loc["Total", leaf] == 1.0


# ---------------------------------------------------------------------------
# Alignment validation
# ---------------------------------------------------------------------------

def test_validate_alignment_passes(Y_hat, coherent_Y_resid, tags_and_leaves, S_df, membership_df):
    tags, leaf_list = tags_and_leaves
    levels = [
        {"name": "store",  "best_models_table": "t", "evaluation_table": "t"},
        {"name": "region", "best_models_table": "t", "evaluation_table": "t"},
        {"name": "total",  "best_models_table": "t", "evaluation_table": "t"},
    ]
    _validate_alignment(Y_hat, coherent_Y_resid, tags, S_df, levels)  # must not raise


def test_validate_alignment_id_mismatch(Y_hat, coherent_Y_resid, tags_and_leaves, S_df):
    tags, _ = tags_and_leaves
    # Add an extra series to Y_hat not in membership
    extra_row = pl.DataFrame([{"unique_id": "Unknown", "ds": DATES_FUTURE[0], "BestModel": 1.0}]).with_columns(
        pl.col("ds").cast(pl.Datetime("us"))
    )
    Y_hat_bad = pl.concat([Y_hat, extra_row])
    levels = [{"name": lvl, "best_models_table": "t", "evaluation_table": "t"} for lvl in tags]
    with pytest.raises(ValueError, match="unique_id mismatch"):
        _validate_alignment(Y_hat_bad, coherent_Y_resid, tags, S_df, levels)


def test_validate_alignment_horizon_mismatch(coherent_Y_resid, tags_and_leaves, S_df):
    tags, _ = tags_and_leaves
    # Give each series a different future ds set
    rows = []
    for i, uid in enumerate(HIERARCHY):
        ds = DATES_FUTURE[i % len(DATES_FUTURE)]
        rows.append({"unique_id": uid, "ds": ds, "BestModel": 1.0})
    Y_hat_bad = pl.DataFrame(rows).with_columns(pl.col("ds").cast(pl.Datetime("us")))
    levels = [{"name": lvl, "best_models_table": "t", "evaluation_table": "t"} for lvl in tags]
    with pytest.raises(ValueError, match="forecast horizon"):
        _validate_alignment(Y_hat_bad, coherent_Y_resid, tags, S_df, levels)


def test_validate_alignment_level_name_mismatch(Y_hat, coherent_Y_resid, tags_and_leaves, S_df):
    tags, _ = tags_and_leaves
    # levels config has a wrong name
    levels = [{"name": "WRONG_LEVEL", "best_models_table": "t", "evaluation_table": "t"}]
    with pytest.raises(ValueError, match="level name mismatch"):
        _validate_alignment(Y_hat, coherent_Y_resid, tags, S_df, levels)


# ---------------------------------------------------------------------------
# Coherence pre-check
# ---------------------------------------------------------------------------

def test_coherence_precheck_passes_on_coherent_data(coherent_Y_resid, membership_df, caplog):
    with caplog.at_level(logging.WARNING, logger="mmf_sa.reconciliation"):
        _coherence_precheck(coherent_Y_resid, membership_df)
    assert not any(r.levelno >= logging.WARNING for r in caplog.records)


def test_coherence_precheck_warns_on_incoherent_data(membership_df, caplog):
    # Build incoherent residuals: North != S1 + S2
    rows = []
    for ds in DATES_HIST:
        for uid, y in [("S1", 10.0), ("S2", 20.0), ("S3", 15.0),
                       ("North", 999.0),  # intentionally wrong
                       ("South", 15.0), ("Total", 45.0)]:
            rows.append({"unique_id": uid, "ds": ds, "y": y, "BestModel": y})
    bad_resid = pl.DataFrame(rows).with_columns(pl.col("ds").cast(pl.Datetime("us")))
    with caplog.at_level(logging.WARNING, logger="mmf_sa.reconciliation"):
        _coherence_precheck(bad_resid, membership_df)
    assert any("Coherence pre-check" in m for m in caplog.messages)


# ---------------------------------------------------------------------------
# MinTrace stability warning
# ---------------------------------------------------------------------------

def test_warn_mintrace_stability_triggers(tags_and_leaves, caplog):
    tags, _ = tags_and_leaves
    n_series = sum(len(ids) for ids in tags.values())
    # Build residuals with fewer time points than series
    few_rows = [{"unique_id": "S1", "ds": DATES_HIST[i], "y": 1.0, "BestModel": 1.0}
                for i in range(max(1, n_series - 2))]
    Y_resid_small = pl.DataFrame(few_rows).with_columns(pl.col("ds").cast(pl.Datetime("us")))
    with caplog.at_level(logging.WARNING, logger="mmf_sa.reconciliation"):
        _warn_mintrace_stability(Y_resid_small, tags, "MinTrace", "mint_shrink")
    assert any("MinTrace stability" in m for m in caplog.messages)


def test_warn_mintrace_stability_silent_for_bottomup(coherent_Y_resid, tags_and_leaves, caplog):
    tags, _ = tags_and_leaves
    with caplog.at_level(logging.WARNING, logger="mmf_sa.reconciliation"):
        _warn_mintrace_stability(coherent_Y_resid, tags, "BottomUp", "mint_shrink")
    assert not any("MinTrace stability" in m for m in caplog.messages)


def test_warn_mintrace_stability_silent_for_wls(coherent_Y_resid, tags_and_leaves, caplog):
    tags, _ = tags_and_leaves
    with caplog.at_level(logging.WARNING, logger="mmf_sa.reconciliation"):
        _warn_mintrace_stability(coherent_Y_resid, tags, "MinTrace", "wls_struct")
    assert not any("MinTrace stability" in m for m in caplog.messages)


# ---------------------------------------------------------------------------
# build_residual_Y_df_from_evaluation
# ---------------------------------------------------------------------------

def test_build_residual_Y_df_correct_shape():
    prediction_length = 3
    n_series = 3
    n_windows = 2
    rows = []
    for uid in ["S1", "S2", "S3"]:
        for w in range(n_windows):
            rows.append({
                "unique_id": uid,
                "backtest_window_start_date": pd.Timestamp("2023-01-01") + pd.DateOffset(months=w * 3),
                "forecast": [float(i) for i in range(prediction_length)],
                "actual":   [float(i + 1) for i in range(prediction_length)],
                "model": "ModelA",
            })
    eval_df = pl.from_pandas(pd.DataFrame(rows))
    bm_df = pl.DataFrame([{"unique_id": uid, "model": "ModelA"} for uid in ["S1", "S2", "S3"]])
    result = build_residual_Y_df_from_evaluation(eval_df, bm_df, freq="M")
    assert set(result.columns) == {"unique_id", "ds", "y", "BestModel"}
    assert result["unique_id"].n_unique() == n_series


def test_build_residual_Y_df_filters_by_best_model():
    rows = [
        {"unique_id": "S1", "backtest_window_start_date": pd.Timestamp("2023-01-01"),
         "forecast": [1.0], "actual": [2.0], "model": "ModelA"},
        {"unique_id": "S1", "backtest_window_start_date": pd.Timestamp("2023-01-01"),
         "forecast": [3.0], "actual": [2.0], "model": "ModelB"},
    ]
    eval_df = pl.from_pandas(pd.DataFrame(rows))
    bm_df = pl.DataFrame([{"unique_id": "S1", "model": "ModelA"}])
    result = build_residual_Y_df_from_evaluation(eval_df, bm_df, freq="M")
    # Only ModelA rows should survive
    assert len(result) == 1
    assert float(result["BestModel"][0]) == 1.0


# ---------------------------------------------------------------------------
# reconcile_core — output schema and coherence
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("method", ["BottomUp", "TopDown", "MinTrace", "ERM"])
def test_reconcile_core_output_schema(method, coherent_Y_resid, Y_hat, tags_and_leaves, S_df):
    hierarchicalforecast = pytest.importorskip("hierarchicalforecast")
    from mmf_sa.reconciliation import reconcile_core

    tags, _ = tags_and_leaves
    result = reconcile_core(Y_hat, coherent_Y_resid, S_df, tags, method=method)
    expected_cols = {"unique_id", "ds", "y_base", "y_reconciled", "hierarchy_level", "reconciliation_method"}
    assert expected_cols.issubset(set(result.columns))
    assert result["reconciliation_method"].unique().to_list() == [method]
    assert result["hierarchy_level"].null_count() == 0


def test_reconcile_core_coherence_bottomup(coherent_Y_resid, Y_hat, tags_and_leaves, S_df):
    """After BottomUp reconciliation, parent y_reconciled == sum of children y_reconciled."""
    hierarchicalforecast = pytest.importorskip("hierarchicalforecast")
    from mmf_sa.reconciliation import reconcile_core

    tags, _ = tags_and_leaves
    result = reconcile_core(Y_hat, coherent_Y_resid, S_df, tags, method="BottomUp").to_pandas()

    for ds in result["ds"].unique():
        day = result[result["ds"] == ds]
        north = day[day["unique_id"] == "North"]["y_reconciled"].values[0]
        s1_s2 = day[day["unique_id"].isin(["S1", "S2"])]["y_reconciled"].sum()
        assert abs(north - s1_s2) < 1e-4, f"Coherence failed North on {ds}: {north} != {s1_s2}"

        total = day[day["unique_id"] == "Total"]["y_reconciled"].values[0]
        all_regions = day[day["unique_id"].isin(["North", "South"])]["y_reconciled"].sum()
        assert abs(total - all_regions) < 1e-4, f"Coherence failed Total on {ds}: {total} != {all_regions}"


def test_unsupported_method_raises():
    from mmf_sa.reconciliation import _build_reconciler
    with pytest.raises(ValueError, match="Unsupported method"):
        _build_reconciler("InvalidMethod")


def test_build_reconciler_invalid_mintrace_method_raises():
    from mmf_sa.reconciliation import _build_reconciler
    pytest.importorskip("hierarchicalforecast")
    with pytest.raises(ValueError, match="Unsupported mintrace_method"):
        _build_reconciler("MinTrace", mintrace_method="invalid_method")


def test_build_reconciler_middleout_with_level():
    from mmf_sa.reconciliation import _build_reconciler
    pytest.importorskip("hierarchicalforecast")
    r = _build_reconciler("MiddleOut", middle_level="region")
    assert r is not None


def test_build_reconciler_middleout_no_level_raises():
    from mmf_sa.reconciliation import _build_reconciler
    pytest.importorskip("hierarchicalforecast")
    with pytest.raises(ValueError, match="middle_level is required"):
        _build_reconciler("MiddleOut")


@pytest.mark.parametrize("middle_level", ["region"])
def test_reconcile_core_middleout(middle_level, coherent_Y_resid, Y_hat, tags_and_leaves, S_df):
    pytest.importorskip("hierarchicalforecast")
    from mmf_sa.reconciliation import reconcile_core

    tags, _ = tags_and_leaves
    result = reconcile_core(Y_hat, coherent_Y_resid, S_df, tags, method="MiddleOut", middle_level=middle_level)
    expected_cols = {"unique_id", "ds", "y_base", "y_reconciled", "hierarchy_level", "reconciliation_method"}
    assert expected_cols.issubset(set(result.columns))
    assert result["reconciliation_method"].unique().to_list() == ["MiddleOut"]
    assert result["hierarchy_level"].null_count() == 0


# ---------------------------------------------------------------------------
# ImportError — graceful handling
# ---------------------------------------------------------------------------

def test_import_error_without_hierarchicalforecast(monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, "hierarchicalforecast", None)
    monkeypatch.setitem(sys.modules, "hierarchicalforecast.core", None)
    monkeypatch.setitem(sys.modules, "hierarchicalforecast.methods", None)
    from mmf_sa.reconciliation import _build_reconciler
    with pytest.raises((ImportError, TypeError)):
        _build_reconciler("MinTrace")


def test_import_error_without_polars(monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, "polars", None)
    import importlib
    import mmf_sa.reconciliation as rec
    monkeypatch.setattr(rec, "_POLARS_AVAILABLE", False)
    with pytest.raises(ImportError, match="polars"):
        rec.run_reconciliation_multilevel(
            spark=None,
            levels=[{"name": "l", "best_models_table": "t", "evaluation_table": "t"}],
            hierarchy_table="h",
            reconciliation_output="o",
            freq="M",
        )
