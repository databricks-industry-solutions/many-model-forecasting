"""Unit tests for the calibration math in mmf_sa.intervals.

Tests cover:
- Finite-sample conformal ranks and infinite-bound flags
- Group merging, shrinkage limits, horizon pooling, clipping and uncrossing
- Golden tests against the reference experiment notebook that chose the defaults,
  on a 30-series M5 sample:
  features and classes, residual conformal (marginal, group, series shrinkage),
  native bands and every CQR variant. The fixture is built by
  tests/unit/data/make_calibration_golden.py.
"""
import json
import pathlib

import numpy as np
import pandas as pd
import pytest

from mmf_sa.intervals.config import resolve_clip_lower
from mmf_sa.intervals.conformal import (
    OTHER_GROUP, conformal_rank, cqr_bounds, cqr_corrections, finalize_bounds, interval_metrics, merge_small_groups,
    residual_bounds, residual_quantiles,
)
from mmf_sa.intervals.features import classify_series, series_features

GOLDEN = pathlib.Path(__file__).parent / "data" / "calibration_golden.npz"
TOL = 1e-9


# --- Ranks and small helpers -----------------------------------------------------------


@pytest.mark.parametrize("n, q, upper, expected", [
    (9, 0.9, True, (9, False)),      # ceil(10 * 0.9) = 9
    (8, 0.9, True, (8, True)),       # ceil(9 * 0.9) = 9 > 8: infinite bound
    (19, 0.95, True, (19, False)),
    (100, 0.95, True, (96, False)),  # ceil(101 * 0.95) = 96
    (100, 0.05, False, (5, False)),  # floor(101 * 0.05) = 5
    (10, 0.05, False, (1, True)),    # floor(11 * 0.05) = 0 < 1
    (10, 0.8, True, (9, False)),     # ceil(8.8) = 9, with the float rounding guard
])
def test_conformal_rank(n, q, upper, expected):
    assert conformal_rank(n, q, upper) == expected


def test_merge_small_groups():
    labels = ["A"] * 5 + ["B"] * 2 + ["C"]
    merged = merge_small_groups(labels, n_windows=3, min_scores_per_group=6)
    assert list(merged) == ["A"] * 5 + ["B"] * 2 + [OTHER_GROUP]
    assert list(merge_small_groups(labels, 3, 1)) == labels


@pytest.fixture
def scores():
    rng = np.random.default_rng(0)
    return rng.standard_normal((40, 6, 5))


def test_series_shrinkage_limits(scores):
    labels = np.array(["g1"] * 20 + ["g2"] * 20)
    g_lo, g_hi, _ = residual_quantiles(scores, 80, "group", labels, min_scores_per_group=1)
    # kappa -> infinity: lambda -> 0, the series target falls back to its group.
    s_lo, s_hi, _ = residual_quantiles(scores, 80, "series", labels, shrinkage_kappa=1e12,
                                           min_scores_per_group=1)
    np.testing.assert_allclose(s_lo, g_lo, atol=1e-9)
    np.testing.assert_allclose(s_hi, g_hi, atol=1e-9)
    # Without horizon pooling, the per-series quantiles come from the W windows of each step.
    p_lo, p_hi, inf = residual_quantiles(scores, 80, "series", labels, shrinkage_kappa=10,
                                             min_scores_per_group=1, pool_horizon=False)
    assert p_lo.shape == p_hi.shape == (40, 5)
    assert (p_lo <= p_hi).all()
    assert inf.all()  # 6 windows are not enough for a finite 80% per-series bound


def test_marginal_equals_group_with_one_label(scores):
    m = residual_quantiles(scores, 95, "marginal")
    g = residual_quantiles(scores, 95, "group", np.array(["x"] * 40))
    for a, b in zip(m, g):
        np.testing.assert_array_equal(a, b)


def test_rejects_non_finite_scores(scores):
    scores[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        residual_quantiles(scores, 80, "marginal")


def test_cqr_rejects_unknown_options(scores):
    y = np.abs(scores)
    sigma = np.ones(40)
    with pytest.raises(ValueError, match="score"):
        cqr_corrections(y, y, y, y, sigma, 80, score="abs", pool="marginal")
    with pytest.raises(ValueError, match="pool"):
        cqr_corrections(y, y, y, y, sigma, 80, pool="store")
    with pytest.raises(ValueError, match="tail split"):
        cqr_corrections(y, y, y, y, sigma, 80, tail_split="upper", pool="marginal")


def test_resolve_clip_lower():
    assert resolve_clip_lower("auto", 0.0) == 0.0
    assert resolve_clip_lower("auto", -1.0) is None
    assert resolve_clip_lower("none", 5.0) is None
    assert resolve_clip_lower(2, -1.0) == 2.0


def test_finalize_bounds_clips_then_uncrosses():
    lo = np.array([[[-1.0, 0.5, 2.0]]])
    hi = np.array([[[-0.5, 1.0, 1.0]]])
    flo, fhi = finalize_bounds(lo, hi, 0.0)
    np.testing.assert_array_equal(flo, [[[0.0, 0.5, 2.0]]])
    np.testing.assert_array_equal(fhi, [[[0.0, 1.0, 2.0]]])
    nlo, nhi = finalize_bounds(lo, hi, None)
    np.testing.assert_array_equal(nlo, lo)
    np.testing.assert_array_equal(nhi, [[[-0.5, 1.0, 2.0]]])


def test_interval_metrics():
    y = np.array([[[0.0, 5.0, 10.0, 3.0]]])
    lo = np.full_like(y, 1.0)
    hi = np.full_like(y, 4.0)
    m = interval_metrics(y, lo, hi, np.array([2.0]), 80)
    assert m["coverage"][0] == pytest.approx(0.25)
    assert m["width"][0] == pytest.approx(1.5)
    # interval score: width 3 + penalties 2/0.2 * (1 + 1 + 6) = 83, over 4 points, / sigma 2
    assert m["msis"][0] == pytest.approx((4 * 3 + 10 * (1 + 1 + 6)) / 4 / 2)


# --- Golden tests against the experiment notebook --------------------------------------


@pytest.fixture(scope="module")
def golden():
    data = np.load(GOLDEN)
    g = {k: data[k] for k in data.files}
    rows = json.loads(str(g.pop("expected")))
    g["expected"] = {(r["method"], r["model"], r["W"], r["level"]): r for r in rows}
    return g


def _check(golden, key, y, lo, hi):
    expected = golden["expected"][key]
    m = interval_metrics(y, lo, hi, golden["sigma"], key[3])
    assert m["coverage"].mean() == pytest.approx(expected["coverage"], abs=TOL), key
    assert m["msis"].mean() == pytest.approx(expected["msis"], abs=TOL), key
    assert m["width"].mean() == pytest.approx(expected["scaled_width"], abs=TOL), key


def test_golden_features(golden):
    rows = [series_features(y, season_length=7) for y in golden["hist_y"]]
    classes = classify_series(pd.DataFrame(rows))
    np.testing.assert_allclose(classes["sigma"], golden["sigma"], rtol=0, atol=1e-12)
    np.testing.assert_array_equal(classes["sb_class"], golden["sb"])
    np.testing.assert_array_equal(classes["abc_class"], golden["abc"])
    np.testing.assert_array_equal(classes["xyz_class"], golden["xyz"])
    np.testing.assert_array_equal(classes["abc_xyz"], golden["abc_xyz"])


@pytest.mark.parametrize("model", ["CrostonSBA", "AutoETS"])
@pytest.mark.parametrize("W", [3, 12])
@pytest.mark.parametrize("level", [80, 95])
def test_golden_residual_conformal(golden, model, W, level):
    sigma = golden["sigma"]
    yhat_cal = np.nan_to_num(np.maximum(golden[f"{model}_cal"], 0.0))[:, -W:, :]
    yhat_test = np.nan_to_num(np.maximum(golden[f"{model}_test"], 0.0))
    scores = (golden["Y_cal"][:, -W:, :] - yhat_cal) / sigma[:, None, None]

    methods = {"marginal": ("marginal", None, None)}
    for col in ("sb", "abc", "abc_xyz"):
        methods[f"group:{col}"] = ("group", golden[col], None)
    for kappa in (5, 10, 20, 40):
        methods[f"series_shrink_nW:abc_xyz:k{kappa}"] = ("series", golden["abc_xyz"], kappa)
        sb_name = "series_shrink_nW" if kappa == 20 else f"series_shrink_nW:sb:k{kappa}"
        methods[sb_name] = ("series", golden["sb"], kappa)

    for name, (target, labels, kappa) in methods.items():
        q_lo, q_hi, _ = residual_quantiles(scores, level, target, labels, shrinkage_kappa=kappa or 10,
                                               min_scores_per_group=1)
        lo, hi = residual_bounds(yhat_test, sigma, q_lo, q_hi)
        # The notebook clipped lo at 0 without uncrossing residual bounds.
        _check(golden, (name, model, W, level), golden["Y_test"], np.maximum(lo, 0.0), hi)


def _native(golden, side, level, part, fallback):
    cube = golden[f"AutoETS_{side}_{level}_{part}"]
    return np.where(np.isfinite(cube), cube, fallback)


@pytest.mark.parametrize("level", [80, 95])
def test_golden_native_band(golden, level):
    yhat_test = np.maximum(golden["AutoETS_test"], 0.0)
    lo = _native(golden, "lo", level, "test", yhat_test)
    hi = _native(golden, "hi", level, "test", yhat_test)
    _check(golden, ("baseline:native", "AutoETS", 0, level), golden["Y_test"], np.maximum(lo, 0.0), hi)


CQR_VARIANTS = {
    "cqr_sym": ("symmetric", "adaptive"),
    "cqr_asym": ("asymmetric", "equal"),
    "cqr_asym_1t": ("asymmetric", "adaptive"),
    "cqr_scaled": ("scaled", "adaptive"),
}
CQR_POOLS = {"marginal": ("marginal", None), "abc_xyz": ("group", None), "series_abc_xyz_k10": ("series", 10)}


@pytest.mark.parametrize("W", [3, 12])
@pytest.mark.parametrize("level", [80, 95])
def test_golden_cqr(golden, W, level):
    sigma = golden["sigma"]
    yhat_cal = np.nan_to_num(np.maximum(golden["AutoETS_cal"], 0.0))
    yhat_test = np.maximum(golden["AutoETS_test"], 0.0)
    # Bands are clipped at 0 in calibration and test alike, as the forecast bands will be.
    lo_c = np.maximum(_native(golden, "lo", level, "cal", yhat_cal), 0.0)[:, -W:, :]
    hi_c = _native(golden, "hi", level, "cal", yhat_cal)[:, -W:, :]
    lo_t = np.maximum(_native(golden, "lo", level, "test", yhat_test), 0.0)
    hi_t = _native(golden, "hi", level, "test", yhat_test)
    y_c = golden["Y_cal"][:, -W:, :]

    for variant, (score, tail_split) in CQR_VARIANTS.items():
        for pool_name, (pool, kappa) in CQR_POOLS.items():
            q_lo, q_hi, _ = cqr_corrections(
                y_c, yhat_cal[:, -W:, :], lo_c, hi_c, sigma, level, score=score, tail_split=tail_split,
                pool=pool, labels=golden["abc_xyz"], shrinkage_kappa=kappa or 10, min_scores_per_group=1)
            lo, hi = cqr_bounds(yhat_test, lo_t, hi_t, sigma, q_lo, q_hi, score)
            lo, hi = finalize_bounds(lo, hi, 0.0)
            _check(golden, (f"{variant}:{pool_name}", "AutoETS", W, level), golden["Y_test"], lo, hi)


def test_golden_covers_all_expected_rows(golden):
    """Every notebook row of the methods MMF implements is checked by a golden test above."""
    residual = {"marginal", "group:sb", "group:abc", "group:abc_xyz", "series_shrink_nW"}
    residual |= {f"series_shrink_nW:{g}:k{k}" for g in ("sb", "abc_xyz") for k in (5, 10, 20, 40)} - {
        "series_shrink_nW:sb:k20"}
    cqr = {f"{v}:{p}" for v in CQR_VARIANTS for p in CQR_POOLS}
    methods = {k[0] for k in golden["expected"]}
    assert residual | cqr | {"baseline:native"} <= methods
    # Not implemented in MMF: business-column groupings, the W*H shrinkage variant, product baselines.
    assert methods - residual - cqr - {"baseline:native"} == {
        "group:err_tier", "group:category", "group:store", "group:sb_x_category", "series_shrink_nWH",
        "baseline:nixtla_series_step", "baseline:autogluon_series_pooled"}
