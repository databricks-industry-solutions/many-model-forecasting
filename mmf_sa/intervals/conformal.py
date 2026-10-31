"""Conformal and CQR calibration of the prediction intervals, as pure numpy/pandas functions.

Backtest values are cubes of shape (N series, W windows, H steps). Interval levels are central levels in percent
(80 means the 10%-90% interval). This is the reference implementation: calibrate_model is the numpy counterpart of the
distributed calibration in intervals.spark, which shares fit_from_statistics with it.

The math was ported from the experiment that chose the defaults.
"""
import logging
import math
from typing import Optional

import numpy as np
import pandas as pd

from mmf_sa.intervals.config import calibration_levels, model_method

_logger = logging.getLogger(__name__)


OTHER_GROUP = "__other__"

MARGINAL_POOL = "__all__"


def conformal_rank(n: int, q: float, upper: bool):
    """
    1-based finite-sample conformal rank of the q-quantile of n scores.
    Parameters:
        n (int): Number of scores.
        q (float): Quantile level in (0, 1).
        upper (bool): True for an upper quantile (ceil((n+1)q)), False for a lower one (floor((n+1)q)).
    Returns: (int, bool): The rank clipped to [1, n], and whether it had to be clipped (an infinite bound).
    """
    pos = round((n + 1) * q, 8)
    k = math.ceil(pos) if upper else math.floor(pos)
    return min(max(k, 1), n), (k < 1 or k > n)


def tail_quantiles(scores: np.ndarray, level: float):
    """
    Lower and upper conformal quantiles of signed scores for a central level, along axis 0.
    Parameters:
        scores (np.ndarray): Scores of shape (n, ...).
        level (float): Central level in percent.
    Returns: (np.ndarray, np.ndarray, bool): q_lo and q_hi of shape scores.shape[1:], and whether a rank was clipped.
    """
    a = (1 - level / 100) / 2
    n = scores.shape[0]
    s = np.sort(scores, axis=0)
    k_lo, inf_lo = conformal_rank(n, a, upper=False)
    k_hi, inf_hi = conformal_rank(n, 1 - a, upper=True)
    return s[k_lo - 1], s[k_hi - 1], (inf_lo or inf_hi)


def upper_quantile(scores: np.ndarray, q: float):
    """
    Upper conformal quantile at level q along axis 0.
    Returns: (np.ndarray, bool): The quantile of shape scores.shape[1:], and whether the rank was clipped.
    """
    k, inf = conformal_rank(scores.shape[0], q, upper=True)
    return np.sort(scores, axis=0)[k - 1], inf


def merge_small_groups(labels, n_windows: int, min_scores_per_group: int) -> np.ndarray:
    """
    Merges groups with fewer than min_scores_per_group scores per step into one OTHER_GROUP pool.
    A group of m series contributes m * n_windows scores per step.
    """
    labels = pd.Series(np.asarray(labels, dtype=object))
    sizes = labels.map(labels.value_counts())
    return labels.where(sizes * n_windows >= min_scores_per_group, OTHER_GROUP).to_numpy()


def _check_cube(name: str, cube: np.ndarray) -> None:
    if cube.ndim != 3:
        raise ValueError(f"{name} must have shape (series, windows, steps), got {cube.shape}")
    if not np.isfinite(cube).all():
        raise ValueError(f"{name} contains non-finite values")


def _labels_or_marginal(labels, n_series: int) -> np.ndarray:
    if labels is None:
        return np.zeros(n_series, dtype=int)
    labels = np.asarray(labels, dtype=object)
    if labels.shape != (n_series,):
        raise ValueError(f"labels must have one entry per series ({n_series}), got shape {labels.shape}")
    return labels


def group_tail_quantiles(scores: np.ndarray, labels, level: float, min_scores_per_group: int):
    """
    Per-group, per-step tail quantiles of signed scores (Mondrian conformal).
    Parameters:
        scores (np.ndarray): Signed scores (N, W, H).
        labels: Group label per series (N,). All series in one pool when None.
        level (float): Central level in percent.
        min_scores_per_group (int): Smaller groups are merged into one pool.
    Returns: (q_lo, q_hi, infinite): q_lo and q_hi of shape (N, H), infinite of shape (N,).
    """
    n, w, h = scores.shape
    labels = merge_small_groups(_labels_or_marginal(labels, n), w, min_scores_per_group)
    q_lo, q_hi = np.empty((n, h)), np.empty((n, h))
    inf = np.zeros(n, dtype=bool)
    for g in pd.unique(labels):
        m = labels == g
        lo, hi, f = tail_quantiles(scores[m].reshape(-1, h), level)
        q_lo[m], q_hi[m], inf[m] = lo, hi, f
    return q_lo, q_hi, inf


def residual_quantiles(
        scores: np.ndarray,
        level: float,
        target: str = "series",
        labels=None,
        shrinkage_kappa: float = 10,
        min_scores_per_group: int = 100,
        pool_horizon="auto",
):
    """
    Conformal quantiles of signed backtest residual scores, for one model and one level.
    Parameters:
        scores (np.ndarray): Signed scores (N, W, H), e.g. (y - yhat) / sigma.
        level (float): Central level in percent.
        target (str): "marginal" (one pool), "group" (one pool per label) or "series"
            (per-series quantiles shrunk toward the group with lambda = W / (W + kappa)).
        labels: Group label per series (N,), used by "group" and "series". None means one pool.
        shrinkage_kappa (float): kappa of the series target.
        min_scores_per_group (int): Smaller groups are merged into one pool.
        pool_horizon: Series target only. True pools the H steps of a series and keeps the
            group's step profile; False estimates each step from the W windows; "auto" pools
            when W < 19 (fewer windows than a finite 95% bound needs).
    Returns: (q_lo, q_hi, infinite): q_lo and q_hi of shape (N, H), in score units; infinite (N,)
        flags series whose quantile rank had to be clipped.
    """
    _check_cube("scores", scores)
    n, w, h = scores.shape
    if target == "marginal":
        return group_tail_quantiles(scores, None, level, min_scores_per_group)
    if target == "group":
        return group_tail_quantiles(scores, labels, level, min_scores_per_group)
    if target != "series":
        raise ValueError(f"Unknown coverage target {target!r}")

    labels = merge_small_groups(_labels_or_marginal(labels, n), w, min_scores_per_group)
    g_lo, g_hi, _ = group_tail_quantiles(scores, labels, level, 1)
    lam = w / (w + shrinkage_kappa)
    pool = (w < 19) if pool_horizon == "auto" else bool(pool_horizon)
    if pool:
        # Per series over its W*H scores; the group's step profile is applied as a ratio.
        s_lo, s_hi, s_inf = tail_quantiles(scores.reshape(n, -1).T, level)
        r_lo, r_hi = np.ones((n, h)), np.ones((n, h))
        for g in pd.unique(labels):
            m = labels == g
            all_lo, all_hi, _ = tail_quantiles(scores[m].reshape(-1), level)
            if abs(all_lo) > 1e-9:
                r_lo[m] = g_lo[m] / all_lo
            if abs(all_hi) > 1e-9:
                r_hi[m] = g_hi[m] / all_hi
        q_lo = lam * s_lo[:, None] * r_lo + (1 - lam) * g_lo
        q_hi = lam * s_hi[:, None] * r_hi + (1 - lam) * g_hi
    else:
        s_lo, s_hi, s_inf = tail_quantiles(scores.transpose(1, 0, 2), level)
        q_lo = lam * s_lo + (1 - lam) * g_lo
        q_hi = lam * s_hi + (1 - lam) * g_hi
    return q_lo, q_hi, np.full(n, bool(s_inf))


def pooled_upper_quantile(scores: np.ndarray, labels, q: float, shrinkage_kappa: float = None,
                          min_scores_per_group: int = 100):
    """
    Upper conformal quantile at level q of CQR scores (N, W, H), per group and step.
    With shrinkage_kappa, the per-series quantile (steps pooled) is shrunk toward the group
    with lambda = W / (W + kappa). The group's step profile is added, not multiplied, because
    CQR scores can be negative or close to zero.
    Returns: (q, infinite): q of shape (N, H) and infinite of shape (N,).
    """
    n, w, h = scores.shape
    labels = merge_small_groups(_labels_or_marginal(labels, n), w, min_scores_per_group)
    out, q_all = np.empty((n, h)), np.empty(n)
    inf = np.zeros(n, dtype=bool)
    for g in pd.unique(labels):
        m = labels == g
        out[m], inf[m] = upper_quantile(scores[m].reshape(-1, h), q)
        q_all[m], _ = upper_quantile(scores[m].reshape(-1), q)
    if shrinkage_kappa is None:
        return out, inf
    q_series, _ = upper_quantile(scores.reshape(n, -1).T, q)
    lam = w / (w + shrinkage_kappa)
    return lam * (q_series[:, None] + out - q_all[:, None]) + (1 - lam) * out, inf


def native_half_width(lo: np.ndarray, hi: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """Half-width of a native band, floored at 1e-3 * sigma so that a zero-width band cannot blow up a score."""
    return np.maximum((hi - lo) / 2, 1e-3 * sigma[:, None, None])


def cqr_corrections(
        y: np.ndarray,
        yhat: np.ndarray,
        lo: np.ndarray,
        hi: np.ndarray,
        sigma: np.ndarray,
        level: float,
        score: str = "asymmetric",
        tail_split: str = "adaptive",
        pool: str = "group",
        labels=None,
        shrinkage_kappa: float = 10,
        min_scores_per_group: int = 100,
):
    """
    CQR corrections of a native band from its misses on the backtest windows (Romano et al., 2019).
    lo and hi must already be clipped the way the forecast bands will be (clip_lower), so that the
    scores are computed against the bounds the corrections are applied to.
    Parameters:
        y, yhat, lo, hi (np.ndarray): Actuals, point forecasts and native band (N, W, H).
        sigma (np.ndarray): Per-series scale (N,).
        level (float): Central level in percent.
        score (str): "asymmetric" (one score per tail), "symmetric" (one score max(lo - y, y - hi))
            or "scaled" (|y - yhat| divided by the native half-width).
        tail_split (str): Asymmetric only. "equal" gives each tail alpha/2. "adaptive" gives the upper
            tail whatever share of alpha the lower tail does not use on the calibration windows.
        pool (str): "marginal", "group" (per label) or "series" (shrunk toward the group).
        labels: Group label per series (N,).
        shrinkage_kappa (float): kappa of the series pool.
        min_scores_per_group (int): Smaller groups are merged into one pool.
    Returns: (q_lo, q_hi, infinite): corrections of shape (N, H), in units of sigma (asymmetric,
        symmetric) or of the native half-width (scaled); infinite (N,) flags clipped ranks.
    """
    for name, cube in (("y", y), ("yhat", yhat), ("lo", lo), ("hi", hi)):
        _check_cube(name, cube)
    if pool not in ("marginal", "group", "series"):
        raise ValueError(f"Unknown CQR pool {pool!r}")
    n = y.shape[0]
    labels = None if pool == "marginal" else labels
    kappa = shrinkage_kappa if pool == "series" else None
    s3 = sigma[:, None, None]
    a = 1 - level / 100

    def pooled(scores, q):
        return pooled_upper_quantile(scores, _labels_or_marginal(labels, n), q, kappa, min_scores_per_group)

    if score == "symmetric":
        q, inf = pooled(np.maximum(lo - y, y - hi) / s3, 1 - a)
        return q, q, inf
    if score == "scaled":
        q, inf = pooled(np.abs(y - yhat) / native_half_width(lo, hi, sigma), 1 - a)
        return q, q, inf
    if score != "asymmetric":
        raise ValueError(f"Unknown CQR score {score!r}")
    s_lo = (lo - y) / s3
    s_hi = (y - hi) / s3
    q_lo, inf_lo = pooled(s_lo, 1 - a / 2)
    if tail_split == "equal":
        q_hi_level = 1 - a / 2
    elif tail_split == "adaptive":
        # Realised lower miss rate of the corrected band; it is close to 0 once lo is clipped.
        m_lo = float(np.mean(s_lo > q_lo[:, None, :]))
        q_hi_level = min(1 - a + m_lo, 1 - a / 2)
    else:
        raise ValueError(f"Unknown CQR tail split {tail_split!r}")
    q_hi, inf_hi = pooled(s_hi, q_hi_level)
    return q_lo, q_hi, inf_lo | inf_hi


def residual_bounds(yhat: np.ndarray, sigma: np.ndarray, q_lo: np.ndarray, q_hi: np.ndarray):
    """Bounds yhat + sigma * q for residual quantiles q of shape (N, H), applied to forecasts (N, K, H)."""
    s3 = sigma[:, None, None]
    return yhat + s3 * q_lo[:, None, :], yhat + s3 * q_hi[:, None, :]


def cqr_bounds(yhat: np.ndarray, lo: np.ndarray, hi: np.ndarray, sigma: np.ndarray,
               q_lo: np.ndarray, q_hi: np.ndarray, score: str = "asymmetric"):
    """Corrected bounds of native bands (N, K, H), for corrections of shape (N, H) from cqr_corrections."""
    if score == "scaled":
        half = native_half_width(lo, hi, sigma)
        return yhat - q_lo[:, None, :] * half, yhat + q_hi[:, None, :] * half
    s3 = sigma[:, None, None]
    return lo - s3 * q_lo[:, None, :], hi + s3 * q_hi[:, None, :]


def finalize_bounds(lo: np.ndarray, hi: np.ndarray, clip_lower=None):
    """Clips lo at clip_lower (if not None), then uncrosses the bounds (hi >= lo)."""
    if clip_lower is not None:
        lo = np.maximum(lo, clip_lower)
    return lo, np.maximum(hi, lo)


def interval_metrics(y: np.ndarray, lo: np.ndarray, hi: np.ndarray, sigma: np.ndarray, level: float):
    """
    Per-series interval metrics on held-out windows.
    Parameters:
        y, lo, hi (np.ndarray): Actuals and bounds (N, K, H).
        sigma (np.ndarray): Per-series scale (N,), used to make width and interval score comparable.
        level (float): Central level in percent.
    Returns: dict of (N,) arrays: coverage, msis (mean interval score / sigma) and width (mean width / sigma).
    """
    a = 1 - level / 100
    s3 = sigma[:, None, None]
    cover = (y >= lo) & (y <= hi)
    iscore = ((hi - lo) + (2 / a) * (lo - y) * (y < lo) + (2 / a) * (y - hi) * (y > hi)) / s3
    return {
        "coverage": cover.mean(axis=(1, 2)),
        "msis": iscore.mean(axis=(1, 2)),
        "width": ((hi - lo) / s3).mean(axis=(1, 2)),
    }


def _native_band(native, level, n, w, h):
    band = (native or {}).get(level)
    if band is None:
        return None
    lo, hi = band
    if lo.shape != (n, w, h) or hi.shape != (n, w, h):
        raise ValueError(f"native band at level {level} must have shape {(n, w, h)}")
    return lo, hi


def fit_plan(pi_conf, has_band: bool, model: Optional[str] = None) -> dict:
    """
    How one model is calibrated at every level: the method, score, pool and settings.
    method auto (or cqr) uses CQR when the model has native bands, otherwise conformal calibration of its
    residuals; native keeps the bands. The method is the model's (model_method).
    """
    cov = pi_conf["coverage"]
    method = model_method(pi_conf, model)
    if method in ("native", "cqr") and not has_band:
        _logger.warning(f"prediction_intervals method {method!r} for {model or 'the model'} needs a native band "
                        "it does not have; falling back to conformal calibration of its residuals.")
    kind = "conformal" if not has_band or method == "conformal" else ("native" if method == "native" else "cqr")
    if kind == "cqr":
        score, target = pi_conf["cqr_score"], pi_conf["cqr_pool"]
    elif kind == "native":
        score, target = "native", "native"
    else:
        score, target = pi_conf["score"], cov["target"]
    return {
        "kind": kind, "score": score, "target": target,
        "kappa": cov["shrinkage_kappa"] if target == "series" else None,
        "tail_split": pi_conf["cqr_tail_split"], "pool_horizon": cov["pool_horizon"],
        "min_scores_per_group": cov["min_scores_per_group"],
    }


def score_cubes(plan: dict, y, yhat, sigma, band=None) -> dict:
    """
    Calibration scores of a plan, as cubes (N, W, H). The Spark path computes the same expressions in SQL.
    band (lo, hi) must already be clipped at clip_lower.
    """
    s3 = sigma[:, None, None]
    if plan["kind"] == "conformal":
        scale = s3 if plan["score"] == "signed_normalized" else np.ones_like(s3)
        return {"e": (y - yhat) / scale}
    if plan["kind"] == "native":
        return {}
    lo, hi = band
    if plan["score"] == "symmetric":
        return {"sym": np.maximum(lo - y, y - hi) / s3}
    if plan["score"] == "scaled":
        return {"scl": np.abs(y - yhat) / native_half_width(lo, hi, sigma)}
    return {"slo": (lo - y) / s3, "shi": (y - hi) / s3}


def pools_for(plan: dict, labels, n_series: int, n_windows: int) -> np.ndarray:
    """Pool of each series for a plan: one marginal pool, or its group label merged for n_windows windows."""
    if plan["target"] in ("marginal", "native") or labels is None:
        return np.full(n_series, MARGINAL_POOL, dtype=object)
    return merge_small_groups(labels, n_windows, plan["min_scores_per_group"])


def stat_requests(plan: dict, level: float, phase: str = "main", m_lo: float = None) -> list:
    """
    Pool order statistics a plan needs at a level: (score, q, upper, steps_pooled) tuples.
    phase "lower" is the first pass of the adaptive CQR split (lower tail only); "main" is everything
    else, with the upper tail at 1 - alpha + m_lo for the adaptive split.
    """
    a = 1 - level / 100
    series = plan["target"] == "series"
    if plan["kind"] == "conformal":
        reqs = [("e", a / 2, False, False), ("e", 1 - a / 2, True, False)]
        if series:
            reqs += [("e", a / 2, False, True), ("e", 1 - a / 2, True, True)]
        return reqs
    if plan["kind"] != "cqr":
        return []

    def upper(score, q):
        return [(score, q, True, False)] + ([(score, q, True, True)] if series else [])

    if plan["score"] == "symmetric":
        return upper("sym", 1 - a)
    if plan["score"] == "scaled":
        return upper("scl", 1 - a)
    lower = upper("slo", 1 - a / 2)
    if plan["tail_split"] == "equal":
        return lower + upper("shi", 1 - a / 2)
    if phase == "lower":
        return lower
    return lower + upper("shi", adaptive_upper_level(level, m_lo))


def adaptive_upper_level(level: float, m_lo: float) -> float:
    """Upper-tail quantile level of the adaptive CQR split, from the lower tail's realised miss rate m_lo."""
    a = 1 - level / 100
    return min(1 - a + m_lo, 1 - a / 2)


def numpy_pool_statistics(cubes: dict, pools: np.ndarray, requests: list) -> dict:
    """
    Pool order statistics: {(pool, step, request): (value, infinite)}, step 0 for steps-pooled requests.
    The reference of the Spark path (intervals.spark.pool_statistics).
    """
    out = {}
    for p in pd.unique(pools):
        m = pools == p
        for req in requests:
            score, q, upper, pooled = req
            sub = cubes[score][m]
            if pooled:
                value, inf = upper_quantile(sub.reshape(-1), q) if upper else _lower_quantile(sub.reshape(-1), q)
                out[(p, 0, req)] = (float(value), inf)
            else:
                h = sub.shape[2]
                values, inf = (upper_quantile(sub.reshape(-1, h), q) if upper
                               else _lower_quantile(sub.reshape(-1, h), q))
                for step in range(h):
                    out[(p, step + 1, req)] = (float(values[step]), inf)
    return out


def _lower_quantile(scores: np.ndarray, q: float):
    k, inf = conformal_rank(scores.shape[0], q, upper=False)
    return np.sort(scores, axis=0)[k - 1], inf


def _lookup(stats: dict, pools: np.ndarray, req: tuple, h: int):
    """Per-series values (N, H) and infinite flags (N,) of a per-step request (looked up once per pool)."""
    uniq, inv = np.unique(np.asarray(pools, dtype=object).astype(str), return_inverse=True)
    by_pool = {str(p): p for p in pools}
    values = np.array([[stats[(by_pool[u], step + 1, req)][0] for step in range(h)] for u in uniq], dtype=float)
    inf = np.array([stats[(by_pool[u], 1, req)][1] for u in uniq], dtype=bool)
    return values.reshape(len(uniq), h)[inv], inf[inv]


def _lookup_pooled(stats: dict, pools: np.ndarray, req: tuple) -> np.ndarray:
    uniq, inv = np.unique(np.asarray(pools, dtype=object).astype(str), return_inverse=True)
    by_pool = {str(p): p for p in pools}
    return np.array([stats[(by_pool[u], 0, req)][0] for u in uniq], dtype=float)[inv]


def _pooled_upper(stats, pools, cubes, score, q, kappa, w, h):
    req = (score, q, True, False)
    out, inf = _lookup(stats, pools, req, h)
    if kappa is None:
        return out, inf
    n = len(pools)
    q_series, _ = upper_quantile(cubes[score].reshape(n, -1).T, q)
    q_all = _lookup_pooled(stats, pools, (score, q, True, True))
    lam = w / (w + kappa)
    return lam * (q_series[:, None] + out - q_all[:, None]) + (1 - lam) * out, inf


def cqr_lower_from_statistics(plan, level, cubes, pools, stats):
    """Lower-tail CQR correction (N, H) of the asymmetric score, from pool statistics."""
    n, w, h = cubes["slo"].shape
    a = 1 - level / 100
    return _pooled_upper(stats, pools, cubes, "slo", 1 - a / 2, plan["kappa"], w, h)


def fit_from_statistics(plan: dict, level: float, cubes: dict, pools: np.ndarray, stats: dict, m_lo: float = None,
                        horizon: int = None):
    """
    Calibration of a set of series from their own score cubes (N, W, H) and the pool statistics, which may
    have been computed over many more series (numpy_pool_statistics or the Spark path).
    horizon is only needed for the native method, which has no scores.
    Returns: (q_lo, q_hi, infinite): (N, H), (N, H), (N,), as residual_quantiles / cqr_corrections.
    """
    n = len(pools)
    if plan["kind"] == "native":
        return np.zeros((n, horizon)), np.zeros((n, horizon)), np.zeros(n, dtype=bool)
    a = 1 - level / 100
    if plan["kind"] == "conformal":
        e = cubes["e"]
        _, w, h = e.shape
        g_lo, inf_lo = _lookup(stats, pools, ("e", a / 2, False, False), h)
        g_hi, inf_hi = _lookup(stats, pools, ("e", 1 - a / 2, True, False), h)
        if plan["target"] != "series":
            return g_lo, g_hi, inf_lo | inf_hi
        lam = w / (w + plan["kappa"])
        pool = (w < 19) if plan["pool_horizon"] == "auto" else bool(plan["pool_horizon"])
        if pool:
            s_lo, s_hi, s_inf = tail_quantiles(e.reshape(n, -1).T, level)
            all_lo = _lookup_pooled(stats, pools, ("e", a / 2, False, True))
            all_hi = _lookup_pooled(stats, pools, ("e", 1 - a / 2, True, True))
            safe_lo = np.where(np.abs(all_lo) > 1e-9, all_lo, 1.0)
            safe_hi = np.where(np.abs(all_hi) > 1e-9, all_hi, 1.0)
            r_lo = np.where((np.abs(all_lo) > 1e-9)[:, None], g_lo / safe_lo[:, None], 1.0)
            r_hi = np.where((np.abs(all_hi) > 1e-9)[:, None], g_hi / safe_hi[:, None], 1.0)
            q_lo = lam * s_lo[:, None] * r_lo + (1 - lam) * g_lo
            q_hi = lam * s_hi[:, None] * r_hi + (1 - lam) * g_hi
        else:
            s_lo, s_hi, s_inf = tail_quantiles(e.transpose(1, 0, 2), level)
            q_lo = lam * s_lo + (1 - lam) * g_lo
            q_hi = lam * s_hi + (1 - lam) * g_hi
        return q_lo, q_hi, np.full(n, bool(s_inf))
    score = {"symmetric": "sym", "scaled": "scl"}.get(plan["score"])
    first = next(iter(cubes.values()))
    _, w, h = first.shape
    if score is not None:
        q, inf = _pooled_upper(stats, pools, cubes, score, 1 - a, plan["kappa"], w, h)
        return q, q, inf
    q_lo, inf_lo = _pooled_upper(stats, pools, cubes, "slo", 1 - a / 2, plan["kappa"], w, h)
    q_hi_level = 1 - a / 2 if plan["tail_split"] == "equal" else adaptive_upper_level(level, m_lo)
    q_hi, inf_hi = _pooled_upper(stats, pools, cubes, "shi", q_hi_level, plan["kappa"], w, h)
    return q_lo, q_hi, inf_lo | inf_hi


def window_bounds(plan, yhat, band, sigma, q_lo, q_hi):
    """Bounds of forecasts (N, K, H) for corrections (N, H): residual, CQR-corrected or native band."""
    if plan["kind"] == "native":
        return band
    if plan["kind"] == "cqr":
        return cqr_bounds(yhat, band[0], band[1], sigma, q_lo, q_hi, plan["score"])
    scale = sigma if plan["score"] == "signed_normalized" else np.ones_like(sigma)
    return residual_bounds(yhat, scale, q_lo, q_hi)


def calibrate_model(y, yhat, sigma, pi_conf, labels=None, native=None, clip_lower=None, model=None):
    """
    Calibrates one model's prediction intervals from its backtest, for every configured level.
    Numpy reference of the distributed calibration (intervals.spark), which shares fit_from_statistics.
    Parameters:
        y, yhat (np.ndarray): Actuals and point forecasts of the backtest windows (N, W, H), oldest window first.
        sigma (np.ndarray): Per-series scale (N,), positive.
        pi_conf: The resolved prediction_intervals block.
        labels: Group label per series (N,), for the group target, the series target's shrinkage
            and the CQR group pool. None means one pool.
        native (dict): Optional native bands {level: (lo, hi)} of shape (N, W, H).
        clip_lower (float): Lower bound of the intervals (resolve_clip_lower), or None.
        model (str): The model's name, to use its method_by_model entry; None uses the method of the block.
    Returns: dict level -> dict with
        method ("conformal", "cqr" or "native"), score, target, q_lo and q_hi (N, H), infinite (N,),
        shrinkage_weight (float, or None), metrics {name: (N, W) array} computed leave-one-window-out
        (empty when W < 2), and native_metrics {name: (N, W) array} for the uncorrected native band.
    """
    _check_cube("y", y)
    _check_cube("yhat", yhat)
    n, w, h = y.shape
    sigma = np.asarray(sigma, dtype=float)
    if sigma.shape != (n,) or not (sigma > 0).all():
        raise ValueError("sigma must be positive, with one value per series")
    has_band = native is not None
    plan = fit_plan(pi_conf, has_band, model)
    results = {}
    for level in calibration_levels(pi_conf):
        band = _native_band(native, level, n, w, h)
        if band is not None and clip_lower is not None:
            # Calibration and forecast bands are clipped alike.
            band = (np.maximum(band[0], clip_lower), band[1])
        cubes = score_cubes(plan, y, yhat, sigma, band)

        def fit(idx):
            sub = {k: v[:, idx] for k, v in cubes.items()}
            pools = pools_for(plan, labels, n, len(idx))
            m_lo = None
            if plan["kind"] == "cqr" and plan["score"] == "asymmetric" and plan["tail_split"] == "adaptive":
                stats = numpy_pool_statistics(sub, pools, stat_requests(plan, level, "lower"))
                q_lo, _ = cqr_lower_from_statistics(plan, level, sub, pools, stats)
                # Realised lower miss rate of the corrected band; it is close to 0 once lo is clipped.
                m_lo = float(np.mean(sub["slo"] > q_lo[:, None, :]))
            stats = numpy_pool_statistics(sub, pools, stat_requests(plan, level, "main", m_lo))
            return fit_from_statistics(plan, level, sub, pools, stats, m_lo, horizon=h)

        def bounds(q_lo, q_hi, j):
            b = None if band is None else (band[0][:, j:j + 1], band[1][:, j:j + 1])
            return window_bounds(plan, yhat[:, j:j + 1], b, sigma, q_lo, q_hi)

        q_lo, q_hi, infinite = fit(np.arange(w))
        metrics = {}
        if w >= 2:
            per_window = []
            for j in range(w):
                lo, hi = finalize_bounds(*bounds(*fit(np.delete(np.arange(w), j))[:2], j), clip_lower)
                per_window.append(interval_metrics(y[:, j:j + 1], lo, hi, sigma, level))
            metrics = {k: np.stack([m[k] for m in per_window], axis=1) for k in per_window[0]}
        native_metrics = {}
        if band is not None:
            per_window = [interval_metrics(y[:, j:j + 1], *finalize_bounds(band[0][:, j:j + 1], band[1][:, j:j + 1],
                                                                          clip_lower), sigma, level)
                          for j in range(w)]
            native_metrics = {k: np.stack([m[k] for m in per_window], axis=1) for k in per_window[0]}
        results[level] = {
            "method": plan["kind"], "score": plan["score"], "target": plan["target"], "q_lo": q_lo, "q_hi": q_hi,
            "infinite": np.asarray(infinite, dtype=bool),
            "shrinkage_weight": (w / (w + plan["kappa"])) if plan["kappa"] else None,
            "metrics": metrics, "native_metrics": native_metrics,
        }
    return results
