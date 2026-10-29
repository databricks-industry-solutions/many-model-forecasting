"""Prediction-interval configuration and calibration.

See docs/specs/confidence_interval.md (§7.3 for the configuration, §11 for the plan).
This module holds the `prediction_intervals` configuration (defaults and validation) and
the calibration math, as pure numpy/pandas functions that the Spark step calls per model.
"""
import logging
import math
import numbers
from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from mmf_sa.exceptions import ConfigurationError

_logger = logging.getLogger(__name__)

# Defaults of the `prediction_intervals` block. The shipped forecasting_conf_*.yaml files
# carry the same block, with intervals disabled so that existing pipelines are unchanged.
DEFAULT_PREDICTION_INTERVALS: Dict[str, Any] = {
    "enabled": False,
    "levels": [80, 95],
    "quantiles": None,
    "method": "auto",
    "cqr_score": "asymmetric",
    "cqr_tail_split": "adaptive",
    "cqr_pool": "group",
    "coverage": {
        "target": "series",
        "group_col": "auto",
        "level_group_col": None,
        "min_scores_per_group": 100,
        "levels_by_group": None,
        "shrinkage_kappa": 10,
        "pool_horizon": "auto",
    },
    "score": "signed_normalized",
    "recency_decay": None,
    "clip_lower": "auto",
    "output_format": "quantile_map",
}

_CHOICES = {
    "method": {"auto", "conformal", "native", "cqr"},
    "cqr_score": {"asymmetric", "scaled", "symmetric"},
    "cqr_tail_split": {"adaptive", "equal"},
    "cqr_pool": {"group", "marginal", "series"},
    "score": {"abs", "signed", "signed_normalized"},
    "output_format": {"quantile_map", "lo_hi", "both"},
}
_COVERAGE_TARGETS = {"marginal", "group", "series"}


def _is_number(value) -> bool:
    return isinstance(value, numbers.Real) and not isinstance(value, bool)


def _check_level(value, where: str) -> None:
    if not _is_number(value) or not 0 < value < 100:
        raise ConfigurationError(f"prediction_intervals.{where}: levels must be numbers in (0, 100), got {value!r}")


def _check_keys(conf: Dict[str, Any], defaults: Dict[str, Any], where: str) -> None:
    unknown = set(conf) - set(defaults)
    if unknown:
        raise ConfigurationError(
            f"Unknown key(s) in {where}: {', '.join(sorted(unknown))}. "
            f"Supported keys: {', '.join(sorted(defaults))}"
        )


def resolve_prediction_intervals_conf(
        conf: Optional[Union[DictConfig, Dict[str, Any]]]) -> DictConfig:
    """
    Merges the user's `prediction_intervals` block over the defaults and validates it.
    Parameters:
        conf (Optional[Union[DictConfig, Dict[str, Any]]]): The user's block, or None.
    Returns: DictConfig: The complete, validated block.
    Raises: ConfigurationError: If a key is unknown or a value is invalid.
    """
    if conf is None:
        user = {}
    elif isinstance(conf, DictConfig):
        user = OmegaConf.to_container(conf, resolve=True)
    else:
        user = dict(conf)
    _check_keys(user, DEFAULT_PREDICTION_INTERVALS, "prediction_intervals")
    if user.get("coverage") is not None:
        _check_keys(user["coverage"], DEFAULT_PREDICTION_INTERVALS["coverage"], "prediction_intervals.coverage")

    merged = OmegaConf.merge(OmegaConf.create(DEFAULT_PREDICTION_INTERVALS), OmegaConf.create(user))
    pi = OmegaConf.to_container(merged, resolve=True)
    cov = pi["coverage"]

    if not isinstance(pi["enabled"], bool):
        raise ConfigurationError(f"prediction_intervals.enabled must be a boolean, got {pi['enabled']!r}")

    # `quantiles`, when set, replaces `levels` (the shipped YAML always carries `levels`).
    if pi["quantiles"] is not None:
        pi["levels"] = None
    elif pi["levels"] is None:
        raise ConfigurationError("prediction_intervals: set `levels` or `quantiles`")
    if pi["levels"] is not None:
        if not pi["levels"]:
            raise ConfigurationError("prediction_intervals.levels must not be empty")
        for level in pi["levels"]:
            _check_level(level, "levels")
    else:
        if not pi["quantiles"]:
            raise ConfigurationError("prediction_intervals.quantiles must not be empty")
        for q in pi["quantiles"]:
            if not _is_number(q) or not 0 < q < 1:
                raise ConfigurationError(
                    f"prediction_intervals.quantiles must be numbers in (0, 1), got {q!r}")

    for key, choices in _CHOICES.items():
        if pi[key] not in choices:
            raise ConfigurationError(
                f"prediction_intervals.{key} must be one of {', '.join(sorted(choices))}, got {pi[key]!r}")
    if pi["cqr_score"] == "scaled" and pi["cqr_pool"] == "series":
        # Near-zero native half-widths make the per-series scaled score blow up (spec §7.6.4).
        raise ConfigurationError("prediction_intervals: cqr_score 'scaled' cannot be combined with cqr_pool 'series'")

    if cov["target"] not in _COVERAGE_TARGETS:
        raise ConfigurationError(
            f"prediction_intervals.coverage.target must be one of {', '.join(sorted(_COVERAGE_TARGETS))}, "
            f"got {cov['target']!r}")
    group_col = cov["group_col"]
    if not (group_col is None or isinstance(group_col, str)
            or (isinstance(group_col, list) and group_col and all(isinstance(c, str) for c in group_col))):
        raise ConfigurationError(
            "prediction_intervals.coverage.group_col must be 'auto', a column name, a list of column names, or null")
    if group_col is None and (cov["target"] == "group" or pi["cqr_pool"] == "group"):
        raise ConfigurationError(
            "prediction_intervals.coverage.group_col is required when coverage.target or cqr_pool is 'group'")
    if not (isinstance(cov["min_scores_per_group"], int) and not isinstance(cov["min_scores_per_group"], bool)
            and cov["min_scores_per_group"] >= 1):
        raise ConfigurationError("prediction_intervals.coverage.min_scores_per_group must be an integer >= 1")
    if not _is_number(cov["shrinkage_kappa"]) or cov["shrinkage_kappa"] <= 0:
        raise ConfigurationError("prediction_intervals.coverage.shrinkage_kappa must be a positive number")
    if cov["pool_horizon"] not in ("auto", True, False):
        raise ConfigurationError("prediction_intervals.coverage.pool_horizon must be 'auto', true or false")
    if cov["levels_by_group"] is not None:
        if cov["level_group_col"] is None:
            raise ConfigurationError(
                "prediction_intervals.coverage.levels_by_group requires coverage.level_group_col")
        if not isinstance(cov["levels_by_group"], dict) or not cov["levels_by_group"]:
            raise ConfigurationError(
                "prediction_intervals.coverage.levels_by_group must map group labels to lists of levels")
        for label, levels in cov["levels_by_group"].items():
            if not isinstance(levels, list) or not levels:
                raise ConfigurationError(
                    f"prediction_intervals.coverage.levels_by_group[{label!r}] must be a non-empty list of levels")
            for level in levels:
                _check_level(level, f"coverage.levels_by_group[{label!r}]")

    decay = pi["recency_decay"]
    if decay is not None and (not _is_number(decay) or not 0 < decay <= 1):
        raise ConfigurationError("prediction_intervals.recency_decay must be null or a number in (0, 1]")
    clip = pi["clip_lower"]
    if not (clip in ("auto", "none") or _is_number(clip)):
        raise ConfigurationError("prediction_intervals.clip_lower must be 'auto', 'none' or a number")

    return OmegaConf.create(pi)


# ---------------------------------------------------------------------------
# Native quantiles (spec sections 7.1 and 7.4).
# ---------------------------------------------------------------------------

# Prefix of the per-quantile columns that a model's predict() adds next to the point forecast
# when native quantiles are requested, e.g. "__q_0.1".
QUANTILE_COLUMN_PREFIX = "__q_"


def quantile_key(q: float) -> str:
    """Key of a quantile level in the quantile maps: no trailing zeros, e.g. "0.1", "0.025", "0.975"."""
    return format(round(float(q), 6), "g")


def level_quantiles(level: float):
    """Lower and upper quantile levels of a central interval level in percent (80 -> 0.1, 0.9)."""
    a = (100 - level) / 200
    return round(a, 6), round(1 - a, 6)


def requested_quantiles(pi_conf) -> list:
    """
    Quantile levels to request from models with native intervals, given the prediction_intervals block.
    Returns the sorted quantiles of every configured level (including levels_by_group) and the
    median, or the configured `quantiles` plus the median.
    """
    if pi_conf.get("quantiles"):
        qs = {round(float(q), 6) for q in pi_conf["quantiles"]}
    else:
        levels = set(pi_conf["levels"] or [])
        for group_levels in ((pi_conf.get("coverage") or {}).get("levels_by_group") or {}).values():
            levels.update(group_levels)
        qs = {q for level in levels for q in level_quantiles(level)}
    qs.add(0.5)
    return sorted(qs)


def central_levels(quantiles) -> list:
    """Central interval levels in percent whose bounds are the given quantiles (0.1 or 0.9 -> 80)."""
    levels = {round(abs(200 * q - 100), 6) for q in quantiles if q != 0.5}
    return sorted(int(lv) if float(lv).is_integer() else lv for lv in levels)


def quantile_columns(frame: pd.DataFrame) -> dict:
    """Maps quantile keys to the QUANTILE_COLUMN_PREFIX columns of a frame."""
    return {c[len(QUANTILE_COLUMN_PREFIX):]: c for c in frame.columns if str(c).startswith(QUANTILE_COLUMN_PREFIX)}


# ---------------------------------------------------------------------------
# Calibration math (docs/specs/confidence_interval.md, sections 7.2 and 7.6).
#
# Pure numpy/pandas, ported from docs/specs/experiments/pi_grouping_m5.py. Arrays of
# backtest values are cubes of shape (N series, W windows, H steps). Interval levels are
# central levels in percent (80 means the 10%-90% interval).
# ---------------------------------------------------------------------------
OTHER_GROUP = "__other__"

# Syntetos-Boylan thresholds (Syntetos, Boylan and Croston, 2005).
SB_ADI = 1.32
SB_CV2 = 0.49
# ABC cumulative volume shares.
ABC_SHARES = (0.80, 0.95)


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


def resolve_clip_lower(clip_lower, min_target: float):
    """
    Lower clipping bound of the intervals (spec section 7.3.1).
    Parameters:
        clip_lower: "auto", "none" or a number.
        min_target (float): Minimum of the training target over all series.
    Returns: float or None: The bound, or None for no clipping. "auto" clips at 0 when min_target >= 0.
    """
    if clip_lower == "none":
        return None
    if clip_lower == "auto":
        return 0.0 if min_target >= 0 else None
    return float(clip_lower)


def finalize_bounds(lo: np.ndarray, hi: np.ndarray, clip_lower=None):
    """Clips lo at clip_lower (if not None), then uncrosses the bounds (hi >= lo)."""
    if clip_lower is not None:
        lo = np.maximum(lo, clip_lower)
    return lo, np.maximum(hi, lo)


def interval_metrics(y: np.ndarray, lo: np.ndarray, hi: np.ndarray, sigma: np.ndarray, level: float):
    """
    Per-series interval metrics on held-out windows (spec section 8).
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


def series_features(y: np.ndarray, season_length: int) -> dict:
    """
    Scale and classification features of one series' recent history.
    Parameters:
        y (np.ndarray): Target values in time order (e.g. the last 365 days before the first backtest window).
        season_length (int): Season length of the seasonal-naive scale.
    Returns: dict: sigma (in-sample seasonal-naive MAE, or mean |y| when that is 0), volume, cv,
        adi (average demand interval) and cv2 (squared CV of non-zero demand).
    """
    y = np.asarray(y, dtype=float)
    snaive_mae = np.mean(np.abs(y[season_length:] - y[:-season_length])) if len(y) > season_length else 0.0
    nz = y[y > 0]
    mean = y.mean() if len(y) else 0.0
    return {
        "sigma": snaive_mae if snaive_mae > 0 else (np.mean(np.abs(y)) if len(y) else 0.0),
        "volume": y.sum(),
        "cv": y.std() / mean if mean > 0 else np.inf,
        "adi": len(y) / max(len(nz), 1),
        "cv2": (nz.std() / nz.mean()) ** 2 if len(nz) > 1 else 0.0,
    }


def classify_series(features: pd.DataFrame) -> pd.DataFrame:
    """
    Adds sb_class (Syntetos-Boylan), abc_class (80/15/5 of volume), xyz_class (terciles of CV, X =
    least volatile) and abc_xyz (e.g. "AX") to a frame of series_features, indexed by series.
    """
    out = features.copy()
    out["sb_class"] = np.select(
        [(out.adi < SB_ADI) & (out.cv2 < SB_CV2), out.adi < SB_ADI, out.cv2 < SB_CV2],
        ["smooth", "erratic", "intermittent"],
        "lumpy",
    )
    share = out["volume"].sort_values(ascending=False).cumsum() / out["volume"].sum()
    out["abc_class"] = (
        pd.cut(share, [-1, ABC_SHARES[0], ABC_SHARES[1], 2], labels=["A", "B", "C"]).astype(str).reindex(out.index)
    )
    if len(out) >= 3:
        out["xyz_class"] = pd.qcut(out["cv"].rank(method="first"), 3, labels=["X", "Y", "Z"]).astype(str)
    else:
        out["xyz_class"] = "X"
    out["abc_xyz"] = out["abc_class"] + out["xyz_class"]
    return out
