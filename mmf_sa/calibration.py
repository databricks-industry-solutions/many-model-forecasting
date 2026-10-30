"""Prediction-interval configuration and calibration.

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
        # Intervals are central: every quantile other than the median needs its mirror 1 - q.
        qs = {round(float(q), 6) for q in pi["quantiles"]}
        unpaired = sorted(q for q in qs if q != 0.5 and round(1 - q, 6) not in qs)
        if unpaired:
            raise ConfigurationError(
                f"prediction_intervals.quantiles must come in central pairs (q and 1 - q); missing the mirror of "
                f"{unpaired}")

    for key, choices in _CHOICES.items():
        if pi[key] not in choices:
            raise ConfigurationError(
                f"prediction_intervals.{key} must be one of {', '.join(sorted(choices))}, got {pi[key]!r}")
    if pi["cqr_score"] == "scaled" and pi["cqr_pool"] == "series":
        # Near-zero native half-widths make the per-series scaled score blow up.
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
# Native quantiles.
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


def quantile_maps(pred_df: pd.DataFrame, group_col: str) -> dict:
    """Maps each series of a multi-series forecast frame to its {quantile key: array} dict (empty without quantiles)."""
    cols = quantile_columns(pred_df)
    if not cols:
        return {}
    keys = pred_df[group_col].to_numpy()
    values = {key: pred_df[c].to_numpy() for key, c in cols.items()}
    return {
        k: {key: [float(x) for x in values[key][i]] for key in values}
        for i, k in enumerate(keys)
    }


def forecast_levels(quantiles) -> list:
    """Quantile levels a foundation-model predict returns for native_quantiles: sorted, rounded, with the median.
    The single source of the level order used to pack and unpack the distributed UDF output."""
    return sorted({round(float(q), 6) for q in quantiles} | {0.5})


def encode_quantile_forecasts(medians: np.ndarray, quantiles: np.ndarray) -> list:
    """Packs each series' median and quantiles into one array, so a UDF can keep returning array<double>."""
    return [np.concatenate([m, q.ravel()]) for m, q in zip(medians, quantiles)]


def decode_quantile_forecasts(forecast_df: pd.DataFrame, target: str, levels, prediction_length: int):
    """Unpacks encode_quantile_forecasts arrays into the target column and one QUANTILE_COLUMN_PREFIX column per level."""
    packed = [np.asarray(v, dtype=np.float64) for v in forecast_df[target]]
    forecast_df[target] = [v[:prediction_length] for v in packed]
    for i, level in enumerate(levels):
        start = prediction_length * (i + 1)
        forecast_df[QUANTILE_COLUMN_PREFIX + quantile_key(level)] = [v[start:start + prediction_length] for v in packed]
    return forecast_df


# ---------------------------------------------------------------------------
# Calibration math.
#
# Pure numpy/pandas, ported from the experiment that chose the defaults. Arrays of
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
    Lower clipping bound of the intervals.
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
    # Ties are broken by series id, so that the Spark path (calibration_spark.classify) gives the same classes.
    ids = np.asarray(out.index.astype(str))
    by_volume = np.lexsort((ids, -out["volume"].to_numpy(dtype=float)))
    share = pd.Series(np.cumsum(out["volume"].to_numpy(dtype=float)[by_volume]) / out["volume"].sum(),
                      index=out.index[by_volume])
    out["abc_class"] = (
        pd.cut(share, [-1, ABC_SHARES[0], ABC_SHARES[1], 2], labels=["A", "B", "C"]).astype(str).reindex(out.index)
    )
    if len(out) >= 3:
        rank = np.empty(len(out))
        rank[np.lexsort((ids, out["cv"].to_numpy(dtype=float)))] = np.arange(1, len(out) + 1)
        out["xyz_class"] = pd.qcut(rank, 3, labels=["X", "Y", "Z"]).astype(str)
    else:
        out["xyz_class"] = "X"
    out["abc_xyz"] = out["abc_class"] + out["xyz_class"]
    return out


# ---------------------------------------------------------------------------
# One model's calibration, as called by Forecaster.run_calibration.
# ---------------------------------------------------------------------------

# Season length of the seasonal-naive scale, and length of the history used for the scale
# and the classes (the periods just before the first backtest window), per frequency.
SEASON_LENGTH = {"H": 24, "D": 7, "W": 52, "M": 12}
FEATURE_PERIODS = {"H": 24 * 28, "D": 365, "W": 104, "M": 24}


def calibration_levels(pi_conf) -> list:
    """Central levels to calibrate: `levels`, or the central levels of `quantiles`."""
    if pi_conf.get("quantiles"):
        return central_levels(pi_conf["quantiles"])
    return sorted(pi_conf["levels"])


def check_supported(pi_conf) -> None:
    """Raises ConfigurationError for settings that are validated but not implemented yet."""
    if pi_conf.get("recency_decay") is not None:
        raise ConfigurationError("prediction_intervals.recency_decay is not supported yet")
    if (pi_conf.get("coverage") or {}).get("levels_by_group") is not None:
        raise ConfigurationError("prediction_intervals.coverage.levels_by_group is not supported yet")
    if pi_conf.get("score") == "abs":
        raise ConfigurationError("prediction_intervals.score 'abs' is not supported yet; use signed_normalized")


def _native_band(native, level, n, w, h):
    band = (native or {}).get(level)
    if band is None:
        return None
    lo, hi = band
    if lo.shape != (n, w, h) or hi.shape != (n, w, h):
        raise ValueError(f"native band at level {level} must have shape {(n, w, h)}")
    return lo, hi


def fit_plan(pi_conf, has_band: bool) -> dict:
    """
    How one model is calibrated at every level: the method, score, pool and settings.
    method auto (or cqr) uses CQR when the model has native bands, otherwise conformal calibration of its
    residuals; native keeps the bands.
    """
    cov = pi_conf["coverage"]
    method = pi_conf["method"]
    if method in ("native", "cqr") and not has_band:
        _logger.warning(f"prediction_intervals.method is {method!r} but the model has no native band; "
                        "falling back to conformal calibration of its residuals.")
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
    The reference of the Spark path (calibration_spark.pool_statistics).
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


def calibrate_model(y, yhat, sigma, pi_conf, labels=None, native=None, clip_lower=None):
    """
    Calibrates one model's prediction intervals from its backtest, for every configured level.
    Numpy reference of the distributed calibration (calibration_spark), which shares fit_from_statistics.
    Parameters:
        y, yhat (np.ndarray): Actuals and point forecasts of the backtest windows (N, W, H), oldest window first.
        sigma (np.ndarray): Per-series scale (N,), positive.
        pi_conf: The resolved prediction_intervals block.
        labels: Group label per series (N,), for the group target, the series target's shrinkage
            and the CQR group pool. None means one pool.
        native (dict): Optional native bands {level: (lo, hi)} of shape (N, W, H).
        clip_lower (float): Lower bound of the intervals (resolve_clip_lower), or None.
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
    plan = fit_plan(pi_conf, has_band)
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


COMPUTED_GROUP_COLS = ("abc_xyz", "sb_class", "abc_class", "xyz_class")
MARGINAL_POOL = "__all__"


def group_columns(pi_conf) -> list:
    """User columns of the training data that coverage.group_col refers to (none for MMF-computed classes)."""
    group_col = pi_conf["coverage"]["group_col"]
    if group_col is None or group_col == "auto":
        return []
    cols = [group_col] if isinstance(group_col, str) else list(group_col)
    return [c for c in cols if c not in COMPUTED_GROUP_COLS]


def group_labels(features: pd.DataFrame, pi_conf):
    """
    Group label per series (in the order of features.index) for coverage.group_col, or None.
    "auto" is ABC x XYZ; several columns are combined as "a|b".
    """
    group_col = pi_conf["coverage"]["group_col"]
    if group_col is None:
        return None
    cols = ["abc_xyz"] if group_col == "auto" else ([group_col] if isinstance(group_col, str) else list(group_col))
    missing = [c for c in cols if c not in features.columns]
    if missing:
        raise ConfigurationError(f"prediction_intervals.coverage.group_col: column(s) {missing} not found")
    return features[cols].astype(str).agg("|".join, axis=1).to_numpy(dtype=object)


def positive_sigma(sigma: np.ndarray):
    """
    Replaces non-positive or missing scales (e.g. all-zero history) by the median positive scale.
    Returns: (np.ndarray, int): The scales and the number of series replaced.
    """
    sigma = np.asarray(sigma, dtype=float)
    bad = ~(np.isfinite(sigma) & (sigma > 0))
    if bad.all():
        return np.ones_like(sigma), int(bad.sum())
    return np.where(bad, np.median(sigma[~bad]), sigma), int(bad.sum())


def backtest_cubes(frame: pd.DataFrame, series_col: str, levels=()):
    """
    Cubes of one model's backtest rows from evaluation_output.
    Parameters:
        frame (pd.DataFrame): Rows with series_col, backtest_window_start_date, forecast and actual
            arrays, and optionally forecast_quantiles maps.
        series_col (str): Series id column.
        levels: Central levels whose native bands to extract from forecast_quantiles.
    Every series keeps its last W windows, W being the most common window count. Series with
    fewer windows, or with arrays of the wrong length or non-finite values, are dropped.
    Missing native quantiles fall back to the point forecast.
    Returns: dict with ids (N,), dates (N, W), y and yhat (N, W, H), native {level: (lo, hi)} or None,
        and dropped (number of series dropped).
    """
    frame = frame.sort_values([series_col, "backtest_window_start_date"])
    counts = frame.groupby(series_col).size()
    w = int(counts.mode().iloc[0])
    h = int(pd.Series([np.size(a) if a is not None else 0 for a in frame["forecast"]]).mode().iloc[0])
    has_q = "forecast_quantiles" in frame.columns
    keys = {lv: tuple(quantile_key(q) for q in level_quantiles(lv)) for lv in levels}
    ids, dates, ys, yhats, bands = [], [], [], [], []
    dropped = 0
    for sid, g in frame.groupby(series_col, sort=True):
        g = g.iloc[-w:]
        if len(g) < w:
            dropped += 1
            continue
        try:
            y = np.stack([np.asarray(a, dtype=float).reshape(-1) for a in g["actual"]])
            yhat = np.stack([np.asarray(a, dtype=float).reshape(-1) for a in g["forecast"]])
        except (ValueError, TypeError):
            dropped += 1
            continue
        if y.shape != (w, h) or yhat.shape != (w, h) or not (np.isfinite(y).all() and np.isfinite(yhat).all()):
            dropped += 1
            continue
        band = {}
        for lv, (k_lo, k_hi) in keys.items():
            los, his = [], []
            for m, fallback in zip(g["forecast_quantiles"] if has_q else [None] * w, yhat):
                # Spark returns map columns as dicts, or as lists of (key, value) pairs with some Arrow versions.
                m = dict(m) if m is not None and not isinstance(m, dict) else (m or {})
                lo = np.asarray(m.get(k_lo, fallback), dtype=float)
                hi = np.asarray(m.get(k_hi, fallback), dtype=float)
                if lo.shape != (h,) or hi.shape != (h,):
                    lo, hi = fallback, fallback
                los.append(np.where(np.isfinite(lo), lo, fallback))
                his.append(np.where(np.isfinite(hi), hi, fallback))
            band[lv] = (np.stack(los), np.stack(his))
        ids.append(sid)
        dates.append(g["backtest_window_start_date"].to_numpy())
        ys.append(y)
        yhats.append(yhat)
        bands.append(band)
    native = None
    if has_q and levels and frame["forecast_quantiles"].notna().any():
        native = {lv: (np.stack([b[lv][0] for b in bands]), np.stack([b[lv][1] for b in bands])) for lv in levels}
    return {
        "ids": np.array(ids, dtype=object),
        "dates": np.array(dates) if dates else np.empty((0, w)),
        "y": np.stack(ys) if ys else np.empty((0, w, h)),
        "yhat": np.stack(yhats) if yhats else np.empty((0, w, h)),
        "native": native if ids else None,
        "dropped": dropped,
    }


def _metric_name(name: str, level: float, native: bool) -> str:
    return f"{'native_' if native else ''}{name}_{level:g}"


def calibration_rows(model: str, series_col: str, ids, results: dict, features: pd.DataFrame, labels,
                     n_windows: int, min_scores_per_group: int, clip_lower) -> pd.DataFrame:
    """
    calibration_output rows of one model: one row per series, step and level.
    q_lo and q_hi are in units of sigma (or 1 for score "signed"; of the native half-width for CQR "scaled").
    """
    n = len(ids)
    feats = features.loc[ids]
    if labels is None:
        pool = np.full(n, MARGINAL_POOL, dtype=object)
    else:
        pool = merge_small_groups(labels, n_windows, min_scores_per_group)
    frames = []
    for level, r in results.items():
        h = r["q_lo"].shape[1]
        pool_key = (np.full(n, MARGINAL_POOL, dtype=object) if r["target"] == "marginal"
                    else np.full(n, "", dtype=object) if r["method"] == "native" else pool)
        frames.append(pd.DataFrame({
            "model": model,
            "method": r["method"],
            "score": r["score"],
            "coverage_target": r["target"],
            series_col: np.repeat(np.asarray(ids, dtype=object), h),
            "pool_key": np.repeat(pool_key, h),
            "step": np.tile(np.arange(1, h + 1), n),
            "level": float(level),
            "q_lo": r["q_lo"].ravel(),
            "q_hi": r["q_hi"].ravel(),
            "n_windows": n_windows,
            "shrinkage_weight": r["shrinkage_weight"],
            "infinite_bound": np.repeat(r["infinite"], h),
            "clip_lower": clip_lower,
            "sigma": np.repeat(feats["sigma"].to_numpy(dtype=float), h),
            "abc_class": np.repeat(feats["abc_class"].to_numpy(dtype=object), h),
            "xyz_class": np.repeat(feats["xyz_class"].to_numpy(dtype=object), h),
            "sb_class": np.repeat(feats["sb_class"].to_numpy(dtype=object), h),
        }))
    return pd.concat(frames, ignore_index=True)


def metric_rows(model: str, series_col: str, ids, dates, results: dict) -> pd.DataFrame:
    """
    Interval metric rows of one model for evaluation_output: one row per series, window and metric
    (coverage_<L>, msis_<L>, width_<L>, computed leave-one-window-out, and native_* for the native band).
    """
    n, w = dates.shape
    frames = []
    for level, r in results.items():
        for native, metrics in ((False, r["metrics"]), (True, r["native_metrics"])):
            for name, values in metrics.items():
                frames.append(pd.DataFrame({
                    series_col: np.repeat(np.asarray(ids, dtype=object), w),
                    "backtest_window_start_date": dates.ravel(),
                    "metric_name": _metric_name(name, level, native),
                    "metric_value": values.ravel(),
                }))
    if not frames:
        return pd.DataFrame(columns=[series_col, "backtest_window_start_date", "metric_name", "metric_value"])
    out = pd.concat(frames, ignore_index=True)
    out["model"] = model
    return out


# ---------------------------------------------------------------------------
# Applying the calibration to forecasts (scoring).
# ---------------------------------------------------------------------------


def quantiles_column(target: str) -> str:
    """Name of the quantile-map column of scoring_output (output_format quantile_map or both)."""
    return f"{target}_quantiles"


def interval_column(target: str, side: str, level: float) -> str:
    """Name of a bound column of scoring_output (output_format lo_hi or both), e.g. y_lo_80 or y_hi_97_5."""
    return f"{target}_{side}_{level:g}".replace(".", "_")


def interval_columns(target: str, levels, output_format: str) -> list:
    """Names of the interval columns that scoring_output gets for an output_format."""
    cols = []
    if output_format in ("quantile_map", "both"):
        cols.append(quantiles_column(target))
    if output_format in ("lo_hi", "both"):
        cols += [interval_column(target, side, lv) for lv in levels for side in ("lo", "hi")]
    return cols


def _as_dict(m):
    if m is None or (isinstance(m, float) and np.isnan(m)):
        return {}
    return dict(m) if not isinstance(m, dict) else m


def _corrections(cal_rows: pd.DataFrame, series_col: str):
    """
    Per-(series, level) corrections from calibration_output rows, plus per-level fallbacks (the
    median correction per step, and the median scale) for series without rows.
    """
    rows = cal_rows.sort_values([series_col, "level", "step"])
    per_series, fallback = {}, {}
    for (sid, level), g in rows.groupby([series_col, "level"], sort=False):
        per_series[(str(sid), float(level))] = {
            "q_lo": g["q_lo"].to_numpy(float), "q_hi": g["q_hi"].to_numpy(float),
            "sigma": float(g["sigma"].iloc[0]), "method": g["method"].iloc[0], "score": g["score"].iloc[0],
        }
    for level, g in rows.groupby("level"):
        steps = g.groupby("step")
        fallback[float(level)] = {
            "q_lo": steps["q_lo"].median().to_numpy(float), "q_hi": steps["q_hi"].median().to_numpy(float),
            "sigma": float(g.drop_duplicates(series_col)["sigma"].median()),
            "method": g["method"].mode().iloc[0], "score": g["score"].mode().iloc[0],
        }
    return per_series, fallback


def series_bounds(yhat: np.ndarray, native: dict, corr: dict, level: float, clip_lower):
    """
    Interval bounds of one series' forecast for one level.
    Parameters:
        yhat (np.ndarray): Point forecast (H,).
        native (dict): The series' native quantiles {key: array (H,)} (may be empty).
        corr (dict): q_lo, q_hi (H,), sigma, method and score of the series from calibration_output.
        level (float): Central level in percent.
        clip_lower (float): Lower bound, or None.
    Returns: (lo, hi) arrays (H,), or None when the method needs a native band that is missing.
    """
    h = len(yhat)
    q_lo, q_hi = corr["q_lo"][:h], corr["q_hi"][:h]
    if len(q_lo) != h:
        return None
    sigma = corr["sigma"]
    if corr["method"] == "conformal":
        scale = sigma if corr["score"] == "signed_normalized" else 1.0
        lo, hi = yhat + scale * q_lo, yhat + scale * q_hi
    else:
        k_lo, k_hi = (quantile_key(q) for q in level_quantiles(level))
        if k_lo not in native or k_hi not in native:
            return None
        lo, hi = np.asarray(native[k_lo], dtype=float), np.asarray(native[k_hi], dtype=float)
        if clip_lower is not None:
            lo = np.maximum(lo, clip_lower)
        if corr["method"] == "cqr":
            if corr["score"] == "scaled":
                half = np.maximum((hi - lo) / 2, 1e-3 * sigma)
                lo, hi = yhat - q_lo * half, yhat + q_hi * half
            else:
                lo, hi = lo - sigma * q_lo, hi + sigma * q_hi
    lo, hi = finalize_bounds(lo, hi, clip_lower)
    return lo, hi


def correction_column(level_index: int, field: str) -> str:
    """Wide column of a forecast batch holding one correction field (q_lo, q_hi, sigma, method, score) of one level."""
    return f"_corr{level_index}_{field}"


CORRECTION_FIELDS = ("q_lo", "q_hi", "sigma", "method", "score")


def _add_intervals(forecasts: pd.DataFrame, corrections, series_col: str, target: str, levels, output_format: str,
                   clip_lower, fallback: dict):
    """
    Shared loop of apply_calibration and apply_joined_corrections.
    corrections(i, sid, level) returns the series' correction dict for a level, or None.
    """
    out = forecasts.drop(columns=["forecast_quantiles"], errors="ignore")
    out = out[[c for c in out.columns if not str(c).startswith("_corr")]].copy()
    natives = (forecasts["forecast_quantiles"].map(_as_dict) if "forecast_quantiles" in forecasts.columns
               else pd.Series([{}] * len(forecasts), index=forecasts.index))
    maps, bounds = [], {(side, lv): [] for lv in levels for side in ("lo", "hi")}
    stats = {"fallback": 0, "null": 0}
    for i, (sid, yhat, native) in enumerate(zip(forecasts[series_col], forecasts[target], natives)):
        if yhat is None:
            maps.append(None)
            for key in bounds:
                bounds[key].append(None)
            continue
        yhat = np.asarray(yhat, dtype=float)
        qmap, used_fallback, missing = {}, False, False
        for lv in levels:
            corr = corrections(i, sid, lv)
            if corr is None:
                corr, used_fallback = fallback.get(float(lv)), True
            b = series_bounds(yhat, native, corr, lv, clip_lower) if corr is not None else None
            if b is None:
                missing = True
                bounds[("lo", lv)].append(None)
                bounds[("hi", lv)].append(None)
                continue
            lo, hi = b
            k_lo, k_hi = (quantile_key(q) for q in level_quantiles(lv))
            qmap[k_lo], qmap[k_hi] = [float(x) for x in lo], [float(x) for x in hi]
            bounds[("lo", lv)].append([float(x) for x in lo])
            bounds[("hi", lv)].append([float(x) for x in hi])
            if corr["method"] != "conformal" and "0.5" in native:
                qmap["0.5"] = [float(x) for x in native["0.5"]]
        stats["fallback"] += used_fallback
        stats["null"] += missing
        maps.append({k: qmap[k] for k in sorted(qmap, key=float)} if qmap else None)
    if output_format in ("quantile_map", "both"):
        out[quantiles_column(target)] = maps
    if output_format in ("lo_hi", "both"):
        for (side, lv), values in bounds.items():
            out[interval_column(target, side, lv)] = values
    return out, stats


def apply_calibration(forecasts: pd.DataFrame, cal_rows: pd.DataFrame, series_col: str, target: str,
                      levels, output_format: str = "quantile_map"):
    """
    Adds the prediction-interval columns of one model's scoring output, from its calibration_output rows.
    Pure pandas reference of the distributed path (apply_joined_corrections), for tests and small runs.
    Parameters:
        forecasts (pd.DataFrame): One row per series with series_col, the target array (point forecast)
            and optionally forecast_quantiles (native quantiles at forecast time).
        cal_rows (pd.DataFrame): The model's calibration_output rows.
        series_col, target (str): Column names.
        levels: Central levels in percent.
        output_format (str): quantile_map, lo_hi or both.
    Series without calibration rows use the per-step median correction of the model. Bounds are null
    when they cannot be computed (no calibration at all, or a missing native band).
    Returns: (pd.DataFrame, dict): The forecasts with the interval columns (forecast_quantiles dropped),
        and counts of series that used the fallback or got null bounds.
    """
    levels = sorted(levels)
    per_series, fallback = _corrections(cal_rows, series_col) if len(cal_rows) else ({}, {})
    clip_lower = None
    if len(cal_rows) and cal_rows["clip_lower"].notna().any():
        clip_lower = float(cal_rows["clip_lower"].dropna().iloc[0])
    return _add_intervals(forecasts, lambda i, sid, lv: per_series.get((str(sid), float(lv))), series_col, target,
                          levels, output_format, clip_lower, fallback)


def apply_joined_corrections(batch: pd.DataFrame, series_col: str, target: str, levels, output_format: str,
                             clip_lower, fallback: dict):
    """
    Adds the prediction-interval columns to a batch of forecasts that carry their own corrections in the
    correction_column(i, field) columns (level i of sorted levels; null when the series has no calibration).
    Runs in mapInPandas at scoring time, so that nothing is collected to the driver.
    Parameters:
        fallback (dict): level -> correction dict used for series without calibration (median_corrections).
    Returns: (pd.DataFrame, dict): As apply_calibration; the correction columns are dropped.
    """
    levels = sorted(levels)
    cols = {(j, f): batch[correction_column(j, f)].to_numpy() for j in range(len(levels)) for f in CORRECTION_FIELDS
            if correction_column(j, f) in batch.columns}
    index = {float(lv): j for j, lv in enumerate(levels)}

    def corrections(i, sid, lv):
        j = index[float(lv)]
        if (j, "q_lo") not in cols:
            return None
        q_lo = cols[(j, "q_lo")][i]
        if q_lo is None or (isinstance(q_lo, float) and np.isnan(q_lo)):
            return None
        return {"q_lo": np.asarray(q_lo, dtype=float), "q_hi": np.asarray(cols[(j, "q_hi")][i], dtype=float),
                "sigma": float(cols[(j, "sigma")][i]), "method": cols[(j, "method")][i],
                "score": cols[(j, "score")][i]}

    return _add_intervals(batch, corrections, series_col, target, levels, output_format, clip_lower, fallback)


def median_corrections(stats: pd.DataFrame) -> dict:
    """
    Fallback corrections per level, from per-(level, step) aggregates of calibration_output with columns
    level, step, q_lo and q_hi (medians over series), sigma (median scale), method and score (most frequent).
    """
    fallback = {}
    for level, g in stats.sort_values(["level", "step"]).groupby("level"):
        fallback[float(level)] = {
            "q_lo": g["q_lo"].to_numpy(float), "q_hi": g["q_hi"].to_numpy(float),
            "sigma": float(g["sigma"].iloc[0]), "method": g["method"].iloc[0], "score": g["score"].iloc[0],
        }
    return fallback
