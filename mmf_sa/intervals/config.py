"""Configuration of the prediction intervals: the `prediction_intervals` block, its defaults and validation.

Also the level helpers that map interval levels (in percent) to quantile levels, and the interval method of each model.
Pure Python and OmegaConf, no Spark.
"""
import numbers
from typing import Any, Dict, Optional, Union

from omegaconf import DictConfig, OmegaConf

from mmf_sa.exceptions import ConfigurationError


# Defaults of the `prediction_intervals` block. The shipped forecasting_conf_*.yaml files
# carry the same block, with intervals disabled so that existing pipelines are unchanged.
DEFAULT_PREDICTION_INTERVALS: Dict[str, Any] = {
    "enabled": False,
    "levels": [80, 95],
    "quantiles": None,
    "method": "auto",
    "method_by_model": {},
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
    if "method_by_model" in user and user["method_by_model"] is None:
        user["method_by_model"] = {}
    if not isinstance(user.get("method_by_model", {}), dict):
        raise ConfigurationError(
            f"prediction_intervals.method_by_model must map model names to methods, got {user['method_by_model']!r}")
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
    for name, method in pi["method_by_model"].items():
        if method not in _CHOICES["method"]:
            raise ConfigurationError(
                f"prediction_intervals.method_by_model[{name!r}] must be one of "
                f"{', '.join(sorted(_CHOICES['method']))}, got {method!r}")
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


def model_method(pi_conf, model: Optional[str] = None) -> str:
    """
    The interval method of one model: its method_by_model entry, or the method of the block when the model
    is not listed. Without a model name, the method of the block.
    """
    if model is None:
        return pi_conf["method"]
    return pi_conf["method_by_model"].get(model, pi_conf["method"])
