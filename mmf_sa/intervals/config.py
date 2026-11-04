"""Configuration of the prediction intervals: the `prediction_intervals` block, its defaults and validation.

Also the level helpers that map interval levels (in percent) to quantile levels. Pure Python and OmegaConf, no Spark.
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
    "output_format": "quantile_map",
}

_OUTPUT_FORMATS = {"quantile_map", "lo_hi", "both"}


def _is_number(value) -> bool:
    return isinstance(value, numbers.Real) and not isinstance(value, bool)


def _check_level(value, where: str) -> None:
    if not _is_number(value) or not 0 < value < 100:
        raise ConfigurationError(f"prediction_intervals.{where}: levels must be numbers in (0, 100), got {value!r}")


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
    unknown = set(user) - set(DEFAULT_PREDICTION_INTERVALS)
    if unknown:
        raise ConfigurationError(
            f"Unknown key(s) in prediction_intervals: {', '.join(sorted(unknown))}. "
            f"Supported keys: {', '.join(sorted(DEFAULT_PREDICTION_INTERVALS))}"
        )

    merged = OmegaConf.merge(OmegaConf.create(DEFAULT_PREDICTION_INTERVALS), OmegaConf.create(user))
    pi = OmegaConf.to_container(merged, resolve=True)

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

    if pi["output_format"] not in _OUTPUT_FORMATS:
        raise ConfigurationError(
            f"prediction_intervals.output_format must be one of {', '.join(sorted(_OUTPUT_FORMATS))}, "
            f"got {pi['output_format']!r}")

    return OmegaConf.create(pi)


def level_quantiles(level: float):
    """Lower and upper quantile levels of a central interval level in percent (80 -> 0.1, 0.9)."""
    a = (100 - level) / 200
    return round(a, 6), round(1 - a, 6)


def requested_quantiles(pi_conf) -> list:
    """
    Quantile levels to request from models with native intervals, given the prediction_intervals block.
    Returns the sorted quantiles of every configured level and the median, or the configured
    `quantiles` plus the median.
    """
    if pi_conf.get("quantiles"):
        qs = {round(float(q), 6) for q in pi_conf["quantiles"]}
    else:
        qs = {q for level in set(pi_conf["levels"] or []) for q in level_quantiles(level)}
    qs.add(0.5)
    return sorted(qs)


def central_levels(quantiles) -> list:
    """Central interval levels in percent whose bounds are the given quantiles (0.1 or 0.9 -> 80)."""
    levels = {round(abs(200 * q - 100), 6) for q in quantiles if q != 0.5}
    return sorted(int(lv) if float(lv).is_integer() else lv for lv in levels)


def interval_levels(pi_conf) -> list:
    """Central levels of the interval columns: `levels`, or the central levels of `quantiles`."""
    if pi_conf.get("quantiles"):
        return central_levels(pi_conf["quantiles"])
    return sorted(pi_conf["levels"])
