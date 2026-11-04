"""The contract between the forecasting models and the prediction intervals.

A model that returns native quantiles adds one column per quantile level next to its point forecast, named
QUANTILE_COLUMN_PREFIX + quantile_key(q). The helpers here name, pack and unpack those columns, and turn them into the
quantile maps written to evaluation_output and scoring_output. The models import only this module (and config for the
level helpers). Pure numpy/pandas.
"""
import numpy as np
import pandas as pd


# Prefix of the per-quantile columns that a model's predict() adds next to the point forecast
# when native quantiles are requested, e.g. "__q_0.1".
QUANTILE_COLUMN_PREFIX = "__q_"

# Fewest sample paths from which a sampling model's native quantiles are estimated well enough at the tails
# (e.g. 2.5% and 97.5%); such models log a warning below it.
MIN_SAMPLES_FOR_QUANTILES = 100


def quantile_key(q: float) -> str:
    """Key of a quantile level in the quantile maps: no trailing zeros, e.g. "0.1", "0.025", "0.975"."""
    return format(round(float(q), 6), "g")


def quantile_columns(frame: pd.DataFrame) -> dict:
    """Maps quantile keys to the QUANTILE_COLUMN_PREFIX columns of a frame."""
    return {c[len(QUANTILE_COLUMN_PREFIX):]: c for c in frame.columns if str(c).startswith(QUANTILE_COLUMN_PREFIX)}


def sorted_quantiles(qmap: dict):
    """
    A quantile map {key: array over the horizon} as written to the output tables: keys whose array is missing or not
    finite are dropped (e.g. levels outside a model's native range), and the values are sorted across levels at each
    step, so that quantiles never cross. Returns None when no key is left.
    """
    arrays = {}
    for key, values in qmap.items():
        if values is None:
            continue
        values = np.asarray(values, dtype=np.float64)
        if values.size and np.isfinite(values).all():
            arrays[key] = values
    if not arrays:
        return None
    keys = sorted(arrays, key=float)
    if len({arrays[k].shape for k in keys}) > 1:
        return None
    stacked = np.sort(np.stack([arrays[k] for k in keys]), axis=0)
    return {key: [float(x) for x in stacked[i]] for i, key in enumerate(keys)}


def quantile_maps(pred_df: pd.DataFrame, group_col: str) -> dict:
    """
    Maps each series of a multi-series forecast frame to its quantile map (sorted_quantiles), or None for a series
    without usable quantiles; {} for a frame without quantile columns.
    """
    cols = quantile_columns(pred_df)
    if not cols:
        return {}
    keys = pred_df[group_col].to_numpy()
    values = {key: pred_df[c].to_numpy() for key, c in cols.items()}
    return {k: sorted_quantiles({key: values[key][i] for key in values}) for i, k in enumerate(keys)}


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
