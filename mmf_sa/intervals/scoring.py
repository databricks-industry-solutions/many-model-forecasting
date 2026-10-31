"""Applying the calibration to forecasts at scoring, and the output rows of the calibration.

The rows are calibration_output and the interval metrics. Pure pandas: apply_calibration is the reference of
apply_joined_corrections, which runs in mapInPandas at scoring time.
"""
import numpy as np
import pandas as pd

from mmf_sa.intervals.config import level_quantiles
from mmf_sa.intervals.conformal import MARGINAL_POOL, finalize_bounds, merge_small_groups
from mmf_sa.intervals.quantiles import quantile_key


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
