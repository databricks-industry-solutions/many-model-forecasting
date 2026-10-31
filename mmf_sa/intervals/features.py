"""Features of the series for the calibration: scale, classes and group labels, and the backtest cubes of one model.

Pure numpy/pandas.
"""
import numpy as np
import pandas as pd

from mmf_sa.exceptions import ConfigurationError

from mmf_sa.intervals.config import level_quantiles
from mmf_sa.intervals.quantiles import quantile_key


# Syntetos-Boylan thresholds (Syntetos, Boylan and Croston, 2005).
SB_ADI = 1.32

SB_CV2 = 0.49

# ABC cumulative volume shares.
ABC_SHARES = (0.80, 0.95)


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
    # Ties are broken by series id, so that the Spark path (intervals.spark.classify) gives the same classes.
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


# Season length of the seasonal-naive scale, and length of the history used for the scale
# and the classes (the periods just before the first backtest window), per frequency.
SEASON_LENGTH = {"H": 24, "D": 7, "W": 52, "M": 12}

FEATURE_PERIODS = {"H": 24 * 28, "D": 365, "W": 104, "M": 24}

COMPUTED_GROUP_COLS = ("abc_xyz", "sb_class", "abc_class", "xyz_class")


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
