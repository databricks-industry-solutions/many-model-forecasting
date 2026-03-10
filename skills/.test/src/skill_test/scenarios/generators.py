import numpy as np
import pandas as pd


def generate_base_series(n_series=50, n_points=365, freq="D", seed=42):
    """Generate clean base time series with trend + seasonality + noise."""
    rng = np.random.default_rng(seed)
    records = []
    for i in range(n_series):
        dates = pd.date_range("2022-01-01", periods=n_points, freq=freq)
        trend = np.linspace(100, 100 + rng.uniform(-50, 150), n_points)
        seasonal = 20 * np.sin(2 * np.pi * np.arange(n_points) / 7)
        noise = rng.normal(0, 5, n_points)
        values = trend + seasonal + noise
        for d, v in zip(dates, values, strict=True):
            records.append({"unique_id": f"series_{i:03d}", "ds": d, "y": v})
    return pd.DataFrame(records)


def inject_missing_values(df, pct=0.1, seed=42):
    """Set random target values to NaN."""
    rng = np.random.default_rng(seed)
    df = df.copy()
    mask = rng.random(len(df)) < pct
    df.loc[mask, "y"] = np.nan
    return df


def inject_missing_blocks(df, block_size=7, n_blocks=3, seed=42):
    """Remove contiguous blocks of data (simulates sensor outages)."""
    rng = np.random.default_rng(seed)
    df = df.copy()
    series_ids = df["unique_id"].unique()
    for sid in series_ids:
        series_mask = df["unique_id"] == sid
        series_indices = df.index[series_mask]
        n_available = len(series_indices)
        if n_available <= block_size:
            continue
        for _ in range(n_blocks):
            start = rng.integers(0, max(1, n_available - block_size))
            block_idx = series_indices[start : start + block_size]
            df.loc[block_idx, "y"] = np.nan
    return df


def inject_outliers(df, pct=0.02, multiplier=10, seed=42):
    """Add extreme point outliers."""
    rng = np.random.default_rng(seed)
    df = df.copy()
    mask = rng.random(len(df)) < pct
    std = df["y"].std()
    signs = rng.choice([-1, 1], size=mask.sum())
    df.loc[mask, "y"] = df.loc[mask, "y"] + signs * multiplier * std
    return df


def inject_level_shift(df, fraction=0.5, magnitude=100):
    """Regime change at a given fraction of each series."""
    df = df.copy()
    for sid in df["unique_id"].unique():
        series_mask = df["unique_id"] == sid
        n = series_mask.sum()
        shift_point = int(n * fraction)
        indices = df.index[series_mask]
        df.loc[indices[shift_point:], "y"] = df.loc[indices[shift_point:], "y"] + magnitude
    return df


def make_irregular_timestamps(df, drop_pct=0.2, seed=42):
    """Randomly drop rows to create irregular timestamps."""
    rng = np.random.default_rng(seed)
    mask = rng.random(len(df)) >= drop_pct
    return df.loc[mask].reset_index(drop=True)


def inject_duplicates(df, dup_pct=0.05, seed=42):
    """Insert exact duplicate rows."""
    rng = np.random.default_rng(seed)
    n_dups = int(len(df) * dup_pct)
    dup_indices = rng.choice(len(df), size=n_dups, replace=True)
    dups = df.iloc[dup_indices].copy()
    return pd.concat([df, dups], ignore_index=True)


def inject_wrong_types(df):
    """Convert target column to string dtype (simulates type corruption)."""
    df = df.copy()
    df["y"] = df["y"].astype(str)
    return df


def create_mixed_frequency_data(n_daily=10, n_weekly=10, n_monthly=5, seed=42):
    """Mix daily/weekly/monthly series in one table."""
    rng = np.random.default_rng(seed)
    records = []

    for i in range(n_daily):
        dates = pd.date_range("2022-01-01", periods=365, freq="D")
        values = 100 + rng.normal(0, 10, len(dates))
        for d, v in zip(dates, values, strict=True):
            records.append({"unique_id": f"daily_{i:03d}", "ds": d, "y": v})

    for i in range(n_weekly):
        dates = pd.date_range("2022-01-01", periods=52, freq="W")
        values = 500 + rng.normal(0, 50, len(dates))
        for d, v in zip(dates, values, strict=True):
            records.append({"unique_id": f"weekly_{i:03d}", "ds": d, "y": v})

    for i in range(n_monthly):
        dates = pd.date_range("2022-01-01", periods=24, freq="MS")
        values = 1000 + rng.normal(0, 100, len(dates))
        for d, v in zip(dates, values, strict=True):
            records.append({"unique_id": f"monthly_{i:03d}", "ds": d, "y": v})

    return pd.DataFrame(records)
