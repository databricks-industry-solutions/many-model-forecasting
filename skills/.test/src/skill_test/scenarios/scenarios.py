import numpy as np

from skill_test.scenarios.generators import (
    create_mixed_frequency_data,
    generate_base_series,
    inject_duplicates,
    inject_missing_values,
    inject_outliers,
    inject_wrong_types,
    make_irregular_timestamps,
)

TEST_SCENARIOS = {
    "clean_baseline": {"missing": 0, "outliers": 0, "negatives": 0, "irregular": False},
    "moderate_noise": {"missing": 0.05, "outliers": 0.01, "negatives": 0.05, "irregular": False},
    "heavy_noise": {"missing": 0.15, "outliers": 0.05, "negatives": 0.15, "irregular": True},
    "boundary_missing": {"missing": 0.199, "outliers": 0, "negatives": 0, "irregular": False},
    "over_threshold": {"missing": 0.25, "outliers": 0, "negatives": 0, "irregular": False},
    "all_negative": {"missing": 0, "outliers": 0, "negatives": 1.0, "irregular": False},
    "intermittent": {"missing": 0, "outliers": 0, "negatives": 0, "zero_pct": 0.85},
    "duplicates": {"missing": 0, "outliers": 0, "negatives": 0, "duplicates": 0.05},
    "wrong_types": {"string_target": True, "mixed_date_formats": True},
    "short_history": {"n_points": 5, "prediction_length": 10},
    "mixed_frequencies": {"daily_series": 10, "weekly_series": 10, "monthly_series": 5},
}


def build_scenario(name, seed=42):
    """Build a DataFrame for a named test scenario."""
    config = TEST_SCENARIOS[name]

    if name == "mixed_frequencies":
        return create_mixed_frequency_data(
            n_daily=config.get("daily_series", 10),
            n_weekly=config.get("weekly_series", 10),
            n_monthly=config.get("monthly_series", 5),
            seed=seed,
        )

    if name == "short_history":
        return generate_base_series(
            n_series=5,
            n_points=config.get("n_points", 5),
            seed=seed,
        )

    if name == "wrong_types":
        df = generate_base_series(n_series=5, n_points=50, seed=seed)
        return inject_wrong_types(df)

    # Standard scenarios: start from base, apply injections
    df = generate_base_series(n_series=10, n_points=100, seed=seed)

    missing_pct = config.get("missing", 0)
    if missing_pct > 0:
        df = inject_missing_values(df, pct=missing_pct, seed=seed)

    outlier_pct = config.get("outliers", 0)
    if outlier_pct > 0:
        df = inject_outliers(df, pct=outlier_pct, seed=seed)

    negative_pct = config.get("negatives", 0)
    if negative_pct > 0:
        rng = np.random.default_rng(seed)
        mask = rng.random(len(df)) < negative_pct
        df.loc[mask, "y"] = -abs(df.loc[mask, "y"])

    if config.get("irregular", False):
        df = make_irregular_timestamps(df, drop_pct=0.1, seed=seed)

    zero_pct = config.get("zero_pct", 0)
    if zero_pct > 0:
        rng = np.random.default_rng(seed + 1)
        mask = rng.random(len(df)) < zero_pct
        df.loc[mask, "y"] = 0.0

    dup_pct = config.get("duplicates", 0)
    if dup_pct > 0:
        df = inject_duplicates(df, dup_pct=dup_pct, seed=seed)

    return df
