"""Synthetic time-series data scenarios for testing MMF skills."""

from skill_test.scenarios.generators import (
    create_mixed_frequency_data,
    generate_base_series,
    inject_duplicates,
    inject_level_shift,
    inject_missing_blocks,
    inject_missing_values,
    inject_outliers,
    inject_wrong_types,
    make_irregular_timestamps,
)
from skill_test.scenarios.scenarios import TEST_SCENARIOS, build_scenario

__all__ = [
    "create_mixed_frequency_data",
    "generate_base_series",
    "inject_duplicates",
    "inject_level_shift",
    "inject_missing_blocks",
    "inject_missing_values",
    "inject_outliers",
    "inject_wrong_types",
    "make_irregular_timestamps",
    "TEST_SCENARIOS",
    "build_scenario",
]
