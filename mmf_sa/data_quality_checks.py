import functools
import logging
from typing import Dict, Any, Union, List, Tuple, Optional, Set
from dataclasses import dataclass
from enum import Enum
from omegaconf import DictConfig
import pandas as pd
import pyspark
import pyspark.sql.functions as F
from pyspark.sql import SparkSession, DataFrame as SparkDataFrame
from pyspark.sql.types import StringType
from mmf_sa.exceptions import (
    DataQualityError,
    ParameterValidationError,
    EmptyDatasetError,
    InvalidConfigurationError
)

_logger = logging.getLogger(__name__)

# Constants
class SupportedFrequencies(Enum):
    """Supported time series frequencies."""
    HOURLY = "H"
    DAILY = "D"
    WEEKLY = "W"
    MONTHLY = "M"

class ExternalRegressorTypes(Enum):
    """Types of external regressors."""
    STATIC_FEATURES = "static_features"
    DYNAMIC_FUTURE_NUMERICAL = "dynamic_future_numerical"
    DYNAMIC_FUTURE_CATEGORICAL = "dynamic_future_categorical"
    DYNAMIC_HISTORICAL_NUMERICAL = "dynamic_historical_numerical"
    DYNAMIC_HISTORICAL_CATEGORICAL = "dynamic_historical_categorical"

# Default thresholds
DEFAULT_MISSING_DATA_THRESHOLD = 0.2
DEFAULT_NEGATIVE_DATA_THRESHOLD = 0.2
DEFAULT_MIN_TRAIN_PREDICT_RATIO = 1.0

@dataclass
class DataQualityThresholds:
    """Configuration class for data quality check thresholds."""
    missing_data_threshold: float = DEFAULT_MISSING_DATA_THRESHOLD
    negative_data_threshold: float = DEFAULT_NEGATIVE_DATA_THRESHOLD
    min_train_predict_ratio: float = DEFAULT_MIN_TRAIN_PREDICT_RATIO

    def __post_init__(self):
        """Validate threshold values."""
        self._validate_thresholds()

    def _validate_thresholds(self):
        """Validate all threshold values."""
        if not (0 <= self.missing_data_threshold <= 1):
            raise ParameterValidationError(
                f"missing_data_threshold must be between 0 and 1, got {self.missing_data_threshold}"
            )
        if not (0 <= self.negative_data_threshold <= 1):
            raise ParameterValidationError(
                f"negative_data_threshold must be between 0 and 1, got {self.negative_data_threshold}"
            )
        if self.min_train_predict_ratio < 0:
            raise ParameterValidationError(
                f"min_train_predict_ratio must be non-negative, got {self.min_train_predict_ratio}"
            )

@dataclass
class DataQualityMetrics:
    """Data structure to track data quality metrics."""
    total_groups: int = 0
    removed_groups: int = 0
    removal_reasons: Dict[str, int] = None

    def __post_init__(self):
        if self.removal_reasons is None:
            self.removal_reasons = {}

    def add_removal_reason(self, reason: str, count: int = 1):
        """Add a removal reason with count."""
        self.removal_reasons[reason] = self.removal_reasons.get(reason, 0) + count

    def get_removal_rate(self) -> float:
        """Get the removal rate as a percentage."""
        if self.total_groups == 0:
            return 0.0
        return (self.removed_groups / self.total_groups) * 100

@dataclass
class ValidationResult:
    """Result of a data quality validation check."""
    is_valid: bool
    reason: Optional[str] = None
    processed_data: Optional[pd.DataFrame] = None

    @classmethod
    def success(cls, processed_data: Optional[pd.DataFrame] = None):
        """Create a successful validation result."""
        return cls(is_valid=True, processed_data=processed_data)

    @classmethod
    def failure(cls, reason: str):
        """Create a failed validation result."""
        return cls(is_valid=False, reason=reason)

class DateOffsetUtility:
    """Utility class for date offset calculations."""

    @staticmethod
    def get_backtest_offset(freq: str, backtest_length: int) -> Optional[pd.DateOffset]:
        """
        Calculate backtest offset based on frequency and length.

        Args:
            freq: Frequency string ("H", "D", "W", "M")
            backtest_length: Length in frequency units

        Returns:
            DateOffset object or None if frequency not supported
        """
        try:
            freq_enum = SupportedFrequencies(freq)
        except ValueError:
            _logger.warning(f"Unsupported frequency: {freq}")
            return None

        offset_map = {
            SupportedFrequencies.HOURLY: lambda length: pd.DateOffset(hours=length),
            SupportedFrequencies.DAILY: lambda length: pd.DateOffset(days=length),
            SupportedFrequencies.WEEKLY: lambda length: pd.DateOffset(weeks=length),
            SupportedFrequencies.MONTHLY: lambda length: pd.DateOffset(months=length)
        }

        return offset_map[freq_enum](backtest_length)

class ExternalRegressorValidator:
    """Validator for external regressors."""

    def __init__(self, conf: DictConfig):
        self.conf = conf
        self.regressor_types = [e.value for e in ExternalRegressorTypes]

    def get_external_regressors(self) -> Dict[str, List[str]]:
        """Get all external regressors from configuration."""
        external_regressors = {}
        for regressor_type in self.regressor_types:
            regressors = self.conf.get(regressor_type, None)
            if regressors:
                external_regressors[regressor_type] = regressors
        return external_regressors

    def has_external_regressors(self) -> bool:
        """Check if any external regressors are configured."""
        return any(
            self.conf.get(regressor_type, None)
            for regressor_type in self.regressor_types
        )

    def validate_resampling_compatibility(self):
        """Validate that resampling is disabled when external regressors are provided."""
        if self.has_external_regressors():
            _logger.info("External regressors detected, checking resampling configuration")
            if self.conf.get("resample"):
                error_msg = "Resampling must be disabled when external regressors are provided"
                _logger.error(error_msg)
                raise InvalidConfigurationError(error_msg)
            _logger.info("External regressors configuration validation passed")
        else:
            _logger.debug("No external regressors detected")

    def check_nulls_in_regressors(self, df: pd.DataFrame, group_id: str) -> ValidationResult:
        """
        Check if external regressors contain null values (Pandas-based, for standalone use).

        Args:
            df: DataFrame for a single group
            group_id: Group identifier for logging

        Returns:
            ValidationResult with check outcome
        """
        external_regressors = self.get_external_regressors()

        for regressor_type, features in external_regressors.items():
            if df[features].isnull().values.any():
                reason = f"null values in {regressor_type.replace('_', ' ')}"
                _logger.debug(f"Group {group_id}: {reason}")
                return ValidationResult.failure(reason)

        return ValidationResult.success()


class DataQualityChecks:
    """
    Spark-native data quality checks for time series data.

    All checks run as distributed Spark operations — the full dataset
    is never collected to the driver node.

    Checks performed:
    - Parameter validation (backtest length, external regressors)
    - Null value checks for external regressors
    - Training period length validation
    - Missing data detection and optional resampling
    - Negative value detection
    """

    _FREQ_INTERVAL_MAP = {
        "H": "INTERVAL 1 HOUR",
        "D": "INTERVAL 1 DAY",
        "W": "INTERVAL 7 DAYS",
        "M": "INTERVAL 1 MONTH",
    }

    def __init__(
        self,
        df: SparkDataFrame,
        conf: DictConfig,
        spark: SparkSession = None,
        thresholds: Optional[DataQualityThresholds] = None,
    ):
        self.df = df
        self.conf = conf
        self.spark = spark or df.sparkSession
        self.thresholds = thresholds or DataQualityThresholds()
        self.metrics = DataQualityMetrics()
        self.regressor_validator = ExternalRegressorValidator(conf)

        self._validate_configuration()

    def _validate_configuration(self):
        """Validate essential configuration parameters."""
        required_params = ["group_id", "date_col", "target", "freq", "backtest_length", "prediction_length"]

        missing_params = [param for param in required_params if param not in self.conf]
        if missing_params:
            raise InvalidConfigurationError(f"Missing required parameters: {missing_params}")

        try:
            SupportedFrequencies(self.conf["freq"])
        except ValueError:
            raise InvalidConfigurationError(f"Unsupported frequency: {self.conf['freq']}")

    def _validate_backtest_length(self):
        """
        Validate that backtest_length contains at least one prediction_length.
        Mandatory check regardless of data_quality_check setting.
        """
        backtest_length = self.conf["backtest_length"]
        prediction_length = self.conf["prediction_length"]

        _logger.debug(f"Checking backtest length: {backtest_length} vs prediction length: {prediction_length}")

        if backtest_length < prediction_length:
            error_msg = (f"Backtest length ({backtest_length}) is shorter than "
                        f"prediction length ({prediction_length})")
            _logger.error(error_msg)
            raise ParameterValidationError(error_msg)

    def _validate_external_regressors(self):
        """Validate external regressors configuration (mandatory check)."""
        self.regressor_validator.validate_resampling_compatibility()

    def _get_backtest_interval_expr(self) -> Optional[str]:
        """Get a SQL INTERVAL expression for the backtest offset."""
        freq = self.conf["freq"]
        length = self.conf["backtest_length"]
        mapping = {
            "H": f"INTERVAL {length} HOURS",
            "D": f"INTERVAL {length} DAYS",
            "W": f"INTERVAL {length * 7} DAYS",
            "M": f"INTERVAL {length} MONTHS",
        }
        return mapping.get(freq)

    def _get_freq_interval_expr(self) -> Optional[str]:
        """Get a SQL INTERVAL expression for one step of the time series frequency."""
        return self._FREQ_INTERVAL_MAP.get(self.conf["freq"])

    # ------------------------------------------------------------------
    # Individual checks — each returns a Spark DF of failed group_ids
    # ------------------------------------------------------------------

    def _find_groups_with_regressor_nulls(self, df: SparkDataFrame) -> SparkDataFrame:
        """Return a single-column DataFrame of groups that have nulls in any external regressor."""
        group_id = self.conf["group_id"]
        external_regressors = self.regressor_validator.get_external_regressors()

        if not external_regressors:
            return df.select(group_id).limit(0)

        all_cols = []
        for features in external_regressors.values():
            all_cols.extend(features)

        if not all_cols:
            return df.select(group_id).limit(0)

        null_aggs = [
            F.sum(F.when(F.isnull(F.col(c)), 1).otherwise(0)).alias(f"_null_{c}")
            for c in all_cols
        ]

        stats = df.groupBy(group_id).agg(*null_aggs)

        any_null = functools.reduce(
            lambda a, b: a | b,
            [F.col(f"_null_{c}") > 0 for c in all_cols]
        )

        return stats.filter(any_null).select(group_id)

    def _find_groups_with_insufficient_training(self, df: SparkDataFrame) -> SparkDataFrame:
        """Return a single-column DataFrame of groups with insufficient training data."""
        group_id = self.conf["group_id"]
        date_col = self.conf["date_col"]
        target = self.conf["target"]

        required_points = (
            self.conf.get("train_predict_ratio", DEFAULT_MIN_TRAIN_PREDICT_RATIO)
            * self.conf["prediction_length"]
        )

        interval_expr = self._get_backtest_interval_expr()
        if interval_expr is None:
            return df.select(group_id).limit(0)

        positive_df = df.filter(F.col(target) > 0)

        group_max = (
            positive_df
            .groupBy(group_id)
            .agg(F.max(date_col).alias("_max_date"))
            .withColumn("_split_date", F.col("_max_date") - F.expr(interval_expr))
        )

        training_counts = (
            positive_df
            .join(group_max.select(group_id, "_split_date"), group_id)
            .filter(F.col(date_col) < F.col("_split_date"))
            .groupBy(group_id)
            .agg(F.count("*").alias("_training_count"))
        )

        all_groups = df.select(group_id).distinct()
        with_counts = (
            all_groups
            .join(training_counts, group_id, "left")
            .na.fill(0, subset=["_training_count"])
        )

        return with_counts.filter(F.col("_training_count") < required_points).select(group_id)

    def _find_groups_with_excessive_negatives(self, df: SparkDataFrame) -> SparkDataFrame:
        """Return a single-column DataFrame of groups exceeding the negative-value threshold."""
        group_id = self.conf["group_id"]
        target = self.conf["target"]
        threshold = self.thresholds.negative_data_threshold

        stats = (
            df.groupBy(group_id)
            .agg(
                F.count("*").alias("_total"),
                F.sum(F.when(F.col(target) < 0, 1).otherwise(0)).alias("_neg_count"),
            )
            .withColumn("_neg_ratio", F.col("_neg_count") / F.col("_total"))
        )

        return stats.filter(F.col("_neg_ratio") > threshold).select(group_id)

    def _handle_missing_entries(
        self, df: SparkDataFrame
    ) -> Tuple[SparkDataFrame, SparkDataFrame]:
        """
        Detect missing time-series entries and optionally resample.

        Returns:
            (clean_df, removed_groups_df) where removed_groups_df is a
            single-column DataFrame of group IDs that were dropped.
        """
        group_id = self.conf["group_id"]
        date_col = self.conf["date_col"]

        freq_interval = self._get_freq_interval_expr()
        if freq_interval is None:
            return df, df.select(group_id).limit(0)

        global_max = df.agg(F.max(date_col).alias("_gmax")).collect()[0]["_gmax"]
        if global_max is None:
            return df, df.select(group_id).limit(0)

        group_stats = df.groupBy(group_id).agg(
            F.min(date_col).alias("_min_date"),
            F.count("*").alias("_actual_count"),
        )

        date_spine = group_stats.select(
            group_id,
            F.explode(
                F.sequence(F.col("_min_date"), F.lit(global_max), F.expr(freq_interval))
            ).alias(date_col),
        )

        expected = date_spine.groupBy(group_id).agg(
            F.count("*").alias("_expected_count")
        )

        missing_stats = (
            group_stats
            .join(expected, group_id)
            .withColumn("_missing_count", F.col("_expected_count") - F.col("_actual_count"))
            .withColumn(
                "_missing_ratio",
                F.when(
                    F.col("_expected_count") > 0,
                    F.col("_missing_count") / F.col("_expected_count"),
                ).otherwise(0),
            )
        )

        if self.conf.get("resample"):
            threshold = self.thresholds.missing_data_threshold
            removed = (
                missing_stats
                .filter((F.col("_missing_count") > 0) & (F.col("_missing_ratio") > threshold))
                .select(group_id)
            )

            surviving_spine = date_spine.join(removed, group_id, "left_anti")
            clean_df = surviving_spine.join(df, [group_id, date_col], "left")

            # Fill gaps with 0 (matches original Pandas fillna(0) behaviour)
            clean_df = clean_df.na.fill(0)
            str_cols = [
                f.name for f in clean_df.schema.fields
                if isinstance(f.dataType, StringType) and f.name not in (group_id, date_col)
            ]
            if str_cols:
                clean_df = clean_df.na.fill("0", subset=str_cols)

            return clean_df, removed
        else:
            removed = (
                missing_stats
                .filter(F.col("_missing_count") > 0)
                .select(group_id)
            )
            clean_df = df.join(removed, group_id, "left_anti")
            return clean_df, removed

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(
        self, verbose: bool = False
    ) -> Tuple[SparkDataFrame, List[str]]:
        """
        Run comprehensive data quality checks on the dataset.

        Args:
            verbose: Whether to log progress and summary information

        Returns:
            Tuple of (clean_df, removed_groups):
                - clean_df: Spark DataFrame with time series that passed all checks
                - removed_groups: List of group IDs that were removed
        """
        if verbose:
            _logger.info("Starting data quality checks...")

        group_id = self.conf["group_id"]
        date_col = self.conf["date_col"]

        df = self.df.withColumn(date_col, F.to_timestamp(F.col(date_col)))

        # Mandatory config-level checks
        if verbose:
            _logger.info("Running mandatory configuration checks...")
        self._validate_external_regressors()
        self._validate_backtest_length()

        removed: List[str] = []

        if self.conf.get("data_quality_check", False):
            if verbose:
                _logger.info("Running optional data quality checks...")

            initial_count = df.select(group_id).distinct().count()

            self.metrics.total_groups = initial_count
            self.metrics.removed_groups = 0
            self.metrics.removal_reasons = {}

            removed_dfs: List[SparkDataFrame] = []

            # Check 1: external regressor nulls
            null_removed = self._find_groups_with_regressor_nulls(df)
            null_count = null_removed.count()
            if null_count > 0:
                self.metrics.removed_groups += null_count
                self.metrics.add_removal_reason(
                    "null values in external regressors", null_count
                )
                df = df.join(null_removed, group_id, "left_anti")
            removed_dfs.append(null_removed)

            # Check 2: insufficient training data
            short_removed = self._find_groups_with_insufficient_training(df)
            short_count = short_removed.count()
            if short_count > 0:
                self.metrics.removed_groups += short_count
                self.metrics.add_removal_reason("insufficient training data", short_count)
                df = df.join(short_removed, group_id, "left_anti")
            removed_dfs.append(short_removed)

            # Check 3: missing entries (+ optional resampling)
            df, missing_removed = self._handle_missing_entries(df)
            missing_count = missing_removed.count()
            if missing_count > 0:
                self.metrics.removed_groups += missing_count
                reason = (
                    f"missing data ratio exceeds threshold ({self.thresholds.missing_data_threshold})"
                    if self.conf.get("resample")
                    else "missing entries detected and resampling disabled"
                )
                self.metrics.add_removal_reason(reason, missing_count)
            removed_dfs.append(missing_removed)

            # Check 4: excessive negative values (runs on potentially resampled data)
            neg_removed = self._find_groups_with_excessive_negatives(df)
            neg_count = neg_removed.count()
            if neg_count > 0:
                self.metrics.removed_groups += neg_count
                self.metrics.add_removal_reason(
                    f"negative data ratio exceeds threshold ({self.thresholds.negative_data_threshold})",
                    neg_count,
                )
                df = df.join(neg_removed, group_id, "left_anti")
            removed_dfs.append(neg_removed)

            all_removed = functools.reduce(SparkDataFrame.union, removed_dfs).distinct()
            removed = sorted([row[0] for row in all_removed.collect()])

            if verbose:
                initial_groups = set(
                    row[0]
                    for row in self.df.select(group_id).distinct().collect()
                )
                final_groups = initial_groups - set(removed)
                self._log_quality_metrics(initial_groups, final_groups, removed)

        else:
            if verbose:
                _logger.info(
                    "Skipping detailed data quality checks (data_quality_check=False)"
                )

        clean_df = df

        if not clean_df.take(1):
            error_msg = "No time series passed the data quality checks"
            _logger.error(error_msg)
            raise EmptyDatasetError(error_msg)

        if verbose:
            final_count = clean_df.select(group_id).distinct().count()
            _logger.info(
                f"Data quality checks completed successfully. "
                f"Final dataset: {clean_df.count()} records across {final_count} groups"
            )

        return clean_df, removed

    # ------------------------------------------------------------------
    # Logging helpers
    # ------------------------------------------------------------------

    def _log_quality_metrics(
        self, initial_groups: Set, final_groups: Set, removed: List[str]
    ):
        """Log comprehensive data quality metrics."""
        initial_count = len(initial_groups)
        final_count = len(final_groups)
        removed_count = len(removed)

        if removed_count > 0:
            removal_rate = self.metrics.get_removal_rate()
            _logger.info("Data quality summary:")
            _logger.info(f"  - Initial groups: {initial_count}")
            _logger.info(f"  - Final groups: {final_count}")
            _logger.info(f"  - Removed groups: {removed_count} ({removal_rate:.1f}%)")

            if self.metrics.removal_reasons:
                _logger.info("  - Removal reasons:")
                for reason, count in self.metrics.removal_reasons.items():
                    percentage = (count / removed_count) * 100
                    _logger.info(f"    * {reason}: {count} ({percentage:.1f}%)")
        else:
            _logger.info("All groups passed data quality checks")

    def get_quality_metrics(self) -> DataQualityMetrics:
        """
        Get comprehensive data quality metrics.

        Returns:
            DataQualityMetrics object with processing statistics
        """
        return self.metrics
