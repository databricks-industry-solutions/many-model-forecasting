import functools
import logging
from typing import Dict, Any, Union, List, Tuple, Optional, Set
from dataclasses import dataclass
from enum import Enum
from omegaconf import DictConfig
import pandas as pd
import pyspark
from pyspark.sql import SparkSession
from mmf_sa.exceptions import (
    DataQualityError,
    ParameterValidationError,
    EmptyDatasetError,
    InvalidConfigurationError
)

# Configure logging
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
        Check if external regressors contain null values.
        
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
    Class to run comprehensive data quality checks on time series data.
    
    This class performs various data quality validations including:
    - Parameter validation (backtest length, external regressors)
    - Null value checks for external regressors
    - Training period length validation
    - Missing data detection and handling
    - Negative value detection
    """
    
    def __init__(
        self,
        df: pyspark.sql.DataFrame,
        conf: DictConfig,
        spark: SparkSession = None,
        thresholds: Optional[DataQualityThresholds] = None,
    ):
        self.df = self._convert_to_pandas(df)
        self.conf = conf
        self.spark = spark
        self.thresholds = thresholds or DataQualityThresholds()
        self.metrics = DataQualityMetrics()
        self.regressor_validator = ExternalRegressorValidator(conf)
        
        # Validate configuration
        self._validate_configuration()
        
        #_logger.info(f"Initialized DataQualityChecks with {len(self.df)} total records")
    
    def _convert_to_pandas(self, df: pyspark.sql.DataFrame) -> pd.DataFrame:
        """Convert Spark DataFrame to pandas with error handling."""
        try:
            return df.toPandas()
        except Exception as e:
            _logger.error(f"Failed to convert Spark DataFrame to pandas: {e}")
            raise DataQualityError(f"DataFrame conversion failed: {e}")
    
    def _validate_configuration(self):
        """Validate essential configuration parameters."""
        required_params = ["group_id", "date_col", "target", "freq", "backtest_length", "prediction_length"]
        
        missing_params = [param for param in required_params if param not in self.conf]
        if missing_params:
            raise InvalidConfigurationError(f"Missing required parameters: {missing_params}")
        
        # Validate frequency
        try:
            SupportedFrequencies(self.conf["freq"])
        except ValueError:
            raise InvalidConfigurationError(f"Unsupported frequency: {self.conf['freq']}")
        
        #_logger.info("Configuration validation passed")

    def _validate_backtest_length(self):
        """
        Validate that backtest_length contains at least one prediction_length.
        This is a mandatory check regardless of data_quality_check setting.
        
        Raises:
            ParameterValidationError: If backtest length is shorter than prediction length
        """
        backtest_length = self.conf["backtest_length"]
        prediction_length = self.conf["prediction_length"]
        
        _logger.debug(f"Checking backtest length: {backtest_length} vs prediction length: {prediction_length}")
        
        if backtest_length < prediction_length:
            error_msg = (f"Backtest length ({backtest_length}) is shorter than "
                        f"prediction length ({prediction_length})")
            _logger.error(error_msg)
            raise ParameterValidationError(error_msg)
        
        #_logger.info("Backtest length validation passed")

    def _validate_external_regressors(self):
        """
        Validate external regressors configuration.
        This is a mandatory check regardless of data_quality_check setting.
        """
        self.regressor_validator.validate_resampling_compatibility()

    def _check_training_period_length(self, df: pd.DataFrame, group_id: str) -> ValidationResult:
        """
        Check if training period meets minimum length requirements.
        
        Args:
            df: DataFrame for a single group
            group_id: Group identifier for logging
            
        Returns:
            ValidationResult with check outcome
        """
        # Filter to positive values for training period calculation
        temp_df = df[df[self.conf["target"]] > 0]
        
        # Calculate backtest offset using utility
        backtest_offset = DateOffsetUtility.get_backtest_offset(
            self.conf["freq"], self.conf["backtest_length"]
        )
        
        if backtest_offset is None:
            _logger.warning(f"Could not calculate backtest offset for group {group_id}")
            return ValidationResult.success()
        
        split_date = temp_df[self.conf["date_col"]].max() - backtest_offset
        training_points = temp_df[temp_df[self.conf["date_col"]] < split_date].count().iloc[0]
        required_points = self.conf.get("train_predict_ratio", DEFAULT_MIN_TRAIN_PREDICT_RATIO) * self.conf["prediction_length"]
        
        if training_points < required_points:
            reason = "insufficient training data"
            _logger.debug(f"Group {group_id}: {reason} ({training_points} < {required_points})")
            return ValidationResult.failure(reason)
        
        return ValidationResult.success()

    def _check_missing_entries(self, df: pd.DataFrame, group_id: str, max_date: pd.Timestamp) -> ValidationResult:
        """
        Check and handle missing entries in the time series.
        
        Args:
            df: DataFrame for a single group
            group_id: Group identifier for logging
            max_date: Maximum date for resampling (if None, uses group's own max date)
            
        Returns:
            ValidationResult with check outcome and processed data
        """
        # Use group's own max date for detecting missing entries within the group
        if max_date is None:
            max_date = df[self.conf["date_col"]].max()
    
        # Create complete date range and resample
        df_indexed = df.set_index(self.conf["date_col"])
        date_idx = pd.date_range(
            start=df[self.conf["date_col"]].min(),
            end=max_date,
            freq=self.conf["freq"],
            name=self.conf["date_col"],
        )
        df_resampled = (
            df_indexed.reindex(date_idx)
            .reset_index()
            .fillna(value=0)
        )
        
        # Check if there are missing entries
        if len(df_resampled) > len(df):
            missing_ratio = (len(df_resampled) - len(df)) / len(df_resampled)
            
            if self.conf.get("resample"):
                if missing_ratio > self.thresholds.missing_data_threshold:
                    reason = f"missing data ratio exceeds threshold ({self.thresholds.missing_data_threshold})"
                    _logger.debug(f"Group {group_id}: missing data ratio ({missing_ratio:.3f}) exceeds threshold ({self.thresholds.missing_data_threshold})")
                    return ValidationResult.failure(reason)
                else:
                    _logger.debug(f"Group {group_id}: resampled {len(df_resampled) - len(df)} missing entries")
                    return ValidationResult.success(df_resampled)
            else:
                reason = "missing entries detected and resampling disabled"
                _logger.debug(f"Group {group_id}: {reason}")
                return ValidationResult.failure(reason)
        
        return ValidationResult.success(df_resampled)

    def _check_negative_entries(self, df: pd.DataFrame, group_id: str) -> ValidationResult:
        """
        Check if the time series has excessive negative entries.
        
        Args:
            df: DataFrame for a single group
            group_id: Group identifier for logging
            
        Returns:
            ValidationResult with check outcome
        """
        positive_df = df[df[self.conf["target"]] >= 0]
        negative_ratio = (len(df) - len(positive_df)) / len(df)
        
        if negative_ratio > self.thresholds.negative_data_threshold:
            reason = f"negative data ratio exceeds threshold ({self.thresholds.negative_data_threshold})"
            _logger.debug(f"Group {group_id}: negative data ratio ({negative_ratio:.3f}) exceeds threshold ({self.thresholds.negative_data_threshold})")
            return ValidationResult.failure(reason)
        
        return ValidationResult.success()

    def _run_group_checks(self, df: pd.DataFrame, max_date: pd.Timestamp) -> pd.DataFrame:
        """
        Run all data quality checks for a single group.
        
        Args:
            df: DataFrame for a single group
            max_date: Maximum date for processing
            
        Returns:
            DataFrame (empty if group fails checks, processed if passes)
        """
        if df.empty:
            return pd.DataFrame()
        
        group_id = df[self.conf["group_id"]].iloc[0]
            
        # Track metrics for this group
        self.metrics.total_groups += 1
        
        # Define check pipeline
        checks = [
            ("external_regressor_nulls", lambda: self.regressor_validator.check_nulls_in_regressors(df, group_id)),
            ("training_period_length", lambda: self._check_training_period_length(df, group_id)),
            ("missing_entries", lambda: self._check_missing_entries(df, group_id, max_date)),
        ]
        
        processed_df = df
        
        # Run checks in sequence
        for check_name, check_func in checks:
            try:
                result = check_func()
                if not result.is_valid:
                    self.metrics.removed_groups += 1
                    self.metrics.add_removal_reason(result.reason)
                    return pd.DataFrame()
                if result.processed_data is not None:
                    processed_df = result.processed_data
            except Exception as e:
                _logger.error(f"Error in {check_name} check for group {group_id}: {e}")
                self.metrics.removed_groups += 1
                self.metrics.add_removal_reason(f"error in {check_name} check")
                return pd.DataFrame()
        
        # Check negative entries on final processed data
        negative_result = self._check_negative_entries(processed_df, group_id)
        if not negative_result.is_valid:
            self.metrics.removed_groups += 1
            self.metrics.add_removal_reason(negative_result.reason)
            return pd.DataFrame()
        
        return processed_df

    def run(self, verbose: bool = False) -> Tuple[Union[pd.DataFrame, pyspark.sql.DataFrame], List[str]]:
        """
        Run comprehensive data quality checks on the dataset.
        
        Args:
            verbose: Whether to log progress and summary information (default: True)
        
        Returns:
            Tuple of (clean_df, removed_groups):
                - clean_df: DataFrame with time series that passed all checks
                - removed_groups: List of group IDs that were removed
        """
        if verbose:
            _logger.info("Starting data quality checks...")
        
        # Prepare data
        try:
            self.df[self.conf["date_col"]] = pd.to_datetime(self.df[self.conf["date_col"]])
            self.df.sort_values(by=self.conf["date_col"], inplace=True)
        except Exception as e:
            _logger.error(f"Failed to prepare data: {e}")
            raise DataQualityError(f"Data preparation failed: {e}")
        
        initial_groups = set(self.df[self.conf["group_id"]].unique())
        if verbose:
            _logger.info(f"Initial dataset: {len(self.df)} records across {len(initial_groups)} groups")
        
        # Run mandatory checks
        if verbose:
            _logger.info("Running mandatory configuration checks...")
        self._validate_external_regressors()
        self._validate_backtest_length()
        
        # Initialize removed groups list
        removed = []
        
        # Run optional detailed checks if enabled
        if self.conf.get("data_quality_check", False):
            if verbose:
                _logger.info("Running optional data quality checks...")
            
            # Reset metrics for group processing
            self.metrics.total_groups = 0
            self.metrics.removed_groups = 0
            self.metrics.removal_reasons = {}
            
            # Create partial function for group checks
            max_date = self.df[self.conf["date_col"]].max()

            group_check_func = functools.partial(
                self._run_group_checks,
                max_date=max_date,
            )
            
            # Apply checks to each group
            clean_df = self.df.groupby(self.conf["group_id"]).apply(group_check_func)
            
            # Handle groupby results and clean up DataFrame structure
            clean_df = self._process_groupby_results(clean_df)
            
            # Calculate removed groups
            final_groups = set(clean_df[self.conf["group_id"]].unique()) if not clean_df.empty else set()
            removed = sorted(list(initial_groups - final_groups))
            
            # Log comprehensive metrics
            if verbose:
                self._log_quality_metrics(initial_groups, final_groups, removed)
            
        else:
            if verbose:
                _logger.info("Skipping detailed data quality checks (data_quality_check=False)")
            clean_df = self.df
        
        # Final validation
        if clean_df.empty:
            error_msg = "No time series passed the data quality checks"
            _logger.error(error_msg)
            raise EmptyDatasetError(error_msg)
        
        if verbose:
            group_col = self.conf['group_id']
            _logger.info(f"Data quality checks completed successfully. "
                        f"Final dataset: {len(clean_df)} records across {len(clean_df[group_col].unique())} groups")
        
        # Convert back to Spark DataFrame
        try:
            clean_df = self.spark.createDataFrame(clean_df)
        except Exception as e:
            _logger.error(f"Failed to convert back to Spark DataFrame: {e}")
            raise DataQualityError(f"Spark DataFrame conversion failed: {e}")
        
        return clean_df, removed
    
    def _process_groupby_results(self, clean_df: pd.DataFrame) -> pd.DataFrame:
        """Process the results from groupby operations."""
        # Handle empty DataFrame case
        if clean_df.empty:
            return clean_df
            
        # Handle groupby results and clean up DataFrame structure
        if isinstance(clean_df.index, pd.MultiIndex):
            clean_df = clean_df.drop(
                columns=[self.conf["group_id"]], errors="ignore"
            ).reset_index()
            clean_df = clean_df[
                clean_df.columns.drop(list(clean_df.filter(regex="level")))
            ]
        else:
            clean_df = clean_df.reset_index()
            clean_df = clean_df[
                clean_df.columns.drop(list(clean_df.filter(regex="index")))
            ]
        return clean_df
    
    def _format_removal_reason(self, reason: str) -> str:
        """
        Format removal reasons for consistent logging.
        
        Args:
            reason: Removal reason string
            
        Returns:
            Formatted reason string
        """
        # Since individual check methods now return formatted strings,
        # we mostly just return the reason as-is, but handle special cases
        
        # Check if reason starts with "null values in" (external regressor check)
        if reason.startswith("null values in"):
            return reason
        
        # Check if reason starts with "error in" (exception handling)
        if reason.startswith("error in"):
            return reason
            
        # Return the reason as-is (already formatted by individual check methods)
        return reason
    
    def _log_quality_metrics(self, initial_groups: Set, final_groups: Set, removed: List[str]):
        """
        Log comprehensive data quality metrics.
        
        Args:
            initial_groups: Set of initial group IDs
            final_groups: Set of final group IDs
            removed: List of removed group IDs
        """
        initial_count = len(initial_groups)
        final_count = len(final_groups)
        removed_count = len(removed)
        
        if removed_count > 0:
            removal_rate = self.metrics.get_removal_rate()
            _logger.info(f"Data quality summary:")
            _logger.info(f"  - Initial groups: {initial_count}")
            _logger.info(f"  - Final groups: {final_count}")
            _logger.info(f"  - Removed groups: {removed_count} ({removal_rate:.1f}%)")
            
            # Log removal reasons
            if self.metrics.removal_reasons:
                _logger.info("  - Removal reasons:")
                for reason, count in self.metrics.removal_reasons.items():
                    percentage = (count / removed_count) * 100
                    formatted_reason = self._format_removal_reason(reason)
                    _logger.info(f"    * {formatted_reason}: {count} ({percentage:.1f}%)")
        else:
            _logger.info("All groups passed data quality checks")
    
    def get_quality_metrics(self) -> DataQualityMetrics:
        """
        Get comprehensive data quality metrics.
        
        Returns:
            DataQualityMetrics object with processing statistics
        """
        return self.metrics
