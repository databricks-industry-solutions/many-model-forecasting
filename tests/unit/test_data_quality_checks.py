"""
Unit tests for the improved data_quality_checks.py module.
Tests cover all new functionality including ValidationResult, ExternalRegressorValidator,
DateOffsetUtility, and the enhanced DataQualityChecks class.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from omegaconf import OmegaConf
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType, TimestampType

from mmf_sa.data_quality_checks import (
    DataQualityChecks,
    DataQualityThresholds,
    DataQualityMetrics,
    ValidationResult,
    ExternalRegressorValidator,
    DateOffsetUtility,
    SupportedFrequencies,
    ExternalRegressorTypes,
    DEFAULT_MISSING_DATA_THRESHOLD,
    DEFAULT_NEGATIVE_DATA_THRESHOLD,
    DEFAULT_MIN_TRAIN_PREDICT_RATIO
)
from mmf_sa.exceptions import (
    ParameterValidationError,
    InvalidConfigurationError,
    EmptyDatasetError,
    DataQualityError
)

from .fixtures import spark_session


class TestValidationResult:
    """Test suite for ValidationResult class."""
    
    def test_success_without_data(self):
        """Test successful validation result without processed data."""
        result = ValidationResult.success()
        assert result.is_valid is True
        assert result.reason is None
        assert result.processed_data is None
    
    def test_success_with_data(self):
        """Test successful validation result with processed data."""
        data = pd.DataFrame({'col': [1, 2, 3]})
        result = ValidationResult.success(data)
        assert result.is_valid is True
        assert result.reason is None
        assert result.processed_data is not None
        assert len(result.processed_data) == 3
    
    def test_failure(self):
        """Test failed validation result."""
        reason = "Test failure reason"
        result = ValidationResult.failure(reason)
        assert result.is_valid is False
        assert result.reason == reason
        assert result.processed_data is None


class TestDataQualityThresholds:
    """Test suite for DataQualityThresholds class."""
    
    def test_default_values(self):
        """Test default threshold values."""
        thresholds = DataQualityThresholds()
        assert thresholds.missing_data_threshold == DEFAULT_MISSING_DATA_THRESHOLD
        assert thresholds.negative_data_threshold == DEFAULT_NEGATIVE_DATA_THRESHOLD
        assert thresholds.min_train_predict_ratio == DEFAULT_MIN_TRAIN_PREDICT_RATIO
    
    def test_custom_values(self):
        """Test custom threshold values."""
        thresholds = DataQualityThresholds(
            missing_data_threshold=0.1,
            negative_data_threshold=0.15,
            min_train_predict_ratio=2.0
        )
        assert thresholds.missing_data_threshold == 0.1
        assert thresholds.negative_data_threshold == 0.15
        assert thresholds.min_train_predict_ratio == 2.0
    
    def test_invalid_missing_data_threshold(self):
        """Test invalid missing data threshold values."""
        with pytest.raises(ParameterValidationError) as exc_info:
            DataQualityThresholds(missing_data_threshold=1.5)
        assert "missing_data_threshold must be between 0 and 1" in str(exc_info.value)
        assert "1.5" in str(exc_info.value)
    
    def test_invalid_negative_data_threshold(self):
        """Test invalid negative data threshold values."""
        with pytest.raises(ParameterValidationError) as exc_info:
            DataQualityThresholds(negative_data_threshold=-0.1)
        assert "negative_data_threshold must be between 0 and 1" in str(exc_info.value)
        assert "-0.1" in str(exc_info.value)
    
    def test_invalid_train_predict_ratio(self):
        """Test invalid train predict ratio values."""
        with pytest.raises(ParameterValidationError) as exc_info:
            DataQualityThresholds(min_train_predict_ratio=-1.0)
        assert "min_train_predict_ratio must be non-negative" in str(exc_info.value)
        assert "-1.0" in str(exc_info.value)


class TestDataQualityMetrics:
    """Test suite for DataQualityMetrics class."""
    
    def test_default_initialization(self):
        """Test default initialization of metrics."""
        metrics = DataQualityMetrics()
        assert metrics.total_groups == 0
        assert metrics.removed_groups == 0
        assert metrics.removal_reasons == {}
    
    def test_add_removal_reason(self):
        """Test adding removal reasons."""
        metrics = DataQualityMetrics()
        metrics.add_removal_reason("test_reason", 2)
        assert metrics.removal_reasons["test_reason"] == 2
        
        # Add more of the same reason
        metrics.add_removal_reason("test_reason", 3)
        assert metrics.removal_reasons["test_reason"] == 5
        
        # Add different reason
        metrics.add_removal_reason("another_reason", 1)
        assert metrics.removal_reasons["another_reason"] == 1
        assert len(metrics.removal_reasons) == 2
    
    def test_get_removal_rate(self):
        """Test removal rate calculation."""
        metrics = DataQualityMetrics()
        
        # Test with zero total groups
        assert metrics.get_removal_rate() == 0.0
        
        # Test with normal values
        metrics.total_groups = 10
        metrics.removed_groups = 2
        assert metrics.get_removal_rate() == 20.0
        
        # Test with all groups removed
        metrics.removed_groups = 10
        assert metrics.get_removal_rate() == 100.0


class TestSupportedFrequencies:
    """Test suite for SupportedFrequencies enum."""
    
    def test_enum_values(self):
        """Test enum values are correct."""
        assert SupportedFrequencies.HOURLY.value == "H"
        assert SupportedFrequencies.DAILY.value == "D"
        assert SupportedFrequencies.WEEKLY.value == "W"
        assert SupportedFrequencies.MONTHLY.value == "M"
    
    def test_enum_validation(self):
        """Test enum validation."""
        valid_values = [freq.value for freq in SupportedFrequencies]
        assert "H" in valid_values
        assert "D" in valid_values
        assert "W" in valid_values
        assert "M" in valid_values
        assert "INVALID" not in valid_values


class TestExternalRegressorTypes:
    """Test suite for ExternalRegressorTypes enum."""
    
    def test_enum_values(self):
        """Test enum values are correct."""
        assert ExternalRegressorTypes.STATIC_FEATURES.value == "static_features"
        assert ExternalRegressorTypes.DYNAMIC_FUTURE_NUMERICAL.value == "dynamic_future_numerical"
        assert ExternalRegressorTypes.DYNAMIC_FUTURE_CATEGORICAL.value == "dynamic_future_categorical"
        assert ExternalRegressorTypes.DYNAMIC_HISTORICAL_NUMERICAL.value == "dynamic_historical_numerical"
        assert ExternalRegressorTypes.DYNAMIC_HISTORICAL_CATEGORICAL.value == "dynamic_historical_categorical"


class TestDateOffsetUtility:
    """Test suite for DateOffsetUtility class."""
    
    def test_valid_frequencies(self):
        """Test valid frequency calculations."""
        # Test hourly
        offset = DateOffsetUtility.get_backtest_offset("H", 24)
        assert offset is not None
        assert offset == pd.DateOffset(hours=24)
        
        # Test daily
        offset = DateOffsetUtility.get_backtest_offset("D", 7)
        assert offset is not None
        assert offset == pd.DateOffset(days=7)
        
        # Test weekly
        offset = DateOffsetUtility.get_backtest_offset("W", 2)
        assert offset is not None
        assert offset == pd.DateOffset(weeks=2)
        
        # Test monthly
        offset = DateOffsetUtility.get_backtest_offset("M", 3)
        assert offset is not None
        assert offset == pd.DateOffset(months=3)
    
    def test_invalid_frequency(self):
        """Test invalid frequency handling."""
        offset = DateOffsetUtility.get_backtest_offset("INVALID", 1)
        assert offset is None
    
    def test_zero_length(self):
        """Test zero length handling."""
        offset = DateOffsetUtility.get_backtest_offset("D", 0)
        assert offset is not None
        assert offset == pd.DateOffset(days=0)


class TestExternalRegressorValidator:
    """Test suite for ExternalRegressorValidator class."""
    
    def test_no_external_regressors(self):
        """Test validator with no external regressors."""
        conf = OmegaConf.create({
            'group_id': 'group_id',
            'date_col': 'date_col',
            'target': 'target'
        })
        validator = ExternalRegressorValidator(conf)
        
        assert not validator.has_external_regressors()
        assert validator.get_external_regressors() == {}
    
    def test_with_external_regressors(self):
        """Test validator with external regressors."""
        conf = OmegaConf.create({
            'group_id': 'group_id',
            'date_col': 'date_col',
            'target': 'target',
            'static_features': ['feature1', 'feature2'],
            'dynamic_future_numerical': ['feature3']
        })
        validator = ExternalRegressorValidator(conf)
        
        assert validator.has_external_regressors()
        external_regressors = validator.get_external_regressors()
        assert 'static_features' in external_regressors
        assert 'dynamic_future_numerical' in external_regressors
        assert external_regressors['static_features'] == ['feature1', 'feature2']
        assert external_regressors['dynamic_future_numerical'] == ['feature3']
    
    def test_resampling_compatibility_valid(self):
        """Test valid resampling compatibility."""
        # No external regressors, resampling enabled - should pass
        conf = OmegaConf.create({
            'group_id': 'group_id',
            'resample': True
        })
        validator = ExternalRegressorValidator(conf)
        validator.validate_resampling_compatibility()  # Should not raise
        
        # External regressors, resampling disabled - should pass
        conf = OmegaConf.create({
            'group_id': 'group_id',
            'static_features': ['feature1'],
            'resample': False
        })
        validator = ExternalRegressorValidator(conf)
        validator.validate_resampling_compatibility()  # Should not raise
    
    def test_resampling_compatibility_invalid(self):
        """Test invalid resampling compatibility."""
        conf = OmegaConf.create({
            'group_id': 'group_id',
            'static_features': ['feature1'],
            'resample': True
        })
        validator = ExternalRegressorValidator(conf)
        
        with pytest.raises(InvalidConfigurationError) as exc_info:
            validator.validate_resampling_compatibility()
        assert "Resampling must be disabled when external regressors are provided" in str(exc_info.value)
    
    def test_check_nulls_in_regressors_valid(self):
        """Test null checks with valid data."""
        conf = OmegaConf.create({
            'static_features': ['feature1', 'feature2'],
            'dynamic_future_numerical': ['feature3']
        })
        validator = ExternalRegressorValidator(conf)
        
        # Create data without nulls
        df = pd.DataFrame({
            'feature1': [1, 2, 3],
            'feature2': ['a', 'b', 'c'],
            'feature3': [1.0, 2.0, 3.0]
        })
        
        result = validator.check_nulls_in_regressors(df, 'test_group')
        assert result.is_valid is True
        assert result.reason is None
    
    def test_check_nulls_in_regressors_invalid(self):
        """Test null checks with invalid data."""
        conf = OmegaConf.create({
            'static_features': ['feature1', 'feature2'],
            'dynamic_future_numerical': ['feature3']
        })
        validator = ExternalRegressorValidator(conf)
        
        # Create data with nulls
        df = pd.DataFrame({
            'feature1': [1, 2, np.nan],
            'feature2': ['a', 'b', 'c'],
            'feature3': [1.0, 2.0, 3.0]
        })
        
        result = validator.check_nulls_in_regressors(df, 'test_group')
        assert result.is_valid is False
        assert "null values in static features" in result.reason


class TestDataQualityChecks:
    """Test suite for DataQualityChecks class."""
    
    def create_sample_data(self):
        """Create sample data for testing."""
        return pd.DataFrame({
            'group_id': ['A', 'A', 'A', 'A', 'B', 'B', 'B', 'B'],
            'date_col': pd.date_range('2023-01-01', periods=8, freq='D'),
            'target': [10, 20, 30, 40, 15, 25, 35, 45],
            'feature1': [1, 2, 3, 4, 5, 6, 7, 8],
            'feature2': ['x', 'y', 'z', 'w', 'x', 'y', 'z', 'w']
        })
    
    def create_basic_config(self):
        """Create basic configuration for testing."""
        return OmegaConf.create({
            'group_id': 'group_id',
            'date_col': 'date_col',
            'target': 'target',
            'freq': 'D',
            'backtest_length': 2,
            'prediction_length': 1,
            'train_predict_ratio': 1.0,
            'data_quality_check': True,
            'resample': False
        })
    
    def test_initialization(self, spark_session):
        """Test DataQualityChecks initialization."""
        sample_data = self.create_sample_data()
        conf = self.create_basic_config()
        
        # Mock Spark DataFrame
        mock_df = Mock()
        mock_df.toPandas.return_value = sample_data
        
        checker = DataQualityChecks(mock_df, conf, spark_session)
        
        assert checker.conf == conf
        assert checker.spark == spark_session
        assert isinstance(checker.thresholds, DataQualityThresholds)
        assert isinstance(checker.metrics, DataQualityMetrics)
        assert isinstance(checker.regressor_validator, ExternalRegressorValidator)
        assert len(checker.df) == 8
    
    def test_configuration_validation_missing_params(self, spark_session):
        """Test configuration validation with missing parameters."""
        sample_data = self.create_sample_data()
        
        # Missing required parameters
        incomplete_conf = OmegaConf.create({
            'group_id': 'group_id',
            'date_col': 'date_col'
            # Missing target, freq, backtest_length, prediction_length
        })
        
        mock_df = Mock()
        mock_df.toPandas.return_value = sample_data
        
        with pytest.raises(InvalidConfigurationError) as exc_info:
            DataQualityChecks(mock_df, incomplete_conf, spark_session)
        assert "Missing required parameters" in str(exc_info.value)
    
    def test_configuration_validation_invalid_frequency(self, spark_session):
        """Test configuration validation with invalid frequency."""
        sample_data = self.create_sample_data()
        conf = self.create_basic_config()
        conf['freq'] = 'INVALID'
        
        mock_df = Mock()
        mock_df.toPandas.return_value = sample_data
        
        with pytest.raises(InvalidConfigurationError) as exc_info:
            DataQualityChecks(mock_df, conf, spark_session)
        assert "Unsupported frequency: INVALID" in str(exc_info.value)
    
    def test_backtest_length_validation_valid(self, spark_session):
        """Test valid backtest length validation."""
        sample_data = self.create_sample_data()
        conf = self.create_basic_config()
        conf['backtest_length'] = 5
        conf['prediction_length'] = 3
        
        mock_df = Mock()
        mock_df.toPandas.return_value = sample_data
        
        checker = DataQualityChecks(mock_df, conf, spark_session)
        checker._validate_backtest_length()  # Should not raise
    
    def test_backtest_length_validation_invalid(self, spark_session):
        """Test invalid backtest length validation."""
        sample_data = self.create_sample_data()
        conf = self.create_basic_config()
        conf['backtest_length'] = 1
        conf['prediction_length'] = 3
        
        mock_df = Mock()
        mock_df.toPandas.return_value = sample_data
        
        checker = DataQualityChecks(mock_df, conf, spark_session)
        
        with pytest.raises(ParameterValidationError) as exc_info:
            checker._validate_backtest_length()
        assert "Backtest length (1) is shorter than prediction length (3)" in str(exc_info.value)
    
    def test_training_period_length_check_valid(self, spark_session):
        """Test valid training period length check."""
        sample_data = self.create_sample_data()
        conf = self.create_basic_config()
        
        mock_df = Mock()
        mock_df.toPandas.return_value = sample_data
        
        checker = DataQualityChecks(mock_df, conf, spark_session)
        
        # Get data for one group
        group_data = sample_data[sample_data['group_id'] == 'A']
        
        result = checker._check_training_period_length(group_data, 'A')
        assert result.is_valid is True
    
    def test_training_period_length_check_invalid(self, spark_session):
        """Test invalid training period length check."""
        # Create data with insufficient training period
        sample_data = pd.DataFrame({
            'group_id': ['A', 'A'],
            'date_col': pd.date_range('2023-01-01', periods=2, freq='D'),
            'target': [10, 20]
        })
        
        conf = self.create_basic_config()
        conf['train_predict_ratio'] = 5.0  # High ratio to make it fail
        
        mock_df = Mock()
        mock_df.toPandas.return_value = sample_data
        
        checker = DataQualityChecks(mock_df, conf, spark_session)
        
        result = checker._check_training_period_length(sample_data, 'A')
        assert result.is_valid is False
        assert result.reason == "insufficient training data"
    
    def test_missing_entries_check_no_resampling(self, spark_session):
        """Test missing entries check without resampling."""
        sample_data = self.create_sample_data()
        conf = self.create_basic_config()
        conf['resample'] = False
        
        mock_df = Mock()
        mock_df.toPandas.return_value = sample_data
        
        checker = DataQualityChecks(mock_df, conf, spark_session)
        
        # Get data for one group
        group_data = sample_data[sample_data['group_id'] == 'A']
        
        result = checker._check_missing_entries(group_data, 'A', group_data['date_col'].max())
        assert result.is_valid is True
    
    def test_missing_entries_check_with_resampling(self, spark_session):
        """Test missing entries check with resampling enabled."""
        # Create data with missing dates but under the threshold
        sample_data = pd.DataFrame({
            'group_id': ['A', 'A', 'A', 'A'],
            'date_col': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-04', '2023-01-05']),  # Missing 01-03 only (20% missing)
            'target': [10, 20, 30, 40]
        })
        
        conf = self.create_basic_config()
        conf['resample'] = True
        
        mock_df = Mock()
        mock_df.toPandas.return_value = sample_data
        
        checker = DataQualityChecks(mock_df, conf, spark_session)
        
        result = checker._check_missing_entries(sample_data, 'A', sample_data['date_col'].max())
        assert result.is_valid is True
        assert result.processed_data is not None
        assert len(result.processed_data) > len(sample_data)  # Should have more rows after resampling
    
    def test_negative_entries_check_valid(self, spark_session):
        """Test negative entries check with valid data."""
        sample_data = self.create_sample_data()
        conf = self.create_basic_config()
        
        mock_df = Mock()
        mock_df.toPandas.return_value = sample_data
        
        checker = DataQualityChecks(mock_df, conf, spark_session)
        
        # Get data for one group
        group_data = sample_data[sample_data['group_id'] == 'A']
        
        result = checker._check_negative_entries(group_data, 'A')
        assert result.is_valid is True
    
    def test_negative_entries_check_invalid(self, spark_session):
        """Test negative entries check with invalid data."""
        # Create data with many negative values
        sample_data = pd.DataFrame({
            'group_id': ['A', 'A', 'A', 'A'],
            'date_col': pd.date_range('2023-01-01', periods=4, freq='D'),
            'target': [10, -20, -30, -40]  # 75% negative
        })
        
        conf = self.create_basic_config()
        
        mock_df = Mock()
        mock_df.toPandas.return_value = sample_data
        
        checker = DataQualityChecks(mock_df, conf, spark_session)
        
        result = checker._check_negative_entries(sample_data, 'A')
        assert result.is_valid is False
        assert result.reason == f"negative data ratio exceeds threshold ({DEFAULT_NEGATIVE_DATA_THRESHOLD})"
    
    def test_run_group_checks_all_pass(self, spark_session):
        """Test group checks when all checks pass."""
        sample_data = self.create_sample_data()
        conf = self.create_basic_config()
        
        mock_df = Mock()
        mock_df.toPandas.return_value = sample_data
        
        checker = DataQualityChecks(mock_df, conf, spark_session)
        
        # Get data for one group
        group_data = sample_data[sample_data['group_id'] == 'A']
        
        result_df = checker._run_group_checks(group_data, group_data['date_col'].max())
        assert not result_df.empty
        assert len(result_df) == len(group_data)
    
    def test_run_group_checks_some_fail(self, spark_session):
        """Test group checks when some checks fail."""
        # Create data that will fail negative entries check
        sample_data = pd.DataFrame({
            'group_id': ['A', 'A', 'A', 'A'],
            'date_col': pd.date_range('2023-01-01', periods=4, freq='D'),
            'target': [10, -20, -30, -40]  # 75% negative
        })
        
        conf = self.create_basic_config()
        
        mock_df = Mock()
        mock_df.toPandas.return_value = sample_data
        
        checker = DataQualityChecks(mock_df, conf, spark_session)
        
        result_df = checker._run_group_checks(sample_data, sample_data['date_col'].max())
        assert result_df.empty  # Should be empty due to failed checks
        assert checker.metrics.removed_groups == 1
    
    def test_run_with_data_quality_checks_disabled(self, spark_session):
        """Test run method with data quality checks disabled."""
        sample_data = self.create_sample_data()
        conf = self.create_basic_config()
        conf['data_quality_check'] = False
        
        mock_df = Mock()
        mock_df.toPandas.return_value = sample_data
        
        # Mock the createDataFrame method
        mock_result_df = Mock()
        spark_session.createDataFrame = Mock(return_value=mock_result_df)
        
        checker = DataQualityChecks(mock_df, conf, spark_session)
        
        clean_df, removed_groups = checker.run()
        
        assert clean_df == mock_result_df
        assert removed_groups == []
    
    def test_empty_dataset_error(self, spark_session):
        """Test empty dataset error handling."""
        # Create empty data with proper data types
        sample_data = pd.DataFrame({
            'group_id': pd.Series([], dtype='object'),
            'date_col': pd.Series([], dtype='datetime64[ns]'),
            'target': pd.Series([], dtype='float64')
        })
        
        conf = self.create_basic_config()
        
        mock_df = Mock()
        mock_df.toPandas.return_value = sample_data
        
        checker = DataQualityChecks(mock_df, conf, spark_session)
        
        with pytest.raises(EmptyDatasetError) as exc_info:
            checker.run()
        assert "No time series passed the data quality checks" in str(exc_info.value)
    
    def test_conversion_error_handling(self, spark_session):
        """Test DataFrame conversion error handling."""
        conf = self.create_basic_config()
        
        # Mock DataFrame that raises exception on toPandas
        mock_df = Mock()
        mock_df.toPandas.side_effect = Exception("Conversion failed")
        
        with pytest.raises(DataQualityError) as exc_info:
            DataQualityChecks(mock_df, conf, spark_session)
        assert "DataFrame conversion failed" in str(exc_info.value)
    
    def test_get_quality_metrics(self, spark_session):
        """Test getting quality metrics."""
        sample_data = self.create_sample_data()
        conf = self.create_basic_config()
        
        mock_df = Mock()
        mock_df.toPandas.return_value = sample_data
        
        checker = DataQualityChecks(mock_df, conf, spark_session)
        
        # Add some test metrics
        checker.metrics.total_groups = 10
        checker.metrics.removed_groups = 2
        checker.metrics.add_removal_reason("test_reason", 2)
        
        metrics = checker.get_quality_metrics()
        
        assert isinstance(metrics, DataQualityMetrics)
        assert metrics.total_groups == 10
        assert metrics.removed_groups == 2
        assert metrics.removal_reasons["test_reason"] == 2
        assert metrics.get_removal_rate() == 20.0
    
    def test_log_quality_metrics_aggregated_only(self, spark_session):
        """Test that logging shows only aggregated numbers."""
        sample_data = self.create_sample_data()
        conf = self.create_basic_config()
        
        mock_df = Mock()
        mock_df.toPandas.return_value = sample_data
        
        checker = DataQualityChecks(mock_df, conf, spark_session)
        
        # Set up metrics
        checker.metrics.total_groups = 10
        checker.metrics.removed_groups = 3
        checker.metrics.add_removal_reason("insufficient training data", 2)
        checker.metrics.add_removal_reason(f"negative data ratio exceeds threshold ({DEFAULT_NEGATIVE_DATA_THRESHOLD})", 1)
        
        initial_groups = set(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'])
        final_groups = set(['A', 'B', 'C', 'D', 'E', 'F', 'G'])
        removed_groups = ['H', 'I', 'J']
        
        # Test logging with removals - should only show aggregated numbers
        with patch('mmf_sa.data_quality_checks._logger') as mock_logger:
            checker._log_quality_metrics(initial_groups, final_groups, removed_groups)
            
            # Check that only aggregated metrics are logged
            mock_logger.info.assert_any_call("Data quality summary:")
            mock_logger.info.assert_any_call("  - Initial groups: 10")
            mock_logger.info.assert_any_call("  - Final groups: 7")
            mock_logger.info.assert_any_call("  - Removed groups: 3 (30.0%)")
            
            # Verify that removal reasons ARE logged but sample groups are NOT
            logged_calls = [call[0][0] for call in mock_logger.info.call_args_list]
            assert any("Removal reasons:" in call for call in logged_calls)
            assert any("insufficient training data" in call for call in logged_calls)
            assert any(f"negative data ratio exceeds threshold ({DEFAULT_NEGATIVE_DATA_THRESHOLD})" in call for call in logged_calls)
            assert not any("Sample removed groups:" in call for call in logged_calls)
        
        # Test logging with no removals
        with patch('mmf_sa.data_quality_checks._logger') as mock_logger:
            checker._log_quality_metrics(initial_groups, initial_groups, [])
            
            mock_logger.info.assert_called_with("All groups passed data quality checks")


class TestDataQualityChecksIntegration:
    """Integration tests for DataQualityChecks."""
    
    def test_full_pipeline_success(self, spark_session):
        """Test full pipeline with successful data quality checks."""
        # Create good quality data
        sample_data = pd.DataFrame({
            'group_id': ['A', 'A', 'A', 'A', 'A', 'A', 'B', 'B', 'B', 'B', 'B', 'B'],
            'date_col': pd.date_range('2023-01-01', periods=12, freq='D'),
            'target': [10, 20, 30, 40, 50, 60, 15, 25, 35, 45, 55, 65]
        })
        
        conf = OmegaConf.create({
            'group_id': 'group_id',
            'date_col': 'date_col',
            'target': 'target',
            'freq': 'D',
            'backtest_length': 3,
            'prediction_length': 2,
            'train_predict_ratio': 1.0,
            'data_quality_check': True,
            'resample': False
        })
        
        mock_df = Mock()
        mock_df.toPandas.return_value = sample_data
        
        # Mock the createDataFrame method
        mock_result_df = Mock()
        spark_session.createDataFrame = Mock(return_value=mock_result_df)
        
        checker = DataQualityChecks(mock_df, conf, spark_session)
        
        clean_df, removed_groups = checker.run()
        
        assert clean_df == mock_result_df
        assert removed_groups == []
        
        metrics = checker.get_quality_metrics()
        assert metrics.total_groups == 2
        assert metrics.removed_groups == 0
        assert metrics.get_removal_rate() == 0.0
    
    def test_full_pipeline_with_removals(self, spark_session):
        """Test full pipeline with some groups removed."""
        # Create mixed quality data - Group A good, Group B mostly negative, Group C insufficient data
        sample_data = pd.DataFrame({
            'group_id': ['A', 'A', 'A', 'A', 'A', 'A', 'B', 'B', 'B', 'B', 'C', 'C'],
            'date_col': pd.date_range('2023-01-01', periods=12, freq='D'),
            'target': [10, 20, 30, 40, 50, 60, -15, -25, -35, -45, 1, 2]  # Group A good, Group B all negative, Group C insufficient
        })
        
        conf = OmegaConf.create({
            'group_id': 'group_id',
            'date_col': 'date_col',
            'target': 'target',
            'freq': 'D',
            'backtest_length': 2,
            'prediction_length': 1,
            'train_predict_ratio': 1.0,
            'data_quality_check': True,
            'resample': False
        })
        
        mock_df = Mock()
        mock_df.toPandas.return_value = sample_data
        
        # Mock the createDataFrame method to return a filtered DataFrame
        mock_result_df = Mock()
        spark_session.createDataFrame = Mock(return_value=mock_result_df)
        
        checker = DataQualityChecks(mock_df, conf, spark_session)
        
        clean_df, removed_groups = checker.run()
        
        assert clean_df == mock_result_df
        assert len(removed_groups) > 0  # Some groups should be removed
        
        metrics = checker.get_quality_metrics()
        assert metrics.total_groups == 3
        assert metrics.removed_groups > 0
        assert metrics.get_removal_rate() > 0.0 