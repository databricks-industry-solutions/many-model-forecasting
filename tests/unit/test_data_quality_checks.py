"""
Unit tests for the data_quality_checks.py module.
Tests cover all functionality including ValidationResult, ExternalRegressorValidator,
DateOffsetUtility, and the Spark-native DataQualityChecks class.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch
from omegaconf import OmegaConf
from pyspark.sql import SparkSession
from pyspark.sql.types import (
    StructType, StructField, StringType, DoubleType, TimestampType
)

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


# ---------------------------------------------------------------------------
# Helper-class tests (unchanged — these are independent of Spark)
# ---------------------------------------------------------------------------

class TestValidationResult:
    """Test suite for ValidationResult class."""

    def test_success_without_data(self):
        result = ValidationResult.success()
        assert result.is_valid is True
        assert result.reason is None
        assert result.processed_data is None

    def test_success_with_data(self):
        data = pd.DataFrame({'col': [1, 2, 3]})
        result = ValidationResult.success(data)
        assert result.is_valid is True
        assert result.reason is None
        assert result.processed_data is not None
        assert len(result.processed_data) == 3

    def test_failure(self):
        reason = "Test failure reason"
        result = ValidationResult.failure(reason)
        assert result.is_valid is False
        assert result.reason == reason
        assert result.processed_data is None


class TestDataQualityThresholds:
    """Test suite for DataQualityThresholds class."""

    def test_default_values(self):
        thresholds = DataQualityThresholds()
        assert thresholds.missing_data_threshold == DEFAULT_MISSING_DATA_THRESHOLD
        assert thresholds.negative_data_threshold == DEFAULT_NEGATIVE_DATA_THRESHOLD
        assert thresholds.min_train_predict_ratio == DEFAULT_MIN_TRAIN_PREDICT_RATIO

    def test_custom_values(self):
        thresholds = DataQualityThresholds(
            missing_data_threshold=0.1,
            negative_data_threshold=0.15,
            min_train_predict_ratio=2.0
        )
        assert thresholds.missing_data_threshold == 0.1
        assert thresholds.negative_data_threshold == 0.15
        assert thresholds.min_train_predict_ratio == 2.0

    def test_invalid_missing_data_threshold(self):
        with pytest.raises(ParameterValidationError) as exc_info:
            DataQualityThresholds(missing_data_threshold=1.5)
        assert "missing_data_threshold must be between 0 and 1" in str(exc_info.value)
        assert "1.5" in str(exc_info.value)

    def test_invalid_negative_data_threshold(self):
        with pytest.raises(ParameterValidationError) as exc_info:
            DataQualityThresholds(negative_data_threshold=-0.1)
        assert "negative_data_threshold must be between 0 and 1" in str(exc_info.value)
        assert "-0.1" in str(exc_info.value)

    def test_invalid_train_predict_ratio(self):
        with pytest.raises(ParameterValidationError) as exc_info:
            DataQualityThresholds(min_train_predict_ratio=-1.0)
        assert "min_train_predict_ratio must be non-negative" in str(exc_info.value)
        assert "-1.0" in str(exc_info.value)


class TestDataQualityMetrics:
    """Test suite for DataQualityMetrics class."""

    def test_default_initialization(self):
        metrics = DataQualityMetrics()
        assert metrics.total_groups == 0
        assert metrics.removed_groups == 0
        assert metrics.removal_reasons == {}

    def test_add_removal_reason(self):
        metrics = DataQualityMetrics()
        metrics.add_removal_reason("test_reason", 2)
        assert metrics.removal_reasons["test_reason"] == 2

        metrics.add_removal_reason("test_reason", 3)
        assert metrics.removal_reasons["test_reason"] == 5

        metrics.add_removal_reason("another_reason", 1)
        assert metrics.removal_reasons["another_reason"] == 1
        assert len(metrics.removal_reasons) == 2

    def test_get_removal_rate(self):
        metrics = DataQualityMetrics()
        assert metrics.get_removal_rate() == 0.0

        metrics.total_groups = 10
        metrics.removed_groups = 2
        assert metrics.get_removal_rate() == 20.0

        metrics.removed_groups = 10
        assert metrics.get_removal_rate() == 100.0


class TestSupportedFrequencies:
    """Test suite for SupportedFrequencies enum."""

    def test_enum_values(self):
        assert SupportedFrequencies.HOURLY.value == "H"
        assert SupportedFrequencies.DAILY.value == "D"
        assert SupportedFrequencies.WEEKLY.value == "W"
        assert SupportedFrequencies.MONTHLY.value == "M"

    def test_enum_validation(self):
        valid_values = [freq.value for freq in SupportedFrequencies]
        assert "H" in valid_values
        assert "D" in valid_values
        assert "W" in valid_values
        assert "M" in valid_values
        assert "INVALID" not in valid_values


class TestExternalRegressorTypes:
    """Test suite for ExternalRegressorTypes enum."""

    def test_enum_values(self):
        assert ExternalRegressorTypes.STATIC_FEATURES.value == "static_features"
        assert ExternalRegressorTypes.DYNAMIC_FUTURE_NUMERICAL.value == "dynamic_future_numerical"
        assert ExternalRegressorTypes.DYNAMIC_FUTURE_CATEGORICAL.value == "dynamic_future_categorical"
        assert ExternalRegressorTypes.DYNAMIC_HISTORICAL_NUMERICAL.value == "dynamic_historical_numerical"
        assert ExternalRegressorTypes.DYNAMIC_HISTORICAL_CATEGORICAL.value == "dynamic_historical_categorical"


class TestDateOffsetUtility:
    """Test suite for DateOffsetUtility class."""

    def test_valid_frequencies(self):
        offset = DateOffsetUtility.get_backtest_offset("H", 24)
        assert offset is not None
        assert offset == pd.DateOffset(hours=24)

        offset = DateOffsetUtility.get_backtest_offset("D", 7)
        assert offset is not None
        assert offset == pd.DateOffset(days=7)

        offset = DateOffsetUtility.get_backtest_offset("W", 2)
        assert offset is not None
        assert offset == pd.DateOffset(weeks=2)

        offset = DateOffsetUtility.get_backtest_offset("M", 3)
        assert offset is not None
        assert offset == pd.DateOffset(months=3)

    def test_invalid_frequency(self):
        offset = DateOffsetUtility.get_backtest_offset("INVALID", 1)
        assert offset is None

    def test_zero_length(self):
        offset = DateOffsetUtility.get_backtest_offset("D", 0)
        assert offset is not None
        assert offset == pd.DateOffset(days=0)


class TestExternalRegressorValidator:
    """Test suite for ExternalRegressorValidator class."""

    def test_no_external_regressors(self):
        conf = OmegaConf.create({
            'group_id': 'group_id',
            'date_col': 'date_col',
            'target': 'target'
        })
        validator = ExternalRegressorValidator(conf)
        assert not validator.has_external_regressors()
        assert validator.get_external_regressors() == {}

    def test_with_external_regressors(self):
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
        conf = OmegaConf.create({
            'group_id': 'group_id',
            'resample': True
        })
        validator = ExternalRegressorValidator(conf)
        validator.validate_resampling_compatibility()

        conf = OmegaConf.create({
            'group_id': 'group_id',
            'static_features': ['feature1'],
            'resample': False
        })
        validator = ExternalRegressorValidator(conf)
        validator.validate_resampling_compatibility()

    def test_resampling_compatibility_invalid(self):
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
        conf = OmegaConf.create({
            'static_features': ['feature1', 'feature2'],
            'dynamic_future_numerical': ['feature3']
        })
        validator = ExternalRegressorValidator(conf)
        df = pd.DataFrame({
            'feature1': [1, 2, 3],
            'feature2': ['a', 'b', 'c'],
            'feature3': [1.0, 2.0, 3.0]
        })
        result = validator.check_nulls_in_regressors(df, 'test_group')
        assert result.is_valid is True
        assert result.reason is None

    def test_check_nulls_in_regressors_invalid(self):
        conf = OmegaConf.create({
            'static_features': ['feature1', 'feature2'],
            'dynamic_future_numerical': ['feature3']
        })
        validator = ExternalRegressorValidator(conf)
        df = pd.DataFrame({
            'feature1': [1, 2, np.nan],
            'feature2': ['a', 'b', 'c'],
            'feature3': [1.0, 2.0, 3.0]
        })
        result = validator.check_nulls_in_regressors(df, 'test_group')
        assert result.is_valid is False
        assert "null values in static features" in result.reason


# ---------------------------------------------------------------------------
# Spark-native DataQualityChecks tests
# ---------------------------------------------------------------------------

def _make_spark_df(spark, pdf):
    """Convert a Pandas DataFrame to a Spark DataFrame, handling timestamps."""
    return spark.createDataFrame(pdf)


def _basic_conf(**overrides):
    """Create a basic OmegaConf config for tests with optional overrides."""
    base = {
        'group_id': 'group_id',
        'date_col': 'date_col',
        'target': 'target',
        'freq': 'D',
        'backtest_length': 2,
        'prediction_length': 1,
        'train_predict_ratio': 1.0,
        'data_quality_check': True,
        'resample': False,
    }
    base.update(overrides)
    return OmegaConf.create(base)


class TestDataQualityChecks:
    """Test suite for the Spark-native DataQualityChecks class."""

    def _sample_pdf(self):
        return pd.DataFrame({
            'group_id': ['A', 'A', 'A', 'A', 'B', 'B', 'B', 'B'],
            'date_col': pd.date_range('2023-01-01', periods=4, freq='D').tolist() * 2,
            'target': [10.0, 20.0, 30.0, 40.0, 15.0, 25.0, 35.0, 45.0],
            'feature1': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        })

    # -- initialisation & config validation --------------------------------

    def test_initialization(self, spark_session):
        sdf = _make_spark_df(spark_session, self._sample_pdf())
        conf = _basic_conf()
        checker = DataQualityChecks(sdf, conf, spark_session)

        assert checker.conf == conf
        assert checker.spark == spark_session
        assert isinstance(checker.thresholds, DataQualityThresholds)
        assert isinstance(checker.metrics, DataQualityMetrics)
        assert isinstance(checker.regressor_validator, ExternalRegressorValidator)

    def test_configuration_validation_missing_params(self, spark_session):
        sdf = _make_spark_df(spark_session, self._sample_pdf())
        incomplete_conf = OmegaConf.create({
            'group_id': 'group_id',
            'date_col': 'date_col',
        })
        with pytest.raises(InvalidConfigurationError) as exc_info:
            DataQualityChecks(sdf, incomplete_conf, spark_session)
        assert "Missing required parameters" in str(exc_info.value)

    def test_configuration_validation_invalid_frequency(self, spark_session):
        sdf = _make_spark_df(spark_session, self._sample_pdf())
        conf = _basic_conf(freq='INVALID')
        with pytest.raises(InvalidConfigurationError) as exc_info:
            DataQualityChecks(sdf, conf, spark_session)
        assert "Unsupported frequency: INVALID" in str(exc_info.value)

    # -- backtest length validation ----------------------------------------

    def test_backtest_length_validation_valid(self, spark_session):
        sdf = _make_spark_df(spark_session, self._sample_pdf())
        conf = _basic_conf(backtest_length=5, prediction_length=3)
        checker = DataQualityChecks(sdf, conf, spark_session)
        checker._validate_backtest_length()

    def test_backtest_length_validation_invalid(self, spark_session):
        sdf = _make_spark_df(spark_session, self._sample_pdf())
        conf = _basic_conf(backtest_length=1, prediction_length=3)
        checker = DataQualityChecks(sdf, conf, spark_session)
        with pytest.raises(ParameterValidationError) as exc_info:
            checker._validate_backtest_length()
        assert "Backtest length (1) is shorter than prediction length (3)" in str(exc_info.value)

    # -- regressor null check ----------------------------------------------

    def test_regressor_nulls_none_configured(self, spark_session):
        sdf = _make_spark_df(spark_session, self._sample_pdf())
        conf = _basic_conf()
        checker = DataQualityChecks(sdf, conf, spark_session)
        removed = checker._find_groups_with_regressor_nulls(sdf)
        assert removed.count() == 0

    def test_regressor_nulls_detected(self, spark_session):
        pdf = pd.DataFrame({
            'group_id': ['A', 'A', 'B', 'B'],
            'date_col': pd.date_range('2023-01-01', periods=2, freq='D').tolist() * 2,
            'target': [10.0, 20.0, 30.0, 40.0],
            'feat': [1.0, None, 3.0, 4.0],
        })
        sdf = _make_spark_df(spark_session, pdf)
        conf = _basic_conf(static_features=['feat'])
        checker = DataQualityChecks(sdf, conf, spark_session)

        removed = checker._find_groups_with_regressor_nulls(sdf)
        removed_ids = {row[0] for row in removed.collect()}
        assert removed_ids == {'A'}

    # -- training period length check --------------------------------------

    def test_training_period_sufficient(self, spark_session):
        pdf = pd.DataFrame({
            'group_id': ['A'] * 10 + ['B'] * 10,
            'date_col': (
                pd.date_range('2023-01-01', periods=10, freq='D').tolist()
                + pd.date_range('2023-01-01', periods=10, freq='D').tolist()
            ),
            'target': [float(i) for i in range(1, 11)] * 2,
        })
        sdf = _make_spark_df(spark_session, pdf)
        conf = _basic_conf(backtest_length=2, prediction_length=1, train_predict_ratio=1.0)
        checker = DataQualityChecks(sdf, conf, spark_session)

        removed = checker._find_groups_with_insufficient_training(sdf)
        assert removed.count() == 0

    def test_training_period_insufficient(self, spark_session):
        pdf = pd.DataFrame({
            'group_id': ['A', 'A'],
            'date_col': pd.date_range('2023-01-01', periods=2, freq='D'),
            'target': [10.0, 20.0],
        })
        sdf = _make_spark_df(spark_session, pdf)
        conf = _basic_conf(train_predict_ratio=5.0)
        checker = DataQualityChecks(sdf, conf, spark_session)

        removed = checker._find_groups_with_insufficient_training(sdf)
        removed_ids = {row[0] for row in removed.collect()}
        assert 'A' in removed_ids

    # -- negative entries check --------------------------------------------

    def test_negative_entries_below_threshold(self, spark_session):
        pdf = pd.DataFrame({
            'group_id': ['A'] * 4,
            'date_col': pd.date_range('2023-01-01', periods=4, freq='D'),
            'target': [10.0, 20.0, 30.0, 40.0],
        })
        sdf = _make_spark_df(spark_session, pdf)
        conf = _basic_conf()
        checker = DataQualityChecks(sdf, conf, spark_session)

        removed = checker._find_groups_with_excessive_negatives(sdf)
        assert removed.count() == 0

    def test_negative_entries_above_threshold(self, spark_session):
        pdf = pd.DataFrame({
            'group_id': ['A'] * 4,
            'date_col': pd.date_range('2023-01-01', periods=4, freq='D'),
            'target': [10.0, -20.0, -30.0, -40.0],
        })
        sdf = _make_spark_df(spark_session, pdf)
        conf = _basic_conf()
        checker = DataQualityChecks(sdf, conf, spark_session)

        removed = checker._find_groups_with_excessive_negatives(sdf)
        removed_ids = {row[0] for row in removed.collect()}
        assert removed_ids == {'A'}

    # -- missing entries check ---------------------------------------------

    def test_missing_entries_no_gaps(self, spark_session):
        pdf = pd.DataFrame({
            'group_id': ['A'] * 4,
            'date_col': pd.date_range('2023-01-01', periods=4, freq='D'),
            'target': [10.0, 20.0, 30.0, 40.0],
        })
        sdf = _make_spark_df(spark_session, pdf)
        conf = _basic_conf(resample=False)
        checker = DataQualityChecks(sdf, conf, spark_session)

        clean_df, removed = checker._handle_missing_entries(sdf)
        assert removed.count() == 0
        assert clean_df.count() == 4

    def test_missing_entries_gaps_resample_false(self, spark_session):
        """Groups with gaps are removed when resample=False."""
        pdf = pd.DataFrame({
            'group_id': ['A', 'A', 'A'],
            'date_col': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-05']),
            'target': [10.0, 20.0, 30.0],
        })
        sdf = _make_spark_df(spark_session, pdf)
        conf = _basic_conf(resample=False)
        checker = DataQualityChecks(sdf, conf, spark_session)

        clean_df, removed = checker._handle_missing_entries(sdf)
        removed_ids = {row[0] for row in removed.collect()}
        assert 'A' in removed_ids

    def test_missing_entries_gaps_resample_true_within_threshold(self, spark_session):
        """Gaps are filled when resample=True and within threshold."""
        pdf = pd.DataFrame({
            'group_id': ['A', 'A', 'A', 'A'],
            'date_col': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-04', '2023-01-05']),
            'target': [10.0, 20.0, 30.0, 40.0],
        })
        sdf = _make_spark_df(spark_session, pdf)
        conf = _basic_conf(resample=True)
        checker = DataQualityChecks(sdf, conf, spark_session)

        clean_df, removed = checker._handle_missing_entries(sdf)
        assert removed.count() == 0
        assert clean_df.count() == 5  # 4 original + 1 gap-filled

    def test_missing_entries_gaps_resample_true_exceeds_threshold(self, spark_session):
        """Groups removed when resample=True but missing ratio exceeds threshold."""
        pdf = pd.DataFrame({
            'group_id': ['A', 'A'],
            'date_col': pd.to_datetime(['2023-01-01', '2023-01-10']),
            'target': [10.0, 20.0],
        })
        sdf = _make_spark_df(spark_session, pdf)
        conf = _basic_conf(resample=True)
        thresholds = DataQualityThresholds(missing_data_threshold=0.2)
        checker = DataQualityChecks(sdf, conf, spark_session, thresholds=thresholds)

        clean_df, removed = checker._handle_missing_entries(sdf)
        removed_ids = {row[0] for row in removed.collect()}
        assert 'A' in removed_ids

    # -- full run() --------------------------------------------------------

    def test_run_data_quality_checks_disabled(self, spark_session):
        pdf = self._sample_pdf()
        sdf = _make_spark_df(spark_session, pdf)
        conf = _basic_conf(data_quality_check=False)
        checker = DataQualityChecks(sdf, conf, spark_session)

        clean_df, removed_groups = checker.run()
        assert removed_groups == []
        assert clean_df.count() == len(pdf)

    def test_run_all_groups_pass(self, spark_session):
        pdf = pd.DataFrame({
            'group_id': ['A'] * 6 + ['B'] * 6,
            'date_col': (
                pd.date_range('2023-01-01', periods=6, freq='D').tolist()
                + pd.date_range('2023-01-01', periods=6, freq='D').tolist()
            ),
            'target': [10.0, 20.0, 30.0, 40.0, 50.0, 60.0] * 2,
        })
        sdf = _make_spark_df(spark_session, pdf)
        conf = _basic_conf(backtest_length=3, prediction_length=2)
        checker = DataQualityChecks(sdf, conf, spark_session)

        clean_df, removed_groups = checker.run()
        assert removed_groups == []
        assert clean_df.count() == 12

        metrics = checker.get_quality_metrics()
        assert metrics.total_groups == 2
        assert metrics.removed_groups == 0
        assert metrics.get_removal_rate() == 0.0

    def test_run_removes_negative_group(self, spark_session):
        pdf = pd.DataFrame({
            'group_id': ['A'] * 6 + ['B'] * 4,
            'date_col': (
                pd.date_range('2023-01-01', periods=6, freq='D').tolist()
                + pd.date_range('2023-01-01', periods=4, freq='D').tolist()
            ),
            'target': [10.0, 20.0, 30.0, 40.0, 50.0, 60.0,
                        10.0, -20.0, -30.0, -40.0],
        })
        sdf = _make_spark_df(spark_session, pdf)
        conf = _basic_conf(backtest_length=2, prediction_length=1)
        checker = DataQualityChecks(sdf, conf, spark_session)

        clean_df, removed_groups = checker.run()
        assert 'B' in removed_groups
        remaining = {row['group_id'] for row in clean_df.select('group_id').distinct().collect()}
        assert remaining == {'A'}

    def test_empty_dataset_error(self, spark_session):
        schema = StructType([
            StructField('group_id', StringType()),
            StructField('date_col', TimestampType()),
            StructField('target', DoubleType()),
        ])
        sdf = spark_session.createDataFrame([], schema)
        conf = _basic_conf()
        checker = DataQualityChecks(sdf, conf, spark_session)

        with pytest.raises(EmptyDatasetError) as exc_info:
            checker.run()
        assert "No time series passed the data quality checks" in str(exc_info.value)

    def test_get_quality_metrics(self, spark_session):
        sdf = _make_spark_df(spark_session, self._sample_pdf())
        conf = _basic_conf()
        checker = DataQualityChecks(sdf, conf, spark_session)

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
        sdf = _make_spark_df(spark_session, self._sample_pdf())
        conf = _basic_conf()
        checker = DataQualityChecks(sdf, conf, spark_session)

        checker.metrics.total_groups = 10
        checker.metrics.removed_groups = 3
        checker.metrics.add_removal_reason("insufficient training data", 2)
        checker.metrics.add_removal_reason(
            f"negative data ratio exceeds threshold ({DEFAULT_NEGATIVE_DATA_THRESHOLD})", 1
        )

        initial_groups = set(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'])
        final_groups = set(['A', 'B', 'C', 'D', 'E', 'F', 'G'])
        removed_groups = ['H', 'I', 'J']

        with patch('mmf_sa.data_quality_checks._logger') as mock_logger:
            checker._log_quality_metrics(initial_groups, final_groups, removed_groups)

            mock_logger.info.assert_any_call("Data quality summary:")
            mock_logger.info.assert_any_call("  - Initial groups: 10")
            mock_logger.info.assert_any_call("  - Final groups: 7")
            mock_logger.info.assert_any_call("  - Removed groups: 3 (30.0%)")

            logged_calls = [call[0][0] for call in mock_logger.info.call_args_list]
            assert any("Removal reasons:" in call for call in logged_calls)
            assert any("insufficient training data" in call for call in logged_calls)
            assert any(
                f"negative data ratio exceeds threshold ({DEFAULT_NEGATIVE_DATA_THRESHOLD})" in call
                for call in logged_calls
            )
            assert not any("Sample removed groups:" in call for call in logged_calls)

        with patch('mmf_sa.data_quality_checks._logger') as mock_logger:
            checker._log_quality_metrics(initial_groups, initial_groups, [])
            mock_logger.info.assert_called_with("All groups passed data quality checks")


class TestDataQualityChecksIntegration:
    """Integration tests for DataQualityChecks."""

    def test_full_pipeline_success(self, spark_session):
        pdf = pd.DataFrame({
            'group_id': ['A'] * 6 + ['B'] * 6,
            'date_col': (
                pd.date_range('2023-01-01', periods=6, freq='D').tolist()
                + pd.date_range('2023-01-01', periods=6, freq='D').tolist()
            ),
            'target': [10.0, 20.0, 30.0, 40.0, 50.0, 60.0,
                        15.0, 25.0, 35.0, 45.0, 55.0, 65.0],
        })
        sdf = _make_spark_df(spark_session, pdf)
        conf = OmegaConf.create({
            'group_id': 'group_id',
            'date_col': 'date_col',
            'target': 'target',
            'freq': 'D',
            'backtest_length': 3,
            'prediction_length': 2,
            'train_predict_ratio': 1.0,
            'data_quality_check': True,
            'resample': False,
        })

        checker = DataQualityChecks(sdf, conf, spark_session)
        clean_df, removed_groups = checker.run()

        assert removed_groups == []
        assert clean_df.count() == 12

        metrics = checker.get_quality_metrics()
        assert metrics.total_groups == 2
        assert metrics.removed_groups == 0
        assert metrics.get_removal_rate() == 0.0

    def test_full_pipeline_with_removals(self, spark_session):
        """Group A is good, Group B is mostly negative, Group C has insufficient data."""
        pdf = pd.DataFrame({
            'group_id': (
                ['A'] * 6 + ['B'] * 4 + ['C'] * 2
            ),
            'date_col': (
                pd.date_range('2023-01-01', periods=6, freq='D').tolist()
                + pd.date_range('2023-01-01', periods=4, freq='D').tolist()
                + pd.date_range('2023-01-01', periods=2, freq='D').tolist()
            ),
            'target': (
                [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]
                + [-15.0, -25.0, -35.0, -45.0]
                + [1.0, 2.0]
            ),
        })
        sdf = _make_spark_df(spark_session, pdf)
        conf = OmegaConf.create({
            'group_id': 'group_id',
            'date_col': 'date_col',
            'target': 'target',
            'freq': 'D',
            'backtest_length': 2,
            'prediction_length': 1,
            'train_predict_ratio': 1.0,
            'data_quality_check': True,
            'resample': False,
        })

        checker = DataQualityChecks(sdf, conf, spark_session)
        clean_df, removed_groups = checker.run()

        assert len(removed_groups) > 0
        remaining = {
            row['group_id']
            for row in clean_df.select('group_id').distinct().collect()
        }
        assert 'A' in remaining

        metrics = checker.get_quality_metrics()
        assert metrics.total_groups == 3
        assert metrics.removed_groups > 0
        assert metrics.get_removal_rate() > 0.0

    def test_full_pipeline_with_resampling(self, spark_session):
        """Verify gap-filling produces the expected number of rows."""
        pdf = pd.DataFrame({
            'group_id': ['A'] * 4 + ['B'] * 5,
            'date_col': (
                pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-04', '2023-01-05']).tolist()
                + pd.date_range('2023-01-01', periods=5, freq='D').tolist()
            ),
            'target': [10.0, 20.0, 30.0, 40.0, 1.0, 2.0, 3.0, 4.0, 5.0],
        })
        sdf = _make_spark_df(spark_session, pdf)
        conf = OmegaConf.create({
            'group_id': 'group_id',
            'date_col': 'date_col',
            'target': 'target',
            'freq': 'D',
            'backtest_length': 2,
            'prediction_length': 1,
            'train_predict_ratio': 1.0,
            'data_quality_check': True,
            'resample': True,
        })

        checker = DataQualityChecks(sdf, conf, spark_session)
        clean_df, removed_groups = checker.run()

        assert removed_groups == []
        # A: 5 days (01-01 to 01-05 with 01-03 filled), B: 5 days
        assert clean_df.count() == 10
