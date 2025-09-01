"""
Common pytest configuration and fixtures for data quality tests.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock
from omegaconf import OmegaConf
from pyspark.sql import SparkSession
import tempfile
import os

# Import existing fixtures
from .unit.fixtures import spark_session


@pytest.fixture
def sample_time_series_data():
    """Create sample time series data for testing."""
    return pd.DataFrame({
        'group_id': ['A', 'A', 'A', 'A', 'A', 'B', 'B', 'B', 'B', 'B'],
        'date_col': pd.date_range('2023-01-01', periods=10, freq='D'),
        'target': [10, 20, 30, 40, 50, 15, 25, 35, 45, 55],
        'feature1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'feature2': ['x', 'y', 'z', 'w', 'v', 'x', 'y', 'z', 'w', 'v']
    })


@pytest.fixture
def basic_forecasting_config():
    """Create basic forecasting configuration for testing."""
    return OmegaConf.create({
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


@pytest.fixture
def config_with_external_regressors():
    """Create configuration with external regressors for testing."""
    return OmegaConf.create({
        'group_id': 'group_id',
        'date_col': 'date_col',
        'target': 'target',
        'freq': 'D',
        'backtest_length': 3,
        'prediction_length': 2,
        'train_predict_ratio': 1.0,
        'data_quality_check': True,
        'resample': False,
        'static_features': ['feature1'],
        'dynamic_future_numerical': ['feature2']
    })


@pytest.fixture
def mock_spark_dataframe():
    """Create a mock Spark DataFrame for testing."""
    mock_df = Mock()
    mock_df.toPandas.return_value = pd.DataFrame({
        'group_id': ['A', 'A', 'B', 'B'],
        'date_col': pd.date_range('2023-01-01', periods=4, freq='D'),
        'target': [10, 20, 30, 40]
    })
    return mock_df


@pytest.fixture
def problematic_data():
    """Create problematic data for testing failure scenarios."""
    return pd.DataFrame({
        'group_id': ['A', 'A', 'A', 'A', 'B', 'B', 'C', 'C'],
        'date_col': pd.date_range('2023-01-01', periods=8, freq='D'),
        'target': [10, 20, 30, 40, -15, -25, 1, 2],  # Group B has negative values, Group C has insufficient data
        'feature1': [1, 2, 3, 4, 5, 6, np.nan, 8],  # Group C has null values
        'feature2': ['x', 'y', 'z', 'w', 'v', 'u', 't', 's']
    })


# Configure pytest to suppress specific warnings
def pytest_configure(config):
    """Configure pytest settings."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )


# Configure logging for tests
def pytest_runtest_setup(item):
    """Setup for each test."""
    import logging
    logging.getLogger("py4j").setLevel(logging.ERROR)
    logging.getLogger("pyspark").setLevel(logging.ERROR)


# Add custom markers for test categorization
pytest_plugins = [] 