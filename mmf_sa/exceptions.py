"""
Custom exception classes for the Many Model Forecasting (MMF) package.

This module provides specific exception types for different error scenarios
to improve error handling and debugging throughout the codebase.
"""


class MMFError(Exception):
    """Base exception class for all MMF-related errors."""
    pass


# Configuration-related exceptions
class ConfigurationError(MMFError):
    """Raised when there are issues with configuration setup or validation."""
    pass


class InvalidConfigurationError(ConfigurationError):
    """Raised when configuration contains invalid values or structure."""
    pass


# Model-related exceptions
class ModelError(MMFError):
    """Base exception for model-related errors."""
    pass


class ModelNotFoundError(ModelError):
    """Raised when a requested model is not found in the registry."""
    pass


class ModelInitializationError(ModelError):
    """Raised when a model fails to initialize properly."""
    pass


class ModelTrainingError(ModelError):
    """Raised when model training fails."""
    pass


class ModelPredictionError(ModelError):
    """Raised when model prediction fails."""
    pass


class UnsupportedModelError(ModelError):
    """Raised when trying to use an unsupported model."""
    pass


# Data-related exceptions
class DataError(MMFError):
    """Base exception for data-related errors."""
    pass


class DataQualityError(DataError):
    """Raised when data quality checks fail."""
    pass


class DataPreparationError(DataError):
    """Raised when data preparation fails."""
    pass


class MissingDataError(DataError):
    """Raised when required data is missing."""
    pass


class InvalidDataError(DataError):
    """Raised when data contains invalid values or structure."""
    pass


class EmptyDatasetError(DataError):
    """Raised when dataset is empty after processing."""
    pass


# Feature-related exceptions
class FeatureError(MMFError):
    """Base exception for feature-related errors."""
    pass


class MissingFeatureError(FeatureError):
    """Raised when required features are missing."""
    pass


class InvalidFeatureError(FeatureError):
    """Raised when features contain invalid values."""
    pass


class FeaturePreparationError(FeatureError):
    """Raised when feature preparation fails."""
    pass


# Validation-related exceptions
class ValidationError(MMFError):
    """Base exception for validation errors."""
    pass


class ParameterValidationError(ValidationError):
    """Raised when parameter validation fails."""
    pass


class MetricValidationError(ValidationError):
    """Raised when metric validation fails."""
    pass


class FrequencyValidationError(ValidationError):
    """Raised when frequency validation fails."""
    pass


# Forecasting-related exceptions
class ForecastingError(MMFError):
    """Base exception for forecasting-related errors."""
    pass


class BacktestError(ForecastingError):
    """Raised when backtesting fails."""
    pass


class EvaluationError(ForecastingError):
    """Raised when model evaluation fails."""
    pass


class ScoringError(ForecastingError):
    """Raised when model scoring fails."""
    pass


class UnsupportedMetricError(ForecastingError):
    """Raised when an unsupported metric is used."""
    pass


class UnsupportedFrequencyError(ForecastingError):
    """Raised when an unsupported frequency is used."""
    pass


# MLflow-related exceptions
class MLflowError(MMFError):
    """Base exception for MLflow-related errors."""
    pass


class ExperimentError(MLflowError):
    """Raised when MLflow experiment operations fail."""
    pass


class ModelRegistryError(MLflowError):
    """Raised when MLflow model registry operations fail."""
    pass


# Infrastructure-related exceptions
class InfrastructureError(MMFError):
    """Base exception for infrastructure-related errors."""
    pass


class SparkError(InfrastructureError):
    """Raised when Spark operations fail."""
    pass


class ResourceError(InfrastructureError):
    """Raised when resource allocation fails."""
    pass


class AcceleratorError(InfrastructureError):
    """Raised when accelerator (GPU/CPU) operations fail."""
    pass


# I/O related exceptions
class MMFIOError(MMFError):
    """Base exception for I/O operations."""
    pass


class MMFFileNotFoundError(MMFIOError):
    """Raised when a required file is not found."""
    pass


class WriteError(MMFIOError):
    """Raised when writing operations fail."""
    pass


class ReadError(MMFIOError):
    """Raised when reading operations fail."""
    pass 