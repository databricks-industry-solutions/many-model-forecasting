# Many-Model Forecasting Architecture

## Overview

The **MMF** (Many-Model Forecasting) is a comprehensive time series forecasting framework that enables evaluation and deployment of multiple forecasting models across many time series. The framework is designed to handle various types of models (local, global, and foundation models) and provides unified APIs for training, evaluation, and scoring.

## Entry Point: `run_forecast` Function

The primary entry point to the MMF is the `run_forecast` function located in `mmf_sa/__init__.py`. This function serves as the main orchestrator that:

1. **Configuration Management**: Handles configuration merging from multiple sources
2. **Forecaster Instantiation**: Creates and configures the `Forecaster` object
3. **Execution Control**: Triggers the evaluation and scoring pipeline

### Function Signature and Parameters

```python
def run_forecast(
    spark: SparkSession,
    train_data: Union[str, pd.DataFrame, DataFrame],
    scoring_data: Union[str, pd.DataFrame, DataFrame] = None,
    evaluation_output: str = None,
    scoring_output: str = None,
    group_id: str,
    date_col: str,
    target: str,
    freq: str,
    prediction_length: int,
    backtest_length: int,
    stride: int,
    metric: str = "smape",
    use_case_name: str = None,
    static_features: List[str] = None,
    dynamic_future_numerical: List[str] = None,
    dynamic_future_categorical: List[str] = None,
    dynamic_historical_numerical: List[str] = None,
    dynamic_historical_categorical: List[str] = None,
    active_models: List[str] = None,
    accelerator: str = "cpu",
    num_nodes: int = 1,
    train_predict_ratio: int = None,
    data_quality_check: bool = False,
    resample: bool = False,
    experiment_path: str = None,
    run_id: str = None,
    conf: Union[str, Dict[str, Any], OmegaConf] = None,
) -> str:
```

### Configuration Hierarchy

The system uses a hierarchical configuration approach. The latter steps override the former steps:

```
Base Configuration (e.g., forecasting_conf_daily.yaml)
    ↓
User Configuration (conf parameter)
    ↓
Function Parameters (direct arguments)
```

## Core Architecture: The Forecaster Class

The `Forecaster` class (`mmf_sa/Forecaster.py`) is the central orchestrator that manages the entire forecasting pipeline.

## Main Execution Flow: `evaluate_score` Method

The `evaluate_score` method is the main method of the `Forecaster` class. It runs data quality checks, performs backtesting (evaluation) for all models in the `active_models` list, and executes forecasting (scoring) for all models in the `active_models` list.

```python
def evaluate_score(self, evaluate: bool = True, score: bool = True) -> str:

    # 1. Data Quality Checks
    print("Run quality checks")
    src_df = self.resolve_source("train_data")
    clean_df, removed = DataQualityChecks(src_df, self.conf, self.spark).run(verbose=True)
    
    # 2. Evaluation
    if evaluate:
        self.evaluate_models()
    
    # 3. Scoring
    if score:
        self.score_models()
    
    return self.run_id
```

## Data Quality Management

### DataQualityChecks Class Architecture

The `DataQualityChecks` class (`mmf_sa/data_quality_checks.py`) provides comprehensive data validation. Checks consistes of mandatory and optional ones. See `mmf_sa/data_quality_checks.py` for details.

#### Key Features:
1. **External Regressor Validation**: Validates that resampling is disabled when external regressors are provided (mandatory)
2. **Training Period Validation**: Ensures that backtest_length contains at least one prediction_length (mandatory)
3. **Missing Data Detection**: Identifies and handles missing values (optional)
4. **Negative Value Validation**: Detects negative target values (optional)


## Model Management System

### Model Registry Architecture

The `ModelRegistry` class manages the lifecycle of forecasting models:

#### Model Configuration Structure:
```yaml
models:
  ModelName:
    module: path.to.model.module
    model_class: ModelClassName
    framework: StatsForecast|NeuralForecast|SKTime|Chronos|Moirai(disabled)|TimesFM
    model_type: local|global|foundation
    model_spec:
      parameter1: value1
      parameter2: value2
```

#### Model Type Classifications:

1. **Local Models**: 
   - Trained independently for each time series
   - Examples: StatsForecastAutoArima, StatsForecastAutoETS, SKTimeProphet
   - Processed using Pandas UDF for parallel execution

2. **Global Models**:
   - Trained on all time series simultaneously
   - Examples: NeuralForecastAutoNBEATSx, NeuralForecastAutoTiDE
   - Centralized training with shared parameters
   - Supports distributed training on single-node multi-GPU and multi-node multi-GPU clusters via `TorchDistributor`

3. **Foundation Models**:
   - Pre-trained models adapted for specific tasks
   - Examples: ChronosBoltBase, TimesFM_2_5_200m
   - Zero-shot learning capabilities

### Abstract Model Interface

All models inherit from `ForecastingRegressor` in `mmf_sa/models/abstract_model.py` which provides:

```python
class ForecastingRegressor(BaseEstimator, RegressorMixin):
    @abstractmethod
    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame
    
    @abstractmethod
    def fit(self, x, y=None)
    
    @abstractmethod
    def predict(self, x, y=None)
    
    @abstractmethod
    def forecast(self, x, spark=None)
    
    def backtest(self, df: pd.DataFrame, start: pd.Timestamp, ...) -> pd.DataFrame
    
    def calculate_metrics(self, hist_df: pd.DataFrame, val_df: pd.DataFrame, ...) -> Dict
```

## Evaluation Pipeline

### Model Evaluation Workflow

The `evaluate_models` method orchestrates model evaluation across different model types:

```python
def evaluate_models(self):
    for model_name in self.model_registry.get_active_model_keys():
        model_conf = self.model_registry.get_model_conf(model_name)
        if model_conf["model_type"] == "local":
            self.evaluate_local_model(model_conf)
        elif model_conf["model_type"] == "global":
            self.evaluate_global_model(model_conf)
        elif model_conf["model_type"] == "foundation":
            self.evaluate_foundation_model(model_conf)
```

### Local Model Evaluation

Local models are evaluated using Spark's `applyInPandas` for distributed processing:

1. **Data Preparation**: Resolves data sources and applies quality checks
2. **Parallel Processing**: Applies `evaluate_one_local_model` and performs backtesting for each group
3. **Results Aggregation**: Combines results from all time series and writes to the `evaluation_output` table
4. **MLflow Logging**: Records metrics per model aggregated over all time series

#### Backtest Process for Local Models:
```
Historical Data → Split by Group (Parallelization) → Split by Backtesting Start Date → Fit Model → Generate Forecasts → Calculate Metrics → Store Results
```

### Global Model Evaluation

Global models follow a two-phase training approach:

1. **Phase 1 - Final Model Training**:
   - Trains on complete dataset (train + validation)
   - Creates model artifacts
   - Registers model in Unity Catalog
   
2. **Phase 2 - Evaluation Training**:
   - Trains only on training data: i.e., all data excluding the `backtest_length`
   - Performs detailed backtesting using the same model
   - Calculates performance metrics and writes the results to the `evaluation_output` table

#### Multi-GPU Distributed Training

When `accelerator="gpu"` is set and the cluster has multiple GPUs (either on a single node or across multiple nodes), global models automatically use Spark's `TorchDistributor` with PyTorch DDP for distributed training. This is controlled by the `num_nodes` parameter:

- **Single-node multi-GPU** (`num_nodes=1`, default): All GPUs on the driver node participate in DDP training. Training data is partitioned to local storage (`/tmp/`).
- **Multi-node multi-GPU** (`num_nodes > 1`): GPUs across worker nodes participate in DDP training. Training data is partitioned to shared storage (`/dbfs/tmp/`) via the DBFS FUSE mount so all nodes can access it. Set `num_nodes` to the number of **worker** nodes. Autoscaling must be disabled to prevent workers from being removed mid-training.

If only a single GPU is available on a single-node cluster, training falls back to standard single-GPU mode without DDP.

### Foundation Model Evaluation

Foundation models leverage pre-trained weights:

1. **Model Loading**: Loads pre-trained foundation model
2. **Adaptation**: Applies model to specific time series format
3. **Evaluation**: Performs inference-based evaluation and writes the results to the `evaluation_output` table
4. **Registration**: Registers adapted model for deployment

## Backtesting Framework

### Backtesting Algorithm

The `backtest` method in `ForecastingRegressor` implements extending window validation:

```python
def backtest(self, df: pd.DataFrame, start: pd.Timestamp, ...) -> pd.DataFrame:
    # Initialize sliding window
    curr_date = start + self.one_ts_offset
    
    # Walk forward through time
    while curr_date + self.prediction_length_offset <= end_date:
        # Create training window
        train_df = df[df[date_col] < curr_date]
        
        # Create validation window
        val_df = df[(df[date_col] >= curr_date) & 
                   (df[date_col] < curr_date + prediction_length_offset)]
        
        # Calculate metrics
        metrics = self.calculate_metrics(train_df, val_df, curr_date)
        
        # Move to next window
        curr_date += stride_offset
```

### Metric Calculation

The system supports multiple out-of-the-box evaluation metrics:
- **SMAPE**: Symmetric Mean Absolute Percentage Error
- **MAPE**: Mean Absolute Percentage Error
- **MAE**: Mean Absolute Error
- **MSE**: Mean Squared Error
- **RMSE**: Root Mean Squared Error

## Scoring Pipeline

### Scoring Workflow

The scoring pipeline generates forecasts:

1. **Model Retrieval**: Loads trained models from MLflow registry (only for global and foundation models)
2. **Data Preparation**: Prepares scoring data with quality checks
3. **Prediction Generation**: Generates forecasts using trained models
4. **Result Storage**: Saves predictions to specified output (`scoring_output`) tables

### Scoring Methods by Model Type

#### Local Model Scoring:
- Uses Pandas UDF for parallel processing
- Generates predictions per time series group
- Saves serialized model artifacts in a Delta table for reproducibility and auditability

#### Global Model Scoring:
- Loads centralized model from MLflow
- Applies model to entire scoring dataset
- Generates predictions for all time series and writes to `scoring_output` tables

#### Foundation Model Scoring:
- Loads pre-trained foundation model
- Applies zero-shot inference
- Generates predictions without additional training and writes to `scoring_output` tables

## Data Flow Architecture

### Data Resolution System

The `resolve_source` method provides flexible data source handling:

```python
def resolve_source(self, key: str) -> DataFrame:
    # Check if data is in data_conf (direct data objects)
    if self.data_conf and key in self.data_conf:
        data = self.data_conf[key]
        if isinstance(data, pd.DataFrame):
            return self.spark.createDataFrame(data)
        elif isinstance(data, DataFrame):
            return data
    
    # Otherwise, read from table name in configuration
    return self.spark.read.table(self.conf[key])
```

### Data Preparation Pipeline

1. **Source Resolution**: Determines data source type (table, DataFrame, etc.)
2. **Quality Checks**: Applies comprehensive data validation
3. **Format Standardization**: Converts to required format for models
4. **Feature Engineering**: Processes external regressors and features

## Integration Points

### MLflow Integration

The system integrates extensively with MLflow for:

1. **Experiment Tracking**: Logs all runs with parameters and metrics
2. **Model Registration**: Stores trained models in Unity Catalog
3. **Version Control**: Tracks model versions and lineage

### Spark Integration

Spark is used for:

1. **Data Processing**: Handles large-scale data operations
2. **Distributed Computing**: Parallel model training and evaluation
3. **Table Operations**: Reads from and writes to Delta tables

## Configuration Management

### Configuration Files

1. **`forecasting_conf_<frequency>.yaml`**: Base configuration with default settings
2. **`models/models_conf.yaml`**: Model definitions and specifications
3. **User Configuration**: Custom overrides and specific settings

## Error Handling and Logging

### Exception Hierarchy

The system defines custom exceptions for different error scenarios:

- `ConfigurationError`: Configuration validation issues
- `ModelError`: Model-specific errors
- `DataError`: Data quality and processing issues
- `EvaluationError`: Evaluation pipeline errors
- `ScoringError`: Scoring pipeline errors

### Advanced Configuration

```python
# Custom configuration
config = {
    "models": {
        "StatsForecastAutoArima": {
            "model_spec": {
                "season_length": 12,
                "approximation": False
            }
        }
    }
}

run_id = run_forecast(
    spark=spark,
    train_data=train_df,  # Pandas DataFrame
    scoring_data=score_df,
    scoring_output="output_table",
    group_id="series_id",
    date_col="timestamp",
    target="value",
    freq="M",
    prediction_length=6,
    backtest_length=12,
    stride=1,
    data_quality_check=True,
    resample=True,
    conf=config
)
```

## Extensibility

### Adding New Models

1. **Create Model Class**: Inherit from `ForecastingRegressor`
2. **Implement Required Methods**: `fit`, `predict`, `forecast`, `prepare_data`
3. **Add Configuration**: Update `models_conf.yaml`
4. **Activate**: Add to active models list

### Custom Metrics

1. **Implement Metric Function**: Create custom metric calculation
2. **Update Abstract Model**: Add metric to `calculate_metrics`
3. **Configuration Update**: Add metric to supported metrics list


## Best Practices

### Data Management
- Implement data quality checks prior to using MMF
- Address missing values and anomalies prior to using MMF
- Obtain sufficient history

### Configuration Management
- Use version control for configuration files
- Document configuration changes
- Test configurations in development environments

### Performance Optimization
- Monitor resource usage
- Optimize Spark configurations
- Use appropriate cluster sizes

This comprehensive architecture enables the MMF to handle complex forecasting scenarios while maintaining flexibility, scalability, and maintainability. 
