---
id: global-models
title: Global Models
sidebar_position: 4
---

# Global Models

Global models leverage patterns across multiple time series, enabling shared learning and improved predictions for each series. You train one model for many or all time series. They can often deliver better performance and robustness for large, similar datasets. MMF supports machine learning models with [mlforecast](https://nixtlaverse.nixtla.io/mlforecast/index.html) and deep learning models with [neuralforecast](https://nixtlaverse.nixtla.io/neuralforecast/index.html). Covariates and hyperparameter tuning are both supported for some models.

## Cluster requirements

Attach the [examples/daily/global_daily_dl.ipynb](https://github.com/databricks-industry-solutions/many-model-forecasting/blob/main/examples/daily/global_daily_dl.ipynb) notebook to a cluster running [DBR 18.0 for ML](https://docs.databricks.com/en/release-notes/runtime/18.0-ml.html) or later. We recommend a GPU cluster such as `g5.12xlarge` (A10G) on AWS or `Standard_NV36ads_A10_v5` on Azure. Both single-node multi-GPU and multi-node multi-GPU clusters are supported.

## Selecting models

```python
active_models = [
    "NeuralForecastRNN",
    "NeuralForecastLSTM",
    "NeuralForecastNBEATSx",
    "NeuralForecastNHITS",
    "NeuralForecastAutoRNN",
    "NeuralForecastAutoLSTM",
    "NeuralForecastAutoNBEATSx",
    "NeuralForecastAutoNHITS",
    "NeuralForecastAutoTiDE",
    "NeuralForecastAutoPatchTST",
]
```

Models prefixed with `Auto` perform hyperparameter optimization within a specified range. A comprehensive list is available in the [models README](https://github.com/databricks-industry-solutions/many-model-forecasting/blob/main/mmf_sa/models/README.md).

## Running the forecast

Global and foundation models are run by looping over `active_models` and calling the [run_daily.ipynb](https://github.com/databricks-industry-solutions/many-model-forecasting/blob/main/examples/run_daily.ipynb) notebook, which in turn calls `run_forecast`:

```python
# Number of nodes for distributed training. Use 1 for single-node multi-GPU,
# or set to the number of worker nodes for multi-node multi-GPU clusters.
num_nodes = 1

for model in active_models:
    dbutils.notebook.run(
        "run_daily",
        timeout_seconds=0,
        arguments={"catalog": catalog, "db": db, "model": model, "run_id": run_id, "num_nodes": str(num_nodes)},
    )
```

Inside `run_daily.ipynb`, `run_forecast` is configured with GPU-specific parameters:

```python
run_forecast(
    spark=spark,
    train_data=f"{catalog}.{db}.m4_daily_train",
    scoring_data=f"{catalog}.{db}.m4_daily_train",
    scoring_output=f"{catalog}.{db}.daily_scoring_output",
    evaluation_output=f"{catalog}.{db}.daily_evaluation_output",
    model_output=f"{catalog}.{db}",
    group_id="unique_id",
    date_col="ds",
    target="y",
    freq="D",
    prediction_length=10,
    backtest_length=30,
    stride=10,
    metric="smape",
    train_predict_ratio=2,
    data_quality_check=True,
    resample=False,
    active_models=[model],
    experiment_path="/Shared/mmf_experiment",
    use_case_name="m4_daily",
    run_id=run_id,
    accelerator="gpu",
    num_nodes=num_nodes,
)
```

### Additional parameters

- `model_output` — where you store your trained model.
- `use_case_name` — used to suffix the model name when registered to Unity Catalog.
- `accelerator` — tells MMF to use GPU instead of CPU.
- `num_nodes` — number of nodes for distributed training (default `1`). For multi-node clusters, set this to the number of **worker** nodes. Autoscaling must be disabled on multi-node GPU clusters to prevent workers from being removed mid-training.

Different loss functions (`smape`, `mae`, `mse`, `rmse`, `mape`, `mase`) are supported for training and evaluating global models via the `loss` field in [models_conf.yaml](https://github.com/databricks-industry-solutions/many-model-forecasting/blob/main/mmf_sa/models/models_conf.yaml).

## Learn more

Read through [global_daily_dl.ipynb](https://github.com/databricks-industry-solutions/many-model-forecasting/blob/main/examples/daily/global_daily_dl.ipynb). An example with exogenous regressors is in [global_external_regressors_daily_dl.ipynb](https://github.com/databricks-industry-solutions/many-model-forecasting/blob/main/examples/external_regressors/global_external_regressors_daily_dl.ipynb).
