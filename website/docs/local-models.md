---
id: local-models
title: Local Models
sidebar_position: 3
---

# Local Models

Local models are used to model individual time series. They can be advantageous over other types of models for their ability to tailor-fit individual series, offer greater interpretability, and require less data. MMF supports models from [statsforecast](https://github.com/Nixtla/statsforecast) and [sktime](https://www.sktime.net/en/stable/). Covariates (exogenous regressors) are currently only supported for some statsforecast models.

## Cluster requirements

Attach the [examples/daily/local_univariate_daily.ipynb](https://github.com/databricks-industry-solutions/many-model-forecasting/blob/main/examples/daily/local_univariate_daily.ipynb) notebook to a cluster running [DBR 17.3 LTS for ML](https://docs.databricks.com/en/release-notes/runtime/17.3lts-ml.html) or later. The cluster can be single-node or multi-node CPU. Set the following Spark configurations before you start:

- `spark.sql.execution.arrow.enabled true`
- `spark.sql.adaptive.enabled false`

## Selecting models

You can specify the models to use in a list:

```python
active_models = [
    "StatsForecastBaselineWindowAverage",
    "StatsForecastBaselineSeasonalWindowAverage",
    "StatsForecastBaselineNaive",
    "StatsForecastBaselineSeasonalNaive",
    "StatsForecastAutoArima",
    "StatsForecastAutoETS",
    "StatsForecastAutoCES",
    "StatsForecastAutoTheta",
    "StatsForecastAutoTbats",
    "StatsForecastAutoMfles",
    "StatsForecastTSB",
    "StatsForecastADIDA",
    "StatsForecastIMAPA",
    "StatsForecastCrostonClassic",
    "StatsForecastCrostonOptimized",
    "StatsForecastCrostonSBA",
    "SKTimeProphet",
]
```

A comprehensive list of supported local models is available in the [models README](https://github.com/databricks-industry-solutions/many-model-forecasting/blob/main/mmf_sa/models/README.md).

## Running the forecast

Run forecasting with the `run_forecast` function and the `active_models` list above. See the full parameter reference in [Getting Started](./getting-started.md).

```python
run_forecast(
    spark=spark,
    train_data=f"{catalog}.{db}.m4_daily_train",
    scoring_data=f"{catalog}.{db}.m4_daily_train",
    scoring_output=f"{catalog}.{db}.daily_scoring_output",
    evaluation_output=f"{catalog}.{db}.daily_evaluation_output",
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
    active_models=active_models,
    experiment_path="/Shared/mmf_experiment",
    use_case_name="m4_daily",
)
```

## Learn more

We encourage you to read through the [local_univariate_daily.ipynb](https://github.com/databricks-industry-solutions/many-model-forecasting/blob/main/examples/daily/local_univariate_daily.ipynb) notebook. An example with exogenous regressors is in [local_univariate_external_regressors_daily.ipynb](https://github.com/databricks-industry-solutions/many-model-forecasting/blob/main/examples/external_regressors/local_univariate_external_regressors_daily.ipynb). See how to define backtesting parameters in the [mmf_sa README](https://github.com/databricks-industry-solutions/many-model-forecasting/blob/main/mmf_sa/README.md#how-backtesting-works).
