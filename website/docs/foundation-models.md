---
id: foundation-models
title: Foundation Models
sidebar_position: 5
---

# Foundation Models

Foundation time series models are mostly transformer-based models pretrained on millions or billions of time points. They can perform analysis (forecasting, anomaly detection, classification) on a previously unseen time series without training or tuning. MMF supports open source models from multiple sources: [Chronos](https://github.com/amazon-science/chronos-forecasting) (Chronos-Bolt and Chronos-2) and [TimesFM](https://github.com/google-research/timesfm). This is a rapidly changing field, and we update the supported models as it evolves.

## Cluster requirements

Attach the [examples/daily/foundation_daily.ipynb](https://github.com/databricks-industry-solutions/many-model-forecasting/blob/main/examples/daily/foundation_daily.ipynb) notebook to a cluster running [DBR 18.0 for ML](https://docs.databricks.com/en/release-notes/runtime/18.0-ml.html) or later. We recommend a single-node cluster with multiple GPUs such as `g5.12xlarge` (A10G) on AWS or `Standard_NV36ads_A10_v5` on Azure. Multi-node setup is currently not supported.

## Serverless GPU

Alternatively, run foundation models on [serverless GPU](https://docs.databricks.com/aws/en/compute/serverless/gpu) compute by passing `serverless=True` to `run_forecast`. This routes Chronos and TimesFM inference through a driver-only predict path instead of Spark Pandas UDFs. It is required on serverless GPU because Spark Connect Python workers are CPU-only; the trade-off versus the classic-cluster path is no multi-GPU data parallelism. See [examples/serverless/foundation_serverless.ipynb](https://github.com/databricks-industry-solutions/many-model-forecasting/blob/main/examples/serverless/foundation_serverless.ipynb) for a runnable example.

## Selecting models

```python
active_models = [
    "ChronosBoltTiny",
    "ChronosBoltMini",
    "ChronosBoltSmall",
    "ChronosBoltBase",
    "Chronos2",
    "Chronos2Small",
    "Chronos2Synth",
    "TimesFM_2_5_200m",
]
```

A comprehensive list of supported models is available in the [models README](https://github.com/databricks-industry-solutions/many-model-forecasting/blob/main/mmf_sa/models/README.md).

## Running the forecast

As with global models, loop over `active_models` and run the [run_daily.ipynb](https://github.com/databricks-industry-solutions/many-model-forecasting/blob/main/examples/run_daily.ipynb) notebook:

```python
for model in active_models:
    dbutils.notebook.run(
        "run_daily",
        timeout_seconds=0,
        arguments={"catalog": catalog, "db": db, "model": model, "run_id": run_id},
    )
```

Since these models are pretrained, no training occurs — during evaluation, the models are logged and registered to Unity Catalog.

## Learn more

Read through [foundation_daily.ipynb](https://github.com/databricks-industry-solutions/many-model-forecasting/blob/main/examples/daily/foundation_daily.ipynb). An example with exogenous regressors is in [foundation_external_regressors_daily.ipynb](https://github.com/databricks-industry-solutions/many-model-forecasting/blob/main/examples/external_regressors/foundation_external_regressors_daily.ipynb).
