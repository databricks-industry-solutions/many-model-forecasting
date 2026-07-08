---
id: intro
title: Introduction
slug: /intro
sidebar_position: 1
---

# Many Model Forecasting by Databricks

Bootstrap your large-scale forecasting solutions on Databricks with the **Many Models Forecasting (MMF)** Solution Accelerator.

MMF accelerates the development of forecasting solutions on Databricks, including the critical phases of data preparation, training, backtesting, evaluation, scoring, and deployment. Adopting a **configuration-over-code** approach, MMF minimizes the need for extensive coding. But with its open and extensible architecture, MMF allows technically proficient users to incorporate new models and features. We recommend users read through the source code and modify it to their specific requirements.

MMF integrates a variety of well-established and cutting-edge algorithms, including [local statistical models](./local-models.md), [global machine learning / deep learning models](./global-models.md), and [foundation time series models](./foundation-models.md). MMF enables parallel modeling of hundreds or thousands of time series by leveraging Spark's distributed compute. Users can apply multiple models at once and select the best performing one for each time series based on their custom metrics.

## What's New

Use a cluster with [Databricks Runtime 17.3 LTS for ML](https://docs.databricks.com/en/release-notes/runtime/17.3lts-ml.html) for local models, and [Databricks Runtime 18.0 for ML](https://docs.databricks.com/en/release-notes/runtime/18.0-ml.html) or later for global and foundation models.

- **May 2026** — Added MLForecast for LightGBM support.
- **May 2026** — All model classes (local, global, and foundation) run on serverless.
- **Mar 2026** — Introduced the **MMF Agent**: a set of skills that guide users through an end-to-end forecasting project (preprocess, profile, provision resources, forecast, evaluate). MMF Agent runs on Genie Code, Claude Code, Cursor, and GitHub Copilot.
- **Feb 2026** — Added an interactive app to explore forecasting results.
- **Feb 2026** — [Chronos-2](https://github.com/amazon-science/chronos-forecasting) models are now available for univariate and covariate forecasting (ChronosT5 decommissioned).
- **Feb 2026** — [TimesFM 2.5](https://github.com/google-research/timesfm) is available for univariate and covariate forecasting (TimesFM 1.0 and 2.0 decommissioned).
- **Feb 2026** — Added multi-node multi-GPU support for global models.

## The Three Model Families

| Family | Best for | Backbone libraries |
| --- | --- | --- |
| [Local](./local-models.md) | Individual series, interpretability, low data | statsforecast, sktime |
| [Global](./global-models.md) | Shared learning across many similar series | mlforecast, neuralforecast |
| [Foundation](./foundation-models.md) | Zero-shot forecasting, no training | Chronos, TimesFM |

Ready to forecast? Head to [Getting Started](./getting-started.md).
