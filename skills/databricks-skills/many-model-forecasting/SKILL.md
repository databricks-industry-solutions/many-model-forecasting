---
name: many-model-forecasting
description: "Kickstart Many Models Forecasting (MMF) projects on Databricks — explore data, profile series, configure clusters, run forecasting pipelines, and evaluate results."
---

# Many Models Forecasting (MMF)

## Overview

Automates the full [Many Models Forecasting](https://github.com/databricks-industry-solutions/many-model-forecasting) workflow on Databricks. Five skills walk you through data preparation, series profiling, cluster setup, pipeline execution, and post-evaluation interactively using Databricks MCP tools and `AskUserQuestion`.

All generated assets are prefixed with a user-provided **use case name** (e.g., `m4`, `rossmann`), allowing multiple forecasting projects to coexist in the same catalog/schema.

## When to Use

Use this skill when a user wants to:
- Run time series forecasting at scale on Databricks
- Set up an MMF project with `mmf_sa` (the Many Model Forecasting solution accelerator)
- Train statistical, global neural, or foundation models across many series
- Explore, clean, and prepare time series data for forecasting
- Profile and classify time series for model selection
- Evaluate and compare forecasting model results

## Workflow

The five skills are designed to run in sequence:

```
/prep-and-clean-data  →  /profile-and-classify-series  →  /provision-forecasting-resources
        ↓                          ↓                                ↓
  Discover, clean &         Statistical profiling,          Configure CPU/GPU
  prepare data              classify forecastability,       cluster(s)
  → {use_case}_train_data   recommend models
                            → {use_case}_series_profile

                    →  /execute-mmf-forecast  →  /post-process-and-evaluate
                              ↓                            ↓
                      Generate notebooks,          Best model selection,
                      submit Workflow job           business-ready summary
                      → {use_case}_evaluation      → {use_case}_best_models
                      → {use_case}_scoring         → {use_case}_evaluation_summary
```

### Skill 1: Prep and Clean Data (`/prep-and-clean-data <catalog> <schema>`)

Collects a use case name, discovers time series tables, maps columns to MMF schema (`unique_id`, `ds`, `y`), applies automated cleaning (imputation, anomaly capping), and creates the `{use_case}_train_data` table.

See: [1-prep-and-clean-data.md](1-prep-and-clean-data.md)

### Skill 2: Profile and Classify Series (`/profile-and-classify-series <catalog> <schema>`)

Calculates statistical properties (stationarity, seasonality, trend, entropy, SNR), partitions series into "high-confidence" and "low-signal" groups, and recommends model families.

See: [2-profile-and-classify-series.md](2-profile-and-classify-series.md)

### Skill 3: Provision Forecasting Resources (`/provision-forecasting-resources`)

Determines required cluster types from profiling output or user input, configures CPU/GPU clusters with correct specs, verifies UC enablement, and optionally reuses existing clusters.

See: [3-provision-forecasting-resources.md](3-provision-forecasting-resources.md)

### Skill 4: Execute MMF Forecast (`/execute-mmf-forecast <catalog> <schema>`)

Validates parameters, generates parameterized notebooks (one per GPU model to avoid CUDA memory issues), submits a Databricks Workflow job, monitors execution, and logs run metadata.

See: [4-execute-mmf-forecast.md](4-execute-mmf-forecast.md)

### Skill 5: Post-Process and Evaluate (`/post-process-and-evaluate <catalog> <schema>`)

Calculates multi-metric evaluation (MAPE, sMAPE, WAPE), selects best model per series, ranks models by win count, and generates a business-ready summary report.

See: [5-post-process-and-evaluate.md](5-post-process-and-evaluate.md)

## Available Models

| Category | Models | Compute |
|----------|--------|---------|
| **Local (CPU)** | StatsForecastAutoArima, AutoETS, AutoCES, AutoTheta, AutoTbats, AutoMfles, SKTimeProphet, and baseline/intermittent models | CPU cluster |
| **Global (GPU)** | NeuralForecastAutoNHITS, AutoPatchTST, AutoRNN, AutoLSTM, AutoNBEATSx, AutoTiDE, and non-auto variants | GPU cluster |
| **Foundation (GPU)** | ChronosBoltTiny/Mini/Small/Base, Chronos2, Chronos2Small, Chronos2Synth, TimesFM_2_5_200m | GPU cluster |

## Cluster Configurations

| Cluster | Runtime | Node type (AWS) | Workers | Use case |
|---------|---------|-----------------|---------|----------|
| `{use_case}_cpu_cluster` | `17.3.x-cpu-ml-scala2.13` | `i3.xlarge` | Dynamic (0-10 based on series count) | Local models + profiling |
| `{use_case}_gpu_cluster` | `18.0.x-gpu-ml-scala2.13` | User-selectable (e.g., `g5.12xlarge`) | 0 (single-node) | Global & foundation models |

See [3-provision-forecasting-resources.md](3-provision-forecasting-resources.md) for Azure and GCP node types.

## Notebook Templates

- [mmf_local_notebook_template.ipynb](mmf_local_notebook_template.ipynb) — CPU models (StatsForecast, Prophet)
- [mmf_gpu_notebook_template.ipynb](mmf_gpu_notebook_template.ipynb) — GPU models (NeuralForecast, Chronos, TimesFM) — one model per session
- [mmf_profiling_notebook_template.ipynb](mmf_profiling_notebook_template.ipynb) — Series profiling (statsmodels, scipy)

## Prerequisites

- Databricks MCP server configured
- A Databricks workspace with time series data
- Unity Catalog access to the target catalog/schema

## MMF Installation

Each generated notebook installs `mmf_sa` at the start:
- Local: `pip install "mmf_sa[local] @ git+https://github.com/databricks-industry-solutions/many-model-forecasting.git@v0.1.2"`
- Global: `pip install "mmf_sa[global] @ git+https://github.com/databricks-industry-solutions/many-model-forecasting.git@v0.1.2"`
- Foundation: `pip install "mmf_sa[foundation] @ git+https://github.com/databricks-industry-solutions/many-model-forecasting.git@v0.1.2"`
