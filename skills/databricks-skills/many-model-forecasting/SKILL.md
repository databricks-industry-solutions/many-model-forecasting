---
name: many-model-forecasting
description: "Kickstart Many Models Forecasting (MMF) projects on Databricks — explore data, configure clusters, and run forecasting pipelines."
---

# Many Models Forecasting (MMF)

## Overview

Automates the full [Many Models Forecasting](https://github.com/databricks-industry-solutions/many-model-forecasting) workflow on Databricks. Three skills walk you through data exploration, cluster setup, and pipeline execution interactively using Databricks MCP tools and `AskUserQuestion`.

## When to Use

Use this skill when a user wants to:
- Run time series forecasting at scale on Databricks
- Set up an MMF project with `mmf_sa` (the Many Model Forecasting solution accelerator)
- Train statistical, global neural, or foundation models across many series
- Explore and prepare time series data for forecasting

## Workflow

The three skills are designed to run in sequence:

```
/explore-data  →  /setup-cluster  →  /run-mmf
    ↓                   ↓                ↓
 Discover &        Configure         Generate notebooks,
 prepare data     cluster(s)         submit Workflow job,
 → mmf_train_data                    analyze results
```

### Skill 1: Explore Data (`/explore-data <catalog> <schema>`)

Discovers time series tables, maps columns to MMF schema (`unique_id`, `ds`, `y`), runs data quality checks, and creates the `mmf_train_data` table.

See: [1-explore-data.md](1-explore-data.md)

### Skill 2: Setup Cluster (`/setup-cluster <model-types>`)

Recommends CPU or GPU cluster configs based on model types (local, global, foundation) and cloud provider.

See: [2-setup-the-mmf-cluster.md](2-setup-the-mmf-cluster.md)

### Skill 3: Run MMF (`/run-mmf <catalog> <schema>`)

Generates parameterized notebooks, submits a Databricks Workflow job, monitors execution, and analyzes results.

See: [3-run-mmf.md](3-run-mmf.md)

## Available Models

| Category | Models | Compute |
|----------|--------|---------|
| **Local (CPU)** | StatsForecastAutoArima, AutoETS, AutoCES, AutoTheta, AutoTbats, AutoMfles, SKTimeProphet, and baseline models | CPU cluster |
| **Global (GPU)** | NeuralForecastNHITS, PatchTST, RNN, LSTM, NBEATSx, TiDE | GPU cluster |
| **Foundation (GPU)** | ChronosBoltTiny/Mini/Small/Base, Chronos2, TimesFM_2_5_200m | GPU cluster |

## Cluster Configurations

| Cluster | Runtime | Node type (AWS) | Workers | Use case |
|---------|---------|-----------------|---------|----------|
| `mmf_cpu_cluster` | `17.3.x-cpu-ml-scala2.13` | `i3.xlarge` | 2 | Local models |
| `mmf_gpu_cluster` | `18.0.x-gpu-ml-scala2.13` | `g5.12xlarge` | 0 (single-node) | Global & foundation models |

See [2-setup-the-mmf-cluster.md](2-setup-the-mmf-cluster.md) for Azure and GCP node types.

## Notebook Templates

- [mmf_local_notebook_template.ipynb](mmf_local_notebook_template.ipynb) — CPU models (StatsForecast, Prophet)
- [mmf_gpu_notebook_template.ipynb](mmf_gpu_notebook_template.ipynb) — GPU models (NeuralForecast, Chronos, TimesFM)

## Prerequisites

- Databricks MCP server configured
- A Databricks workspace with time series data
- Unity Catalog access to the target catalog/schema

## MMF Installation

Each generated notebook installs `mmf_sa` at the start:
- Local: `pip install "mmf_sa[local] @ git+https://github.com/databricks-industry-solutions/many-model-forecasting.git@v0.1.2"`
- Global: `pip install "mmf_sa[global] @ git+https://github.com/databricks-industry-solutions/many-model-forecasting.git@v0.1.2"`
- Foundation: `pip install "mmf_sa[foundation] @ git+https://github.com/databricks-industry-solutions/many-model-forecasting.git@v0.1.2"`