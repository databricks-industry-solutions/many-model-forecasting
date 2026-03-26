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

## Interaction Model — Mandatory STOP Gates

This workflow is **interactive by design**. At multiple points the agent MUST pause and wait for explicit user input before proceeding. These are called **STOP gates**.

**Global STOP gates (apply across all skills):**

1. **Catalog & Schema** — Always ask the user which catalog and schema to use. Do NOT assume or reuse values from prior runs. Do NOT proceed until the user confirms.
2. **Step transitions** — After completing each skill (e.g., prep-and-clean-data), present a summary of what was done and ask the user whether to proceed to the next skill. Do NOT auto-advance.

**Per-skill STOP gates are documented in each skill file.** Look for the `⛔ STOP GATE` markers.

## Workflow

The five skills are designed to run in sequence, but **Skill 2 (profiling) is optional**:

```
/prep-and-clean-data  →  /profile-and-classify-series (OPTIONAL)  →  /provision-forecasting-resources
        ↓                          ↓                                          ↓
  Discover, clean &         Statistical profiling,                  Ask user which models,
  prepare data              classify forecastability,               configure CPU/GPU
  → {use_case}_train_data   recommend models                       cluster(s)
                            → {use_case}_series_profile

                    →  /execute-mmf-forecast  →  /post-process-and-evaluate
                              ↓                            ↓
                      Generate notebooks,          Best model selection,
                      submit 3 parallel jobs       business-ready summary
                      → {use_case}_evaluation      → {use_case}_best_models
                      → {use_case}_scoring         → {use_case}_evaluation_summary
```

### Skill 1: Prep and Clean Data (`/prep-and-clean-data`)

Asks for catalog/schema, collects a use case name, discovers time series tables, maps columns to MMF schema (`unique_id`, `ds`, `y`), asks the user how to impute missing data, generates an anomaly analysis report, asks the user how to handle anomalies, creates the `{use_case}_train_data` table, and generates a **reproducibility notebook** that records all decisions.

**STOP gates:** catalog/schema, imputation strategy, anomaly handling.

See: [1-prep-and-clean-data.md](1-prep-and-clean-data.md)

### Skill 2: Profile and Classify Series (`/profile-and-classify-series`) — OPTIONAL

Calculates statistical properties (stationarity, seasonality, trend, entropy, SNR), partitions series into "high-confidence" and "low-signal" groups, and recommends model families. Runs on **serverless compute**.

**This step is optional.** If skipped, the user manually selects models in Skill 3. Inform the user of estimated runtime:
- **< 100 series**: ~2–5 minutes
- **100–1,000 series**: ~5–15 minutes
- **1,000–10,000 series**: ~15–45 minutes
- **> 10,000 series**: ~1–2 hours (consider sampling)

See: [2-profile-and-classify-series.md](2-profile-and-classify-series.md)

### Skill 3: Provision Forecasting Resources (`/provision-forecasting-resources`)

Asks the user which models to run, determines required cluster types, asks the user which clusters to start and confirms configuration. GPU clusters always use **single-node**.

**STOP gates:** model selection, cluster selection.

See: [3-provision-forecasting-resources.md](3-provision-forecasting-resources.md)

### Skill 4: Execute MMF Forecast (`/execute-mmf-forecast`)

Validates parameters, asks the user about backtesting setup, generates notebooks using the **orchestrator + run_gpu** pattern (one `run_gpu` notebook invoked per model via `dbutils.notebook.run()`), creates **one job per model class** (local, global, foundation) and triggers them **in parallel**.

**STOP gates:** backtesting setup, model confirmation.

See: [4-execute-mmf-forecast.md](4-execute-mmf-forecast.md)

### Skill 5: Post-Process and Evaluate (`/post-process-and-evaluate`)

Calculates multi-metric evaluation (MAPE, sMAPE, WAPE), selects best model per series, ranks models by win count, generates a business-ready summary report, and produces a **reproducibility notebook** for re-running the evaluation.

See: [5-post-process-and-evaluate.md](5-post-process-and-evaluate.md)

## Available Models

| Category | Models | Compute |
|----------|--------|---------|
| **Local (CPU)** | StatsForecastAutoArima, AutoETS, AutoCES, AutoTheta, AutoTbats, AutoMfles, SKTimeProphet, and baseline/intermittent models | CPU cluster |
| **Global (GPU)** | NeuralForecastAutoNHITS, AutoPatchTST, AutoRNN, AutoLSTM, AutoNBEATSx, AutoTiDE, and non-auto variants | GPU cluster (single-node) |
| **Foundation (GPU)** | ChronosBoltTiny/Mini/Small/Base, Chronos2, Chronos2Small, Chronos2Synth, TimesFM_2_5_200m | GPU cluster (single-node) |

## Cluster Configurations

| Cluster | Runtime | Node type (AWS) | Workers | Use case |
|---------|---------|-----------------|---------|----------|
| `{use_case}_cpu_cluster` | `17.3.x-cpu-ml-scala2.13` | `i3.xlarge` | Dynamic (0-10 based on series count) | Local models |
| `{use_case}_gpu_cluster` | `18.0.x-gpu-ml-scala2.13` | User-selectable (e.g., `g5.12xlarge`) | **0 (single-node always)** | Global & foundation models |

See [3-provision-forecasting-resources.md](3-provision-forecasting-resources.md) for Azure and GCP node types.

## Notebook Templates

### Pipeline notebooks (generated during execution)
- [mmf_local_notebook_template.ipynb](mmf_local_notebook_template.ipynb) — CPU models (StatsForecast, Prophet)
- [mmf_gpu_run_notebook_template.ipynb](mmf_gpu_run_notebook_template.ipynb) — GPU single-model runner (receives model name via widget, used by orchestrators)
- [mmf_gpu_orchestrator_notebook_template.ipynb](mmf_gpu_orchestrator_notebook_template.ipynb) — GPU orchestrator (holds model list, invokes run_gpu per model via `dbutils.notebook.run()`)
- [mmf_profiling_notebook_template.ipynb](mmf_profiling_notebook_template.ipynb) — Series profiling (statsmodels, scipy)

### Reproducibility notebooks (generated after interactive sessions)
- [mmf_prep_notebook_template.ipynb](mmf_prep_notebook_template.ipynb) — Data preparation replay (Skill 1): records all interactive decisions (table selection, column mapping, imputation strategy, anomaly capping) and re-creates `{use_case}_train_data` and `{use_case}_cleaning_report`
- [mmf_post_process_notebook_template.ipynb](mmf_post_process_notebook_template.ipynb) — Post-processing replay (Skill 5): re-runs best-model selection, evaluation summary, and business report generation

## Job Architecture (Skill 4)

Three separate jobs run **in parallel**, one per model class:

```
┌─────────────────────────┐   ┌──────────────────────────────┐   ┌──────────────────────────────────┐
│ Job: {use_case}_local   │   │ Job: {use_case}_global       │   │ Job: {use_case}_foundation       │
│ Cluster: CPU            │   │ Cluster: GPU (single-node)   │   │ Cluster: GPU (single-node)       │
│ Notebook: run_local     │   │ Notebook: orchestrator_global│   │ Notebook: orchestrator_foundation│
│ (all local models)      │   │  └→ run_gpu (per model)      │   │  └→ run_gpu (per model)          │
└─────────────────────────┘   └──────────────────────────────┘   └──────────────────────────────────┘
         ▲                              ▲                                    ▲
         └──────────────────────────────┴────────────────────────────────────┘
                                   Triggered in parallel
```

## Prerequisites

- Databricks MCP server configured
- A Databricks workspace with time series data
- Unity Catalog access to the target catalog/schema

## MMF Installation

Each generated notebook installs `mmf_sa` at the start:
- Local: `pip install "mmf_sa[local] @ git+https://github.com/databricks-industry-solutions/many-model-forecasting.git@v0.1.2"`
- Global: `pip install "mmf_sa[global] @ git+https://github.com/databricks-industry-solutions/many-model-forecasting.git@v0.1.2"`
- Foundation: `pip install "mmf_sa[foundation] @ git+https://github.com/databricks-industry-solutions/many-model-forecasting.git@v0.1.2"`
