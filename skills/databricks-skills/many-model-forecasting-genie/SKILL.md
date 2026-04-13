---
name: many-model-forecasting
description: "Run Many Models Forecasting (MMF) projects on Databricks — explore data, profile series, configure clusters, run forecasting pipelines, and evaluate results."
---

# Many Models Forecasting (MMF)

## Overview

Guides you through the full [Many Models Forecasting](https://github.com/databricks-industry-solutions/many-model-forecasting) workflow on Databricks. Five steps walk you through data preparation, series profiling, cluster setup, pipeline execution, and post-evaluation interactively.

All generated assets are prefixed with a user-provided **use case name** (e.g., `m4`, `rossmann`), allowing multiple forecasting projects to coexist in the same catalog/schema.

## When to Use

Use this skill when a user wants to:
- Run time series forecasting at scale on Databricks
- Set up an MMF project with `mmf_sa` (the Many Model Forecasting solution accelerator)
- Train statistical, global neural, or foundation models across many series
- Explore, clean, and prepare time series data for forecasting
- Profile and classify time series for model selection
- Evaluate and compare forecasting model results

**Step 0 (Generate Sample Data) is for demos and testing only.** Production users with real data start at Step 1.

## Interaction Model — Mandatory STOP Gates

This workflow is **interactive by design**. At multiple points you MUST pause and ask the user for input before proceeding. These are called **STOP gates**.

**Global STOP gates (apply across all steps):**

1. **Catalog & Schema** — Always ask the user which catalog and schema to use. Do NOT assume or reuse values from prior runs. Do NOT proceed until the user confirms.
2. **Step transitions** — After completing each step, present a summary of what was done and ask the user whether to proceed to the next step. Do NOT auto-advance.

**Per-step STOP gates are documented in each step file.** Look for the `⛔ STOP GATE` markers.

## Agent Behavior Rules

These rules apply across all steps and override default agent behavior.

### Do not execute pipeline code inline

All MMF pipeline code — profiling, model training, and scoring — MUST run inside a Databricks notebook on cluster or serverless compute. **Never** run `mmf_sa`, profiling, or model training code directly in the conversation. Always:
1. Generate the notebook from the template (replacing `{placeholder}` tokens)
2. Upload it to the workspace
3. Trigger a run (job or `runs/submit`)

Inline SQL queries for data exploration and table creation are fine — they are not pipeline code.

### Confirm before creating workspace artifacts

Before creating or modifying any workspace artifact (notebook, job, or run submission), you MUST:
1. Explain what you are about to create and why
2. Show a summary of the content or configuration (path, parameters, cluster type)
3. Explicitly ask the user for confirmation before proceeding

Do NOT upload a notebook, create a job, or submit a run until the user confirms.

### Use Databricks platform tools first

Always use native Databricks tools (Jobs API, `runs/submit`) to create and trigger jobs. Do NOT use CLI commands or write job-creation code inline. CLI and SDK code are last resort only if the platform tools are unavailable.

### Do not narrate internal implementation choices

Handle internal decisions silently — tool availability, fallback strategies, agent delegation. Only communicate what is relevant to the user: what was done, what comes next, and what you need from them. Do not say things like "tool X is not available, so I'll use Y instead."

## Workflow

The steps are designed to run in sequence. **Step 0 is optional** (demos and testing only). **Step 2 (profiling) is also optional**:

```
[OPTIONAL] Step 0: Generate Sample Data
        ↓ (skips Step 1 — data already in `{use_case}_train_data`)
/prep-and-clean-data  →  /profile-and-classify-series (OPTIONAL)  →  /provision-forecasting-resources
        ↓                          ↓                                          ↓
  Discover, clean &         Statistical profiling,                  Ask user which models,
  prepare data              classify forecastability,               configure CPU/GPU
  → {use_case}_train_data   recommend models, ask user              cluster(s)
                            how to handle non-forecastable
                            series (include / fallback /
                            separate job)
                            → {use_case}_series_profile
                            → {use_case}_pipeline_config

                    →  /execute-mmf-forecast  →  /post-process-and-evaluate
                              ↓                            ↓
                      Generate notebooks,          Best model selection,
                      upload & trigger jobs        merge forecastable +
                      (+ NF job if separate)       non-forecastable results,
                      → {use_case}_evaluation      business-ready summary
                      → {use_case}_scoring         → {use_case}_best_models
                                                     (with forecast_source)
                                                   → {use_case}_evaluation_summary
```

### Non-forecastable series handling

After Step 2 classifies series into "high-confidence" and "low-signal" groups, the user chooses one of three strategies:

| Strategy | Description | Downstream effect |
|----------|-------------|-------------------|
| **Include** (Option A) | Keep all series together | No table splitting; all series use same models |
| **Fallback** (Option B) | Exclude + apply simple rule (Naive, Seasonal Naive, Mean, or Zero) | Filtered training table; fallback forecasts produced immediately; no cluster needed for NF series |
| **Separate job** (Option C) | Exclude + run a dedicated pipeline with user-selected models | Filtered training tables; separate job with its own cluster sizing; full backtest evaluation for NF series |

Results from all strategies are merged in Step 5 with a `forecast_source` column tracking provenance.

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

The following notebooks are included in this skill folder and should be uploaded to the Databricks workspace during execution:

- [notebooks/mmf_generate_data_notebook_template.ipynb](notebooks/mmf_generate_data_notebook_template.ipynb) — Sample data generation (Step 0)
- [notebooks/mmf_local_notebook_template.ipynb](notebooks/mmf_local_notebook_template.ipynb) — CPU models (StatsForecast, Prophet)
- [notebooks/mmf_gpu_run_notebook_template.ipynb](notebooks/mmf_gpu_run_notebook_template.ipynb) — GPU single-model runner (receives model name via widget)
- [notebooks/mmf_gpu_orchestrator_notebook_template.ipynb](notebooks/mmf_gpu_orchestrator_notebook_template.ipynb) — GPU orchestrator (loops models via `dbutils.notebook.run()`)
- [notebooks/mmf_profiling_notebook_template.ipynb](notebooks/mmf_profiling_notebook_template.ipynb) — Series profiling

## Job Architecture (Step 4)

Up to six separate jobs run **in parallel**:

```
── Main Pipeline ──────────────────────────────────────────────────────────────────────────────────────
┌─────────────────────────┐   ┌──────────────────────────────┐   ┌──────────────────────────────────┐
│ Job: {use_case}_local   │   │ Job: {use_case}_global       │   │ Job: {use_case}_foundation       │
│ Cluster: CPU            │   │ Cluster: GPU (single-node)   │   │ Cluster: GPU (single-node)       │
│ Notebook: run_local     │   │ Notebook: orchestrator_global│   │ Notebook: orchestrator_foundation│
│ (all local models)      │   │  └→ run_gpu (per model)      │   │  └→ run_gpu (per model)          │
└─────────────────────────┘   └──────────────────────────────┘   └──────────────────────────────────┘
```

## Prerequisites

- A Databricks workspace with Unity Catalog enabled
- Time series data accessible in a Delta table
- Permissions to create clusters and jobs in the workspace

## MMF Installation

Each generated notebook installs `mmf_sa` at the start:
- Local: `pip install "mmf_sa[local] @ git+https://github.com/databricks-industry-solutions/many-model-forecasting.git@v0.1.2"`
- Global: `pip install "mmf_sa[global] @ git+https://github.com/databricks-industry-solutions/many-model-forecasting.git@v0.1.2"`
- Foundation: `pip install "mmf_sa[foundation] @ git+https://github.com/databricks-industry-solutions/many-model-forecasting.git@v0.1.2"`
