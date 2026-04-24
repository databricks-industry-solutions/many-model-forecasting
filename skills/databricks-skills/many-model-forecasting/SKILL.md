---
name: many-model-forecasting
description: "Kickstart Many Models Forecasting (MMF) projects on Databricks — explore data, profile series, configure clusters, run forecasting pipelines, and evaluate results."
---

# Many Models Forecasting (MMF)

> ⛔ **MANDATORY — Read before taking any action.**
> Before doing anything in an MMF project — querying data, creating resources, generating notebooks, or running jobs — read this file completely. Then identify which skill the user needs and read that skill file in full before proceeding. Do NOT act on user instructions until you have read the relevant skill file.

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
3. **Forecast problem brief (`{forecast_problem_brief}`)** — Captured in Skill 1 (STOP GATE 0b) as a short problem statement (metric meaning, use, horizon, series shape, exogenous intent). Carry it in the conversation through all skills. If it is missing (user skipped Skill 1 or new session), **reconfirm or capture a minimal brief in Skill 2** (Step 2). Any **optional research** (web, Databricks docs, extended reasoning, Skill 2 Step 8b feature-engineering research) must stay **scoped** to `{forecast_problem_brief}` — do not run generic time-series research without tying it back to the brief.

**Per-skill STOP gates are documented in each skill file.** Look for the `⛔ STOP GATE` markers.

## Workflow

The five skills are designed to run in sequence, but **Skill 2 (profiling) is optional**:

```
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
                      submit parallel jobs         merge forecastable +
                      (+ NF job if separate)       non-forecastable results,
                      → {use_case}_evaluation      business-ready summary
                      → {use_case}_scoring         → {use_case}_best_models
                                                     (with forecast_source)
                                                   → {use_case}_evaluation_summary
```

### Non-forecastable series handling

After Skill 2 classifies series into "high-confidence" and "low-signal" groups, the user chooses one of three strategies:

| Strategy | Description | Downstream effect |
|----------|-------------|-------------------|
| **Include** (Option A) | Keep all series together | No table splitting; all series use same models |
| **Fallback** (Option B) | Exclude + apply simple rule (Naive, Seasonal Naive, Mean, or Zero) | Filtered training table; fallback forecasts produced immediately; no cluster needed for NF series |
| **Separate job** (Option C) | Exclude + run a dedicated pipeline with user-selected models | Filtered training tables; separate job with its own cluster sizing; full backtest evaluation for NF series |

Results from all strategies are merged in Skill 5 with a `forecast_source` column tracking provenance.

### Skill 1: Prep and Clean Data (`/prep-and-clean-data`)

Asks for catalog/schema, collects a use case name, captures **`{forecast_problem_brief}`** (what is being forecast and why), discovers time series tables, maps columns to MMF schema (`unique_id`, `ds`, `y`), asks the user how to impute missing data, generates an anomaly analysis report, asks the user how to handle anomalies, creates the `{use_case}_train_data` table, and generates a **reproducibility notebook** that records all decisions.

**STOP gates:** catalog/schema, **forecast problem brief**, imputation strategy, anomaly handling.

See: [1-prep-and-clean-data.md](1-prep-and-clean-data.md)

### Skill 2: Profile and Classify Series (`/profile-and-classify-series`) — OPTIONAL

Calculates statistical properties (stationarity, seasonality, trend, entropy, SNR), partitions series into "high-confidence" and "low-signal" groups, recommends model families, and **asks the user how to handle non-forecastable series** (include, fallback, or separate job). Runs on **serverless compute**.

**STOP gates:** confirm parameters before profiling (including **`{forecast_problem_brief}`**); optional **deep research on feature engineering** after `{use_case}_series_profile` exists and classification/recommendations are summarized (see Step 8b — grounds research in profiling metadata, `{use_case}_train_data` columns, and **`{forecast_problem_brief}`**).

**This step is optional.** If skipped, the user manually selects models in Skill 3 and all series are treated as forecastable. Inform the user of estimated runtime:
- **< 100 series**: ~2–5 minutes
- **100–1,000 series**: ~5–15 minutes
- **1,000–10,000 series**: ~15–45 minutes
- **> 10,000 series**: ~1–2 hours (consider sampling)

**STOP gates:** catalog/schema, non-forecastable strategy selection.

See: [2-profile-and-classify-series.md](2-profile-and-classify-series.md)

### Skill 3: Provision Forecasting Resources (`/provision-forecasting-resources`)

Asks the user which models to run, determines required cluster types, asks the user which clusters to start and confirms configuration. GPU clusters always use **single-node**.

**STOP gates:** model selection, cluster selection.

See: [3-provision-forecasting-resources.md](3-provision-forecasting-resources.md)

### Skill 4: Execute MMF Forecast (`/execute-mmf-forecast`)

Validates parameters, asks the user about backtesting setup, generates notebooks using the **orchestrator + run_gpu** pattern (one `run_gpu` notebook invoked per model via `dbutils.notebook.run()`), creates **one job per model class** (local, global, foundation) and triggers them **in parallel**. The pipeline runs in **univariate mode by default** — all covariate lists (`static_features`, `dynamic_historical_*`, `dynamic_future_*`) default to `[]` and `scoring_table` defaults to `""`. **Avoid `dynamic_future_*`** unless the user has confirmed known future regressors and a pre-built scoring table; prefer `static_features` and `dynamic_historical_*` when covariates are needed. See the Feature type decision guide in [4-execute-mmf-forecast.md](4-execute-mmf-forecast.md).

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
- [mmf_local_notebook_template.ipynb](mmf_local_notebook_template.ipynb) — CPU models (StatsForecast, Prophet); covariate lists default to `[]` (univariate). Avoid `dynamic_future_*` — see Feature type decision guide in Skill 4.
- [mmf_gpu_run_notebook_template.ipynb](mmf_gpu_run_notebook_template.ipynb) — GPU single-model runner (widgets include covariate columns, defaulting to empty/univariate; used by orchestrators)
- [mmf_gpu_orchestrator_notebook_template.ipynb](mmf_gpu_orchestrator_notebook_template.ipynb) — GPU orchestrator (model list + covariate lists passed into `run_gpu`; defaults to `[]` for all covariates)
- [mmf_profiling_notebook_template.ipynb](mmf_profiling_notebook_template.ipynb) — Series profiling (statsmodels, scipy)

### Reproducibility notebooks (generated after interactive sessions)
- [mmf_prep_notebook_template.ipynb](mmf_prep_notebook_template.ipynb) — Data preparation replay (Skill 1): records all interactive decisions (table selection, column mapping, imputation strategy, anomaly capping) and re-creates `{use_case}_train_data` and `{use_case}_cleaning_report`
- [mmf_post_process_notebook_template.ipynb](mmf_post_process_notebook_template.ipynb) — Post-processing replay (Skill 5): re-runs best-model selection, evaluation summary, and business report generation

## Job Architecture (Skill 4)

Up to six separate jobs run **in parallel** — three for the main pipeline, plus up to three for non-forecastable series (if `separate_job` strategy):

```
── Main Pipeline (forecastable series) ──────────────────────────────────────────────────────────────
┌─────────────────────────┐   ┌──────────────────────────────┐   ┌──────────────────────────────────┐
│ Job: {use_case}_local   │   │ Job: {use_case}_global       │   │ Job: {use_case}_foundation       │
│ Cluster: CPU            │   │ Cluster: GPU (single-node)   │   │ Cluster: GPU (single-node)       │
│ Notebook: run_local     │   │ Notebook: orchestrator_global│   │ Notebook: orchestrator_foundation│
│ (all local models)      │   │  └→ run_gpu (per model)      │   │  └→ run_gpu (per model)          │
└─────────────────────────┘   └──────────────────────────────┘   └──────────────────────────────────┘

── Non-Forecastable Pipeline (separate_job strategy only) ───────────────────────────────────────────
┌──────────────────────────┐   ┌───────────────────────────────┐
│ Job: {use_case}_nf_local │   │ Job: {use_case}_nf_global/    │
│ Cluster: NF CPU          │   │      {use_case}_nf_foundation │
│ Notebook: run_local_nf   │   │ Cluster: NF GPU (single-node) │
│ (NF local models)        │   │ Notebook: orchestrator_*_nf   │
└──────────────────────────┘   └───────────────────────────────┘
         ▲                              ▲
         └──────────────────────────────┘
              All jobs triggered in parallel
```

## Prerequisites

- Databricks MCP server configured
- A Databricks workspace with time series data
- Unity Catalog access to the target catalog/schema
- **Skill files uploaded** to `/Workspace/Users/{your_email}/.assistant/skills/many-model-forecasting/`
- **Assistant instructions configured** at `/Workspace/Users/{your_email}/.assistant_instructions.md` — this file is loaded automatically by Genie Code at the start of every conversation. It should include instructions to always read the MMF skill files before taking any action, and to respect all STOP gates. Without this file, the agent may ignore skill instructions and proceed autonomously.

## MMF Installation

Each generated notebook installs `mmf_sa` at the start:
- Local: `pip install "mmf_sa[local] @ git+https://github.com/databricks-industry-solutions/many-model-forecasting.git@main"`
- Global: `pip install "mmf_sa[global] @ git+https://github.com/databricks-industry-solutions/many-model-forecasting.git@main"`
- Foundation: `pip install "mmf_sa[foundation] @ git+https://github.com/databricks-industry-solutions/many-model-forecasting.git@main"`
