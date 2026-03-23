[![Skills CI](https://github.com/databricks-industry-solutions/many-model-forecasting/actions/workflows/skills-ci.yml/badge.svg)](https://github.com/databricks-industry-solutions/many-model-forecasting/actions/workflows/skills-ci.yml)

# Many-Model Forecasting (MMF) Dev Kit

A focused development kit for the **Many-Model Forecasting** skill, enabling AI coding assistants to build time series forecasting pipelines on Databricks.

## What's Included

| Component | Description |
|-----------|-------------|
| [`databricks-skills/many-model-forecasting/`](databricks-skills/many-model-forecasting/) | The MMF skill — patterns and best practices for forecasting on Databricks |
| [`.test/`](.test/) | Test infrastructure for evaluating the skill |

## The MMF Skill

The Many-Model Forecasting skill teaches AI assistants how to:

- **Prepare and clean** time series data with automated imputation and anomaly capping
- **Profile and classify** series by forecastability using statistical properties (ADF, STL, spectral entropy, SNR)
- **Provision** the right Databricks clusters (CPU or GPU) based on model requirements
- **Execute** forecasting pipelines using **StatsForecast**, **NeuralForecast**, **Chronos**, and **TimesFM** models
- **Evaluate** results with multi-metric analysis, best-model selection per series, and business-ready summaries

All generated assets are prefixed with a user-provided **use case name** (e.g., `m4`, `rossmann`), allowing multiple forecasting projects to coexist in the same catalog/schema.

### The 5-Skill Pipeline

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

| Skill | Command | Description |
|-------|---------|-------------|
| 1 | `/prep-and-clean-data` | Discover tables, map columns, impute missing data, cap anomalies |
| 2 | `/profile-and-classify-series` | Compute statistical properties, classify forecastability, recommend models |
| 3 | `/provision-forecasting-resources` | Configure CPU/GPU clusters with dynamic sizing and UC verification |
| 4 | `/execute-mmf-forecast` | Generate notebooks, submit Workflow job (one task per GPU model), monitor |
| 5 | `/post-process-and-evaluate` | Best-model selection, WAPE/sMAPE metrics, evaluation summary |

Skills can be run end-to-end or individually. Skill 2 (profiling) is optional — Skills 3 and 4 fall back to manual configuration if the profile table doesn't exist.

## Prerequisites

This skill depends on the **Databricks MCP tools** (e.g., `connect_to_workspace`, `execute_parameterized_sql`) provided by [`ai-dev-kit`](https://github.com/databricks-solutions/ai-dev-kit). Make sure the Databricks MCP server from `ai-dev-kit` is configured in your AI coding tool before using the skill.

## Installing the Skill into Your Project

Download `install.py` and run it — no need to clone the entire repo:

```bash
curl -O https://raw.githubusercontent.com/databricks-industry-solutions/many-model-forecasting/main/skills/install.py
python3 install.py --target /path/to/your-project
```

The installer downloads the skill files from GitHub automatically and configures your AI coding tools (Claude Code, Cursor, Gemini CLI).

```bash
# Preview what will be created
python3 install.py --target /path/to/your-project --dry-run

# Configure only specific tools
python3 install.py --target /path/to/your-project --tools claude cursor
```

If you already have the repo cloned, it uses the local files instead of downloading.

After installation your project will contain:

```
your-project/
├── CLAUDE.md                              # created or updated
├── GEMINI.md                              # created or updated
├── .cursor/rules/many-model-forecasting.mdc
└── databricks-skills/many-model-forecasting/
    ├── SKILL.md
    ├── 1-prep-and-clean-data.md
    ├── 2-profile-and-classify-series.md
    ├── 3-provision-forecasting-resources.md
    ├── 4-execute-mmf-forecast.md
    ├── 5-post-process-and-evaluate.md
    ├── mmf_local_notebook_template.ipynb
    ├── mmf_gpu_notebook_template.ipynb
    └── mmf_profiling_notebook_template.ipynb
```

Re-running the installer is safe — it updates existing configurations without duplication.

## Running Tests

From the `.test/` directory:

```bash
# Unit tests
uv run --extra dev python -m pytest tests/test_scorers.py -v

# Tier 1 agent tests
uv run --extra tier1 python -m pytest tests/tier1/ -v

# Skill evaluation
uv run --extra dev python scripts/run_eval.py many-model-forecasting
```

## Demo

The following demo shows the MMF slash commands in action against a real Databricks workspace. These commands are implemented as Claude Code slash commands backed by Databricks MCP tools: `/prep-and-clean-data` discovers and cleans the time series data, `/provision-forecasting-resources` configures the right cluster type based on the models you want to run, and `/execute-mmf-forecast` launches the full forecasting pipeline — all from the terminal, driven by an AI coding assistant.

<img src="mmf-demo.svg" width="750" alt="MMF Dev Kit demo"/>
