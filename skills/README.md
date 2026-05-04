# Many-Model Forecasting (MMF) Agent

A focused skill set for the **Many-Model Forecasting**, enabling AI coding assistants to build time series forecasting pipelines on Databricks.

## What's Included


| Component                                                                                | Description                                                               |
| ---------------------------------------------------------------------------------------- | ------------------------------------------------------------------------- |
| `[databricks-skills/many-model-forecasting/](databricks-skills/many-model-forecasting/)` | The MMF skill — patterns and best practices for forecasting on Databricks |
| `[.test/](.test/)`                                                                       | Test infrastructure for evaluating the skill                              |


## The MMF Skill

The Many-Model Forecasting skill teaches AI assistants how to:

- **Prepare and clean** time series data with automated imputation and anomaly capping
- **Profile and classify** series by forecastability using statistical properties (ADF, STL, spectral entropy, SNR)
- **Handle non-forecastable series** — the user chooses to keep them, apply a simple fallback (Naive, Seasonal Naive, Mean, Zero), or run a separate pipeline with dedicated models
- **Provision** the right Databricks clusters (CPU with 16 or 32 vCPU options, or GPU) based on model requirements and dataset size
- **Execute** forecasting pipelines using **StatsForecast**, **SKTime**, **NeuralForecast**, **Chronos**, and **TimesFM** models
- **Evaluate** results with multi-metric analysis, best-model selection per series, and business-ready summaries with `forecast_source` tracking

All generated assets are prefixed with a user-provided **use case name** (e.g., `m4`, `rossmann`), allowing multiple forecasting projects to coexist in the same catalog/schema.

### The 5-Skill Pipeline

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


| Skill | Command                            | Description                                                                                                                                            |
| ----- | ---------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 1     | `/prep-and-clean-data`             | Discover tables, map columns, ask user about imputation and anomaly handling                                                                           |
| 2     | `/profile-and-classify-series`     | **(Optional)** Compute statistical properties, classify forecastability, recommend models, ask user how to handle non-forecastable series (serverless) |
| 3     | `/provision-forecasting-resources` | Ask user which models and clusters, configure CPU (16 vCPU default, 32 vCPU option) / GPU; size separate NF cluster if needed                          |
| 4     | `/execute-mmf-forecast`            | Generate orchestrator + run notebooks, create one job per model class (+ NF jobs), run in parallel                                                     |
| 5     | `/post-process-and-evaluate`       | Best-model selection, WAPE/sMAPE metrics, merge NF results, evaluation summary with `forecast_source`                                                  |


### Non-Forecastable Series Handling

After Skill 2 classifies series, the user chooses how to handle low-signal (non-forecastable) series:


| Strategy                    | Description                                                        | Downstream effect                                                                            |
| --------------------------- | ------------------------------------------------------------------ | -------------------------------------------------------------------------------------------- |
| **Include** (Option A)      | Keep all series together                                           | No table splitting; all series use same models                                               |
| **Fallback** (Option B)     | Exclude + apply simple rule (Naive, Seasonal Naive, Mean, or Zero) | Filtered training table; fallback forecasts produced immediately; no cluster needed          |
| **Separate job** (Option C) | Exclude + run a dedicated pipeline with user-selected models       | Filtered training tables; separate job with its own cluster sizing; full backtest evaluation |


The user's choice is stored in `{use_case}_pipeline_config` and read by Skills 3–5. Results from all strategies are merged in Skill 5 with a `forecast_source` column tracking provenance (`main_pipeline`, `fallback`, or `non_forecastable_pipeline`).

The workflow is **interactive** — the agent pauses at STOP gates to ask the user for decisions (catalog/schema, imputation strategy, anomaly handling, non-forecastable strategy, model selection, cluster configuration, backtesting setup). Skills can be run end-to-end or individually. Skill 2 (profiling) is optional — Skills 3 and 4 fall back to manual configuration if the profile table doesn't exist.

## Prerequisites & Installation

The skill ships as a set of Markdown files plus notebook templates that are loaded into the AI coding assistant's context at runtime. How you install them depends on which assistant you use.

Pick the section that matches your assistant:

- [Databricks Genie Code (Databricks Assistant)](#option-a--databricks-genie-code-databricks-assistant) — runs inside the Databricks workspace
- [Claude Code, Cursor, GitHub Copilot, Gemini CLI](#option-b--claude-code-cursor-github-copilot-gemini-cli) — local IDE / CLI assistants

### Option A — Databricks Genie Code

Genie Code loads context from your Databricks **Workspace** files. Install the skill once into your home folder and the assistant will pick it up on every session.

The expected layout, where `{user}` is your Databricks user folder name (typically your email address):

```
/Workspace/Users/{user}/
├── .assistant_instructions.md             # global agent instructions
└── .assistant/
    └── skills/
        ├── SKILL.md
        ├── 1-prep-and-clean-data.md
        ├── 2-profile-and-classify-series.md
        ├── 3-provision-forecasting-resources.md
        ├── 4-execute-mmf-forecast.md
        ├── 5-post-process-and-evaluate.md
        ├── mmf_local_notebook_template.ipynb
        ├── mmf_gpu_run_notebook_template.ipynb
        ├── mmf_gpu_orchestrator_notebook_template.ipynb
        ├── mmf_profiling_notebook_template.ipynb
        ├── mmf_prep_notebook_template.ipynb
        └── mmf_post_process_notebook_template.ipynb
```

There are two files / folders to upload:

1. **The skill bundle** — the **contents** of `[skills/databricks-skills/many-model-forecasting/](databricks-skills/many-model-forecasting/)` go directly into `/Workspace/Users/{user}/.assistant/skills/`.
2. **The agent instructions** — the contents of `[skills/assistant_instructions.md](assistant_instructions.md)` should be copied to `/Workspace/Users/{user}/.assistant_instructions.md` (note the leading dot and the `.md` extension; this file lives **at the root** of your user folder, not inside `.assistant/`).

Before uploading, edit the `### User Context` section near the bottom of `assistant_instructions.md` to set your workspace URL and email.

#### Upload via the Databricks CLI (recommended)

From a clone of this repo:

```bash
# Set your Databricks user folder name (usually your email)
USER_FOLDER="your-email@your-domain.com"

# 1) Upload the skill bundle directly into .assistant/skills/
#    (contents of many-model-forecasting/ land at the root of skills/)
databricks workspace import-dir \
  skills/databricks-skills/many-model-forecasting \
  /Users/${USER_FOLDER}/.assistant/skills \
  --overwrite

# 2) Upload the agent instructions to the user folder root
databricks workspace import \
  skills/assistant_instructions.md \
  /Users/${USER_FOLDER}/.assistant_instructions.md \
  --format AUTO --language MARKDOWN --overwrite
```

#### Upload via the Workspace UI

1. In the Databricks workspace, navigate to your user folder (`/Users/{user}/`).
2. Import every file from `skills/databricks-skills/many-model-forecasting/` directly into `.assistant/skills/`.
3. Import `skills/assistant_instructions.md` into your user folder root and rename it to `.assistant_instructions.md` simply copy and paste the content over to the exisiting `.assistant_instructions.md`.

#### Verify

Open Databricks Genie Code and ask: `What skills do you have access to?` — it should mention the Many-Model Forecasting skill and the five sub-skills. If not, double-check that the files are at the exact paths above (Genie Code is path-sensitive) and that the leading dots on `.assistant/` and `.assistant_instructions.md` were preserved.

Re-uploading is safe — it overwrites the existing files and picks up any updates to the skill.

### Option B — Claude Code, Cursor, GitHub Copilot, Gemini CLI

In this case, the skill calls Databricks MCP tools (e.g., `connect_to_workspace`, `execute_parameterized_sql`) provided by [ai-dev-kit](https://github.com/databricks-solutions/ai-dev-kit). Make sure the Databricks MCP server from `ai-dev-kit` is configured (or available) in your assistant before running any skill command.

These assistants read the skill files from your local project directory. Use the bundled `install.py` script — no need to clone the repo.

```bash
curl -O https://raw.githubusercontent.com/databricks-industry-solutions/many-model-forecasting/main/skills/install.py
python3 install.py --target /path/to/your-project
```

The installer downloads the skill files from GitHub automatically and configures your AI coding tools (Claude Code, Cursor, Gemini CLI, Copilot).

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
    ├── mmf_gpu_run_notebook_template.ipynb
    ├── mmf_gpu_orchestrator_notebook_template.ipynb
    ├── mmf_profiling_notebook_template.ipynb
    ├── mmf_prep_notebook_template.ipynb
    └── mmf_post_process_notebook_template.ipynb
```

Re-running the installer is safe — it updates existing configurations without duplication.

## Quickstart

Want to try the MMF agent end-to-end without bringing your own data? You can generate a synthetic time series dataset directly in your Databricks workspace and then point the agent at it.

1. Upload [`skills/synthetic_data_generation.ipynb`](synthetic_data_generation.ipynb) to your Databricks workspace.
2. Open the notebook, attach it to **serverless compute**, and run all cells. It will write a synthetic training table into the catalog/schema you specify in the notebook widgets.
3. Start a session with the MMF agent (Genie Code, Claude Code, Cursor, etc.) and run `/prep-and-clean-data`, pointing it at the table you just created. From there, continue through the rest of the 5-skill pipeline (`/profile-and-classify-series` → `/provision-forecasting-resources` → `/execute-mmf-forecast` → `/post-process-and-evaluate`).

This is the fastest way to see the full forecasting workflow run against a real Databricks workspace.

## Demo

The following recordings show the MMF Agent in action against a real Databricks workspace, driven from **Genie Code**. Start with the highlight reel for a feel of the end-to-end flow, then dive into the per-skill recordings for a closer look at each step.

### Highlight reel — end-to-end workflow

A short, sped-up walkthrough of the full 5-skill pipeline, from raw table to evaluated forecasts.

https://github.com/user-attachments/assets/3cda5e11-6dbb-41f7-aca0-6ed11c123fc5

### Step-by-step recordings

Each section below shows a detailed recording of a single skill, with commentary on what the agent is doing and the decisions the user is asked to make.

#### 1. `/prep-and-clean-data`

Discovers the source table, maps columns, and walks the user through imputation and anomaly handling. Produces `{use_case}_train_data`.

https://github.com/user-attachments/assets/7d31cc20-c91f-489e-a820-1f8ae254dd80

#### 2. `/profile-and-classify-series` *(optional)*

Computes statistical properties (ADF, STL, spectral entropy, SNR), classifies series by forecastability, recommends models, and asks how to handle non-forecastable series. Produces `{use_case}_series_profile` and `{use_case}_pipeline_config`.

https://github.com/user-attachments/assets/9a9d9e09-68fa-4fda-a2b8-545752ae9a83

#### 3. `/provision-forecasting-resources`

Asks which models to run, then sizes and provisions the appropriate CPU (16 or 32 vCPU) or GPU cluster — including a separate cluster for non-forecastable series if requested.

https://github.com/user-attachments/assets/9586f2a5-abfb-4f00-b255-dd177a2b87eb

#### 4. `/execute-mmf-forecast`

Generates the orchestrator and per-model run notebooks, then submits one Databricks job per model class (plus any non-forecastable jobs) and runs them in parallel. Produces `{use_case}_evaluation` and `{use_case}_scoring`.

https://github.com/user-attachments/assets/d7dd8ee6-8f2e-402b-9548-c386a663df0e

#### 5. `/post-process-and-evaluate`

Selects the best model per series, merges main-pipeline and non-forecastable results, and produces a business-ready summary with `forecast_source` provenance. Produces `{use_case}_best_models` and `{use_case}_evaluation_summary`.

https://github.com/user-attachments/assets/e03a61d5-4c88-4521-b151-fc55e0e7495a
