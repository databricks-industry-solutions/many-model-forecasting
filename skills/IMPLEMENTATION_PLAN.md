# MMF Agent Skill-Kit: Implementation Plan

## Overview

This document provides a step-by-step coding plan for building the MMF Agent Skill-Kit. The plan is organized into 7 phases, estimated at ~3-4 days of implementation. Each phase produces independently testable artifacts.

**Golden rule**: All changes go into `skills/databricks-skills/many-model-forecasting/`. Zero changes to `mmf_sa/`.

**Merge strategy**: The original 3-skill pipeline (`1-explore-data.md` → `2-setup-the-mmf-cluster.md` → `3-run-mmf.md`) is merged into the new 5-skill pipeline. Skills 1, 3, and 4 incorporate the original content verbatim and extend it. The old files are then removed.

**Asset naming convention**: At the start of the pipeline (Skill 1, Step 0), the agent collects a short **use case name** from the user (e.g., `m4`, `rossmann`). All Delta tables created by the pipeline are prefixed with this name: `{use_case}_train_data`, `{use_case}_cleaning_report`, `{use_case}_series_profile`, etc. This maps directly to `mmf_sa.run_forecast`'s `use_case_name` parameter and allows multiple forecasting projects to coexist in the same schema.

---

## Phase 0: Preparation (0.5 day)

### 0.1 Understand the reuse mapping

Each new file either supersedes an original or is entirely new:

| New file | Relationship to original |
|----------|------------------------|
| `1-prep-and-clean-data.md` | **Supersedes** `1-explore-data.md`. Includes Steps 1-5 verbatim, removes Step 6 (subsumed), renumbers Step 7→6, adds Steps 7-9. |
| `2-profile-and-classify-series.md` | **New** — no original to reuse. |
| `3-provision-forecasting-resources.md` | **Supersedes** `2-setup-the-mmf-cluster.md`. Includes Steps 1-5 verbatim, adds cluster reuse + UC check. |
| `4-execute-mmf-forecast.md` | **Supersedes** `3-run-mmf.md`. Includes Steps 1-7, params, model list verbatim, adds pre-flight validation + backtest strategy. GPU models run one-per-session (CUDA constraint). Analysis moved to Skill 5. |
| `5-post-process-and-evaluate.md` | **New** — automates `post-evaluation-analysis.ipynb` patterns. |

### 0.2 Create a working branch

```bash
git checkout -b feature/mmf-agent-skill-kit
```

### 0.3 Verify existing tests pass

```bash
cd skills/.test
uv run --extra dev python -m pytest tests/test_scorers.py -v
```

---

## Phase 1: Skill 1 — `prep_and_clean_data` (0.5 day)

### Step 1.1: Create `1-prep-and-clean-data.md`

**File**: `skills/databricks-skills/many-model-forecasting/1-prep-and-clean-data.md`

**Action**: Create a new markdown file by starting from the full content of `1-explore-data.md` and extending it.

1. **Copy the content of `1-explore-data.md`** as the starting point. This provides workspace connection, table discovery, column type matching, profiling queries, frequency detection, user validation, and training data creation.

2. **Update the header** to reflect the new name:
   ```markdown
   # Prep and Clean Data

   **Slash command:** `/prep-and-clean-data <catalog> <schema>`

   Collects a use case name, connects to a Databricks workspace, discovers time
   series tables, maps columns to the MMF schema (`unique_id`, `ds`, `y`),
   applies automated cleaning (imputation, anomaly capping), and creates the
   `{use_case}_train_data` table ready for forecasting.
   ```

3. **Prepend Step 0: Collect use case name** — before any data exploration, ask the user for a short use case identifier via `AskUserQuestion`:
   - Prompt: "Provide a short use case name (e.g., m4, rossmann, retail_sales). This will prefix all tables and assets created by the pipeline."
   - Validate: lowercase alphanumeric + underscores only, 1-30 characters, cannot start with a number
   - Store as `{use_case}` for all subsequent table names

4. **Keep Steps 1-5 exactly as they are** in `1-explore-data.md`. Every SQL query, every `AskUserQuestion`, every threshold — identical.

5. **Remove original Step 6** (Data quality checks). Its missing-value diagnostics are subsumed by Step 7 (missing data assessment & imputation). Its negative-value diagnostics are subsumed by Step 8 (IQR-based anomaly detection). A read-only diagnostic before actionable cleaning steps adds no value.

6. **Renumber original Step 7 as Step 6** (Create `{use_case}_train_data`) with the existing daily/weekly/monthly SQL variants, renaming the output table from `mmf_train_data` to `{use_case}_train_data`. This creates the base table; Steps 7-8 clean it in-place.

7. **Append Step 7**: Missing Data Assessment & Imputation (sub-steps 7a-7c)
   - **7a**: Generate a date spine per series using `SEQUENCE(min_ds, max_ds, INTERVAL)` at the detected `{freq}` and `LEFT JOIN` actual data. This detects both explicit NULLs and implicit gaps (missing rows) in a single pass. Use `INTERVAL 1 HOUR` for hourly, `INTERVAL 1 DAY` for daily, `INTERVAL 7 DAY` for weekly, `INTERVAL 1 MONTH` for monthly.
   - **7b**: Present summary to the user with categorized breakdown (counts based on spine, not just existing rows):
     - Complete series (no gaps)
     - `< 5%` missing → suggest linear interpolation
     - `5-20%` missing → suggest forward fill
     - `> 20%` missing → suggest exclusion
   - `AskUserQuestion` with options:
     - (a) Apply suggested strategy
     - (b) Use a single strategy for all series (interpolation / forward fill / fill with 0 / drop nulls)
     - (c) Skip imputation
     - (d) Adjust the exclusion threshold
   - **7c**: Replace `{use_case}_train_data` with the spine-joined version (making implicit gaps into explicit NULL rows), then apply chosen imputation via SQL window functions (`LAG`/`LEAD` for interpolation, `LAST_VALUE(... IGNORE NULLS)` for forward fill, `COALESCE(y, 0)` for zero-fill)
   - Exclude series above threshold from `{use_case}_train_data`
   - Log imputed counts and excluded series for the cleaning report

8. **Append Step 8**: Anomaly Detection & Capping
   - SQL query using `PERCENTILE` to compute Q1, Q3, IQR per series from `{use_case}_train_data`
   - Flag values outside `[Q1 - 1.5×IQR, Q3 + 1.5×IQR]` for detection/reporting
   - Present anomaly summary to the user (affected series, total anomaly count, ranges)
   - `AskUserQuestion` with options for capping range:
     - (a) Cap at 1.5×IQR (moderate — default)
     - (b) Cap at 3.0×IQR (conservative — extreme outliers only)
     - (c) Custom multiplier (user enters a value)
     - (d) Skip anomaly capping
   - If user confirms, apply capping via `UPDATE` with `CASE WHEN` using the user-chosen `{iqr_multiplier}`
   - Log capped counts per series for the cleaning report

9. **Append Step 9**: Create Cleaning Report
   ```sql
   CREATE OR REPLACE TABLE {catalog}.{schema}.{use_case}_cleaning_report AS ...
   ```
   - Columns: `unique_id`, `original_count`, `final_count`, `missing_filled`, `imputation_method`, `anomalies_capped`, `iqr_multiplier`, `excluded`, `exclusion_reason`
   - Present summary to user

10. **Update Outputs section** to include both `{use_case}_train_data` and `{use_case}_cleaning_report`

**All cleaning logic is expressed as SQL** executed via `execute_parameterized_sql` — consistent with the original `1-explore-data.md` pattern.

### Step 1.2: Validate the skill document

- All SQL queries syntactically valid
- `{placeholder}` tokens consistent with existing convention (now including `{use_case}`)
- `{use_case}_train_data` output schema exactly: `unique_id` (STRING), `ds` (TIMESTAMP), `y` (DOUBLE)
- Step 0 collects use case name before any table discovery
- `AskUserQuestion` at both cleaning steps (7 and 8) with suggested defaults and user override options
- Steps 1-5 byte-for-byte identical to `1-explore-data.md`
- Original Step 6 removed (not present in the new skill document)

---

## Phase 2: Skill 2 — `profile_and_classify_series` (1 day)

This is entirely new — no original file to reuse. Requires both a markdown step-file and a notebook template.

### Step 2.1: Create `mmf_profiling_notebook_template.ipynb`

**File**: `skills/databricks-skills/many-model-forecasting/mmf_profiling_notebook_template.ipynb`

**Action**: Create a Jupyter notebook. Use the exact same cell structure as `mmf_local_notebook_template.ipynb` (markdown title → `%pip install` → `%restart_python` → parameters → logic → output).

**Cell 0** (markdown):
```
# MMF Series Profiling
Computes statistical properties for each time series and classifies them.
```

**Cell 1** (code — pip install):
```python
%pip install statsmodels scipy
%restart_python
```

**Cell 2** (code — parameters with `{placeholder}` tokens):
```python
catalog = "{catalog}"
schema = "{schema}"
use_case = "{use_case}"
train_table = "{train_table}"
freq = "{freq}"
prediction_length = {prediction_length}
```

**Cell 3** (code — load data):
```python
df = spark.table(f"{catalog}.{schema}.{train_table}")
print(f"Loaded {df.count()} rows for use case: {use_case}")
```

**Cell 4** (code — define `profile_series` UDF):

Computes per-series: ADF p-value, seasonality strength (STL), trend strength (STL), spectral entropy, lag-1 autocorrelation, SNR, sparsity, CV. Full implementation in `AGENT_DESIGN_SPEC.md` Phase 2 Cell 4.

**Cell 5** (code — apply profiling via `groupby().applyInPandas()`):

Define `output_schema` with `StructType`, apply `profile_series` over `unique_id` groups.

**Cell 6** (code — classify and recommend):

`forecastability_class` assignment (high_confidence / low_signal) based on thresholds. Model recommendation logic mapping characteristics → model names. Derive `model_types_needed` from recommended model names.

**Cell 7** (code — write output):

Write to `{catalog}.{schema}.{use_case}_series_profile` Delta table. Print summary.

### Step 2.2: Create `2-profile-and-classify-series.md`

**File**: `skills/databricks-skills/many-model-forecasting/2-profile-and-classify-series.md`

**Action**: Write a markdown skill document following the pattern established by `3-run-mmf.md` (parameter table, placeholder table, step-by-step with SQL and MCP tools):

1. **Step 1**: Verify `{use_case}_train_data` exists (SQL via MCP)
2. **Step 2**: Gather parameters (`catalog`, `schema`, `use_case`, `freq`, `prediction_length`) via `AskUserQuestion`
3. **Step 3**: Generate notebook from `mmf_profiling_notebook_template.ipynb` by replacing `{placeholder}` tokens. **CRITICAL: Copy the template VERBATIM** — same rule as `3-run-mmf.md` Step 2.
4. **Step 4**: Upload notebook to workspace via MCP `upload_notebook`
5. **Step 5**: Create a single-task Workflow job on the **CPU cluster** (profiling is CPU-bound, use same CPU cluster spec from `2-setup-the-mmf-cluster.md`)
6. **Step 6**: Run job via `run_job` and monitor via `get_job_run`
7. **Step 7**: Query `{use_case}_series_profile` and present classification summary to user
8. **Step 8**: Present model recommendations via `AskUserQuestion` for confirmation

Include the full parameter table, placeholder mapping table, and output schema.

---

## Phase 3: Skill 3 — `provision_forecasting_resources` (0.5 day)

### Step 3.1: Create `3-provision-forecasting-resources.md`

**File**: `skills/databricks-skills/many-model-forecasting/3-provision-forecasting-resources.md`

**Action**: Create a markdown file by starting from the full content of `2-setup-the-mmf-cluster.md` and extending it.

1. **Copy the entire content of `2-setup-the-mmf-cluster.md`** as the starting point.

2. **Update the header**:
   ```markdown
   # Provision Forecasting Resources

   **Slash command:** `/provision-forecasting-resources`

   Determines required cluster types (from profiling or user input), configures
   clusters with the correct specs, verifies Unity Catalog enablement, and
   optionally reuses or restarts existing clusters.
   ```

3. **Enhance Step 1** (Determine model classes):
   - Add a preamble that checks for `{use_case}_series_profile`:
     ```sql
     SELECT DISTINCT model_types_needed
     FROM {catalog}.{schema}.{use_case}_series_profile
     WHERE forecastability_class = 'high_confidence'
     ```
   - In both cases (profile exists or not), present a multi-select `AskUserQuestion` showing all three model classes with their full model lists:
     - **Local models (CPU):** all 17 StatsForecast + SKTimeProphet models
     - **Global models (GPU):** all 10 NeuralForecast models
     - **Foundation models (GPU):** all 8 Chronos + TimesFM models
   - If profile exists → pre-select the auto-detected classes as a suggestion
   - If profile does NOT exist → no pre-selection; user chooses freely
   - User can select any combination (local, global, foundation, or all three)

4. **Keep Step 2 verbatim** (Determine cloud provider — `AskUserQuestion`)

5. **Insert new Step 3**: Check existing clusters
   - Use `list_clusters` MCP tool
   - Match by runtime version (`17.3.x-cpu-ml-scala2.13` or `18.0.x-gpu-ml-scala2.13`)
   - Decision logic:
     - RUNNING → "Reuse?"
     - TERMINATED → "Start it?" (use `start_cluster`)
     - None found → proceed to ephemeral config

6. **Keep original Step 3 as Step 4** (Select cluster configuration — CPU/GPU tables, decision logic). Enhance with user-selectable worker counts:
   - **CPU cluster**: Query `COUNT(DISTINCT unique_id)` from `{use_case}_train_data`, suggest workers based on series count (<100→0 single-node, 100-1K→4, 1K-10K→6, 10K-100K→8, >100K→10). `AskUserQuestion` to confirm or override.
   - **GPU cluster**: Always single-node (0 workers). `AskUserQuestion` to let user choose the GPU instance type (controls number of GPUs: 1, 2, 4, or 8 per cloud provider).

7. **Insert new Step 5**: UC enablement check
   - Verify `data_security_mode` is `USER_ISOLATION` or `SINGLE_USER`
   - If missing → warn user, suggest adding Spark config

8. **Keep original Step 4 as Step 6** (Present and validate — `AskUserQuestion`)

9. **Keep original Step 5 as Step 7** (Save configuration)

10. **Keep MMF Installation section and Outputs section verbatim**

---

## Phase 4: Skill 4 — `execute_mmf_forecast` (0.75 day)

### Step 4.0: Modify `mmf_gpu_notebook_template.ipynb`

**File**: `skills/databricks-skills/many-model-forecasting/mmf_gpu_notebook_template.ipynb`

**Why**: The current template passes the full `active_models` list to a single `run_forecast()` call. This fails because PyTorch cannot flush CUDA memory between models within the same Python process. The official example notebooks (`examples/monthly/global_monthly.ipynb`, `examples/daily/global_daily.ipynb`) solve this by running each GPU model in a separate session via `dbutils.notebook.run()`. The template must be updated to accept a single model.

**Changes**:

1. **Cell 3** — Replace `active_models = {active_models}` with single-model parameters:
   ```python
   model = "{model}"
   run_id = "{run_id}"
   num_nodes = {num_nodes}
   ```
   Remove the `active_models` line.

2. **Cell 4** — Update `run_forecast()` call:
   - Change `active_models=active_models` → `active_models=[model]`
   - Add `run_id=run_id` (groups results across separate GPU sessions)
   - Add `model_output=catalog + "." + schema` (matches example pattern)
   - Add `num_nodes=num_nodes` (enables multi-node GPU clusters)
   - Change output table names to use `{use_case}` prefix:
     - `scoring_output=catalog + "." + schema + ".{use_case}_scoring_output"`
     - `evaluation_output=catalog + "." + schema + ".{use_case}_evaluation_output"`
   - Change experiment path to `"/Users/" + user + "/mmf/{use_case}"`
   - Change `use_case_name="{use_case}"`

### Step 4.1: Create `4-execute-mmf-forecast.md`

**File**: `skills/databricks-skills/many-model-forecasting/4-execute-mmf-forecast.md`

**Action**: Create a markdown file by starting from the full content of `3-run-mmf.md` and extending it.

1. **Copy the entire content of `3-run-mmf.md`** as the starting point.

2. **Update the header**:
   ```markdown
   # Execute MMF Forecast

   **Slash command:** `/execute-mmf-forecast <catalog> <schema>`

   Validates parameters, generates and submits Many Models Forecasting notebooks
   to Databricks, monitors execution, and logs run metadata.
   ```

3. **Keep the Parameters table verbatim** from `3-run-mmf.md`.

4. **Keep the Available model names section verbatim** from `3-run-mmf.md`.

5. **Prepend new Step 1**: Pre-flight parameter validation
   - Validation rules (backtest_length >= prediction_length, stride, freq, model names)
   - Table existence check: verify `{use_case}_train_data` exists and has required schema via SQL
   - Model type vs cluster type consistency warning

6. **Prepend new Step 1a**: Profile-aware model selection (optional)
   - Check if `{use_case}_series_profile` exists
   - If yes → query recommended models, propose as `active_models`, `AskUserQuestion`
   - If no → skip (use default or user-specified `active_models`)

7. **Prepend new Step 1b**: Define backtest strategy
   - After `prediction_length` is known, present a visual explanation of how backtesting works (sliding window diagram)
   - `AskUserQuestion` with five strategy options:
     - (a) Quick validation: 1 window (`backtest_length = prediction_length`, `stride = prediction_length`)
     - (b) Standard: 3 windows (`backtest_length = 3 × prediction_length`, `stride = prediction_length`) — recommended default
     - (c) Thorough: 5 windows (`backtest_length = 5 × prediction_length`, `stride = prediction_length`)
     - (d) Overlapping: dense evaluation (`backtest_length = 3 × prediction_length`, `stride = 1`)
     - (e) Custom: user enters `backtest_length` and `stride` manually
   - Derive and validate parameters, report resulting number of backtest windows

8. **Keep original Step 1 as Step 2** (Gather and validate parameters) — verbatim.

9. **Extend original Step 2 as Step 3** (Generate notebooks):
    - Keep the **CRITICAL** verbatim template copy rule
    - Generate a shared `run_id` (UUID) for the entire run
    - **Local models**: single notebook from `mmf_local_notebook_template.ipynb` with all local models in `{active_models}` (unchanged behavior)
    - **GPU models (CHANGED)**: generate **one notebook per GPU model** from `mmf_gpu_notebook_template.ipynb`. Each gets:
      - `{model}` = single model name (e.g. `NeuralForecastAutoNHITS`)
      - `{pip_extras}` = `[global]` for `NeuralForecast*` models, `[foundation]` for others
      - `{run_id}` = the shared UUID
      - `{num_nodes}` = `1` (single-node default)
      - `{use_case}` = use case name (for output table names and experiment path)
    - This is required because **PyTorch GPU models cannot flush CUDA memory within the same process** — running multiple GPU models sequentially in one session causes OOM errors. The official MMF example notebooks (`examples/monthly/global_monthly.ipynb`) use the same one-model-per-session pattern via `dbutils.notebook.run()`.
    - Update placeholder table to include `{model}`, `{run_id}`, `{num_nodes}`, `{use_case}` for GPU notebooks

10. **Extend original Step 3 as Step 4** (Upload notebooks):
    - Upload `notebooks/{use_case}/run_local` — single local notebook
    - Upload `notebooks/{use_case}/run_{model_name}` — one per GPU model

11. **Extend original Step 4 as Step 5** (Create Workflow job):
    - **Local models task**: single task (unchanged)
    - **GPU model tasks (CHANGED)**: one task per GPU model, **chained sequentially** via `depends_on` to prevent CUDA memory conflicts on the shared GPU cluster
    - Example task chain: `gpu_NeuralForecastAutoNHITS` → `gpu_ChronosBoltBase` → `gpu_Chronos2`
    - The local models task and the first GPU task can run in parallel (different clusters)
    - Ephemeral job cluster specs — verbatim from `3-run-mmf.md` (with `{use_case}` prefixed names)
    - Serverless alternative — applicable to local models only (GPU models require ML Runtime clusters)

12. **Enhance original Step 5 as Step 6** (Monitor execution):
    - Base polling behavior unchanged
    - Add structured status logging with per-model-task progress reporting

10. **Simplify original Step 6 as Step 7** (Log run metadata):
    - **Remove** the three analysis SQL queries (best model, avg metric, worst series) — these are now handled exclusively by Skill 5
    - **Keep only**: write run metadata to `{use_case}_run_metadata` table
    - Query and report row counts for `{use_case}_evaluation_output` and `{use_case}_scoring_output`

11. **Replace original Step 7 with Step 8** (Hand off to post-processing):
    - Present completion summary (row counts, table names, MLflow experiment path)
    - Direct user to run Skill 5 (`/post-process-and-evaluate`) for detailed results analysis
    - Add a note explaining that all analysis (best model selection, model ranking, metric summaries, business reporting) is consolidated in Skill 5

12. **Update Outputs section**: all output tables now prefixed with `{use_case}` — `{use_case}_evaluation_output`, `{use_case}_scoring_output`, `{use_case}_run_metadata`

---

## Phase 5: Skill 5 — `post_process_and_evaluate` (0.5 day)

### Step 5.1: Create `5-post-process-and-evaluate.md`

**File**: `skills/databricks-skills/many-model-forecasting/5-post-process-and-evaluate.md`

**Action**: Create a markdown skill document (entirely new, no original to start from):

1. **Step 1**: Verify output tables exist (SQL via MCP)

2. **Step 2**: Compute multi-metric evaluation
   - Average primary metric per model per series
   - WAPE calculation using Spark SQL higher-order functions (`TRANSFORM`, `AGGREGATE`, `ARRAYS_ZIP`)

3. **Step 3**: Best model selection per series
   - Reuse SQL pattern from `post-evaluation-analysis.ipynb` Cell 7
   - Create `{use_case}_best_models` table

4. **Step 4**: Model ranking
   - Reuse SQL pattern from `post-evaluation-analysis.ipynb` Cell 9
   - Add percentage column

5. **Step 5**: Create `{use_case}_evaluation_summary` table

6. **Step 6**: Business-ready report (present structured summary to user)

7. **Step 7**: Cross-reference with profiling (conditional — only if `{use_case}_series_profile` exists)

8. **Step 8**: Suggest next steps (decision tree based on results quality)

All queries use `execute_parameterized_sql` MCP tool. No notebook template needed.

---

## Phase 6: Update Supporting Files (0.25 day)

### Step 6.1: Rewrite `SKILL.md`

**File**: `skills/databricks-skills/many-model-forecasting/SKILL.md`

**Changes**: Replace the old 3-skill workflow diagram and skill descriptions with the new 5-skill pipeline. Keep the YAML frontmatter, Overview, Prerequisites, and Cluster Configurations sections largely unchanged. Specifically:

- Replace the workflow diagram with the 5-step pipeline
- Replace Skill 1/2/3 descriptions with Skill 1-5 descriptions
- Update "See:" links to point to new file names
- Add `mmf_profiling_notebook_template.ipynb` to the templates section

### Step 6.2: Update `install.py`

**File**: `skills/install.py`

**Replace** the `SKILL_FILES` list:

```python
SKILL_FILES = [
    "SKILL.md",
    "1-prep-and-clean-data.md",
    "2-profile-and-classify-series.md",
    "3-provision-forecasting-resources.md",
    "4-execute-mmf-forecast.md",
    "5-post-process-and-evaluate.md",
    "mmf_local_notebook_template.ipynb",
    "mmf_gpu_notebook_template.ipynb",
    "mmf_profiling_notebook_template.ipynb",
]
```

Update `SKILL_REFERENCE_BLOCK` and `CURSOR_RULE_CONTENT` to reference the new files:

```python
SKILL_REFERENCE_BLOCK = """\
## Many-Model Forecasting Skill

This project includes the Many-Model Forecasting (MMF) skill for Databricks.
Read these files to learn the patterns before starting any forecasting task:

- `databricks-skills/many-model-forecasting/SKILL.md` — overview and workflow
- `databricks-skills/many-model-forecasting/1-prep-and-clean-data.md` — data discovery, quality checks, and cleaning
- `databricks-skills/many-model-forecasting/2-profile-and-classify-series.md` — series profiling and classification
- `databricks-skills/many-model-forecasting/3-provision-forecasting-resources.md` — cluster setup and provisioning
- `databricks-skills/many-model-forecasting/4-execute-mmf-forecast.md` — running the forecasting pipeline
- `databricks-skills/many-model-forecasting/5-post-process-and-evaluate.md` — post-processing and evaluation
- `databricks-skills/many-model-forecasting/mmf_local_notebook_template.ipynb` — local notebook template
- `databricks-skills/many-model-forecasting/mmf_gpu_notebook_template.ipynb` — GPU notebook template
- `databricks-skills/many-model-forecasting/mmf_profiling_notebook_template.ipynb` — profiling notebook template
"""
```

### Step 6.3: Update root `CLAUDE.md`

**File**: `CLAUDE.md` (root)

**Replace** the content within the `<!-- mmf-dev-kit:skills -->` marker block with references to the new file names (same list as `SKILL_REFERENCE_BLOCK`).

### Step 6.4: Remove retired files

**Delete** the original skill files that are now superseded:

```bash
rm skills/databricks-skills/many-model-forecasting/1-explore-data.md
rm skills/databricks-skills/many-model-forecasting/2-setup-the-mmf-cluster.md
rm skills/databricks-skills/many-model-forecasting/3-run-mmf.md
```

---

## Phase 7: Testing (0.5 day)

### Step 7.1: Update `manifest.yaml`

**File**: `skills/.test/skills/many-model-forecasting/manifest.yaml`

**Changes**:

1. Add new triggers:
   ```yaml
   triggers:
     # ... existing triggers ...
     - prep and clean data
     - profile series
     - classify series
     - provision cluster
     - provision resources
     - execute forecast
     - post evaluation
     - evaluate forecast
   ```

2. Update trace expectations for the merged skills:
   ```yaml
   trace_expectations:
     tool_limits:
       # ... existing limits ...
       mcp__databricks-v2__list_clusters: 5
       mcp__databricks-v2__start_cluster: 3
       mcp__databricks-v2__create_cluster: 3
   ```

3. Update `default_guidelines` to reference new file names.

### Step 7.2: Update `ground_truth.yaml`

**File**: `skills/.test/skills/many-model-forecasting/ground_truth.yaml`

**Add expected facts** for the merged / new skills:

For Skill 1 (merged):
- Must collect `use_case` name before any data exploration (Step 0)
- Must include the original explore-data SQL (SHOW TABLES, DESCRIBE TABLE, frequency detection)
- Must NOT include original Step 6 (read-only quality diagnostic) — removed
- Step 7 (imputation): must present missing data summary and `AskUserQuestion` with options (suggested strategy / single strategy / skip / adjust threshold)
- Step 8 (anomaly): must present anomaly summary and `AskUserQuestion` for capping range (1.5×IQR / 3.0×IQR / custom / skip)
- Cleaning report must include `imputation_method` and `iqr_multiplier` columns
- Output table named `{use_case}_train_data` with schema: `unique_id` (STRING), `ds` (TIMESTAMP), `y` (DOUBLE)

For Skill 2 (new):
- Must accept `use_case` parameter and write to `{use_case}_series_profile`
- Profiling must compute: `adf_pvalue`, `seasonality_strength`, `spectral_entropy`
- Classification: exactly two classes (`high_confidence`, `low_signal`)
- Model recommendations: only names from `models_conf.yaml`

For Skill 3 (merged):
- Must include the original cluster specs (runtime, node types)
- Must check for UC enablement

For Skill 4 (merged):
- Must include the original notebook generation rules (VERBATIM template copy)
- Must generate one notebook per GPU model (not one per model class) due to CUDA memory constraint
- Must create one Workflow task per GPU model, chained sequentially via `depends_on`
- Must pass a shared `run_id` to all GPU notebook tasks
- Must include pre-flight validation against `{use_case}_train_data`
- Must include run metadata table (`{use_case}_run_metadata`)
- All output table names must use `{use_case}` prefix

For Skill 5 (new):
- Must accept `use_case` parameter and prefix all table names
- Best model selection SQL must use `RANK()` window function into `{use_case}_best_models`
- WAPE calculation must use `ARRAYS_ZIP` / `AGGREGATE`
- Evaluation summary written to `{use_case}_evaluation_summary`

### Step 7.3: Add Tier 1 tests

**File**: `skills/.test/tests/tier1/test_profile_series.py`

Test that the profiling notebook template:
- Contains all required `{placeholder}` tokens
- Defines the `profile_series` function with correct return schema
- Writes to `{use_case}_series_profile` table

**File**: `skills/.test/tests/tier1/test_post_evaluate.py`

Test that the post-evaluation skill:
- Contains the correct SQL for best model selection
- Creates `{use_case}_best_models` and `{use_case}_evaluation_summary` tables
- Cross-references with `{use_case}_series_profile` when available

### Step 7.4: Update existing Tier 1 tests

Update references in existing tests to point to new file names:
- `test_explore_data.py` → update to reference `1-prep-and-clean-data.md`
- `test_setup_cluster.py` → update to reference `3-provision-forecasting-resources.md`
- `test_run_mmf.py` → update to reference `4-execute-mmf-forecast.md`

### Step 7.5: Run full test suite

```bash
cd skills/.test

# Unit tests
uv run --extra dev python -m pytest tests/test_scorers.py -v

# Tier 1 agent tests (with DuckDB mock)
uv run --extra tier1 python -m pytest tests/tier1/ -v

# Skill evaluation
uv run --extra dev python scripts/run_eval.py many-model-forecasting
```

---

## Implementation Order & Dependencies

```
Phase 0: Preparation
    │
    ├── Phase 1: Skill 1 — prep_and_clean_data
    │     Start from 1-explore-data.md, append cleaning steps
    │
    ├── Phase 2: Skill 2 — profile_and_classify_series
    │     Entirely new (notebook template + step-file)
    │
    ├── Phase 3: Skill 3 — provision_forecasting_resources
    │     Start from 2-setup-the-mmf-cluster.md, add cluster reuse + UC check
    │
    ├── Phase 4: Skill 4 — execute_mmf_forecast
    │     Modify GPU template (single-model-per-session), start from 3-run-mmf.md,
    │     prepend validation, one task per GPU model, hand off analysis to Skill 5
    │
    ├── Phase 5: Skill 5 — post_process_and_evaluate
    │     Entirely new (SQL-only step-file)
    │
    ├── Phase 6: Update supporting files + remove retired originals
    │
    └── Phase 7: Testing
```

Phases 1-5 can be **written** in parallel since they are independent markdown files. The runtime dependency (each skill's output feeds the next) only matters during execution.

---

## File Checklist

| # | File | Action | Phase | Starts From |
|---|------|--------|-------|-------------|
| 1 | `1-prep-and-clean-data.md` | Create | 1 | Copy of `1-explore-data.md` (minus Step 6) + Steps 7-9 |
| 2 | `2-profile-and-classify-series.md` | Create | 2 | Written from scratch |
| 3 | `mmf_profiling_notebook_template.ipynb` | Create | 2 | Written from scratch |
| 4 | `3-provision-forecasting-resources.md` | Create | 3 | Copy of `2-setup-the-mmf-cluster.md` + Steps 3,5 |
| 5 | `mmf_gpu_notebook_template.ipynb` | **Modify** | 4 | Single-model-per-session: `{model}`, `{run_id}`, `{num_nodes}` |
| 6 | `4-execute-mmf-forecast.md` | Create | 4 | Copy of `3-run-mmf.md` + Steps 1,1a,1b; one task per GPU model |
| 7 | `5-post-process-and-evaluate.md` | Create | 5 | Written from scratch |
| 8 | `SKILL.md` | Rewrite | 6 | — |
| 9 | `install.py` | Update | 6 | — |
| 10 | `CLAUDE.md` (root) | Update | 6 | — |
| 11 | `1-explore-data.md` | **Delete** | 6 | — |
| 12 | `2-setup-the-mmf-cluster.md` | **Delete** | 6 | — |
| 13 | `3-run-mmf.md` | **Delete** | 6 | — |
| 14 | `manifest.yaml` | Update | 7 | — |
| 15 | `ground_truth.yaml` | Update | 7 | — |
| 16 | `test_profile_series.py` | Create | 7 | — |
| 17 | `test_post_evaluate.py` | Create | 7 | — |
| 18 | Existing tier 1 tests | Update refs | 7 | — |

**Total new files**: 6 (5 skill docs + 1 notebook template)
**Total modified files**: 6 (`mmf_gpu_notebook_template.ipynb`, `SKILL.md`, `install.py`, `CLAUDE.md`, `manifest.yaml`, `ground_truth.yaml`)
**Total deleted files**: 3 (original skill docs)
**Total updated test files**: ~3 (existing tier 1 tests with renamed references)
**Files in `mmf_sa/`**: 0 (zero changes to core library)

---

## Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| Profiling UDF fails on large datasets | Add `.limit(10000)` per series in profiling; document memory requirements |
| STL decomposition fails on short series | Try-except with fallback to 0.0 for all STL-based metrics |
| WAPE SQL with higher-order functions not supported on older runtimes | Provide fallback SQL using `LATERAL VIEW EXPLODE` |
| `models_conf.yaml` model names change in future `mmf_sa` versions | Validate model names at runtime in Skill 4 pre-flight check |
| User skips profiling (Skill 2) | Skills 3 and 4 gracefully fall back: Skill 3 asks user for model types, Skill 4 uses default `active_models` |
| Invalid use case name | Step 0 validates: lowercase alphanumeric + underscores, 1-30 chars, no leading digits. Re-prompt on failure |
| Existing tests break after file renames | Phase 7 Step 7.4 explicitly updates all test references |
| Consumers of the old `install.py` get stale files | `install.py` is updated in Phase 6 to ship only new files |
| CUDA OOM when running multiple GPU models | GPU template runs one model per session; Skill 4 creates one task per GPU model, chained sequentially via `depends_on`. Matches the official example pattern (`examples/monthly/global_monthly.ipynb`) |
| Many GPU models → many sequential tasks → long total runtime | Expected; user is warned at Step 5 with estimated task count. Can be parallelized with multiple GPU clusters (future enhancement) |

---

## Definition of Done

Each skill is considered complete when:

1. **Reuse verified**: For Skills 1, 3, 4 — every step that came from the original file is preserved (diff only shows additions and table name changes from `mmf_*` to `{use_case}_*`). Exception: original Step 6 is intentionally removed from Skill 1 (subsumed by Steps 7 and 8).
2. **Pattern followed**: The markdown skill document follows the existing pattern (numbered steps, SQL blocks, `AskUserQuestion` checkpoints). Cleaning steps (7 and 8) must each have an `AskUserQuestion` with suggested defaults and user override options.
3. **Templates correct**: Any notebook templates use `{placeholder}` substitution identical to existing `mmf_local_notebook_template.ipynb`
4. **Schemas documented**: Output table schemas listed with column names and types
5. **Graceful fallbacks**: Each skill handles missing upstream outputs (e.g., no `{use_case}_series_profile`) by falling back to manual/default behavior
6. **Old files removed**: The three original files are deleted; no references to them remain in `SKILL.md`, `install.py`, `CLAUDE.md`, or tests
7. **Tests pass**: `test_scorers.py`, tier 1 tests, `run_eval.py` all green
8. **Installer works**: `install.py` copies exactly the new files to target projects
