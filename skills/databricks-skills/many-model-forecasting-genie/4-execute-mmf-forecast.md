# Execute MMF Forecast

Validates parameters, asks the user about backtesting setup, generates notebooks
using the **orchestrator + run_gpu** pattern, creates **one job per model class**
(local, global, foundation), and triggers them **in parallel**.

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `use_case` | From Step 1 | Use case name — prefixes all output tables |
| `freq` | Auto-detected from data | Time series frequency (`D`, `W`, `M`, `H`) |
| `prediction_length` | Ask user | Forecast horizon in time steps |
| `backtest_length` | Derived from backtest strategy | Total historical points reserved for backtesting |
| `stride` | Derived from backtest strategy | Step size between backtest windows |
| `metric` | `smape` | Evaluation metric (`smape`, `mape`, `mae`, `mse`, `rmse`) |
| `active_models` | From Step 3 | Models to run (grouped by class) |
| `train_data` | `<catalog>.<schema>.{use_case}_train_data` | Input table — see Step 0a for routing |
| `evaluation_output` | `<catalog>.<schema>.{use_case}_evaluation_output` | Evaluation results table |
| `scoring_output` | `<catalog>.<schema>.{use_case}_scoring_output` | Scoring results table |
| `group_id` | `unique_id` | Column name for series identifier |
| `date_col` | `ds` | Column name for timestamp |
| `target` | `y` | Column name for target value |

## Available model names

**Local models (CPU):** `StatsForecastBaselineWindowAverage`, `StatsForecastBaselineSeasonalWindowAverage`, `StatsForecastBaselineNaive`, `StatsForecastBaselineSeasonalNaive`, `StatsForecastAutoArima`, `StatsForecastAutoETS`, `StatsForecastAutoCES`, `StatsForecastAutoTheta`, `StatsForecastAutoTbats`, `StatsForecastAutoMfles`, `StatsForecastTSB`, `StatsForecastADIDA`, `StatsForecastIMAPA`, `StatsForecastCrostonClassic`, `StatsForecastCrostonOptimized`, `StatsForecastCrostonSBA`, `SKTimeProphet`

**Global models (GPU):** `NeuralForecastRNN`, `NeuralForecastLSTM`, `NeuralForecastNBEATSx`, `NeuralForecastNHITS`, `NeuralForecastAutoRNN`, `NeuralForecastAutoLSTM`, `NeuralForecastAutoNBEATSx`, `NeuralForecastAutoNHITS`, `NeuralForecastAutoTiDE`, `NeuralForecastAutoPatchTST`

**Foundation models (GPU):** `ChronosBoltTiny`, `ChronosBoltMini`, `ChronosBoltSmall`, `ChronosBoltBase`, `Chronos2`, `Chronos2Small`, `Chronos2Synth`, `TimesFM_2_5_200m`

## Agent Rules (apply to this entire step)

- **Use Databricks platform tools first.** Create and trigger jobs using the Jobs API (`jobs/create`, `jobs/run-now`) or `runs/submit`. Do NOT use CLI commands or write job-creation code inline.
- **Do NOT run MMF pipeline code inline.** Notebook generation, model training, and scoring MUST happen inside a Databricks notebook on a cluster. Never run `mmf_sa` or model training code directly in the conversation.
- **Confirm before acting.** Before uploading a notebook, creating a job, or triggering a run, explain what you are about to do, show the configuration, and ask the user for confirmation.

## Steps

### Step 0a: Read non-forecastable strategy from pipeline config

```sql
SELECT non_forecastable_strategy, fallback_method, non_forecastable_models,
       n_forecastable, n_non_forecastable
FROM {catalog}.{schema}.{use_case}_pipeline_config
WHERE use_case = '{use_case}'
```

| Strategy | Main pipeline train table | Non-forecastable handling |
|----------|--------------------------|--------------------------|
| `include` (or config missing) | `{use_case}_train_data` | All series run together |
| `fallback` | `{use_case}_train_data_forecastable` | Already handled in Step 2 |
| `separate_job` | `{use_case}_train_data_forecastable` | Separate job created in Step 5a |

Set `{train_table}` to the correct table name for all subsequent steps.

### Step 1: Pre-flight parameter validation

| Validation | Rule | Error |
|-----------|------|-------|
| `backtest_length >= prediction_length` | Mandatory | "Backtest length must be >= prediction length" |
| `stride <= backtest_length` | Mandatory | "Stride must be <= backtest length" |
| `freq` ∈ {H, D, W, M} | Mandatory | "Unsupported frequency" |
| `active_models` all in model names list above | Mandatory | "Unknown model: {name}" |
| `train_table` exists and has required schema | Mandatory | "Training table missing or invalid schema" |

```sql
SELECT COUNT(*) AS count FROM {catalog}.{schema}.{train_table}
```

### ⛔ STOP GATE — Step 1b: Ask user about backtesting setup

**Always ask the user about their backtesting configuration. Do NOT proceed until the user confirms.**

Explain backtesting and ask:

> "Backtesting evaluates model accuracy by simulating forecasts on historical data.
> MMF slides a window across your history and produces a forecast at each position.
>
> Key parameters:
> - prediction_length: {prediction_length} (already set — your forecast horizon)
> - backtest_length: how much history to reserve for backtest windows
> - stride: how far to shift between windows (smaller = more windows = slower but more robust)
> - number of backtest windows = (backtest_length - prediction_length) / stride + 1
>
> How would you like to configure backtesting?
>
> (a) Quick validation — 1 backtest window (fastest)
>     → backtest_length = {prediction_length}, stride = {prediction_length}
>
> (b) Standard — 3 backtest windows (recommended)
>     → backtest_length = 3 × {prediction_length}, stride = {prediction_length}
>
> (c) Thorough — 5+ sliding windows (most robust)
>     → backtest_length = 5 × {prediction_length}, stride = {prediction_length}
>
> (d) Overlapping windows — dense evaluation
>     → backtest_length = 3 × {prediction_length}, stride = 1
>
> (e) Custom — specify backtest_length and stride manually"

**WAIT for the user to respond.**

### Step 2: Validate parameters

Confirm the input table exists and present all parameters to the user for final validation before generating notebooks.

### Step 3: Generate notebooks

**CRITICAL: Do NOT execute MMF pipeline code inline. All model training and scoring MUST run inside a Databricks notebook on a cluster. Generate notebooks from the templates and upload them to the workspace.**

**CRITICAL: Use templates from the `notebooks/` subfolder of this skill. Copy verbatim, only replacing `{placeholder}` tokens. Do NOT add, remove, or modify any other code.**

Generate a shared `run_id` (UUID) that will be passed to all notebooks.

#### 3a: Local models notebook

Generate from `notebooks/mmf_local_notebook_template.ipynb` with all local models in `{active_models}`.

##### Placeholder values for local notebook

| Placeholder | Value |
|-------------|-------|
| `{catalog}` | user's catalog |
| `{schema}` | user's schema |
| `{train_table}` | Determined by Step 0a |
| `{freq}` | detected or user-specified frequency |
| `{prediction_length}` | user-specified forecast horizon (integer) |
| `{backtest_length}` | derived from backtest strategy (integer) |
| `{stride}` | derived from backtest strategy (integer) |
| `{metric}` | `smape` (default) |
| `{active_models}` | Python list literal, e.g. `["StatsForecastAutoArima", "StatsForecastAutoETS"]` |
| `{group_id}` | `unique_id` |
| `{date_col}` | `ds` |
| `{target}` | `y` |
| `{use_case}` | use case name |

#### 3b: GPU run notebook (static — no placeholder substitution)

Copy `notebooks/mmf_gpu_run_notebook_template.ipynb` **as-is**. This notebook:
- Receives all parameters via `dbutils.widgets`
- Auto-detects whether the model is global or foundation and installs the correct `mmf_sa` extras
- Runs a single model per invocation
- Is called by the orchestrator notebooks via `dbutils.notebook.run()`

**Do NOT modify this template.**

#### 3c: GPU orchestrator notebooks (one per model class)

Generate from `notebooks/mmf_gpu_orchestrator_notebook_template.ipynb` for each GPU model class. Each orchestrator:
- Holds the list of active models for its class
- Loops through the models and calls `run_gpu` for each via `dbutils.notebook.run()`

> **Why orchestrator + run_gpu:** PyTorch allocates CUDA memory that cannot be freed within the same Python process. The `dbutils.notebook.run()` pattern gives each model a fresh kernel, avoiding OOM errors.

##### Placeholder values for orchestrator notebooks

| Placeholder | Value |
|-------------|-------|
| `{catalog}` | user's catalog |
| `{schema}` | user's schema |
| `{use_case}` | use case name |
| `{train_table}` | Determined by Step 0a |
| `{freq}` | detected or user-specified frequency |
| `{prediction_length}` | user-specified forecast horizon (integer) |
| `{backtest_length}` | derived from backtest strategy (integer) |
| `{stride}` | derived from backtest strategy (integer) |
| `{metric}` | `smape` (default) |
| `{active_models}` | Python list literal of models for this class only |
| `{num_nodes}` | `1` (single-node always) |

Generate up to two orchestrators:
- `orchestrator_global` — if any global models selected
- `orchestrator_foundation` — if any foundation models selected

### Step 4: Upload notebooks to workspace

#### ⛔ STOP GATE — Confirm before uploading

Before uploading, present a summary and ask the user:

> "I am about to upload the following notebooks to the workspace:
> {list each notebook with its full path and key parameters}
>
> Shall I proceed?"

**Do NOT upload until the user confirms.**

For each notebook, read the corresponding template from the `notebooks/` subfolder of this skill, replace the `{placeholder}` tokens with the actual values, and upload directly to the Databricks workspace. Use `{home_path}` = `/Workspace/Users/{current_user_email}`.

Upload these notebooks to `{home_path}/mmf-skills-test/notebooks/{use_case}/`:
- `04_run_local` — generated from `notebooks/mmf_local_notebook_template.ipynb` with all local models
- `04_run_gpu` — copied as-is from `notebooks/mmf_gpu_run_notebook_template.ipynb` (no placeholder substitution)
- `04_orchestrator_global` — generated from `notebooks/mmf_gpu_orchestrator_notebook_template.ipynb` (if any global models selected)
- `04_orchestrator_foundation` — generated from `notebooks/mmf_gpu_orchestrator_notebook_template.ipynb` (if any foundation models selected)

### Step 5: Create one job per model class (triggered in parallel)

**Create separate Workflow jobs for each model class and trigger them all in parallel.**

#### ⛔ STOP GATE — Confirm before creating and triggering jobs

Before creating any job, present a summary and ask the user:

> "I am about to create and trigger the following Databricks jobs:
> {list each job with name, notebook path, cluster type, and key parameters}
>
> All jobs will be triggered in parallel. Shall I proceed?"

**Do NOT create or trigger any job until the user confirms.**

#### ⛔ GUARDRAIL — Step 5a: Job ownership check

<!-- BUG IDENTIFIED: 2026-03-26 — lourdes.martinez@databricks.com
     PROBLEM: Job creation is idempotent by name — if a job with the same name exists
     (owned by ANY user), it silently returns that existing job. If owned by a different
     user, this causes wrong notebooks to run or permission errors.
     PROPOSED FIX: Always check job ownership before creating.
-->

**Before creating any job:**
1. Get the current user's identity (email).
2. Use job name pattern: `{use_case}_{class}_forecasting_{username}_{YYYYMMDD}`
   - Example: `synthetic_local_forecasting_lourdes.martinez_20260326`
3. Search for existing jobs with that name. If found, verify the creator matches the current user before reusing.

#### Job 1: Local models (if any local models selected)

```json
{
  "name": "{use_case}_local_forecasting",
  "tasks": [{
    "task_key": "local_models",
    "notebook_task": {
      "notebook_path": "{home_path}/mmf-skills-test/notebooks/{use_case}/04_run_local"
    },
    "job_cluster_key": "{use_case}_cpu_cluster"
  }],
  "job_clusters": [{
    "job_cluster_key": "{use_case}_cpu_cluster",
    "new_cluster": {
      "spark_version": "17.3.x-cpu-ml-scala2.13",
      "node_type_id": "{cpu_node_type}",
      "num_workers": {cpu_workers},
      "data_security_mode": "SINGLE_USER",
      "spark_conf": {
        "spark.sql.execution.arrow.enabled": "true",
        "spark.sql.adaptive.enabled": "false",
        "spark.databricks.delta.formatCheck.enabled": "false",
        "spark.databricks.delta.schema.autoMerge.enabled": "true"
      }
    }
  }]
}
```

#### Job 2: Global models (if any global models selected)

```json
{
  "name": "{use_case}_global_forecasting",
  "tasks": [{
    "task_key": "global_models",
    "notebook_task": {
      "notebook_path": "{home_path}/mmf-skills-test/notebooks/{use_case}/04_orchestrator_global"
    },
    "job_cluster_key": "{use_case}_gpu_cluster"
  }],
  "job_clusters": [{
    "job_cluster_key": "{use_case}_gpu_cluster",
    "new_cluster": {
      "spark_version": "18.0.x-gpu-ml-scala2.13",
      "node_type_id": "{gpu_node_type}",
      "num_workers": 0,
      "data_security_mode": "SINGLE_USER",
      "spark_conf": {
        "spark.master": "local[*]",
        "spark.databricks.cluster.profile": "singleNode",
        "spark.databricks.delta.formatCheck.enabled": "false",
        "spark.databricks.delta.schema.autoMerge.enabled": "true"
      },
      "custom_tags": {"ResourceClass": "SingleNode"}
    }
  }]
}
```

#### Job 3: Foundation models (if any foundation models selected)

Same pattern as Job 2 with `04_orchestrator_foundation` notebook.

**Important notes:**
- All clusters use `data_security_mode: "SINGLE_USER"` — ML runtimes reject `USER_ISOLATION`.
- GPU clusters are **always single-node** (`num_workers: 0`) with `spark.master: local[*]`, `spark.databricks.cluster.profile: singleNode`, and `custom_tags: {"ResourceClass": "SingleNode"}`. All three are required.
- Global and foundation jobs each get their own ephemeral GPU cluster so they run **in parallel**.

#### Serverless alternative (local models only)

```json
{
  "name": "{use_case}_local_forecasting",
  "tasks": [{
    "task_key": "local_models",
    "notebook_task": {"notebook_path": "{home_path}/mmf-skills-test/notebooks/{use_case}/04_run_local"},
    "environment_key": "Default"
  }],
  "environments": [{"environment_key": "Default", "spec": {"client": "1"}}]
}
```

GPU models require ML Runtime clusters and cannot use serverless.

#### Step 5a: Non-forecastable separate job (if `separate_job` strategy)

**Only create these jobs if `non_forecastable_strategy == 'separate_job'`.**

Generate notebooks using the same templates with:
- `{train_table}` → `{use_case}_train_data_non_forecastable`
- `{active_models}` → the non-forecastable models selected in Steps 2/3
- Output table suffix → `{use_case}_nf` (to avoid overwriting main pipeline outputs)

Upload NF notebooks as:
- `04_nf_run_local`, `04_nf_run_gpu`, `04_nf_orchestrator_global`, `04_nf_orchestrator_foundation` (as applicable)

Jobs follow the same pattern as the main pipeline, named `{use_case}_nf_local_forecasting`, `{use_case}_nf_global_forecasting`, `{use_case}_nf_foundation_forecasting`.

**Launch all non-forecastable jobs in parallel with the main pipeline jobs.**

### Step 6: Monitor execution

Poll all job run statuses until completion. Report structured status updates per job showing RUNNING / SUCCEEDED / FAILED states.

### Step 7: Log run metadata

```sql
CREATE OR REPLACE TABLE {catalog}.{schema}.{use_case}_run_metadata AS
SELECT
  '{use_case}' AS use_case,
  '{run_id}' AS run_id,
  CURRENT_TIMESTAMP() AS run_date,
  '{freq}' AS freq,
  {prediction_length} AS prediction_length,
  {backtest_length} AS backtest_length,
  {stride} AS stride,
  '{metric}' AS metric,
  '{active_models_str}' AS active_models,
  '{non_forecastable_strategy}' AS non_forecastable_strategy,
  '{train_table}' AS train_table
```

Confirm output tables were written:

```sql
SELECT COUNT(*) AS eval_rows FROM {catalog}.{schema}.{use_case}_evaluation_output
```
```sql
SELECT COUNT(*) AS score_rows FROM {catalog}.{schema}.{use_case}_scoring_output
```

### ⛔ STOP GATE — Step 8: Hand off to post-processing

Ask the user:

> "✅ Forecast run complete.
>
> Main pipeline (forecastable series):
> - Evaluation output: {eval_rows} rows in {use_case}_evaluation_output
> - Scoring output: {score_rows} rows in {use_case}_scoring_output
> - Run metadata logged to {use_case}_run_metadata
>
> Would you like to proceed to post-processing and evaluation?
> (a) Yes, proceed
> (b) No, stop here — I'll come back later"

**Do NOT proceed until the user responds.**

## Outputs

- Delta table `{use_case}_evaluation_output` — backtest metrics per model per series
- Delta table `{use_case}_scoring_output` — forward-looking forecasts
- Delta table `{use_case}_run_metadata` — run parameters and audit trail
- MLflow experiment at `/Users/{user}/mmf/{use_case}`
- (if `separate_job`) `{use_case}_nf_evaluation_output` and `{use_case}_nf_scoring_output`
