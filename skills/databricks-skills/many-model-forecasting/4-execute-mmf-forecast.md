# Execute MMF Forecast

**Slash command:** `/execute-mmf-forecast <catalog> <schema>`

Validates parameters, generates and submits Many Models Forecasting notebooks
to Databricks, monitors execution, and logs run metadata.

## Parameters

Gather these from the user (with sensible defaults):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `use_case` | From Skill 1 | Use case name — prefixes all output tables |
| `freq` | Auto-detected from data | Time series frequency (`D`, `W`, `M`, `H`) |
| `prediction_length` | Ask user | Forecast horizon in time steps |
| `backtest_length` | Derived from backtest strategy | Total historical points reserved for backtesting |
| `stride` | Derived from backtest strategy | Step size between backtest windows |
| `metric` | `smape` | Evaluation metric (`smape`, `mape`, `mae`, `mse`, `rmse`) |
| `active_models` | `["StatsForecastAutoArima", "StatsForecastAutoETS", "StatsForecastAutoCES", "StatsForecastAutoTheta"]` | Models to run |
| `train_data` | `<catalog>.<schema>.{use_case}_train_data` | Input table (from `/prep-and-clean-data`) |
| `evaluation_output` | `<catalog>.<schema>.{use_case}_evaluation_output` | Evaluation results table |
| `scoring_output` | `<catalog>.<schema>.{use_case}_scoring_output` | Scoring results table |
| `group_id` | `unique_id` | Column name for series identifier |
| `date_col` | `ds` | Column name for timestamp |
| `target` | `y` | Column name for target value |

## Available model names

Use these exact names in the `active_models` parameter:

**Local models (CPU):** `StatsForecastBaselineWindowAverage`, `StatsForecastBaselineSeasonalWindowAverage`, `StatsForecastBaselineNaive`, `StatsForecastBaselineSeasonalNaive`, `StatsForecastAutoArima`, `StatsForecastAutoETS`, `StatsForecastAutoCES`, `StatsForecastAutoTheta`, `StatsForecastAutoTbats`, `StatsForecastAutoMfles`, `StatsForecastTSB`, `StatsForecastADIDA`, `StatsForecastIMAPA`, `StatsForecastCrostonClassic`, `StatsForecastCrostonOptimized`, `StatsForecastCrostonSBA`, `SKTimeProphet`

**Global models (GPU):** `NeuralForecastRNN`, `NeuralForecastLSTM`, `NeuralForecastNBEATSx`, `NeuralForecastNHITS`, `NeuralForecastAutoRNN`, `NeuralForecastAutoLSTM`, `NeuralForecastAutoNBEATSx`, `NeuralForecastAutoNHITS`, `NeuralForecastAutoTiDE`, `NeuralForecastAutoPatchTST`

**Foundation models (GPU):** `ChronosBoltTiny`, `ChronosBoltMini`, `ChronosBoltSmall`, `ChronosBoltBase`, `Chronos2`, `Chronos2Small`, `Chronos2Synth`, `TimesFM_2_5_200m`

## Steps

### Step 1: Pre-flight parameter validation

Before generating notebooks, validate:

| Validation | Rule | Error |
|-----------|------|-------|
| `backtest_length >= prediction_length` | Mandatory | "Backtest length must be >= prediction length" |
| `stride <= backtest_length` | Mandatory | "Stride must be <= backtest length" |
| `freq` ∈ {H, D, W, M} | Mandatory | "Unsupported frequency" |
| `active_models` all in model names list above | Mandatory | "Unknown model: {name}" |
| Model types consistent with cluster config | Warning | "GPU models selected but only CPU cluster configured" |
| `train_table` exists and has required schema | Mandatory | "Training table missing or invalid schema" |

```sql
SELECT COUNT(*) AS count FROM {catalog}.{schema}.{use_case}_train_data
```

### Step 1a: Profile-aware model selection (optional)

If `{use_case}_series_profile` exists, read recommended models:
```sql
SELECT COLLECT_SET(recommended_models) AS all_recommended
FROM {catalog}.{schema}.{use_case}_series_profile
WHERE forecastability_class = 'high_confidence'
```

Parse the comma-separated model names and propose them as `active_models`. Present to user via `AskUserQuestion` for confirmation.

If `{use_case}_series_profile` does NOT exist → skip this step; use the user-specified or default `active_models`.

### Step 1b: Define backtest strategy

After the user specifies `prediction_length`, guide them through configuring the backtest:

```
"Backtesting evaluates model accuracy by simulating forecasts on historical data.
 MMF slides a window across your history and produces a forecast at each position:

 ◄──────────── training data ─────────────►◄── forecast ──►
 |================================================|-----------|  Window 1
      |================================================|-----------|  Window 2 (shifted by stride)
           |================================================|-----------|  Window 3
                                                    ◄─stride─►

 Key parameters:
 • prediction_length: {prediction_length} (already set — your forecast horizon)
 • backtest_length: how much history to reserve for backtest windows
 • stride: how far to shift between windows (smaller = more windows = slower but more robust)
 • number of backtest windows = (backtest_length - prediction_length) / stride + 1"
```

Then present backtest strategy options:

```
AskUserQuestion:
  "How would you like to configure backtesting?

   (a) Quick validation — 1 backtest window (fastest)
       → backtest_length = {prediction_length}, stride = {prediction_length}
       → 1 window: fast but less robust

   (b) Standard — 3 backtest windows (recommended)
       → backtest_length = 3 × {prediction_length}, stride = {prediction_length}
       → 3 non-overlapping windows: good balance of speed and reliability

   (c) Thorough — 5+ sliding windows (most robust)
       → backtest_length = 5 × {prediction_length}, stride = {prediction_length}
       → 5 non-overlapping windows: best accuracy estimate, slower

   (d) Overlapping windows — dense evaluation
       → backtest_length = 3 × {prediction_length}, stride = 1
       → Many overlapping windows: most statistically robust, slowest

   (e) Custom — specify backtest_length and stride manually"
```

Derive `backtest_length` and `stride` from the user's choice. For option (e), ask the user to enter values directly and validate:
- `backtest_length >= prediction_length`
- `stride >= 1`
- `stride <= backtest_length`

Report the resulting number of backtest windows: `(backtest_length - prediction_length) / stride + 1`.

### Step 2: Gather and validate parameters

Confirm the input table exists:
```sql
SELECT COUNT(*) AS count FROM {catalog}.{schema}.{use_case}_train_data
```

Present all parameters to the user via `AskUserQuestion` for validation.

### Step 3: Generate notebooks by substituting placeholders

**CRITICAL: Copy the template VERBATIM from the template files, only replacing the `{placeholder}` tokens with actual values. Do NOT add, remove, or modify any other code. The templates are complete and production-ready.**

Generate a shared `run_id` (UUID) that will be passed to all notebooks, grouping results across separate GPU sessions.

#### Local models notebook

Generate a single notebook from `mmf_local_notebook_template.ipynb` with all local models in `{active_models}`. Local models (CPU/statsforecast) do not have CUDA memory constraints and can run together.

#### Placeholder values for local notebook

| Placeholder | Value |
|-------------|-------|
| `{catalog}` | user's catalog |
| `{schema}` | user's schema |
| `{train_table}` | `{use_case}_train_data` |
| `{freq}` | detected or user-specified frequency |
| `{prediction_length}` | user-specified forecast horizon (integer) |
| `{backtest_length}` | derived from backtest strategy (integer) |
| `{stride}` | derived from backtest strategy (integer) |
| `{metric}` | `smape` (default) |
| `{active_models}` | Python list literal, e.g. `["StatsForecastAutoArima", "StatsForecastAutoETS"]` |
| `{group_id}` | `unique_id` (default) |
| `{date_col}` | `ds` (default) |
| `{target}` | `y` (default) |
| `{use_case}` | use case name (for output table names and experiment path) |

Use the template from:
- [mmf_local_notebook_template.ipynb](mmf_local_notebook_template.ipynb) for `notebooks/{use_case}/run_local`

#### GPU model notebooks (one per model)

For each GPU model (global or foundation), generate a **separate notebook** from `mmf_gpu_notebook_template.ipynb`. Each notebook receives a single model.

> **Why one notebook per GPU model.** PyTorch allocates CUDA memory that cannot be freed within the same Python process. Running multiple GPU models in sequence causes OOM when the second model loads. The example notebooks (`examples/monthly/global_monthly.ipynb`) solve this via `dbutils.notebook.run()` per model — each invocation gets a fresh kernel. The Workflow job equivalent is one task per GPU model.

#### Placeholder values for GPU notebooks

| Placeholder | Value |
|-------------|-------|
| `{catalog}` | user's catalog |
| `{schema}` | user's schema |
| `{train_table}` | `{use_case}_train_data` |
| `{freq}` | detected or user-specified frequency |
| `{prediction_length}` | user-specified forecast horizon (integer) |
| `{backtest_length}` | derived from backtest strategy (integer) |
| `{stride}` | derived from backtest strategy (integer) |
| `{metric}` | `smape` (default) |
| `{model}` | single model name, e.g. `NeuralForecastAutoNHITS` |
| `{run_id}` | shared UUID for the entire run |
| `{num_nodes}` | `1` (default, single-node) |
| `{group_id}` | `unique_id` (default) |
| `{date_col}` | `ds` (default) |
| `{target}` | `y` (default) |
| `{pip_extras}` | `[global]` for NeuralForecast models, `[foundation]` for others |
| `{use_case}` | use case name (for output table names and experiment path) |

Use the template from:
- [mmf_gpu_notebook_template.ipynb](mmf_gpu_notebook_template.ipynb) — one notebook per GPU model

### Step 4: Upload notebooks

Upload generated notebooks to the workspace:
- `notebooks/{use_case}/run_local` — local models (single notebook, all local models)
- `notebooks/{use_case}/run_{model_name}` — one per GPU model (e.g., `run_NeuralForecastAutoNHITS`, `run_ChronosBoltBase`)

### Step 5: Create Workflow job

Create a multi-task Workflow job. **GPU models get one task each, chained sequentially** to prevent CUDA memory conflicts on the shared GPU cluster:

| Task key | Notebook | Cluster key | Depends on |
|----------|----------|-------------|------------|
| `local_models` | `notebooks/{use_case}/run_local` | `{use_case}_cpu_cluster` | — |
| `gpu_{model_1}` | `notebooks/{use_case}/run_{model_1}` | `{use_case}_gpu_cluster` | — |
| `gpu_{model_2}` | `notebooks/{use_case}/run_{model_2}` | `{use_case}_gpu_cluster` | `gpu_{model_1}` |
| `gpu_{model_3}` | `notebooks/{use_case}/run_{model_3}` | `{use_case}_gpu_cluster` | `gpu_{model_2}` |
| ... | ... | ... | previous GPU task |

- The `local_models` task runs independently (in parallel with GPU tasks if both CPU and GPU clusters exist).
- GPU tasks are **chained sequentially** via `depends_on` to ensure only one GPU model is loaded at a time.
- Only include tasks for the model classes the user selected.

Job clusters (ephemeral, created with the job):

| Cluster key | Runtime | Node type (AWS) | Workers | Spark config |
|-------------|---------|-----------------|---------|--------------|
| `{use_case}_cpu_cluster` | `17.3.x-cpu-ml-scala2.13` | `i3.xlarge` | Dynamic (from Skill 3) | `spark.sql.execution.arrow.enabled=true`, `spark.sql.adaptive.enabled=false`, `spark.databricks.delta.formatCheck.enabled=false`, `spark.databricks.delta.schema.autoMerge.enabled=true` |
| `{use_case}_gpu_cluster` | `18.0.x-gpu-ml-scala2.13` | `g5.12xlarge` (default) | 0 (single-node) | `spark.databricks.delta.formatCheck.enabled=false`, `spark.databricks.delta.schema.autoMerge.enabled=true` |

Use `create_job` to create the Workflow, then `run_job` to start it.

#### Serverless alternative

If the user requests **serverless compute**, omit `job_clusters` entirely and use `environment_key` in each task instead of `job_cluster_key`:

```json
{
  "name": "{use_case}_forecasting",
  "tasks": [{
    "task_key": "local_models",
    "notebook_task": {"notebook_path": "notebooks/{use_case}/run_local"},
    "environment_key": "Default"
  }],
  "environments": [{
    "environment_key": "Default",
    "spec": {"client": "1"}
  }]
}
```

This is faster (no cluster startup) and works well for local CPU models. GPU models require ML Runtime clusters and cannot use serverless.

### Step 6: Monitor execution

Poll the job run status until completion. Report progress to the user with structured status updates showing per-model progress:

```
[HH:MM:SS] Job run started (run_id: {run_id})
[HH:MM:SS] Task local_models: RUNNING
[HH:MM:SS] Task gpu_NeuralForecastAutoNHITS: RUNNING
[HH:MM:SS] Task local_models: SUCCEEDED (duration: 12m 34s)
[HH:MM:SS] Task gpu_NeuralForecastAutoNHITS: SUCCEEDED (duration: 5m 12s)
[HH:MM:SS] Task gpu_ChronosBoltBase: RUNNING
[HH:MM:SS] Task gpu_ChronosBoltBase: SUCCEEDED (duration: 3m 45s)
[HH:MM:SS] All tasks completed. Overall status: SUCCEEDED
```

### Step 7: Log run metadata

Write run metadata to `{use_case}_run_metadata` table for audit and reproducibility:

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
  '{active_models_str}' AS active_models
```

Confirm to the user that evaluation and scoring output tables have been written, and report row counts:

```sql
SELECT COUNT(*) AS eval_rows FROM {catalog}.{schema}.{use_case}_evaluation_output
```
```sql
SELECT COUNT(*) AS score_rows FROM {catalog}.{schema}.{use_case}_scoring_output
```

### Step 8: Hand off to post-processing

Present to the user:

```
"Forecast run complete.
 • Evaluation output: {eval_rows} rows in {use_case}_evaluation_output
 • Scoring output: {score_rows} rows in {use_case}_scoring_output
 • Run metadata logged to {use_case}_run_metadata
 • MLflow experiment: /Users/{user}/mmf/{use_case}

 → Run /post-process-and-evaluate to analyze results, select best models,
   and generate a business-ready summary."
```

> **Note — analysis moved to Skill 5.** Result analysis queries (best model per series, avg metric per model, worst series) and business reporting are consolidated in Skill 5 (`/post-process-and-evaluate`) to avoid duplication.

## Outputs

- Delta table `<catalog>.<schema>.{use_case}_evaluation_output` — backtest metrics per model per series
- Delta table `<catalog>.<schema>.{use_case}_scoring_output` — forward-looking forecasts
- Delta table `<catalog>.<schema>.{use_case}_run_metadata` — run parameters and audit trail
- MLflow experiment at `/Users/<user>/mmf/{use_case}`
