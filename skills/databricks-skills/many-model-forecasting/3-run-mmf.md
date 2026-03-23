# Run MMF

**Slash command:** `/run-mmf <catalog> <schema>`

Generates and submits a Many Models Forecasting notebook to Databricks,
monitors execution, and summarizes results.

## Parameters

Gather these from the user (with sensible defaults):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `freq` | Auto-detected from data | Time series frequency (`D`, `W`, `M`, `H`) |
| `prediction_length` | Ask user | Forecast horizon in time steps |
| `backtest_length` | `3 * prediction_length` | Number of historical points for backtesting |
| `stride` | `prediction_length` | Step size between backtest windows |
| `metric` | `smape` | Evaluation metric (`smape`, `mape`, `mae`, `mse`, `rmse`) |
| `active_models` | `["StatsForecastAutoArima", "StatsForecastAutoETS", "StatsForecastAutoCES", "StatsForecastAutoTheta"]` | Models to run |
| `train_data` | `<catalog>.<schema>.mmf_train_data` | Input table (from `/explore-data`) |
| `group_id` | `unique_id` | Column name for series identifier |
| `date_col` | `ds` | Column name for timestamp |
| `target` | `y` | Column name for target value |

## Available model names

Use these exact names in the `active_models` parameter:

**Local models (CPU):** `StatsForecastBaselineWindowAverage`, `StatsForecastBaselineSeasonalWindowAverage`, `StatsForecastBaselineNaive`, `StatsForecastBaselineSeasonalNaive`, `StatsForecastAutoArima`, `StatsForecastAutoETS`, `StatsForecastAutoCES`, `StatsForecastAutoTheta`, `StatsForecastAutoTbats`, `StatsForecastAutoMfles`, `StatsForecastTSB`, `StatsForecastADIDA`, `StatsForecastIMAPA`, `StatsForecastCrostonClassic`, `StatsForecastCrostonOptimized`, `StatsForecastCrostonSBA`, `SKTimeProphet`

**Global models (GPU):** `NeuralForecastRNN`, `NeuralForecastLSTM`, `NeuralForecastNBEATSx`, `NeuralForecastNHITS`, `NeuralForecastAutoRNN`, `NeuralForecastAutoLSTM`, `NeuralForecastAutoNBEATSx`, `NeuralForecastAutoNHITS`, `NeuralForecastAutoTiDE`, `NeuralForecastAutoPatchTST`

**Foundation models (GPU):** `ChronosBoltTiny`, `ChronosBoltMini`, `ChronosBoltSmall`, `ChronosBoltBase`, `Chronos2`, `Chronos2Small`, `Chronos2Synth`, `TimesFM_2_5_200m`

## Steps

### Step 1: Gather and validate parameters

Confirm the input table exists:
```sql
SELECT COUNT(*) AS count FROM {catalog}.{schema}.mmf_train_data
```

Present all parameters to the user via `AskUserQuestion` for validation.

### Step 2: Generate notebooks by substituting placeholders

**CRITICAL: Copy the template VERBATIM from the template files, only replacing the `{placeholder}` tokens with actual values. Do NOT add, remove, or modify any other code. The templates are complete and production-ready.**

#### Placeholder values

| Placeholder | Value |
|-------------|-------|
| `{catalog}` | user's catalog |
| `{schema}` | user's schema |
| `{train_table}` | `mmf_train_data` |
| `{freq}` | detected or user-specified frequency |
| `{prediction_length}` | user-specified forecast horizon (integer) |
| `{backtest_length}` | `3 * prediction_length` (default, integer) |
| `{stride}` | `prediction_length` (default, integer) |
| `{metric}` | `smape` (default) |
| `{active_models}` | Python list literal, e.g. `["StatsForecastAutoArima", "StatsForecastAutoETS"]` |
| `{group_id}` | `unique_id` (default) |
| `{date_col}` | `ds` (default) |
| `{target}` | `y` (default) |
| `{pip_extras}` | `[foundation]` or `[global]` (GPU template only) |

Use the templates from:
- [mmf_local_notebook_template.ipynb](mmf_local_notebook_template.ipynb) for `notebooks/run_local`
- [mmf_gpu_notebook_template.ipynb](mmf_gpu_notebook_template.ipynb) for `notebooks/run_foundation` (with `{pip_extras}` = `[foundation]`) and `notebooks/run_global` (with `{pip_extras}` = `[global]`)

### Step 3: Upload notebooks

Upload the generated notebooks to the Databricks workspace at:
- `notebooks/run_local` — local models notebook
- `notebooks/run_foundation` — foundation models notebook
- `notebooks/run_global` — global models notebook

### Step 4: Create Workflow job

Create a Databricks multi-task Workflow job with these task definitions:

| Task key | Notebook | Cluster key | Description |
|----------|----------|-------------|-------------|
| `mmf_local_models` | `notebooks/run_local` | `mmf_cpu_cluster` | Runs local models via Spark Pandas UDFs |
| `mmf_foundation_models` | `notebooks/run_foundation` | `mmf_gpu_cluster` | Runs foundation models on GPU |
| `mmf_global_models` | `notebooks/run_global` | `mmf_gpu_cluster` | Runs global models on GPU |

Only include tasks matching the user's selected model types. Tasks are independent and run in parallel.

Job clusters (ephemeral, created with the job):

| Cluster key | Runtime | Node type (AWS) | Workers | Spark config |
|-------------|---------|-----------------|---------|--------------|
| `mmf_cpu_cluster` | `17.3.x-cpu-ml-scala2.13` | `i3.xlarge` | 2 | `spark.sql.execution.arrow.enabled=true`, `spark.sql.adaptive.enabled=false`, `spark.databricks.delta.formatCheck.enabled=false`, `spark.databricks.delta.schema.autoMerge.enabled=true` |
| `mmf_gpu_cluster` | `18.0.x-gpu-ml-scala2.13` | `g5.12xlarge` | 0 (single-node) | `spark.databricks.delta.formatCheck.enabled=false`, `spark.databricks.delta.schema.autoMerge.enabled=true` |

Use `create_job` to create the Workflow, then `run_job` to start it.

#### Serverless alternative

If the user requests **serverless compute**, omit `job_clusters` entirely and use `environment_key` in each task instead of `job_cluster_key`:

```json
{
  "name": "mmf_forecasting",
  "tasks": [{
    "task_key": "mmf_local_models",
    "notebook_task": {"notebook_path": "notebooks/run_local"},
    "environment_key": "Default"
  }],
  "environments": [{
    "environment_key": "Default",
    "spec": {"client": "1"}
  }]
}
```

This is faster (no cluster startup) and works well for local CPU models.

### Step 5: Monitor execution

Poll the job run status until completion. Report progress to the user.

### Step 6: Analyze results

Read the evaluation output and present:
- Best model per series (by chosen metric)
- Average metric across all series per model
- Worst-performing series (potential data quality issues)

```sql
-- Best model per series
SELECT unique_id, model, {metric}
FROM {catalog}.{schema}.mmf_evaluation_output
WHERE (unique_id, {metric}) IN (
  SELECT unique_id, MIN({metric})
  FROM {catalog}.{schema}.mmf_evaluation_output
  GROUP BY unique_id
)
ORDER BY {metric}

-- Average metric per model
SELECT model, ROUND(AVG({metric}), 4) AS avg_metric, COUNT(*) AS series_count
FROM {catalog}.{schema}.mmf_evaluation_output
GROUP BY model
ORDER BY avg_metric

-- Worst-performing series
SELECT unique_id, model, {metric}
FROM {catalog}.{schema}.mmf_evaluation_output
ORDER BY {metric} DESC
LIMIT 20
```

### Step 7: Suggest next steps

Based on results:
- If local models performed well: "Consider running with global models for potentially better accuracy."
- If some series have poor metrics: "These series may need more data or different frequency."
- If the user wants to iterate: allow re-running with different models or parameters.

## Outputs

- Delta table `<catalog>.<schema>.mmf_evaluation_output` — backtest metrics per model per series
- Delta table `<catalog>.<schema>.mmf_scoring_output` — forward-looking forecasts
- MLflow experiment at `/Users/<user>/mmf/<use_case_name>`
- Summary of results presented to the user