# Execute MMF Forecast

**Slash command:** `/execute-mmf-forecast`

Validates parameters, asks the user about backtesting setup, generates notebooks
using the **orchestrator + run_gpu** pattern, creates **one job per model class**
(local, global, foundation), and triggers them **in parallel**.

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
| `active_models` | From Skill 3 | Models to run (grouped by class) |
| `train_data` | `<catalog>.<schema>.{use_case}_train_data` | Input table — see Step 0a for routing based on non-forecastable strategy |
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

### Step 0a: Read non-forecastable strategy from pipeline config

Check if `{use_case}_pipeline_config` exists:

```sql
SELECT non_forecastable_strategy, fallback_method, non_forecastable_models,
       n_forecastable, n_non_forecastable
FROM {catalog}.{schema}.{use_case}_pipeline_config
WHERE use_case = '{use_case}'
```

Based on the strategy, determine which training table to use for the **main pipeline**:

| Strategy | Main pipeline train table | Non-forecastable handling |
|----------|--------------------------|--------------------------|
| `include` (or config missing) | `{use_case}_train_data` | All series run together — no separate handling |
| `fallback` | `{use_case}_train_data_forecastable` | Already handled in Skill 2 — fallback forecasts in `{use_case}_scoring_output_non_forecastable` |
| `separate_job` | `{use_case}_train_data_forecastable` | Separate job created in Step 5a below |

Set `{train_table}` to the correct table name for all subsequent notebook generation steps.

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
-- Use the train table determined in Step 0a
SELECT COUNT(*) AS count FROM {catalog}.{schema}.{train_table}
```

### Step 1a: Profile-aware model selection (optional)

If `{use_case}_series_profile` exists, read recommended models:
```sql
SELECT COLLECT_SET(recommended_models) AS all_recommended
FROM {catalog}.{schema}.{use_case}_series_profile
WHERE forecastability_class = 'high_confidence'
```

Parse the comma-separated model names and propose them as `active_models`. Present to user via `AskUserQuestion` for confirmation.

If `{use_case}_series_profile` does NOT exist → skip this step; use the models selected in Skill 3.

### ⛔ STOP GATE — Step 1b: Ask user about backtesting setup

**Always ask the user about their backtesting configuration. Do NOT proceed until the user confirms.**

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

**WAIT for the user to respond. Do NOT derive backtest parameters without user input.**

Derive `backtest_length` and `stride` from the user's choice. For option (e), ask the user to enter values directly and validate:
- `backtest_length >= prediction_length`
- `stride >= 1`
- `stride <= backtest_length`

Report the resulting number of backtest windows: `(backtest_length - prediction_length) / stride + 1`.

### Step 2: Gather and validate parameters

Confirm the input table exists:
```sql
SELECT COUNT(*) AS count FROM {catalog}.{schema}.{train_table}
```

Present all parameters to the user via `AskUserQuestion` for validation.

### Feature type decision guide

`mmf_sa` supports five covariate types. **By default, always use `[]` for every covariate list and `""` for `scoring_table`** — the pipeline runs in univariate mode and all model families (local, global, foundation) work well without covariates.

| Feature type | Recommended? | When values are needed | Safe default |
|---|---|---|---|
| `static_features` | **Yes** — use when available | Constant per `group_id`, always known (e.g. `store_state`, `dept_id`) | `[]` |
| `dynamic_historical_numerical` | **Yes** — for global models | Past-only; NOT needed at forecast time (e.g. lagged sales, rolling averages) | `[]` |
| `dynamic_historical_categorical` | **Yes** — for global models | Past-only; NOT needed at forecast time | `[]` |
| `dynamic_future_numerical` | **AVOID** | Must provide values for **every future `ds`** in the forecast horizon via a separate scoring table. If missing, the pipeline errors or **silently drops series**. | `[]` |
| `dynamic_future_categorical` | **AVOID** | Same as above — requires known future values for every forecast date | `[]` |

> **Why avoid `dynamic_future_*`.** These features require the user to build and maintain a **scoring table** with one row per `unique_id` x future `ds`, pre-populated with the regressor values. Most many-series forecasting use cases (retail demand, financial metrics, IoT telemetry) do not have reliably known future regressors. When `dynamic_future_*` columns are specified but the scoring table is incomplete, `mmf_sa` either raises an error or silently removes the affected series from the forecast — producing no output for those series with no warning in the final results.
>
> **When `dynamic_future_*` is appropriate:** Only when the user's `{forecast_problem_brief}` explicitly mentions known future regressors (e.g., planned promotional calendars, contractual pricing schedules, weather forecasts) **and** the user confirms they have a pre-built scoring table. Even then, prefer `static_features` for attributes that are constant per series and `dynamic_historical_*` for signals derivable from past data.

**Always substitute `[]` for all `dynamic_future_*` placeholders and `""` for `{scoring_table}` unless the user explicitly requests future exogenous regressors and confirms they have a scoring table.**

### Step 3: Generate notebooks

**CRITICAL: Copy templates VERBATIM, only replacing `{placeholder}` tokens with actual values. Do NOT add, remove, or modify any other code. The templates are complete and production-ready.**

Generate a shared `run_id` (UUID) that will be passed to all notebooks, grouping results across separate sessions.

#### 3a: Local models notebook

Generate a single notebook from `mmf_local_notebook_template.ipynb` with all local models in `{active_models}`. Local models (CPU/statsforecast) do not have CUDA memory constraints and can run together.

##### Placeholder values for local notebook

| Placeholder | Value |
|-------------|-------|
| `{catalog}` | user's catalog |
| `{schema}` | user's schema |
| `{train_table}` | Determined by Step 0a: `{use_case}_train_data` (include), `{use_case}_train_data_forecastable` (fallback/separate_job) |
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
| `{static_features}` | Python list of column names constant per `group_id`, e.g. `[]` or `["dept_id","state_id"]`. Safe to use when available. |
| `{dynamic_future_numerical}` | **AVOID — always `[]`** unless user has a scoring table with known future values. See [Feature type decision guide](#feature-type-decision-guide). |
| `{dynamic_future_categorical}` | **AVOID — always `[]`**. Same as above. |
| `{dynamic_historical_numerical}` | Python list; past-only signals for **NeuralForecast** global models (e.g. lagged features), or `[]`. Safe to use. |
| `{dynamic_historical_categorical}` | Python list, or `[]`. Safe to use. |
| `{scoring_table}` | **Always `""`** unless `dynamic_future_*` is in use (not recommended). Empty string means `scoring_data` = `train_table`. |

Use the template from:
- [mmf_local_notebook_template.ipynb](mmf_local_notebook_template.ipynb) → save locally as `notebooks/{use_case}/run_local.ipynb`

#### 3b: GPU run notebook (static — no placeholder substitution)

Copy the `mmf_gpu_run_notebook_template.ipynb` **as-is** to `notebooks/{use_case}/run_gpu.ipynb`. This notebook:
- Receives all parameters via `dbutils.widgets` (catalog, schema, model, run_id, etc.)
- Auto-detects whether the model is global or foundation and installs the correct `mmf_sa` extras
- Runs a single model per invocation
- Is called by the orchestrator notebooks via `dbutils.notebook.run()`

**Do NOT modify this template.** Copy it verbatim.

Use the template from:
- [mmf_gpu_run_notebook_template.ipynb](mmf_gpu_run_notebook_template.ipynb) → save locally as `notebooks/{use_case}/run_gpu.ipynb`

#### 3c: GPU orchestrator notebooks (one per model class)

For each GPU model class (global and/or foundation), generate an **orchestrator notebook** from `mmf_gpu_orchestrator_notebook_template.ipynb`. Each orchestrator:
- Holds the list of active models for its class
- Loops through the models and calls `run_gpu` for each via `dbutils.notebook.run()`
- Each `dbutils.notebook.run()` invocation gets a fresh Python process, avoiding CUDA memory conflicts

> **Why orchestrator + run_gpu.** PyTorch allocates CUDA memory that cannot be freed within the same Python process. Running multiple GPU models in sequence causes OOM when the second model loads. The `dbutils.notebook.run()` pattern gives each model a fresh kernel. This is the same pattern used in the [examples folder](../../examples/monthly/global_monthly.ipynb).

##### Placeholder values for orchestrator notebooks

| Placeholder | Value |
|-------------|-------|
| `{catalog}` | user's catalog |
| `{schema}` | user's schema |
| `{use_case}` | use case name |
| `{train_table}` | Determined by Step 0a: `{use_case}_train_data` (include), `{use_case}_train_data_forecastable` (fallback/separate_job) |
| `{freq}` | detected or user-specified frequency |
| `{prediction_length}` | user-specified forecast horizon (integer) |
| `{backtest_length}` | derived from backtest strategy (integer) |
| `{stride}` | derived from backtest strategy (integer) |
| `{metric}` | `smape` (default) |
| `{active_models}` | Python list literal of models for this class only, e.g. `["NeuralForecastAutoNHITS", "NeuralForecastAutoPatchTST"]` |
| `{group_id}` | `unique_id` (default) |
| `{date_col}` | `ds` (default) |
| `{target}` | `y` (default) |
| `{num_nodes}` | `1` (single-node always) |
| `{static_features}` | Same as local notebook — Python list literal, often `[]`. Safe to use. |
| `{dynamic_future_numerical}` | **AVOID — always `[]`** unless user has a scoring table. See [Feature type decision guide](#feature-type-decision-guide). |
| `{dynamic_future_categorical}` | **AVOID — always `[]`**. Same as above. |
| `{dynamic_historical_numerical}` | Past-only signals for **global** models (lag/roll features), or `[]`. Safe to use. |
| `{dynamic_historical_categorical}` | Same — often `[]`. Safe to use. |
| `{scoring_table}` | **Always `""`** unless `dynamic_future_*` is in use (not recommended). |

Use the template from:
- [mmf_gpu_orchestrator_notebook_template.ipynb](mmf_gpu_orchestrator_notebook_template.ipynb)

Generate up to two orchestrators, saving locally:
- `notebooks/{use_case}/orchestrator_global.ipynb` — if any global models selected
- `notebooks/{use_case}/orchestrator_foundation.ipynb` — if any foundation models selected

#### Covariates and model coverage (reference)

> **Default: univariate mode.** All model families work well without covariates. Always generate notebooks with `[]` for every covariate list and `""` for `scoring_table` unless the user explicitly requests otherwise. See [Feature type decision guide](#feature-type-decision-guide).

`mmf_sa.run_forecast` maps columns into StatsForecast / NeuralForecast / Chronos / TimesFM pipelines:

- **StatsForecast (local)**: runs univariate by default. Can accept `dynamic_future_*` covariates but these require a scoring table — **avoid unless the user explicitly provides one**.
- **NeuralForecast (global)**: benefits from **`static_features`** and **`dynamic_historical_*`** (past-only signals used during training). Can also accept `dynamic_future_*` but same caveat applies.
- **Foundation (Chronos / TimesFM)**: benefits from **`static_features`**. Can accept `dynamic_future_*` in the current implementation but does not use `dynamic_historical_*`.

**If the user insists on future exogenous regressors** (rare — only with confirmed known future data like promotional calendars), follow the pattern in [examples/run_external_regressors_daily.ipynb](../../examples/run_external_regressors_daily.ipynb): training table with history + target, a **separate** scoring table with future dates and exogenous columns, and set `{scoring_table}` accordingly.

### Step 4: Import notebooks into Databricks workspace

> ⚠️ **Do NOT use `upload_file` for notebooks.** The `upload_file` MCP tool creates a workspace FILE, not a NOTEBOOK. Databricks job tasks require a proper NOTEBOOK object. Using `upload_file` will cause every job to fail immediately with: `'<path>' is not a notebook`.

Use the **Databricks CLI** with `--format JUPYTER` to import each notebook. If a path already exists as a FILE from a prior failed `upload_file`, delete it first before importing.

Import all notebooks:

```bash
# Local models notebook
databricks workspace import /notebooks/{use_case}/run_local \
  --file /tmp/{use_case}_run_local.ipynb \
  --format JUPYTER --overwrite

# GPU run notebook (static — uploaded verbatim)
databricks workspace import /notebooks/{use_case}/run_gpu \
  --file /tmp/{use_case}_run_gpu.ipynb \
  --format JUPYTER --overwrite

# Global orchestrator (if global models selected)
databricks workspace import /notebooks/{use_case}/orchestrator_global \
  --file /tmp/{use_case}_orchestrator_global.ipynb \
  --format JUPYTER --overwrite

# Foundation orchestrator (if foundation models selected)
databricks workspace import /notebooks/{use_case}/orchestrator_foundation \
  --file /tmp/{use_case}_orchestrator_foundation.ipynb \
  --format JUPYTER --overwrite
```

### Step 4b: Verify all notebooks imported correctly

```bash
databricks workspace list /notebooks/{use_case}/
```

Confirm every path shows `object_type: NOTEBOOK` (not `FILE`). Fix any mismatches before creating jobs.

The local copies in `notebooks/{use_case}/` serve as version-controllable artifacts of the generated pipeline.

### Step 5: Create one job per model class (triggered in parallel)

**Create separate Workflow jobs for each model class and trigger them all in parallel.** This maximizes throughput — local models run on CPU while GPU models run concurrently.

#### Job 1: Local models (if any local models selected)

```json
{
  "name": "{use_case}_local_forecasting",
  "tasks": [{
    "task_key": "local_models",
    "notebook_task": {
      "notebook_path": "notebooks/{use_case}/run_local"
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
      "notebook_path": "notebooks/{use_case}/orchestrator_global"
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
      "custom_tags": {
        "ResourceClass": "SingleNode"
      }
    }
  }]
}
```

#### Job 3: Foundation models (if any foundation models selected)

```json
{
  "name": "{use_case}_foundation_forecasting",
  "tasks": [{
    "task_key": "foundation_models",
    "notebook_task": {
      "notebook_path": "notebooks/{use_case}/orchestrator_foundation"
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
      "custom_tags": {
        "ResourceClass": "SingleNode"
      }
    }
  }]
}
```

**Important notes:**
- **All clusters** use `data_security_mode: "SINGLE_USER"` because ML runtimes (`*-cpu-ml-*` and `*-gpu-ml-*`) reject `USER_ISOLATION`.
- GPU clusters are **always single-node** (`num_workers: 0`) with `spark.master: local[*]`, `spark.databricks.cluster.profile: singleNode`, and `custom_tags: {"ResourceClass": "SingleNode"}`. All three are required for proper single-node mode.
- Global and foundation jobs each get their own ephemeral GPU cluster so they can run **in parallel**.
- Only create jobs for model classes the user actually selected.

Use `create_job` to create each job, then `run_job` to start **all jobs simultaneously**.

#### Serverless alternative (local models only)

If the user requests **serverless compute** for local models, omit `job_clusters` entirely and use `environment_key` instead of `job_cluster_key`:

```json
{
  "name": "{use_case}_local_forecasting",
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

GPU models require ML Runtime clusters and cannot use serverless.

#### Step 5a: Non-forecastable separate job (if `separate_job` strategy)

**Only create these jobs if `non_forecastable_strategy == 'separate_job'`.** Skip entirely for `include` or `fallback` strategies.

Generate notebooks for the non-forecastable pipeline using the same templates, but with:
- `{train_table}` → `{use_case}_train_data_non_forecastable`
- `{active_models}` → the non-forecastable models selected in Skill 2/3
- `{use_case}` output table suffix → `{use_case}_nf` (to avoid overwriting main pipeline outputs)

The evaluation and scoring output tables for non-forecastable series are:
- `{use_case}_nf_evaluation_output`
- `{use_case}_nf_scoring_output`

##### Non-forecastable local models notebook

If the non-forecastable models include local (CPU) models, generate a notebook from `mmf_local_notebook_template.ipynb`:
- Save locally as `notebooks/{use_case}/run_local_nf.ipynb`
- Upload to workspace at `notebooks/{use_case}/run_local_nf`

##### Non-forecastable GPU orchestrator notebooks

If the non-forecastable models include global or foundation models, generate orchestrator notebooks:
- `notebooks/{use_case}/orchestrator_global_nf.ipynb` — if any global models
- `notebooks/{use_case}/orchestrator_foundation_nf.ipynb` — if any foundation models

These orchestrators call the same `run_gpu` notebook (shared with the main pipeline).

##### Job definitions for non-forecastable pipeline

**Job NF-1: Non-forecastable local models** (if any NF local models):

```json
{
  "name": "{use_case}_nf_local_forecasting",
  "tasks": [{
    "task_key": "nf_local_models",
    "notebook_task": {
      "notebook_path": "notebooks/{use_case}/run_local_nf"
    },
    "job_cluster_key": "{use_case}_nf_cpu_cluster"
  }],
  "job_clusters": [{
    "job_cluster_key": "{use_case}_nf_cpu_cluster",
    "new_cluster": {
      "spark_version": "17.3.x-cpu-ml-scala2.13",
      "node_type_id": "{cpu_node_type}",
      "num_workers": {nf_cpu_workers},
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

**Job NF-2: Non-forecastable global models** (if any NF global models):

```json
{
  "name": "{use_case}_nf_global_forecasting",
  "tasks": [{
    "task_key": "nf_global_models",
    "notebook_task": {
      "notebook_path": "notebooks/{use_case}/orchestrator_global_nf"
    },
    "job_cluster_key": "{use_case}_nf_gpu_cluster"
  }],
  "job_clusters": [{
    "job_cluster_key": "{use_case}_nf_gpu_cluster",
    "new_cluster": {
      "spark_version": "18.0.x-gpu-ml-scala2.13",
      "node_type_id": "{nf_gpu_node_type}",
      "num_workers": 0,
      "data_security_mode": "SINGLE_USER",
      "spark_conf": {
        "spark.master": "local[*]",
        "spark.databricks.cluster.profile": "singleNode",
        "spark.databricks.delta.formatCheck.enabled": "false",
        "spark.databricks.delta.schema.autoMerge.enabled": "true"
      },
      "custom_tags": {
        "ResourceClass": "SingleNode"
      }
    }
  }]
}
```

**Job NF-3: Non-forecastable foundation models** (if any NF foundation models):

Same pattern as NF-2, with `orchestrator_foundation_nf` notebook.

**Launch all non-forecastable jobs in parallel with the main pipeline jobs.** All jobs (main + NF) are triggered simultaneously.

### Step 6: Monitor execution

Poll all job run statuses until completion. Report progress to the user with structured status updates showing per-job progress:

```
[HH:MM:SS] Triggered {n_jobs} jobs in parallel:
[HH:MM:SS]   Job {use_case}_local_forecasting (run_id: {local_run_id})
[HH:MM:SS]   Job {use_case}_global_forecasting (run_id: {global_run_id})
[HH:MM:SS]   Job {use_case}_foundation_forecasting (run_id: {foundation_run_id})
{if separate_job:
[HH:MM:SS]   Job {use_case}_nf_local_forecasting (run_id: {nf_local_run_id})
}

[HH:MM:SS] {use_case}_local_forecasting: RUNNING
[HH:MM:SS] {use_case}_global_forecasting: RUNNING (orchestrator running NeuralForecastAutoNHITS...)
[HH:MM:SS] {use_case}_foundation_forecasting: RUNNING (orchestrator running ChronosBoltBase...)
{if separate_job:
[HH:MM:SS] {use_case}_nf_local_forecasting: RUNNING
}
[HH:MM:SS] {use_case}_local_forecasting: SUCCEEDED (duration: 12m 34s)
[HH:MM:SS] {use_case}_global_forecasting: SUCCEEDED (duration: 25m 12s)
[HH:MM:SS] {use_case}_foundation_forecasting: SUCCEEDED (duration: 18m 45s)
{if separate_job:
[HH:MM:SS] {use_case}_nf_local_forecasting: SUCCEEDED (duration: 5m 20s)
}
[HH:MM:SS] All jobs completed. Overall status: SUCCEEDED
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
  '{active_models_str}' AS active_models,
  '{non_forecastable_strategy}' AS non_forecastable_strategy,
  '{train_table}' AS train_table
```

Confirm to the user that evaluation and scoring output tables have been written, and report row counts:

```sql
SELECT COUNT(*) AS eval_rows FROM {catalog}.{schema}.{use_case}_evaluation_output
```
```sql
SELECT COUNT(*) AS score_rows FROM {catalog}.{schema}.{use_case}_scoring_output
```

If `separate_job` strategy, also report non-forecastable output counts:
```sql
SELECT COUNT(*) AS nf_eval_rows FROM {catalog}.{schema}.{use_case}_nf_evaluation_output
```
```sql
SELECT COUNT(*) AS nf_score_rows FROM {catalog}.{schema}.{use_case}_nf_scoring_output
```

### ⛔ STOP GATE — Step 8: Hand off to post-processing

Present to the user and ask whether to proceed:

```
AskUserQuestion:
  "✅ Forecast run complete.

   Main pipeline (forecastable series):
   • Evaluation output: {eval_rows} rows in {use_case}_evaluation_output
   • Scoring output: {score_rows} rows in {use_case}_scoring_output

   {if separate_job:
   Non-forecastable pipeline:
   • Evaluation output: {nf_eval_rows} rows in {use_case}_nf_evaluation_output
   • Scoring output: {nf_score_rows} rows in {use_case}_nf_scoring_output
   }

   {if fallback:
   Non-forecastable series: handled by {fallback_method} fallback (in {use_case}_scoring_output_non_forecastable)
   }

   • Run metadata logged to {use_case}_run_metadata
   • MLflow experiment: /Users/{user}/mmf/{use_case}

   Would you like to proceed to post-processing and evaluation?
   (a) Yes, proceed to /post-process-and-evaluate
   (b) No, stop here — I'll come back later"
```

**Do NOT proceed until the user responds.**

> **Note — analysis moved to Skill 5.** Result analysis queries (best model per series, avg metric per model, worst series) and business reporting are consolidated in Skill 5 (`/post-process-and-evaluate`) to avoid duplication.

## Outputs

**Always produced:**
- Delta table `<catalog>.<schema>.{use_case}_evaluation_output` — backtest metrics per model per series (forecastable series only when strategy is `fallback` or `separate_job`)
- Delta table `<catalog>.<schema>.{use_case}_scoring_output` — forward-looking forecasts (forecastable series only when strategy is `fallback` or `separate_job`)
- Delta table `<catalog>.<schema>.{use_case}_run_metadata` — run parameters, audit trail, and non-forecastable strategy
- MLflow experiment at `/Users/<user>/mmf/{use_case}`

**Produced when `separate_job` strategy:**
- Delta table `<catalog>.<schema>.{use_case}_nf_evaluation_output` — backtest metrics for non-forecastable series
- Delta table `<catalog>.<schema>.{use_case}_nf_scoring_output` — forward-looking forecasts for non-forecastable series
