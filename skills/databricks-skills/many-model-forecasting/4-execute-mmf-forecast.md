# Execute MMF Forecast

> ⛔ **MANDATORY:** If you have not read [SKILL.md](SKILL.md) yet, read it now before proceeding. Do NOT take any action until you have read both SKILL.md and this file in full.

**Slash command:** `/execute-mmf-forecast`

Validates parameters, asks the user about backtesting setup, generates notebooks
using the **orchestrator + run_gpu** pattern, creates **one job per model class**
(local, global, foundation), and triggers them **in parallel**.

## ⛔ Preconditions — DO NOT SKIP

> **This skill is NOT the entry point of the MMF workflow.** Even if the user said "let's forecast" or "use MMF" or "run the forecast," the agent must NOT start here unless ALL preconditions below are satisfied. If any are missing, route the user back to the earliest unmet skill — do not improvise inputs.

| Precondition | How to verify | If missing |
|---|---|---|
| `{catalog}.{schema}.{use_case}_train_data` exists (or `_train_data_forecastable` for fallback/separate_job) | `get_table` or `SELECT 1 FROM ... LIMIT 1` | Go back to **Skill 1 (`/prep-and-clean-data`)** |
| `active_models` has been chosen by the user (grouped by local / global / foundation) | Check conversation state | Go back to **Skill 3 (`/provision-forecasting-resources`)** |
| Cluster config has been confirmed by the user | Check conversation state | Go back to **Skill 3** |
| `{forecast_problem_brief}` is in conversation context | Check prior turns | Reconfirm with the user |
| `{use_case}` name is known | Check conversation state | Go back to **Skill 1** |

**Verification routine the agent MUST run before any other Step in this skill:**

1. Ask the user (or recall from context) the `{use_case}`, `{catalog}`, `{schema}`, and `{freq}`.
2. Run `SELECT 1 FROM {catalog}.{schema}.{use_case}_train_data LIMIT 1` (or `_train_data_forecastable`). If it errors, **stop** and tell the user: *"I can't find your training table — Skill 1 needs to run first. Want me to start Skill 1 (`/prep-and-clean-data`)?"*
3. Confirm the user has an `active_models` selection. If not, **stop** and route to Skill 3.
4. **Verify date-alignment of the training table against `{freq}`.** Skill 4 must NOT run the forecast if `{use_case}_train_data` contains misaligned dates (e.g. monthly data not snapped to month-end). Run the check matching `{freq}`:

   - For `{freq} == "M"`:

     ```sql
     SELECT COUNT(*) AS misaligned
     FROM {catalog}.{schema}.{use_case}_train_data
     WHERE ds <> LAST_DAY(ds)
     ```

   - For `{freq} == "W"`:

     ```sql
     SELECT COUNT(*) AS misaligned
     FROM {catalog}.{schema}.{use_case}_train_data
     WHERE ds <> DATE_TRUNC('week', ds) + INTERVAL 6 DAY
     ```

   - For `{freq} ∈ {D, H}`: skip this check.

   If `misaligned > 0`, **stop**, tell the user the training table is misaligned for the declared frequency, and offer to re-run Skill 1 Step 6 to rebuild it. **Do NOT attempt the forecast** — `run_forecast` indexes by `ds` and a misaligned monthly column produces silently wrong results.

   If `{use_case}_scoring_data` exists, run the same check against it.

5. Only then proceed to Step 0a below.

**Never default `active_models`, `prediction_length`, or `freq` silently** — these come from prior skills or must be explicitly asked.

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

### ⛔ STOP GATE — Step 0b: Get current user and confirm project folder

Call `get_current_user()` to obtain the authenticated user's full email (e.g. `user@databricks.com`). Derive:
- `{username}` = local part before `@` (e.g. `user`)
- `{YYYYMMDD}` = today's date (e.g. `20260420`)

If `{project_folder}`, `{notebook_base_path}`, and `{experiment_path}` were already confirmed in Skill 1, reuse them. If not known (new session or Skill 1 was skipped), ask:

```
AskUserQuestion:
  "Where would you like to store the notebooks for this project?

   (a) Use an existing folder — provide the folder name (e.g. 'my-project')
   (b) Create a new folder — provide a name and I will create /Users/{full_email}/{name}/notebooks/
   (c) Use default — I will create /Users/{full_email}/{use_case}/notebooks/

  Options: [free text — user picks (a), (b), or (c) and provides a name if needed]"
```

**WAIT for the user to respond. Do NOT create any folders or upload any notebooks until the user answers.**

Set:
- `{project_folder}` = user-provided name, or `{use_case}` if they pick (c)
- `{notebook_base_path}` = `/Users/{full_email}/{project_folder}/notebooks`
- `{experiment_path}` = `/Users/{full_email}/{project_folder}/experiments/{use_case}` *(MLflow experiment — sibling of `notebooks/`, NEVER overlapping)*

> ⚠️ **MLflow path-collision rule.** `{experiment_path}` must be a disjoint sibling of `{notebook_base_path}`. The pattern above is the only safe default — Databricks silently fails to register an experiment whose path is equal to, an ancestor of, or a descendant of a notebook folder, and the failure surfaces deep inside `run_forecast` as `AttributeError: 'NoneType' object has no attribute 'experiment_id'`. Step 1c below re-validates and pre-creates the experiment via the API to fail-fast on any collision. **Never** fall back to legacy patterns like `/Users/{full_email}/mmf/{use_case}` or any path that overlaps `{notebook_base_path}`.

Store these for use in all subsequent steps (notebook paths, job names, tags, **and notebook/job parameters that pass `experiment_path` into `run_forecast`**).

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

### Step 1c: Validate and pre-create the MLflow experiment (MANDATORY — fail-fast)

> ⛔ **This step is not optional.** It exists to prevent the most common silent failure of `mmf_sa.run_forecast`:
>
> ```
> AttributeError: 'NoneType' object has no attribute 'experiment_id'
>   at Forecaster.set_mlflow_experiment ...
>   MlflowClient().get_experiment_by_name(self.conf["experiment_path"]).experiment_id
> ```
>
> The cause is a Workspace **path collision**: `{experiment_path}` either equals, lives inside, or is the parent of a folder that already contains notebooks. MLflow on Databricks cannot reliably create or look up an experiment whose path overlaps a notebook folder, so `get_experiment_by_name(...)` returns `None`. We must validate the path and pre-create the experiment **before** any job is launched.

#### 1c.1 — Disjointness check

`{experiment_path}` must be a strict sibling of `{notebook_base_path}` — not equal, not an ancestor, not a descendant. Run this check (the agent's tool of choice — Python locally or Genie Code):

```python
def _is_inside(child, parent):
    child = child.rstrip("/") + "/"
    parent = parent.rstrip("/") + "/"
    return child == parent or child.startswith(parent)

assert experiment_path != notebook_base_path, (
    "experiment_path equals notebook_base_path — they must be different folders."
)
assert not _is_inside(experiment_path, notebook_base_path), (
    f"experiment_path ({experiment_path}) is INSIDE notebook_base_path "
    f"({notebook_base_path}). MLflow will fail with AttributeError on experiment_id."
)
assert not _is_inside(notebook_base_path, experiment_path), (
    f"notebook_base_path ({notebook_base_path}) is INSIDE experiment_path "
    f"({experiment_path}). MLflow will fail with AttributeError on experiment_id."
)
print(f"Path disjointness OK:\n  notebooks   : {notebook_base_path}\n  experiments : {experiment_path}")
```

If any assertion fails, **stop**, tell the user the chosen folders overlap, and ask them to rename `{project_folder}` (or pick a fully-disjoint custom `{experiment_path}`). Do NOT proceed.

#### 1c.2 — Workspace-object check

The Workspace path that will host the experiment must NOT already exist as a `NOTEBOOK` or `DIRECTORY` containing notebooks.

**Genie Code (Databricks-native) — Python SDK:**

```python
from databricks.sdk import WorkspaceClient
from databricks.sdk.errors import NotFound

w = WorkspaceClient()

try:
    obj = w.workspace.get_status(path=experiment_path)
    if obj.object_type.value == "NOTEBOOK":
        raise RuntimeError(
            f"{experiment_path} already exists as a NOTEBOOK. "
            "Pick a different experiment_path that does not overlap any notebook."
        )
    if obj.object_type.value == "DIRECTORY":
        # Directory is OK only if it is empty or already an MLflow experiment parent.
        children = list(w.workspace.list(path=experiment_path))
        if any(c.object_type.value == "NOTEBOOK" for c in children):
            raise RuntimeError(
                f"{experiment_path} is a workspace folder containing notebooks. "
                "MLflow cannot create an experiment at this path. Pick a disjoint path."
            )
except NotFound:
    pass  # Path does not exist yet — perfect, MLflow will create it.
```

**External agent — Databricks CLI:**

```bash
databricks workspace get-status "{experiment_path}" || true
# If the command prints object_type=NOTEBOOK or a DIRECTORY containing notebooks, abort.
```

#### 1c.3 — Pre-create the experiment via the API

Create the MLflow experiment up-front so any remaining issue surfaces here, not deep inside `run_forecast`. Idempotent — if it already exists, capture the existing experiment_id.

```python
from mlflow.tracking import MlflowClient
from mlflow.exceptions import RestException

client = MlflowClient()

existing = client.get_experiment_by_name(experiment_path)
if existing is not None:
    experiment_id = existing.experiment_id
    print(f"MLflow experiment already exists: {experiment_path} (id={experiment_id})")
else:
    try:
        experiment_id = client.create_experiment(name=experiment_path)
        print(f"Created MLflow experiment: {experiment_path} (id={experiment_id})")
    except RestException as e:
        # RESOURCE_ALREADY_EXISTS race-condition fallback
        existing = client.get_experiment_by_name(experiment_path)
        if existing is None:
            raise RuntimeError(
                f"Failed to create MLflow experiment at {experiment_path}: {e}. "
                "This usually means the path collides with a notebook folder. "
                "Rename the project folder so experiments/ and notebooks/ are siblings, "
                "then retry."
            )
        experiment_id = existing.experiment_id
        print(f"MLflow experiment already exists (race): {experiment_path} (id={experiment_id})")

assert experiment_id is not None, "MLflow experiment_id is None after create — refuse to launch jobs."
```

If this step succeeds, the experiment is registered and `Forecaster.set_mlflow_experiment` will resolve it cleanly inside every job. If it fails, **do NOT launch any job** — instead route the user back to Step 0b to pick a different `{project_folder}` or `{experiment_path}`.

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
| `dynamic_future_numerical` | **Use only if Skill 1 created a scoring table** | Must provide values for **every future `ds`** in the forecast horizon via a scoring table. If missing, the pipeline errors or **silently drops series**. | `[]` |
| `dynamic_future_categorical` | **Use only if Skill 1 created a scoring table** | Same as above — requires known future values for every forecast date | `[]` |

> **When to use `dynamic_future_*`.** If Skill 1 created `{use_case}_scoring_data` with dynamic future regressors, use those column names here and set `{scoring_table}` to `{use_case}_scoring_data`. The scoring table was already validated in Skill 1 (coverage, NULLs, series completeness). If Skill 1 did NOT create a scoring table (user chose univariate mode or only static/historical regressors), always use `[]` for `dynamic_future_*` and `""` for `{scoring_table}`.
>
> **Warning:** When `dynamic_future_*` columns are specified but the scoring table is incomplete, `mmf_sa` either raises an error or silently removes the affected series from the forecast — producing no output for those series with no warning in the final results.

**Use `[]` for all `dynamic_future_*` placeholders and `""` for `{scoring_table}` unless Skill 1 created `{use_case}_scoring_data`.** If the scoring table exists, populate the `dynamic_future_*` lists from the column names carried forward from Skill 1 and set `{scoring_table}` to `{use_case}_scoring_data`.

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
| `{dynamic_future_numerical}` | `[]` unless Skill 1 created `{use_case}_scoring_data` with future numerical regressors. If so, use the column names from Skill 1 (e.g. `["planned_price", "temperature"]`). See [Feature type decision guide](#feature-type-decision-guide). |
| `{dynamic_future_categorical}` | `[]` unless Skill 1 created `{use_case}_scoring_data` with future categorical regressors. If so, use the column names from Skill 1 (e.g. `["promo", "holiday"]`). |
| `{dynamic_historical_numerical}` | Python list; past-only signals for **NeuralForecast** global models (e.g. lagged features), or `[]`. Safe to use. |
| `{dynamic_historical_categorical}` | Python list, or `[]`. Safe to use. |
| `{scoring_table}` | `""` unless Skill 1 created `{use_case}_scoring_data`. If scoring table exists, set to `{use_case}_scoring_data`. Empty string means `scoring_data` = `train_table`. |

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

> **Why orchestrator + run_gpu.** PyTorch allocates CUDA memory that cannot be freed within the same Python process. Running multiple GPU models in sequence causes OOM when the second model loads. The `dbutils.notebook.run()` pattern gives each model a fresh kernel. This is the same pattern used in the [examples folder](https://github.com/databricks-industry-solutions/many-model-forecasting/blob/main/examples/monthly/global_monthly.ipynb).

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
| `{dynamic_future_numerical}` | `[]` unless Skill 1 created `{use_case}_scoring_data` with future numerical regressors. If so, use column names from Skill 1. See [Feature type decision guide](#feature-type-decision-guide). |
| `{dynamic_future_categorical}` | `[]` unless Skill 1 created `{use_case}_scoring_data` with future categorical regressors. If so, use column names from Skill 1. |
| `{dynamic_historical_numerical}` | Past-only signals for **global** models (lag/roll features), or `[]`. Safe to use. |
| `{dynamic_historical_categorical}` | Same — often `[]`. Safe to use. |
| `{scoring_table}` | `""` unless Skill 1 created `{use_case}_scoring_data`. If scoring table exists, set to `{use_case}_scoring_data`. |

Use the template from:
- [mmf_gpu_orchestrator_notebook_template.ipynb](mmf_gpu_orchestrator_notebook_template.ipynb)

Generate up to two orchestrators, saving locally:
- `notebooks/{use_case}/orchestrator_global.ipynb` — if any global models selected
- `notebooks/{use_case}/orchestrator_foundation.ipynb` — if any foundation models selected

#### Frequency-specific configuration (automatic)

`run_forecast` **automatically loads the correct frequency-specific YAML config** from the installed `mmf_sa` package based on the `freq` parameter. No explicit `conf` argument is needed in the generated notebooks.

**Config resolution order (later overrides earlier):**

1. **Base config** — auto-selected by `freq` from the `mmf_sa` package:
   - `freq="H"` → `forecasting_conf_hourly.yaml` (`season_length: 24`, `window_size: 24`)
   - `freq="D"` → `forecasting_conf_daily.yaml` (`season_length: 7`, `window_size: 7`)
   - `freq="W"` → `forecasting_conf_weekly.yaml`
   - `freq="M"` → `forecasting_conf_monthly.yaml` (`season_length: 12`, `window_size: 12`)
2. **User `conf` kwarg** (optional) — a dict, YAML file path, or OmegaConf object merged on top of the base
3. **Explicit keyword arguments** — `active_models`, `prediction_length`, `metric`, etc. override everything

The base configs set `season_length` and `window_size` for models like `StatsForecastAutoArima`, `StatsForecastAutoETS`, `StatsForecastAutoTheta`, etc. The templates pass `freq` as a keyword argument, so the right config is loaded automatically — **do NOT pass `conf` in the generated notebooks**.

The source YAML files live in the `mmf_sa` package directory (e.g., [`mmf_sa/forecasting_conf_daily.yaml`](https://github.com/databricks-industry-solutions/many-model-forecasting/blob/main/mmf_sa/forecasting_conf_daily.yaml)) and are bundled when `mmf_sa` is installed.

**Advanced: overriding model hyperparameters.** If the user wants to tune `season_length` or other model-specific parameters, pass a `conf` dict to `run_forecast` that overrides only the desired keys. The base config's defaults still apply for everything else:

```python
custom_conf = {
    "models": {
        "StatsForecastAutoArima": {
            "model_spec": {
                "season_length": 52  # override for weekly with yearly seasonality
            }
        }
    }
}

run_forecast(
    ...,
    conf=custom_conf,
)
```

**Do NOT pass `conf` by default.** Only add it when the user explicitly requests hyperparameter tuning. The auto-selected base config is correct for the vast majority of use cases.

#### Covariates and model coverage (reference)

> **Default: univariate mode.** All model families work well without covariates. Generate notebooks with `[]` for every covariate list and `""` for `scoring_table` unless Skill 1 configured exogenous regressors and created `{use_case}_scoring_data`. See [Feature type decision guide](#feature-type-decision-guide).

`mmf_sa.run_forecast` maps columns into StatsForecast / NeuralForecast / Chronos / TimesFM pipelines:

- **StatsForecast (local)**: runs univariate by default. Can accept `dynamic_future_*` covariates but these require a scoring table — **avoid unless the user explicitly provides one**.
- **NeuralForecast (global)**: benefits from **`static_features`** and **`dynamic_historical_*`** (past-only signals used during training). Can also accept `dynamic_future_*` but same caveat applies.
- **Foundation (Chronos / TimesFM)**: benefits from **`static_features`**. Can accept `dynamic_future_*` in the current implementation but does not use `dynamic_historical_*`.

**If `{use_case}_scoring_data` exists from Skill 1**, set `{scoring_table}` to `{use_case}_scoring_data` and populate `dynamic_future_*` lists with the column names carried forward. The scoring table was already validated in Skill 1. See also [run_external_regressors_daily.ipynb](https://github.com/databricks-industry-solutions/many-model-forecasting/blob/main/examples/run_external_regressors_daily.ipynb) for the reference pattern.

### Step 4: Import notebooks into Databricks workspace

The notebooks **must** be imported as proper NOTEBOOK objects (not FILEs). Databricks job tasks require this — a FILE will cause every job to fail immediately with: `'<path>' is not a notebook`.

Use the method available in your environment:

**External agent (Claude Code, Cursor, Copilot, etc.) — Databricks CLI:**

```bash
# Local models notebook
databricks workspace import {notebook_base_path}/run_local \
  --file /tmp/{use_case}_run_local.ipynb \
  --format JUPYTER --overwrite

# GPU run notebook (static — uploaded verbatim)
databricks workspace import {notebook_base_path}/run_gpu \
  --file /tmp/{use_case}_run_gpu.ipynb \
  --format JUPYTER --overwrite

# Global orchestrator (if global models selected)
databricks workspace import {notebook_base_path}/orchestrator_global \
  --file /tmp/{use_case}_orchestrator_global.ipynb \
  --format JUPYTER --overwrite

# Foundation orchestrator (if foundation models selected)
databricks workspace import {notebook_base_path}/orchestrator_foundation \
  --file /tmp/{use_case}_orchestrator_foundation.ipynb \
  --format JUPYTER --overwrite
```
> ⚠️ Do NOT use the `upload_file` MCP tool — it creates a FILE, not a NOTEBOOK. If a path already exists as a FILE from a prior failed upload, delete it first before importing.

**Genie Code (Databricks-native) — Python SDK:**

```python
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.workspace import ImportFormat
import base64

w = WorkspaceClient()

notebooks = {
    "run_local": "/tmp/{use_case}_run_local.ipynb",
    "run_gpu": "/tmp/{use_case}_run_gpu.ipynb",
    "orchestrator_global": "/tmp/{use_case}_orchestrator_global.ipynb",      # if global models selected
    "orchestrator_foundation": "/tmp/{use_case}_orchestrator_foundation.ipynb",  # if foundation models selected
}

for name, local_path in notebooks.items():
    with open(local_path, "rb") as f:
        content = base64.b64encode(f.read()).decode("utf-8")
    w.workspace.import_(
        path=f"{notebook_base_path}/{name}",
        format=ImportFormat.JUPYTER,
        overwrite=True,
        content=content
    )
```

### Step 4b: Verify all notebooks imported correctly

**External agent — Databricks CLI:**
```bash
databricks workspace list {notebook_base_path}/
```

**Genie Code — Python SDK:**
```python
for item in w.workspace.list(path="{notebook_base_path}/"):
    print(item.path, item.object_type)
```

Confirm every path shows `object_type: NOTEBOOK` (not `FILE`). Fix any mismatches before creating jobs.

The local copies in `notebooks/{use_case}/` serve as version-controllable artifacts of the generated pipeline.

### Step 5: Upsert one job per model class

Job names follow the pattern `{use_case}_{type}_forecasting_{username}` (no date — one persistent job per type per user):
- Local: `{use_case}_local_forecasting_{username}`
- Global: `{use_case}_global_forecasting_{username}`
- Foundation: `{use_case}_foundation_forecasting_{username}`

For each model class selected, **upsert** the job:
1. Search for an existing job with that exact name owned by `{full_email}`
2. If found → **update** it with the new notebook path and cluster config
3. If not found → **create** it

This ensures there is always exactly one job per type per user — no accumulation of stale jobs.

> ⚠️ **One job per model class — no exceptions.** Do NOT combine model classes into a single multi-task job. Local, global, and foundation models require different compute (CPU vs GPU) and must run independently in parallel.

---

### Step 5b: Create one job per model class (triggered in parallel)

> ⚠️ **One job per model class — no exceptions.** Do NOT combine model classes into a single multi-task job. Create separate jobs for local, global, and foundation models. This is required for correct cluster assignment (CPU vs GPU) and independent parallelism.

> ⛔ **MANDATORY `spark_version` per job class — DO NOT SUBSTITUTE.**
> The three JSON templates below have intentionally different `spark_version` values. They are **not interchangeable** and must be used exactly as written:
>
> | Job | `spark_version` (required) |
> |---|---|
> | Local models | `17.3.x-cpu-ml-scala2.13` |
> | Global models | `18.0.x-gpu-ml-scala2.13` |
> | Foundation models | `18.0.x-gpu-ml-scala2.13` |
> | NF Local models *(if `separate_job`)* | `17.3.x-cpu-ml-scala2.13` |
> | NF Global / NF Foundation *(if `separate_job`)* | `18.0.x-gpu-ml-scala2.13` |
>
> The agent is FORBIDDEN from:
> - Reusing the local job's `spark_version` (`17.3.x-cpu-ml-scala2.13`) inside the global or foundation JSON. A GPU node type with a CPU runtime fails to start or silently runs on CPU.
> - Substituting `17.3.x-gpu-ml-scala2.13` for the GPU jobs because it "looks like LTS." The GPU pipeline is pinned to **18.0** and `mmf_sa[global]` / `mmf_sa[foundation]` are tested against it.
> - Using a non-ML runtime (anything not ending in `-ml-scala2.13`).
> - Copy-pasting one job's `job_clusters` block into another. Build each job's `job_clusters` block from its own template.
>
> **Step 5c (immediately after job creation) reads back the `spark_version` of every job cluster via `get_job` and aborts if any does not match this table. The verification is not optional.**

All jobs must include:
- `tags`: `{ "mmf-agent": "", "databricks-ai-dev-kit": "" }`
- `description`: human-readable summary — e.g. `"MMF {type} forecasting | use_case={use_case} | catalog={catalog}.{schema} | models={active_models} | horizon={prediction_length} | runtime={spark_version} | created={YYYYMMDD}"`

**Create separate Workflow jobs for each model class and trigger them all in parallel.** This maximizes throughput — local models run on CPU while GPU models run concurrently.

#### Job 1: Local models (if any local models selected)

```json
{
  "name": "{use_case}_local_forecasting_{username}",
  "description": "MMF local forecasting | use_case={use_case} | catalog={catalog}.{schema} | models={active_local_models} | horizon={prediction_length} | created={YYYYMMDD}",
  "tags": {
    "aidevkit_project": "mmf-agent",
    "created_by": "databricks-ai-dev-kit"
  },
  "tasks": [{
    "task_key": "local_models",
    "notebook_task": {
      "notebook_path": "{notebook_base_path}/run_local",
      "base_parameters": {
        "experiment_path": "{experiment_path}"
      }
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
  "name": "{use_case}_global_forecasting_{username}",
  "description": "MMF global forecasting | use_case={use_case} | catalog={catalog}.{schema} | models={active_global_models} | horizon={prediction_length} | created={YYYYMMDD}",
  "tags": {
    "aidevkit_project": "mmf-agent",
    "created_by": "databricks-ai-dev-kit"
  },
  "tasks": [{
    "task_key": "global_models",
    "notebook_task": {
      "notebook_path": "{notebook_base_path}/orchestrator_global",
      "base_parameters": {
        "experiment_path": "{experiment_path}"
      }
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
  "name": "{use_case}_foundation_forecasting_{username}",
  "description": "MMF foundation forecasting | use_case={use_case} | catalog={catalog}.{schema} | models={active_foundation_models} | horizon={prediction_length} | created={YYYYMMDD}",
  "tags": {
    "aidevkit_project": "mmf-agent",
    "created_by": "databricks-ai-dev-kit"
  },
  "tasks": [{
    "task_key": "foundation_models",
    "notebook_task": {
      "notebook_path": "{notebook_base_path}/orchestrator_foundation",
      "base_parameters": {
        "experiment_path": "{experiment_path}"
      }
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

Use `create_job` (or `update_job` on upsert) to create each job. **After `create_job` / `update_job` and BEFORE `run_job`, run Step 5c.**

### Step 5c: Verify each job's cluster runtime (MANDATORY)

> ⛔ **This step is not optional and cannot be skipped.** It exists to catch the most common failure mode: the agent copying the local job's `spark_version` into the global or foundation JSON (e.g. using `17.3.x-cpu-ml-scala2.13` for a GPU cluster). The verification reads the runtime back from the API after job creation, so it does not depend on the agent's intent.

For every job created in Step 5b (and Step 5a if `separate_job`), call `get_job` and inspect each entry in `job_clusters[*].new_cluster.spark_version`. Required values:

| Job name | Required `spark_version` |
|---|---|
| `{use_case}_local_forecasting_{username}` | `17.3.x-cpu-ml-scala2.13` |
| `{use_case}_global_forecasting_{username}` | `18.0.x-gpu-ml-scala2.13` |
| `{use_case}_foundation_forecasting_{username}` | `18.0.x-gpu-ml-scala2.13` |
| `{use_case}_nf_local_forecasting_{username}` | `17.3.x-cpu-ml-scala2.13` |
| `{use_case}_nf_global_forecasting_{username}` | `18.0.x-gpu-ml-scala2.13` |
| `{use_case}_nf_foundation_forecasting_{username}` | `18.0.x-gpu-ml-scala2.13` |

For each job:

1. Call `get_job(job_id=...)`.
2. For each cluster in `settings.job_clusters[*]`, check `new_cluster.spark_version`.
3. Compare to the expected value from the table above.
4. If **any** value does not match, do NOT call `run_job`. Instead:
   - Print a clear error including the job name, the cluster key, the actual `spark_version`, and the expected `spark_version`.
   - Call `update_job` with the corrected JSON template from Step 5b (or 5a) — using the exact `spark_version` from the table above.
   - Re-run Step 5c to verify the fix.
   - Only after the check passes for **all** jobs may the agent proceed to `run_job`.

Report a one-line summary to the user before triggering runs:

```
Cluster runtime check:
  • {use_case}_local_forecasting_{username}      → 17.3.x-cpu-ml-scala2.13  ✓
  • {use_case}_global_forecasting_{username}     → 18.0.x-gpu-ml-scala2.13  ✓
  • {use_case}_foundation_forecasting_{username} → 18.0.x-gpu-ml-scala2.13  ✓
All runtimes match the pinned versions. Proceeding to run_job.
```

Then call `run_job` for **all jobs simultaneously**.

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
      "notebook_path": "notebooks/{use_case}/run_local_nf",
      "base_parameters": {
        "experiment_path": "{experiment_path}"
      }
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
      "notebook_path": "notebooks/{use_case}/orchestrator_global_nf",
      "base_parameters": {
        "experiment_path": "{experiment_path}"
      }
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
   • MLflow experiment: {experiment_path}

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
- MLflow experiment at `{experiment_path}` (default: `/Users/{full_email}/{project_folder}/experiments/{use_case}`) — pre-created and validated in Step 1c, never overlapping `{notebook_base_path}`

**Produced when `separate_job` strategy:**
- Delta table `<catalog>.<schema>.{use_case}_nf_evaluation_output` — backtest metrics for non-forecastable series
- Delta table `<catalog>.<schema>.{use_case}_nf_scoring_output` — forward-looking forecasts for non-forecastable series

## ⛔ Step-transition gate — Ask the user before moving on

After all forecast jobs have been triggered (or completed), the agent MUST stop and ask before starting Skill 5. **Do NOT auto-advance.**

```
AskUserQuestion:
  "Skill 4 (Execute MMF Forecast) is complete.

  Triggered jobs:
    {list of job names with run URLs}

  Outputs (once jobs finish):
    • {catalog}.{schema}.{use_case}_evaluation_output
    • {catalog}.{schema}.{use_case}_scoring_output

  Ready to proceed to Skill 5 (Post-Process and Evaluate) once jobs complete?
    (a) Yes — wait for jobs, then run Skill 5
    (b) Run Skill 5 now (jobs are already complete)
    (c) Stop here for now"
  Options: [a, b, c]
```
