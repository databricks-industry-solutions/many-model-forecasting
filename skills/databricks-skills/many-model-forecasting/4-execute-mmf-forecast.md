# Execute MMF Forecast

> ⛔ **MANDATORY:** If you have not read [SKILL.md](SKILL.md) yet, read it now before proceeding. Do NOT take any action until you have read both SKILL.md and this file in full.

**Slash command:** `/execute-mmf-forecast`

Validates parameters, asks the user about backtesting setup, generates notebooks
(the **orchestrator + run_gpu** pattern for GPU classes on classic GPU clusters, a
single **run_serverless_{class}** notebook for GPU classes on serverless GPU), creates
**one job per model class** (local, global_ml, global, foundation, foundationforecast),
and triggers them **in parallel**. With calibrated prediction intervals (Step 1d option (d)), it also creates a CPU
calibration job, and an orchestrating job that runs the forecast jobs in parallel and the calibration after them.

## ⛔ Preconditions — DO NOT SKIP

> **This skill is NOT the entry point of the MMF workflow.** Even if the user said "let's forecast" or "use MMF" or "run the forecast," the agent must NOT start here unless ALL preconditions below are satisfied. If any are missing, route the user back to the earliest unmet skill — do not improvise inputs.

| Precondition | How to verify | If missing |
|---|---|---|
| `{catalog}.{schema}.{use_case}_train_data` exists (or `_train_data_forecastable` for fallback/separate_job) | `get_table` or `SELECT 1 FROM ... LIMIT 1` | Go back to **Skill 1 (`/prep-and-clean-data`)** |
| `active_models` has been chosen by the user (grouped by local / global_ml / global / foundation / foundationforecast) | Check conversation state | Go back to **Skill 3 (`/provision-forecasting-resources`)** |
| Cluster config, including the compute type (classic GPU or serverless GPU) of each GPU class, has been confirmed by the user | Check conversation state | Go back to **Skill 3** |
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
| `prediction_intervals` | `None` (disabled) | Prediction-interval settings, e.g. `{"enabled": True, "levels": [80, 95]}` — see Step 1d |

## Available model names

Use these exact names in the `active_models` parameter:

**Local models (CPU multi-node):** `StatsForecastBaselineWindowAverage`, `StatsForecastBaselineSeasonalWindowAverage`, `StatsForecastBaselineNaive`, `StatsForecastBaselineSeasonalNaive`, `StatsForecastAutoArima`, `StatsForecastAutoETS`, `StatsForecastAutoCES`, `StatsForecastAutoTheta`, `StatsForecastAutoTbats`, `StatsForecastAutoMfles`, `StatsForecastTSB`, `StatsForecastADIDA`, `StatsForecastIMAPA`, `StatsForecastCrostonClassic`, `StatsForecastCrostonOptimized`, `StatsForecastCrostonSBA`, `SKTimeProphet`

**Global ML models (CPU single-node):** `MLForecastLGBM`, `MLForecastAutoLGBM`

**Global DL models (GPU single-node):** `NeuralForecastRNN`, `NeuralForecastLSTM`, `NeuralForecastNBEATSx`, `NeuralForecastNHITS`, `NeuralForecastAutoRNN`, `NeuralForecastAutoLSTM`, `NeuralForecastAutoNBEATSx`, `NeuralForecastAutoNHITS`, `NeuralForecastAutoTiDE`, `NeuralForecastAutoPatchTST`

**Foundation models (GPU single-node):** `ChronosBoltTiny`, `ChronosBoltMini`, `ChronosBoltSmall`, `ChronosBoltBase`, `Chronos2`, `Chronos2Small`, `Chronos2Synth`, `TimesFM_2_5_200m`

**Foundationforecast models (GPU single-node, one GPU):** `FFChronos2`, `FFChronos2Small`, `FFTimesFM_2_5_200m`, `FFTimesFM_3_0`, `FFTiRex2`, `FFToto`, `FFToto2_22m`, `FFToto2_313m`, `FFToto2_1B`, `FFFlowState`, `FFPatchTSTFM_R2`, `FFPatchTSTFM_R1`, `FFTafsutBase`, `FFT0Beta`, `FFT0Gamma`, `FFTimestral_5M`, `FFTimestral_28M`, `FFTimestral_123M`, `FFTimestral_209M`. These are pretrained models run through the [foundationforecast](https://github.com/TimeCopilot/foundationforecast) package. They need a different torch build from the foundation models, so they always run in their own job. Ten of them use known-future covariates (see the [Feature type decision guide](#feature-type-decision-guide)).

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
| `separate_job` | `{use_case}_train_data_forecastable` | Separate jobs created in Step 5b below |

Set `{train_table}` to the correct table name for all subsequent notebook generation steps.

If Skill 1 created `{use_case}_scoring_data`, route the scoring table the same way: `{use_case}_scoring_data` for `include`, `{use_case}_scoring_data_forecastable` for `fallback` and `separate_job` (Skill 2 Step 10 creates it). This is the `{scoring_table}` value below. A scoring table holding series that are not in `{train_table}` breaks the run (Skill 1 Step 8d).

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
| `FF*` models selected → a foundationforecast compute (classic `{use_case}_ff_gpu_cluster` or serverless GPU) is configured | Mandatory | "Foundationforecast models need their own GPU compute; they cannot share the foundation models' environment" |
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

### ⛔ STOP GATE — Step 1d: Ask user about prediction intervals

**Always ask. Do NOT proceed until the user answers.**

```
AskUserQuestion:
  "Do you want prediction intervals (lower/upper bounds) with the forecasts?

   (a) No — point forecasts only (default)

   (b) Yes, 80% and 95% native intervals, for the models that have them
       → prediction_intervals = {"enabled": True, "levels": [80, 95]}

   (c) Yes, custom levels or quantiles
       → e.g. {"enabled": True, "levels": [90]} or {"enabled": True, "quantiles": [0.1, 0.5, 0.9]}

   (d) Yes, calibrated intervals (80% and 95%, or custom levels) for every model
       → the native intervals of (b)/(c), plus a calibration job on CPU after the forecast jobs"
```

Tell the user clearly what they get: the models' **native** intervals, **uncalibrated**, and only for the models that have them: statsforecast AutoETS, AutoArima, AutoCES, AutoTheta, AutoTbats, Naive and SeasonalNaive; the Chronos and TimesFM models; SKTimeProphet; and the FoundationForecast FF* models. All other models (e.g. CrostonSBA, ADIDA, MLForecast, NeuralForecast) get NULL interval columns. The coverage of the native bands can be off nominal; Skill 5 measures it on the backtest. Some models only have quantiles from 10% to 90% (Chronos-Bolt, TimesFM, several FF* models): their 95% bounds are NULL. The intervals land in `scoring_output` as a `y_quantiles` map, and in `evaluation_output` as `forecast_quantiles` for each backtest window.

For option (d), add: the calibration job corrects the native bands (or, for models without them, builds intervals
from the backtest errors) so that they reach the requested coverage on the backtest, for every model. It runs
`mmf_sa.calibrate_intervals` on CPU after all forecast jobs, reading only their tables, and writes
`{use_case}_scoring_output_intervals` (the calibrated intervals) plus `{use_case}_evaluation_output_calibration` and
`{use_case}_evaluation_output_interval_metrics` (out-of-sample coverage, which Skill 5 reports). It needs at least 2
backtest windows, and 3 or more give better per-series intervals (Step 1b).

- Option (a): set `{prediction_intervals}` to `None` in every notebook.
- Options (b)/(c): set `{prediction_intervals}` to the Python dict literal in every notebook (local, global ML, the GPU orchestrators and the serverless GPU notebooks; the Step 3e non-forecastable notebooks use the same value). If the user wants bound columns for SQL or dashboards, add `"output_format": "both"` (adds `y_lo_80`, `y_hi_80`, ...).
- Option (d): do the same as (b)/(c), so that the models with native intervals write their bands for the calibration, and set `{calibrate_intervals}` to `True`: Step 3f generates the calibration notebook with the same levels (`{calibration_levels}`), and Step 5b creates the calibration and orchestrating jobs. For options (a)-(c), `{calibrate_intervals}` is `False`.

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

Creating an experiment fails with `Parent directory does not exist` if its parent folder (`/Users/{full_email}/{project_folder}/experiments`) is missing, so create the folder first: `databricks workspace mkdirs /Users/{full_email}/{project_folder}/experiments` (CLI) or `w.workspace.mkdirs(...)` (SDK).

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

`mmf_sa` supports five covariate types. **By default, always use `[]` for every covariate list and `""` for `scoring_table`** — the pipeline runs in univariate mode and all model families (local, global_ml, global_dl, foundation, foundationforecast) work well without covariates.

| Feature type | Recommended? | When values are needed | Safe default |
|---|---|---|---|
| `static_features` | **Yes** — use when available | Constant per `group_id`, always known (e.g. `store_state`, `dept_id`) | `[]` |
| `dynamic_historical_numerical` | **Yes** — for global_ml and global_dl models | Past-only; NOT needed at forecast time (e.g. lagged sales, rolling averages) | `[]` |
| `dynamic_historical_categorical` | **Yes** — for global_ml and global_dl models | Past-only; NOT needed at forecast time | `[]` |
| `dynamic_future_numerical` | **Use only if Skill 1 created a scoring table** | Must provide values for **every future `ds`** in the forecast horizon via a scoring table. If missing, the pipeline errors or **silently drops series**. | `[]` |
| `dynamic_future_categorical` | **Use only if Skill 1 created a scoring table** | Same as above — requires known future values for every forecast date | `[]` |

> **When to use `dynamic_future_*`.** If Skill 1 created `{use_case}_scoring_data` with dynamic future regressors, use those column names here and set `{scoring_table}` as routed in Step 0a. The scoring table was already validated in Skill 1 (coverage, NULLs, series completeness). If Skill 1 did NOT create a scoring table (user chose univariate mode or only static/historical regressors), always use `[]` for `dynamic_future_*` and `""` for `{scoring_table}`.
>
> **Strongest support: `global_ml`.** When `dynamic_future_*` regressors are configured, MLForecast LightGBM (`MLForecastLGBM`, `MLForecastAutoLGBM`) has the most thoroughly validated end-to-end coverage among MMF model classes: the eval path builds a complete per-(unique_id, future_ds) `X_df` skeleton from the `new_df`'s last dates and left-merges the future regressors; the scoring path consumes the unioned `train + scoring_data` table identically. If your project relies on known future regressors, prefer `global_ml` as the primary class (it's CPU-only, so cost is also moderate). NeuralForecast variants in `global_dl` and `Chronos*` foundation models accept `dynamic_future_*` too but with framework-specific caveats.
>
> **Zero-shot alternative: `foundationforecast`.** Ten `FF*` models use `dynamic_future_*` covariates without training: `FFChronos2`, `FFChronos2Small`, `FFTimesFM_3_0`, `FFTiRex2`, `FFT0Beta`, `FFT0Gamma`, `FFTimestral_5M`, `FFTimestral_28M`, `FFTimestral_123M`, `FFTimestral_209M`. Run them next to `global_ml` when known future regressors exist. Their rules are stricter than the other classes':
> - They use **only** `dynamic_future_*`. They ignore `static_features` and `dynamic_historical_*`. The other nine `FF*` models ignore all covariates and log a warning.
> - Every series needs a covariate value for each of the `prediction_length` dates after its last `ds`, in the scoring table. A missing date, a duplicate date or a NULL value **raises an error** instead of dropping the series.
> - Only `FFChronos2` and `FFChronos2Small` accept string categoricals. The other covariate models need numeric columns, so pass categorical columns to them already encoded as numbers (Skill 1 Step 6b).
> - Weekly series must all fall on the same weekday.
> - `FFTimesFM_3_0` and `FFPatchTSTFM_R1` have non-commercial licenses, and the `FFTimestral_*` checkpoints need a commercial license from Mistral above USD 20M monthly revenue. Mention this when the user selects them.
>
> **Warning:** When `dynamic_future_*` columns are specified but the scoring table is incomplete, `mmf_sa` either raises an error or silently removes the affected series from the forecast — producing no output for those series with no warning in the final results.

**Use `[]` for all `dynamic_future_*` placeholders and `""` for `{scoring_table}` unless Skill 1 created `{use_case}_scoring_data`.** If the scoring table exists, populate the `dynamic_future_*` lists from the column names carried forward from Skill 1 and set `{scoring_table}` as routed in Step 0a.

### Step 3: Generate notebooks

**CRITICAL: Copy templates VERBATIM, only replacing `{placeholder}` tokens with actual values. Do NOT add, remove, or modify any other code. The templates are complete and production-ready.**

> ℹ️ **`run_id` is per-job, not shared.** Each model-class job calls `run_forecast`, which
> generates its **own** `run_id` (UUID) internally. Do **not** try to force a single shared
> `run_id` across jobs — Skill 5 handles this correctly by selecting the **latest `run_id` per
> model** so all models compete together. (An earlier version pinned Skill 5 to a single global
> latest run, which silently excluded every model except the last job to finish.)

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
| `{scoring_table}` | `""` unless Skill 1 created `{use_case}_scoring_data`. If scoring table exists, set it as routed in Step 0a (`{use_case}_scoring_data` or `{use_case}_scoring_data_forecastable`). Empty string means `scoring_data` = `train_table`. |
| `{prediction_intervals}` | `None`, or the Python dict literal chosen in Step 1d, e.g. `{"enabled": True, "levels": [80, 95]}` |

Use the template from:
- [mmf_local_notebook_template.ipynb](mmf_local_notebook_template.ipynb) → save locally as `notebooks/{use_case}/run_local.ipynb`

#### 3a-bis: Global ML models notebook (if any MLForecast model selected)

Generate a single notebook from `mmf_global_ml_notebook_template.ipynb` with all MLForecast models in `{active_models}`. Same all-models-in-one-shot pattern as `run_local` because LightGBM has no CUDA-memory accumulation problem — both `MLForecastLGBM` and `MLForecastAutoLGBM` can run sequentially in the same Python process.

##### Placeholder values for global ML notebook

Identical to the local notebook placeholders, with `{active_models}` filtered to the MLForecast models only. The only structural difference vs `run_local` is the install line (`mmf_sa[global]` vs `mmf_sa[local]`).

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
| `{active_models}` | Python list literal of MLForecast models only, e.g. `["MLForecastLGBM", "MLForecastAutoLGBM"]` |
| `{group_id}`, `{date_col}`, `{target}`, `{use_case}` | as for the local notebook |
| `{static_features}` | Same as local — can populate when known per-series static columns exist |
| `{dynamic_future_numerical}`, `{dynamic_future_categorical}` | **Populate these when Skill 1 created `{use_case}_scoring_data`** — `global_ml` has the most thoroughly validated `dynamic_future_*` support of any class |
| `{dynamic_historical_numerical}`, `{dynamic_historical_categorical}` | Past-only signals; safe to use |
| `{scoring_table}` | `""` unless Skill 1 created `{use_case}_scoring_data`. If so, set it as routed in Step 0a. |
| `{prediction_intervals}` | Same as the local notebook (Step 1d) |

Use the template from:
- [mmf_global_ml_notebook_template.ipynb](mmf_global_ml_notebook_template.ipynb) → save locally as `notebooks/{use_case}/run_global_ml.ipynb`

> **Which notebooks each GPU class gets** depends on the compute type the user chose for it in Skill 3:
> - **Classic GPU cluster** → `run_gpu` (3b) + one orchestrator per class (3c).
> - **Serverless GPU** → one `run_serverless_{class}` notebook per class (3d). No `run_gpu` or orchestrator for that class.
>
> Generate `run_gpu` only if at least one GPU class runs on a classic cluster.

#### 3b: GPU run notebook (static — no placeholder substitution)

Copy the `mmf_gpu_run_notebook_template.ipynb` **as-is** to `notebooks/{use_case}/run_gpu.ipynb`. This notebook:
- Receives all parameters via `dbutils.widgets` (catalog, schema, model, run_id, etc.)
- Auto-detects whether the model is global DL, foundation or foundationforecast and installs the correct `mmf_sa` extras (foundationforecast: `mmf_sa[foundationforecast]` with the CUDA 12.8 torch build)
- Runs a single model per invocation
- Is called by the orchestrator notebooks via `dbutils.notebook.run()`

**Do NOT modify this template.** Copy it verbatim.

Use the template from:
- [mmf_gpu_run_notebook_template.ipynb](mmf_gpu_run_notebook_template.ipynb) → save locally as `notebooks/{use_case}/run_gpu.ipynb`

#### 3c: GPU orchestrator notebooks (one per model class)

For each GPU model class on a classic GPU cluster (global, foundation and/or foundationforecast), generate an **orchestrator notebook** from `mmf_gpu_orchestrator_notebook_template.ipynb`. Each orchestrator:
- Holds the list of active models for its class
- Loops through the models and calls `run_gpu` for each via `dbutils.notebook.run()`
- Each `dbutils.notebook.run()` invocation gets a fresh Python process, avoiding CUDA memory conflicts

> **Why orchestrator + run_gpu.** PyTorch allocates CUDA memory that cannot be freed within the same Python process. Running multiple GPU models in sequence causes OOM when the second model loads. The `dbutils.notebook.run()` pattern gives each model a fresh kernel. This is the same pattern used in the [examples folder](https://github.com/databricks-industry-solutions/many-model-forecasting/blob/main/examples/monthly/global_monthly_dl.ipynb).

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
| `{scoring_table}` | `""` unless Skill 1 created `{use_case}_scoring_data`. If scoring table exists, set it as routed in Step 0a. |
| `{prediction_intervals}` | Same as the local notebook (Step 1d). The orchestrator forwards it to each `run_gpu` call as JSON. |

Use the template from:
- [mmf_gpu_orchestrator_notebook_template.ipynb](mmf_gpu_orchestrator_notebook_template.ipynb)

Generate up to three orchestrators, saving locally:
- `notebooks/{use_case}/orchestrator_global.ipynb` — if any global models selected
- `notebooks/{use_case}/orchestrator_foundation.ipynb` — if any foundation models selected
- `notebooks/{use_case}/orchestrator_foundationforecast.ipynb` — if any `FF*` models selected

#### 3d: Serverless GPU notebooks (one per GPU class on serverless GPU)

For each GPU class the user put on serverless GPU in Skill 3, generate one notebook from `mmf_gpu_serverless_notebook_template.ipynb`. It installs the class's packages once, then calls `run_forecast(..., serverless=True)` once per model, in the same notebook.

> **Why not orchestrator + run_gpu on serverless.** Serverless GPU tasks get their environment from the job's `environments` spec and `%pip`; `dbutils.notebook.run()` child notebooks on serverless GPU have not been validated with MMF. Calling `run_forecast` once per model in one notebook is the tested pattern: MMF releases each model's weights before loading the next. `serverless=True` moves foundation and foundationforecast inference to the driver's GPU, because serverless Spark workers have no GPU.

Placeholders: the same as the orchestrator table above (without `{num_nodes}`), plus `{pip_install_args}`. Fill `{pip_install_args}` with exactly one of these, by class:

| Class | `{pip_install_args}` |
|---|---|
| Global DL | `"mmf_sa[global] @ git+https://github.com/databricks-industry-solutions/many-model-forecasting.git@main"` |
| Foundation | `"mmf_sa[foundation] @ git+https://github.com/databricks-industry-solutions/many-model-forecasting.git@main"` |
| Foundationforecast | `--extra-index-url https://download.pytorch.org/whl/cu128 "mmf_sa[foundationforecast] @ git+https://github.com/databricks-industry-solutions/many-model-forecasting.git@main" torch==2.11.0+cu128 torchvision==0.26.0+cu128 torchaudio==2.11.0+cu128` |

Use the template from:
- [mmf_gpu_serverless_notebook_template.ipynb](mmf_gpu_serverless_notebook_template.ipynb)

Save locally, one per serverless class:
- `notebooks/{use_case}/run_serverless_global.ipynb`
- `notebooks/{use_case}/run_serverless_foundation.ipynb`
- `notebooks/{use_case}/run_serverless_foundationforecast.ipynb`

#### 3e: Non-forecastable notebooks (if `separate_job` strategy)

**Only generate these if `non_forecastable_strategy == 'separate_job'`.** Skip for `include` or `fallback`.

Generate notebooks for the non-forecastable pipeline using the same templates, but with:
- `{train_table}` → `{use_case}_train_data_non_forecastable`
- `{scoring_table}` → `{use_case}_scoring_data_non_forecastable` if Skill 1 created a scoring table, otherwise `""`
- `{active_models}` → the non-forecastable models selected in Skill 2/3
- `{use_case}` output table suffix → `{use_case}_nf` (to avoid overwriting main pipeline outputs)

The evaluation and scoring output tables for non-forecastable series are:
- `{use_case}_nf_evaluation_output`
- `{use_case}_nf_scoring_output`

##### Non-forecastable local models notebook

If the non-forecastable models include local (CPU multi-node) models, generate a notebook from `mmf_local_notebook_template.ipynb`:
- Save locally as `notebooks/{use_case}/run_local_nf.ipynb`; Step 4 imports it as `{notebook_base_path}/run_local_nf`

##### Non-forecastable global ML models notebook

If the non-forecastable models include any `MLForecast*` model, generate a notebook from `mmf_global_ml_notebook_template.ipynb`:
- Save locally as `notebooks/{use_case}/run_global_ml_nf.ipynb`; Step 4 imports it as `{notebook_base_path}/run_global_ml_nf`

Identical placeholder substitution to the main `run_global_ml` notebook, except `{train_table}` points at `{use_case}_train_data_non_forecastable` and the output tables use the `_nf_` infix (see the start of 3e).

##### Non-forecastable GPU orchestrator notebooks

If the non-forecastable models include global DL, foundation or foundationforecast models, generate orchestrator notebooks:
- `notebooks/{use_case}/orchestrator_global_nf.ipynb` — if any global DL models
- `notebooks/{use_case}/orchestrator_foundation_nf.ipynb` — if any foundation models
- `notebooks/{use_case}/orchestrator_foundationforecast_nf.ipynb` — if any `FF*` models

These orchestrators call the same `run_gpu` notebook (shared with the main pipeline).

If a GPU class runs on serverless GPU, generate `notebooks/{use_case}/run_serverless_{class}_nf.ipynb` from the serverless template instead, with the same `{train_table}` and `{use_case}_nf` substitutions. Step 5b creates its job from the [serverless GPU job template](#serverless-gpu-jobs-global-dl-foundation-foundationforecast) with job name `{use_case}_nf_{class}_forecasting_{username}`.

#### 3f: Calibration notebook (only if `{calibrate_intervals}`, Step 1d option (d))

Generate one notebook from `mmf_calibration_notebook_template.ipynb`. It calls `mmf_sa.calibrate_intervals` on the
tables the forecast jobs write, and fails if a model could not be calibrated. It installs `mmf_sa` without extras:
it needs no model library and no GPU.

| Placeholder | Value |
|---|---|
| `{catalog}`, `{schema}`, `{use_case}`, `{freq}`, `{group_id}`, `{date_col}`, `{target}` | Same as the forecast notebooks |
| `{train_table}` | Same as the forecast notebooks (Step 0a): the series' scale is computed from it |
| `{calibration_levels}` | Python list of the Step 1d levels, e.g. `[80, 95]` |
| `{output_format}` | `quantile_map`, `lo_hi` or `both`, as in `{prediction_intervals}` (default `quantile_map`) |
| `{calibrate_non_forecastable}` | `True` if `separate_job` and the Step 3e notebooks have prediction intervals, else `False` |
| `{pool}` | `marginal` (default): all series in one pool. The CQR corrections are pooled with no per-series shrinkage; the conformal intervals shrink each series' tails toward the pool. `group` only if the user asks to pool by a column of the training data |
| `{group_col}` | `None` (default), or with `{pool}` = `group` the column name as a string literal (e.g. `"category"`), or `"auto"` for MMF's ABC x XYZ classes. Tell the user that ABC x XYZ is validated on M5 retail demand only, and is meaningless for targets that are not volumes |

Use the template from:
- [mmf_calibration_notebook_template.ipynb](mmf_calibration_notebook_template.ipynb) → save locally as `notebooks/{use_case}/calibrate_intervals.ipynb`

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
- **Foundationforecast (`FF*`)**: ten models use **`dynamic_future_*`** only; the rest are univariate. See the foundationforecast note in the [Feature type decision guide](#feature-type-decision-guide).

**If `{use_case}_scoring_data` exists from Skill 1**, set `{scoring_table}` as routed in Step 0a and populate `dynamic_future_*` lists with the column names carried forward. The scoring table was already validated in Skill 1. See also [run_external_regressors_daily.ipynb](https://github.com/databricks-industry-solutions/many-model-forecasting/blob/main/examples/run_external_regressors_daily.ipynb) for the reference pattern.

### Step 4: Import notebooks into Databricks workspace

The notebooks **must** be imported as proper NOTEBOOK objects (not FILEs). Databricks job tasks require this — a FILE will cause every job to fail immediately with: `'<path>' is not a notebook`.

Use the method available in your environment:

**External agent (Claude Code, Cursor, Copilot, etc.) — Databricks CLI:**

```bash
# Local models notebook
databricks workspace import {notebook_base_path}/run_local \
  --file /tmp/{use_case}_run_local.ipynb \
  --format JUPYTER --overwrite

# Global ML models notebook (if any MLForecast model selected)
databricks workspace import {notebook_base_path}/run_global_ml \
  --file /tmp/{use_case}_run_global_ml.ipynb \
  --format JUPYTER --overwrite

# GPU run notebook (static — uploaded verbatim; only if a GPU class runs on a classic cluster)
databricks workspace import {notebook_base_path}/run_gpu \
  --file /tmp/{use_case}_run_gpu.ipynb \
  --format JUPYTER --overwrite

# Global DL orchestrator (if global DL models selected)
databricks workspace import {notebook_base_path}/orchestrator_global \
  --file /tmp/{use_case}_orchestrator_global.ipynb \
  --format JUPYTER --overwrite

# Foundation orchestrator (if foundation models selected)
databricks workspace import {notebook_base_path}/orchestrator_foundation \
  --file /tmp/{use_case}_orchestrator_foundation.ipynb \
  --format JUPYTER --overwrite

# Foundationforecast orchestrator (if FF* models selected, classic GPU)
databricks workspace import {notebook_base_path}/orchestrator_foundationforecast \
  --file /tmp/{use_case}_orchestrator_foundationforecast.ipynb \
  --format JUPYTER --overwrite

# Serverless GPU notebooks (one per GPU class on serverless GPU; {class} = global, foundation or foundationforecast)
databricks workspace import {notebook_base_path}/run_serverless_{class} \
  --file /tmp/{use_case}_run_serverless_{class}.ipynb \
  --format JUPYTER --overwrite

# Calibration notebook (only if calibrated intervals, Step 3f)
databricks workspace import {notebook_base_path}/calibrate_intervals \
  --file /tmp/{use_case}_calibrate_intervals.ipynb \
  --format JUPYTER --overwrite

# Non-forecastable notebooks (only if separate_job; see Step 3e). Same pattern with an _nf suffix
# for each one generated: run_local_nf, run_global_ml_nf, orchestrator_{class}_nf, run_serverless_{class}_nf
databricks workspace import {notebook_base_path}/run_local_nf \
  --file /tmp/{use_case}_run_local_nf.ipynb \
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
    "run_global_ml": "/tmp/{use_case}_run_global_ml.ipynb",          # if any MLForecast model selected
    "run_gpu": "/tmp/{use_case}_run_gpu.ipynb",                               # if a GPU class runs on a classic cluster
    "orchestrator_global": "/tmp/{use_case}_orchestrator_global.ipynb",       # if global DL models selected (classic)
    "orchestrator_foundation": "/tmp/{use_case}_orchestrator_foundation.ipynb",  # if foundation models selected (classic)
    "orchestrator_foundationforecast": "/tmp/{use_case}_orchestrator_foundationforecast.ipynb",  # if FF* models selected (classic)
    "run_serverless_foundationforecast": "/tmp/{use_case}_run_serverless_foundationforecast.ipynb",  # one entry per GPU class on serverless GPU
    "calibrate_intervals": "/tmp/{use_case}_calibrate_intervals.ipynb",  # calibrated intervals only (Step 3f)
    "run_local_nf": "/tmp/{use_case}_run_local_nf.ipynb",  # separate_job only: one entry per Step 3e notebook (_nf suffix)
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
- Global ML: `{use_case}_global_ml_forecasting_{username}`
- Global (DL): `{use_case}_global_forecasting_{username}`
- Foundation: `{use_case}_foundation_forecasting_{username}`
- Foundationforecast: `{use_case}_foundationforecast_forecasting_{username}`
- Calibration (only if `{calibrate_intervals}`): `{use_case}_calibrate_intervals_{username}`
- Orchestrating job (only if `{calibrate_intervals}`): `{use_case}_forecast_and_calibrate_{username}`

The name is the same whether the class runs on a classic GPU cluster or on serverless GPU. If the user switches a class between the two, update the job with the other template (classic jobs have `job_clusters`; serverless jobs have `environments` instead).

For each model class selected, **upsert** the job:
1. Search for an existing job with that exact name owned by `{full_email}`
2. If found → **update** it with the new notebook path and cluster config
3. If not found → **create** it

This ensures there is always exactly one job per type per user — no accumulation of stale jobs.

> ⚠️ **One job per model class — no exceptions.** Do NOT combine model classes into a single multi-task job. Local, global ML, global DL, foundation and foundationforecast models require different compute (multi-node CPU vs single-node CPU vs single-node GPU) or different Python environments, and must run independently in parallel. The orchestrating job of calibrated intervals does not break this rule: it holds only "Run Job" tasks, each of which runs one of these jobs on its own compute.

---

### Step 5b: Create one job per model class (triggered in parallel)

> ⚠️ **One job per model class — no exceptions.** Do NOT combine model classes into a single multi-task job. Create separate jobs for local, global_ml, global, foundation and foundationforecast models. This is required for correct cluster assignment (multi-node CPU vs single-node CPU vs single-node GPU) and independent parallelism.

> ⛔ **MANDATORY `spark_version` + topology per job class — DO NOT SUBSTITUTE.**
> The JSON templates below have intentionally different `spark_version` AND `num_workers` values. They are **not interchangeable** and must be used exactly as written:
>
> | Job | `spark_version` (required) | `num_workers` (required) |
> |---|---|---|
> | Local models | `18.x-cpu-ml-scala2.13` | `{cpu_workers}` (multi-node, sized by series count; single-node settings if 0) |
> | Global ML models | `18.x-cpu-ml-scala2.13` | **`0` (single-node always)** + `spark.master=local[*]` + singleNode profile + ResourceClass tag |
> | Global DL models (classic GPU) | `18.x-gpu-ml-scala2.13` | `0` (single-node always) |
> | Foundation models (classic GPU) | `18.x-gpu-ml-scala2.13` | `0` (single-node always) |
> | Foundationforecast models (classic GPU) | `18.x-gpu-ml-scala2.13` | `0` (single-node always) |
> | NF Local models *(if `separate_job`)* | `18.x-cpu-ml-scala2.13` | `{nf_cpu_workers}` (single-node settings if 0) |
> | NF Global ML models *(if `separate_job`)* | `18.x-cpu-ml-scala2.13` | `0` (single-node) |
> | NF Global / NF Foundation / NF Foundationforecast *(if `separate_job`, classic GPU)* | `18.x-gpu-ml-scala2.13` | `0` (single-node) |
>
> GPU classes on **serverless GPU** have no `job_clusters` and no `spark_version`. They use the [serverless GPU job template](#serverless-gpu-jobs-global-dl-foundation-foundationforecast) below instead.
>
> The agent is FORBIDDEN from:
> - Reusing the local job's `spark_version` (`18.x-cpu-ml-scala2.13`) inside the global DL or foundation JSON. A GPU node type with a CPU runtime fails to start or silently runs on CPU.
> - **Reusing the local job's `job_clusters` block (multi-node) inside the global_ml JSON.** The two share `18.x-cpu-ml-scala2.13` but the topology is incompatible — global_ml needs `num_workers: 0`, `spark.master=local[*]`, `spark.databricks.cluster.profile=singleNode`, and `custom_tags: {"ResourceClass": "SingleNode"}`. A local job only needs single-node mode when it has 0 workers (see the note after Job 1).
> - Substituting a `17.3.x` (or any non-`18.x`) runtime for the GPU jobs because it "looks like LTS." The whole pipeline is now standardized on **18 ML** (`18.x`) and `mmf_sa[global]` / `mmf_sa[foundation]` are tested against it.
> - Using a non-ML runtime (anything not ending in `-ml-scala2.13`).
> - Copy-pasting one job's `job_clusters` block into another. Build each job's `job_clusters` block from its own template.
>
> **Step 5c (immediately after job creation) reads back the `spark_version` AND `num_workers` of every job cluster via `get_job` and aborts if either does not match this table. The verification is not optional.**

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
      "spark_version": "18.x-cpu-ml-scala2.13",
      "node_type_id": "{cpu_node_type}",
      "num_workers": {cpu_workers},
      "data_security_mode": "SINGLE_USER",
      {cpu_availability_attributes},
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

> **0 workers (single-node).** If `{cpu_workers}` is 0, a multi-node `spark_conf` leaves the cluster with no executors. Keep the runtime, node type and the two Arrow/AQE settings, and add the single-node settings from Job 2: `"spark.master": "local[*]"`, `"spark.databricks.cluster.profile": "singleNode"` and `"custom_tags": {"ResourceClass": "SingleNode"}`. The Pandas UDFs then run in parallel on the driver's cores. The same rule applies to NF-1 when `{nf_cpu_workers}` is 0.

#### Job 2: Global ML models (if any MLForecast model selected)

```json
{
  "name": "{use_case}_global_ml_forecasting_{username}",
  "description": "MMF global_ml forecasting | use_case={use_case} | catalog={catalog}.{schema} | models={active_global_ml_models} | horizon={prediction_length} | created={YYYYMMDD}",
  "tags": {
    "aidevkit_project": "mmf-agent",
    "created_by": "databricks-ai-dev-kit"
  },
  "tasks": [{
    "task_key": "global_ml_models",
    "notebook_task": {
      "notebook_path": "{notebook_base_path}/run_global_ml",
      "base_parameters": {
        "experiment_path": "{experiment_path}"
      }
    },
    "job_cluster_key": "{use_case}_ml_cluster"
  }],
  "job_clusters": [{
    "job_cluster_key": "{use_case}_ml_cluster",
    "new_cluster": {
      "spark_version": "18.x-cpu-ml-scala2.13",
      "node_type_id": "{ml_node_type}",
      "num_workers": 0,
      "data_security_mode": "SINGLE_USER",
      {on_demand_attributes},
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

> **Note the `spark_conf` triple.** `spark.master=local[*]` + `spark.databricks.cluster.profile=singleNode` + `custom_tags.ResourceClass=SingleNode` MUST all be present. Missing any one degrades to multi-node mode (which then conflicts with `num_workers: 0`).

#### Job 3: Global DL models (if any global DL models selected, classic GPU)

```json
{
  "name": "{use_case}_global_forecasting_{username}",
  "description": "MMF global_dl forecasting | use_case={use_case} | catalog={catalog}.{schema} | models={active_global_models} | horizon={prediction_length} | created={YYYYMMDD}",
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
      "spark_version": "18.x-gpu-ml-scala2.13",
      "node_type_id": "{gpu_node_type}",
      "num_workers": 0,
      "data_security_mode": "SINGLE_USER",
      {on_demand_attributes},
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

#### Job 4: Foundation models (if any foundation models selected, classic GPU)

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
      "spark_version": "18.x-gpu-ml-scala2.13",
      "node_type_id": "{gpu_node_type}",
      "num_workers": 0,
      "data_security_mode": "SINGLE_USER",
      {on_demand_attributes},
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

#### Job 5: Foundationforecast models (if any `FF*` models selected, classic GPU)

Same cluster as Job 4, but its own job and its own cluster `{use_case}_ff_gpu_cluster`: `run_gpu` installs `mmf_sa[foundationforecast]` with a different torch build, which must not share an environment with the foundation models. FF inference uses one GPU, so `{ff_gpu_node_type}` is a single-GPU instance (Skill 3).

```json
{
  "name": "{use_case}_foundationforecast_forecasting_{username}",
  "description": "MMF foundationforecast forecasting | use_case={use_case} | catalog={catalog}.{schema} | models={active_ff_models} | horizon={prediction_length} | created={YYYYMMDD}",
  "tags": {
    "aidevkit_project": "mmf-agent",
    "created_by": "databricks-ai-dev-kit"
  },
  "tasks": [{
    "task_key": "foundationforecast_models",
    "notebook_task": {
      "notebook_path": "{notebook_base_path}/orchestrator_foundationforecast",
      "base_parameters": {
        "experiment_path": "{experiment_path}"
      }
    },
    "job_cluster_key": "{use_case}_ff_gpu_cluster"
  }],
  "job_clusters": [{
    "job_cluster_key": "{use_case}_ff_gpu_cluster",
    "new_cluster": {
      "spark_version": "18.x-gpu-ml-scala2.13",
      "node_type_id": "{ff_gpu_node_type}",
      "num_workers": 0,
      "data_security_mode": "SINGLE_USER",
      {on_demand_attributes},
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

#### Serverless GPU jobs (global DL, foundation, foundationforecast)

Use this template instead of Job 3, 4 or 5 for each GPU class the user put on serverless GPU in Skill 3. `{class}` is `global`, `foundation` or `foundationforecast`, and `{active_class_models}` is that class's model list.

```json
{
  "name": "{use_case}_{class}_forecasting_{username}",
  "description": "MMF {class} forecasting | use_case={use_case} | catalog={catalog}.{schema} | models={active_class_models} | horizon={prediction_length} | compute=serverless GPU A10, AI v6 | created={YYYYMMDD}",
  "tags": {
    "aidevkit_project": "mmf-agent",
    "created_by": "databricks-ai-dev-kit"
  },
  "tasks": [{
    "task_key": "{class}_models",
    "notebook_task": {
      "notebook_path": "{notebook_base_path}/run_serverless_{class}",
      "source": "WORKSPACE",
      "base_parameters": {
        "experiment_path": "{experiment_path}"
      }
    },
    "compute": {
      "hardware_accelerator": "GPU_1xA10"
    },
    "environment_key": "gpu_env",
    "timeout_seconds": 21600
  }],
  "environments": [{
    "environment_key": "gpu_env",
    "spec": {
      "base_environment": "{ai_v6_base_environment}"
    }
  }]
}
```

**`{ai_v6_base_environment}`:**
1. Use `workspace-base-environments/databricks_ai_v6`, the Databricks-provided AI v6 environment.
2. If `create_job` / `update_job` rejects it (some workspaces accept only a workspace YAML path as `base_environment`), upload [serverless_gpu_ai_v6_environment.yml](serverless_gpu_ai_v6_environment.yml) **as a file** (not a notebook) to `/Workspace/Users/{full_email}/{project_folder}/env/databricks_ai_v6.yml`, and use that `/Workspace/...` path. The file pins the AI v6 packages and uses public package indexes.

**Serverless GPU rules:**
- **No `job_clusters`, `spark_version` or `node_type_id`.** The task's `compute.hardware_accelerator` picks the GPU and the environment picks the software.
- **One A10 GPU per task.** NeuralForecast trains on that single GPU; foundation and foundationforecast inference runs on the driver's GPU (`serverless=True` in the notebook).
- **Startup can take several minutes** while the environment installs, and A10 capacity can run out in some regions. If the run fails with a capacity or environment-timeout error, retry it; if it keeps failing, offer to switch that class to a classic GPU cluster (Skill 3).
- **Serverless tasks retry automatically** after some failures. When monitoring (Step 6), read the run's overall state, not the first task attempt.

**Important notes:**
- **All clusters** use `data_security_mode: "SINGLE_USER"` because ML runtimes (`*-cpu-ml-*` and `*-gpu-ml-*`) reject `USER_ISOLATION`.
- GPU clusters are **always single-node** (`num_workers: 0`) with `spark.master: local[*]`, `spark.databricks.cluster.profile: singleNode`, and `custom_tags: {"ResourceClass": "SingleNode"}`. All three are required for proper single-node mode.
- Global, foundation and foundationforecast jobs each get their own ephemeral GPU cluster or serverless GPU task so they can run **in parallel**.
- **Output tables.** Step 5d creates `{use_case}_evaluation_output` and `{use_case}_scoring_output` before the jobs start, so parallel first writes cannot collide.
- **Availability attributes.** Replace `{on_demand_attributes}` and `{cpu_availability_attributes}` with the block for the workspace's cloud from Skill 3 [Instance availability](3-provision-forecasting-resources.md#instance-availability-all-classic-clusters). Never omit them: on AWS the default puts even the driver on spot.
- Only create jobs for model classes the user actually selected.

#### Serverless CPU alternative (local models only)

If the user requests **serverless compute** for local models, omit `job_clusters` entirely and use `environment_key` instead of `job_cluster_key`:

```json
{
  "name": "{use_case}_local_forecasting_{username}",
  "tasks": [{
    "task_key": "local_models",
    "notebook_task": {"notebook_path": "{notebook_base_path}/run_local"},
    "environment_key": "Default"
  }],
  "environments": [{
    "environment_key": "Default",
    "spec": {"client": "1"}
  }]
}
```

GPU classes on serverless use the [serverless GPU job template](#serverless-gpu-jobs-global-dl-foundation-foundationforecast), not this one: this CPU environment has no GPU.

#### Non-forecastable jobs (if `separate_job` strategy)

**Only create these jobs if `non_forecastable_strategy == 'separate_job'`.** Skip entirely for `include` or `fallback` strategies.

The notebooks for these jobs were generated in Step 3e and imported in Step 4. Their output tables are `{use_case}_nf_evaluation_output` and `{use_case}_nf_scoring_output`.

##### Job definitions for non-forecastable pipeline

**Job NF-1: Non-forecastable local models** (if any NF local models). If `{nf_cpu_workers}` is 0, apply the single-node settings from the note after Job 1:

```json
{
  "name": "{use_case}_nf_local_forecasting_{username}",
  "tasks": [{
    "task_key": "nf_local_models",
    "notebook_task": {
      "notebook_path": "{notebook_base_path}/run_local_nf",
      "base_parameters": {
        "experiment_path": "{experiment_path}"
      }
    },
    "job_cluster_key": "{use_case}_nf_cpu_cluster"
  }],
  "job_clusters": [{
    "job_cluster_key": "{use_case}_nf_cpu_cluster",
    "new_cluster": {
      "spark_version": "18.x-cpu-ml-scala2.13",
      "node_type_id": "{cpu_node_type}",
      "num_workers": {nf_cpu_workers},
      "data_security_mode": "SINGLE_USER",
      {cpu_availability_attributes},
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

**Job NF-2: Non-forecastable global ML models** (if any NF `MLForecast*` models):

```json
{
  "name": "{use_case}_nf_global_ml_forecasting_{username}",
  "tasks": [{
    "task_key": "nf_global_ml_models",
    "notebook_task": {
      "notebook_path": "{notebook_base_path}/run_global_ml_nf",
      "base_parameters": {
        "experiment_path": "{experiment_path}"
      }
    },
    "job_cluster_key": "{use_case}_nf_ml_cluster"
  }],
  "job_clusters": [{
    "job_cluster_key": "{use_case}_nf_ml_cluster",
    "new_cluster": {
      "spark_version": "18.x-cpu-ml-scala2.13",
      "node_type_id": "{nf_ml_node_type}",
      "num_workers": 0,
      "data_security_mode": "SINGLE_USER",
      {on_demand_attributes},
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

**Job NF-3: Non-forecastable global DL models** (if any NF global DL models):

```json
{
  "name": "{use_case}_nf_global_forecasting_{username}",
  "tasks": [{
    "task_key": "nf_global_models",
    "notebook_task": {
      "notebook_path": "{notebook_base_path}/orchestrator_global_nf",
      "base_parameters": {
        "experiment_path": "{experiment_path}"
      }
    },
    "job_cluster_key": "{use_case}_nf_gpu_cluster"
  }],
  "job_clusters": [{
    "job_cluster_key": "{use_case}_nf_gpu_cluster",
    "new_cluster": {
      "spark_version": "18.x-gpu-ml-scala2.13",
      "node_type_id": "{nf_gpu_node_type}",
      "num_workers": 0,
      "data_security_mode": "SINGLE_USER",
      {on_demand_attributes},
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

**Job NF-4: Non-forecastable foundation models** (if any NF foundation models):

Same pattern as NF-3 (global DL), with job name `{use_case}_nf_foundation_forecasting_{username}` and the `orchestrator_foundation_nf` notebook.

**Job NF-5: Non-forecastable foundationforecast models** (if any NF `FF*` models):

Same pattern as NF-3, with job name `{use_case}_nf_foundationforecast_forecasting_{username}`, the `orchestrator_foundationforecast_nf` notebook and its own cluster `{use_case}_nf_ff_gpu_cluster` (same config as `{use_case}_ff_gpu_cluster`).

**Do not trigger these jobs here.** Step 5c verifies them together with the main jobs and then triggers all jobs (main + NF) simultaneously.

Use `create_job` (or `update_job` on upsert) to create each job. **After `create_job` / `update_job` and BEFORE `run_job`, run Step 5c.**

#### Calibration jobs (only if `{calibrate_intervals}`, Step 1d option (d))

The calibration is CPU work on the forecast tables: it must **never** run in a GPU job. Create two jobs after the
forecast jobs (including the non-forecastable ones), so that their job IDs are known.

**Job 6: Calibration.** One task running the Step 3f notebook on a classic CPU cluster. Size it like the local job
(`{cpu_node_type}`, `{cpu_workers}` from Skill 3; single-node settings if 0 workers, as in the note after Job 1); the
driver must be on-demand (`{cpu_availability_attributes}`):

```json
{
  "name": "{use_case}_calibrate_intervals_{username}",
  "description": "MMF interval calibration | use_case={use_case} | catalog={catalog}.{schema} | levels={calibration_levels} | created={YYYYMMDD}",
  "tags": {
    "aidevkit_project": "mmf-agent",
    "created_by": "databricks-ai-dev-kit"
  },
  "tasks": [{
    "task_key": "calibrate_intervals",
    "notebook_task": {
      "notebook_path": "{notebook_base_path}/calibrate_intervals",
      "source": "WORKSPACE"
    },
    "job_cluster_key": "{use_case}_calibration_cluster",
    "timeout_seconds": 21600
  }],
  "job_clusters": [{
    "job_cluster_key": "{use_case}_calibration_cluster",
    "new_cluster": {
      "spark_version": "18.x-cpu-ml-scala2.13",
      "node_type_id": "{cpu_node_type}",
      "num_workers": {cpu_workers},
      "data_security_mode": "SINGLE_USER",
      {cpu_availability_attributes},
      "spark_conf": {
        "spark.sql.adaptive.enabled": "false",
        "spark.databricks.delta.schema.autoMerge.enabled": "true"
      }
    }
  }]
}
```

If the user prefers **serverless CPU**, omit `job_clusters`, replace `job_cluster_key` with `"environment_key": "Default"`,
and add `"environments": [{"environment_key": "Default", "spec": {"client": "4"}}]`. On serverless, the calibration
writes its intermediate results to temporary tables `{use_case}_pi_tmp_*`, which it drops at the end.

**Orchestrating job.** One "Run Job" task per forecast job created in Step 5b (main and non-forecastable), with no
dependencies between them so that they run in parallel, and a "Run Job" task for Job 6 that depends on all of them.
It has no cluster of its own:

```json
{
  "name": "{use_case}_forecast_and_calibrate_{username}",
  "description": "MMF forecasting and interval calibration | use_case={use_case} | catalog={catalog}.{schema} | created={YYYYMMDD}",
  "tags": {
    "aidevkit_project": "mmf-agent",
    "created_by": "databricks-ai-dev-kit"
  },
  "tasks": [
    {"task_key": "local_forecasting", "run_job_task": {"job_id": {local_job_id}}},
    {"task_key": "foundation_forecasting", "run_job_task": {"job_id": {foundation_job_id}}},
    {"task_key": "calibrate_intervals", "run_job_task": {"job_id": {calibration_job_id}},
     "depends_on": [{"task_key": "local_forecasting"}, {"task_key": "foundation_forecasting"}],
     "run_if": "ALL_SUCCESS"}
  ]
}
```

List one `*_forecasting` task per forecast job that exists (`global_ml`, `global`, `foundationforecast`, and the
`nf_*` jobs), and put every one of them in `depends_on`. With `run_if: ALL_SUCCESS`, a failed forecast job skips the
calibration: fix and rerun that job, then run `{use_case}_calibrate_intervals_{username}` on its own. It calibrates
the latest run of each model, so it does not need the other forecast jobs to run again. Re-running the calibration
alone (other levels, after a forecast rerun) never reruns the models.

### Step 5c: Verify each job's cluster runtime (MANDATORY)

> ⛔ **This step is not optional and cannot be skipped.** It exists to catch the most common failure mode: the agent copying the local job's `spark_version` into the global or foundation JSON (e.g. using `18.x-cpu-ml-scala2.13` for a GPU cluster). The verification reads the runtime back from the API after job creation, so it does not depend on the agent's intent.

For every job created in Step 5b (including the non-forecastable jobs if `separate_job`), call `get_job` and inspect each entry in `job_clusters[*].new_cluster.spark_version` AND `job_clusters[*].new_cluster.num_workers`. Required values:

| Job name | Required `spark_version` | Required topology |
|---|---|---|
| `{use_case}_local_forecasting_{username}` | `18.x-cpu-ml-scala2.13` | multi-node (`num_workers > 0`, `local[*]` NOT set), or single-node settings if `num_workers == 0` |
| `{use_case}_global_ml_forecasting_{username}` | `18.x-cpu-ml-scala2.13` | **single-node** (`num_workers == 0`, `spark.master==local[*]`, `cluster.profile==singleNode`, `custom_tags.ResourceClass==SingleNode`) |
| `{use_case}_global_forecasting_{username}` | `18.x-gpu-ml-scala2.13` | single-node (`num_workers == 0`) |
| `{use_case}_foundation_forecasting_{username}` | `18.x-gpu-ml-scala2.13` | single-node (`num_workers == 0`) |
| `{use_case}_foundationforecast_forecasting_{username}` | `18.x-gpu-ml-scala2.13` | single-node (`num_workers == 0`) |
| `{use_case}_nf_local_forecasting_{username}` | `18.x-cpu-ml-scala2.13` | multi-node, or single-node settings if `num_workers == 0` |
| `{use_case}_nf_global_ml_forecasting_{username}` | `18.x-cpu-ml-scala2.13` | single-node |
| `{use_case}_nf_global_forecasting_{username}` | `18.x-gpu-ml-scala2.13` | single-node |
| `{use_case}_nf_foundation_forecasting_{username}` | `18.x-gpu-ml-scala2.13` | single-node |
| `{use_case}_nf_foundationforecast_forecasting_{username}` | `18.x-gpu-ml-scala2.13` | single-node |
| `{use_case}_calibrate_intervals_{username}` *(calibrated intervals)* | `18.x-cpu-ml-scala2.13` | like the local job; never a GPU node type |

The GPU rows apply to classes on a classic GPU cluster. For a GPU class on **serverless GPU**, check instead that:
- `settings.job_clusters` is absent or empty;
- the task has `environment_key` and `compute.hardware_accelerator == "GPU_1xA10"`, and no `job_cluster_key`;
- the matching `settings.environments[*].spec.base_environment` is `workspace-base-environments/databricks_ai_v6` or the uploaded AI v6 YAML path;
- the task's notebook is `run_serverless_{class}`.

For the **orchestrating job** (calibrated intervals), check that it has no `job_clusters` and no `environments`, that
every task is a `run_job_task` whose `job_id` is one of the jobs above, and that `calibrate_intervals` depends on every
other task. For a calibration job on **serverless CPU**, check that it has no `job_clusters` and no
`compute.hardware_accelerator`.

For each job:

1. Call `get_job(job_id=...)`.
2. For each cluster in `settings.job_clusters[*]`, check `new_cluster.spark_version` AND `new_cluster.num_workers` (plus, for single-node clusters, `spark_conf['spark.master']` and `custom_tags['ResourceClass']`). Also check that `new_cluster.aws_attributes` / `azure_attributes` / `gcp_attributes` is present with `first_on_demand >= 1`, and that `availability` is on-demand unless this is a local CPU cluster with workers and the user chose spot workers.
3. Compare to the expected values from the table above.
4. If **any** value does not match, do NOT call `run_job`. Instead:
   - Print a clear error including the job name, the cluster key, the actual values, and the expected values.
   - Call `update_job` with the corrected JSON template from Step 5b — using the exact values from the table above.
   - Re-run Step 5c to verify the fix.
   - Only after the check passes for **all** jobs may the agent proceed to `run_job`.

Report a one-line summary to the user before triggering runs:

```
Cluster runtime check:
  • {use_case}_local_forecasting_{username}        → 18.x-cpu-ml-scala2.13  multi-node    ✓
  • {use_case}_global_ml_forecasting_{username}    → 18.x-cpu-ml-scala2.13  single-node   ✓
  • {use_case}_global_forecasting_{username}       → 18.x-gpu-ml-scala2.13  single-node   ✓
  • {use_case}_foundation_forecasting_{username}   → 18.x-gpu-ml-scala2.13  single-node   ✓
  • {use_case}_foundationforecast_forecasting_{username} → serverless GPU_1xA10, AI v6   ✓
All runtimes and topologies match the pinned values. Proceeding to run_job.
```

### Step 5d: Pre-create the output tables (MANDATORY before the parallel trigger)

> ⚠️ Each model's first write is `saveAsTable(..., mode="append")`. When several jobs start at once and the table does not exist yet, two of them can both try to create it, and the loser fails that model with `[TABLE_OR_VIEW_ALREADY_EXISTS]`. MMF catches the error, so the job still reports SUCCESS but the model has no rows. Creating the tables first removes the race.

Run once per `{use_case}` (and once for `{use_case}_nf` if `separate_job`). Use the same `{group_id}`, `{date_col}` and `{target}` names as in the notebooks:

```sql
CREATE TABLE IF NOT EXISTS {catalog}.{schema}.{use_case}_evaluation_output (
  {group_id} STRING, backtest_window_start_date TIMESTAMP, metric_name STRING, metric_value DOUBLE,
  forecast ARRAY<DOUBLE>, actual ARRAY<DOUBLE>, model_pickle BINARY, run_id STRING, run_date TIMESTAMP,
  model STRING, use_case STRING, model_uri STRING);

CREATE TABLE IF NOT EXISTS {catalog}.{schema}.{use_case}_scoring_output (
  {group_id} STRING, {date_col} ARRAY<TIMESTAMP>, {target} ARRAY<DOUBLE>, model STRING, run_id STRING,
  run_date TIMESTAMP, use_case STRING, model_pickle BINARY, model_uri STRING);
```

**If prediction intervals are enabled (Step 1d), run these instead.** They add the interval columns, so that the jobs' first writes do not change the schema at the same time (`mergeSchema`), which can fail the same way. Run them once per `{use_case}` (and once for `{use_case}_nf` if `separate_job`):

```sql
CREATE TABLE IF NOT EXISTS {catalog}.{schema}.{use_case}_evaluation_output (
  {group_id} STRING, backtest_window_start_date TIMESTAMP, metric_name STRING, metric_value DOUBLE,
  forecast ARRAY<DOUBLE>, actual ARRAY<DOUBLE>, model_pickle BINARY,
  forecast_quantiles MAP<STRING, ARRAY<DOUBLE>>,
  run_id STRING, run_date TIMESTAMP, model STRING, use_case STRING, model_uri STRING);

CREATE TABLE IF NOT EXISTS {catalog}.{schema}.{use_case}_scoring_output (
  {group_id} STRING, {date_col} ARRAY<TIMESTAMP>, {target} ARRAY<DOUBLE>, model STRING, run_id STRING,
  run_date TIMESTAMP, use_case STRING, model_pickle BINARY, model_uri STRING,
  {target}_quantiles MAP<STRING, ARRAY<DOUBLE>>);
```

If the tables already exist without the interval columns, add them: `ALTER TABLE … ADD COLUMNS (forecast_quantiles MAP<STRING, ARRAY<DOUBLE>>)` on `evaluation_output` and `ALTER TABLE … ADD COLUMNS ({target}_quantiles MAP<STRING, ARRAY<DOUBLE>>)` on `scoring_output`. With `"output_format": "lo_hi"` or `"both"`, also add one `ARRAY<DOUBLE>` column per bound to `scoring_output`, named `{target}_lo_<L>` and `{target}_hi_<L>` (a level of 97.5 becomes `97_5`). For the default levels:

```sql
ALTER TABLE {catalog}.{schema}.{use_case}_scoring_output ADD COLUMNS (
  {target}_lo_80 ARRAY<DOUBLE>, {target}_hi_80 ARRAY<DOUBLE>, {target}_lo_95 ARRAY<DOUBLE>, {target}_hi_95 ARRAY<DOUBLE>);
```

With `"output_format": "lo_hi"`, MMF writes these columns and not `{target}_quantiles`, which then stays null. These schemas follow `mmf_sa/intervals/scoring.py` (`interval_columns`); keep them in sync if it changes.

The calibration tables need no pre-creation: one job writes them, after the forecast jobs.

Then call `run_job` for **all jobs simultaneously**. With calibrated intervals, call `run_job` on the orchestrating job
`{use_case}_forecast_and_calibrate_{username}` only: it starts the forecast jobs in parallel, and the calibration after
them.

### Step 6: Monitor execution

Poll all job run statuses until completion. Report progress to the user with structured status updates showing per-job progress:

```
[HH:MM:SS] Triggered {n_jobs} jobs in parallel:
[HH:MM:SS]   Job {use_case}_local_forecasting_{username}       (run_id: {local_run_id})
[HH:MM:SS]   Job {use_case}_global_ml_forecasting_{username}   (run_id: {global_ml_run_id})
[HH:MM:SS]   Job {use_case}_global_forecasting_{username}      (run_id: {global_run_id})
[HH:MM:SS]   Job {use_case}_foundation_forecasting_{username}  (run_id: {foundation_run_id})
[HH:MM:SS]   Job {use_case}_foundationforecast_forecasting_{username}  (run_id: {ff_run_id})
{if separate_job:
[HH:MM:SS]   Job {use_case}_nf_local_forecasting_{username}       (run_id: {nf_local_run_id})
[HH:MM:SS]   Job {use_case}_nf_global_ml_forecasting_{username}   (run_id: {nf_global_ml_run_id})
[HH:MM:SS]   Job {use_case}_nf_{class}_forecasting_{username}     (run_id: ...)  one per NF GPU class
}

[HH:MM:SS] {use_case}_local_forecasting_{username}:       RUNNING
[HH:MM:SS] {use_case}_global_ml_forecasting_{username}:   RUNNING (MLForecastLGBM, MLForecastAutoLGBM in one process)
[HH:MM:SS] {use_case}_global_forecasting_{username}:      RUNNING (orchestrator running NeuralForecastAutoNHITS...)
[HH:MM:SS] {use_case}_foundation_forecasting_{username}:  RUNNING (orchestrator running ChronosBoltBase...)
[HH:MM:SS] {use_case}_foundationforecast_forecasting_{username}:  RUNNING (serverless GPU, 4/19 models done)
[HH:MM:SS] {use_case}_local_forecasting_{username}:       SUCCEEDED (duration: 12m 34s)
[HH:MM:SS] {use_case}_global_ml_forecasting_{username}:   SUCCEEDED (duration: 8m 12s)
[HH:MM:SS] {use_case}_global_forecasting_{username}:      SUCCEEDED (duration: 25m 12s)
[HH:MM:SS] {use_case}_foundation_forecasting_{username}:  SUCCEEDED (duration: 18m 45s)
[HH:MM:SS] {use_case}_foundationforecast_forecasting_{username}:  SUCCEEDED (duration: 80m 10s)
[HH:MM:SS] All jobs completed. Overall status: SUCCEEDED
```

- **Progress of a GPU job** comes from the evaluation table: `SELECT COUNT(DISTINCT model) FROM {catalog}.{schema}.{use_case}_evaluation_output WHERE run_id = ...`, or simply the distinct models written since the job started.
- **Serverless GPU jobs** retry failed tasks automatically, so a run can show a failed first attempt and still be running. Report the run's overall `life_cycle_state` / `result_state`; report a failed attempt only if the run itself ends failed.
- **A SUCCEEDED job is not proof that its models ran.** `run_forecast` logs a model that fails (for example `RuntimeError: CUDA unknown error` on a faulty GPU host) and moves on, so a job can succeed with no rows written. When each job finishes, check that every model in its `active_models` list has rows in `{use_case}_evaluation_output` and `{use_case}_scoring_output`:
  ```sql
  SELECT model, COUNT(DISTINCT unique_id) AS n_series
  FROM {catalog}.{schema}.{use_case}_evaluation_output
  WHERE run_id = '{run_id}'   -- or: run_date >= the job's start time
  GROUP BY model
  ```
  Report missing models as failed, and show the error from that model's `run_gpu` child run (classic) or the job's notebook output (serverless). If every model in a classic GPU job failed with the same CUDA error, rerun the job once to get a fresh cluster before changing anything else. On short series, some Auto models fail by design: `NeuralForecastAutoRNN` / `NeuralForecastAutoLSTM` with `No windows available for training` and `MLForecastAutoLGBM` with `Input data must be 2 dimensional and non empty` (seen with 60 monthly points and `backtest_length=36`); report them as unsupported for this data rather than rerunning.
- **With calibrated intervals**, monitor the orchestrating job: each `*_forecasting` task shows the state of its forecast job's run (open the child run for details), and `calibrate_intervals` starts when all of them have succeeded. Report it as skipped, not failed, when a forecast job failed. When it finishes, its notebook output lists, per model, the method of each level and the leave-one-window-out coverage.
- **Foundationforecast is the slowest class.** 19 models over 1,000 monthly series took 80–100 minutes on serverless GPU and 2.5–3 hours on a single-A10 classic cluster (`g5.xlarge`, `Standard_NV36ads_A10_v5`) in testing.

### Step 7: Log run metadata

Write run metadata to `{use_case}_run_metadata` table for audit and reproducibility:

```sql
CREATE OR REPLACE TABLE {catalog}.{schema}.{use_case}_run_metadata AS
SELECT
  '{use_case}' AS use_case,
  -- Each model-class job writes its own run_id (local/global_ml generate one internally;
  -- each GPU orchestrator generates one and shares it across the models in that job).
  -- Capture every run_id present in the evaluation output so lineage reflects all jobs
  -- that contributed — Skill 5 selects the latest run_id per model across these.
  (SELECT ARRAY_AGG(DISTINCT run_id)
   FROM {catalog}.{schema}.{use_case}_evaluation_output) AS run_ids,
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

**Produced with calibrated intervals (Step 1d option (d)):**
- Delta table `<catalog>.<schema>.{use_case}_scoring_output_intervals` — the calibrated interval columns of each model's latest forecasts
- Delta table `<catalog>.<schema>.{use_case}_evaluation_output_calibration` — the corrections per model, series, horizon step and level
- Delta table `<catalog>.<schema>.{use_case}_evaluation_output_interval_metrics` — out-of-sample coverage, miss rates, MSIS, width and WIS per series and backtest window
- (the same `{use_case}_nf_*` tables when `separate_job` with intervals)

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
