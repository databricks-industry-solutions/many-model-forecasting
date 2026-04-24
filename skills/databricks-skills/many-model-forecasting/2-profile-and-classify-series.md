# Profile and Classify Series (OPTIONAL)

> ⛔ **MANDATORY:** If you have not read [SKILL.md](SKILL.md) yet, read it now before proceeding. Do NOT take any action until you have read both SKILL.md and this file in full.

**Slash command:** `/profile-and-classify-series`

**This skill is optional.** If the user skips it, they manually select models in Skill 3.

Calculates statistical properties for each time series, partitions data into
"High-Confidence" (forecastable) and "Low-Signal" (non-forecastable) groups,
and recommends specific MMF model classes for each partition. Runs on **serverless compute**.

## Estimated Runtime

Inform the user of approximate profiling times before they commit:

| Series count | Estimated time | Notes |
|-------------|---------------|-------|
| < 100 | ~2–5 minutes | Quick validation |
| 100–1,000 | ~5–15 minutes | Typical small-to-medium project |
| 1,000–10,000 | ~15–45 minutes | Large project; serverless helps |
| > 10,000 | ~1–2 hours | Consider sampling a subset first |

The profiling involves STL decomposition, ADF tests, and spectral analysis per series. Serverless compute avoids cluster startup overhead, but wall-clock time scales linearly with series count.

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `catalog` | Ask user | Unity Catalog name |
| `schema` | Ask user | Schema name |
| `use_case` | From Skill 1 | Use case name (prefixes all assets) |
| `train_table` | `{use_case}_train_data` | Training table created by Skill 1 |
| `freq` | Auto-detected from data | Time series frequency (`H`, `D`, `W`, `M`) |
| `prediction_length` | Ask user | Forecast horizon (needed for classification thresholds) |
| `forecast_problem_brief` | From Skill 1, or capture at Step 2 | Short problem statement: what `y` means, business use, horizon intent, series shape, exogenous vs many univariate (see Skill 1 Step 0b) |

## Placeholder values

| Placeholder | Value |
|-------------|-------|
| `{catalog}` | user's catalog |
| `{schema}` | user's schema |
| `{use_case}` | use case name from Skill 1 |
| `{train_table}` | `{use_case}_train_data` |
| `{freq}` | detected or user-specified frequency |
| `{prediction_length}` | user-specified forecast horizon (integer) |
| `{forecast_problem_brief}` | from Skill 1 conversation context, or collected at Step 2 if missing |
| `{username}` | current user (without `@domain`) — from `get_current_user()` |
| `{project_folder}` | user's project folder name — asked in Step 2b (default: `{use_case}`) |
| `{notebook_base_path}` | `/Users/{full_email}/{project_folder}/notebooks` — derived from `get_current_user()` + user choice |

Use the template from:
- [mmf_profiling_notebook_template.ipynb](mmf_profiling_notebook_template.ipynb)

## Steps

### Step 1: Verify training data exists

```sql
SELECT COUNT(*) AS count FROM {catalog}.{schema}.{use_case}_train_data
```

If the table does not exist or is empty, instruct the user to run `/prep-and-clean-data` first.

### ⛔ STOP GATE — Step 2: Confirm catalog/schema and gather parameters

**Always ask the user for catalog and schema. Do NOT assume or reuse values.**

Use `AskUserQuestion` to confirm:
- `catalog` and `schema`
- `use_case` name
- **`{forecast_problem_brief}`** — use the brief from Skill 1 if present; if missing (skipped prep or new session), ask the same minimal questions as Skill 1 Step 0b and store a 3–6 line summary **before** running profiling
- `freq` (detected frequency from Skill 1, or ask user)
- `prediction_length` (forecast horizon — needed for series length classification)

Also inform the user of the estimated runtime based on the series count:

```
AskUserQuestion:
  "Profiling will analyze {n_series} time series.
   Estimated runtime: {estimated_time}

   Parameters:
   • Catalog: {catalog}
   • Schema: {schema}
   • Use case: {use_case}
   • Forecast brief: {forecast_problem_brief}
   • Frequency: {freq}
   • Prediction length: {prediction_length}

   The job will run on serverless compute (no cluster startup delay).

   Proceed with profiling?
   (a) Yes, run profiling
   (b) No, skip profiling and go to model selection"
```

**Do NOT proceed until the user confirms.**

### Step 2b: Get current user and project folder

Call `get_current_user()` to obtain `{full_email}` and derive `{username}` (local part before `@`).

If `{project_folder}` and `{notebook_base_path}` were already confirmed in Skill 1, reuse them. If not known (new session or Skill 1 was skipped), ask:

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

Store these for use in all subsequent steps.

### Step 3: Generate notebook from template

**CRITICAL: Copy the template VERBATIM from `mmf_profiling_notebook_template.ipynb`, only replacing the `{placeholder}` tokens with actual values. Do NOT add, remove, or modify any other code.**

Replace these placeholders:
- `{catalog}` → user's catalog
- `{schema}` → user's schema
- `{use_case}` → use case name
- `{train_table}` → `{use_case}_train_data`
- `{freq}` → detected frequency
- `{prediction_length}` → user-specified forecast horizon

Save the filled-in notebook locally as `/tmp/{use_case}_run_profiling.ipynb`.

### Step 4: Import notebook into Databricks workspace

The notebook **must** be imported as a proper NOTEBOOK object (not a FILE). Databricks job tasks require this — a FILE will cause the job to fail immediately with: `'<path>' is not a notebook`.

Use the method available in your environment:

**External agent (Claude Code, Cursor, Copilot, etc.) — Databricks CLI:**
```bash
databricks workspace import {notebook_base_path}/run_profiling \
  --file /tmp/{use_case}_run_profiling.ipynb \
  --format JUPYTER \
  --overwrite
```
> ⚠️ Do NOT use the `upload_file` MCP tool — it creates a FILE, not a NOTEBOOK.

If the path already exists as a FILE (e.g. from a prior failed upload), delete it first:
```bash
databricks workspace delete {notebook_base_path}/run_profiling
databricks workspace import {notebook_base_path}/run_profiling \
  --file /tmp/{use_case}_run_profiling.ipynb \
  --format JUPYTER
```

**Genie Code (Databricks-native) — Python SDK:**
```python
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.workspace import ImportFormat
import base64

w = WorkspaceClient()
with open("/tmp/{use_case}_run_profiling.ipynb", "rb") as f:
    content = base64.b64encode(f.read()).decode("utf-8")
w.workspace.import_(
    path="{notebook_base_path}/run_profiling",
    format=ImportFormat.JUPYTER,
    overwrite=True,
    content=content
)
```

### Step 4b: Verify notebook type

After import, confirm the workspace object is a NOTEBOOK (not a FILE).

**External agent — Databricks CLI:**
```bash
databricks workspace get-status {notebook_base_path}/run_profiling
```

**Genie Code — Python SDK:**
```python
obj = w.workspace.get_status(path="{notebook_base_path}/run_profiling")
print(obj.object_type)  # must be NOTEBOOK
```

The returned `object_type` **must** be `NOTEBOOK`. If it is `FILE`, the job will fail — delete and re-import before proceeding.

### Step 5: Create Workflow job on serverless compute

Job name: `{use_case}_profiling_{username}` (no date — one persistent job per user, upsert approach).

**Upsert:** search for an existing job with that exact name owned by `{full_email}`. If found → update it. If not found → create it.

Create a single-task Workflow job on **serverless compute** (profiling benefits from instant startup):

```json
{
  "name": "{use_case}_profiling_{username}",
  "description": "MMF profiling | use_case={use_case} | catalog={catalog}.{schema} | created={YYYYMMDD}",
  "tags": {
    "aidevkit_project": "mmf-agent",
    "created_by": "databricks-ai-dev-kit"
  },
  "tasks": [{
    "task_key": "profile_series",
    "notebook_task": {
      "notebook_path": "{notebook_base_path}/run_profiling"
    },
    "environment_key": "Default"
  }],
  "environments": [{
    "environment_key": "Default",
    "spec": {
      "client": "1"
    }
  }]
}
```

Use `create_job` to create the job, then `run_job` to start it.

### Step 6: Monitor execution

Poll the job run status via `get_job_run` until completion. Report progress to the user.

### Step 7: Query profiling results

```sql
SELECT forecastability_class, COUNT(*) AS series_count
FROM {catalog}.{schema}.{use_case}_series_profile
GROUP BY forecastability_class
```

Present the classification summary:
- Total series profiled
- High-confidence count and percentage
- Low-signal count and percentage

### Step 8: Present model recommendations

```sql
SELECT DISTINCT recommended_models, model_types_needed, COUNT(*) AS series_count
FROM {catalog}.{schema}.{use_case}_series_profile
WHERE forecastability_class = 'high_confidence'
GROUP BY recommended_models, model_types_needed
ORDER BY series_count DESC
```

Use `AskUserQuestion` to let the user review and confirm:
- Recommended model families
- Required compute types (local/global/foundation)
- Whether to adjust classification thresholds

### ⛔ STOP GATE — Step 8b: Optional deep research on feature engineering (post-profiling)

**Prerequisites (do not skip):** The profiling job has **finished successfully**, `{catalog}.{schema}.{use_case}_series_profile` exists, and you have already presented the **Step 7** classification summary and **Step 8** model-recommendation breakdown. Optional deep research must be grounded in **three** inputs: **`{forecast_problem_brief}`** (domain, metric meaning, intermittency, horizon, exogenous intent), **profiling outputs**, and **`{use_case}_train_data` columns**.

**Before** asking the user, gather context with SQL (adjust placeholders):

1. **Training table columns** — `{use_case}_train_data`:

```sql
SELECT column_name, data_type
FROM information_schema.columns
WHERE table_catalog = '{catalog}'
  AND table_schema = '{schema}'
  AND table_name = '{use_case}_train_data'
ORDER BY ordinal_position
```

2. **Profiling metadata snapshot** — `{use_case}_series_profile` (dataset-level signals the research should reference):

```sql
SELECT
  COUNT(*) AS series_count,
  ROUND(AVG(sparsity), 4)       AS avg_sparsity,
  ROUND(AVG(spectral_entropy), 4) AS avg_spectral_entropy,
  ROUND(AVG(snr), 4)            AS avg_snr,
  ROUND(AVG(seasonality_strength), 4) AS avg_seasonality,
  ROUND(AVG(trend_strength), 4) AS avg_trend
FROM {catalog}.{schema}.{use_case}_series_profile
```

Optionally add `MIN`/`MAX` on key metrics or a small `GROUP BY forecastability_class` recap if it helps the narrative.

```
AskUserQuestion:
  "Profiling is complete. I can run an optional deep research pass grounded in:

   • Forecast problem brief: {forecast_problem_brief}
   • Training table columns: {column_list_summary}
   • Profiling signals (e.g., avg sparsity {avg_sparsity}, avg spectral entropy {avg_entropy}, high vs low-signal split from Step 7)
   • Recommended model families from Step 8

   Would you like deep research on additional feature engineering — scoped to your brief, intermittent/seasonal patterns, and the models recommended?

   (a) Yes — research (web + authoritative sources as needed) and return a concise, actionable feature-engineering brief; scope all findings to the forecast brief; do not alter Delta tables unless I explicitly ask
   (b) No — skip research and continue to the handoff (Step 9)"
```

**Do NOT proceed until the user responds.**

- If **(a)**: Tie recommendations to **`{forecast_problem_brief}`** (domain, metric, intermittency, horizon, exogenous intent), **actual column names**, **freq**, **prediction_length**, and **profiling statistics** (sparsity, entropy, SNR, seasonality/trend, forecastability split). Connect ideas to the **recommended model families** (e.g., Poisson loss for neural models on counts, ADIDA/aggregation when sparsity is high). **Feature type preference:** recommend `static_features` (constant per series, always known) and `dynamic_historical_*` (past-only signals for NeuralForecast global models) over `dynamic_future_*`. Only suggest `dynamic_future_*` features if the user's `{forecast_problem_brief}` explicitly mentions known future regressors (e.g., planned promotions, contractual pricing, scheduled events) **and** the user confirms they can provide a scoring table with those values for every future `ds`. In all other cases, steer the user toward univariate forecasting augmented with `static_features` and `dynamic_historical_*`. After the brief, **do not** auto-advance — offer implementation only if the user asks, then present **Step 9**.
- If **(b)**: Go directly to **Step 9**.
### ⛔ STOP GATE — Step 9: Decide how to handle non-forecastable series

**Always present this decision to the user. Do NOT proceed until the user selects an option.**

Query counts for the prompt:
```sql
SELECT
  COUNT(CASE WHEN forecastability_class = 'high_confidence' THEN 1 END) AS n_forecastable,
  COUNT(CASE WHEN forecastability_class = 'low_signal' THEN 1 END) AS n_non_forecastable,
  COUNT(*) AS n_total
FROM {catalog}.{schema}.{use_case}_series_profile
```

```
AskUserQuestion:
  "Your dataset contains {n_non_forecastable} non-forecastable (low-signal) series
   out of {n_total} total ({non_forecastable_pct}%).

   These series have high noise, insufficient length, or too many zeros
   to produce reliable forecasts. How would you like to handle them?

   (a) Keep all together — forecast everything with the same models
       The non-forecastable series stay in the dataset. All models run on all series.
       Simple, but may waste compute and pull down aggregate metrics.
       Best when the low-signal count is small (< 5-10% of total).

   (b) Exclude + automatic fallback — remove from main pipeline, apply a simple rule
       Non-forecastable series are excluded from model training/evaluation.
       A lightweight fallback produces their forecasts immediately:
         • Seasonal Naive — repeat the last complete seasonal cycle
         • Naive (last value) — carry forward the last observed value
         • Historical mean — average of all observed values
         • Zero — fill forecast horizon with zeros
       Fast, cheap, and honest. Fallback results merge back in post-processing.

   (c) Exclude + separate forecasting job — run them through their own pipeline
       Non-forecastable series get their own dedicated job with models you choose
       (e.g., baseline, intermittent demand, or even foundation models).
       Cluster sizing will be computed from the non-forecastable dataset size.
       Results merge back in post-processing with full evaluation metrics.
       Best when the low-signal count is large or you suspect specialized models
       (TSB, ADIDA, IMAPA, Croston) may recover signal."
```

**WAIT for the user to respond. Do NOT proceed until the user has selected an option.**

### Step 10: Execute the chosen non-forecastable strategy

Based on the user's choice in Step 9:

#### Option (a): Keep all together

No table changes. Record the strategy:

```sql
CREATE OR REPLACE TABLE {catalog}.{schema}.{use_case}_pipeline_config AS
SELECT
  '{use_case}' AS use_case,
  'include' AS non_forecastable_strategy,
  NULL AS fallback_method,
  NULL AS non_forecastable_models,
  {n_forecastable} AS n_forecastable,
  {n_non_forecastable} AS n_non_forecastable,
  CURRENT_TIMESTAMP() AS created_at
```

The main pipeline will use `{use_case}_train_data` as-is.

#### Option (b): Exclude + automatic fallback

First, ask which fallback method to use:

```
AskUserQuestion:
  "Which fallback method for the {n_non_forecastable} non-forecastable series?

   (1) Seasonal Naive — repeat the last seasonal cycle (good default if data has any pattern)
   (2) Naive (last value) — carry forward the final observation
   (3) Historical mean — use the average of all observations
   (4) Zero — fill with zeros (for intermittent/sparse series you want to suppress)"
```

Then create the filtered training table and compute fallback forecasts:

```sql
-- Forecastable series only (used by main pipeline in Skills 3-4)
CREATE OR REPLACE TABLE {catalog}.{schema}.{use_case}_train_data_forecastable AS
SELECT t.*
FROM {catalog}.{schema}.{use_case}_train_data t
INNER JOIN {catalog}.{schema}.{use_case}_series_profile p
  ON t.unique_id = p.unique_id
WHERE p.forecastability_class = 'high_confidence'
```

Compute the fallback forecasts on serverless (lightweight SQL/Spark operation):

**Seasonal Naive fallback:**
```sql
CREATE OR REPLACE TABLE {catalog}.{schema}.{use_case}_scoring_output_non_forecastable AS
WITH last_season AS (
  SELECT unique_id, ds, y,
         ROW_NUMBER() OVER (PARTITION BY unique_id ORDER BY ds DESC) AS rn
  FROM {catalog}.{schema}.{use_case}_train_data t
  INNER JOIN {catalog}.{schema}.{use_case}_series_profile p
    ON t.unique_id = p.unique_id
  WHERE p.forecastability_class = 'low_signal'
),
-- Take the last {prediction_length} observations as the forecast
seasonal_values AS (
  SELECT unique_id, COLLECT_LIST(y) AS y
  FROM last_season
  WHERE rn <= {prediction_length}
  GROUP BY unique_id
)
SELECT unique_id, 'SeasonalNaiveFallback' AS model, y
FROM seasonal_values
```

**Naive (last value) fallback:**
```sql
CREATE OR REPLACE TABLE {catalog}.{schema}.{use_case}_scoring_output_non_forecastable AS
WITH last_values AS (
  SELECT unique_id,
         FIRST_VALUE(y) OVER (PARTITION BY unique_id ORDER BY ds DESC) AS last_y
  FROM {catalog}.{schema}.{use_case}_train_data t
  INNER JOIN {catalog}.{schema}.{use_case}_series_profile p
    ON t.unique_id = p.unique_id
  WHERE p.forecastability_class = 'low_signal'
),
distinct_series AS (
  SELECT DISTINCT unique_id, last_y FROM last_values
)
SELECT unique_id, 'NaiveFallback' AS model,
       ARRAY_REPEAT(last_y, {prediction_length}) AS y
FROM distinct_series
```

**Historical mean fallback:**
```sql
CREATE OR REPLACE TABLE {catalog}.{schema}.{use_case}_scoring_output_non_forecastable AS
WITH series_means AS (
  SELECT t.unique_id, AVG(t.y) AS mean_y
  FROM {catalog}.{schema}.{use_case}_train_data t
  INNER JOIN {catalog}.{schema}.{use_case}_series_profile p
    ON t.unique_id = p.unique_id
  WHERE p.forecastability_class = 'low_signal'
  GROUP BY t.unique_id
)
SELECT unique_id, 'HistoricalMeanFallback' AS model,
       ARRAY_REPEAT(mean_y, {prediction_length}) AS y
FROM series_means
```

**Zero fallback:**
```sql
CREATE OR REPLACE TABLE {catalog}.{schema}.{use_case}_scoring_output_non_forecastable AS
SELECT DISTINCT t.unique_id, 'ZeroFallback' AS model,
       ARRAY_REPEAT(CAST(0.0 AS DOUBLE), {prediction_length}) AS y
FROM {catalog}.{schema}.{use_case}_train_data t
INNER JOIN {catalog}.{schema}.{use_case}_series_profile p
  ON t.unique_id = p.unique_id
WHERE p.forecastability_class = 'low_signal'
```

Record the strategy:
```sql
CREATE OR REPLACE TABLE {catalog}.{schema}.{use_case}_pipeline_config AS
SELECT
  '{use_case}' AS use_case,
  'fallback' AS non_forecastable_strategy,
  '{fallback_method}' AS fallback_method,
  NULL AS non_forecastable_models,
  {n_forecastable} AS n_forecastable,
  {n_non_forecastable} AS n_non_forecastable,
  CURRENT_TIMESTAMP() AS created_at
```

Verify:
```sql
SELECT COUNT(DISTINCT unique_id) AS fallback_series
FROM {catalog}.{schema}.{use_case}_scoring_output_non_forecastable
```

Report to the user: "{fallback_series} non-forecastable series handled with {fallback_method} fallback. Results stored in `{use_case}_scoring_output_non_forecastable`."

#### Option (c): Exclude + separate forecasting job

Create both filtered training tables:

```sql
-- Forecastable series only (used by main pipeline in Skills 3-4)
CREATE OR REPLACE TABLE {catalog}.{schema}.{use_case}_train_data_forecastable AS
SELECT t.*
FROM {catalog}.{schema}.{use_case}_train_data t
INNER JOIN {catalog}.{schema}.{use_case}_series_profile p
  ON t.unique_id = p.unique_id
WHERE p.forecastability_class = 'high_confidence'
```

```sql
-- Non-forecastable series (used by separate job in Skill 4)
CREATE OR REPLACE TABLE {catalog}.{schema}.{use_case}_train_data_non_forecastable AS
SELECT t.*
FROM {catalog}.{schema}.{use_case}_train_data t
INNER JOIN {catalog}.{schema}.{use_case}_series_profile p
  ON t.unique_id = p.unique_id
WHERE p.forecastability_class = 'low_signal'
```

Then ask the user which models to run on the non-forecastable series:

```
AskUserQuestion:
  "Which models should the non-forecastable series be forecasted with?
   Recommended models for low-signal series are pre-selected.

   BASELINE MODELS (recommended):
   [x] StatsForecastBaselineNaive
   [x] StatsForecastBaselineSeasonalNaive

   INTERMITTENT DEMAND MODELS (recommended if sparsity is high):
   [ ] StatsForecastTSB
   [ ] StatsForecastADIDA
   [ ] StatsForecastIMAPA
   [ ] StatsForecastCrostonClassic
   [ ] StatsForecastCrostonOptimized
   [ ] StatsForecastCrostonSBA

   OTHER LOCAL MODELS:
   [ ] StatsForecastAutoArima
   [ ] StatsForecastAutoETS
   [ ] StatsForecastAutoCES

   FOUNDATION MODELS (zero-shot, no training needed):
   [ ] ChronosBoltTiny
   [ ] ChronosBoltMini
   [ ] ChronosBoltSmall
   [ ] ChronosBoltBase

   List the model names you want to run."
```

Record the strategy:
```sql
CREATE OR REPLACE TABLE {catalog}.{schema}.{use_case}_pipeline_config AS
SELECT
  '{use_case}' AS use_case,
  'separate_job' AS non_forecastable_strategy,
  NULL AS fallback_method,
  '{non_forecastable_models_str}' AS non_forecastable_models,
  {n_forecastable} AS n_forecastable,
  {n_non_forecastable} AS n_non_forecastable,
  CURRENT_TIMESTAMP() AS created_at
```

Verify both tables:
```sql
SELECT 'forecastable' AS partition, COUNT(DISTINCT unique_id) AS n_series
FROM {catalog}.{schema}.{use_case}_train_data_forecastable
UNION ALL
SELECT 'non_forecastable', COUNT(DISTINCT unique_id)
FROM {catalog}.{schema}.{use_case}_train_data_non_forecastable
```

## Statistical Properties Computed

| Property | Method | Library |
|----------|--------|---------|
| **Stationarity** | Augmented Dickey-Fuller p-value | `statsmodels.tsa.stattools.adfuller` |
| **Seasonality Strength** | STL decomposition: `1 - Var(remainder) / Var(deseasonalized)` | `statsmodels.tsa.seasonal.STL` |
| **Trend Strength** | STL decomposition: `1 - Var(remainder) / Var(detrended)` | `statsmodels.tsa.seasonal.STL` |
| **Spectral Entropy** | Normalized Shannon entropy of spectral density | `scipy.signal.periodogram` + `scipy.stats.entropy` |
| **Autocorrelation (lag-1)** | Pearson autocorrelation at lag 1 | `pandas.Series.autocorr` |
| **Signal-to-Noise Ratio** | `mean(y)² / var(y)` | native pandas |
| **Sparsity** | Fraction of zero or near-zero values | native pandas |
| **Coefficient of Variation** | `std(y) / mean(y)` | native pandas |
| **Series Length** | Number of observations | native pandas |

## Classification Logic

```
High-Confidence (Forecastable):
  - spectral_entropy < 0.8
  - series_length >= 2 * prediction_length
  - sparsity < 0.5
  - snr > 0.1

Low-Signal (Non-Forecastable):
  - Everything else
```

## Model Recommendation Logic

| Series Characteristics | Recommended Models | Rationale |
|----------------------|-------------------|-----------|
| Strong seasonality (>0.6) + stationary | `StatsForecastAutoArima`, `StatsForecastAutoETS`, `StatsForecastAutoTheta`, `ChronosBoltBase`, `Chronos2`, `TimesFM_2_5_200m` | Classical models excel with clear seasonal patterns; foundation models as benchmark |
| Strong trend (>0.6) + weak seasonality | `StatsForecastAutoArima`, `SKTimeProphet`, `NeuralForecastAutoNHITS`, `ChronosBoltBase`, `Chronos2`, `TimesFM_2_5_200m` | ARIMA captures trends; Prophet handles changepoints |
| High complexity (entropy >0.6) + long series (>200) | `NeuralForecastAutoNHITS`, `NeuralForecastAutoPatchTST`, `ChronosBoltBase`, `Chronos2`, `TimesFM_2_5_200m` | Neural models learn complex patterns; largest foundation models for zero-shot |
| Short series (<50 points) | `StatsForecastAutoETS`, `StatsForecastAutoCES`, `ChronosBoltBase`, `Chronos2`, `TimesFM_2_5_200m` | Simple models + zero-shot foundation models |
| Intermittent/sparse (sparsity >0.3) | `StatsForecastTSB`, `StatsForecastADIDA`, `StatsForecastIMAPA`, `StatsForecastCrostonClassic` | Specialized intermittent demand models |
| General / mixed characteristics | `StatsForecastAutoArima`, `NeuralForecastAutoNHITS`, `ChronosBoltBase`, `Chronos2`, `TimesFM_2_5_200m` | Broad coverage across model families |
| Low-signal (non-forecastable) | `StatsForecastBaselineNaive`, `StatsForecastBaselineSeasonalNaive` | Baseline only; flag for human review |

## ⛔ STOP GATE — Step 11: Confirm before proceeding to next skill

```
AskUserQuestion:
  "✅ Profiling complete for use case '{use_case}'.

   Summary:
   • Total series profiled: {total}
   • High-confidence (forecastable): {high} ({high_pct}%)
   • Low-signal (non-forecastable): {low} ({low_pct}%)
   • Non-forecastable strategy: {strategy_description}
   • Recommended model types: {model_types}
   • Profile table: {catalog}.{schema}.{use_case}_series_profile
   • Pipeline config: {catalog}.{schema}.{use_case}_pipeline_config
   {if strategy == 'fallback': '• Fallback forecasts: {catalog}.{schema}.{use_case}_scoring_output_non_forecastable'}
   {if strategy != 'include': '• Forecastable training data: {catalog}.{schema}.{use_case}_train_data_forecastable'}
   {if strategy == 'separate_job': '• Non-forecastable training data: {catalog}.{schema}.{use_case}_train_data_non_forecastable'}
   {if strategy == 'separate_job': '• Non-forecastable models: {non_forecastable_models}'}

   Would you like to proceed to cluster provisioning and model selection?
   (a) Yes, proceed to /provision-forecasting-resources
   (b) No, stop here — I'll come back later"
```

**Do NOT proceed until the user responds.**

## Output

**Table**: `<catalog>.<schema>.{use_case}_series_profile`

| Column | Type | Description |
|--------|------|-------------|
| `unique_id` | STRING | Series identifier |
| `series_length` | INT | Number of observations |
| `adf_pvalue` | DOUBLE | ADF test p-value (stationarity) |
| `seasonality_strength` | DOUBLE | 0-1 seasonality measure |
| `trend_strength` | DOUBLE | 0-1 trend measure |
| `spectral_entropy` | DOUBLE | 0-1 entropy measure |
| `autocorrelation_lag1` | DOUBLE | Lag-1 autocorrelation |
| `snr` | DOUBLE | Signal-to-noise ratio |
| `sparsity` | DOUBLE | Fraction of zero values |
| `cv` | DOUBLE | Coefficient of variation |
| `forecastability_class` | STRING | `high_confidence` or `low_signal` |
| `recommended_models` | STRING | Comma-separated model names |
| `model_types_needed` | STRING | `local`, `local,foundation`, etc. |

**Table**: `<catalog>.<schema>.{use_case}_pipeline_config`

| Column | Type | Description |
|--------|------|-------------|
| `use_case` | STRING | Use case name |
| `non_forecastable_strategy` | STRING | `include`, `fallback`, or `separate_job` |
| `fallback_method` | STRING | Fallback method name (null if not `fallback`) |
| `non_forecastable_models` | STRING | Comma-separated model names (null if not `separate_job`) |
| `n_forecastable` | INT | Count of high-confidence series |
| `n_non_forecastable` | INT | Count of low-signal series |
| `created_at` | TIMESTAMP | When the config was created |

**Table** (conditional): `<catalog>.<schema>.{use_case}_train_data_forecastable`

Created when strategy is `fallback` or `separate_job`. Contains only high-confidence series from the training data. Same schema as `{use_case}_train_data`.

**Table** (conditional): `<catalog>.<schema>.{use_case}_train_data_non_forecastable`

Created when strategy is `separate_job`. Contains only low-signal series from the training data. Same schema as `{use_case}_train_data`.

**Table** (conditional): `<catalog>.<schema>.{use_case}_scoring_output_non_forecastable`

Created when strategy is `fallback`. Contains fallback forecasts for non-forecastable series.

| Column | Type | Description |
|--------|------|-------------|
| `unique_id` | STRING | Series identifier |
| `model` | STRING | Fallback method name (e.g., `SeasonalNaiveFallback`) |
| `y` | ARRAY&lt;DOUBLE&gt; | Forecast values |
