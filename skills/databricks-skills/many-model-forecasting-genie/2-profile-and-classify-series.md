# Profile and Classify Series (OPTIONAL)

**This step is optional.** If the user skips it, they manually select models in Step 3.

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

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `catalog` | Ask user | Unity Catalog name |
| `schema` | Ask user | Schema name |
| `use_case` | From Step 1 | Use case name (prefixes all assets) |
| `train_table` | `{use_case}_train_data` | Training table created by Step 1 |
| `freq` | Auto-detected from data | Time series frequency (`H`, `D`, `W`, `M`) |
| `prediction_length` | Ask user | Forecast horizon (needed for classification thresholds) |

## Steps

### Step 1: Verify training data exists

```sql
SELECT COUNT(*) AS count FROM {catalog}.{schema}.{use_case}_train_data
```

If the table does not exist or is empty, instruct the user to run the data preparation step first.

### ⛔ STOP GATE — Step 2: Confirm catalog/schema and gather parameters

**Always ask the user for catalog and schema. Do NOT assume or reuse values.**

Ask the user:

> "Profiling will analyze {n_series} time series.
> Estimated runtime: {estimated_time}
>
> Parameters:
> - Catalog: {catalog}
> - Schema: {schema}
> - Use case: {use_case}
> - Frequency: {freq}
> - Prediction length: {prediction_length}
>
> The job will run on serverless compute (no cluster startup delay).
>
> Proceed with profiling?
> (a) Yes, run profiling
> (b) No, skip profiling and go to model selection"

**Do NOT proceed until the user confirms.**

### Step 3: Generate notebook from template

**CRITICAL: Do NOT execute profiling code inline. All profiling runs MUST happen inside a Databricks notebook on cluster or serverless compute. Generate the notebook from the template and upload it to the workspace.**

**CRITICAL: Use the template at `notebooks/mmf_profiling_notebook_template.ipynb` (in this skill folder). Copy it verbatim, only replacing the `{placeholder}` tokens with actual values. Do NOT add, remove, or modify any other code.**

Replace these placeholders:
- `{catalog}` → user's catalog
- `{schema}` → user's schema
- `{use_case}` → use case name
- `{train_table}` → `{use_case}_train_data`
- `{freq}` → detected frequency
- `{prediction_length}` → user-specified forecast horizon

### ⛔ STOP GATE — Step 4: Confirm before uploading notebook

Before uploading, present a summary and ask the user:

> "I am about to upload the profiling notebook to the workspace:
> - Path: `{home_path}/mmf-skills-test/notebooks/{use_case}/02_run_profiling`
> - Parameters: catalog={catalog}, schema={schema}, use_case={use_case}, freq={freq}, prediction_length={prediction_length}
>
> Shall I proceed?"

**Do NOT upload until the user confirms.**

Upload the resulting notebook to the Databricks workspace at:

```
{home_path}/mmf-skills-test/notebooks/{use_case}/02_run_profiling
```

Where `{home_path}` = `/Workspace/Users/{current_user_email}`.

### Step 5: Execute profiling

**Never create a persistent job for profiling.** Profiling is a one-time analytical step. Use the strategy below.

#### Step 5a: Determine execution strategy

```sql
SELECT COUNT(DISTINCT unique_id) AS n_series
FROM {catalog}.{schema}.{use_case}_train_data
```

- **n_series < 300**: Execute the notebook directly on serverless compute — no job or run submission needed.
- **n_series ≥ 300**: Submit a one-time run using `runs/submit` with a small CPU cluster. This provides Spark parallelism for `applyInPandas` across series. The cluster is destroyed automatically after the run and is not saved to the Workflows UI.

#### ⛔ STOP GATE — Step 5b: Confirm before executing

Before running, ask the user:

> "I am about to execute the profiling notebook:
> - Path: `{home_path}/mmf-skills-test/notebooks/{use_case}/02_run_profiling`
> - Execution mode: {serverless direct execution / one-time run submission with CPU cluster ({n_workers} workers)}
> - Estimated runtime: {estimated_time}
>
> Shall I proceed?"

**Do NOT execute until the user confirms.**

#### Step 5c: Execute

**If n_series < 300 — serverless direct execution:**

Execute the profiling notebook directly on serverless compute using the available Databricks platform tools.

**If n_series ≥ 300 — one-time run submission:**

Submit using `runs/submit` (not `jobs/create`). Size workers by series count:

| Series count | Workers |
|---|---|
| 300–1,000 | 2 |
| 1,000–5,000 | 4 |
| > 5,000 | 6 |

```json
{
  "run_name": "{use_case}_profiling_{username_without_domain}_{YYYYMMDD}",
  "tasks": [{
    "task_key": "profile_series",
    "notebook_task": {
      "notebook_path": "{home_path}/mmf-skills-test/notebooks/{use_case}/02_run_profiling"
    },
    "job_cluster_key": "profiling_cluster"
  }],
  "job_clusters": [{
    "job_cluster_key": "profiling_cluster",
    "new_cluster": {
      "spark_version": "17.3.x-cpu-ml-scala2.13",
      "node_type_id": "{cpu_node_type}",
      "num_workers": {n_workers},
      "data_security_mode": "SINGLE_USER",
      "spark_conf": {
        "spark.sql.execution.arrow.enabled": "true",
        "spark.sql.adaptive.enabled": "false"
      }
    }
  }]
}
```

Where `{cpu_node_type}` is the node type for the user's cloud provider (see Step 3 of `3-provision-forecasting-resources.md` for options).

Submit the run, then monitor it.

### Step 6: Monitor execution

Poll the job run status until completion. Report progress to the user.

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

Ask the user to review and confirm:
- Recommended model families
- Required compute types (local/global/foundation)
- Whether to adjust classification thresholds

### ⛔ STOP GATE — Step 9: Decide how to handle non-forecastable series

**Always present this decision to the user. Do NOT proceed until the user selects an option.**

Query counts:
```sql
SELECT
  COUNT(CASE WHEN forecastability_class = 'high_confidence' THEN 1 END) AS n_forecastable,
  COUNT(CASE WHEN forecastability_class = 'low_signal' THEN 1 END) AS n_non_forecastable,
  COUNT(*) AS n_total
FROM {catalog}.{schema}.{use_case}_series_profile
```

Ask the user:

> "Your dataset contains {n_non_forecastable} non-forecastable (low-signal) series out of {n_total} total ({non_forecastable_pct}%).
>
> These series have high noise, insufficient length, or too many zeros to produce reliable forecasts. How would you like to handle them?
>
> (a) Keep all together — forecast everything with the same models
>     All series stay in the dataset. Simple, but may waste compute and pull down aggregate metrics.
>     Best when the low-signal count is small (< 5-10% of total).
>
> (b) Exclude + automatic fallback — remove from main pipeline, apply a simple rule
>     Non-forecastable series are excluded from model training/evaluation.
>     A lightweight fallback produces their forecasts immediately:
>       - Seasonal Naive — repeat the last complete seasonal cycle
>       - Naive (last value) — carry forward the last observed value
>       - Historical mean — average of all observed values
>       - Zero — fill forecast horizon with zeros
>
> (c) Exclude + separate forecasting job — run them through their own pipeline
>     Non-forecastable series get their own dedicated job with models you choose.
>     Results merge back in post-processing with full evaluation metrics."

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

#### Option (b): Exclude + automatic fallback

First, ask which fallback method to use:

> "Which fallback method for the {n_non_forecastable} non-forecastable series?
>
> (1) Seasonal Naive — repeat the last seasonal cycle
> (2) Naive (last value) — carry forward the final observation
> (3) Historical mean — use the average of all observations
> (4) Zero — fill with zeros (for intermittent/sparse series)"

Then create the filtered training table and compute fallback forecasts:

```sql
-- Forecastable series only (used by main pipeline in Steps 3-4)
CREATE OR REPLACE TABLE {catalog}.{schema}.{use_case}_train_data_forecastable AS
SELECT t.*
FROM {catalog}.{schema}.{use_case}_train_data t
INNER JOIN {catalog}.{schema}.{use_case}_series_profile p
  ON t.unique_id = p.unique_id
WHERE p.forecastability_class = 'high_confidence'
```

Compute the fallback forecasts on serverless:

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
distinct_series AS (SELECT DISTINCT unique_id, last_y FROM last_values)
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

#### Option (c): Exclude + separate forecasting job

Create both filtered training tables:

```sql
CREATE OR REPLACE TABLE {catalog}.{schema}.{use_case}_train_data_forecastable AS
SELECT t.* FROM {catalog}.{schema}.{use_case}_train_data t
INNER JOIN {catalog}.{schema}.{use_case}_series_profile p ON t.unique_id = p.unique_id
WHERE p.forecastability_class = 'high_confidence'
```

```sql
CREATE OR REPLACE TABLE {catalog}.{schema}.{use_case}_train_data_non_forecastable AS
SELECT t.* FROM {catalog}.{schema}.{use_case}_train_data t
INNER JOIN {catalog}.{schema}.{use_case}_series_profile p ON t.unique_id = p.unique_id
WHERE p.forecastability_class = 'low_signal'
```

Then ask the user which models to run on the non-forecastable series:

> "Which models should the non-forecastable series be forecasted with?
>
> BASELINE MODELS (recommended):
> [x] StatsForecastBaselineNaive
> [x] StatsForecastBaselineSeasonalNaive
>
> INTERMITTENT DEMAND MODELS (recommended if sparsity is high):
> [ ] StatsForecastTSB / StatsForecastADIDA / StatsForecastIMAPA
> [ ] StatsForecastCrostonClassic / StatsForecastCrostonOptimized / StatsForecastCrostonSBA
>
> OTHER LOCAL MODELS:
> [ ] StatsForecastAutoArima / StatsForecastAutoETS / StatsForecastAutoCES
>
> FOUNDATION MODELS (zero-shot, no training needed):
> [ ] ChronosBoltTiny / ChronosBoltMini / ChronosBoltSmall / ChronosBoltBase
>
> List the model names you want to run."

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

## Statistical Properties Computed

| Property | Method | Library |
|----------|--------|---------|
| **Stationarity** | Augmented Dickey-Fuller p-value | `statsmodels.tsa.stattools.adfuller` |
| **Seasonality Strength** | STL decomposition | `statsmodels.tsa.seasonal.STL` |
| **Trend Strength** | STL decomposition | `statsmodels.tsa.seasonal.STL` |
| **Spectral Entropy** | Normalized Shannon entropy of spectral density | `scipy.signal.periodogram` |
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

| Series Characteristics | Recommended Models |
|----------------------|-------------------|
| Strong seasonality (>0.6) + stationary | `StatsForecastAutoArima`, `AutoETS`, `AutoTheta`, `ChronosBoltBase`, `Chronos2`, `TimesFM_2_5_200m` |
| Strong trend (>0.6) + weak seasonality | `StatsForecastAutoArima`, `SKTimeProphet`, `NeuralForecastAutoNHITS`, `ChronosBoltBase` |
| High complexity (entropy >0.6) + long series (>200) | `NeuralForecastAutoNHITS`, `NeuralForecastAutoPatchTST`, `ChronosBoltBase`, `Chronos2` |
| Short series (<50 points) | `StatsForecastAutoETS`, `StatsForecastAutoCES`, `ChronosBoltBase` |
| Intermittent/sparse (sparsity >0.3) | `StatsForecastTSB`, `StatsForecastADIDA`, `StatsForecastIMAPA`, `StatsForecastCrostonClassic` |
| General / mixed characteristics | `StatsForecastAutoArima`, `NeuralForecastAutoNHITS`, `ChronosBoltBase`, `Chronos2` |
| Low-signal (non-forecastable) | `StatsForecastBaselineNaive`, `StatsForecastBaselineSeasonalNaive` |

## ⛔ STOP GATE — Step 11: Confirm before proceeding

Ask the user:

> "✅ Profiling complete for use case '{use_case}'.
>
> Summary:
> - Total series profiled: {total}
> - High-confidence (forecastable): {high} ({high_pct}%)
> - Low-signal (non-forecastable): {low} ({low_pct}%)
> - Non-forecastable strategy: {strategy_description}
> - Recommended model types: {model_types}
> - Profile table: {catalog}.{schema}.{use_case}_series_profile
> - Pipeline config: {catalog}.{schema}.{use_case}_pipeline_config
>
> Would you like to proceed to cluster provisioning and model selection?
> (a) Yes, proceed
> (b) No, stop here — I'll come back later"

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
