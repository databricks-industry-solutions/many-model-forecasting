# Profile and Classify Series (OPTIONAL)

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

## Placeholder values

| Placeholder | Value |
|-------------|-------|
| `{catalog}` | user's catalog |
| `{schema}` | user's schema |
| `{use_case}` | use case name from Skill 1 |
| `{train_table}` | `{use_case}_train_data` |
| `{freq}` | detected or user-specified frequency |
| `{prediction_length}` | user-specified forecast horizon (integer) |

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
   • Frequency: {freq}
   • Prediction length: {prediction_length}

   The job will run on serverless compute (no cluster startup delay).

   Proceed with profiling?
   (a) Yes, run profiling
   (b) No, skip profiling and go to model selection"
```

**Do NOT proceed until the user confirms.**

### Step 3: Generate notebook from template

**CRITICAL: Copy the template VERBATIM from `mmf_profiling_notebook_template.ipynb`, only replacing the `{placeholder}` tokens with actual values. Do NOT add, remove, or modify any other code.**

Replace these placeholders:
- `{catalog}` → user's catalog
- `{schema}` → user's schema
- `{use_case}` → use case name
- `{train_table}` → `{use_case}_train_data`
- `{freq}` → detected frequency
- `{prediction_length}` → user-specified forecast horizon

### Step 4: Save notebook locally and upload

Save the generated notebook to the **local project directory** at:
- `notebooks/{use_case}/run_profiling.ipynb`

Then upload it to the Databricks workspace at `notebooks/{use_case}/run_profiling`.

### Step 5: Create Workflow job on serverless compute

Create a single-task Workflow job on **serverless compute** (profiling is CPU-bound and benefits from instant startup):

```json
{
  "name": "{use_case}_profiling",
  "tasks": [{
    "task_key": "profile_series",
    "notebook_task": {
      "notebook_path": "notebooks/{use_case}/run_profiling"
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

## ⛔ STOP GATE — Step 9: Confirm before proceeding to next skill

```
AskUserQuestion:
  "✅ Profiling complete for use case '{use_case}'.

   Summary:
   • Total series profiled: {total}
   • High-confidence: {high} ({high_pct}%)
   • Low-signal: {low} ({low_pct}%)
   • Recommended model types: {model_types}
   • Profile table: {catalog}.{schema}.{use_case}_series_profile

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
