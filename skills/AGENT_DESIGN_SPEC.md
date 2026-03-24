# MMF Agent Skill-Kit: Technical Design Specification

## 1. Executive Summary

This document specifies the architecture for transforming the Many-Model Forecasting (MMF) framework into an **Agent-Native Skill-Kit**. The kit wraps the existing `mmf_sa` library with five high-level, function-calling compatible skills that an AI agent can orchestrate autonomously — from raw data discovery through to business-ready forecast evaluation.

The design philosophy is **minimal intrusion**: the core `mmf_sa` package remains untouched. All new capabilities are implemented as skill documents and notebook templates inside the existing `skills/databricks-skills/many-model-forecasting/` directory, following the established pattern of markdown step-files and `.ipynb` templates.

**Key design decision — merge, not fork**: The new 5-skill pipeline *replaces* the original 3-skill pipeline (`explore-data` → `setup-cluster` → `run-mmf`). Skills 1, 3, and 4 directly incorporate the full content of the originals and extend them with new capabilities. The original files (`1-explore-data.md`, `2-setup-the-mmf-cluster.md`, `3-run-mmf.md`) are retired. There is one canonical pipeline, not two parallel ones.

**Key design decision — use-case-prefixed assets**: At the very start of the pipeline (Skill 1, before any table discovery), the agent asks the user for a short **use case name** (e.g., `m4`, `rossmann`, `retail_sales`). This name prefixes every Delta table the pipeline creates, allowing multiple forecasting projects to coexist in the same schema without collision. The prefix maps directly to `mmf_sa`'s existing `use_case_name` parameter.

---

## 2. Architecture Overview

### 2.1 Original Architecture (Retired)

```
/explore-data  →  /setup-cluster  →  /run-mmf
     ↓                  ↓                ↓
  Discover &        Configure         Generate notebooks,
  prepare data     cluster(s)         submit Workflow job,
  → mmf_train_data                   analyze results
```

These three files are superseded by the five skills below. Their complete logic is preserved inside the new skills.

### 2.2 New Architecture (5-Skill Pipeline)

```
┌──────────────────────────────────────────────────────────────────────────┐
│                        MMF Agent Skill-Kit                               │
│                                                                          │
│  1  /prep-and-clean-data                                                 │
│       ↓ Asks for use_case name (e.g. "m4", "rossmann")                   │
│       ↓ Incorporates ALL of 1-explore-data.md (Steps 1-7)                │
│       ↓ Adds: anomaly detection, imputation, cleaning report             │
│       ↓ Output: {use_case}_train_data + {use_case}_cleaning_report       │
│                                                                          │
│  2  /profile-and-classify-series              (entirely new)             │
│       ↓ Statistical profiling + forecastability classification           │
│       ↓ Output: {use_case}_series_profile + model recommendations        │
│                                                                          │
│  3  /provision-forecasting-resources                                     │
│       ↓ Incorporates ALL of 2-setup-the-mmf-cluster.md (Steps 1-5)       │
│       ↓ Adds: auto-detect from profile, cluster reuse, UC verification   │
│       ↓ Output: Validated cluster configuration(s)                       │
│                                                                          │
│  4  /execute-mmf-forecast                                                │
│       ↓ Incorporates ALL of 3-run-mmf.md (Steps 1-7)                     │
│       ↓ Adds: pre-flight validation, profile-aware models, run summary   │
│       ↓ Output: {use_case}_evaluation_output, {use_case}_scoring_output, │
│       ↓         MLflow experiment, {use_case}_run_metadata               │
│                                                                          │
│  5  /post-process-and-evaluate                (entirely new)             │
│       ↓ Automates post-evaluation-analysis.ipynb patterns                │
│       ↓ Output: {use_case}_best_models + {use_case}_evaluation_summary   │
└──────────────────────────────────────────────────────────────────────────┘
```

### 2.3 Reuse Mapping

The following table shows exactly how each original step is reused:

| Original File | Original Step | New Skill | New Step | Change |
|---------------|--------------|-----------|----------|--------|
| `1-explore-data.md` | Step 1: Connect to workspace | Skill 1 | Step 1 | Verbatim |
| `1-explore-data.md` | Step 2: List tables | Skill 1 | Step 2 | Verbatim |
| `1-explore-data.md` | Step 3: Identify TS candidates | Skill 1 | Step 3 | Verbatim |
| `1-explore-data.md` | Step 4: Profile candidates | Skill 1 | Step 4 | Verbatim |
| `1-explore-data.md` | Step 4a: Detect frequency | Skill 1 | Step 4a | Verbatim |
| `1-explore-data.md` | Step 5: Validate with user | Skill 1 | Step 5 | Verbatim |
| `1-explore-data.md` | Step 6: Data quality checks | Skill 1 | — | **Removed**: subsumed by Steps 7 and 8 |
| `1-explore-data.md` | Step 7: Create mmf_train_data | Skill 1 | Step 6 | Renumbered; output table prefixed with `{use_case}` |
| — | — | Skill 1 | Step 7 | **New**: missing data assessment & imputation |
| — | — | Skill 1 | Step 8 | **New**: anomaly detection & capping |
| — | — | Skill 1 | Step 9 | **New**: cleaning report |
| `2-setup-the-mmf-cluster.md` | Step 1: Determine model types | Skill 3 | Step 1 | Extended (auto-detect option) |
| `2-setup-the-mmf-cluster.md` | Step 2: Determine cloud provider | Skill 3 | Step 2 | Verbatim |
| `2-setup-the-mmf-cluster.md` | Step 3: Select cluster config | Skill 3 | Step 4 | Verbatim (specs unchanged) |
| `2-setup-the-mmf-cluster.md` | Decision logic | Skill 3 | Step 4 | Verbatim |
| `2-setup-the-mmf-cluster.md` | Step 4: Present and validate | Skill 3 | Step 6 | Verbatim |
| `2-setup-the-mmf-cluster.md` | Step 5: Save configuration | Skill 3 | Step 7 | Verbatim |
| — | — | Skill 3 | Step 3 | **New**: check existing clusters |
| — | — | Skill 3 | Step 5 | **New**: UC enablement check |
| `3-run-mmf.md` | Parameters table | Skill 4 | Parameters | Verbatim |
| `3-run-mmf.md` | Available model names | Skill 4 | Model names | Verbatim |
| `3-run-mmf.md` | Step 1: Gather params | Skill 4 | Step 2 | Verbatim |
| `3-run-mmf.md` | Step 2: Generate notebooks | Skill 4 | Step 3 | Extended (one notebook per GPU model — CUDA memory constraint) |
| `3-run-mmf.md` | Step 3: Upload notebooks | Skill 4 | Step 4 | Extended (upload per-model GPU notebooks) |
| `3-run-mmf.md` | Step 4: Create Workflow job | Skill 4 | Step 5 | Extended (one task per GPU model, chained sequentially) |
| `3-run-mmf.md` | Serverless alternative | Skill 4 | Step 5 | Retained (local models only; GPU requires ML Runtime) |
| `3-run-mmf.md` | Step 5: Monitor execution | Skill 4 | Step 6 | Extended (structured status) |
| `3-run-mmf.md` | Step 6: Analyze results | Skill 4 | Step 7 | Simplified (metadata only; analysis moved to Skill 5) |
| `3-run-mmf.md` | Step 7: Suggest next steps | Skill 4 | Step 8 | Replaced with hand-off to Skill 5 |
| — | — | Skill 4 | Step 1 | **New**: pre-flight validation |
| — | — | Skill 4 | Step 1a | **New**: profile-aware models |
| — | — | Skill 4 | Step 1b | **New**: backtest strategy |

### 2.4 Use-Case-Prefixed Asset Naming

Every Delta table created by the pipeline is prefixed with the user's **use case name** (`{use_case}`). This is collected once in Skill 1 and propagated to all downstream skills.

| Asset | Old name (hardcoded) | New name (prefixed) |
|-------|---------------------|---------------------|
| Training data | `mmf_train_data` | `{use_case}_train_data` |
| Cleaning report | `mmf_cleaning_report` | `{use_case}_cleaning_report` |
| Series profile | `mmf_series_profile` | `{use_case}_series_profile` |
| Evaluation output | `mmf_evaluation_output` | `{use_case}_evaluation_output` |
| Scoring output | `mmf_scoring_output` | `{use_case}_scoring_output` |
| Run metadata | `mmf_run_metadata` | `{use_case}_run_metadata` |
| Best models | `mmf_best_models` | `{use_case}_best_models` |
| Evaluation summary | `mmf_evaluation_summary` | `{use_case}_evaluation_summary` |
| MLflow experiment | `/Users/{user}/mmf/{use_case}` | Same pattern — `use_case_name` feeds directly into `mmf_sa.run_forecast` |
| CPU cluster name | `mmf_cpu_cluster` | `{use_case}_cpu_cluster` |
| GPU cluster name | `mmf_gpu_cluster` | `{use_case}_gpu_cluster` |
| Workflow job name | `mmf_forecasting` | `{use_case}_forecasting` |
| Notebooks path | `notebooks/run_local` | `notebooks/{use_case}/run_local` |

The `{use_case}` value maps directly to `mmf_sa.run_forecast`'s `use_case_name` parameter, which is already used for the `use_case` column in evaluation/scoring output tables and the MLflow experiment path.

### 2.5 Design Principles

1. **Reuse over rewrite**: Skills 1, 3, and 4 incorporate the original text of `1-explore-data.md`, `2-setup-the-mmf-cluster.md`, and `3-run-mmf.md` and extend with new sections. Original Step 6 (read-only quality diagnostic) is the sole removal — subsumed by actionable cleaning steps.
2. **Single canonical pipeline**: The original 3 files are retired. `SKILL.md` points only to the 5-skill pipeline.
3. **Use-case isolation**: All assets are prefixed with the use case name, enabling multiple forecasting projects in the same catalog/schema.
4. **Notebook templates for compute-heavy work**: Statistical profiling requires Spark + pandas UDFs, delivered as `mmf_profiling_notebook_template.ipynb` — same pattern as the existing `mmf_local_notebook_template.ipynb`.
5. **MCP-first execution**: All SQL, cluster management, and job operations use Databricks MCP tools.
6. **Minimal intrusion**: Zero changes to `mmf_sa/` source code.

---

## 3. Skill Specifications

### 3.1 Skill 1: `prep_and_clean_data`

**File**: `skills/databricks-skills/many-model-forecasting/1-prep-and-clean-data.md`
**Replaces**: `1-explore-data.md`
**Slash command**: `/prep-and-clean-data <catalog> <schema>`

#### Purpose
Collect the use case name, discover time series tables, map columns to MMF schema, run data quality checks, apply automated cleaning (imputation, anomaly detection, timestamp alignment), and create the `{use_case}_train_data` table.

#### Inputs
| Input | Type | Description |
|-------|------|-------------|
| `catalog` | string | Unity Catalog name |
| `schema` | string | Schema name |

#### Steps — Use Case Name Collection

**Step 0: Collect use case name (new)**

Before any data exploration, ask the user for a short use case identifier:

```
AskUserQuestion:
  "Provide a short use case name (e.g., m4, rossmann, retail_sales).
   This will prefix all tables and assets created by the pipeline."
  Options: [free text]
```

Validation rules:
- Lowercase alphanumeric + underscores only (must be valid as a Delta table name component)
- 1-30 characters
- Cannot start with a number

Store as `{use_case}` and propagate to all downstream skills. All table names in the pipeline use the pattern `{use_case}_<asset_name>` (e.g., `m4_train_data`, `rossmann_cleaning_report`).

#### Steps — Reused from `1-explore-data.md`

**Steps 1-5 are copied verbatim from `1-explore-data.md`**:

- **Step 1**: Connect to workspace (`connect_to_workspace`)
- **Step 2**: List tables (`SHOW TABLES IN {catalog}.{schema}`)
- **Step 3**: Identify time series candidates (`DESCRIBE TABLE`, column type matching)
- **Step 4**: Profile candidates (row count, date range, distinct groups)
- **Step 4a**: Detect source frequency (avg gap query)
- **Step 5**: Validate with user (`AskUserQuestion` — table, columns, regressors)

All SQL queries, thresholds, and `AskUserQuestion` checkpoints are identical to the original.

> **Note — original Step 6 (Data quality checks) removed.** Its missing-value diagnostics are subsumed by Step 7 (which assesses and acts on missing data). Its negative-value diagnostics are subsumed by Step 8 (IQR-based anomaly detection catches domain-inappropriate values). Keeping a read-only diagnostic before the actionable steps added no value.

#### Steps — Data Preparation & Cleaning

**Step 6: Create {use_case}_train_data**

The CREATE TABLE SQL from `1-explore-data.md` Step 7 is preserved (hourly / daily / weekly / monthly variants with frequency-aware alignment), with the output table name `{use_case}_train_data`. This creates the base table with the MMF-required schema (`unique_id`, `ds`, `y`). Steps 7 and 8 then clean this table in-place.

**Step 7: Missing Data Assessment & Imputation**

Missing data in time series comes in two forms, both of which must be detected:
- **Explicit NULLs**: Rows exist but `y` is NULL.
- **Implicit gaps**: Entire rows are absent between the first and last timestamp of a series.

The current `{use_case}_train_data` table may have implicit gaps (e.g., daily data with Jan 1, Jan 2, Jan 5 — missing Jan 3 and Jan 4 as rows). A query that only counts `y IS NULL` would report 0% missing for such a series. To catch both forms, generate an expected date spine per series and left-join actual data.

**Step 7a: Generate date spine and detect all gaps**

```sql
-- Build the expected regular grid per series based on detected {freq}
WITH series_bounds AS (
  SELECT unique_id, MIN(ds) AS min_ds, MAX(ds) AS max_ds
  FROM {catalog}.{schema}.{use_case}_train_data
  GROUP BY unique_id
),
-- Generate expected timestamps at the detected frequency
-- Daily example (replace INTERVAL for H/W/M):
date_spine AS (
  SELECT s.unique_id, EXPLODE(SEQUENCE(s.min_ds, s.max_ds, INTERVAL 1 DAY)) AS expected_ds
  FROM series_bounds s
),
-- Left join actual data onto the spine
spine_joined AS (
  SELECT
    sp.unique_id,
    sp.expected_ds AS ds,
    t.y
  FROM date_spine sp
  LEFT JOIN {catalog}.{schema}.{use_case}_train_data t
    ON sp.unique_id = t.unique_id AND sp.expected_ds = t.ds
)
SELECT
  unique_id,
  COUNT(*) AS expected_count,
  SUM(CASE WHEN y IS NOT NULL THEN 1 ELSE 0 END) AS actual_count,
  SUM(CASE WHEN y IS NULL THEN 1 ELSE 0 END) AS missing_count,
  ROUND(SUM(CASE WHEN y IS NULL THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS missing_pct
FROM spine_joined
GROUP BY unique_id
ORDER BY missing_pct DESC
```

The `SEQUENCE` interval changes by frequency:
- Hourly (`freq=H`): `INTERVAL 1 HOUR`
- Daily (`freq=D`): `INTERVAL 1 DAY`
- Weekly (`freq=W`): `INTERVAL 7 DAY`
- Monthly (`freq=M`): `INTERVAL 1 MONTH`

This catches both explicit NULLs (rows with `y IS NULL`) and implicit gaps (rows that don't exist) in a single pass.

**Step 7b: Present summary and confirm strategy with user**

```
AskUserQuestion:
  "Missing data summary (including implicit date gaps):
   - {n_clean} series are complete (no gaps)
   - {n_low} series have < 5% missing → Suggest: linear interpolation (avg of neighbors)
   - {n_mid} series have 5-20% missing → Suggest: forward fill (last known value)
   - {n_high} series have > 20% missing → Suggest: exclude from forecasting

   How would you like to proceed?
   (a) Apply suggested strategy
   (b) Use a single strategy for all (interpolation / forward fill / fill with 0 / drop nulls)
   (c) Skip imputation — keep nulls as-is
   (d) Adjust the exclusion threshold (currently 20%)"
```

**Step 7c: Backfill the date spine and apply imputation**

First, replace `{use_case}_train_data` with the complete spine so that implicit gaps become explicit NULL rows:

```sql
CREATE OR REPLACE TABLE {catalog}.{schema}.{use_case}_train_data AS
SELECT sp.unique_id, sp.expected_ds AS ds, t.y
FROM date_spine sp
LEFT JOIN {catalog}.{schema}.{use_case}_train_data t
  ON sp.unique_id = t.unique_id AND sp.expected_ds = t.ds
```

Then apply the chosen imputation on the now-explicit NULLs:

- **Linear interpolation**: `(LAG(y IGNORE NULLS) + LEAD(y IGNORE NULLS)) / 2`
- **Forward fill**: `LAST_VALUE(y IGNORE NULLS) OVER (PARTITION BY unique_id ORDER BY ds)`
- **Fill with 0**: `COALESCE(y, 0)` — appropriate for count/demand data where absence means zero activity
- **Exclusion**: Remove series exceeding the threshold from `{use_case}_train_data`

Log the count of imputed values per series and excluded series for the cleaning report.

**Step 8: Anomaly Detection & Capping**

First, compute IQR statistics and flag anomalies per series:

```sql
WITH stats AS (
  SELECT unique_id,
         PERCENTILE(y, 0.25) AS q1,
         PERCENTILE(y, 0.75) AS q3,
         PERCENTILE(y, 0.75) - PERCENTILE(y, 0.25) AS iqr
  FROM {catalog}.{schema}.{use_case}_train_data
  GROUP BY unique_id
),
flagged AS (
  SELECT t.unique_id, t.ds, t.y,
         s.q1, s.q3, s.iqr,
         s.q1 - 1.5 * s.iqr AS lower_bound,
         s.q3 + 1.5 * s.iqr AS upper_bound,
         CASE WHEN t.y < s.q1 - 1.5 * s.iqr OR t.y > s.q3 + 1.5 * s.iqr
              THEN TRUE ELSE FALSE END AS is_anomaly
  FROM {catalog}.{schema}.{use_case}_train_data t
  JOIN stats s ON t.unique_id = s.unique_id
)
SELECT unique_id,
       COUNT(*) AS total_points,
       SUM(CASE WHEN is_anomaly THEN 1 ELSE 0 END) AS anomaly_count,
       ROUND(SUM(CASE WHEN is_anomaly THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS anomaly_pct,
       MIN(CASE WHEN is_anomaly THEN y END) AS min_anomaly,
       MAX(CASE WHEN is_anomaly THEN y END) AS max_anomaly,
       MIN(lower_bound) AS lower_bound,
       MAX(upper_bound) AS upper_bound
FROM flagged
GROUP BY unique_id
HAVING anomaly_count > 0
ORDER BY anomaly_pct DESC
```

Present the anomaly summary and let the user define the capping range:

```
AskUserQuestion:
  "Anomaly detection summary (IQR-based):
   - {n_clean} series have no anomalies
   - {n_affected} series have outliers ({total_anomalies} points total, {anomaly_pct}% overall)
   - Default capping range: [Q1 - 1.5×IQR, Q3 + 1.5×IQR]

   How would you like to proceed?
   (a) Cap at 1.5×IQR (default — moderate, removes typical outliers)
   (b) Cap at 3.0×IQR (conservative — only removes extreme outliers)
   (c) Custom multiplier: enter a value (e.g., 2.0)
   (d) Skip anomaly capping — keep all values as-is"
```

If the user chooses (a), (b), or (c), apply capping using the chosen `{iqr_multiplier}`:

```sql
UPDATE {catalog}.{schema}.{use_case}_train_data AS t
SET y = CASE
  WHEN t.y < s.q1 - {iqr_multiplier} * s.iqr THEN s.q1 - {iqr_multiplier} * s.iqr
  WHEN t.y > s.q3 + {iqr_multiplier} * s.iqr THEN s.q3 + {iqr_multiplier} * s.iqr
  ELSE t.y
END
FROM (
  SELECT unique_id,
         PERCENTILE(y, 0.25) AS q1,
         PERCENTILE(y, 0.75) AS q3,
         PERCENTILE(y, 0.75) - PERCENTILE(y, 0.25) AS iqr
  FROM {catalog}.{schema}.{use_case}_train_data
  GROUP BY unique_id
) s
WHERE t.unique_id = s.unique_id
```

Log the count of capped values per series for the cleaning report.

**Step 9: Create {use_case}_cleaning_report**

```sql
CREATE OR REPLACE TABLE {catalog}.{schema}.{use_case}_cleaning_report AS
SELECT
  unique_id,
  original_count,
  final_count,
  missing_filled,
  imputation_method,
  anomalies_capped,
  iqr_multiplier,
  CASE WHEN excluded THEN TRUE ELSE FALSE END AS excluded,
  exclusion_reason
FROM ...
```

Present cleaning summary to the user.

#### Output Schema

Primary output — `{use_case}_train_data` (table name is now prefixed; schema unchanged from original):

| Column | Type | Description |
|--------|------|-------------|
| `unique_id` | STRING | Series identifier |
| `ds` | TIMESTAMP | Aligned timestamp |
| `y` | DOUBLE | Cleaned target value |

Secondary output — `{use_case}_cleaning_report`:

| Column | Type | Description |
|--------|------|-------------|
| `unique_id` | STRING | Series identifier |
| `original_count` | INT | Rows before cleaning |
| `final_count` | INT | Rows after cleaning |
| `missing_filled` | INT | Number of imputed values |
| `imputation_method` | STRING | Method used: `interpolation`, `forward_fill`, or `none` |
| `anomalies_capped` | INT | Number of capped outliers |
| `iqr_multiplier` | DOUBLE | IQR multiplier used for capping (NULL if skipped) |
| `excluded` | BOOLEAN | Whether series was excluded |
| `exclusion_reason` | STRING | Reason if excluded (e.g., `missing_pct > 20%`) |

---

### 3.2 Skill 2: `profile_and_classify_series`

**File**: `skills/databricks-skills/many-model-forecasting/2-profile-and-classify-series.md`
**Template**: `skills/databricks-skills/many-model-forecasting/mmf_profiling_notebook_template.ipynb`
**Slash command**: `/profile-and-classify-series <catalog> <schema>`
**Entirely new** — no original file to reuse.

#### Purpose
Calculate statistical properties for each time series and partition data into "High-Confidence" (forecastable) and "Low-Signal" (non-forecastable) groups. Recommend specific MMF model classes for each partition.

#### Inputs
| Input | Type | Description |
|-------|------|-------------|
| `catalog` | string | Unity Catalog name |
| `schema` | string | Schema name |
| `use_case` | string | Use case name from Skill 1 (prefixes all assets) |
| `train_table` | string | Training table (default: `{use_case}_train_data`) |
| `freq` | string | Detected or user-specified frequency |

#### Logic

This skill requires PySpark + pandas computation that exceeds what MCP SQL alone can do. It follows the existing template pattern: the agent generates a notebook from `mmf_profiling_notebook_template.ipynb`, uploads it, and runs it as a Databricks job task.

**Statistical Properties Computed (per series)**:

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

**Classification Logic**:

```
High-Confidence (Forecastable):
  - spectral_entropy < 0.8
  - series_length >= 2 * prediction_length
  - sparsity < 0.5
  - snr > 0.1

Low-Signal (Non-Forecastable):
  - Everything else
```

Thresholds are configurable via `AskUserQuestion`.

**Model Recommendation Logic**:

| Series Characteristics | Recommended Models | Rationale |
|----------------------|-------------------|-----------|
| Strong seasonality (>0.6) + stationary | `StatsForecastAutoArima`, `StatsForecastAutoETS`, `StatsForecastAutoTheta`, `ChronosBoltBase`, `Chronos2`, `TimesFM_2_5_200m` | Classical models excel with clear seasonal patterns; foundation models as benchmark |
| Strong trend (>0.6) + weak seasonality | `StatsForecastAutoArima`, `SKTimeProphet`, `NeuralForecastAutoNHITS`, `ChronosBoltBase`, `Chronos2`, `TimesFM_2_5_200m` | ARIMA captures trends; Prophet handles changepoints; foundation models as benchmark |
| High complexity (entropy >0.6) + long series (>200) | `NeuralForecastAutoNHITS`, `NeuralForecastAutoPatchTST`, `ChronosBoltBase`, `Chronos2`, `TimesFM_2_5_200m` | Auto neural models learn complex non-linear patterns; largest foundation models for best zero-shot quality |
| Short series (<50 points) | `StatsForecastAutoETS`, `StatsForecastAutoCES`, `ChronosBoltBase`, `Chronos2`, `TimesFM_2_5_200m` | Simple models + largest zero-shot foundation models (no training needed) |
| Intermittent/sparse (sparsity >0.3) | `StatsForecastTSB`, `StatsForecastADIDA`, `StatsForecastIMAPA`, `StatsForecastCrostonClassic` | Specialized intermittent demand models |
| General / mixed characteristics | `StatsForecastAutoArima`, `NeuralForecastAutoNHITS`, `ChronosBoltBase`, `Chronos2`, `TimesFM_2_5_200m` | Broad coverage across model families; largest foundation models |
| Low-signal (non-forecastable) | `StatsForecastBaselineNaive`, `StatsForecastBaselineSeasonalNaive` | Baseline only; flag for human review |

#### Output

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
| `forecastability_class` | STRING | "high_confidence" or "low_signal" |
| `recommended_models` | STRING | Comma-separated model names |
| `model_types_needed` | STRING | "local", "local,foundation", etc. |

---

### 3.3 Skill 3: `provision_forecasting_resources`

**File**: `skills/databricks-skills/many-model-forecasting/3-provision-forecasting-resources.md`
**Replaces**: `2-setup-the-mmf-cluster.md`
**Slash command**: `/provision-forecasting-resources`

#### Purpose
Determine required cluster types (from profile or user), configure clusters with the correct specs, verify Unity Catalog enablement, and optionally reuse or restart existing clusters.

#### Inputs
| Input | Type | Description |
|-------|------|-------------|
| `catalog` | string | Unity Catalog name |
| `schema` | string | Schema name |
| `use_case` | string | Use case name from Skill 1 (to read `{use_case}_series_profile` if it exists) |
| `cloud_provider` | string | AWS, Azure, or GCP |

#### Steps — Reused from `2-setup-the-mmf-cluster.md`

**Step 1: Determine model classes (enhanced)**

First, check if `{use_case}_series_profile` exists:
```sql
SELECT DISTINCT model_types_needed
FROM {catalog}.{schema}.{use_case}_series_profile
WHERE forecastability_class = 'high_confidence'
```

If the table exists → auto-detect required model classes from the profiling output and present as a suggestion.

In both cases (profile exists or not), present the full model catalog and let the user select one or more model classes:

```
AskUserQuestion:
  "Which model classes do you want to run? Select one or more:

   [ ] Local models (CPU cluster)
       StatsForecastBaselineWindowAverage, StatsForecastBaselineSeasonalWindowAverage,
       StatsForecastBaselineNaive, StatsForecastBaselineSeasonalNaive,
       StatsForecastAutoArima, StatsForecastAutoETS, StatsForecastAutoCES,
       StatsForecastAutoTheta, StatsForecastAutoTbats, StatsForecastAutoMfles,
       StatsForecastTSB, StatsForecastADIDA, StatsForecastIMAPA,
       StatsForecastCrostonClassic, StatsForecastCrostonOptimized,
       StatsForecastCrostonSBA, SKTimeProphet

   [ ] Global models (GPU cluster)
       NeuralForecastRNN, NeuralForecastLSTM, NeuralForecastNBEATSx,
       NeuralForecastNHITS, NeuralForecastAutoRNN, NeuralForecastAutoLSTM,
       NeuralForecastAutoNBEATSx, NeuralForecastAutoNHITS,
       NeuralForecastAutoTiDE, NeuralForecastAutoPatchTST

   [ ] Foundation models (GPU cluster)
       ChronosBoltTiny, ChronosBoltMini, ChronosBoltSmall, ChronosBoltBase,
       Chronos2, Chronos2Small, Chronos2Synth, TimesFM_2_5_200m

   {if profile exists: 'Based on series profiling, suggested classes: {auto_detected_classes}'}
   You can select any combination (e.g., local + foundation)."
  Options: [local, global, foundation] (multi-select)
```

The user's selection determines which clusters are provisioned:
- **Local only** → CPU cluster
- **Global only** → GPU cluster
- **Foundation only** → GPU cluster
- **Local + Global** → CPU + GPU clusters
- **Local + Foundation** → CPU + GPU clusters
- **Global + Foundation** → GPU cluster
- **All three** → CPU + GPU clusters

**Step 2: Determine cloud provider** — Verbatim from `2-setup-the-mmf-cluster.md` Step 2.

Ask the user via `AskUserQuestion`: AWS, Azure, or GCP.

**Step 3: Check existing clusters (new)**

Use MCP `list_clusters` to find clusters matching the required configurations:
- Match by runtime version (`17.3.x-cpu-ml-scala2.13` for CPU, `18.0.x-gpu-ml-scala2.13` for GPU)
- Match by node type for the target cloud provider

Decision logic:
```
if matching_cluster.state == "RUNNING":
    → "Found running cluster {name}. Reuse it?"
elif matching_cluster.state == "TERMINATED":
    → "Found terminated cluster {name}. Start it?"
    → Use start_cluster MCP tool if confirmed
else:
    → Proceed to generate ephemeral job cluster config
```

**Step 4: Select cluster configuration** — Verbatim from `2-setup-the-mmf-cluster.md` Step 3.

All cluster specifications are unchanged:

**CPU Cluster (`{use_case}_cpu_cluster`)**:

| Setting | Value |
|---------|-------|
| **Runtime** | `17.3.x-cpu-ml-scala2.13` |
| **Node type (AWS)** | `i3.xlarge` |
| **Node type (Azure)** | `Standard_DS3_v2` |
| **Node type (GCP)** | `n1-standard-4` |
| **Workers** | Dynamic — see sizing logic below |
| **Spark config** | `spark.sql.execution.arrow.enabled=true`, `spark.sql.adaptive.enabled=false`, `spark.databricks.delta.formatCheck.enabled=false`, `spark.databricks.delta.schema.autoMerge.enabled=true` |

**CPU worker sizing logic:**

Query the number of distinct series:
```sql
SELECT COUNT(DISTINCT unique_id) AS n_series
FROM {catalog}.{schema}.{use_case}_train_data
```

Suggest workers based on series count:

| Series count | Suggested workers | Rationale |
|-------------|-------------------|-----------|
| < 100 | 0 (single-node) | No parallelism needed |
| 100 – 1,000 | 4 | Moderate parallelism |
| 1,000 – 10,000 | 6 | Each worker handles ~1,500 series |
| 10,000 – 100,000 | 8 | High parallelism for large-scale |
| > 100,000 | 10 | Maximum recommended; beyond this consider partitioning |

Present the suggestion and let the user override:

```
AskUserQuestion:
  "Your dataset has {n_series} distinct time series.
   Suggested CPU workers: {suggested_workers}

   How many workers would you like?
   (a) Use suggested: {suggested_workers} workers
   (b) Custom: enter a number (min 1, max 64)"
```

**GPU Cluster (`{use_case}_gpu_cluster`)**:

| Setting | Value |
|---------|-------|
| **Runtime** | `18.0.x-gpu-ml-scala2.13` |
| **Node type** | User-selectable — see GPU instance options below |
| **Workers** | 0 (single-node) |
| **Spark config** | `spark.databricks.delta.formatCheck.enabled=false`, `spark.databricks.delta.schema.autoMerge.enabled=true` |

**GPU instance selection:**

The cluster runs as single-node (0 workers). The user chooses the GPU instance type, which determines the number of GPUs available:

```
AskUserQuestion:
  "Select a GPU instance type:

   AWS:
   (a) g5.xlarge    — 1× A10G GPU, 24 GB  (small foundation models)
   (b) g5.2xlarge   — 1× A10G GPU, 24 GB, more CPU/RAM
   (c) g5.12xlarge  — 4× A10G GPUs, 96 GB  (recommended for global + foundation)
   (d) g5.48xlarge  — 8× A10G GPUs, 192 GB (large-scale training)

   Azure:
   (a) Standard_NC4as_T4_v3    — 1× T4 GPU, 16 GB
   (b) Standard_NC8as_T4_v3    — 1× T4 GPU, 16 GB, more CPU/RAM
   (c) Standard_NV36ads_A10_v5 — 2× A10 GPUs, 48 GB  (recommended)
   (d) Standard_NC24ads_A100_v4 — 1× A100 GPU, 80 GB (large models)

   GCP:
   (a) g2-standard-4   — 1× L4 GPU, 24 GB
   (b) g2-standard-8   — 1× L4 GPU, 24 GB, more CPU/RAM
   (c) g2-standard-48  — 4× L4 GPUs, 96 GB  (recommended)
   (d) a2-highgpu-1g   — 1× A100 GPU, 40 GB (large models)"
```

**Decision logic** — verbatim from `2-setup-the-mmf-cluster.md`:
- Local models only → CPU cluster only
- Global models → GPU cluster required
- Foundation models → GPU cluster required
- Local + global/foundation → Both CPU and GPU clusters

**Step 5: Unity Catalog enablement verification (new)**

All clusters MUST have UC enabled. Verify:
```json
{
  "data_security_mode": "USER_ISOLATION",
  "spark_conf": {
    "spark.databricks.unityCatalog.enabled": "true"
  }
}
```

If cluster doesn't have UC → warn user, suggest adding required Spark config.

**Step 6: Present and validate** — Verbatim from `2-setup-the-mmf-cluster.md` Step 4.

Use `AskUserQuestion` to confirm the complete configuration.

**Step 7: Save configuration** — Verbatim from `2-setup-the-mmf-cluster.md` Step 5.

Save the cluster configuration locally for Skill 4 to consume.

**MMF Installation** section — verbatim from `2-setup-the-mmf-cluster.md`.

#### Output
- Validated cluster configuration(s) with UC enablement
- Cluster IDs if existing clusters are reused
- Configuration JSON for ephemeral job clusters

---

### 3.4 Skill 4: `execute_mmf_forecast`

**File**: `skills/databricks-skills/many-model-forecasting/4-execute-mmf-forecast.md`
**Replaces**: `3-run-mmf.md`
**Slash command**: `/execute-mmf-forecast <catalog> <schema>`

#### Purpose
Validate parameters, optionally read model recommendations from profiling, generate and submit forecasting notebooks (one per GPU model to avoid CUDA memory issues), monitor execution, and log run metadata.

#### Parameters — Verbatim from `3-run-mmf.md` (with use-case-prefixed defaults)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `use_case` | From Skill 1 | Use case name — prefixes all output tables |
| `freq` | Auto-detected from data | Time series frequency (`D`, `W`, `M`, `H`) |
| `prediction_length` | Ask user | Forecast horizon in time steps |
| `backtest_length` | Derived from backtest strategy | Total historical points reserved for backtesting |
| `stride` | Derived from backtest strategy | Step size between backtest windows |
| `metric` | `smape` | Evaluation metric (`smape`, `mape`, `mae`, `mse`, `rmse`) |
| `active_models` | `["StatsForecastAutoArima", "StatsForecastAutoETS", "StatsForecastAutoCES", "StatsForecastAutoTheta"]` | Models to run |
| `train_data` | `<catalog>.<schema>.{use_case}_train_data` | Input table |
| `evaluation_output` | `<catalog>.<schema>.{use_case}_evaluation_output` | Evaluation results table |
| `scoring_output` | `<catalog>.<schema>.{use_case}_scoring_output` | Scoring results table |
| `group_id` | `unique_id` | Column name for series identifier |
| `date_col` | `ds` | Column name for timestamp |
| `target` | `y` | Column name for target value |

#### Available model names — Verbatim from `3-run-mmf.md`

**Local models (CPU):** `StatsForecastBaselineWindowAverage`, `StatsForecastBaselineSeasonalWindowAverage`, `StatsForecastBaselineNaive`, `StatsForecastBaselineSeasonalNaive`, `StatsForecastAutoArima`, `StatsForecastAutoETS`, `StatsForecastAutoCES`, `StatsForecastAutoTheta`, `StatsForecastAutoTbats`, `StatsForecastAutoMfles`, `StatsForecastTSB`, `StatsForecastADIDA`, `StatsForecastIMAPA`, `StatsForecastCrostonClassic`, `StatsForecastCrostonOptimized`, `StatsForecastCrostonSBA`, `SKTimeProphet`

**Global models (GPU):** `NeuralForecastRNN`, `NeuralForecastLSTM`, `NeuralForecastNBEATSx`, `NeuralForecastNHITS`, `NeuralForecastAutoRNN`, `NeuralForecastAutoLSTM`, `NeuralForecastAutoNBEATSx`, `NeuralForecastAutoNHITS`, `NeuralForecastAutoTiDE`, `NeuralForecastAutoPatchTST`

**Foundation models (GPU):** `ChronosBoltTiny`, `ChronosBoltMini`, `ChronosBoltSmall`, `ChronosBoltBase`, `Chronos2`, `Chronos2Small`, `Chronos2Synth`, `TimesFM_2_5_200m`

#### Steps — New (Pre-flight)

**Step 1: Pre-flight parameter validation (new)**

Before generating notebooks, validate:

| Validation | Rule | Error |
|-----------|------|-------|
| `backtest_length >= prediction_length` | Mandatory | "Backtest length must be >= prediction length" |
| `stride <= backtest_length` | Mandatory | "Stride must be <= backtest length" |
| `freq` ∈ {H, D, W, M} | Mandatory | "Unsupported frequency" |
| `active_models` all in model names list above | Mandatory | "Unknown model: {name}" |
| Model types consistent with cluster config | Warning | "GPU models selected but only CPU cluster configured" |
| `train_table` exists and has required schema | Mandatory | "Training table missing or invalid schema" |

**Step 1a: Profile-aware model selection (new, optional)**

If `{use_case}_series_profile` exists, read recommended models:
```sql
SELECT COLLECT_SET(recommended_models) AS all_recommended
FROM {catalog}.{schema}.{use_case}_series_profile
WHERE forecastability_class = 'high_confidence'
```

Parse the comma-separated model names and propose them as `active_models`. Present to user via `AskUserQuestion` for confirmation.

If `{use_case}_series_profile` does NOT exist → skip this step; use the user-specified or default `active_models`.

**Step 1b: Define backtest strategy (new)**

After the user specifies `prediction_length`, guide them through configuring the backtest. First, explain the concept visually:

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

#### Steps — Reused from `3-run-mmf.md`

**Step 2: Gather and validate parameters** — Verbatim from `3-run-mmf.md` Step 1.

Confirm input table exists, present all parameters (including the derived `backtest_length` and `stride`) via `AskUserQuestion`.

**Step 3: Generate notebooks by substituting placeholders** — Extended from `3-run-mmf.md` Step 2.

**CRITICAL: Copy the template VERBATIM from the template files, only replacing the `{placeholder}` tokens.**

Generate a shared `run_id` (UUID) that will be passed to all notebooks, grouping results across separate GPU sessions.

**Local models notebook**: Generate a single notebook from `mmf_local_notebook_template.ipynb` with all local models in `{active_models}`. Local models (CPU/statsforecast) do not have CUDA memory constraints and can run together.

**GPU model notebooks (one per model)**: For each GPU model (global or foundation), generate a separate notebook from `mmf_gpu_notebook_template.ipynb`. Each notebook receives:
- `{model}` — a single model name (e.g., `NeuralForecastAutoNHITS`)
- `{pip_extras}` — `[global]` for NeuralForecast models, `[foundation]` for others
- `{run_id}` — the shared run ID
- `{num_nodes}` — number of GPU nodes (default `1` for single-node)
- All other common placeholders (catalog, schema, freq, etc.)

> **Why one notebook per GPU model.** PyTorch allocates CUDA memory that cannot be freed within the same Python process. Running multiple GPU models in sequence causes OOM when the second model loads. The example notebooks (`examples/monthly/global_monthly.ipynb`) solve this via `dbutils.notebook.run()` per model — each invocation gets a fresh kernel. The Workflow job equivalent is one task per GPU model.

Updated placeholder table for GPU notebooks:

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

**Step 4: Upload notebooks** — Extended from `3-run-mmf.md` Step 3.

Upload generated notebooks to the workspace:
- `notebooks/{use_case}/run_local` — local models (single notebook, all local models)
- `notebooks/{use_case}/run_{model_name}` — one per GPU model (e.g., `run_NeuralForecastAutoNHITS`, `run_ChronosBoltBase`)

**Step 5: Create Workflow job** — Extended from `3-run-mmf.md` Step 4.

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
- Ephemeral job cluster specs — verbatim from `3-run-mmf.md` Step 4 (with `{use_case}` prefixed names).
- Serverless alternative — verbatim from `3-run-mmf.md` Step 4 (applicable to local models only; GPU models require ML Runtime clusters).

**Step 6: Monitor execution (enhanced)**

Base behavior from `3-run-mmf.md` Step 5 (poll job run status until completion), enhanced with structured status updates showing per-model progress:

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

**Step 7: Log run metadata**

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

**Step 8: Hand off to post-processing**

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

> **Note — analysis moved to Skill 5.** Previous design had result analysis queries (best model per series, avg metric per model, worst series) and a run summary notebook in Skill 4. These overlapped with Skill 5's more thorough analysis (multi-metric evaluation, WAPE, profiling cross-reference, business reporting). All analysis is now consolidated in Skill 5 to avoid duplication.

#### Outputs
- Delta table `<catalog>.<schema>.{use_case}_evaluation_output` — verbatim from `3-run-mmf.md` (name now prefixed)
- Delta table `<catalog>.<schema>.{use_case}_scoring_output` — verbatim from `3-run-mmf.md` (name now prefixed)
- MLflow experiment at `/Users/{user}/mmf/{use_case}` — `use_case` maps to `run_forecast(use_case_name=...)`
- Delta table `<catalog>.<schema>.{use_case}_run_metadata`

---

### 3.5 Skill 5: `post_process_and_evaluate`

**File**: `skills/databricks-skills/many-model-forecasting/5-post-process-and-evaluate.md`
**Slash command**: `/post-process-and-evaluate <catalog> <schema>`
**Entirely new** — automates patterns from `examples/post-evaluation-analysis.ipynb`.

#### Purpose
Calculate multiple accuracy metrics, perform best-model selection per series, and format results for business consumption.

#### Inputs
| Input | Type | Description |
|-------|------|-------------|
| `catalog` | string | Unity Catalog name |
| `schema` | string | Schema name |
| `use_case` | string | Use case name from Skill 1 (prefixes all table names) |
| `metric` | string | Primary metric (default: `smape`) |
| `evaluation_table` | string | Default: `{use_case}_evaluation_output` |
| `scoring_table` | string | Default: `{use_case}_scoring_output` |

#### Logic

**Step 1: Verify outputs exist**
```sql
SELECT COUNT(*) AS eval_count FROM {catalog}.{schema}.{evaluation_table}
```
```sql
SELECT COUNT(*) AS score_count FROM {catalog}.{schema}.{scoring_table}
```

**Step 2: Compute multi-metric evaluation**

Calculate MAPE, sMAPE, and WAPE from the stored `forecast` and `actual` arrays:

```sql
SELECT
    unique_id,
    model,
    AVG(metric_value) AS avg_primary_metric,
    AVG(
      AGGREGATE(
        TRANSFORM(
          ARRAYS_ZIP(forecast, actual),
          x -> ABS(x.actual - x.forecast)
        ), CAST(0.0 AS DOUBLE), (acc, x) -> acc + x
      ) /
      NULLIF(AGGREGATE(
        TRANSFORM(actual, x -> ABS(x)),
        CAST(0.0 AS DOUBLE), (acc, x) -> acc + x
      ), 0)
    ) AS wape
FROM {catalog}.{schema}.{evaluation_table}
GROUP BY unique_id, model
```

**Step 3: Best model selection per series**

Pattern from `post-evaluation-analysis.ipynb` Cell 7:

```sql
CREATE OR REPLACE TABLE {catalog}.{schema}.{use_case}_best_models AS
SELECT eval.unique_id, eval.model, eval.avg_metric, score.ds, score.y
FROM (
  SELECT unique_id, model, avg_metric,
         RANK() OVER (PARTITION BY unique_id ORDER BY avg_metric ASC) AS rank
  FROM (
    SELECT unique_id, model, AVG(metric_value) AS avg_metric
    FROM {catalog}.{schema}.{evaluation_table}
    GROUP BY unique_id, model
    HAVING AVG(metric_value) IS NOT NULL
  )
) AS eval
INNER JOIN {catalog}.{schema}.{scoring_table} AS score
  ON eval.unique_id = score.unique_id AND eval.model = score.model
WHERE eval.rank = 1
```

**Step 4: Model ranking (wins count)**

Pattern from `post-evaluation-analysis.ipynb` Cell 9:

```sql
SELECT model, COUNT(*) AS wins_count,
       ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 2) AS wins_pct
FROM {catalog}.{schema}.{use_case}_best_models
GROUP BY model
ORDER BY wins_count DESC
```

**Step 5: Business-ready summary report**

Present to the user:
- Total series evaluated
- Number of distinct models that "won" at least one series
- Top 3 models by wins count
- Average metric across all series (overall forecast quality)
- Worst 10 series (potential data quality issues for re-investigation)

**Step 6: Cross-reference with profiling (if available)**

If `{use_case}_series_profile` exists, join with best model results:
```sql
SELECT
    b.model,
    p.forecastability_class,
    COUNT(*) AS series_count,
    AVG(b.avg_metric) AS avg_metric
FROM {catalog}.{schema}.{use_case}_best_models b
LEFT JOIN {catalog}.{schema}.{use_case}_series_profile p
  ON b.unique_id = p.unique_id
GROUP BY b.model, p.forecastability_class
ORDER BY p.forecastability_class, avg_metric
```

**Step 7: Suggest next steps**

Based on results:
- If many series have high error: suggest re-running with different models or checking data quality
- If foundation models won most series: suggest using them for scoring
- If local and foundation results are similar: suggest using local models for cost efficiency
- If `low_signal` series still have high error: confirm they should be excluded from business decisions

#### Outputs

**Table**: `<catalog>.<schema>.{use_case}_best_models`

| Column | Type | Description |
|--------|------|-------------|
| `unique_id` | STRING | Series identifier |
| `model` | STRING | Best model name |
| `avg_metric` | DOUBLE | Average backtest metric |
| `ds` | ARRAY<TIMESTAMP> | Forecast dates |
| `y` | ARRAY<DOUBLE> | Forecast values |

**Table**: `<catalog>.<schema>.{use_case}_evaluation_summary`

| Column | Type | Description |
|--------|------|-------------|
| `model` | STRING | Model name |
| `wins_count` | INT | Number of series where this model was best |
| `wins_pct` | DOUBLE | Percentage of total series |
| `avg_smape` | DOUBLE | Average sMAPE across all series |
| `avg_wape` | DOUBLE | Average WAPE across all series |

---

## 4. File Layout

### 4.1 Final State of the Skill Folder

```
skills/databricks-skills/many-model-forecasting/
├── SKILL.md                                      # REWRITTEN: single 5-skill pipeline
├── 1-prep-and-clean-data.md                      # NEW (supersedes 1-explore-data.md)
├── 2-profile-and-classify-series.md              # NEW
├── 3-provision-forecasting-resources.md          # NEW (supersedes 2-setup-the-mmf-cluster.md)
├── 4-execute-mmf-forecast.md                     # NEW (supersedes 3-run-mmf.md)
├── 5-post-process-and-evaluate.md                # NEW
├── mmf_local_notebook_template.ipynb             # UNCHANGED
├── mmf_gpu_notebook_template.ipynb               # MODIFIED (single-model-per-session)
└── mmf_profiling_notebook_template.ipynb          # NEW
```

### 4.2 Retired Files (Removed)

```
1-explore-data.md          # Superseded by 1-prep-and-clean-data.md
2-setup-the-mmf-cluster.md # Superseded by 3-provision-forecasting-resources.md
3-run-mmf.md               # Superseded by 4-execute-mmf-forecast.md
```

### 4.3 Changes to Supporting Files

| File | Change | Scope |
|------|--------|-------|
| `SKILL.md` | Rewrite: remove old 3-skill workflow, define 5-skill pipeline | Full rewrite of workflow section |
| `install.py` | Replace `SKILL_FILES` list entries: old → new | ~8 line changes |
| `CLAUDE.md` (root) | Replace skill references in `<!-- mmf-dev-kit:skills -->` block | ~6 line changes |

---

## 5. Data Flow

```
                  ┌─────────────────────────────────────┐
                  │     Unity Catalog: <catalog>.<schema> │
                  └──────────────┬──────────────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │  Raw time series table   │
                    │  (user's source data)    │
                    └────────────┬────────────┘
                                 │
                  ┌──────────────▼──────────────┐
                  │  Skill 1: prep_and_clean    │
                  │  Step 0: collect use_case   │
                  │  (incorporates explore-data │
                  │   Steps 1-7 + cleaning)     │
                  └──────────┬────────┬─────────┘
                             │        │
         ┌───────────────────▼──┐  ┌──▼──────────────────────┐
         │{use_case}_train_data │  │{use_case}_cleaning_report│
         └──────┬───────────────┘  └──────────────────────────┘
                │
          ┌─────▼───────────────┐
          │ Skill 2: profile    │
          │ (entirely new)      │
          └──────┬────┬─────────┘
                 │    │
     ┌───────────▼┐  ┌▼───────────────────┐
     │{use_case}  │  │ Model type needs    │
     │_series     │  │ (local/GPU/found.)  │
     │_profile    │  │                     │
     └─────┬──────┘  └────────┬───────────┘
           │                  │
           │    ┌─────────────▼──────────────┐
           │    │ Skill 3: provision         │
           │    │ (incorporates setup-cluster│
           │    │  Steps 1-5 + reuse + UC)  │
           │    └─────────────┬──────────────┘
           │                  │
           │         Cluster config(s)
           │                  │
     ┌─────▼──────────────────▼──────────┐
     │ Skill 4: execute                   │
     │ (incorporates run-mmf             │
     │  Steps 1-7 + validation)          │
     └──────┬────────┬──────────┬────────┘
            │        │          │
   ┌──────────────▼──┐ ┌──▼──────────┐ ┌▼───────────┐
   │{use_case}       │ │{use_case}   │ │{use_case}   │
   │_evaluation      │ │_scoring     │ │_run         │
   │_output          │ │_output      │ │_metadata    │
   └────┬────────────┘ └────┬────────┘ └────────────┘
        │                   │
     ┌──▼───────────────────▼────────────┐
     │ Skill 5: post_process             │
     │ (entirely new — automates         │
     │  post-evaluation-analysis.ipynb)  │
     └──────┬────────────────┬───────────┘
            │                │
   ┌────────▼──────────┐ ┌──▼──────────────────┐
   │{use_case}         │ │{use_case}            │
   │_best_models       │ │_evaluation_summary   │
   └───────────────────┘ └─────────────────────┘
```

---

## 6. MCP Tool Dependencies

| MCP Tool | Used By Skills | Purpose |
|----------|---------------|---------|
| `connect_to_workspace` | 1, 2, 3, 4, 5 | Establish workspace connection |
| `execute_parameterized_sql` | 1, 2, 4, 5 | Run SQL queries against UC tables |
| `create_job` | 2, 4 | Create Databricks Workflow jobs |
| `run_job` | 2, 4 | Start job runs |
| `get_job_run` | 2, 4 | Monitor job status |
| `list_clusters` | 3 | Discover existing clusters |
| `get_cluster` | 3 | Check cluster status |
| `start_cluster` | 3 | Restart terminated clusters |
| `create_cluster` | 3 | Create new clusters |
| `upload_notebook` | 2, 4 | Upload generated notebooks |
| `AskUserQuestion` | All | User confirmation checkpoints |

---

## 7. Notebook Template Specifications

### 7.1 `mmf_profiling_notebook_template.ipynb` (new)

**Purpose**: Compute statistical properties for each series using pandas UDFs on Spark.

**Structure**:
1. Title cell (markdown)
2. `%pip install statsmodels scipy` + `%restart_python`
3. Parameters cell (catalog, schema, use_case, train_table, freq, prediction_length — all with `{placeholder}`)
4. Load data from `{use_case}_train_data`
5. Define profiling UDF (computes all 9 statistical properties per series)
6. Apply UDF via `groupby().applyInPandas()`
7. Classify into high_confidence / low_signal
8. Recommend models based on classification rules
9. Write `{use_case}_series_profile` to Delta table

**Compute**: Runs on the CPU cluster (profiling is CPU-bound).

### 7.2 `mmf_gpu_notebook_template.ipynb` (MODIFIED — single-model-per-session)

> **CUDA memory constraint.** PyTorch GPU models allocate CUDA memory that cannot be reliably freed within the same Python process. Running multiple GPU models sequentially in one session causes out-of-memory errors when the second model loads while the first model's CUDA tensors are still resident. The official MMF example notebooks (`examples/monthly/global_monthly.ipynb`, `examples/daily/global_daily.ipynb`) solve this by calling `dbutils.notebook.run()` per model — each invocation gets a fresh kernel with clean CUDA state.

The GPU template must be updated to accept a **single model** per invocation, matching the example pattern.

**Changes from current template:**

| Aspect | Current (broken) | Updated |
|--------|-----------------|---------|
| Cell 3: model parameter | `active_models = {active_models}` (list) | `model = "{model}"` (single string) |
| Cell 3: run_id | Not present | `run_id = "{run_id}"` |
| Cell 3: num_nodes | Not present | `num_nodes = {num_nodes}` |
| Cell 4: `run_forecast()` | `active_models=active_models` | `active_models=[model]` |
| Cell 4: `run_forecast()` | No `run_id` | `run_id=run_id` |
| Cell 4: `run_forecast()` | No `model_output` | `model_output=catalog + "." + schema` |
| Cell 4: `run_forecast()` | No `num_nodes` | `num_nodes=num_nodes` |
| Cell 1: pip extras | `{pip_extras}` (unchanged) | `{pip_extras}` (unchanged — agent determines per model) |

**Updated Cell 3:**
```python
catalog = "{catalog}"
schema = "{schema}"
train_table = "{train_table}"
freq = "{freq}"
prediction_length = {prediction_length}
backtest_length = {backtest_length}
stride = {stride}
metric = "{metric}"
model = "{model}"
run_id = "{run_id}"
num_nodes = {num_nodes}
group_id = "{group_id}"
date_col = "{date_col}"
target = "{target}"
```

**Updated Cell 4:**
```python
from mmf_sa import run_forecast

user = spark.sql("SELECT current_user() AS user").collect()[0]["user"]

run_forecast(
    spark=spark,
    train_data=catalog + "." + schema + "." + train_table,
    scoring_data=catalog + "." + schema + "." + train_table,
    scoring_output=catalog + "." + schema + ".{use_case}_scoring_output",
    evaluation_output=catalog + "." + schema + ".{use_case}_evaluation_output",
    model_output=catalog + "." + schema,
    group_id=group_id,
    date_col=date_col,
    target=target,
    freq=freq,
    prediction_length=prediction_length,
    backtest_length=backtest_length,
    stride=stride,
    metric=metric,
    train_predict_ratio=1,
    data_quality_check=True,
    resample=False,
    active_models=[model],
    experiment_path="/Users/" + user + "/mmf/{use_case}",
    use_case_name="{use_case}",
    run_id=run_id,
    accelerator="gpu",
    num_nodes=num_nodes,
)
```

**Compute**: Each invocation runs on the GPU cluster. A shared `run_id` groups results across all model invocations.

### 7.3 `mmf_local_notebook_template.ipynb` (unchanged)

Local models (statsforecast, Prophet) run on CPU and do not have CUDA memory constraints. The local template continues to pass the full `active_models` list in a single `run_forecast()` call.

---

## 8. Interaction Patterns

### 8.1 Fully Autonomous Mode

```
Agent: "I'll run the full MMF pipeline. What's a short name for this use case?"
User: "rossmann"
Agent: "Got it — all assets will be prefixed with 'rossmann'. Starting with data preparation..."
  → /prep-and-clean-data mmf_catalog sales   (use_case=rossmann)
  → /profile-and-classify-series mmf_catalog sales
  → /provision-forecasting-resources
  → /execute-mmf-forecast mmf_catalog sales
  → /post-process-and-evaluate mmf_catalog sales
Agent: "Pipeline complete. Tables: rossmann_train_data, rossmann_best_models, rossmann_evaluation_summary"
```

### 8.2 Interactive Mode (with checkpoints)

Each skill has `AskUserQuestion` checkpoints where the agent pauses:

1. **Skill 1**: "I found 3 candidate tables. Which should I use?" (from original Step 5)
2. **Skill 2**: "Based on profiling, I recommend these model families: [local, foundation]. Proceed?"
3. **Skill 3**: "I found an existing GPU cluster. Reuse it?"
4. **Skill 4**: "Here are the forecast parameters. Confirm to submit the job?" (from original Step 1)
5. **Skill 5**: "Results ready. Would you like me to drill into specific series?"

### 8.3 Partial Execution

Skills can be run independently. For example:
- Run only Skill 5 against existing evaluation results
- Run only Skill 2 to profile data without forecasting
- Run Skills 1 → 3 → 4 (skip profiling — Skill 3 falls back to asking the user, Skill 4 uses default models)

---

## 9. Error Handling

| Error Scenario | Handling | Recovery |
|---------------|----------|----------|
| Table not found | Report to user, suggest alternatives | `AskUserQuestion` for correction |
| No forecastable series after profiling | Warn user, suggest lowering thresholds | Adjustable classification thresholds |
| Cluster start fails | Report error, provide manual instructions | Fallback to ephemeral job clusters |
| Job run fails | Report task-level errors, suggest fixes | Re-run failed tasks only |
| Model not in registry | Validate against model names list before submission | Pre-flight validation in Skill 4 |
| Empty evaluation output | Check job logs, verify input data | Re-run with debug logging |
| CUDA OOM on GPU task | Should not occur with single-model-per-task pattern; if it does, suggest smaller GPU instance or reduce batch size | Select a larger GPU instance in Skill 3 |
| `{use_case}_series_profile` missing | Skills 3 and 4 gracefully fall back to manual mode | No error — just skip auto-detect |

---

## 10. Testing Strategy

### 10.1 Extend Existing Test Framework

| Test Artifact | Location | Purpose |
|-------------|----------|---------|
| `manifest.yaml` | Update `scorers.enabled` and `trace_expectations` | Add expectations for new skills |
| `ground_truth.yaml` | Add expected facts for new skills | Validate profiling outputs, model recommendations |
| Tier 1 tests | `.test/tests/tier1/test_profile_series.py` | DuckDB-mocked profiling tests |
| Tier 1 tests | `.test/tests/tier1/test_post_evaluate.py` | DuckDB-mocked evaluation tests |

### 10.2 Quality Gates

| Gate | Threshold | Applies To |
|------|-----------|-----------|
| `syntax_valid` | 100% | All generated SQL and Python |
| `pattern_adherence` | 90% | Template placeholder substitution |
| `no_hallucinated_apis` | 100% | Model names, MCP tool names |
| `execution_success` | 80% | End-to-end skill execution |
| `profiling_consistency` | 95% | Classification thresholds produce expected partitions |
