# Prep and Clean Data

Asks the user for catalog, schema, and use case name, discovers time series tables,
maps columns to the MMF schema (`unique_id`, `ds`, `y`),
asks the user how to impute missing data, generates an anomaly analysis report,
asks the user how to handle anomalies, and creates the `{use_case}_train_data` table
ready for forecasting.

## Steps

### Step 0: Collect use case name

Before any data exploration, ask the user:

> "Provide a short use case name (e.g., m4, rossmann, retail_sales).
> This will prefix all tables and assets created by the pipeline."

Validation rules:
- Lowercase alphanumeric + underscores only (must be valid as a Delta table name component)
- 1-30 characters
- Cannot start with a number

Store as `{use_case}` and propagate to all downstream steps. All table names in the pipeline use the pattern `{use_case}_<asset_name>` (e.g., `m4_train_data`, `rossmann_cleaning_report`).

### ⛔ STOP GATE — Step 0a: Ask for catalog and schema

**Do NOT assume catalog or schema values. Do NOT reuse values from prior runs. Always ask.**

Ask the user:

> "Which Unity Catalog catalog and schema should I use to discover your time series data?
>
> - Catalog: (e.g., main, ml_dev, my_catalog)
> - Schema: (e.g., default, forecasting, my_schema)
>
> I'll search for time series tables in this location."

**Do NOT proceed until the user provides both catalog and schema.**

Store as `{catalog}` and `{schema}`.

### Step 1: List tables

Execute SQL to discover tables in the given catalog and schema:

```sql
SHOW TABLES IN {catalog}.{schema}
```

Present the table list to the user.

### Step 2: Identify time series candidates

For each table, execute:

```sql
DESCRIBE TABLE {catalog}.{schema}.{table_name}
```

Identify time series candidates by checking column types:
- **Timestamp/date column (`ds`)**: columns with type `TIMESTAMP`, `DATE`, or `STRING` with date-like names (`ds`, `date`, `timestamp`, `time`, `datetime`)
- **Target numeric column (`y`)**: columns with type `DOUBLE`, `FLOAT`, `INT`, `BIGINT`, `DECIMAL` with names suggesting a metric (`y`, `value`, `target`, `amount`, `quantity`, `sales`, `count`)
- **Group identifier (`unique_id`)**: columns with type `STRING`, `INT`, or `BIGINT` that appear to be categorical identifiers (`unique_id`, `id`, `store_id`, `product_id`, `sku`, `series_id`)

A valid time series table MUST have all three: a date column, a numeric column, and a group column.

### Step 3: Profile candidates

For each candidate table, execute these queries:

**Row count:**
```sql
SELECT COUNT(*) AS count FROM {catalog}.{schema}.{table_name}
```

**Date range:**
```sql
SELECT MIN({ds_col}) AS min_ds, MAX({ds_col}) AS max_ds FROM {catalog}.{schema}.{table_name}
```

**Number of distinct groups:**
```sql
SELECT COUNT(DISTINCT {unique_id_col}) AS unique_count FROM {catalog}.{schema}.{table_name}
```

Present the profiling results to the user and propose which table to use and which columns map to `unique_id`, `ds`, and `y`.

### Step 3a: Detect source frequency

Measure the average gap between consecutive timestamps to determine the source data frequency:

```sql
SELECT ROUND(AVG(diff_days), 1) AS avg_gap, MIN(diff_days) AS min_gap, MAX(diff_days) AS max_gap
FROM (
  SELECT DATEDIFF(DAY, LAG({ds_col}) OVER (PARTITION BY {unique_id_col} ORDER BY {ds_col}), {ds_col}) AS diff_days
  FROM {catalog}.{schema}.{table_name}
) sub WHERE diff_days IS NOT NULL
```

**Interpretation:**
- `avg_gap` < 0.1 = **Hourly** (`freq=H`)
- `avg_gap` ~1 = **Daily** (`freq=D`)
- `avg_gap` ~7 = **Weekly** (`freq=W`)
- `avg_gap` ~28-31 = **Monthly** (`freq=M`)

Report the detected frequency to the user.

### Step 4: Validate with user

Ask the user to confirm:
- Source table selection
- Column mapping: `unique_id`, `ds`, `y`
- Any optional exogenous regressors to include

### Step 5: Create {use_case}_train_data

Create the training table with the MMF-required schema. The SQL depends on the detected (or user-specified) frequency to ensure dates are properly aligned:

**Hourly (`freq=H`) — no date transformation needed:**
```sql
CREATE OR REPLACE TABLE {catalog}.{schema}.{use_case}_train_data AS
SELECT
    CAST({unique_id_col} AS STRING) AS unique_id,
    CAST({ds_col} AS TIMESTAMP) AS ds,
    CAST({y_col} AS DOUBLE) AS y
FROM {catalog}.{schema}.{table_name}
WHERE {y_col} IS NOT NULL
```

**Daily (`freq=D`) — no date transformation needed:**
```sql
CREATE OR REPLACE TABLE {catalog}.{schema}.{use_case}_train_data AS
SELECT
    CAST({unique_id_col} AS STRING) AS unique_id,
    CAST({ds_col} AS DATE) AS ds,
    CAST({y_col} AS DOUBLE) AS y
FROM {catalog}.{schema}.{table_name}
WHERE {y_col} IS NOT NULL
```

**Weekly (`freq=W`) — align dates to Sunday (end of ISO week) and aggregate:**
```sql
CREATE OR REPLACE TABLE {catalog}.{schema}.{use_case}_train_data AS
SELECT
    CAST({unique_id_col} AS STRING) AS unique_id,
    CAST(DATE_TRUNC('week', {ds_col}) + INTERVAL 6 DAY AS DATE) AS ds,
    SUM(CAST({y_col} AS DOUBLE)) AS y
FROM {catalog}.{schema}.{table_name}
WHERE {y_col} IS NOT NULL
GROUP BY {unique_id_col}, DATE_TRUNC('week', {ds_col}) + INTERVAL 6 DAY
```

**Monthly (`freq=M`) — align dates to month-end and aggregate:**
```sql
CREATE OR REPLACE TABLE {catalog}.{schema}.{use_case}_train_data AS
SELECT
    CAST({unique_id_col} AS STRING) AS unique_id,
    CAST(LAST_DAY({ds_col}) AS DATE) AS ds,
    SUM(CAST({y_col} AS DOUBLE)) AS y
FROM {catalog}.{schema}.{table_name}
WHERE {y_col} IS NOT NULL
GROUP BY {unique_id_col}, LAST_DAY({ds_col})
```

If exogenous regressors were selected, include them as additional columns (use `SUM` for numeric regressors when aggregating).

Verify creation:
```sql
SELECT COUNT(*) AS count FROM {catalog}.{schema}.{use_case}_train_data
```

### Step 6: Missing Data Assessment & Imputation

Missing data in time series comes in three forms, all of which must be detected:
- **Explicit NULLs**: Rows exist but `y` is NULL.
- **Interior gaps**: Entire rows are absent between the first and last timestamp of a series.
- **Trailing gaps**: A series ends before the global max date. For example, if most series run through Dec 2024 but three end at Nov 2024, those three have a one-month trailing gap. This is critical to detect because MMF expects all series to share the same end date.

#### Step 6a: Generate date spine and detect all gaps

Use each series' own `min_ds` (series may legitimately start at different times) but the **global** `max_ds` across all series (so trailing gaps are detected):

```sql
WITH global_max AS (
  SELECT MAX(ds) AS max_ds
  FROM {catalog}.{schema}.{use_case}_train_data
),
series_bounds AS (
  SELECT unique_id, MIN(ds) AS min_ds
  FROM {catalog}.{schema}.{use_case}_train_data
  GROUP BY unique_id
),
date_spine AS (
  SELECT s.unique_id, EXPLODE(SEQUENCE(s.min_ds, g.max_ds, INTERVAL 1 DAY)) AS expected_ds
  FROM series_bounds s
  CROSS JOIN global_max g
),
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

Before presenting the summary, also identify series with **trailing gaps** — i.e., series whose last observed data point is earlier than the global maximum date:

```sql
WITH global_max AS (
  SELECT MAX(ds) AS max_ds FROM {catalog}.{schema}.{use_case}_train_data
),
series_last AS (
  SELECT unique_id, MAX(ds) AS last_ds
  FROM {catalog}.{schema}.{use_case}_train_data
  WHERE y IS NOT NULL
  GROUP BY unique_id
)
SELECT s.unique_id, s.last_ds, g.max_ds,
       DATEDIFF(g.max_ds, s.last_ds) AS trailing_gap_days
FROM series_last s CROSS JOIN global_max g
WHERE s.last_ds < g.max_ds
ORDER BY trailing_gap_days DESC
```

Report the trailing-gap series as part of the summary below.

#### ⛔ STOP GATE — Step 6b: Present summary and ask user for imputation strategy

**Do NOT proceed until the user chooses an imputation strategy.**

Present the missing data summary to the user and ask:

> "Missing data summary (including interior and trailing gaps):
> - {n_clean} series are complete (no gaps)
> - {n_low} series have < 5% missing → Suggest: linear interpolation (avg of neighbors)
> - {n_mid} series have 5-20% missing → Suggest: forward fill (last known value)
> - {n_high} series have > 20% missing → Suggest: exclude from forecasting
>
> Trailing gaps detected:
> - {n_trailing} series end before the global max date ({global_max_ds})
> - These series are missing the most recent data points and will show as NULL rows after the spine backfill.
> [list each trailing-gap series, its last observed date, and the gap size]
>
> How would you like to proceed?
> (a) Apply suggested strategy
> (b) Use a single strategy for all (interpolation / forward fill / fill with 0 / drop nulls)
> (c) Skip imputation — keep nulls as-is
> (d) Adjust the exclusion threshold (currently 20%)"

**WAIT for the user to respond. Do NOT apply any imputation until the user confirms their choice.**

#### Step 6c: Backfill the date spine and apply imputation

First, replace `{use_case}_train_data` with the complete spine (using global `max_ds`) so that implicit gaps and trailing gaps become explicit NULL rows:

```sql
CREATE OR REPLACE TABLE {catalog}.{schema}.{use_case}_train_data AS
WITH global_max AS (
  SELECT MAX(ds) AS max_ds FROM {catalog}.{schema}.{use_case}_train_data
),
series_bounds AS (
  SELECT unique_id, MIN(ds) AS min_ds
  FROM {catalog}.{schema}.{use_case}_train_data
  GROUP BY unique_id
),
date_spine AS (
  SELECT s.unique_id, EXPLODE(SEQUENCE(s.min_ds, g.max_ds, INTERVAL 1 DAY)) AS expected_ds
  FROM series_bounds s CROSS JOIN global_max g
)
SELECT sp.unique_id, sp.expected_ds AS ds, t.y
FROM date_spine sp
LEFT JOIN {catalog}.{schema}.{use_case}_train_data t
  ON sp.unique_id = t.unique_id AND sp.expected_ds = t.ds
```

The `SEQUENCE` interval must match the detected frequency (same as Step 6a).

Then apply the chosen imputation on the now-explicit NULLs:

<!-- BUG IDENTIFIED: 2026-03-26 — lourdes.martinez@databricks.com
     PROBLEM: The original linear interpolation expression uses LAG(y IGNORE NULLS),
     which is invalid Databricks SQL syntax. Databricks requires IGNORE NULLS to be
     placed OUTSIDE the function call (not inside the parentheses). The query fails
     with a syntax error at runtime.
     The same issue applies to LEAD(y IGNORE NULLS).

     PROPOSED FIX (⚠️ requires thorough testing — not yet validated in execution):
     Use LAST_VALUE / FIRST_VALUE with IGNORE NULLS outside the call, combined with
     a window frame to approximate the average of the previous and next known values.
     See corrected expression below.

     ORIGINAL (broken — kept for reference):
     - Linear interpolation: (LAG(y IGNORE NULLS) + LEAD(y IGNORE NULLS)) / 2
-->

- **Linear interpolation** *(fix proposal — ⚠️ requires thorough testing)*:
  ```sql
  (LAST_VALUE(y IGNORE NULLS) OVER (PARTITION BY unique_id ORDER BY ds
     ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
   +
   FIRST_VALUE(y IGNORE NULLS) OVER (PARTITION BY unique_id ORDER BY ds
     ROWS BETWEEN 1 FOLLOWING AND UNBOUNDED FOLLOWING)
  ) / 2.0
  ```
- **Forward fill**: `LAST_VALUE(y IGNORE NULLS) OVER (PARTITION BY unique_id ORDER BY ds ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)`
- **Fill with 0**: `COALESCE(y, 0)` — appropriate for count/demand data where absence means zero activity
- **Exclusion**: Remove series exceeding the threshold from `{use_case}_train_data`

Log the count of imputed values per series and excluded series for the cleaning report.

### Step 7: Anomaly Detection & Analysis Report

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

#### Step 7a: Generate anomaly analysis report

**Always generate and present an anomaly report before asking the user how to handle anomalies.**

Present the following report to the user:

```
═══════════════════════════════════════════════════════
          ANOMALY ANALYSIS REPORT
          Use case: {use_case}
═══════════════════════════════════════════════════════

OVERALL SUMMARY
  Total series analyzed:     {total_series}
  Series with anomalies:     {n_affected} ({affected_pct}%)
  Series without anomalies:  {n_clean} ({clean_pct}%)
  Total anomalous points:    {total_anomalies} out of {total_points} ({overall_anomaly_pct}%)

SEVERITY DISTRIBUTION
  Low    (< 1% anomalous):   {n_low_severity} series
  Medium (1-5% anomalous):   {n_med_severity} series
  High   (> 5% anomalous):   {n_high_severity} series

TOP 10 MOST AFFECTED SERIES
  ┌──────────────┬─────────┬───────────┬────────────┬────────────┬──────────────┐
  │ unique_id    │ total   │ anomalies │ anomaly_%  │ min_anom   │ max_anom     │
  ├──────────────┼─────────┼───────────┼────────────┼────────────┼──────────────┤
  │ {id_1}       │ {n_1}   │ {a_1}     │ {p_1}%     │ {min_1}    │ {max_1}      │
  │ ...          │ ...     │ ...       │ ...        │ ...        │ ...          │
  └──────────────┴─────────┴───────────┴────────────┴────────────┴──────────────┘

IQR BOUNDS (1.5× multiplier)
  Typical lower bound range: {min_lower} to {max_lower}
  Typical upper bound range: {min_upper} to {max_upper}

═══════════════════════════════════════════════════════
```

#### ⛔ STOP GATE — Step 7b: Ask user how to handle anomalies

**Do NOT proceed until the user chooses an anomaly handling strategy.**

Present the anomaly report above and ask the user:

> "Based on the anomaly analysis report above:
> - {n_clean} series have no anomalies
> - {n_affected} series have outliers ({total_anomalies} points total, {anomaly_pct}% overall)
> - Default capping range: [Q1 - 1.5×IQR, Q3 + 1.5×IQR]
>
> How would you like to proceed?
> (a) Cap at 1.5×IQR (default — moderate, removes typical outliers)
> (b) Cap at 3.0×IQR (conservative — only removes extreme outliers)
> (c) Custom multiplier: enter a value (e.g., 2.0)
> (d) Skip anomaly capping — keep all values as-is"

**WAIT for the user to respond. Do NOT apply any capping until the user confirms their choice.**

#### Step 7c: Apply capping (if chosen)

If the user chooses (a), (b), or (c), apply capping using the chosen `{iqr_multiplier}`:

<!-- BUG IDENTIFIED: 2026-03-26 — lourdes.martinez@databricks.com
     PROBLEM: The original capping query uses UPDATE ... FROM ... syntax, which is
     NOT supported in Databricks SQL. Databricks does not allow a FROM clause in
     UPDATE statements — it raises a parse error at runtime.

     PROPOSED FIX (⚠️ requires thorough testing — validated once in execution 2026-03-26):
     Replace with CREATE OR REPLACE TABLE ... AS SELECT ... JOIN pattern.
     This rewrites the full table with capped values in a single scan, which is
     also more performant on Delta than row-by-row UPDATE.

     ORIGINAL (broken — kept for reference):
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
-->

```sql
-- FIX PROPOSAL (⚠️ requires thorough testing):
CREATE OR REPLACE TABLE {catalog}.{schema}.{use_case}_train_data AS
SELECT
  t.unique_id,
  t.ds,
  CASE
    WHEN t.y < s.q1 - {iqr_multiplier} * s.iqr THEN s.q1 - {iqr_multiplier} * s.iqr
    WHEN t.y > s.q3 + {iqr_multiplier} * s.iqr THEN s.q3 + {iqr_multiplier} * s.iqr
    ELSE t.y
  END AS y
FROM {catalog}.{schema}.{use_case}_train_data t
JOIN (
  SELECT unique_id,
         PERCENTILE(y, 0.25) AS q1,
         PERCENTILE(y, 0.75) AS q3,
         PERCENTILE(y, 0.75) - PERCENTILE(y, 0.25) AS iqr
  FROM {catalog}.{schema}.{use_case}_train_data
  GROUP BY unique_id
) s ON t.unique_id = s.unique_id
```

Log the count of capped values per series for the cleaning report.

### Step 8: Create {use_case}_cleaning_report

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

### Step 9: Generate reproducibility notebook

After all interactive decisions have been made, generate a self-contained notebook that replays the entire data preparation pipeline.

**CRITICAL: Do NOT execute this code inline. Generate the notebook from the template, upload it to the workspace. Never run data preparation pipeline code directly in the conversation.**

**CRITICAL: Use the template at `notebooks/mmf_prep_notebook_template.ipynb` (in this skill folder). Copy it verbatim, only replacing the `{placeholder}` tokens with actual values. Do NOT add, remove, or modify any other code.**

Replace these placeholders:
- `{catalog}` → user's catalog
- `{schema}` → user's schema
- `{use_case}` → use case name
- `{source_table}` → selected source table name (table name only, not fully qualified)
- `{unique_id_col}` → source column mapped to `unique_id`
- `{ds_col}` → source column mapped to `ds`
- `{y_col}` → source column mapped to `y`
- `{freq}` → detected or user-specified frequency (`H`, `D`, `W`, `M`)
- `{imputation_method}` → chosen imputation strategy (`interpolation`, `forward_fill`, `fill_zero`, `none`)
- `{exclusion_threshold}` → exclusion threshold as integer (e.g., `20`)
- `{iqr_multiplier}` → IQR multiplier as float (e.g., `1.5`; `0` = skip capping)

#### ⛔ STOP GATE — Confirm before uploading notebook

Before uploading, present a summary and ask the user:

> "I am about to upload the data preparation notebook to the workspace:
> - Path: `{home_path}/mmf-skills-test/notebooks/{use_case}/01_prep_data`
> - Parameters: catalog={catalog}, schema={schema}, use_case={use_case}, freq={freq}, imputation={imputation_method}, iqr_multiplier={iqr_multiplier}
>
> Shall I proceed?"

**Do NOT upload until the user confirms.**

Upload the generated notebook to the Databricks workspace at:
```
{home_path}/mmf-skills-test/notebooks/{use_case}/01_prep_data
```

Where `{home_path}` = `/Workspace/Users/{current_user_email}`.

### ⛔ STOP GATE — Step 10: Confirm before proceeding to next step

Present a summary of what was done and ask the user:

> "✅ Data preparation complete for use case '{use_case}'.
>
> Summary:
> - Training table: {catalog}.{schema}.{use_case}_train_data
> - Series count: {n_series}
> - Date range: {min_date} → {max_date}
> - Frequency: {freq}
> - Imputation: {imputation_summary}
> - Anomalies: {anomaly_summary}
> - Cleaning report: {catalog}.{schema}.{use_case}_cleaning_report
> - Reproducibility notebook: notebooks/{use_case}/01_prep_data
>
> Would you like to proceed to the next step?
> (a) Run profiling & classification (optional — estimates series forecastability and recommends models)
> (b) Skip profiling and go directly to cluster provisioning & model selection
> (c) Stop here — I'll come back later"

**Do NOT proceed until the user responds.**

## Outputs

- A Delta table `<catalog>.<schema>.{use_case}_train_data` with columns `unique_id` (STRING), `ds` (DATE for D/W/M, TIMESTAMP for H), `y` (DOUBLE)
- A Delta table `<catalog>.<schema>.{use_case}_cleaning_report` with columns: `unique_id`, `original_count`, `final_count`, `missing_filled`, `imputation_method`, `anomalies_capped`, `iqr_multiplier`, `excluded`, `exclusion_reason`
- A reproducibility notebook uploaded to the workspace at `notebooks/{use_case}/01_prep_data`
- A summary: number of series, date range, detected frequency, cleaning actions taken
