# Prep and Clean Data

**Slash command:** `/prep-and-clean-data`

Asks the user for catalog, schema, use case name, and a **forecast problem brief** (`{forecast_problem_brief}`), connects to a Databricks workspace,
discovers time series tables, maps columns to the MMF schema (`unique_id`, `ds`, `y`),
asks the user how to impute missing data, generates an anomaly analysis report,
asks the user how to handle anomalies, and creates the `{use_case}_train_data` table
ready for forecasting.

## Steps

### Step 0: Collect use case name

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

### ⛔ STOP GATE — Step 0a: Ask for catalog and schema

**Do NOT assume catalog or schema values. Do NOT reuse values from prior runs. Always ask.**

```
AskUserQuestion:
  "Which Unity Catalog catalog and schema should I use to discover your time series data?

   • Catalog: (e.g., main, ml_dev, my_catalog)
   • Schema:  (e.g., default, forecasting, my_schema)

   I'll search for time series tables in this location."
  Options: [free text — user provides catalog and schema]
```

**Do NOT proceed until the user provides both catalog and schema.**

Store as `{catalog}` and `{schema}`.

### ⛔ STOP GATE — Step 0b: Forecast problem brief

**Do NOT connect to the workspace or run discovery SQL until the user has provided a minimal forecast problem brief (or one clarifying round).**

```
AskUserQuestion:
  "What use case are we solving?

   In a few sentences, describe the forecasting problem — what are you trying
   to predict and why? For example:

     'We forecast weekly unit sales per SKU×store to drive replenishment
      orders. Demand is highly seasonal with many slow-moving items.
      We need a 13-week horizon. No external regressors — just historical sales.'

   Things that help me tailor every downstream decision:
   • What the target variable (y) represents
   • How the forecast will be used (operations, finance, capacity, etc.)
   • Rough horizon you care about
   • Whether your series are smooth, seasonal, intermittent/sparse, or unknown
   • Univariate series vs. exogenous / covariate

   Reply in free text; I'll condense to a short brief for downstream steps."
  Options: [free text]
```

Store a normalized **3–6 line** summary as `{forecast_problem_brief}`. Carry it in the conversation through all downstream skills (reconfirm in Skill 2 if context is missing).

#### Optional research and deep documentation

Any **web search**, **Databricks documentation**, or **extended reasoning** beyond the scripted SQL in this skill is optional and must be **scoped** to `{forecast_problem_brief}` (domain, meaning of `y`, intermittency, horizon, exogenous intent). Do **not** run broad, generic "time series cleaning" research without tying conclusions to the brief. If the brief is too vague to scope research, ask **one clarifying question** before searching. When suggesting imputation, fill-zero vs interpolation, IQR capping, or exogenous regressors, **cite the brief** in one line (e.g. "Given intermittent retail demand …").

### Step 1: Connect to workspace

Use `connect_to_workspace` to connect to the user's Databricks workspace.

### Step 2: List tables

Run SQL to discover tables in the given catalog and schema:

```sql
SHOW TABLES IN {catalog}.{schema}
```

Present the table list to the user.

### Step 3: Identify time series candidates

For each table, run:

```sql
DESCRIBE TABLE {catalog}.{schema}.{table_name}
```

Identify time series candidates by checking column types:

- **Timestamp/date column (`ds`)**: columns with type `TIMESTAMP`, `DATE`, or `STRING` with date-like names (`ds`, `date`, `timestamp`, `time`, `datetime`)
- **Target numeric column (`y`)**: columns with type `DOUBLE`, `FLOAT`, `INT`, `BIGINT`, `DECIMAL` with names suggesting a metric (`y`, `value`, `target`, `amount`, `quantity`, `sales`, `count`)
- **Group identifier (`unique_id`)**: columns with type `STRING`, `INT`, or `BIGINT` that appear to be categorical identifiers (`unique_id`, `id`, `store_id`, `product_id`, `sku`, `series_id`)

A valid time series table MUST have all three: a date column, a numeric column, and a group column.

### Step 4: Profile candidates

For each candidate table, run these queries:

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

### Step 4a: Detect source frequency

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

### ⛔ STOP GATE — Step 5: Validate with user

**Do NOT proceed until the user explicitly confirms the table and column mapping.**

```
AskUserQuestion:
  "Based on the profiling above, here is my proposed setup:

   • Forecast brief: {forecast_problem_brief} (one-line check: does this still match your goal?)
   • Source table:  {catalog}.{schema}.{table_name}
   • unique_id:     {unique_id_col}   (series identifier)
   • ds:            {ds_col}          (timestamp/date column)
   • y:             {y_col}           (target variable to forecast)
   • Frequency:     {detected_freq}

   ⚠️  The target variable (y) determines what will be forecasted.
   Please confirm or correct:
   (a) Confirm — use {y_col} as the target variable and proceed
   (b) Change target variable — specify a different column name
   (c) Change table — specify a different table to use
   (d) Change any column mapping — specify corrections"
```

**WAIT for the user to respond. Do NOT create any tables or run any further queries until the user confirms or corrects the mapping.**

If the user selects (b), (c), or (d), update the mapping accordingly and re-present this confirmation prompt before proceeding.

### ⛔ STOP GATE — Step 5a: Ask about exogenous regressors

**Only ask this after the user has confirmed the column mapping above. Ask one question at a time.**

```
AskUserQuestion:
  "Does your source table have any exogenous regressor columns you want to
   include alongside the target variable (y)?

   (1) Yes — I have additional columns to include
   (2) No — run in univariate mode (no exogenous regressors)"
```

**WAIT for the user to respond.**

If the user selects (2), set all regressor lists to `[]` and skip to Step 6.

If the user selects (1), ask which types they have:

```
AskUserQuestion:
  "Which types of regressors do you have? Select all that apply:

   (a) Static features — constant per series / group
       (e.g., store_state, dept_id, product_category)
   (b) Dynamic historical — past-only values, NOT needed at forecast time
       (e.g., lagged sales, rolling averages, past event flags)
   (c) Dynamic future — values known for BOTH past AND future dates
       (e.g., planned price, temperature forecast, promo flag, holiday indicator)

   Which types? (e.g., 'a and c', or just 'c')"
```

**WAIT for the user to respond.**

For each type the user selected, ask a separate follow-up question to collect the column names:

**If static features selected:**
```
AskUserQuestion:
  "List the static feature column names (constant per series):
   e.g., store_state, dept_id"
```

**WAIT for the user to respond.** Store as `{static_features}`.

**If dynamic historical selected:**
```
AskUserQuestion:
  "List the dynamic historical column names (past-only, not needed at forecast time).
   Separate numerical and categorical:
   • Numerical (e.g., lagged_sales, rolling_avg):
   • Categorical (e.g., past_event_type):"
```

**WAIT for the user to respond.** Store as `{dynamic_historical_numerical}` and `{dynamic_historical_categorical}`.

**If dynamic future selected:**
```
AskUserQuestion:
  "List the dynamic future column names (known for both past AND future dates).
   Separate numerical and categorical:
   • Numerical (e.g., planned_price, temperature_forecast):
   • Categorical (e.g., promo, day_of_week, holiday, open_closed):

   ⚠️  You MUST have known future values for every series × every future date
   in the forecast horizon. I will create a scoring table from this data."
```

**WAIT for the user to respond.** Store as `{dynamic_future_numerical}` and `{dynamic_future_categorical}`.

### ⛔ STOP GATE — Step 5b: Ask where future regressor values come from

**Only ask this if the user selected dynamic future regressors above.**

```
AskUserQuestion:
  "Where do the future values for {dynamic_future_columns} come from?

   (a) Same source table — rows beyond the last training date already contain
       the future regressor values (e.g., a calendar or schedule table that
       extends into the future)
   (b) Separate table — provide the table name containing future-dated rows
       with the regressor columns"
```

**WAIT for the user to respond.**

Store the answer as `{future_regressor_source}` (`same_table` or the name of the separate table). If the user selects (b), ask for the table name in a follow-up question before proceeding.

### Step 6: Create {use_case}_train_data

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

If exogenous regressors were selected, include them as additional columns (use `SUM` for numeric regressors when aggregating, and `FIRST` or `MAX` for categorical regressors when aggregating).

Verify creation:

```sql
SELECT COUNT(*) AS count FROM {catalog}.{schema}.{use_case}_train_data
```

### Step 6a: Create {use_case}_scoring_data (if dynamic future regressors)

**Only run this step if the user selected dynamic future regressors (`dynamic_future_numerical` or `dynamic_future_categorical`) in Step 5.**

The scoring table must contain one row per `unique_id` × future `ds` for each date in the forecast horizon, with all dynamic future regressor columns populated. During scoring, `run_forecast` unions this table with `train_data` via `unionByName(score_df, allowMissingColumns=True)` — columns present in `train_data` but absent from the scoring table (such as `y`) are automatically filled with NULL.

**Required schema — must match the Rossmann example pattern** (see [local_univariate_external_regressors_daily.ipynb](https://github.com/databricks-industry-solutions/many-model-forecasting/blob/main/examples/external_regressors/local_univariate_external_regressors_daily.ipynb)):

| Column | Type | Description |
|--------|------|-------------|
| `unique_id` | STRING | Series identifier (same column as in `train_data`) |
| `ds` | DATE or TIMESTAMP | Future dates only — dates beyond the last training date |
| Each `dynamic_future_numerical` column | DOUBLE / FLOAT / INT | Known future numerical regressor values |
| Each `dynamic_future_categorical` column | STRING / INT | Known future categorical regressor values |

**Do NOT include the target column (`y`).** The `allowMissingColumns=True` union fills it as NULL automatically. This matches the Rossmann example where `rossmann_daily_test` has `Store`, `Date`, and the regressor columns but no `Sales`.

**Do NOT include `static_features` or `dynamic_historical_*` columns.** These are either constant per series (extracted from training data by the model) or past-only (not relevant for future dates). The `allowMissingColumns=True` union handles them.

#### Source: same table (option a)

If the future regressor values come from the **same source table** (rows beyond the training period's end date):

```sql
CREATE OR REPLACE TABLE {catalog}.{schema}.{use_case}_scoring_data AS
SELECT
    CAST({unique_id_col} AS STRING) AS unique_id,
    CAST({ds_col} AS DATE) AS ds,
    {dynamic_future_columns}
FROM {catalog}.{schema}.{table_name}
WHERE {ds_col} > (SELECT MAX(ds) FROM {catalog}.{schema}.{use_case}_train_data)
```

Adjust the `CAST({ds_col} ...)` and date filtering to match the detected frequency (TIMESTAMP for hourly, DATE for daily/weekly/monthly). For weekly/monthly frequencies, apply the same alignment logic as Step 6 (e.g., `DATE_TRUNC('week', ...) + INTERVAL 6 DAY` for weekly, `LAST_DAY(...)` for monthly) and aggregate regressors accordingly.

#### Source: separate table (option b)

If the future regressor values come from a **separate table**:

```sql
CREATE OR REPLACE TABLE {catalog}.{schema}.{use_case}_scoring_data AS
SELECT
    CAST({future_unique_id_col} AS STRING) AS unique_id,
    CAST({future_ds_col} AS DATE) AS ds,
    {dynamic_future_columns}
FROM {catalog}.{schema}.{future_regressor_table}
WHERE {future_ds_col} > (SELECT MAX(ds) FROM {catalog}.{schema}.{use_case}_train_data)
```

Ask the user to confirm the column mapping for `unique_id` and `ds` in the separate table if the column names differ from the source training table.

#### Validation

After creating the scoring table, verify coverage:

```sql
-- Check row count and date range
SELECT
    COUNT(*) AS scoring_rows,
    COUNT(DISTINCT unique_id) AS n_series,
    MIN(ds) AS min_future_ds,
    MAX(ds) AS max_future_ds
FROM {catalog}.{schema}.{use_case}_scoring_data
```

```sql
-- Check that every training series has scoring rows
SELECT COUNT(*) AS series_missing_scoring
FROM (
    SELECT DISTINCT unique_id FROM {catalog}.{schema}.{use_case}_train_data
    EXCEPT
    SELECT DISTINCT unique_id FROM {catalog}.{schema}.{use_case}_scoring_data
)
```

```sql
-- Check for NULL values in dynamic future regressor columns
SELECT
    {for each col in dynamic_future_columns:
    SUM(CASE WHEN {col} IS NULL THEN 1 ELSE 0 END) AS {col}_nulls,}
    COUNT(*) AS total_rows
FROM {catalog}.{schema}.{use_case}_scoring_data
```

**If `series_missing_scoring > 0`**, warn the user:
> "⚠️ {series_missing_scoring} series in training data have no future regressor values in the scoring table. These series will be dropped from scoring. Would you like to (a) proceed anyway, (b) exclude those series from training too, or (c) provide a corrected scoring source?"

#### ⛔ STOP GATE — Scoring data NULL handling

**If any dynamic future columns have NULLs, always ask the user how to handle them. Do NOT proceed until the user confirms.**

Present the NULL summary and ask for a strategy:

```
AskUserQuestion:
  "The scoring table has NULL values in some dynamic future regressor columns:

   {for each column with nulls:
   • {col}: {n_nulls} NULLs out of {total_rows} rows ({null_pct}%)}

   NULL regressor values will cause errors or silently drop affected series
   during forecasting. How would you like to fill them?

   (a) Forward fill — carry the last known value from training data forward
       (good for slowly changing features like price or staffing level)
   (b) Fill with a specific value — enter a default per column
       (good for binary flags like promo=0, open=1)
   (c) Fill with the column's mode (most frequent value)
       (good for categorical columns like day_of_week or holiday)
   (d) Fill with the column's mean (numerical columns only)
       (good for continuous regressors like temperature)
   (e) Drop rows with NULLs — remove incomplete future dates
   (f) Proceed as-is — keep NULLs (⚠️ may cause pipeline errors)"
```

**WAIT for the user to respond. Do NOT apply any fill strategy until the user confirms.**

Apply the chosen strategy:

- **(a) Forward fill**: For each series, carry the last known value from the training table into the scoring rows:
  ```sql
  -- For each column with NULLs, get the last known value per series from train_data
  UPDATE {catalog}.{schema}.{use_case}_scoring_data s
  SET {col} = (
      SELECT {col}
      FROM {catalog}.{schema}.{use_case}_train_data t
      WHERE t.unique_id = s.unique_id
        AND t.{col} IS NOT NULL
      ORDER BY t.ds DESC
      LIMIT 1
  )
  WHERE s.{col} IS NULL
  ```
- **(b) Fill with specific value**: Ask the user for a default value per column, then:
  ```sql
  UPDATE {catalog}.{schema}.{use_case}_scoring_data
  SET {col} = {user_specified_default}
  WHERE {col} IS NULL
  ```
- **(c) Fill with mode**:
  ```sql
  UPDATE {catalog}.{schema}.{use_case}_scoring_data
  SET {col} = (
      SELECT {col} FROM {catalog}.{schema}.{use_case}_train_data
      GROUP BY {col} ORDER BY COUNT(*) DESC LIMIT 1
  )
  WHERE {col} IS NULL
  ```
- **(d) Fill with mean** (numerical columns only):
  ```sql
  UPDATE {catalog}.{schema}.{use_case}_scoring_data
  SET {col} = (SELECT AVG({col}) FROM {catalog}.{schema}.{use_case}_train_data)
  WHERE {col} IS NULL
  ```
- **(e) Drop rows**: Remove incomplete rows from the scoring table:
  ```sql
  DELETE FROM {catalog}.{schema}.{use_case}_scoring_data
  WHERE {col_1} IS NULL OR {col_2} IS NULL OR ...
  ```
- **(f) Proceed as-is**: Skip — warn the user again that this may cause pipeline errors.

After applying the chosen strategy (unless option f), verify no NULLs remain:

```sql
SELECT
    {for each col in dynamic_future_columns:
    SUM(CASE WHEN {col} IS NULL THEN 1 ELSE 0 END) AS {col}_nulls,}
    COUNT(*) AS total_rows
FROM {catalog}.{schema}.{use_case}_scoring_data
```

If NULLs still remain (e.g., forward fill had no non-NULL training value), warn the user and suggest a fallback strategy.

Present the final scoring table summary to the user before continuing.

### Step 7: Missing Data Assessment & Imputation

Missing data in time series comes in three forms, all of which must be detected:

- **Explicit NULLs**: Rows exist but `y` is NULL.
- **Interior gaps**: Entire rows are absent between the first and last timestamp of a series.
- **Trailing gaps**: A series ends before the global max date. For example, if most series run through Dec 2024 but three end at Nov 2024, those three have a one-month trailing gap. This is critical to detect because MMF expects all series to share the same end date.

#### Step 7a: Generate date spine and detect all gaps

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

#### ⛔ STOP GATE — Step 7b: Present summary and ask user for imputation strategy

**Do NOT proceed until the user chooses an imputation strategy.**

```
AskUserQuestion:
  "Missing data summary (including interior and trailing gaps):
   - {n_clean} series are complete (no gaps)
   - {n_low} series have < 5% missing → Suggest: linear interpolation (avg of neighbors)
   - {n_mid} series have 5-20% missing → Suggest: forward fill (last known value)
   - {n_high} series have > 20% missing → Suggest: exclude from forecasting

   Trailing gaps detected:
   - {n_trailing} series end before the global max date ({global_max_ds})
   - These series are missing the most recent data points and will show as
     NULL rows after the spine backfill. They are included in the counts above.
   [list each trailing-gap series, its last observed date, and the gap size]

   How would you like to proceed?
   (a) Apply suggested strategy
   (b) Use a single strategy for all (interpolation / forward fill / fill with 0 / drop nulls)
   (c) Skip imputation — keep nulls as-is
   (d) Adjust the exclusion threshold (currently 20%)"
```

**WAIT for the user to respond. Do NOT apply any imputation until the user confirms their choice.**

#### Step 7c: Backfill the date spine and apply imputation

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

The `SEQUENCE` interval must match the detected frequency (same as Step 7a).

Then apply the chosen imputation on the now-explicit NULLs:

- **Linear interpolation**: `(LAG(y IGNORE NULLS) + LEAD(y IGNORE NULLS)) / 2`
- **Forward fill**: `LAST_VALUE(y IGNORE NULLS) OVER (PARTITION BY unique_id ORDER BY ds)`
- **Fill with 0**: `COALESCE(y, 0)` — appropriate for count/demand data where absence means zero activity
- **Exclusion**: Remove series exceeding the threshold from `{use_case}_train_data`

> **Trailing-gap fallback (required after interpolation).** Linear interpolation needs a future neighbor via `LEAD(y IGNORE NULLS)`. For trailing gaps — rows beyond a series' last observed value — there is no future data point, so `LEAD` returns NULL and interpolation leaves those rows unfilled. After applying interpolation, always run a forward-fill pass on any remaining NULLs:
>
> ```sql
> UPDATE {catalog}.{schema}.{use_case}_train_data t
> SET y = (
>   SELECT LAST_VALUE(t2.y IGNORE NULLS) OVER (
>     PARTITION BY t2.unique_id ORDER BY t2.ds
>     ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
>   )
>   FROM {catalog}.{schema}.{use_case}_train_data t2
>   WHERE t2.unique_id = t.unique_id AND t2.ds = t.ds
> )
> WHERE t.y IS NULL
> ```
>
> This only affects trailing positions where interpolation could not reach. Forward-fill and fill-with-0 strategies do not need this extra step.

After imputation, verify no NULLs remain:

```sql
SELECT COUNT(*) AS remaining_nulls
FROM {catalog}.{schema}.{use_case}_train_data
WHERE y IS NULL
```

If `remaining_nulls > 0`, warn the user and suggest excluding those series or switching to forward-fill.

Log the count of imputed values per series and excluded series for the cleaning report.

### Step 8: Anomaly Detection & Analysis Report

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

#### Step 8a: Generate anomaly analysis report

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

#### ⛔ STOP GATE — Step 8b: Ask user how to handle anomalies

**Do NOT proceed until the user chooses an anomaly handling strategy.**

```
AskUserQuestion:
  "Based on the anomaly analysis report above:
   - {n_clean} series have no anomalies
   - {n_affected} series have outliers ({total_anomalies} points total, {anomaly_pct}% overall)
   - Default capping range: [Q1 - 1.5×IQR, Q3 + 1.5×IQR]

   How would you like to proceed?
   (a) Cap at 1.5×IQR (default — moderate, removes typical outliers)
   (b) Cap at 3.0×IQR (conservative — only removes extreme outliers)
   (c) Custom multiplier: enter a value (e.g., 2.0)
   (d) Skip anomaly capping — keep all values as-is"
```

**WAIT for the user to respond. Do NOT apply any capping until the user confirms their choice.**

#### Step 8c: Apply capping (if chosen)

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

### Step 9: Create {use_case}_cleaning_report

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

### Step 10: Generate reproducibility notebook

After all interactive decisions have been made, generate a self-contained notebook that replays the entire data preparation pipeline. This notebook allows the user (or a teammate) to re-run the exact same prep with the same parameters without going through the interactive session again.

**CRITICAL: Copy the template VERBATIM from `mmf_prep_notebook_template.ipynb`, only replacing the `{placeholder}` tokens with actual values. Do NOT add, remove, or modify any other code.**

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

Save the generated notebook to the **local project directory** at:

- `notebooks/{use_case}/prep_data.ipynb`

Then upload it to the Databricks workspace at `notebooks/{use_case}/prep_data`.

Use the template from:

- [mmf_prep_notebook_template.ipynb](mmf_prep_notebook_template.ipynb)

### ⛔ STOP GATE — Step 11: Confirm before proceeding to next skill

Present a summary of what was done and ask whether to proceed:

```
AskUserQuestion:
  "✅ Data preparation complete for use case '{use_case}'.

   Summary:
   • Forecast brief: {forecast_problem_brief}
   • Training table: {catalog}.{schema}.{use_case}_train_data
   • Series count: {n_series}
   • Date range: {min_date} → {max_date}
   • Frequency: {freq}
   • Imputation: {imputation_summary}
   • Anomalies: {anomaly_summary}
   • Cleaning report: {catalog}.{schema}.{use_case}_cleaning_report
   {if dynamic_future_regressors:
   • Scoring table: {catalog}.{schema}.{use_case}_scoring_data
     — {scoring_n_series} series × {scoring_date_range} future dates
     — Dynamic future columns: {dynamic_future_columns}
   }
   {if static_features: '• Static features: {static_features}'}
   {if dynamic_historical: '• Dynamic historical features: {dynamic_historical_columns}'}
   • Reproducibility notebook: notebooks/{use_case}/prep_data

   Would you like to proceed to the next step?
   (a) Run profiling & classification (optional — estimates series forecastability and recommends models)
   (b) Skip profiling and go directly to cluster provisioning & model selection
   (c) Stop here — I'll come back later"
```

**Do NOT proceed until the user responds.**

## Outputs

- A conversation-carried **`{forecast_problem_brief}`** (3–6 lines) for downstream skills and optional research scoping
- A Delta table `<catalog>.<schema>.{use_case}_train_data` with columns `unique_id` (STRING), `ds` (DATE for D/W/M, TIMESTAMP for H), `y` (DOUBLE), plus any exogenous regressor columns
- A Delta table `<catalog>.<schema>.{use_case}_cleaning_report` with columns: `unique_id`, `original_count`, `final_count`, `missing_filled`, `imputation_method`, `anomalies_capped`, `iqr_multiplier`, `excluded`, `exclusion_reason`
- **(If dynamic future regressors)** A Delta table `<catalog>.<schema>.{use_case}_scoring_data` with columns `unique_id` (STRING), `ds` (DATE/TIMESTAMP — future dates only), plus all dynamic future regressor columns. Does NOT include `y`, `static_features`, or `dynamic_historical_*` columns — `run_forecast` unions this with `train_data` via `allowMissingColumns=True`. This table is passed as `scoring_data` to `run_forecast` in Skill 4.
- Exogenous regressor column lists carried forward: `{static_features}`, `{dynamic_future_numerical}`, `{dynamic_future_categorical}`, `{dynamic_historical_numerical}`, `{dynamic_historical_categorical}`
- A reproducibility notebook uploaded to `notebooks/{use_case}/prep_data` that can re-create the training table with the same parameters
- A summary: number of series, date range, detected frequency, cleaning actions taken

