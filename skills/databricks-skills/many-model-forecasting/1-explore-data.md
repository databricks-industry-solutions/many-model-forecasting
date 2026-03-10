# Explore Data

**Slash command:** `/explore-data <catalog> <schema>`

Connects to a Databricks workspace, discovers time series tables, maps columns
to the MMF schema (`unique_id`, `ds`, `y`), runs data quality checks, and
creates the `mmf_train_data` table ready for forecasting.

## Steps

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
- `avg_gap` ~1 = **Daily** (`freq=D`)
- `avg_gap` ~7 = **Weekly** (`freq=W`)
- `avg_gap` ~28-31 = **Monthly** (`freq=M`)

Report the detected frequency to the user.

### Step 5: Validate with user

Use `AskUserQuestion` to confirm:
- Source table selection
- Column mapping: `unique_id`, `ds`, `y`
- Any optional exogenous regressors to include

### Step 6: Data quality checks

Run quality checks on the selected table:

**Missing values per series:**
```sql
SELECT {unique_id_col},
       COUNT(*) AS total,
       SUM(CASE WHEN {y_col} IS NULL THEN 1 ELSE 0 END) AS missing_y,
       ROUND(SUM(CASE WHEN {y_col} IS NULL THEN 1 ELSE 0 END) / COUNT(*) * 100, 2) AS missing_pct
FROM {catalog}.{schema}.{table_name}
GROUP BY {unique_id_col}
HAVING missing_pct > 0
ORDER BY missing_pct DESC
```

**Negative values per series:**
```sql
SELECT {unique_id_col},
       COUNT(*) AS total,
       SUM(CASE WHEN {y_col} < 0 THEN 1 ELSE 0 END) AS negative_count,
       ROUND(SUM(CASE WHEN {y_col} < 0 THEN 1 ELSE 0 END) / COUNT(*) * 100, 2) AS negative_pct
FROM {catalog}.{schema}.{table_name}
GROUP BY {unique_id_col}
HAVING negative_count > 0
ORDER BY negative_pct DESC
```

**Thresholds:**
- Series with > 20% missing values: flag for removal
- Series with > 20% negative values: flag for removal
- Report total series that pass vs. fail quality checks

### Step 7: Create mmf_train_data

Create the training table with the MMF-required schema. The SQL depends on the detected (or user-specified) frequency to ensure dates are properly aligned:

**Daily (`freq=D`) — no date transformation needed:**
```sql
CREATE OR REPLACE TABLE {catalog}.{schema}.mmf_train_data AS
SELECT
    CAST({unique_id_col} AS STRING) AS unique_id,
    CAST({ds_col} AS TIMESTAMP) AS ds,
    CAST({y_col} AS DOUBLE) AS y
FROM {catalog}.{schema}.{table_name}
WHERE {y_col} IS NOT NULL
```

**Weekly (`freq=W`) — align dates to Sunday (end of ISO week) and aggregate:**
```sql
CREATE OR REPLACE TABLE {catalog}.{schema}.mmf_train_data AS
SELECT
    CAST({unique_id_col} AS STRING) AS unique_id,
    CAST(DATE_TRUNC('week', {ds_col}) + INTERVAL 6 DAY AS TIMESTAMP) AS ds,
    SUM(CAST({y_col} AS DOUBLE)) AS y
FROM {catalog}.{schema}.{table_name}
WHERE {y_col} IS NOT NULL
GROUP BY {unique_id_col}, DATE_TRUNC('week', {ds_col}) + INTERVAL 6 DAY
```

**Monthly (`freq=M`) — align dates to month-end and aggregate:**
```sql
CREATE OR REPLACE TABLE {catalog}.{schema}.mmf_train_data AS
SELECT
    CAST({unique_id_col} AS STRING) AS unique_id,
    CAST(LAST_DAY({ds_col}) AS TIMESTAMP) AS ds,
    SUM(CAST({y_col} AS DOUBLE)) AS y
FROM {catalog}.{schema}.{table_name}
WHERE {y_col} IS NOT NULL
GROUP BY {unique_id_col}, LAST_DAY({ds_col})
```

If exogenous regressors were selected, include them as additional columns (use `SUM` for numeric regressors when aggregating).

Verify creation:
```sql
SELECT COUNT(*) AS count FROM {catalog}.{schema}.mmf_train_data
```

## Outputs

- A Delta table `<catalog>.<schema>.mmf_train_data` with columns `unique_id` (STRING), `ds` (TIMESTAMP), `y` (DOUBLE)
- A summary: number of series, date range, detected frequency, data quality report