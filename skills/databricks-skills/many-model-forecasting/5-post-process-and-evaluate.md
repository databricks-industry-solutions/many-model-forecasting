# Post-Process and Evaluate

**Slash command:** `/post-process-and-evaluate <catalog> <schema>`

Calculates multiple accuracy metrics, performs best-model selection per series,
and formats results for business consumption.

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `catalog` | Ask user | Unity Catalog name |
| `schema` | Ask user | Schema name |
| `use_case` | From Skill 1 | Use case name (prefixes all table names) |
| `metric` | `smape` | Primary evaluation metric |
| `evaluation_table` | `{use_case}_evaluation_output` | Evaluation output from Skill 4 |
| `scoring_table` | `{use_case}_scoring_output` | Scoring output from Skill 4 |

## Steps

### Step 1: Verify outputs exist

```sql
SELECT COUNT(*) AS eval_count FROM {catalog}.{schema}.{use_case}_evaluation_output
```
```sql
SELECT COUNT(*) AS score_count FROM {catalog}.{schema}.{use_case}_scoring_output
```

If either table is empty or missing, instruct the user to run `/execute-mmf-forecast` first.

### Step 2: Compute multi-metric evaluation

Calculate the primary metric and WAPE from the stored `forecast` and `actual` arrays:

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
FROM {catalog}.{schema}.{use_case}_evaluation_output
GROUP BY unique_id, model
```

### Step 3: Best model selection per series

Select the best-performing model for each time series based on the primary metric:

```sql
CREATE OR REPLACE TABLE {catalog}.{schema}.{use_case}_best_models AS
SELECT eval.unique_id, eval.model, eval.avg_metric, score.ds, score.y
FROM (
  SELECT unique_id, model, avg_metric,
         RANK() OVER (PARTITION BY unique_id ORDER BY avg_metric ASC) AS rank
  FROM (
    SELECT unique_id, model, AVG(metric_value) AS avg_metric
    FROM {catalog}.{schema}.{use_case}_evaluation_output
    GROUP BY unique_id, model
    HAVING AVG(metric_value) IS NOT NULL
  )
) AS eval
INNER JOIN {catalog}.{schema}.{use_case}_scoring_output AS score
  ON eval.unique_id = score.unique_id AND eval.model = score.model
WHERE eval.rank = 1
```

Verify:
```sql
SELECT COUNT(*) AS best_model_count FROM {catalog}.{schema}.{use_case}_best_models
```

### Step 4: Model ranking (wins count)

Count how many series each model won:

```sql
SELECT model, COUNT(*) AS wins_count,
       ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 2) AS wins_pct
FROM {catalog}.{schema}.{use_case}_best_models
GROUP BY model
ORDER BY wins_count DESC
```

### Step 5: Create evaluation summary table

```sql
CREATE OR REPLACE TABLE {catalog}.{schema}.{use_case}_evaluation_summary AS
SELECT
    model,
    COUNT(*) AS wins_count,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 2) AS wins_pct,
    ROUND(AVG(avg_metric), 4) AS avg_smape,
    NULL AS avg_wape
FROM {catalog}.{schema}.{use_case}_best_models
GROUP BY model
ORDER BY wins_count DESC
```

### Step 6: Business-ready summary report

Present to the user:
- Total series evaluated
- Number of distinct models that "won" at least one series
- Top 3 models by wins count
- Average metric across all series (overall forecast quality)
- Worst 10 series (potential data quality issues for re-investigation)

```sql
-- Overall forecast quality
SELECT
  COUNT(DISTINCT unique_id) AS total_series,
  COUNT(DISTINCT model) AS distinct_winning_models,
  ROUND(AVG(avg_metric), 4) AS overall_avg_metric
FROM {catalog}.{schema}.{use_case}_best_models
```

```sql
-- Worst-performing series
SELECT unique_id, model, avg_metric
FROM {catalog}.{schema}.{use_case}_best_models
ORDER BY avg_metric DESC
LIMIT 10
```

### Step 7: Cross-reference with profiling (if available)

If `{use_case}_series_profile` exists, join with best model results to analyze performance by forecastability class:

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

If the profile table does not exist, skip this step.

### Step 8: Suggest next steps

Based on results:
- If many series have high error: suggest re-running with different models or checking data quality
- If foundation models won most series: suggest using them for scoring
- If local and foundation results are similar: suggest using local models for cost efficiency
- If `low_signal` series still have high error: confirm they should be excluded from business decisions
- If the user wants to iterate: allow re-running with different models or parameters

## Outputs

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
