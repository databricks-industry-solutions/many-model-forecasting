# Post-Process and Evaluate

Calculates multiple accuracy metrics, performs best-model selection per series,
and formats results for business consumption.

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `catalog` | Ask user | Unity Catalog name |
| `schema` | Ask user | Schema name |
| `use_case` | From Step 1 | Use case name (prefixes all table names) |
| `metric` | `smape` | Primary evaluation metric |
| `evaluation_table` | `{use_case}_evaluation_output` | Evaluation output from Step 4 |
| `scoring_table` | `{use_case}_scoring_output` | Scoring output from Step 4 |

## Steps

### ⛔ STOP GATE — Step 0: Ask for catalog and schema

**Always ask the user for catalog and schema. Do NOT assume or reuse values.**

Ask the user:

> "Which catalog and schema contain the forecast outputs?
> - Catalog: (e.g., main, ml_dev)
> - Schema: (e.g., default, forecasting)
> - Use case name: (e.g., m4, rossmann)"

**Do NOT proceed until the user provides catalog, schema, and use case name.**

### Step 0a: Read non-forecastable strategy from pipeline config

```sql
SELECT non_forecastable_strategy, fallback_method, non_forecastable_models,
       n_forecastable, n_non_forecastable
FROM {catalog}.{schema}.{use_case}_pipeline_config
WHERE use_case = '{use_case}'
```

If the table does not exist, treat strategy as `include`.

### Step 1: Verify outputs exist

```sql
SELECT COUNT(*) AS eval_count FROM {catalog}.{schema}.{use_case}_evaluation_output
```
```sql
SELECT COUNT(*) AS score_count FROM {catalog}.{schema}.{use_case}_scoring_output
```

If either table is empty or missing, instruct the user to run the forecast execution step first.

If `separate_job` strategy, also verify:
```sql
SELECT COUNT(*) AS nf_eval_count FROM {catalog}.{schema}.{use_case}_nf_evaluation_output
```

If `fallback` strategy, verify:
```sql
SELECT COUNT(*) AS fallback_count FROM {catalog}.{schema}.{use_case}_scoring_output_non_forecastable
```

### Step 2: Compute multi-metric evaluation

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

### Step 3: Best model selection per series (forecastable)

```sql
CREATE OR REPLACE TABLE {catalog}.{schema}.{use_case}_best_models_forecastable AS
SELECT eval.unique_id, eval.model, eval.avg_metric, score.ds, score.y, 'main_pipeline' AS forecast_source
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

### Step 3a: Incorporate non-forecastable series results

#### If strategy is `include` (or config missing)

```sql
CREATE OR REPLACE TABLE {catalog}.{schema}.{use_case}_best_models AS
SELECT unique_id, model, avg_metric, ds, y, 'main_pipeline' AS forecast_source
FROM {catalog}.{schema}.{use_case}_best_models_forecastable
```

#### If strategy is `fallback`

```sql
CREATE OR REPLACE TABLE {catalog}.{schema}.{use_case}_best_models AS
SELECT unique_id, model, avg_metric, ds, y, forecast_source
FROM {catalog}.{schema}.{use_case}_best_models_forecastable

UNION ALL

SELECT unique_id, model, NULL AS avg_metric, NULL AS ds, y, 'fallback' AS forecast_source
FROM {catalog}.{schema}.{use_case}_scoring_output_non_forecastable
```

#### If strategy is `separate_job`

```sql
CREATE OR REPLACE TABLE {catalog}.{schema}.{use_case}_best_models AS
SELECT unique_id, model, avg_metric, ds, y, 'main_pipeline' AS forecast_source
FROM {catalog}.{schema}.{use_case}_best_models_forecastable

UNION ALL

SELECT nf_eval.unique_id, nf_eval.model, nf_eval.avg_metric, nf_score.ds, nf_score.y,
       'non_forecastable_pipeline' AS forecast_source
FROM (
  SELECT unique_id, model, avg_metric,
         RANK() OVER (PARTITION BY unique_id ORDER BY avg_metric ASC) AS rank
  FROM (
    SELECT unique_id, model, AVG(metric_value) AS avg_metric
    FROM {catalog}.{schema}.{use_case}_nf_evaluation_output
    GROUP BY unique_id, model
    HAVING AVG(metric_value) IS NOT NULL
  )
) AS nf_eval
INNER JOIN {catalog}.{schema}.{use_case}_nf_scoring_output AS nf_score
  ON nf_eval.unique_id = nf_score.unique_id AND nf_eval.model = nf_score.model
WHERE nf_eval.rank = 1
```

### Step 4: Model ranking (wins count)

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

```sql
-- Overall forecast quality
SELECT
  COUNT(DISTINCT unique_id) AS total_series,
  COUNT(DISTINCT model) AS distinct_winning_models,
  ROUND(AVG(avg_metric), 4) AS overall_avg_metric
FROM {catalog}.{schema}.{use_case}_best_models
```

```sql
-- Breakdown by forecast source
SELECT
  forecast_source,
  COUNT(DISTINCT unique_id) AS series_count,
  COUNT(DISTINCT model) AS distinct_models,
  ROUND(AVG(avg_metric), 4) AS avg_metric
FROM {catalog}.{schema}.{use_case}_best_models
GROUP BY forecast_source
```

```sql
-- Worst-performing series (excluding fallback which has no metric)
SELECT unique_id, model, avg_metric, forecast_source
FROM {catalog}.{schema}.{use_case}_best_models
WHERE avg_metric IS NOT NULL
ORDER BY avg_metric DESC
LIMIT 10
```

### Step 7: Cross-reference with profiling (if available)

If `{use_case}_series_profile` exists:

```sql
SELECT
    b.model,
    b.forecast_source,
    p.forecastability_class,
    COUNT(*) AS series_count,
    ROUND(AVG(b.avg_metric), 4) AS avg_metric
FROM {catalog}.{schema}.{use_case}_best_models b
LEFT JOIN {catalog}.{schema}.{use_case}_series_profile p
  ON b.unique_id = p.unique_id
GROUP BY b.model, b.forecast_source, p.forecastability_class
ORDER BY p.forecastability_class, b.forecast_source, avg_metric
```

If the profile table does not exist, skip this step.

### Step 8: Suggest next steps

Based on results:
- If many series have high error: suggest re-running with different models or checking data quality
- If foundation models won most series: suggest using them for scoring
- If local and foundation results are similar: suggest using local models for cost efficiency
- If `include` strategy and low_signal series have high error: suggest re-running with `fallback` or `separate_job`
- If `fallback` strategy: note that fallback series have no evaluation metrics — suggest upgrading to `separate_job` if the user wants backtested accuracy

### Step 9: Generate reproducibility notebook

**CRITICAL: Do NOT execute this code inline. Generate the notebook from the template, upload it to the workspace. Never run post-processing pipeline code directly in the conversation.**

**CRITICAL: Use the template at `notebooks/mmf_post_process_notebook_template.ipynb` (in this skill folder). Copy it verbatim, only replacing the `{placeholder}` tokens.**

Replace these placeholders:
- `{catalog}` → user's catalog
- `{schema}` → user's schema
- `{use_case}` → use case name
- `{metric}` → primary evaluation metric (e.g., `smape`)

#### ⛔ STOP GATE — Confirm before uploading notebook

Before uploading, present a summary and ask the user:

> "I am about to upload the post-processing notebook to the workspace:
> - Path: `{home_path}/mmf-skills-test/notebooks/{use_case}/05_post_process`
> - Parameters: catalog={catalog}, schema={schema}, use_case={use_case}, metric={metric}
>
> Shall I proceed?"

**Do NOT upload until the user confirms.**

Upload the generated notebook directly to the Databricks workspace at:
```
{home_path}/mmf-skills-test/notebooks/{use_case}/05_post_process
```

Where `{home_path}` = `/Workspace/Users/{current_user_email}`.

### ⛔ STOP GATE — Step 10: Final confirmation

Ask the user:

> "✅ Post-processing and evaluation complete for use case '{use_case}'.
>
> Summary:
> - Total series: {total_series}
> - Winning models: {n_winning_models}
> - Top model: {top_model} ({top_wins} wins, {top_pct}%)
> - Overall avg {metric}: {overall_avg} (forecastable series only)
> - Best models table: {catalog}.{schema}.{use_case}_best_models
> - Evaluation summary: {catalog}.{schema}.{use_case}_evaluation_summary
> - Reproducibility notebook: notebooks/{use_case}/05_post_process
>
> What would you like to do next?
> (a) Re-run forecasting with different models or parameters
> (b) Explore results further (drill into specific series or models)
> (c) Done — all outputs are ready for business use"

## Outputs

**Table**: `<catalog>.<schema>.{use_case}_best_models`

| Column | Type | Description |
|--------|------|-------------|
| `unique_id` | STRING | Series identifier |
| `model` | STRING | Best model name (or fallback method name) |
| `avg_metric` | DOUBLE | Average backtest metric (NULL for fallback series) |
| `ds` | ARRAY&lt;TIMESTAMP&gt; | Forecast dates |
| `y` | ARRAY&lt;DOUBLE&gt; | Forecast values |
| `forecast_source` | STRING | `main_pipeline`, `non_forecastable_pipeline`, or `fallback` |

**Table**: `<catalog>.<schema>.{use_case}_evaluation_summary`

| Column | Type | Description |
|--------|------|-------------|
| `model` | STRING | Model name |
| `wins_count` | INT | Number of series where this model was best |
| `wins_pct` | DOUBLE | Percentage of total series |
| `avg_smape` | DOUBLE | Average sMAPE across all series |
| `avg_wape` | DOUBLE | Average WAPE across all series |

**Notebook**: `notebooks/{use_case}/05_post_process` — reproducibility notebook
