# Post-Process and Evaluate

**Slash command:** `/post-process-and-evaluate`

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

### ⛔ STOP GATE — Step 0: Ask for catalog and schema

**Always ask the user for catalog and schema. Do NOT assume or reuse values.**

```
AskUserQuestion:
  "Which catalog and schema contain the forecast outputs?
   • Catalog: (e.g., main, ml_dev)
   • Schema:  (e.g., default, forecasting)
   • Use case name: (e.g., m4, rossmann)"
```

**Do NOT proceed until the user provides catalog, schema, and use case name.**

### Step 0a: Read non-forecastable strategy from pipeline config

Check if `{use_case}_pipeline_config` exists:

```sql
SELECT non_forecastable_strategy, fallback_method, non_forecastable_models,
       n_forecastable, n_non_forecastable
FROM {catalog}.{schema}.{use_case}_pipeline_config
WHERE use_case = '{use_case}'
```

If the table does not exist, treat strategy as `include`. The strategy determines which tables to look for and how to merge results.

### Step 1: Verify outputs exist

```sql
SELECT COUNT(*) AS eval_count FROM {catalog}.{schema}.{use_case}_evaluation_output
```
```sql
SELECT COUNT(*) AS score_count FROM {catalog}.{schema}.{use_case}_scoring_output
```

If either table is empty or missing, instruct the user to run `/execute-mmf-forecast` first.

If `separate_job` strategy, also verify non-forecastable outputs:
```sql
SELECT COUNT(*) AS nf_eval_count FROM {catalog}.{schema}.{use_case}_nf_evaluation_output
```
```sql
SELECT COUNT(*) AS nf_score_count FROM {catalog}.{schema}.{use_case}_nf_scoring_output
```

If `fallback` strategy, verify fallback outputs:
```sql
SELECT COUNT(*) AS fallback_count FROM {catalog}.{schema}.{use_case}_scoring_output_non_forecastable
```

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

### Step 3: Best model selection per series (forecastable)

Select the best-performing model for each forecastable time series based on the primary metric:

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

Based on the non-forecastable strategy from Step 0a:

#### If strategy is `include` (or config missing)

All series are already in the main evaluation/scoring tables. No merging needed:

```sql
CREATE OR REPLACE TABLE {catalog}.{schema}.{use_case}_best_models AS
SELECT unique_id, model, avg_metric, ds, y, 'main_pipeline' AS forecast_source
FROM {catalog}.{schema}.{use_case}_best_models_forecastable
```

#### If strategy is `fallback`

Combine forecastable best models with fallback forecasts. Fallback series have no evaluation metrics (no backtest was run):

```sql
CREATE OR REPLACE TABLE {catalog}.{schema}.{use_case}_best_models AS
SELECT unique_id, model, avg_metric, ds, y, forecast_source
FROM {catalog}.{schema}.{use_case}_best_models_forecastable

UNION ALL

SELECT unique_id, model, NULL AS avg_metric, NULL AS ds, y, 'fallback' AS forecast_source
FROM {catalog}.{schema}.{use_case}_scoring_output_non_forecastable
```

#### If strategy is `separate_job`

Run best-model selection on the non-forecastable evaluation outputs, then merge:

```sql
CREATE OR REPLACE TABLE {catalog}.{schema}.{use_case}_best_models AS
-- Forecastable best models
SELECT unique_id, model, avg_metric, ds, y, 'main_pipeline' AS forecast_source
FROM {catalog}.{schema}.{use_case}_best_models_forecastable

UNION ALL

-- Non-forecastable best models (from separate pipeline)
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

Verify the combined result:
```sql
SELECT forecast_source, COUNT(*) AS series_count
FROM {catalog}.{schema}.{use_case}_best_models
GROUP BY forecast_source
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
- Total series evaluated (broken down by forecast source)
- Number of distinct models that "won" at least one series
- Top 3 models by wins count
- Average metric across all series (overall forecast quality)
- Worst 10 series (potential data quality issues for re-investigation)

```sql
-- Overall forecast quality (combined)
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

If fallback series exist, report them separately:
```sql
SELECT COUNT(*) AS fallback_series, model AS fallback_method
FROM {catalog}.{schema}.{use_case}_best_models
WHERE forecast_source = 'fallback'
GROUP BY model
```

### Step 7: Cross-reference with profiling (if available)

If `{use_case}_series_profile` exists, join with best model results to analyze performance by forecastability class and forecast source:

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
- If the user wants to iterate: allow re-running with different models or parameters

**Strategy-specific suggestions:**
- If `include` strategy and low_signal series have high error: suggest re-running with `fallback` or `separate_job` strategy to isolate them
- If `fallback` strategy: report that fallback series have no evaluation metrics — suggest upgrading to `separate_job` if the user wants backtested accuracy for those series
- If `separate_job` strategy and non-forecastable series still perform poorly: confirm they should be flagged in business outputs; consider switching to `fallback` to save compute
- If `separate_job` strategy and non-forecastable series perform well with specialized models: the separate pipeline was worthwhile — highlight which models recovered signal

### Step 9: Generate reproducibility notebook

After all evaluation queries have been run, generate a self-contained notebook that replays the entire post-processing and evaluation pipeline. This notebook allows the user (or a teammate) to re-run the exact same evaluation and best-model selection without going through the interactive session again.

**CRITICAL: Copy the template VERBATIM from `mmf_post_process_notebook_template.ipynb`, only replacing the `{placeholder}` tokens with actual values. Do NOT add, remove, or modify any other code.**

Replace these placeholders:
- `{catalog}` → user's catalog
- `{schema}` → user's schema
- `{use_case}` → use case name
- `{metric}` → primary evaluation metric (e.g., `smape`)

Save the generated notebook to the **local project directory** at:
- `notebooks/{use_case}/post_process.ipynb`

Then upload it to the Databricks workspace at `notebooks/{use_case}/post_process`.

Use the template from:
- [mmf_post_process_notebook_template.ipynb](mmf_post_process_notebook_template.ipynb)

### ⛔ STOP GATE — Step 10: Final confirmation

```
AskUserQuestion:
  "✅ Post-processing and evaluation complete for use case '{use_case}'.

   Summary:
   • Total series: {total_series}
     {if strategy != 'include':
     — Forecastable (main pipeline): {n_forecastable}
     — Non-forecastable ({strategy}): {n_non_forecastable}
     }
   • Winning models: {n_winning_models}
   • Top model: {top_model} ({top_wins} wins, {top_pct}%)
   • Overall avg {metric}: {overall_avg} (forecastable series only)
   • Best models table: {catalog}.{schema}.{use_case}_best_models
   • Evaluation summary: {catalog}.{schema}.{use_case}_evaluation_summary
   • Reproducibility notebook: notebooks/{use_case}/post_process

   What would you like to do next?
   (a) Re-run forecasting with different models or parameters
   (b) Explore results further (drill into specific series or models)
   {if strategy == 'include': '(c) Re-run with non-forecastable series handled separately'}
   {if strategy == 'fallback': '(c) Upgrade non-forecastable series to a separate pipeline (for backtested metrics)'}
   (d) Done — all outputs are ready for business use"
```

## Outputs

**Table**: `<catalog>.<schema>.{use_case}_best_models`

Combined table of best forecasts for all series (forecastable + non-forecastable).

| Column | Type | Description |
|--------|------|-------------|
| `unique_id` | STRING | Series identifier |
| `model` | STRING | Best model name (or fallback method name) |
| `avg_metric` | DOUBLE | Average backtest metric (NULL for fallback series) |
| `ds` | ARRAY&lt;TIMESTAMP&gt; | Forecast dates (NULL for fallback series) |
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

**Notebook**: `notebooks/{use_case}/post_process` — reproducibility notebook that re-creates both output tables

---

## Deploy the Results Explorer App

The `apps/` folder at the project root contains a ready-to-deploy **Dash** application
("MMF Results Explorer") that lets the user interactively browse evaluation and
scoring tables produced by the forecasting pipeline. It supports:

- Selecting catalog, schema, training/evaluation/scoring tables from dropdowns
- Filtering by run date, run ID, group IDs, and model(s)
- Auto-detecting column roles and best-model selection per series
- Plotting historical + forecast curves for selected time series
- Browsing backtest windows with forecast-vs-actual overlays
- A model ranking chart showing how often each model was chosen as best

### Prerequisites

1. A Databricks workspace with the **Apps** feature enabled.
2. A SQL warehouse the app can connect to.
3. The app's service principal needs `CAN USE` on the warehouse and `SELECT`
   on the evaluation, scoring, and training tables.

### Deploy Steps

Run the following CLI commands (replace `<your-email>` and `YOUR_PROFILE`):

```bash
# 1. Create the app resource
databricks apps create mmf-app --profile YOUR_PROFILE

# 2. Upload the app code to the workspace
databricks workspace import-dir apps \
  /Workspace/Users/<your-email>/apps/mmf-app --profile YOUR_PROFILE

# 3. Deploy the app
databricks apps deploy mmf-app \
  --source-code-path /Workspace/Users/<your-email>/apps/mmf-app --profile YOUR_PROFILE
```

After deploying, open the Databricks Apps UI and **add a SQL warehouse resource**
with the key `sql-warehouse` so the app can auto-detect the warehouse ID.
Alternatively, the user can paste a warehouse ID directly in the app's form.

### Redeploy After Changes

If the app code is updated, re-upload and redeploy:

```bash
databricks workspace delete /Workspace/Users/<your-email>/apps/mmf-app \
  --recursive --profile YOUR_PROFILE

databricks workspace import-dir apps \
  /Workspace/Users/<your-email>/apps/mmf-app --profile YOUR_PROFILE

databricks apps deploy mmf-app \
  --source-code-path /Workspace/Users/<your-email>/apps/mmf-app --profile YOUR_PROFILE
```

### What the App Contains

| File | Purpose |
|------|---------|
| `apps/app.py` | Dash application — layout, callbacks, SQL queries against evaluation/scoring/training tables |
| `apps/app.yaml` | Databricks Apps manifest — sets the run command and binds the `sql-warehouse` resource |
| `apps/requirements.txt` | Python dependencies (`dash-bootstrap-components`, `databricks-sql-connector`, `databricks-sdk`, `pandas`) |
| `apps/README.md` | Detailed deploy/redeploy instructions and input reference |

### How It Works

The app connects to the SQL warehouse using `databricks-sql-connector` and the
workspace SDK for authentication. When the user enters a catalog and schema, it
lists available tables and auto-detects column roles (group ID, forecast arrays,
metric values, etc.). Clicking **Load** joins the evaluation and scoring tables
to show forecasts alongside historical training data. If backtest windows are
present in the evaluation table, the app renders tabbed backtest charts with
forecast-vs-actual overlays so the user can visually assess model performance
across different holdout periods.
