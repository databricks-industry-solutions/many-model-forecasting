# MMF Results Explorer

Dash app for browsing evaluation and scoring tables produced by `run_forecast`.

## Inputs

| Field | Required | Description |
|-------|----------|-------------|
| Catalog | Yes | Unity Catalog name |
| Schema | Yes | Schema name |
| Evaluation Table | No | Table written to `evaluation_output` by `run_forecast` |
| Scoring Table | No | Table written to `scoring_output` by `run_forecast` |
| Run ID | No | Filters both tables to a single run |

Binary columns (e.g. `model_pickle`) are excluded automatically. Array columns are displayed as strings. Results are capped at 10,000 rows.

## Prerequisites

- A SQL warehouse with a resource key `sql-warehouse` added to the app in the Databricks Apps UI.
- The app's service principal needs `CAN USE` on the warehouse and `SELECT` on the target tables.

## Deploy

```bash
databricks apps create mmf-app --profile YOUR_PROFILE

databricks workspace import-dir apps \
  /Workspace/Users/<your-email>/apps/mmf-app --profile YOUR_PROFILE

databricks apps deploy mmf-app \
  --source-code-path /Workspace/Users/<your-email>/apps/mmf-app --profile YOUR_PROFILE
```

## Redeploy after changes

```bash
databricks workspace delete /Workspace/Users/<your-email>/apps/mmf-app --recursive --profile aws
databricks workspace import-dir apps \
  /Workspace/Users/<your-email>/apps/mmf-app --profile YOUR_PROFILE
databricks apps deploy mmf-app \
  --source-code-path /Workspace/Users/<your-email>/apps/mmf-app --profile YOUR_PROFILE
```
