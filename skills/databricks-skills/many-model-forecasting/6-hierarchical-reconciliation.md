# Hierarchical Reconciliation

> ⛔ **MANDATORY:** If you have not read [SKILL.md](SKILL.md) yet, read it now before proceeding. Do NOT take any action until you have read both SKILL.md and this file in full.

**Slash command:** `/hierarchical-reconciliation`

Applies hierarchical reconciliation to MMF forecasts, making them coherent across all levels of a hierarchy (e.g., SKU → Category → Total). Works with any forecast table — by default uses the best-model output from Skill 5, but can reconcile any table with `(unique_id, ds, forecast_value)` columns.

**This skill is optional — only run it if the use case has a meaningful hierarchy.**

## Preconditions

> ⛔ **Verify before starting this skill.** If preconditions are missing, do NOT improvise — route the user back.

| Precondition | How to verify | If missing |
|---|---|---|
| `{catalog}.{schema}.{use_case}_evaluation_output` exists and is populated | `SELECT COUNT(*) FROM ...` | Go back to **Skill 4 (`/execute-mmf-forecast`)** — `evaluation_output` is required for all reconciliation methods |
| A forecast table with `(unique_id, ds, y)` columns exists | Confirmed in Step 0a | Route back to **Skill 5** or ask the user which table to use |

> ℹ️ `_hierarchy_S` and `_hierarchy_tags` are **not** required upfront — Step 1a auto-derives them from `train_data` if missing.

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `catalog` | From prior skills | Unity Catalog name |
| `schema` | From prior skills | Schema name |
| `use_case` | From Skill 1 | Use case name (prefixes all table names) |
| `freq` | From Skill 1 | Frequency code — H \| D \| W \| M |
| `date_col` | `ds` | Date column name |
| `target` | `y` | Target column name |
| `reconciliation_method` | `MinTrace` | Reconciliation method — see Step 2 |

## Steps

### ⛔ STOP GATE — Step 0: Confirm catalog, schema, use case

If not already known from prior skills, ask in plain text (do NOT use AskUserQuestion — this is free text):

> "To get started I need three things:
>  • Catalog:
>  • Schema:
>  • Use case name (the prefix for your tables, e.g. `rossmann`, `retail_sales`):"

**Do NOT proceed until the user provides all three.**

Call `get_current_user()` to obtain `{full_email}`. Then set:
- `{notebook_base_path}` = `/Users/{full_email}/{use_case}/notebooks` (or carry forward from Skill 1 if available in context)

### ⛔ STOP GATE — Step 0a: Confirm forecast table

First, silently check if the default table exists:

```sql
SELECT COUNT(*) AS n FROM {catalog}.{schema}.{use_case}_best_models
```

- If it exists (`n > 0`): present it as option (a) recommended.
- If it does not exist: present only option (b) — do NOT show option (a).

```
AskUserQuestion:
  "Which table contains the forecasts you want to reconcile?

   (a) {catalog}.{schema}.{use_case}_best_models — default output from Skill 5  [only show if exists]
   (b) A different table — I will provide the full name (catalog.schema.table)

   Options: [a, b]"
```

- If **(a)**: set `{forecast_table}` = `{catalog}.{schema}.{use_case}_best_models`
- If **(b)**: ask in plain text "Provide the full table name (catalog.schema.table):" and store as `{forecast_table}`

**Do NOT proceed until `{forecast_table}` is confirmed.**

### Step 0b: Auto-detect frequency

Once `{forecast_table}` is known, detect `{freq}` automatically — do NOT ask the user:

```sql
SELECT ds FROM {forecast_table} WHERE ds IS NOT NULL LIMIT 1000
```

Compute the median gap between consecutive distinct `ds` values:
- Gap ≈ 1 day → `D`
- Gap ≈ 7 days → `W`
- Gap ≈ 28–31 days → `M`
- Gap < 1 day → `H`

If `{freq}` was already known from prior skills, skip this step and carry it forward.

Do NOT narrate this detection to the user. Proceed silently with `{freq}` set.

Only ask the user if the gap is ambiguous (e.g. mixed gaps or fewer than 2 distinct dates):

```
AskUserQuestion:
  "Could not determine frequency automatically. What is the forecast frequency?

   (a) Daily (D)
   (b) Weekly (W)
   (c) Monthly (M)
   (d) Hourly (H)

   Options: [a, b, c, d]"
```

### Step 1: Verify forecast inputs

Check that the required tables exist and are populated:

```sql
SELECT COUNT(*) AS n_forecast FROM {forecast_table}
```
```sql
SELECT COUNT(*) AS n_evaluation FROM {catalog}.{schema}.{use_case}_evaluation_output
```

If either is missing or empty, route back to the appropriate skill (Skill 4 or Skill 5).

> ℹ️ **Hierarchy metadata is handled automatically** — the reconciliation notebook includes a conditional cell that derives `_hierarchy_S` and `_hierarchy_tags` from `train_data` if they are missing. No separate job needed. Proceed directly to Step 2.

### ⛔ STOP GATE — Step 2: Propose and confirm reconciliation method

Present MinTrace as the recommendation with reasoning, then ask:

> "I recommend **MinTrace** (`mint_shrink`) — it minimizes forecast error variance by estimating the covariance across all hierarchy levels. It is the statistically optimal method when backtest residuals are available.
>
> Alternatives: BottomUp (aggregates leaf forecasts upward), TopDown (distributes from the total downward), MiddleOut (anchors at an intermediate level), ERM (learns an optimal reconciliation matrix).

```
AskUserQuestion:
  "Should we use MinTrace or do you prefer a different method?

   (a) MinTrace — recommended
   (b) I want to choose a different method

   Options: [a, b]"
```

- If **(a)**: set `{reconciliation_method}` = `MinTrace`. Proceed to Step 3.
- If **(b)**:

```
AskUserQuestion:
  "Which method would you like to use?

   (a) BottomUp — aggregates leaf-level forecasts upward
   (b) TopDown — distributes the top-level forecast downward
   (c) MiddleOut — anchors at an intermediate level
   (d) ERM — learns an optimal reconciliation matrix

   Options: [a, b, c, d]"
```

Map: (a) → `BottomUp`, (b) → `TopDown`, (c) → `MiddleOut`, (d) → `ERM`. Store as `{reconciliation_method}`.

### Step 3: Generate reconciliation notebook

Generate `{notebook_base_path}/run_reconciliation.ipynb` from the template `mmf_reconciliation_notebook_template.ipynb`, filling in:

| Placeholder | Value |
|-------------|-------|
| `{full_email}` | from `get_current_user()` — used in install path |
| `{catalog}` | confirmed catalog |
| `{schema}` | confirmed schema |
| `{use_case}` | confirmed use case |
| `{forecast_table}` | confirmed in Step 0a — full 3-part table name |
| `{freq}` | frequency — H \| D \| W \| M |
| `{date_col}` | `ds` (or user-specified) |
| `{target}` | `y` (or user-specified) |
| `{reconciliation_method}` | method confirmed in Step 2 |

### Step 4: Run on classic compute (Single Node, memory-optimized)

Reconciliation is a matrix operation that requires classic compute — `toArrow()` and scipy sparse are not available on Spark Connect (serverless). Run on a **Single Node** job cluster with DBR ML.

> ⛔ **MANDATORY cluster config — DO NOT use serverless.**
> ```json
> {
>   "spark_version": "17.3.x-cpu-ml-scala2.13",
>   "num_workers": 0,
>   "data_security_mode": "SINGLE_USER",
>   "spark_conf": {
>     "spark.master": "local[*]",
>     "spark.databricks.cluster.profile": "singleNode",
>     "spark.databricks.unityCatalog.enabled": "true"
>   },
>   "custom_tags": { "ResourceClass": "SingleNode" }
> }
> ```
> Node type: memory-optimized. Default recommendation by cloud:
> - **AWS**: `r6id.2xlarge` (64 GB) — suitable for up to ~10k leaf series with MinTrace
> - **Azure**: `Standard_E16ads_v5` (128 GB)
> - **GCP**: `n2-highmem-16` (128 GB)
>
> For larger hierarchies (>10k leaves), upgrade to the next tier or switch `reconciliation_method` to `wls_var`. See the design doc for sizing guidance.

Job name pattern: `{use_case}_reconciliation_{username}` (upsert — no accumulation of stale jobs).

Upsert the job (same pattern as Skill 4 Step 5):
1. Search for an existing job named `{use_case}_reconciliation_{username}` owned by `{full_email}`
2. If found → update it with the new notebook path and cluster config
3. If not found → create it

Then trigger a run. Poll status and report progress:

```
[HH:MM:SS] {use_case}_reconciliation: RUNNING
[HH:MM:SS] {use_case}_reconciliation: SUCCEEDED (duration: Xm Ys)
```

If the job **fails**: stop immediately, report the error to the user in plain language, and ask how to proceed. Do NOT continue to Step 5.

The output table will be `{catalog}.{schema}.{use_case}_reconciliation_output`.

### Step 5: Validate coherence and summarize

After the notebook completes, run a coherence spot-check:

```sql
SELECT
  hierarchy_level,
  reconciliation_method,
  COUNT(DISTINCT unique_id) AS n_series,
  COUNT(*) AS n_rows,
  ROUND(AVG(ABS(y_base - y_reconciled)), 4) AS avg_adjustment
FROM {catalog}.{schema}.{use_case}_reconciliation_output
GROUP BY hierarchy_level, reconciliation_method
ORDER BY hierarchy_level
```

Present a summary to the user:
- How many series were reconciled at each level
- Average adjustment magnitude (how much forecasts changed)
- Confirm coherence is achieved

Example narrative:
> "Reconciliation complete. The forecasts at the `store` level were adjusted by an average of X units to ensure they sum correctly to the `region` and `country` levels. All {n} hierarchy relationships are now coherent."

### Step 6: Generate reproducibility notebook

Generate a reproducibility notebook at `{notebook_base_path}/run_reconciliation_repro.ipynb` — identical to the run notebook, allowing the user to re-run reconciliation independently.

### ⛔ STOP GATE — Step 7: Final confirmation

```
AskUserQuestion:
  "✅ Hierarchical reconciliation complete for use case '{use_case}'.

   Summary:
   • Method: {reconciliation_method}
   • Output table: {catalog}.{schema}.{use_case}_reconciliation_output
   • Reproducibility notebook: {notebook_base_path}/run_reconciliation_repro

   What would you like to do next?
   (a) Done — reconciled forecasts are ready for business use
   (b) Re-run with a different reconciliation method

   Options: [a, b]"
```

## Output Tables

| Table | Description |
|-------|-------------|
| `{use_case}_reconciliation_output` | Reconciled forecasts across all hierarchy levels |

**Schema:**

| Column | Type | Description |
|--------|------|-------------|
| `unique_id` | string | Series identifier (encodes hierarchy level, e.g. `USA/California/Store1`) |
| `ds` | timestamp | Forecast date |
| `y_base` | double | Original forecast before reconciliation |
| `y_reconciled` | double | Coherent forecast after reconciliation |
| `hierarchy_level` | string | Which hierarchy level this series belongs to |
| `reconciliation_method` | string | Method used (e.g. `MinTrace`) |

## Key Concepts

**Why reconcile?** Without reconciliation, the sum of store-level forecasts will not equal the region forecast, which will not equal the country forecast. This creates inconsistencies when different teams use forecasts at different levels for planning.

**How hierarchy metadata is built:** There are two paths depending on the data:
- **Leaf-level data** (no `/` in unique_id): Skill 1 called `run_aggregation()`, which created series at all levels (`USA`, `USA/California`, `USA/California/Store1`) and saved `_hierarchy_S` and `_hierarchy_tags`.
- **Pre-aggregated data** (unique_ids already use `/`): Skill 1 called `derive_hierarchy_from_unique_ids()`, or Skill 6 Step 1a derives them on the fly from `train_data`. In either case `_hierarchy_S` and `_hierarchy_tags` are available before reconciliation runs.

**MinTrace uses backtest residuals:** MinTrace estimates how correlated the forecast errors are across the hierarchy to find the optimal adjustment weights. It uses out-of-sample backtest residuals from `_evaluation_output` — this is statistically better than in-sample fits because it avoids optimism bias, especially important when MMF selects different models per series. All methods supported by Skill 6 use `_evaluation_output`.
