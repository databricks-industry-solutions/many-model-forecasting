# Hierarchical Reconciliation

> ⛔ **MANDATORY:** If you have not read [SKILL.md](SKILL.md) yet, read it now before proceeding. Do NOT take any action until you have read both SKILL.md and this file in full.

**Slash command:** `/hierarchical-reconciliation`

Applies hierarchical reconciliation to MMF forecasts produced independently per hierarchy level, making them coherent across all levels (e.g., store → region → country). Consumes one `best_models` table and one `evaluation_output` table per level, plus a user-provided membership table describing parent–child relationships. Produces a single `reconciliation_output` table with coherent forecasts across the full hierarchy.

**This skill is optional — only run it if the use case has a meaningful hierarchy.**

> ℹ️ Before starting this skill, read the **Hierarchical Reconciliation** section in the README. It explains the multi-level workflow: Skills 1–5 must have been run once per hierarchy level before Skill 6 can reconcile.

## Preconditions

> ⛔ **Verify before starting this skill.** If preconditions are missing, do NOT improvise — route the user back.

| Precondition | How to verify | If missing |
|---|---|---|
| N × `{level}_best_models` tables exist (one per hierarchy level) | `SHOW TABLES IN {catalog}.{schema}` — look for `*_best_models` pattern | Route back to **Skill 5** for each missing level |
| N × `{level}_evaluation_output` tables exist (one per hierarchy level) | Same as above — look for `*_evaluation_output` pattern | Route back to **Skill 4** for each missing level |
| A membership table (adjacency list) exists as a Delta table | User provides the fully-qualified name — Step 2 verifies | Ask the user to create it (see README for schema) |

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `catalog` | From prior skills | Unity Catalog name |
| `schema` | From prior skills | Schema name |
| `use_case` | From Skill 1 | Use case name (prefixes output table) |
| `freq` | Auto-detected | Frequency code — H \| D \| W \| M |
| `date_col` | `ds` | Date column name in best_models tables |
| `target` | `y` | Target column name in best_models tables |
| `reconciliation_method` | `MinTrace` | Reconciliation method — see Step 3 |

## Steps

### ⛔ STOP GATE — Step 0: Confirm catalog, schema, use case

If not already known from prior skills, ask in plain text (do NOT use AskUserQuestion — this is free text):

> "To get started I need three things:
>  • Catalog:
>  • Schema:
>  • Use case name (the prefix for your output table, e.g. `rossmann`, `retail_sales`):"

**Do NOT proceed until the user provides all three.**

Call `get_current_user()` to obtain `{full_email}`.

If `{project_folder}` and `{notebook_base_path}` are already known from prior skills (Skills 1–5), carry them forward — do NOT ask again.

If not already known (user started directly from Skill 6), ask:

```
AskUserQuestion:
  "Where would you like to store the notebooks for this project?

   (a) Use an existing folder — provide the folder name (e.g. 'my-project')
   (b) Create a new folder — provide a name and I will create /Users/{full_email}/{name}/notebooks/
   (c) Use default — I will create /Users/{full_email}/{use_case}/notebooks/

  Options: [free text — user picks (a), (b), or (c) and provides a name if needed]"
```

**WAIT for the user to respond.** Then set:
- `{project_folder}` = user-provided name, or `{use_case}` if they pick (c)
- `{notebook_base_path}` = `/Users/{full_email}/{project_folder}/notebooks`

### ⛔ STOP GATE — Step 1: Discover and confirm level tables

Scan the schema for tables matching the MMF output patterns:

```sql
SHOW TABLES IN {catalog}.{schema}
```

Filter for tables ending in `_best_models` and `_evaluation_output`. Group them into pairs by matching prefix (e.g. `stores_best_models` + `stores_evaluation_output` → level `stores`).

Present the discovered pairs to the user and ask them to confirm the level names and their order from **most granular (leaves) to most aggregate (root)**:

> "I found the following level table pairs in `{catalog}.{schema}`:
>
> | # | Level name | best_models table | evaluation_output table |
> |---|---|---|---|
> | 1 | stores | {catalog}.{schema}.stores_best_models | {catalog}.{schema}.stores_evaluation_output |
> | 2 | regions | {catalog}.{schema}.regions_best_models | {catalog}.{schema}.regions_evaluation_output |
> | 3 | country | {catalog}.{schema}.country_best_models | {catalog}.{schema}.country_evaluation_output |
>
> Is this correct? Please confirm the order (most granular → most aggregate) and correct any table names if needed."

**WAIT for the user to confirm.** Store the confirmed list as `{levels}` (ordered leaf→root). If the user provides corrections, apply them before proceeding.

If no matching pairs are found, tell the user in plain text that no `*_best_models` / `*_evaluation_output` pairs were found in the schema, and route them back to Skills 4–5.

**Auto-detect `{freq}` silently** once `{levels}` is confirmed — sample `ds` values from the first level's `best_models` table:

```sql
SELECT ds FROM {first_level_best_models_table} WHERE ds IS NOT NULL LIMIT 1000
```

Compute the median gap between consecutive distinct `ds` values:
- Gap ≈ 1 day → `D`; gap ≈ 7 days → `W`; gap ≈ 28–31 days → `M`; gap < 1 day → `H`

Do NOT narrate this detection. If freq was already known from prior skills, carry it forward.
Only ask if the gap is ambiguous:

```
AskUserQuestion:
  "Could not determine frequency automatically. What is the forecast frequency?

   (a) Daily (D)
   (b) Weekly (W)
   (c) Monthly (M)
   (d) Hourly (H)

   Options: [a, b, c, d]"
```

### ⛔ STOP GATE — Step 2: Confirm membership table

Ask the user in plain text:

> "Provide the fully-qualified name of your membership table (format: `catalog.schema.table_name`).
>
> The table must be an adjacency list with these columns:
> - `unique_id` (string) — series identifier, must match exactly the IDs in your level tables
> - `level_name` (string) — level this series belongs to (e.g. `store`, `region`, `country`)
> - `parent_unique_id` (string, nullable) — parent series ID; NULL only for the single root"

**WAIT for the user to provide the table name.** Store as `{hierarchy_table}`.

Then verify it exists and has the required columns:

```sql
SELECT unique_id, level_name, parent_unique_id FROM {hierarchy_table} LIMIT 5
```

If the query fails or required columns are missing, report the issue and ask the user to check the table.

### ⛔ STOP GATE — Step 3: Propose and confirm reconciliation method

Present MinTrace as the recommendation with reasoning, then ask:

> "I recommend **MinTrace** (`mint_shrink`) — it minimizes forecast error variance by estimating the covariance structure across all hierarchy levels using the backtest residuals. It is the statistically optimal method when sufficient residual samples are available.
>
> Alternatives:
> - **BottomUp** — aggregates leaf-level forecasts upward; simple and robust
> - **TopDown** — distributes the top-level forecast downward by historical proportions
> - **MiddleOut** — anchors at an intermediate level and reconciles up and down
> - **ERM** — learns an optimal reconciliation matrix from residuals"

```
AskUserQuestion:
  "Which reconciliation method would you like to use?

   (a) MinTrace — recommended, statistically optimal
   (b) BottomUp — aggregates leaf forecasts upward; simple and robust
   (c) TopDown — distributes the top-level forecast downward by historical proportions
   (d) MiddleOut or ERM — anchor at an intermediate level (MiddleOut) or learn an optimal reconciliation matrix (ERM)

   Options: [a, b, c, d]"
```

Map: (a) → `MinTrace`, (b) → `BottomUp`, (c) → `TopDown`. If **(d)** is selected, ask in plain text: *"MiddleOut or ERM?"* and map accordingly. Store as `{reconciliation_method}`.

If **MiddleOut** is selected, ask in plain text:

> "Which level should be the middle anchor? Choose from the levels you confirmed in Step 1 (e.g. `region` in a store → region → country hierarchy):"

**WAIT for the user to respond.** Verify the provided name matches one of the confirmed level names in `{levels}`. Store as `{middle_level}`.

If **(a) MinTrace** is selected, also ask about the sub-method:

```
AskUserQuestion:
  "MinTrace sub-method — `mint_shrink` is the default and works well in most cases.
   Use `wls_struct` or `wls_var` if you have few backtest samples at upper levels.

   (a) mint_shrink — default, shrinkage covariance estimator
   (b) wls_struct — structural weights (number of bottom-level series)
   (c) wls_var — variance scaling weights
   (d) mint_cov — full sample covariance (requires many residual samples)

   Options: [a, b, c, d]"
```

Map: (a) → `mint_shrink`, (b) → `wls_struct`, (c) → `wls_var`, (d) → `mint_cov`. Store as `{mintrace_method}`.

### Step 4: Generate reconciliation notebook

Generate `{notebook_base_path}/run_reconciliation.ipynb` from the template `mmf_reconciliation_notebook_template.ipynb`, filling in:

| Placeholder | Value |
|-------------|-------|
| `{full_email}` | from `get_current_user()` |
| `{catalog}` | confirmed catalog |
| `{schema}` | confirmed schema |
| `{use_case}` | confirmed use case |
| `{levels}` | confirmed levels list (ordered leaf→root) — each entry must use **exactly** these keys: `name`, `best_models_table`, `evaluation_table`. Example: `[{"name": "store", "best_models_table": "cat.schema.store_best_models", "evaluation_table": "cat.schema.store_evaluation_output"}, ...]` |
| `{hierarchy_table}` | confirmed membership table name |
| `{freq}` | detected or confirmed frequency |
| `{date_col}` | `ds` (or user-specified) |
| `{target}` | `y` (or user-specified) |
| `{reconciliation_method}` | method confirmed in Step 3 |
| `{mintrace_method}` | sub-method confirmed in Step 3 (only if MinTrace) |
| `{middle_level}` | middle anchor level confirmed in Step 3 (only if MiddleOut) |

### Step 5: Run on classic compute (Single Node, memory-optimized)

Reconciliation requires classic compute — `toArrow()` and scipy sparse are not available on Spark Connect (serverless). Run on a **Single Node** job cluster with DBR ML.

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
> For larger hierarchies (>10k leaves), upgrade to the next tier or switch to `wls_struct`.

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

If the job **fails**: stop immediately, report the error to the user in plain language, and ask how to proceed. Do NOT continue to Step 6.

The output table will be `{catalog}.{schema}.{use_case}_reconciliation_output`.

### Step 6: Validate coherence and summarize

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
- Confirm the output table is ready

### ⛔ STOP GATE — Step 7: Final confirmation

```
AskUserQuestion:
  "✅ Hierarchical reconciliation complete for use case '{use_case}'.

   Summary:
   • Method: {reconciliation_method}
   • Levels reconciled: {n_levels}
   • Output table: {catalog}.{schema}.{use_case}_reconciliation_output

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
| `unique_id` | string | Series identifier |
| `ds` | timestamp | Forecast date |
| `y_base` | double | Original forecast before reconciliation |
| `y_reconciled` | double | Coherent forecast after reconciliation |
| `hierarchy_level` | string | Level this series belongs to |
| `reconciliation_method` | string | Method used (e.g. `MinTrace`) |

## Key Concepts

**Why reconcile?** Without reconciliation, store-level forecasts will not sum to the region forecast, which will not sum to the country forecast. This creates inconsistencies when different teams plan from different levels of the hierarchy.

**MinTrace uses backtest residuals:** MinTrace estimates how correlated the forecast errors are across the hierarchy to find the optimal adjustment weights. It uses out-of-sample backtest residuals from `evaluation_output` — this avoids overfitting bias, especially important when MMF selects different models per series.

**ID matching is strict:** `unique_id` values in the membership table must match exactly — same case, no extra whitespace — the `unique_id` values in all level tables. Mismatches will be reported as hard errors before reconciliation runs.
