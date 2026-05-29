# Hierarchical Reconciliation

> ⛔ **MANDATORY:** If you have not read [SKILL.md](SKILL.md) yet, read it now before proceeding. Do NOT take any action until you have read both SKILL.md and this file in full.

**Slash command:** `/reconcile-hierarchical`

Applies hierarchical reconciliation to MMF forecasts, making them coherent across all levels of a hierarchy (e.g., SKU → Category → Total). Reads the best-model forecasts from Skill 5 and produces a reconciled output table.

**This skill is optional — only run it if the use case has a meaningful hierarchy.**

## Preconditions

> ⛔ **Verify before starting this skill.** If preconditions are missing, do NOT improvise — route the user back.

| Precondition | How to verify | If missing |
|---|---|---|
| `{catalog}.{schema}.{use_case}_best_models` exists and is populated | `SELECT COUNT(*) FROM ...` | Go back to **Skill 5 (`/post-process-and-evaluate`)** |
| `{catalog}.{schema}.{use_case}_hierarchy_S` exists | `SELECT COUNT(*) FROM ...` | Skill 1 hierarchical prep step was not run — go back to **Skill 1** |
| `{catalog}.{schema}.{use_case}_hierarchy_tags` exists | `SELECT COUNT(*) FROM ...` | Same as above |
| `{catalog}.{schema}.{use_case}_fitted_output` exists (MinTrace/ERM only) | `SELECT COUNT(*) FROM ...` | Required for MinTrace and ERM — `fitted_output` must have been passed to `run_forecast()` in Skill 4 |

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `catalog` | From prior skills | Unity Catalog name |
| `schema` | From prior skills | Schema name |
| `use_case` | From Skill 1 | Use case name (prefixes all table names) |
| `date_col` | `ds` | Date column name |
| `target` | `y` | Target column name |
| `reconciliation_method` | `MinTrace` | Reconciliation method — see Step 2 |

## Steps

### ⛔ STOP GATE — Step 0: Confirm catalog, schema, use case

If not already known from prior skills, ask:

```
AskUserQuestion:
  "Which catalog, schema and use case name contain the forecast outputs?
   • Catalog:
   • Schema:
   • Use case name:"
```

**Do NOT proceed until confirmed.**

---

### Step 1: Verify preconditions

Run the following checks:

```sql
SELECT COUNT(*) AS n_best_models FROM {catalog}.{schema}.{use_case}_best_models
```
```sql
SELECT COUNT(*) AS n_S FROM {catalog}.{schema}.{use_case}_hierarchy_S
```
```sql
SELECT COUNT(*) AS n_tags FROM {catalog}.{schema}.{use_case}_hierarchy_tags
```

If any table is missing or empty, stop and route back to the appropriate skill.

Show the user the hierarchy levels found:

```sql
SELECT DISTINCT level_name FROM {catalog}.{schema}.{use_case}_hierarchy_tags ORDER BY level_name
```

---

### Step 2: Propose reconciliation method

Do NOT ask "which method do you want?" — propose one with reasoning:

> "I recommend **MinTrace** (`mint_shrink`) as it minimizes forecast error variance by estimating the error covariance matrix across all hierarchy levels — this is the statistically optimal approach.
>
> Alternatives:
> - **BottomUp** — aggregates leaf forecasts upward. Simple, no fitted values needed. Good baseline.
> - **TopDown** — distributes top-level forecast downward. Works well when top-level demand is more reliable.
> - **MiddleOut** — anchors at a middle level. Useful when middle aggregations are most trustworthy.
> - **ERM** — learns an optimal reconciliation matrix. Needs fitted values like MinTrace.
>
> Unless you have a reason to prefer otherwise, I'll use MinTrace."

Wait for user confirmation or correction.

**Note:** MinTrace and ERM require `{use_case}_fitted_output` from `run_forecast()`. If that table is missing, default to BottomUp and explain why.

---

### Step 3: Generate reconciliation notebook

Generate `{notebook_base_path}/run_reconciliation.ipynb` from the template `mmf_reconciliation_notebook_template.ipynb`, filling in:

| Placeholder | Value |
|-------------|-------|
| `{catalog}` | confirmed catalog |
| `{schema}` | confirmed schema |
| `{use_case}` | confirmed use case |
| `{date_col}` | `ds` (or user-specified) |
| `{target}` | `y` (or user-specified) |
| `{reconciliation_method}` | method confirmed in Step 2 |

---

### Step 4: Run on serverless compute

Reconciliation is a matrix operation — no GPU needed. Run on **serverless** compute.

Tell the user:
> "This notebook runs on serverless compute — no cluster setup needed."

Run the notebook. The output table will be `{catalog}.{schema}.{use_case}_reconciliation_output`.

---

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

---

### Step 6: Generate reproducibility notebook

Generate a reproducibility notebook at `{notebook_base_path}/run_reconciliation_repro.ipynb` — identical to the run notebook, allowing the user to re-run reconciliation independently.

---

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

---

## Key Concepts

**Why reconcile?** Without reconciliation, the sum of store-level forecasts will not equal the region forecast, which will not equal the country forecast. This creates inconsistencies when different teams use forecasts at different levels for planning.

**What `aggregate()` did in Skill 1:** When Skill 1 detected a hierarchy and called `aggregate()`, it created series at all levels (`USA`, `USA/California`, `USA/California/Store1`) and saved the summation matrix (`_hierarchy_S`) and level metadata (`_hierarchy_tags`). Skill 6 uses those to apply the reconciliation.

**MinTrace needs fitted values:** MinTrace estimates how correlated the forecast errors are across the hierarchy to find the optimal adjustment weights. It needs the in-sample model predictions (`_fitted_output`) to compute these correlations. BottomUp and TopDown do not need this.
